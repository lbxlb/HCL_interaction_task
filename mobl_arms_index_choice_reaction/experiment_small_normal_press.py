"""临时模型上的小幅法向动态接触试验；不修改正式 XML 或任务日志。"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from simulator import IndexSimulator
from task import ChoicePanelTask


INTERNAL_GEOM_PAIRS = (('thorax', 'scapula'), ('thorax', 'humerus'),
                       ('clavicle', 'scapula'), ('scapula', 'humerus'),
                       ('humerus', 'radius'))


def collision_pair_audit(model):
    pairs = []
    for pair in INTERNAL_GEOM_PAIRS:
        entries = []
        for name in pair:
            gid = model.geom(name).id
            bid = int(model.geom_bodyid[gid])
            entries.append(dict(geom=name, body=model.body(bid).name,
                                parent_body=model.body(int(model.body_parentid[bid])).name,
                                contype=int(model.geom_contype[gid]),
                                conaffinity=int(model.geom_conaffinity[gid]),
                                body_geoms=[model.geom(i).name for i in range(model.ngeom)
                                            if int(model.geom_bodyid[i]) == bid]))
        pairs.append(entries)
    return dict(nexclude=int(model.nexclude), pairs=pairs)


def build_experiment_xml(panel_xml, pointing_xml, output_path, fixture_positions,
                         excluded_body_pairs=()):
    panel = ET.fromstring(panel_xml.read_bytes())
    pointing = ET.parse(pointing_xml).getroot()
    assert panel.find('equality') is None
    equality = ET.SubElement(panel, 'equality')
    relations = pointing.findall('./equality/joint')
    assert len(relations) == 12
    for relation in relations:
        equality.append(copy.deepcopy(relation))
    for name in ('head_rot', 'elv_angle', 'shoulder_elv', 'pro_sup'):
        ET.SubElement(equality, 'joint', name=f'fixture_{name}', joint1=name,
                      polycoef=f'{fixture_positions[name]:.17g} 0 0 0 0',
                      solref='0.004 1', solimp='0.9999 0.9999 0.001 0.5 2')
    if excluded_body_pairs:
        contact = panel.find('contact')
        if contact is None:
            contact = ET.SubElement(panel, 'contact')
        for first, second in excluded_body_pairs:
            ET.SubElement(contact, 'exclude', body1=first, body2=second)
    actuator = ET.SubElement(panel, 'actuator')
    ET.SubElement(actuator, 'motor', name='press_elbow_motor',
                  joint='elbow_flexion', gear='1',
                  ctrllimited='true', ctrlrange='-10 10')
    ET.SubElement(actuator, 'motor', name='press_shoulder_motor',
                  joint='shoulder_rot', gear='1',
                  ctrllimited='true', ctrlrange='-10 10')
    for asset in panel.findall('./asset/*'):
        if 'file' in asset.attrib:
            asset.set('file', str((panel_xml.parent / asset.get('file')).resolve()))
    ET.ElementTree(panel).write(output_path, encoding='utf-8')


def joint_displacement(model, data, point_jac, normal, distance, dofs):
    delta = np.linalg.lstsq(point_jac[:, dofs], normal * distance, rcond=None)[0]
    return delta


def dynamics_snapshot(model, data, step):
    """只读取当前已计算的状态，不调用 forward 或改变求解器状态。"""
    contacts = []
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, force)
        contacts.append(dict(
            geoms=[model.geom(int(g)).name for g in (contact.geom1, contact.geom2)],
            distance_m=float(contact.dist), normal_force_N=float(force[0])))
    equality_rows = np.flatnonzero(
        data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
    equality = [dict(name=model.equality(int(data.efc_id[row])).name,
                     residual_rad=float(data.efc_pos[row]),
                     row_force=float(data.efc_force[row])) for row in equality_rows]
    constraint_forces = {}
    contact_types = [int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                     int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                     int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)]
    equality_mask = np.zeros(data.nefc, dtype=bool)
    equality_mask[equality_rows] = True
    contact_mask = np.isin(data.efc_type[:data.nefc], contact_types)
    for name, mask in (('equality', equality_mask), ('contact', contact_mask),
                       ('other', ~(equality_mask | contact_mask))):
        projected = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(model, data, projected,
                            np.where(mask, data.efc_force[:data.nefc], 0.0))
        constraint_forces[name] = projected.tolist()
    assert np.allclose(sum(np.array(value) for value in constraint_forces.values()),
                       data.qfrc_constraint, rtol=1e-9, atol=1e-8)
    return dict(step=step, time_s=float(data.time), ctrl=data.ctrl.tolist(),
                qpos=data.qpos.tolist(), qvel=data.qvel.tolist(), qacc=data.qacc.tolist(),
                generalized_forces={name: getattr(data, 'qfrc_' + name).tolist()
                                    for name in ('bias', 'passive', 'actuator', 'applied', 'constraint')},
                constraint_force_components=constraint_forces,
                equalities=equality, contacts=contacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('press', 'hold-only'), default='press',
                        help='hold-only keeps the initial joint target for the entire run')
    parser.add_argument('--diagnose', action='store_true',
                        help='record initial and first 20 post-step dynamics snapshots')
    parser.add_argument('--exclude-internal-pairs', action='store_true',
                        help='exclude five selected internal body pairs in temporary XML')
    parser.add_argument('--target-source', choices=('original', 'scan-candidate'),
                        default='original',
                        help='use the existing inward-offset target or the selected static scan pose')
    parser.add_argument('--hold-steps', type=int, default=30,
                        help='number of final target-hold steps; default preserves the 100-step run')
    args = parser.parse_args()
    if args.hold_steps < 1:
        parser.error('--hold-steps must be positive')
    if args.target_source == 'scan-candidate' and (args.mode != 'press'
                                                   or not args.exclude_internal_pairs):
        parser.error('--target-source scan-candidate requires --mode press and '
                     '--exclude-internal-pairs')
    base = Path(__file__).resolve().parent
    root = base.parent
    panel_xml = base / 'simulation.xml'
    pointing_xml = root / 'mobl_arms_index_pointing/simulation.xml'
    scan_path = root / 'artifacts/trajectory_check_2026-10-21/shoulder_elbow_reachability.json'
    output_directory = root / 'artifacts/trajectory_check_2026-10-22'
    if args.target_source == 'scan-candidate':
        output = output_directory / 'small_normal_press_scan_candidate_filtered.json'
        if args.hold_steps != 30:
            output = output.with_name(
                f'small_normal_press_scan_candidate_filtered_hold{args.hold_steps}.json')
    else:
        output = output_directory / (
            'small_normal_press.json' if args.mode == 'press' else 'small_normal_hold_only.json')
        if args.exclude_internal_pairs:
            output = output.with_name('small_normal_press_filtered.json' if args.mode == 'press'
                                      else 'small_normal_hold_filtered.json')
        if args.mode == 'press' and args.hold_steps != 30:
            suffix = '_filtered' if args.exclude_internal_pairs else ''
            output = output.with_name(f'small_normal_press{suffix}_hold{args.hold_steps}.json')
    config = yaml.safe_load((base / 'config.yaml').read_text(encoding='utf-8'))
    scan = json.loads(scan_path.read_text(encoding='utf-8'))
    config['target_button'] = int(scan['target_button'])
    nearest = scan['nearest_surface']
    pose = {'shoulder_rot': float(nearest['shoulder_rot_rad']),
            'elbow_flexion': float(nearest['elbow_flexion_rad'])}
    selected_candidate = None
    if args.target_source == 'scan-candidate':
        target_scan_path = output_directory / 'filtered_press_target_scan.json'
        target_scan = json.loads(target_scan_path.read_text(encoding='utf-8'))
        selected_candidate = target_scan[
            'selected_candidate_by_tangential_alignment_within_0_2_to_0_4mm_clearance']
    originals = panel_xml.read_bytes(), pointing_xml.read_bytes()

    with contextlib.redirect_stdout(io.StringIO()):
        source_sim = IndexSimulator(base / 'config.yaml', panel_xml)
        source_sim.reset()
    try:
        initial = source_sim.data.qpos.copy()
        dt = float(source_sim.model.opt.timestep)
        fixture_positions = {
            name: float(initial[source_sim.model.jnt_qposadr[source_sim.model.joint(name).id]])
            for name in ('head_rot', 'elv_angle', 'shoulder_elv', 'pro_sup')
        }
        formal_panel_audit = collision_pair_audit(source_sim.model)
        excluded_body_pairs = [tuple(entry['body'] for entry in pair)
                               for pair in formal_panel_audit['pairs']] if args.exclude_internal_pairs else []
    finally:
        source_sim.close()

    try:
        with tempfile.TemporaryDirectory(prefix='small_normal_press_') as directory:
            experiment_xml = Path(directory) / 'experiment.xml'
            build_experiment_xml(panel_xml, pointing_xml, experiment_xml, fixture_positions,
                                 excluded_body_pairs)
            model = mujoco.MjModel.from_xml_path(str(experiment_xml))
            model.opt.timestep = dt
            data = mujoco.MjData(model)
            data.qpos[:] = initial
            shoulder = model.joint('shoulder_rot')
            elbow = model.joint('elbow_flexion')
            joint_ids = [shoulder.id, elbow.id]
            qadr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
            dofs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
            data.qpos[qadr] = [pose['shoulder_rot'], pose['elbow_flexion']]
            mujoco.mj_forward(model, data)

            button = model.geom(f"button-{config['target_button']}").id
            finger = model.geom('hand_2distph').id
            site = model.site('fingertip').id
            button_rotation = data.geom_xmat[button].reshape(3, 3).copy()
            normal = button_rotation[:, 2]
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            q_contact = data.qpos[qadr].copy()
            if selected_candidate is None:
                in_delta = joint_displacement(model, data, jacp, normal, -0.0008, dofs)
                q_press = q_contact + in_delta
            else:
                q_press = np.radians([
                    selected_candidate['shoulder_rot_deg'],
                    selected_candidate['elbow_flexion_deg']])
                in_delta = q_press - q_contact
            # Start clearly outside the geom margins so task contact cannot be pre-triggered.
            out_delta = joint_displacement(model, data, jacp, normal, 0.006, dofs)
            q_out = q_contact + out_delta

            def set_pose(q):
                data.qpos[qadr] = q
                mujoco.mj_forward(model, data)
                return float(mujoco.mj_geomDistance(model, data, finger, button, 10.0, None))

            initial_contact_distance = set_pose(q_contact)
            outside_distance = set_pose(q_out)
            press_target_distance = set_pose(q_press)
            press_target_pair_detected = any(
                {int(data.contact[i].geom1), int(data.contact[i].geom2)} == {finger, button}
                for i in range(data.ncon))
            target_margin = float(model.geom_margin[button] + model.geom_margin[finger])
            data.qpos[qadr] = q_out
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            q_initial = data.qpos.copy()
            qvel_initial = data.qvel.copy()

            class SimView:
                pass

            view = SimView()
            view.model, view.data, view.config = model, data, config
            view.finger_geom_name = 'hand_2distph'
            task_probe = ChoicePanelTask(config, view)
            initial_task_contact = bool(task_probe._check_button_contact(config['target_button']))
            initial_button_contacts = [
                dict(distance_m=float(data.contact[i].dist))
                for i in range(data.ncon)
                if button in (data.contact[i].geom1, data.contact[i].geom2)]
            if initial_task_contact or outside_distance <= target_margin + 0.001:
                raise RuntimeError(
                    f'起始状态不够远离接触区: distance={outside_distance:.6g} m, '
                    f'margin_sum={target_margin:.6g} m, contact={initial_task_contact}')
            shoulder_motor = model.actuator('press_shoulder_motor').id
            elbow_motor = model.actuator('press_elbow_motor').id
            kp, kd, limit = 50.0, 8.0, 10.0
            settle_steps, ramp_steps, hold_steps = 20, 50, args.hold_steps
            total_steps = settle_steps + ramp_steps + hold_steps
            contact_forces = []
            distances = []
            first_task_contact = None
            saturation_count = 0
            force_vector = np.zeros(6)
            first_contact_snapshot = None
            first_geometric_touch = None
            target_contact_distances = []
            samples = []
            peak_qvel = 0.0
            dynamics = [dynamics_snapshot(model, data, 0)] if args.diagnose else []
            for step in range(total_steps):
                if args.mode == 'hold-only' or step < settle_steps:
                    blend, blend_rate = 0.0, 0.0
                elif step < settle_steps + ramp_steps:
                    phase = (step - settle_steps) / ramp_steps
                    blend = 3 * phase**2 - 2 * phase**3
                    blend_rate = (6 * phase - 6 * phase**2) / (ramp_steps * dt)
                else:
                    blend, blend_rate = 1.0, 0.0
                q_target = q_out + blend * (q_press - q_out)
                qdot_target = blend_rate * (q_press - q_out)
                raw = []
                for index, dof in enumerate(dofs):
                    raw.append(float(
                        kp * (q_target[index] - data.qpos[qadr[index]])
                        + kd * (qdot_target[index] - data.qvel[dof])
                        + data.qfrc_bias[dof] - data.qfrc_passive[dof]))
                data.ctrl[shoulder_motor] = float(np.clip(raw[0], -limit, limit))
                data.ctrl[elbow_motor] = float(np.clip(raw[1], -limit, limit))
                saturation_count += sum(abs(value) > limit for value in raw)
                mujoco.mj_step(model, data)
                # Refresh derived geometry, contacts and forces at the newly integrated state.
                mujoco.mj_forward(model, data)
                if args.diagnose and step < settle_steps:
                    dynamics.append(dynamics_snapshot(model, data, step + 1))
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))))
                distance = float(mujoco.mj_geomDistance(model, data, finger, button, 10.0, None))
                distances.append(distance)
                task_contact = bool(task_probe._check_button_contact(config['target_button']))
                if task_contact and first_task_contact is None:
                    first_task_contact = step + 1
                    first_contact_snapshot = dict(step=step + 1, time_s=(step + 1) * dt,
                                                  geom_distance_m=distance)
                button_normal_force = 0.0
                button_contacts = []
                for ci in range(data.ncon):
                    contact = data.contact[ci]
                    if button not in (contact.geom1, contact.geom2):
                        continue
                    mujoco.mj_contactForce(model, data, ci, force_vector)
                    normal_force = float(force_vector[0])
                    button_normal_force += normal_force
                    button_contacts.append(dict(
                        other_geom=mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM,
                            contact.geom2 if contact.geom1 == button else contact.geom1),
                        distance_m=float(contact.dist),
                        normal_force_N=normal_force,
                        point_world_m=contact.pos.tolist(),
                        normal_world=contact.frame[:3].tolist()))
                if button_contacts:
                    target_contact_distances.extend(item['distance_m'] for item in button_contacts)
                if first_contact_snapshot and step + 1 == first_contact_snapshot['step']:
                    first_contact_snapshot['matching_hand_geom_contact_distances_m'] = [
                        item['distance_m'] for item in button_contacts
                        if item['other_geom'] == 'hand_2distph']
                # Use matching contact pairs to confirm overlap; zero distance alone is ambiguous.
                if any(item['other_geom'] == 'hand_2distph' and item['distance_m'] <= 0
                       for item in button_contacts) and first_geometric_touch is None:
                    first_geometric_touch = dict(step=step + 1, time_s=(step + 1) * dt,
                                                 geom_distance_m=distance)
                contact_forces.append(button_normal_force)
                samples.append(dict(step=step + 1, time_s=float(data.time),
                                    phase=('hold-only' if args.mode == 'hold-only' else
                                           'settle' if step < settle_steps else
                                           'ramp' if step < settle_steps + ramp_steps else 'hold'),
                                    geom_distance_m=distance, task_contact=task_contact,
                                    button_normal_force_N=button_normal_force,
                                    q_actual_rad=data.qpos[qadr].tolist(),
                                    q_target_rad=q_target.tolist(),
                                    q_error_rad=(q_target - data.qpos[qadr]).tolist(),
                                    qvel_actual_rad_s=data.qvel[dofs].tolist(),
                                    qvel_target_rad_s=qdot_target.tolist(),
                                    qvel_error_rad_s=(qdot_target - data.qvel[dofs]).tolist(),
                                    applied_torque_Nm=data.ctrl[[shoulder_motor, elbow_motor]].tolist(),
                                    button_contacts=button_contacts))

            distance_comparisons = [
                dict(step=sample['step'],
                     geom_distance_m=sample['geom_distance_m'],
                     contact_distance_m=contact['distance_m'],
                     absolute_difference_m=abs(sample['geom_distance_m'] - contact['distance_m']))
                for sample in samples for contact in sample['button_contacts']
                if contact['other_geom'] == 'hand_2distph']
            distance_disagreements = [item for item in distance_comparisons
                                      if item['absolute_difference_m'] > 1e-5]
            report = dict(
                experiment='temporary model; short PD-driven normal approach and press',
                mode=args.mode,
                collision_filter=dict(excluded_body_pairs=excluded_body_pairs,
                                      formal_panel_audit=formal_panel_audit,
                                      press_target_finger_button_contact_detected=press_target_pair_detected,
                                      experiment_nexclude=int(model.nexclude)),
                initial_state=dict(qpos=q_initial.tolist(), qvel=qvel_initial.tolist()),
                dynamics_diagnostics=dict(
                    initial_control_note='step 0 has zero motor controls before the first PD command',
                    sampling='read-only initial and post-step snapshots; no extra forward calls',
                    joints=[dict(name=model.joint(jid).name,
                                 qpos_address=int(model.jnt_qposadr[jid]),
                                 dof_address=int(model.jnt_dofadr[jid]))
                            for jid in range(model.njnt)],
                    snapshots=dynamics),
                source_pose_deg=dict(shoulder_rot=nearest['shoulder_rot_deg'],
                                     elbow_flexion=nearest['elbow_flexion_deg']),
                target_source=args.target_source,
                target_pose_deg=dict(shoulder_rot=float(np.degrees(q_press[0])),
                                     elbow_flexion=float(np.degrees(q_press[1]))),
                selected_scan_candidate=selected_candidate,
                model_changes=('temporary XML only: 12 candidate equalities, four fixture equalities, '
                               'two bounded test motors; '
                               + ('five selected body-pair exclusions; ' if excluded_body_pairs else '')
                               + 'formal files unchanged'),
                timestep_s=dt,
                sampling='mj_step then mj_forward; geometry, contacts and forces at post-step state',
                distance_diagnostics=dict(
                    comparison_threshold_m=1e-5,
                    matching_pair_comparisons=len(distance_comparisons),
                    max_absolute_difference_m=max(
                        (item['absolute_difference_m'] for item in distance_comparisons), default=None),
                    disagreements=distance_disagreements,
                    interpretation=('Disagreements are diagnostic flags, not a proven cause. '
                                    'A zero geomDistance value alone does not establish surface touch.')),
                samples=samples,
                press_geometry=dict(
                    target_source=args.target_source,
                    selected_scan_candidate=selected_candidate,
                    button_front_normal_world=normal.tolist(),
                    out_delta_rad=out_delta.tolist(), in_delta_rad=in_delta.tolist(),
                    outside_q_rad=q_out.tolist(), press_target_q_rad=q_press.tolist(),
                    contact_distance_at_source_m=initial_contact_distance,
                    collision_distance_at_outside_start_m=outside_distance,
                    collision_distance_at_press_target_m=press_target_distance,
                    geom_margin_sum_mm=target_margin * 1000,
                    requested_outward_tip_offset_mm=6.0,
                    linearized_outward_tip_displacement_mm=(jacp[:, dofs] @ out_delta * 1000).tolist(),
                    linearized_outward_normal_component_mm=float(normal @ jacp[:, dofs] @ out_delta * 1000),
                    requested_inward_offset_past_source_mm=(0.8 if selected_candidate is None else None)),
                controller=dict(kp=kp, kd=kd, torque_limit_Nm=limit,
                                settle_steps=settle_steps, ramp_steps=ramp_steps,
                                hold_steps=hold_steps, steps=total_steps,
                                saturation_count=saturation_count),
                result=dict(
                    initial_task_contact=initial_task_contact,
                    initial_button_contacts=initial_button_contacts,
                    first_task_contact_step=first_task_contact,
                    first_task_contact_time_s=(first_task_contact * dt
                                               if first_task_contact is not None else None),
                    first_task_contact_snapshot=first_contact_snapshot,
                    first_geometric_touch=first_geometric_touch,
                    min_contact_pair_distance_m=min(target_contact_distances, default=None),
                    max_contact_pair_distance_m=max(target_contact_distances, default=None),
                    final_task_contact=bool(task_probe._check_button_contact(config['target_button'])),
                    contact_force_peak_N=max(contact_forces, default=0.0),
                    contact_force_final_N=contact_forces[-1] if contact_forces else 0.0,
                    contact_force_samples_N=contact_forces,
                    min_geom_distance_m=min(distances),
                    final_geom_distance_m=distances[-1] if distances else None,
                    q_final_rad=data.qpos[qadr].tolist(),
                    max_abs_controlled_joint_drift_rad=np.max(np.abs(
                        np.array([sample['q_actual_rad'] for sample in samples]) - q_out),
                        axis=0).tolist(),
                    max_abs_qvel_rad_s=peak_qvel,
                    final_max_abs_qvel_rad_s=float(np.max(np.abs(data.qvel))),
                    warning_counts=[int(w.number) for w in data.warning],
                    contact_during_settle=any(x['task_contact'] for x in samples[:settle_steps]),
                    button_contacts_final=button_contacts),
                limitations=('Task contact check was queried without task.update to avoid writing '
                              'episode_results.json. The temporary motors and fixture equalities do not '
                              'yet establish a validated controller for the formal model.'))
            assert panel_xml.read_bytes() == originals[0]
            assert pointing_xml.read_bytes() == originals[1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({
                'mode': args.mode,
                'outside_start_distance_mm': outside_distance * 1000,
                'geom_margin_sum_mm': target_margin * 1000,
                'initial_task_contact': initial_task_contact,
                'source_pose_distance_mm': initial_contact_distance * 1000,
                'press_target_distance_mm': press_target_distance * 1000,
                'first_task_contact_step': first_task_contact,
                'first_task_contact_snapshot': first_contact_snapshot,
                'first_geometric_touch': first_geometric_touch,
                'contact_force_peak_N': report['result']['contact_force_peak_N'],
                'contact_force_final_N': report['result']['contact_force_final_N'],
                'min_geom_distance_mm': report['result']['min_geom_distance_m'] * 1000,
                'final_geom_distance_mm': report['result']['final_geom_distance_m'] * 1000,
                'final_task_contact': report['result']['final_task_contact'],
                'controller_saturation_count': saturation_count,
                'output': str(output)}, ensure_ascii=False, indent=2))
    finally:
        assert panel_xml.read_bytes() == originals[0]
        assert pointing_xml.read_bytes() == originals[1]


if __name__ == '__main__':
    main()
