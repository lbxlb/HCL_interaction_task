"""完整临时手臂模型的按钮无接触基线；不修改正式 XML。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from experiment_small_normal_press import (
    build_experiment_xml, collision_pair_audit, joint_displacement,
)
from simulator import IndexSimulator


def main(observation_hook=None, output_stem='button_slider_full_baseline', press=False, release=False):
    if release and not press:
        raise ValueError('Release experiment requires press=True')
    base = Path(__file__).resolve().parent
    root = base.parent
    out = root / 'artifacts/trajectory_check_2026-10-22'
    panel = base / 'simulation.xml'
    pointing = root / 'mobl_arms_index_pointing/simulation.xml'
    originals = [p.read_bytes() for p in (panel, pointing)]
    scan = json.loads((root / 'artifacts/trajectory_check_2026-10-21/'
                      'shoulder_elbow_reachability.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', panel)
        sim.reset()
    try:
        source = sim.model
        initial = sim.data.qpos.copy()
        fixture = {n: float(initial[source.jnt_qposadr[source.joint(n).id]])
                   for n in ('head_rot', 'elv_angle', 'shoulder_elv', 'pro_sup')}
        pairs = [tuple(e['body'] for e in pair)
                 for pair in collision_pair_audit(source)['pairs']]
        button_name = f"button-{scan['target_button']}"
        with tempfile.TemporaryDirectory(prefix='full_slider_baseline_') as directory:
            xml = Path(directory) / 'baseline.xml'
            build_experiment_xml(panel, pointing, xml, fixture, pairs)
            tree = ET.parse(xml)
            body = tree.getroot().find(f"./worldbody/body[@name='{button_name}']")
            ET.SubElement(body, 'joint', name='probe_button_slide', type='slide',
                          axis='0 0 -1', limited='true', range='0 0.005',
                          stiffness='1000', damping='10', springref='0',
                          solreflimit='0.004 1')
            tree.write(xml, encoding='utf-8')
            model = mujoco.MjModel.from_xml_path(str(xml))
            model.opt.timestep = 0.002
            source_integrator = mujoco.mjtIntegrator(source.opt.integrator).name
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            data = mujoco.MjData(model)
            observation = None
            # A new joint shifts array addresses: copy full blocks by name.
            for jid in range(source.njnt):
                dest = int(model.jnt_qposadr[model.joint(source.joint(jid).name).id])
                start = int(source.jnt_qposadr[jid])
                end = int(source.jnt_qposadr[jid + 1]) if jid + 1 < source.njnt else source.nq
                data.qpos[dest:dest + end - start] = initial[start:end]
                np.testing.assert_array_equal(data.qpos[dest:dest + end - start], initial[start:end])
            finger, button = model.geom('hand_2distph').id, model.geom(button_name).id
            original_margin = float(model.geom_margin[finger])
            model.geom_margin[finger] = 0.0
            joints = [model.joint(n).id for n in ('shoulder_rot', 'elbow_flexion')]
            qadr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
            data.qpos[qadr] = [scan['nearest_surface'][n + '_rad']
                               for n in ('shoulder_rot', 'elbow_flexion')]
            mujoco.mj_forward(model, data)
            normal = data.geom_xmat[button].reshape(3, 3)[:, 2].copy()
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, model.site('fingertip').id)
            q_out = data.qpos[qadr] + joint_displacement(model, data, jac, normal, 0.006, dofs)
            q_press = q_out.copy()
            if press:
                target_scan = json.loads((out / 'filtered_press_target_scan.json').read_text())
                candidate = target_scan['selected_candidate_by_tangential_alignment_within_0_2_to_0_4mm_clearance']
                data.qpos[qadr] = np.radians([candidate['shoulder_rot_deg'], candidate['elbow_flexion_deg']])
                mujoco.mj_forward(model, data)
                mujoco.mj_jacSite(model, data, jac, None, model.site('fingertip').id)
                q_press = data.qpos[qadr] + joint_displacement(model, data, jac, normal, -0.0008, dofs)
            data.qpos[qadr] = q_out
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            slider = model.joint('probe_button_slide').id
            sq, sd = int(model.jnt_qposadr[slider]), int(model.jnt_dofadr[slider])
            gravity_force = -float(data.qfrc_bias[sd])
            model.qpos_spring[sq] = -gravity_force / 1000.0
            mujoco.mj_forward(model, data)
            motors = [model.actuator(n).id for n in ('press_shoulder_motor', 'press_elbow_motor')]
            rows = []
            saturation_count = 0
            q_target = q_out.copy()

            def observe(step):
                nonlocal observation
                # Forward only on a separate state; leave integration warm-start untouched.
                observation = copy.copy(data)
                mujoco.mj_forward(model, observation)
                contacts = []
                button_contacts = []
                force = np.zeros(6)
                for ci, c in enumerate(observation.contact):
                    names = [model.geom(int(g)).name for g in (c.geom1, c.geom2)]
                    contacts.append(names)
                    if button in (c.geom1, c.geom2):
                        mujoco.mj_contactForce(model, observation, ci, force)
                        button_contacts.append(dict(geoms=names, distance_m=float(c.dist),
                                                    normal_force_N=float(force[0])))
                row = dict(step=step, time_s=float(data.time),
                           slider_position_m=float(data.qpos[sq]),
                           slider_velocity_m_s=float(data.qvel[sd]),
                           finger_button_distance_m=float(mujoco.mj_geomDistance(
                               model, observation, finger, button, 10, None)),
                           button_contacts=button_contacts, all_contact_pairs=contacts,
                           max_abs_hinge_velocity_rad_s=float(np.max(np.abs(
                               data.qvel[model.jnt_dofadr[model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE)]]))),
                           arm_error_rad=(q_target - data.qpos[qadr]).tolist(),
                           q_target_rad=q_target.tolist(),
                           motor_torque_Nm=data.ctrl[motors].tolist())
                rows.append(row)
                if observation_hook is not None:
                    observation_hook(model, observation, finger, button, row)

            observe(0)
            assert not rows[0]['button_contacts'], 'Initial button contact invalidates baseline'
            for step in range(1, 501 if release else 201):
                phase = np.clip(((step - 1) * model.opt.timestep - 0.04) / 0.2, 0, 1) if press else 0.0
                blend = 3 * phase**2 - 2 * phase**3
                rate = (6 * phase - 6 * phase**2) / 0.2 if press else 0.0
                if release and (step - 1) * model.opt.timestep >= 0.4:
                    phase = np.clip(((step - 1) * model.opt.timestep - 0.4) / 0.2, 0, 1)
                    blend = 1 - (3 * phase**2 - 2 * phase**3)
                    rate = -(6 * phase - 6 * phase**2) / 0.2
                q_target = q_out + blend * (q_press - q_out)
                qdot_target = rate * (q_press - q_out)
                raw = (50 * (q_target - data.qpos[qadr]) + 8 * (qdot_target - data.qvel[dofs])
                       + observation.qfrc_bias[dofs] - observation.qfrc_passive[dofs])
                saturation_count += int(np.count_nonzero(np.abs(raw) > 10))
                data.ctrl[motors] = np.clip(raw, -10, 10)
                mujoco.mj_step(model, data)
                assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                assert abs(data.time - step * model.opt.timestep) < 1e-10
                observe(step)
            warnings = [dict(name=mujoco.mjtWarning(i).name, count=int(data.warning[i].number))
                        for i in range(mujoco.mjtWarning.mjNWARNING) if data.warning[i].number]
            max_q = max(abs(r['slider_position_m']) for r in rows)
            contact_steps = sum(bool(r['button_contacts']) for r in rows)
            checks = dict(no_button_contact=contact_steps == 0,
                          zero_position_within_1nm=max_q < 1e-9,
                          no_warning=not warnings, no_motor_saturation=saturation_count == 0)
            if press:
                checks = dict(no_warning=not warnings, no_motor_saturation=saturation_count == 0,
                              slider_within_diagnostic_bounds=all(-0.0001 <= r['slider_position_m'] <= 0.0051 for r in rows))
            report = dict(
                experiment=('full temporary arm; press until 400ms, retract 400-600ms, settle until 1s'
                            if release else 'full temporary arm; 40ms settle, 200ms ramp, 160ms hold; no release'
                            if press else 'full temporary arm; outside-pose hold only; no press input'),
                press_mode=press,
                release_mode=release,
                mujoco_version=mujoco.__version__, timestep_s=float(model.opt.timestep),
                duration_s=float(data.time), source_integrator=source_integrator,
                integrator=mujoco.mjtIntegrator(model.opt.integrator).name,
                controller=dict(kp=50, kd=8, torque_limit_Nm=10,
                                feedforward='bias - passive, evaluated on copied current state'),
                target_button=button_name, internal_exclusions=pairs,
                copied_source_state_by_joint_name=True,
                finger_margin_original_m=original_margin, finger_margin_probe_m=0,
                slider=dict(axis_local=[0, 0, -1], range_m=model.jnt_range[slider].tolist(),
                            stiffness_N_m=float(model.jnt_stiffness[slider]),
                            damping_Ns_m=float(model.dof_damping[sd]),
                            solreflimit=model.jnt_solref[slider].tolist(),
                            mass_kg=float(model.body_mass[model.body(button_name).id]),
                            gravity_force_N=gravity_force,
                            spring_reference_m=float(model.qpos_spring[sq])),
                summary=dict(max_abs_slider_position_m=max_q,
                             max_abs_slider_velocity_m_s=max(abs(r['slider_velocity_m_s']) for r in rows),
                             min_finger_button_distance_m=min(r['finger_button_distance_m'] for r in rows),
                             min_positive_distance_query_m=min(
                                 (r['finger_button_distance_m'] for r in rows
                                  if r['finger_button_distance_m'] > 0), default=None),
                             zero_distance_without_matching_contact_samples=sum(
                                 r['finger_button_distance_m'] == 0 and not any(
                                     'hand_2distph' in c['geoms'] for c in r['button_contacts'])
                                 for r in rows),
                             button_contact_sample_count=contact_steps,
                             max_slider_position_m=max(r['slider_position_m'] for r in rows),
                             peak_button_normal_force_N=max(sum(c['normal_force_N'] for c in r['button_contacts']) for r in rows),
                             first_matching_geometric_contact_time_s=next((r['time_s'] for r in rows if any(
                                 'hand_2distph' in c['geoms'] and c['distance_m'] <= 0 for c in r['button_contacts'])), None),
                             all_contact_sample_count=sum(bool(r['all_contact_pairs']) for r in rows),
                             max_abs_hinge_velocity_rad_s=max(r['max_abs_hinge_velocity_rad_s'] for r in rows),
                             max_abs_arm_error_rad=float(np.max(np.abs([r['arm_error_rad'] for r in rows]))),
                             motor_saturation_count=saturation_count, warnings=warnings),
                diagnostic_checks=checks, samples=rows,
                sampling='Includes t=0 and every post-step state; contacts refreshed on separate MjData.',
                distance_caveat='Some mj_geomDistance queries may return zero without a matching '
                                'contact. Preserve raw values; do not interpret as proven touching '
                                'or claim continuous positive clearance from this query alone.',
                limitations='Temporary fixtures/exclusions, zero finger margin and implicitfast; '
                            '1 nm is a numerical zero diagnostic, not hardware tolerance. '
                            'No upper-limit loading or task success validated; release is tested only when release_mode is true. '
                            'Press-mode bounds [-0.1,5.1]mm are diagnostic guards, not hardware tolerances.')
    finally:
        sim.close()
        assert [p.read_bytes() for p in (panel, pointing)] == originals
    report['formal_xml_unchanged'] = True
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{output_stem}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    times = [r['time_s'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(times, [r['slider_position_m'] * 1e6 for r in rows])
    axes[0].set_ylabel('Slider position (um)')
    axes[0].set_title('Full temporary arm: short press' if press else 'Full temporary arm: no-contact hold baseline')
    axes[1].plot(times, [r['finger_button_distance_m'] * 1000
                        if r['finger_button_distance_m'] > 0 else np.nan for r in rows])
    axes[1].set_ylabel('Finger-button gap (mm)')
    axes[1].set_title('Positive distance queries; ambiguous zeros omitted '
                      f"({report['summary']['zero_distance_without_matching_contact_samples']} samples)")
    axes[1].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'{output_stem}.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(summary=report['summary'], checks=checks,
                          slider=report['slider'], formal_xml_unchanged=True), indent=2))
    assert all(checks.values()), 'Diagnostic guards failed; inspect JSON before further experiments'


if __name__ == '__main__':
    main()
