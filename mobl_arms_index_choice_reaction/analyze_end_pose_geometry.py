"""比较过滤推进长实验末态与目标态的静态几何关系。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np


def box_signed_distance(point_world, model, data, geom_id):
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    local = rotation.T @ (point_world - data.geom_xpos[geom_id])
    offset = np.abs(local) - model.geom_size[geom_id, :3]
    outside = np.maximum(offset, 0.0)
    return float(np.linalg.norm(outside) + min(float(np.max(offset)), 0.0)), local


def inspect_pose(model, data, qpos, shoulder_q, elbow_q, button_id, finger_id, site_id):
    data.qpos[:] = qpos
    data.qpos[model.jnt_qposadr[model.joint('shoulder_rot').id]] = shoulder_q
    data.qpos[model.jnt_qposadr[model.joint('elbow_flexion').id]] = elbow_q
    mujoco.mj_forward(model, data)
    button_center = data.geom_xpos[button_id].copy()
    button_rotation = data.geom_xmat[button_id].reshape(3, 3).copy()
    button_normal = button_rotation[:, 2].copy()
    fingertip = data.site_xpos[site_id].copy()
    site_distance, fingertip_local = box_signed_distance(fingertip, model, data, button_id)
    closest_segment = np.zeros(6)
    geom_distance = float(mujoco.mj_geomDistance(
        model, data, finger_id, button_id, 10.0, closest_segment))
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if button_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == button_id else contact.geom1
        contact_local = button_rotation.T @ (contact.pos - button_center)
        contacts.append(dict(
            other_geom=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
            distance_m=float(contact.dist),
            point_button_local_m=contact_local.tolist(),
            normal_dot_button_front=float(np.dot(contact.frame[:3], button_normal))))
    return dict(
        shoulder_rot_rad=float(shoulder_q), elbow_flexion_rad=float(elbow_q),
        shoulder_rot_deg=float(np.degrees(shoulder_q)),
        elbow_flexion_deg=float(np.degrees(elbow_q)),
        fingertip_world_m=fingertip.tolist(),
        fingertip_button_local_m=fingertip_local.tolist(),
        fingertip_surface_signed_distance_m=site_distance,
        fingertip_surface_gap_mm=site_distance * 1000,
        hand_geom_signed_distance_m=geom_distance,
        hand_geom_signed_distance_mm=geom_distance * 1000,
        hand_geom_closest_segment_world_m=closest_segment.reshape(2, 3).tolist(),
        button_front_normal_world=button_normal.tolist(),
        button_half_size_m=model.geom_size[button_id, :3].tolist(),
        button_contacts=contacts,
        formal_model_contact_count=int(data.ncon))


def main():
    root = Path(__file__).resolve().parent.parent
    directory = root / 'artifacts/trajectory_check_2026-10-22'
    report = json.loads((directory / 'small_normal_press_filtered_hold200.json').read_text())
    xml_path = Path(__file__).resolve().parent / 'simulation.xml'
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    qpos = np.asarray(report['initial_state']['qpos'], dtype=float)
    actual = np.asarray(report['result']['q_final_rad'], dtype=float)
    target = np.asarray(report['press_geometry']['press_target_q_rad'], dtype=float)
    button_id = model.geom('button-0').id
    finger_id = model.geom('hand_2distph').id
    site_id = model.site('fingertip').id
    actual_pose = inspect_pose(model, data, qpos, actual[0], actual[1], button_id, finger_id, site_id)
    target_pose = inspect_pose(model, data, qpos, target[0], target[1], button_id, finger_id, site_id)
    normal = np.asarray(actual_pose['button_front_normal_world'])
    tip_delta = np.asarray(actual_pose['fingertip_world_m']) - np.asarray(target_pose['fingertip_world_m'])
    normal_delta = float(np.dot(normal, tip_delta))
    tangent_delta = float(np.linalg.norm(tip_delta - normal * normal_delta))
    actual_contacts = {c['other_geom']: c for c in actual_pose['button_contacts']}
    dynamic_contact = next((c for c in report['samples'][-1]['button_contacts']
                            if c['other_geom'] == 'hand_2distph'), None)
    summary = dict(
        source='initial full qpos from dynamic experiment; only shoulder_rot and elbow_flexion replaced',
        model='formal panel XML for FK/contact geometry inspection; no simulation step',
        dynamic_reference=dict(
            final_q_rad=actual.tolist(),
            target_q_rad=target.tolist(),
            final_contact_dist_m=(dynamic_contact['distance_m'] if dynamic_contact else None),
            final_geom_distance_m=report['result']['final_geom_distance_m']),
        actual_pose=actual_pose,
        target_pose=target_pose,
        actual_minus_target=dict(
            joint_error_rad=(target - actual).tolist(),
            joint_error_deg=np.degrees(target - actual).tolist(),
            fingertip_delta_world_m=tip_delta.tolist(),
            fingertip_delta_normal_m=normal_delta,
            fingertip_delta_normal_mm=normal_delta * 1000,
            fingertip_delta_tangent_m=tangent_delta,
            fingertip_delta_tangent_mm=tangent_delta * 1000),
        interpretation=('This is a static FK comparison with all non-controlled qpos held at the '
                        'recorded initial values. It quantifies the measured end-state geometry; it '
                        'does not reconstruct unrecorded full-body dynamic qpos or prove a successful press.'))
    (directory / 'end_pose_geometry.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    labels = ['actual end', 'target']
    site_gaps = [actual_pose['fingertip_surface_gap_mm'], target_pose['fingertip_surface_gap_mm']]
    geom_gaps = [actual_pose['hand_geom_signed_distance_mm'], target_pose['hand_geom_signed_distance_mm']]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(2)
    axes[0].bar(x, site_gaps, color=['tab:blue', 'tab:orange'])
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('Fingertip site signed distance (mm)')
    axes[0].set_title('Point-to-button surface')
    axes[1].bar(x, geom_gaps, color=['tab:blue', 'tab:orange'])
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel('hand_2distph signed distance (mm)')
    axes[1].set_title('Collision geom distance')
    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
    figure.savefig(directory / 'end_pose_geometry.png', dpi=160)
    plt.close(figure)
    print(json.dumps(dict(
        actual_site_gap_mm=actual_pose['fingertip_surface_gap_mm'],
        target_site_gap_mm=target_pose['fingertip_surface_gap_mm'],
        actual_hand_geom_distance_mm=actual_pose['hand_geom_signed_distance_mm'],
        target_hand_geom_distance_mm=target_pose['hand_geom_signed_distance_mm'],
        actual_minus_target_normal_delta_mm=normal_delta * 1000,
        actual_minus_target_tangent_delta_mm=tangent_delta * 1000,
        formal_actual_contacts=actual_contacts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
