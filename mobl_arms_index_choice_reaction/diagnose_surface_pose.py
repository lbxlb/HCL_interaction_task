"""检查近表面姿态的手部碰撞几何与候选关节等式残差。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simulator import IndexSimulator


def pose_data(model, data, pose):
    mujoco.mj_resetData(model, data)
    for name, angle in pose.items():
        jid = model.joint(name).id
        data.qpos[model.jnt_qposadr[jid]] = angle
    mujoco.mj_forward(model, data)


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f'geom_{gid}'


def inspect_contact(model, data, pose, target_name):
    pose_data(model, data, pose)
    target = model.geom(target_name).id
    hand = [gid for gid in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or '').startswith('hand_')]
    distances = [dict(hand_geom=geom_name(model, gid),
                      distance_m=float(mujoco.mj_geomDistance(model, data, gid, target, 10.0, None)))
                 for gid in hand]
    distances.sort(key=lambda item: item['distance_m'])
    fingertip = data.site_xpos[model.site('fingertip').id].copy()
    button_rotation = data.geom_xmat[target].reshape(3, 3).copy()
    button_center = data.geom_xpos[target].copy()
    fingertip_local = button_rotation.T @ (fingertip - button_center)
    closest_segment = np.zeros(6, dtype=np.float64)
    finger_id = model.geom('hand_2distph').id
    finger_button_distance = float(mujoco.mj_geomDistance(
        model, data, finger_id, target, 10.0, closest_segment))
    contact_geometry = []
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if target in (contact.geom1, contact.geom2):
            other = contact.geom2 if contact.geom1 == target else contact.geom1
            contacts.append(dict(other_geom=geom_name(model, other), distance_m=float(contact.dist)))
            normal = contact.frame[:3].copy()
            contact_local = button_rotation.T @ (contact.pos - button_center)
            contact_geometry.append(dict(
                other_geom=geom_name(model, other),
                contact_point_world_m=contact.pos.tolist(),
                contact_point_button_local_m=contact_local.tolist(),
                contact_normal_world=normal.tolist(),
                normal_dot_button_front=float(np.dot(normal, button_rotation[:, 2])),
                abs_normal_alignment_with_button_front=float(abs(np.dot(normal, button_rotation[:, 2])))))
    return dict(fingertip_world_m=data.site_xpos[model.site('fingertip').id].tolist(),
                button_front_normal_world=button_rotation[:, 2].tolist(),
                fingertip_button_local_m=fingertip_local.tolist(),
                button_half_size_m=model.geom_size[target, :3].tolist(),
                finger_button_nearest_segment_world_m=closest_segment.reshape(2, 3).tolist(),
                finger_button_signed_distance_m=finger_button_distance,
                formal_model_equality_count=int(model.neq),
                finger_to_button_distances=distances,
                overlapping_hand_geoms=[x for x in distances if x['distance_m'] < 0],
                button_contacts=contacts, button_contact_geometry=contact_geometry,
                total_contact_count=int(data.ncon),
                note='所有 hand_ 前缀几何体均计算距离；距离为负代表几何重叠，实际接触还受碰撞过滤影响。')


def make_candidate_model(panel_xml, pointing_xml, directory):
    panel = ET.fromstring(panel_xml.read_bytes())
    pointing = ET.parse(pointing_xml).getroot()
    assert panel.find('equality') is None, '面板正式模型已有 equality，请复核'
    candidates = pointing.findall('./equality/joint')
    assert len(candidates) == 12
    eq = ET.SubElement(panel, 'equality')
    for item in candidates:
        eq.append(copy.deepcopy(item))
    for asset in panel.findall('./asset/*'):
        if 'file' in asset.attrib:
            asset.set('file', str((panel_xml.parent / asset.get('file')).resolve()))
    path = Path(directory) / 'panel_candidate_equalities.xml'
    ET.ElementTree(panel).write(path, encoding='utf-8')
    return path, [dict(item.attrib) for item in candidates]


def measure_residuals(model, data, pose, candidates):
    pose_data(model, data, pose)
    result = []
    for item in candidates:
        eqid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, item['name'])
        rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                              & (data.efc_id == eqid))
        result.append(dict(name=item['name'], active_rows=int(len(rows)),
                           residual_rad=float(data.efc_pos[rows[0]]) if len(rows) else None))
    return result


def main():
    base = Path(__file__).resolve().parent
    root = base.parent
    scan_path = root / 'artifacts/trajectory_check_2026-10-21/shoulder_elbow_reachability.json'
    output = root / 'artifacts/trajectory_check_2026-10-21/surface_pose_diagnostics.json'
    scan = json.loads(scan_path.read_text(encoding='utf-8'))
    nearest = scan['nearest_surface']
    pose = {'shoulder_rot': nearest['shoulder_rot_rad'],
            'elbow_flexion': nearest['elbow_flexion_rad']}
    panel_xml = base / 'simulation.xml'
    pointing_xml = root / 'mobl_arms_index_pointing/simulation.xml'
    originals = panel_xml.read_bytes(), pointing_xml.read_bytes()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', panel_xml)
        sim.reset()
    try:
        formal = inspect_contact(sim.model, sim.data, pose, f"button-{scan['target_button']}")
        with tempfile.TemporaryDirectory(prefix='surface_pose_eq_') as directory:
            temp_xml, candidates = make_candidate_model(panel_xml, pointing_xml, directory)
            candidate_model = mujoco.MjModel.from_xml_path(str(temp_xml))
            candidate_data = mujoco.MjData(candidate_model)
            residuals = measure_residuals(candidate_model, candidate_data, pose, candidates)
        assert panel_xml.read_bytes() == originals[0]
        assert pointing_xml.read_bytes() == originals[1]
        report = dict(
            pose=dict(shoulder_rot_deg=nearest['shoulder_rot_deg'],
                      elbow_flexion_deg=nearest['elbow_flexion_deg']),
            formal_panel=formal,
            candidate_equalities_source=str(pointing_xml.relative_to(root)),
            candidate_equalities_count=len(candidates),
            candidate_equality_residuals=residuals,
            max_abs_candidate_residual_rad=max(abs(x['residual_rad']) for x in residuals
                                               if x['residual_rad'] is not None),
            interpretation='等式残差来自仅用于诊断的临时模型；这些约束不在面板正式 XML 中。')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(
            pose_deg=report['pose'],
            formal_model_equality_count=formal['formal_model_equality_count'],
            overlapping_hand_geoms=formal['overlapping_hand_geoms'],
            button_contacts=formal['button_contacts'],
            candidate_equality_residual_count=len(residuals),
            max_abs_candidate_residual_rad=report['max_abs_candidate_residual_rad'],
            output=str(output)), ensure_ascii=False, indent=2))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
