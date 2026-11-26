"""仅保留手指网格和按钮盒体，重现已保存的三个相邻静态姿态。"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnose_slider_distance_query import query


def main():
    base = Path(__file__).resolve().parent
    out = base.parent / 'artifacts/trajectory_check_2026-10-22'
    formal = [base / 'simulation.xml', base.parent / 'mobl_arms_index_pointing/simulation.xml']
    originals = [p.read_bytes() for p in formal]
    audit = json.loads((out / 'slider_distance_query_audit.json').read_text())
    first_bad = audit['summary']['original_zero_steps'][0]
    frames = [r for r in audit['all_samples'] if r['step'] in (first_bad - 1, first_bad, first_bad + 1)]
    results = []
    for frame in frames:
        xml = ET.Element('mujoco', model=f"distance_minimal_step_{frame['step']}")
        ET.SubElement(xml, 'compiler', angle='radian')
        assets = ET.SubElement(xml, 'asset')
        ET.SubElement(assets, 'mesh', name='finger_mesh',
                      file=str(base / 'assets/hand_2distph_pointing.stl'))
        world = ET.SubElement(xml, 'worldbody')
        for index, geom in enumerate(frame['geoms']):
            body = ET.SubElement(world, 'body', name=f'body_{index}', mocap='true')
            if index == 0:
                ET.SubElement(body, 'geom', name='finger', type='mesh', mesh='finger_mesh', margin='0')
            else:
                ET.SubElement(body, 'geom', name='button', type='box', size='0.025 0.025 0.01', margin='0')
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        # The mesh compiler shifts/rotates mesh geoms. Undo that local geom pose;
        # the saved world transform already includes the original compiler offset.
        model.geom_pos[:] = 0
        model.geom_quat[:] = [1, 0, 0, 0]
        # Compiled sameframe optimization encodes the old mesh/inertial pose.
        model.geom_sameframe[:] = 0
        data = mujoco.MjData(model)
        gids = [model.geom(n).id for n in ('finger', 'button')]
        target_pos = np.array([g['position_m'] for g in frame['geoms']])
        target_rot = np.array([g['rotation'] for g in frame['geoms']])
        for index in range(2):
            data.mocap_pos[index] = target_pos[index]
            mujoco.mju_mat2Quat(data.mocap_quat[index], target_rot[index])
        mujoco.mj_forward(model, data)
        normal_query = query(model, data, *gids, 10.0)
        rotation_error = float(np.max(np.abs(data.geom_xmat[gids] - target_rot)))
        position_error = float(np.max(np.abs(data.geom_xpos[gids] - target_pos)))
        assert position_error < 1e-14 and rotation_error < 1e-14
        # A matrix -> quaternion -> matrix roundtrip can perturb a near-degenerate
        # solver input. For query-only diagnosis also restore the exact saved matrices.
        data.geom_xpos[gids] = target_pos
        data.geom_xmat[gids] = target_rot
        exact_query = query(model, data, *gids, 10.0)
        vertices = model.mesh_vert.astype(float)
        world_vertices = vertices @ target_rot[0].reshape(3, 3).T + target_pos[0]
        normal = target_rot[1].reshape(3, 3)[:, 2]
        gap = float(np.min(world_vertices @ normal) - target_pos[1] @ normal - model.geom_size[gids[1], 2])
        np.testing.assert_allclose(gap, frame['mesh_button_front_separating_plane_gap_m'], atol=1e-14, rtol=0)
        results.append(dict(step=frame['step'], original_full_model_distance_m=frame['original_distance_m'],
                            minimal_forward_query=normal_query, minimal_exact_transform_query=exact_query,
                            world_position_roundtrip_error_m=position_error,
                            world_rotation_roundtrip_error=rotation_error,
                            separating_plane_gap_m=gap, nq=model.nq, nv=model.nv,
                            ngeom=model.ngeom, nu=model.nu,
                            source_world_geoms=frame['geoms'],
                            exact_transform_reproduces_original=exact_query['distance_m'] == frame['original_distance_m']))
        # Binary snapshot retains the compiled mesh and adjusted local geom poses.
        model.body_pos[1:3] = data.mocap_pos
        model.body_quat[1:3] = data.mocap_quat
        snapshot = out / f"distance_minimal_step_{frame['step']}.mjb"
        mujoco.mj_saveModel(model, str(snapshot))
        reloaded = mujoco.MjModel.from_binary_path(str(snapshot))
        fresh = mujoco.MjData(reloaded)
        mujoco.mj_forward(reloaded, fresh)
        reloaded_query = query(reloaded, fresh, *gids, 10.0)
        assert reloaded_query['distance_m'] == normal_query['distance_m']
        results[-1]['binary_reload_query'] = reloaded_query
        results[-1]['snapshot_file'] = snapshot.name
    assert [p.read_bytes() for p in formal] == originals
    report = dict(experiment=__doc__, mujoco_version=mujoco.__version__, samples=results,
                  formal_xml_unchanged=True,
                  limitations='Three separated static states only. Exact-transform variant overwrites '
                              'derived geom arrays for distance query only; no collision or dynamics '
                              'claim is made for those overwritten arrays. Roundtrip variant uses '
                              'normal mj_forward. No formal model changes or press input.')
    path = out / 'slider_distance_minimal_audit.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps([dict(step=r['step'], full_distance_m=r['original_full_model_distance_m'],
                           minimal_distance_m=r['minimal_forward_query']['distance_m'],
                           reload_distance_m=r['binary_reload_query']['distance_m'],
                           plane_gap_m=r['separating_plane_gap_m']) for r in results], indent=2))


if __name__ == '__main__':
    main()
