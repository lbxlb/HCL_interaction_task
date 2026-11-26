"""固定三个最小姿态，用显式接触对交叉核验距离查询。"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnose_slider_distance_query import query


def make_proximity_observer(base, margin):
    """两几何体独立观测器；margin=gap，只用于分离姿态的近距离记录。"""
    xml = ET.Element('mujoco', model='minimal_explicit_pair_probe')
    asset = ET.SubElement(xml, 'asset')
    ET.SubElement(asset, 'mesh', name='finger_mesh',
                  file=str(base / 'assets/hand_2distph_pointing.stl'))
    world = ET.SubElement(xml, 'worldbody')
    for index in range(2):
        body = ET.SubElement(world, 'body', name=f'probe_{index}', mocap='true')
        if index == 0:
            ET.SubElement(body, 'geom', name='finger', type='mesh', mesh='finger_mesh')
        else:
            ET.SubElement(body, 'geom', name='button', type='box', size='0.025 0.025 0.01')
    contact = ET.SubElement(xml, 'contact')
    ET.SubElement(contact, 'pair', geom1='finger', geom2='button',
                  margin=str(margin), gap=str(margin))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    model.geom_pos[:] = 0
    model.geom_quat[:] = [1, 0, 0, 0]
    model.geom_sameframe[:] = 0
    return model, mujoco.MjData(model), [model.geom(n).id for n in ('finger', 'button')]


def main():
    base = Path(__file__).resolve().parent
    out = base.parent / 'artifacts/trajectory_check_2026-10-22'
    paths = [base / 'simulation.xml', base.parent / 'mobl_arms_index_pointing/simulation.xml']
    originals = [p.read_bytes() for p in paths]
    prior = json.loads((out / 'slider_distance_minimal_audit.json').read_text())
    rows = []
    for frame in prior['samples']:
        for margin in (0.0, 0.005, 0.01):
            # gap=margin requests proximity records without active constraints for
            # positive separation. Explicit pair avoids automatic body-pair filtering.
            model, data, gids = make_proximity_observer(base, margin)
            for index, geom in enumerate(frame['source_world_geoms']):
                data.mocap_pos[index] = geom['position_m']
                mujoco.mju_mat2Quat(data.mocap_quat[index], np.array(geom['rotation']))
            mujoco.mj_forward(model, data)
            for index, geom in enumerate(frame['source_world_geoms']):
                np.testing.assert_allclose(data.geom_xpos[gids[index]], geom['position_m'], atol=1e-14, rtol=0)
                np.testing.assert_allclose(data.geom_xmat[gids[index]], geom['rotation'], atol=1e-14, rtol=0)
            before = query(model, data, *gids, 0.01)
            mujoco.mj_collision(model, data)
            contacts = [dict(distance_m=float(c.dist), position_m=c.pos.tolist(),
                             normal=c.frame[:3].tolist(), includemargin_m=float(c.includemargin),
                             efc_address=int(c.efc_address)) for c in data.contact]
            after = query(model, data, *gids, 0.01)
            assert before == after
            assert data.nefc == 0, 'Diagnostic proximity records must not add constraints'
            assert not any(w.number for w in data.warning)
            if margin == 0:
                assert not contacts
            else:
                assert contacts
                for c in contacts:
                    assert c['distance_m'] > 0 and c['efc_address'] == -1
                    assert c['includemargin_m'] == 0
                    assert abs(c['distance_m'] - frame['separating_plane_gap_m']) < 1e-8
            rows.append(dict(step=frame['step'], pair_margin_m=margin, pair_gap_m=margin,
                             distance_query=before, contacts=contacts,
                             separating_plane_gap_m=frame['separating_plane_gap_m'],
                             query_unchanged_after_collision=True, nefc_after_forward=int(data.nefc),
                             warnings=[int(w.number) for w in data.warning]))
    assert [p.read_bytes() for p in paths] == originals
    summary = [dict(step=r['step'], margin_mm=r['pair_margin_m'] * 1000,
                    query_mm=r['distance_query']['distance_m'] * 1000,
                    contact_distances_mm=[c['distance_m'] * 1000 for c in r['contacts']],
                    plane_gap_mm=r['separating_plane_gap_m'] * 1000) for r in rows]
    report = dict(experiment=__doc__, mujoco_version=mujoco.__version__, samples=rows, summary=summary,
                  formal_xml_unchanged=True,
                  diagnostic_checks_passed=True, comparison_tolerance_m=1e-8,
                  max_contact_vs_plane_difference_m=max(
                      abs(c['distance_m'] - r['separating_plane_gap_m'])
                      for r in rows for c in r['contacts']),
                  limitations='Three separated poses, explicit pairs and diagnostic margin/gap only. '
                              'Proximity contacts are not physical touch or button success. '
                              'No mj_step or force-control test; not a formal-model repair.')
    (out / 'slider_distance_contact_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
