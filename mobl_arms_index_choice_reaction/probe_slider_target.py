"""临时按钮滑块的静态目标探针；不推进动力学，不改正式 XML。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from experiment_small_normal_press import build_experiment_xml, collision_pair_audit
from simulator import IndexSimulator


def main():
    base = Path(__file__).resolve().parent
    root = base.parent
    output_dir = root / 'artifacts/trajectory_check_2026-10-22'
    panel = base / 'simulation.xml'
    pointing = root / 'mobl_arms_index_pointing/simulation.xml'
    originals = [p.read_bytes() for p in (panel, pointing)]
    scan = json.loads((output_dir / 'filtered_press_target_scan.json').read_text())
    candidate = scan['selected_candidate_by_tangential_alignment_within_0_2_to_0_4mm_clearance']
    button_name = f"button-{scan['target_button']}"
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', panel)
        sim.reset()
    try:
        source = sim.model
        initial = sim.data.qpos.copy()
        fixture = {n: float(initial[source.jnt_qposadr[source.joint(n).id]])
                   for n in ('head_rot', 'elv_angle', 'shoulder_elv', 'pro_sup')}
        pairs = [tuple(e['body'] for e in pair) for pair in collision_pair_audit(source)['pairs']]
        with tempfile.TemporaryDirectory(prefix='slider_target_probe_') as directory:
            xml = Path(directory) / 'probe.xml'
            build_experiment_xml(panel, pointing, xml, fixture, pairs)
            tree = ET.parse(xml)
            body = tree.getroot().find(f"./worldbody/body[@name='{button_name}']")
            ET.SubElement(body, 'joint', name='probe_button_slide', type='slide',
                          axis='0 0 -1', limited='true', range='0 0.005',
                          stiffness='1000', springref='0', damping='10')
            tree.write(xml, encoding='utf-8')
            model = mujoco.MjModel.from_xml_path(str(xml))
            data = mujoco.MjData(model)
            # Inserting a slider changes qpos addresses: map each named joint block.
            for jid in range(source.njnt):
                target = model.joint(source.joint(jid).name).id
                start = int(source.jnt_qposadr[jid])
                end = int(source.jnt_qposadr[jid + 1]) if jid + 1 < source.njnt else source.nq
                dest = int(model.jnt_qposadr[target])
                data.qpos[dest:dest + end - start] = initial[start:end]
                np.testing.assert_array_equal(data.qpos[dest:dest + end - start], initial[start:end])
            finger = model.geom('hand_2distph').id
            button = model.geom(button_name).id
            original_margin = float(model.geom_margin[finger])
            model.geom_margin[finger] = 0
            joints = [model.joint(n).id for n in ('shoulder_rot', 'elbow_flexion')]
            qadr = model.jnt_qposadr[joints]
            dofs = model.jnt_dofadr[joints]
            q = np.radians([candidate['shoulder_rot_deg'], candidate['elbow_flexion_deg']])
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            normal = data.geom_xmat[button].reshape(3, 3)[:, 2].copy()
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, model.site('fingertip').id)
            records = []
            slider = model.joint('probe_button_slide').id
            for offset in (0, 0.3, 0.8, 1.0, 1.5, 2.0):
                delta = np.linalg.lstsq(jac[:, dofs], -normal * offset / 1000, rcond=None)[0]
                data.qpos[qadr] = q + delta
                mujoco.mj_forward(model, data)
                assert data.qpos[model.jnt_qposadr[slider]] == 0
                contacts = [float(c.dist) * 1000 for c in data.contact
                            if {int(c.geom1), int(c.geom2)} == {finger, button}]
                records.append(dict(inward_site_offset_requested_mm=offset,
                                    joint_delta_deg=np.degrees(delta).tolist(),
                                    geom_distance_mm=float(mujoco.mj_geomDistance(
                                        model, data, finger, button, 10, None)) * 1000,
                                    matching_contact_distance_mm=contacts))
            report = dict(method='static FK only; slider held at q=0; no mj_step',
                          mujoco_version=mujoco.__version__, target_button=button_name,
                          source_nq=source.nq, temporary_nq=model.nq,
                          initial_state_mapping='joint names and full qpos blocks',
                          internal_exclusions=pairs,
                          finger_margin_original_m=original_margin, finger_margin_probe_m=0,
                          slider=dict(axis_local=[0, 0, -1], range_m=[0, 0.005],
                                      stiffness_N_m=1000, damping_Ns_m=10,
                                      inferred_mass_kg=float(model.body_mass[model.body(button_name).id])),
                          samples=records,
                          limitations='Spring/damping/travel are provisional, not calibrated. '
                          'Negative distances are deliberately imposed FK overlap at q_slide=0; '
                          'no dynamic press, return, or success threshold has been validated.')
    finally:
        sim.close()
        assert [p.read_bytes() for p in (panel, pointing)] == originals
    path = output_dir / 'slider_target_probe.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
