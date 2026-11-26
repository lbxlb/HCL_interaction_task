"""扫描末态附近的肩肘姿态，寻找无穿透且接近按钮表面的静态候选点。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mujoco
import numpy as np
import yaml

import experiment_small_normal_press as experiment
from simulator import IndexSimulator


def main():
    base = Path(__file__).resolve().parent
    root = base.parent
    directory = root / 'artifacts/trajectory_check_2026-10-22'
    dynamic_path = directory / 'small_normal_press_filtered_hold200.json'
    reference_path = root / 'artifacts/trajectory_check_2026-10-21/surface_pose_diagnostics.json'
    panel_xml = base / 'simulation.xml'
    pointing_xml = root / 'mobl_arms_index_pointing/simulation.xml'
    dynamic = json.loads(dynamic_path.read_text(encoding='utf-8'))
    reference = json.loads(reference_path.read_text(encoding='utf-8'))['formal_panel']
    target_button = int(yaml.safe_load((base / 'config.yaml').read_text())['target_button'])
    target_local_xy = np.asarray(reference['fingertip_button_local_m'][:2], dtype=float)
    actual_q = np.asarray(dynamic['result']['q_final_rad'], dtype=float)
    target_q = np.asarray(dynamic['press_geometry']['press_target_q_rad'], dtype=float)
    originals = panel_xml.read_bytes(), pointing_xml.read_bytes()
    with contextlib.redirect_stdout(io.StringIO()):
        source = IndexSimulator(base / 'config.yaml', panel_xml)
        source.reset()
    try:
        reset_qpos = source.data.qpos.copy()
        controlled_qpos_addresses = [int(source.model.jnt_qposadr[source.model.joint(name).id])
                                     for name in ('shoulder_rot', 'elbow_flexion')]
        fixture = {name: float(reset_qpos[source.model.jnt_qposadr[source.model.joint(name).id]])
                   for name in ('head_rot', 'elv_angle', 'shoulder_elv', 'pro_sup')}
    finally:
        source.close()
    qpos0 = np.asarray(dynamic['initial_state']['qpos'], dtype=float)
    free_qpos = np.ones(len(qpos0), dtype=bool)
    free_qpos[controlled_qpos_addresses] = False
    np.testing.assert_array_equal(qpos0[free_qpos], reset_qpos[free_qpos])
    np.testing.assert_allclose(qpos0[controlled_qpos_addresses],
                               dynamic['press_geometry']['outside_q_rad'], rtol=0, atol=1e-14)

    shoulder_deg = np.arange(np.degrees(actual_q[0]) - 0.6,
                              np.degrees(actual_q[0]) + 0.6001, 0.05)
    elbow_deg = np.arange(np.degrees(actual_q[1]) - 0.9,
                           np.degrees(actual_q[1]) + 0.9001, 0.05)
    shape = (len(shoulder_deg), len(elbow_deg))
    geom_distance = np.full(shape, np.nan)
    pair_distance = np.full(shape, np.nan)
    site_normal_gap = np.full(shape, np.nan)
    site_tangent_error = np.full(shape, np.nan)
    contact_present = np.zeros(shape, dtype=bool)
    disagreement = np.zeros(shape, dtype=bool)
    records = []
    excluded_pairs = [tuple(entry['body'] for entry in pair)
                      for pair in experiment.collision_pair_audit(source.model)['pairs']]
    try:
        with tempfile.TemporaryDirectory(prefix='filtered_target_scan_') as temp:
            xml = Path(temp) / 'static_scan.xml'
            experiment.build_experiment_xml(panel_xml, pointing_xml, xml, fixture, excluded_pairs)
            model = mujoco.MjModel.from_xml_path(str(xml))
            data = mujoco.MjData(model)
            assert model.nexclude == 5
            button = model.geom(f'button-{target_button}').id
            finger = model.geom('hand_2distph').id
            fingertip = model.site('fingertip').id
            half = model.geom_size[button, :3].copy()
            margin_sum = float(model.geom_margin[button] + model.geom_margin[finger])
            for i, shoulder in enumerate(shoulder_deg):
                for j, elbow in enumerate(elbow_deg):
                    data.qpos[:] = qpos0
                    data.qpos[model.jnt_qposadr[model.joint('shoulder_rot').id]] = np.radians(shoulder)
                    data.qpos[model.jnt_qposadr[model.joint('elbow_flexion').id]] = np.radians(elbow)
                    mujoco.mj_forward(model, data)
                    d_geom = float(mujoco.mj_geomDistance(model, data, finger, button, 10.0, None))
                    geom_distance[i, j] = d_geom * 1000
                    matching = [data.contact[k] for k in range(data.ncon)
                                if {int(data.contact[k].geom1), int(data.contact[k].geom2)}
                                == {finger, button}]
                    d_pair = min((float(c.dist) for c in matching), default=np.nan)
                    if np.isfinite(d_pair):
                        contact_present[i, j] = True
                        pair_distance[i, j] = d_pair * 1000
                        disagreement[i, j] = abs(d_geom - d_pair) > 1e-5
                    button_rotation = data.geom_xmat[button].reshape(3, 3)
                    site_local = button_rotation.T @ (data.site_xpos[fingertip] - data.geom_xpos[button])
                    # Positive means the site lies outside the button front face.
                    site_normal_gap[i, j] = (site_local[2] - half[2]) * 1000
                    site_tangent_error[i, j] = np.linalg.norm(site_local[:2] * 1000 - target_local_xy * 1000)
                    if np.isfinite(d_pair) and d_pair >= 0 and d_pair <= margin_sum:
                        records.append(dict(
                            shoulder_rot_deg=float(shoulder), elbow_flexion_deg=float(elbow),
                            geom_distance_mm=float(d_geom * 1000),
                            contact_pair_distance_mm=float(d_pair * 1000),
                            fingertip_site_front_offset_mm=float(site_normal_gap[i, j]),
                            fingertip_site_tangent_error_from_reference_mm=float(site_tangent_error[i, j]),
                            fingertip_button_local_m=site_local.tolist(),
                            matching_contact_count=len(matching),
                            distance_disagreement=bool(disagreement[i, j])))
    finally:
        assert panel_xml.read_bytes() == originals[0]
        assert pointing_xml.read_bytes() == originals[1]

    if not records:
        raise RuntimeError('扫描范围内没有非负的 margin 接触候选点。')
    records.sort(key=lambda r: (r['contact_pair_distance_mm'],
                                r['fingertip_site_tangent_error_from_reference_mm']))
    closest_surface = records[0]
    clearance_band = [record for record in records
                      if 0.2 <= record['contact_pair_distance_mm'] <= 0.4]
    if not clearance_band:
        raise RuntimeError('扫描范围内没有位于 0.2–0.4 mm 正间距带内的非穿透候选点。')
    selected = min(clearance_band,
                   key=lambda r: r['fingertip_site_tangent_error_from_reference_mm'])
    assert selected['geom_distance_mm'] > 0 and not selected['distance_disagreement']
    assert np.isfinite(geom_distance).all() and np.isfinite(site_normal_gap).all()
    ii_actual = int(np.argmin(np.abs(shoulder_deg - np.degrees(actual_q[0])))
                    + 0)
    jj_actual = int(np.argmin(np.abs(elbow_deg - np.degrees(actual_q[1])))
                    + 0)
    ii_target = int(np.argmin(np.abs(shoulder_deg - np.degrees(target_q[0])))
                    + 0)
    jj_target = int(np.argmin(np.abs(elbow_deg - np.degrees(target_q[1])))
                    + 0)
    summary = dict(
        method='static FK grid; non-controlled qpos held at reset values; no mj_step',
        model_changes='temporary XML only: same five internal body exclusions and candidate equalities',
        target_button=target_button,
        grid=dict(shoulder_rot_range_deg=[float(shoulder_deg[0]), float(shoulder_deg[-1])],
                  elbow_flexion_range_deg=[float(elbow_deg[0]), float(elbow_deg[-1])],
                  resolution_deg=0.05, shape=list(shape), points=int(np.prod(shape))),
        geometry=dict(button_half_size_m=half.tolist(), combined_geom_margin_mm=margin_sum * 1000,
                      site_tangent_reference_local_mm=(target_local_xy * 1000).tolist()),
        source_poses=dict(
            dynamic_actual=dict(shoulder_rot_deg=float(np.degrees(actual_q[0])),
                                elbow_flexion_deg=float(np.degrees(actual_q[1]))),
            press_target=dict(shoulder_rot_deg=float(np.degrees(target_q[0])),
                              elbow_flexion_deg=float(np.degrees(target_q[1])))),
        candidate_count=len(records),
        fingertip_site_outside_candidates_count=sum(r['fingertip_site_front_offset_mm'] >= 0
                                                    for r in records),
        fingertip_site_on_or_behind_face_candidates_count=sum(
            r['fingertip_site_front_offset_mm'] < 0 for r in records),
        candidate_definition=('matching finger-button contact exists; contact.dist >= 0; '
                              'contact.dist <= combined geom margins'),
        closest_surface_candidate=closest_surface,
        selected_candidate_by_tangential_alignment_within_0_2_to_0_4mm_clearance=selected,
        selected_clearance_band_mm=[0.2, 0.4],
        clearance_band_is_only_a_diagnostic_ranking_cutoff=True,
        top_candidates=records[:10],
        actual_grid_sample=dict(geom_distance_mm=float(geom_distance[ii_actual, jj_actual]),
                                contact_pair_distance_mm=(float(pair_distance[ii_actual, jj_actual])
                                                         if contact_present[ii_actual, jj_actual] else None),
                                site_front_offset_mm=float(site_normal_gap[ii_actual, jj_actual]),
                                site_tangent_error_mm=float(site_tangent_error[ii_actual, jj_actual])),
        target_grid_sample=dict(geom_distance_mm=float(geom_distance[ii_target, jj_target]),
                                contact_pair_distance_mm=(float(pair_distance[ii_target, jj_target])
                                                         if contact_present[ii_target, jj_target] else None),
                                site_front_offset_mm=float(site_normal_gap[ii_target, jj_target]),
                                site_tangent_error_mm=float(site_tangent_error[ii_target, jj_target])),
        mj_geomDistance_contact_disagreement_count=int(disagreement.sum()),
        candidates=records,
        interpretation=('Closest-surface ranking uses contact-pair distance. The selected candidate '
                        'minimizes tangential error within a diagnostic 0.2 to 0.4 mm clearance band. '
                        'Site front offset is positive outside. Grid samples near the old target are '
                        'rounded poses, not the exact target. These are static pairwise candidates, '
                        'not full-body collision-free or dynamically validated targets.'))
    (directory / 'filtered_press_target_scan.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    extent = [elbow_deg[0] - 0.025, elbow_deg[-1] + 0.025,
              shoulder_deg[0] - 0.025, shoulder_deg[-1] + 0.025]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [
        (geom_distance, 'mj_geomDistance (mm)', 'coolwarm', TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)),
        (pair_distance, 'contact.dist (mm; only matching pair)', 'viridis', None),
        (site_normal_gap, 'fingertip site front offset from face (mm)', 'coolwarm', TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)),
        (site_tangent_error, 'site tangential error from reference (mm)', 'magma', None)]
    for ax, (grid, title, cmap, norm) in zip(axes.flat, panels):
        image = ax.imshow(grid, origin='lower', aspect='auto', extent=extent, cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax, label=title)
        ax.scatter(np.degrees(actual_q[1]), np.degrees(actual_q[0]), marker='o',
                   facecolors='none', edgecolors='white', s=80, label='dynamic actual')
        ax.scatter(np.degrees(target_q[1]), np.degrees(target_q[0]), marker='x',
                   color='black', s=70, label='press target')
        ax.scatter(selected['elbow_flexion_deg'], selected['shoulder_rot_deg'],
                   marker='*', color='yellow', edgecolors='black', s=150, label='selected')
        ax.set_xlabel('Elbow flexion (deg)')
        ax.set_ylabel('Shoulder rotation (deg)')
        ax.set_title(title)
        ax.legend(fontsize=7, loc='best')
    fig.savefig(directory / 'filtered_press_target_scan.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(
        grid=summary['grid'], candidate_count=len(records),
        closest_surface_candidate=closest_surface,
        selected_candidate_by_tangential_alignment_within_0_2_to_0_4mm_clearance=selected,
        top_candidates=records[:5],
        actual_grid_sample=summary['actual_grid_sample'],
        target_grid_sample=summary['target_grid_sample'],
        mj_geomDistance_contact_disagreement_count=int(disagreement.sum())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
