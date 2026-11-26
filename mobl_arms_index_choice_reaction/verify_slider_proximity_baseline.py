"""离线核验 201 个无载状态的独立近距离观测，不运行主仿真。"""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from diagnose_slider_distance_contact import make_proximity_observer
from diagnose_slider_distance_query import query


def main():
    base = Path(__file__).resolve().parent
    out = base.parent / 'artifacts/trajectory_check_2026-10-22'
    source = out / 'slider_distance_query_audit.json'
    baseline = out / 'button_slider_full_baseline.json'
    watched = [source, baseline, base / 'simulation.xml',
               base.parent / 'mobl_arms_index_pointing/simulation.xml']
    original_bytes = [p.read_bytes() for p in watched]
    audit = json.loads(original_bytes[0])
    original_rows = json.loads(original_bytes[1])['samples']
    model, data, gids = make_proximity_observer(base, 0.005)
    mesh = int(model.geom_dataid[gids[0]])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start:start + count].astype(float)
    rows = []
    for frame, original in zip(audit['all_samples'], original_rows):
        assert frame['step'] == original['step']
        assert frame['original_distance_m'] == original['finger_button_distance_m']
        for i, geom in enumerate(frame['geoms']):
            data.mocap_pos[i] = geom['position_m']
            mujoco.mju_mat2Quat(data.mocap_quat[i], np.array(geom['rotation']))
        mujoco.mj_forward(model, data)
        for i, geom in enumerate(frame['geoms']):
            np.testing.assert_allclose(data.geom_xpos[gids[i]], geom['position_m'], atol=1e-14, rtol=0)
            np.testing.assert_allclose(data.geom_xmat[gids[i]], geom['rotation'], atol=1e-14, rtol=0)
        # An independent distance witness: the support vertex must project inside
        # the finite button face. Otherwise the plane gap is only a lower bound.
        world_vertices = vertices @ data.geom_xmat[gids[0]].reshape(3, 3).T + data.geom_xpos[gids[0]]
        local = (world_vertices - data.geom_xpos[gids[1]]) @ data.geom_xmat[gids[1]].reshape(3, 3)
        support = local[int(np.argmin(local[:, 2]))]
        inside = bool(np.all(np.abs(support[:2]) <= model.geom_size[gids[1], :2]))
        gap = float(support[2] - model.geom_size[gids[1], 2])
        assert inside and gap > 0
        np.testing.assert_allclose(gap, frame['mesh_button_front_separating_plane_gap_m'], atol=1e-14, rtol=0)
        contacts = [c for c in data.contact if {int(c.geom1), int(c.geom2)} == set(gids)]
        assert len(contacts) == 1
        c = contacts[0]
        assert data.nefc == 0 and c.efc_address == -1 and c.includemargin == 0
        assert c.dist > 0 and abs(float(c.dist) - gap) < 1e-8
        assert not any(w.number for w in data.warning)
        rows.append(dict(step=frame['step'], time_s=frame['time_s'],
                         original_distance_query_m=frame['original_distance_m'],
                         observer_distance_query=query(model, data, *gids, 0.01),
                         proximity_distance_m=float(c.dist), separating_plane_gap_m=gap,
                         support_vertex_button_local_m=support.tolist(),
                         support_projection_inside_button_face=inside,
                         contact_position_m=c.pos.tolist(), contact_normal=c.frame[:3].tolist(),
                         nefc=int(data.nefc), efc_address=int(c.efc_address)))
    assert len(rows) == len(audit['all_samples']) == len(original_rows) == 201
    assert [p.read_bytes() for p in watched] == original_bytes
    summary = dict(samples=len(rows), proximity_positive_samples=sum(r['proximity_distance_m'] > 0 for r in rows),
                   original_zero_steps=[r['step'] for r in rows if r['original_distance_query_m'] == 0],
                   observer_query_zero_steps=[r['step'] for r in rows if r['observer_distance_query']['distance_m'] == 0],
                   min_proximity_distance_m=min(r['proximity_distance_m'] for r in rows),
                   max_contact_vs_plane_difference_m=max(abs(r['proximity_distance_m'] - r['separating_plane_gap_m']) for r in rows),
                   support_projection_inside_face_samples=sum(r['support_projection_inside_button_face'] for r in rows),
                   active_constraint_count_max=max(r['nefc'] for r in rows),
                   warnings=0, formal_xml_and_input_artifacts_unchanged=True,
                   main_simulation_executed=False, diagnostic_checks_passed=True)
    report = dict(experiment=__doc__, mujoco_version=mujoco.__version__, summary=summary, samples=rows,
                  observer=dict(pair_margin_m=0.005, pair_gap_m=0.005, comparison_tolerance_m=1e-8),
                  input_sha256={str(p.relative_to(base.parent)): hashlib.sha256(b).hexdigest()
                                for p, b in zip(watched[:2], original_bytes[:2])},
                  limitations='Saved separated baseline states only. No press/release or near-zero '
                              'or penetration validation. Offline inputs are unchanged; no claim '
                              'of dynamically integrating an observer into the main controller.')
    path = out / 'slider_proximity_baseline_verification.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    times = [r['time_s'] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(times, [r['original_distance_query_m'] * 1000 for r in rows], '.', label='Original distance query')
    axes[0].plot(times, [r['proximity_distance_m'] * 1000 for r in rows], label='Independent proximity observer')
    axes[0].set(ylabel='Distance (mm)', title='201 saved no-load states: independent observer verification')
    axes[0].legend()
    axes[1].plot(times, [(r['proximity_distance_m'] - r['separating_plane_gap_m']) * 1e9 for r in rows])
    axes[1].set(xlabel='Time (s)', ylabel='Observer minus geometry (nm)')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
