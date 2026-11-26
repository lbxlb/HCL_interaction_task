"""重放无接触基线，对比距离查询上限、几何顺序和独立状态重算。"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from test_button_slider_full_baseline import main as run_baseline


def query(model, data, first, second, bound):
    segment = np.full(6, np.nan)
    distance = float(mujoco.mj_geomDistance(model, data, first, second, bound, segment))
    return dict(distmax_m=bound, distance_m=distance,
                segment_world_m=segment.tolist(),
                segment_length_m=float(np.linalg.norm(segment[3:] - segment[:3])))


def main():
    records = []

    def inspect(model, data, finger, button, row):
        variants = {}
        for bound in (0.01, 0.1, 1.0, 10.0):
            variants[str(bound)] = query(model, data, finger, button, bound)
        reverse = query(model, data, button, finger, 10.0)
        repeats = [query(model, data, finger, button, 10.0)['distance_m'] for _ in range(3)]
        # Rebuild kinematics without carrying any copied derived arrays.
        fresh = mujoco.MjData(model)
        fresh.qpos[:] = data.qpos
        fresh.qvel[:] = data.qvel
        fresh.ctrl[:] = data.ctrl
        fresh.act[:] = data.act
        fresh.time = data.time
        mujoco.mj_forward(model, fresh)
        fresh_query = query(model, fresh, finger, button, 10.0)
        # Independent separating-plane check, not an exact closest-distance query.
        assert model.geom_type[finger] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_type[button] == mujoco.mjtGeom.mjGEOM_BOX
        mesh = int(model.geom_dataid[finger])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start + count].astype(float)
        world_vertices = vertices @ data.geom_xmat[finger].reshape(3, 3).T + data.geom_xpos[finger]
        normal = data.geom_xmat[button].reshape(3, 3)[:, 2]
        front_plane = float(data.geom_xpos[button] @ normal + model.geom_size[button, 2])
        plane_gap = float(np.min(world_vertices @ normal) - front_plane)
        records.append(dict(
            step=row['step'], time_s=row['time_s'], original_distance_m=row['finger_button_distance_m'],
            queries=variants, reverse_query=reverse, same_state_repeats_m=repeats,
            fresh_state_query=fresh_query,
            mesh_button_front_separating_plane_gap_m=plane_gap,
            fresh_geometry_max_difference=float(max(
                np.max(np.abs(fresh.geom_xpos[[finger, button]] - data.geom_xpos[[finger, button]])),
                np.max(np.abs(fresh.geom_xmat[[finger, button]] - data.geom_xmat[[finger, button]])))),
            geoms=[dict(name=model.geom(g).name, type=mujoco.mjtGeom(model.geom_type[g]).name,
                        position_m=data.geom_xpos[g].tolist(), rotation=data.geom_xmat[g].tolist())
                   for g in (finger, button)],
            qpos=data.qpos.tolist(), button_contacts=row['button_contacts']))

    run_baseline(observation_hook=inspect, output_stem='button_slider_distance_replay_baseline')
    bad = [r['step'] for r in records if r['original_distance_m'] == 0]
    selected = {s + d for s in bad for d in (-1, 0, 1)}
    summary = dict(
        samples=len(records), original_zero_steps=bad,
        zero_count_by_distmax={str(b): sum(r['queries'][str(b)]['distance_m'] == 0 for r in records)
                               for b in (0.01, 0.1, 1.0, 10.0)},
        repeat_mismatch_samples=sum(any(v != r['original_distance_m'] for v in r['same_state_repeats_m'])
                                    for r in records),
        fresh_state_mismatch_samples=sum(abs(r['fresh_state_query']['distance_m'] - r['original_distance_m']) > 1e-12
                                        for r in records),
        reverse_zero_samples=sum(r['reverse_query']['distance_m'] == 0 for r in records),
        max_fresh_geometry_difference=max(r['fresh_geometry_max_difference'] for r in records),
        min_separating_plane_gap_m=min(r['mesh_button_front_separating_plane_gap_m'] for r in records),
        min_separating_plane_gap_at_zero_queries_m=min(
            (r['mesh_button_front_separating_plane_gap_m'] for r in records
             if r['original_distance_m'] == 0), default=None),
        max_positive_query_difference_m=max(
            max((v['distance_m'] for v in r['queries'].values() if v['distance_m'] > 0), default=0)
            - min((v['distance_m'] for v in r['queries'].values() if v['distance_m'] > 0), default=0)
            for r in records),
    )
    report = dict(experiment=__doc__, mujoco_version=mujoco.__version__, summary=summary,
                  selected_anomalies_and_neighbors=[r for r in records if r['step'] in selected],
                  all_samples=records,
                  limitations='Query comparisons diagnose measurement only; no model/solver parameter '
                              'changes and no press input. No automatic replacement of zero values.')
    path = Path(__file__).resolve().parents[1] / 'artifacts/trajectory_check_2026-10-22/slider_distance_query_audit.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    fig, ax = plt.subplots(figsize=(9, 4))
    times = [r['time_s'] for r in records]
    ax.plot(times, [r['original_distance_m'] * 1000 for r in records],
            '.', label='Distance query (raw)')
    ax.plot(times, [r['mesh_button_front_separating_plane_gap_m'] * 1000 for r in records],
            label='Independent separating-plane gap')
    ax.set(xlabel='Time (s)', ylabel='Distance (mm)',
           title='10 zero query results despite a positive geometric separating gap')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
