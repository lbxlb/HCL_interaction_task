"""固定无载基线状态，对比凸距离计算设置；主仿真设置不变。"""
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from diagnose_slider_distance_query import query
from test_button_slider_full_baseline import main as run_baseline


CASES = (
    ('default', {}),
    ('iterations_100', {'ccd_iterations': 100}),
    ('tolerance_1e_8', {'ccd_tolerance': 1e-8}),
    ('nativeccd_disabled', {'disable_nativeccd': True}),
    ('multiccd_enabled', {'enable_multiccd': True}),
)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'artifacts/trajectory_check_2026-10-22'
    prior = json.loads((out / 'slider_distance_query_audit.json').read_text())
    reference = {r['step']: r for r in prior['all_samples']}
    clones = {}
    samples = []

    def inspect(model, data, finger, button, row):
        if not clones:
            for name, changes in CASES:
                probe = copy.copy(model)
                for option in ('ccd_iterations', 'ccd_tolerance'):
                    if option in changes:
                        setattr(probe.opt, option, changes[option])
                if changes.get('disable_nativeccd'):
                    probe.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
                if changes.get('enable_multiccd'):
                    probe.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
                clones[name] = (probe, mujoco.MjData(probe))
        old = reference[row['step']]
        np.testing.assert_array_equal(data.qpos, old['qpos'])
        assert row['finger_button_distance_m'] == old['original_distance_m']
        results = {}
        for name, (probe, state) in clones.items():
            state.qpos[:] = data.qpos
            state.qvel[:] = data.qvel
            state.act[:] = data.act
            state.ctrl[:] = data.ctrl
            state.time = data.time
            mujoco.mj_forward(probe, state)
            # Verify the computation setting did not change geometry placement.
            np.testing.assert_array_equal(state.geom_xpos, data.geom_xpos)
            np.testing.assert_array_equal(state.geom_xmat, data.geom_xmat)
            result = query(probe, state, finger, button, 10.0)
            result['matching_contacts'] = [float(c.dist) for c in state.contact
                                          if {int(c.geom1), int(c.geom2)} == {finger, button}]
            result['warnings'] = [int(w.number) for w in state.warning]
            results[name] = result
        samples.append(dict(step=row['step'], time_s=row['time_s'],
                            separating_plane_gap_m=old['mesh_button_front_separating_plane_gap_m'],
                            cases=results))

    run_baseline(observation_hook=inspect, output_stem='button_slider_settings_replay_baseline')
    previous_baseline = json.loads((out / 'button_slider_full_baseline.json').read_text())
    replay = json.loads((out / 'button_slider_settings_replay_baseline.json').read_text())
    assert replay['samples'] == previous_baseline['samples']
    summaries = {}
    for name, (probe, _) in clones.items():
        values = [r['cases'][name]['distance_m'] for r in samples]
        summaries[name] = dict(
            ccd_iterations=int(probe.opt.ccd_iterations), ccd_tolerance=float(probe.opt.ccd_tolerance),
            nativeccd_enabled=not bool(probe.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
            multiccd_enabled=bool(probe.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD)),
            zero_steps=[r['step'] for r in samples if r['cases'][name]['distance_m'] == 0],
            negative_count=sum(v < 0 for v in values), min_distance_m=min(values), max_distance_m=max(values),
            max_absolute_difference_from_plane_gap_m=max(
                abs(r['cases'][name]['distance_m'] - r['separating_plane_gap_m']) for r in samples),
            below_plane_lower_bound_count=sum(
                r['cases'][name]['distance_m'] < r['separating_plane_gap_m'] - 1e-8 for r in samples),
            matching_contact_samples=sum(bool(r['cases'][name]['matching_contacts']) for r in samples),
            warning_count=sum(sum(r['cases'][name]['warnings']) for r in samples),
        )
    report = dict(experiment=__doc__, mujoco_version=mujoco.__version__, samples=samples,
                  summary=summaries, replay_identical_to_original_baseline=True,
                  formal_xml_unchanged=replay['formal_xml_unchanged'],
                  comparison_tolerance_m=1e-8,
                  comparison_tolerance_note='10 nm is a diagnostic allowance for comparing a '
                                            'query with a geometric lower bound, not a contact threshold.',
                  conclusion='None of the tested settings provides a valid fix on all 201 states. '
                             'Disabling nativeccd removes zeros but produces negative distances '
                             'despite a positive separating-plane gap; do not adopt it.',
                  limitations='Only separated mesh-box states are tested. No inference about '
                              'penetration distances, contact dynamics or other geometries. '
                              'Positive plane gap is a lower bound, not a universal exact distance.')
    path = out / 'slider_distance_settings_audit.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    times = [r['time_s'] for r in samples]
    for name, _ in CASES:
        axes[0].plot(times, [r['cases'][name]['distance_m'] * 1000 for r in samples],
                     '.', markersize=3, label=name)
        axes[1].plot(times, [(r['cases'][name]['distance_m'] - r['separating_plane_gap_m']) * 1e6
                            for r in samples], '.', markersize=3, label=name)
    axes[0].plot(times, [r['separating_plane_gap_m'] * 1000 for r in samples],
                 color='black', linewidth=1, label='Independent plane gap lower bound')
    axes[1].axhline(0, color='black', linewidth=0.7)
    axes[0].set(ylabel='Distance query (mm)', title='Fixed geometry: convex distance settings comparison')
    axes[1].set(xlabel='Time (s)', ylabel='Query minus plane gap (um)')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
