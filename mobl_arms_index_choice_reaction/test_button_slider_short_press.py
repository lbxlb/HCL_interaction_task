"""短时推进与独立近距离观测；不修改正式任务或 XML。"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from diagnose_slider_distance_contact import make_proximity_observer
from test_button_slider_full_baseline import main as run_experiment


def main(release=False):
    base = Path(__file__).resolve().parent
    model, data, gids = make_proximity_observer(base, 0.005)

    def observe(source, state, finger, button, row):
        for i, gid in enumerate((finger, button)):
            data.mocap_pos[i] = state.geom_xpos[gid]
            mujoco.mju_mat2Quat(data.mocap_quat[i], state.geom_xmat[gid])
        mujoco.mj_forward(model, data)
        for i, gid in enumerate((finger, button)):
            np.testing.assert_allclose(data.geom_xpos[gids[i]], state.geom_xpos[gid], atol=1e-14, rtol=0)
            np.testing.assert_allclose(data.geom_xmat[gids[i]], state.geom_xmat[gid], atol=1e-14, rtol=0)
        row['observer_contact_distances_m'] = [float(c.dist) for c in data.contact]
        row['observer_constraint_count'] = int(data.nefc)
        # Any observer constraints during overlap remain isolated from source data.
        row['observer_warnings'] = [int(w.number) for w in data.warning]
        assert not any(row['observer_warnings'])

    stem = 'button_slider_press_release' if release else 'button_slider_short_press'
    run_experiment(observation_hook=observe, output_stem=stem, press=True, release=release)
    out = base.parent / 'artifacts/trajectory_check_2026-10-22'
    path = out / f'{stem}.json'
    report = json.loads(path.read_text())
    report['observer_note'] = ('Independent model; diagnostic margin=gap=5mm. Positive records are proximity only. '
                               'Observer constraints, if any during overlap, do not feed back to main data. '
                               'No claim of exact observer distances during penetration.')
    rows = report['samples']
    # Numerical evidence of mechanical interaction, not a task-success threshold.
    report['short_press_checks'] = dict(
        matching_contact_with_positive_force=any(
            'hand_2distph' in c['geoms'] and c['distance_m'] <= 0 and c['normal_force_N'] > 0
            for r in rows for c in r['button_contacts']),
        positive_button_travel_above_1um=report['summary']['max_slider_position_m'] > 1e-6,
        formal_xml_unchanged=report['formal_xml_unchanged'],
    )
    comparisons = [abs(c['distance_m'] - d) for r in rows for c in r['button_contacts']
                   if 'hand_2distph' in c['geoms'] for d in r['observer_contact_distances_m']]
    report['summary']['max_main_vs_observer_contact_difference_m'] = max(comparisons, default=None)
    if release:
        tail = [r for r in rows if r['time_s'] >= 0.9 - 1e-10]
        detached = next((r for r in rows if r['time_s'] > 0.4 and not r['button_contacts']), None)
        report['release_summary'] = dict(
            first_contact_free_time_after_retract_s=detached['time_s'] if detached else None,
            final_position_m=rows[-1]['slider_position_m'],
            final_velocity_m_s=rows[-1]['slider_velocity_m_s'],
            last_100ms_max_abs_position_m=max(abs(r['slider_position_m']) for r in tail),
            last_100ms_max_abs_velocity_m_s=max(abs(r['slider_velocity_m_s']) for r in tail),
        )
        report['release_checks'] = dict(
            last_100ms_contact_free=all(not r['button_contacts'] for r in tail),
            last_100ms_position_within_1um=all(abs(r['slider_position_m']) < 1e-6 for r in tail),
            last_100ms_speed_below_10um_s=all(abs(r['slider_velocity_m_s']) < 1e-5 for r in tail),
            last_100ms_positive_observed_gap=all(
                r['observer_contact_distances_m'] and min(r['observer_contact_distances_m']) > 0 for r in tail),
        )
        report['release_tolerance_note'] = 'Numerical experiment tolerances only; not hardware or task success thresholds.'
        previous = json.loads((out / 'button_slider_short_press.json').read_text())
        report['release_checks']['original_press_prefix_unchanged'] = rows[:len(previous['samples'])] == previous['samples']
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t = [r['time_s'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, [r['slider_position_m'] * 1000 for r in rows])
    axes[0].set(ylabel='Button travel (mm)', title=(
        'Temporary full model: press, retract, spring return' if release
        else 'Temporary full model: short finger advance, no release'))
    axes[1].plot(t, [sum(c['normal_force_N'] for c in r['button_contacts']) for r in rows])
    axes[1].set_ylabel('Button normal force (N)')
    axes[2].plot(t, [min(r['observer_contact_distances_m'], default=np.nan) * 1000 for r in rows], label='Observer')
    axes[2].plot(t, [min((c['distance_m'] for c in r['button_contacts'] if 'hand_2distph' in c['geoms']), default=np.nan) * 1000 for r in rows], '--', label='Main matching contact')
    axes[2].axhline(0, color='black', linewidth=0.7)
    axes[2].set(xlabel='Time (s)', ylabel='Contact distance (mm)')
    axes[2].legend()
    for ax in axes:
        ax.axvline(0.04, color='gray', linestyle=':')
        ax.axvline(0.24, color='gray', linestyle=':')
        if release:
            ax.axvline(0.4, color='gray', linestyle=':')
            ax.axvline(0.6, color='gray', linestyle=':')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f'{stem}_overview.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report['summary'], indent=2))
    assert all(report['short_press_checks'].values()), 'Short press evidence missing; inspect saved report'
    if release:
        print(json.dumps(dict(summary=report['release_summary'], checks=report['release_checks']), indent=2))
        assert all(report['release_checks'].values()), 'Release checks failed; inspect saved report'


if __name__ == '__main__':
    main()
