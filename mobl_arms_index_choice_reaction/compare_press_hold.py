"""验证推进/仅保持实验的控制变量，并绘制关节漂移及接触力对照。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filtered', action='store_true', help='compare the five-pair filtered cases')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-22'
    press = json.loads((directory / ('small_normal_press_filtered.json' if args.filtered
                                     else 'small_normal_press.json')).read_text())
    hold = json.loads((directory / ('small_normal_hold_filtered.json' if args.filtered
                                    else 'small_normal_hold_only.json')).read_text())
    assert press['mode'] == 'press' and hold['mode'] == 'hold-only'
    for key in ('initial_state', 'press_geometry', 'model_changes', 'timestep_s'):
        assert press[key] == hold[key], key
    for key in ('kp', 'kd', 'torque_limit_Nm', 'settle_steps', 'ramp_steps', 'hold_steps', 'steps'):
        assert press['controller'][key] == hold['controller'][key], key
    settle = press['controller']['settle_steps']
    if args.filtered:
        assert press['collision_filter'] == hold['collision_filter']
        assert len(press['collision_filter']['excluded_body_pairs']) == 5
        assert not any(sample['task_contact'] for sample in hold['samples'])
        assert not any(sample['task_contact'] for sample in press['samples'][:settle])
        assert press['result']['first_task_contact_time_s'] > settle * press['timestep_s']
        assert press['result']['min_contact_pair_distance_m'] > 0
    fields = ('time_s', 'q_actual_rad', 'q_target_rad', 'applied_torque_Nm',
              'geom_distance_m', 'task_contact', 'button_normal_force_N', 'button_contacts')
    assert len(press['samples']) == len(hold['samples']) == press['controller']['steps']
    for first, second in zip(press['samples'][:settle], hold['samples'][:settle]):
        for field in fields:
            assert first[field] == second[field], (first['step'], field)
    q_start = np.array(hold['press_geometry']['outside_q_rad'])
    for sample in hold['samples']:
        np.testing.assert_array_equal(sample['q_target_rad'], q_start)
    assert any(sample['q_target_rad'] != q_start.tolist() for sample in press['samples'][settle:])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    summary = dict(initial_state_and_parameters_identical=True,
                   first_20_steps_identical=True, hold_target_constant=True, cases={})
    for report, label, style in ((press, 'press', '-'), (hold, 'hold only', '--')):
        samples = report['samples']
        times = np.array([s['time_s'] * 1000 for s in samples])
        actual = np.array([s['q_actual_rad'] for s in samples])
        target = np.array([s['q_target_rad'] for s in samples])
        force = np.array([s['button_normal_force_N'] for s in samples])
        assert np.isfinite(actual).all() and np.isfinite(force).all()
        assert not any(report['result']['warning_counts'])
        for index in range(2):
            axes[index].plot(times, np.degrees(actual[:, index] - q_start[index]),
                             style, label=f'{label}: actual')
            axes[index].plot(times, np.degrees(target[:, index] - q_start[index]),
                             style, alpha=0.45, label=f'{label}: target')
        axes[2].plot(times, force, style, label=label)
        summary['cases'][report['mode']] = dict(
            first_task_contact_time_s=report['result']['first_task_contact_time_s'],
            peak_normal_force_N=report['result']['contact_force_peak_N'],
            final_normal_force_N=report['result']['contact_force_final_N'],
            first_geometric_touch=report['result']['first_geometric_touch'],
            min_contact_pair_distance_m=report['result']['min_contact_pair_distance_m'],
            final_tracking_error_deg=np.degrees(actual[-1] - target[-1]).tolist(),
            final_joint_drift_deg=np.degrees(actual[-1] - q_start).tolist(),
            max_abs_joint_drift_deg=np.degrees(np.max(np.abs(actual - q_start), axis=0)).tolist())
    axes[0].set_ylabel('Shoulder drift (deg)')
    axes[1].set_ylabel('Elbow drift (deg)')
    axes[2].set_ylabel('Normal force (N)')
    axes[2].set_xlabel('Simulation time (ms)')
    axes[0].set_title('Filtered model: approach triggers margin contact; no surface touch yet'
                      if args.filtered else 'No-approach control: early contact occurs in both cases')
    for ax in axes:
        ax.axvline(settle * press['timestep_s'] * 1000, color='gray', linestyle=':')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    stem = 'filtered_press_hold_comparison' if args.filtered else 'press_hold_comparison'
    fig.savefig(directory / (stem + '.png'), dpi=160)
    plt.close(fig)
    (directory / (stem + '.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
