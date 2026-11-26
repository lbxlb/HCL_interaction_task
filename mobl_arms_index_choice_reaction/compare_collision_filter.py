"""比较仅保持实验在五对内部 body 碰撞排除前后的结果。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from experiment_small_normal_press import collision_pair_audit


def main():
    root = Path(__file__).resolve().parent.parent
    directory = root / 'artifacts/trajectory_check_2026-10-22'
    baseline = json.loads((directory / 'small_normal_hold_only.json').read_text())
    filtered = json.loads((directory / 'small_normal_hold_filtered.json').read_text())
    for key in ('initial_state', 'press_geometry', 'timestep_s'):
        assert baseline[key] == filtered[key], key
    assert baseline['mode'] == filtered['mode'] == 'hold-only'
    for key in ('kp', 'kd', 'torque_limit_Nm', 'settle_steps', 'ramp_steps', 'hold_steps', 'steps'):
        assert baseline['controller'][key] == filtered['controller'][key], key
    assert baseline['collision_filter']['excluded_body_pairs'] == []
    excluded = filtered['collision_filter']['excluded_body_pairs']
    assert len(excluded) == filtered['collision_filter']['experiment_nexclude'] == 5
    assert filtered['collision_filter']['formal_panel_audit'] == baseline['collision_filter']['formal_panel_audit']
    for pair in filtered['collision_filter']['formal_panel_audit']['pairs']:
        assert all(entry['body_geoms'] == [entry['geom']] for entry in pair)
    assert filtered['collision_filter']['press_target_finger_button_contact_detected']
    assert all(not snapshot['contacts'] for snapshot in filtered['dynamics_diagnostics']['snapshots'])
    assert not any(sample['task_contact'] for sample in filtered['samples'])
    assert baseline['result']['first_task_contact_step'] == 13
    pointing = mujoco.MjModel.from_xml_path(str(root / 'mobl_arms_index_pointing/simulation.xml'))
    summary = dict(pointing_model_audit=collision_pair_audit(pointing),
                   excluded_body_pairs=excluded,
                   press_pose_contact_check_passed=True, cases={})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    start = np.array(filtered['press_geometry']['outside_q_rad'])
    for report, label in ((baseline, 'original collisions'), (filtered, 'five pairs excluded')):
        samples = report['samples']
        assert len(samples) == 100 and not any(report['result']['warning_counts'])
        target = np.array([sample['q_target_rad'] for sample in samples])
        np.testing.assert_array_equal(target, np.tile(start, (len(samples), 1)))
        actual = np.array([sample['q_actual_rad'] for sample in samples])
        times = [sample['time_s'] * 1000 for sample in samples]
        assert np.isfinite(actual).all()
        for index in range(2):
            axes[index].plot(times, np.degrees(actual[:, index] - start[index]), label=label)
        axes[2].plot(times, [sample['button_normal_force_N'] for sample in samples], label=label)
        summary['cases'][label] = dict(
            first_task_contact_time_s=report['result']['first_task_contact_time_s'],
            max_abs_joint_drift_deg=np.degrees(report['result']['max_abs_controlled_joint_drift_rad']).tolist(),
            peak_button_force_N=report['result']['contact_force_peak_N'],
            saturation_count=report['controller']['saturation_count'],
            final_geom_distance_mm=report['result']['final_geom_distance_m'] * 1000)
    axes[0].set_title('Hold only: effect of excluding five selected internal body pairs')
    axes[0].set_ylabel('Shoulder drift (deg)')
    axes[1].set_ylabel('Elbow drift (deg)')
    axes[2].set_ylabel('Button normal force (N)')
    axes[2].set_xlabel('Simulation time (ms)')
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(directory / 'collision_filter_comparison.png', dpi=160)
    plt.close(fig)
    (directory / 'collision_filter_comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary['cases'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
