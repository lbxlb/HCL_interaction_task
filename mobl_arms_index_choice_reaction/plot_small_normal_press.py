"""绘制短时接触试验的两种距离读数与接触力，不推进仿真。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='small_normal_press.json',
                        help='report filename in the 2026-10-22 artifact directory')
    parser.add_argument('--output',
                        help='PNG filename; defaults to the input filename with a .png extension')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-22'
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = directory / input_path
    report = json.loads(input_path.read_text(encoding='utf-8'))
    samples = report['samples']
    controller = report['controller']
    assert len(samples) == controller['steps']
    times = np.array([sample['time_s'] * 1000 for sample in samples])
    distances = np.array([sample['geom_distance_m'] * 1000 for sample in samples])
    pair_distances = np.array([
        min((contact['distance_m'] * 1000 for contact in sample['button_contacts']
             if contact['other_geom'] == 'hand_2distph'), default=np.nan)
        for sample in samples])
    forces = np.array([sample['button_normal_force_N'] for sample in samples])
    q_actual = np.degrees(np.array([sample['q_actual_rad'] for sample in samples]))
    q_target = np.degrees(np.array([sample['q_target_rad'] for sample in samples]))
    q_error = np.degrees(np.array([sample['q_error_rad'] for sample in samples]))
    qvel = np.array([sample['qvel_actual_rad_s'] for sample in samples])
    assert np.isfinite(times).all() and np.isfinite(distances).all() and np.isfinite(forces).all()
    assert np.all(np.diff(times) > 0)
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    axes[0].plot(times, distances, label='mj_geomDistance', linewidth=1.4)
    axes[0].plot(times, pair_distances, label='finger-button contact.dist', linewidth=1.4)
    flagged = (np.isfinite(pair_distances)
               & (np.abs(distances - pair_distances)
                  > report['distance_diagnostics']['comparison_threshold_m'] * 1000))
    axes[0].scatter(times[flagged], distances[flagged], color='red', marker='x',
                    label='distance disagreement > 0.01 mm', zorder=4)
    unpaired_zero = (distances == 0) & ~np.isfinite(pair_distances)
    axes[0].scatter(times[unpaired_zero], distances[unpaired_zero], color='tab:purple',
                    marker='s', label='zero geomDistance, no matched contact', zorder=4)
    axes[0].axhline(0, color='black', linewidth=0.8)
    margin_mm = report['press_geometry']['geom_margin_sum_mm']
    axes[0].axhline(margin_mm, color='gray', linestyle=':', label='combined geom margin')
    axes[0].set_ylabel('Distance (mm)')
    axes[0].legend(fontsize=8)
    joint_names = ('shoulder_rot', 'elbow_flexion')
    for index, name in enumerate(joint_names):
        axes[1].plot(times, q_actual[:, index], label=f'{name} actual')
        axes[1].plot(times, q_target[:, index], linestyle='--', alpha=0.75,
                     label=f'{name} target')
        axes[2].plot(times, q_error[:, index], label=f'{name} error')
        axes[3].plot(times, qvel[:, index], label=f'{name} velocity')
    axes[1].set_ylabel('Joint angle (deg)')
    axes[1].legend(fontsize=8, ncol=2)
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].set_ylabel('Target − actual (deg)')
    axes[2].legend(fontsize=8)
    axes[3].set_ylabel('Joint velocity (rad/s)')
    axes[3].legend(fontsize=8)
    axes[4].plot(times, forces, color='tab:orange', label='all button contact normal forces')
    axes[4].set_ylabel('Normal force (N)')
    axes[4].set_xlabel('Simulation time (ms)')
    axes[4].legend(fontsize=8)
    ramp_start = controller['settle_steps'] * report['timestep_s'] * 1000
    hold_start = (controller['settle_steps'] + controller['ramp_steps']) * report['timestep_s'] * 1000
    for ax in axes:
        ax.axvline(ramp_start, color='gray', linestyle='--')
        ax.axvline(hold_start, color='gray', linestyle='--')
        ax.grid(alpha=0.25)
    target_source = report.get('target_source', report.get('mode', 'unknown'))
    axes[0].set_title(f"Temporary model: {target_source}; margin contact is not button activation")
    axes[4].text(ramp_start + 2, 0.95, 'ramp starts', transform=axes[4].get_xaxis_transform(), va='top')
    axes[4].text(hold_start + 2, 0.95, 'hold starts', transform=axes[4].get_xaxis_transform(), va='top')
    output = Path(args.output) if args.output else input_path.with_suffix('.png')
    if not output.is_absolute():
        output = directory / output
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)
    print(f'Distance disagreement samples: {int(flagged.sum())}')


if __name__ == '__main__':
    main()
