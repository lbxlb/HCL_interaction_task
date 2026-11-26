"""从多目标隔离实验 JSON 绘制角度、跟踪误差和力矩。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_dual_motion(report, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for run in report['runs']:
        samples = run['samples']
        t = [s['time_s'] for s in samples]
        moving = any(run['shoulder_target_sequence_offsets_rad'])
        label = 'Both joints move' if moving else 'Shoulder holds initial angle'
        for column, keys in enumerate((('desired_rad', 'actual_rad', 'torque_Nm'),
                                      ('shoulder_desired_offset_rad', 'shoulder_offset_rad', 'shoulder_torque_Nm'))):
            desired = np.array([s[keys[0]] for s in samples])
            actual = np.array([s[keys[1]] for s in samples])
            line, = axes[0, column].plot(t, actual, label=label)
            axes[0, column].plot(t, desired, '--', color=line.get_color(), alpha=0.6)
            axes[1, column].plot(t, np.degrees(desired - actual), color=line.get_color())
            axes[2, column].plot(t, [s[keys[2]] for s in samples], color=line.get_color())
    axes[0, 0].set_title('Elbow angle (rad)')
    axes[0, 1].set_title('Shoulder angle relative to initial (rad)')
    for column in range(2):
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_ylabel('Tracking error (deg)')
        axes[2, column].set_ylabel('Torque (N m)')
        axes[2, column].set_xlabel('Time (s)')
    for ax in axes.flat:
        for start in range(4):
            if start:
                ax.axvline(start, color='grey', linestyle='--', alpha=0.3)
            ax.axvline(start + report['approach_seconds'], color='grey', linestyle=':', alpha=0.3)
        ax.grid(alpha=0.25)
    fig.suptitle('Two-joint compensated PD: solid actual, dashed desired; remaining joints constrained')
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding='utf-8'))
    if report.get('experiment') == 'shoulder_elbow_dual_motion':
        output = args.input.with_suffix('.png')
        plot_dual_motion(report, output)
        print(output)
        if all('fingertip_tracking' in run for run in report['runs']):
            from measure_fingertip import plot
            tip_output = args.input.with_name(args.input.stem + '_position.png')
            plot(report, tip_output)
            print(tip_output)
        return
    release_test = report.get('experiment') == 'shoulder_rot_fixture_on_off'
    hold_test = report.get('experiment') == 'shoulder_hold_on_off'
    shoulder_test = release_test or hold_test
    fig, axes = plt.subplots(5 if hold_test else (4 if release_test else 3), 1,
                             figsize=(10, 13 if hold_test else (11 if release_test else 9)), sharex=True)
    for run in report['runs']:
        samples = run['samples']
        t = np.array([s['time_s'] for s in samples])
        desired = np.array([s['desired_rad'] for s in samples])
        actual = np.array([s['actual_rad'] for s in samples])
        label = ('Continuous sequence' if 'target_sequence_offsets_rad' in run
                 else f"{run['target_offset_rad']:+.2f} rad")
        if release_test:
            label = 'Shoulder fixture ON' if run['shoulder_fixture_active'] else 'Shoulder fixture OFF'
        if hold_test:
            label = 'Shoulder PD ON' if run['shoulder_hold_control'] else 'Shoulder motor zero torque'
        line, = axes[0].plot(t, actual, label=label)
        axes[0].plot(t, desired, '--', color=line.get_color(), alpha=0.65)
        axes[1].plot(t, np.degrees(desired - actual), color=line.get_color())
        axes[2].plot(t, [s['torque_Nm'] for s in samples], color=line.get_color())
        if shoulder_test:
            axes[3].plot(t, np.degrees([s['shoulder_offset_rad'] for s in samples]),
                         color=line.get_color())
        if hold_test:
            axes[4].plot(t, [s['shoulder_torque_Nm'] for s in samples], color=line.get_color())
    axes[0].set_title('Isolated elbow: compensated PD (solid actual, dashed desired)')
    axes[0].set_ylabel('Angle (rad)')
    axes[0].legend(ncol=4)
    axes[1].set_ylabel('Desired - actual (deg)')
    axes[2].set_ylabel('Torque (N m)')
    if shoulder_test:
        axes[3].set_ylabel('Shoulder drift (deg)')
    if hold_test:
        axes[4].set_ylabel('Shoulder torque (N m)')
        axes[0].set_title('Elbow motion with shoulder hold control (solid actual, dashed desired)')
    axes[-1].set_xlabel('Time (s)')
    for ax in axes:
        starts = ([i * report['segment_seconds'] for i in range(len(report['target_sequence_offsets_rad']))]
                  if 'target_sequence_offsets_rad' in report else [0.0])
        for start in starts:
            if start:
                ax.axvline(start, color='grey', linestyle='--', alpha=0.4)
            ax.axvline(start + report['approach_seconds'], color='grey', linestyle=':', alpha=0.5)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output = args.input.with_suffix('.png')
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
