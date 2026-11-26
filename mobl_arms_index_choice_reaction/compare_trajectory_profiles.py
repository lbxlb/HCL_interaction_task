"""对比两份双轴指尖报告；只比较肩肘同时运动组。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cosine', type=Path)
    parser.add_argument('quintic', type=Path)
    args = parser.parse_args()
    reports = [json.loads(p.read_text(encoding='utf-8')) for p in (args.cosine, args.quintic)]
    assert reports[0].get('trajectory_profile', 'cosine') == 'cosine'
    assert reports[1]['trajectory_profile'] == 'quintic'
    for key in ('kp', 'kd', 'shoulder_kp', 'shoulder_kd', 'approach_seconds',
                'torque_limit_Nm', 'shoulder_torque_limit_Nm', 'fixture_joints',
                'segment_seconds', 'target_sequence_offsets_rad'):
        assert reports[0][key] == reports[1][key], key
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    summary = {}
    for label, report in zip(('cosine', 'quintic'), reports):
        run, = [r for r in report['runs'] if any(r['shoulder_target_sequence_offsets_rad'])]
        tip = run['fingertip_tracking']
        samples = run['samples']
        t = [s['time_s'] for s in samples]
        assert len(tip['samples']) == len(t) == 2001
        assert np.allclose(t, [s['time_s'] for s in tip['samples']], atol=1e-12, rtol=0)
        axes[0, 0].plot(t, [s['error_norm_m'] * 1000 for s in tip['samples']], label=label)
        axes[0, 1].plot(t, np.degrees([s['desired_rad'] - s['actual_rad'] for s in samples]), label=label)
        axes[1, 0].plot(t, [s['torque_Nm'] for s in samples], label=label)
        axes[1, 1].plot(t, [s['shoulder_torque_Nm'] for s in samples], label=label)
        summary[label] = dict(tip_peak_mm=tip['max_error_m'] * 1000,
                             tip_rms_mm=tip['rms_error_m'] * 1000,
                             tip_worst_tail_mm=max(s['max_error_last_0_2_seconds_m'] for s in tip['segments']) * 1000,
                             elbow_peak_torque_Nm=run['peak_abs_torque_Nm'],
                             shoulder_peak_torque_Nm=run['shoulder_peak_abs_torque_Nm'])
    for ax, title, unit in zip(axes.flat,
                              ('Fingertip error norm', 'Elbow tracking error', 'Elbow torque', 'Shoulder torque'),
                              ('mm', 'deg', 'N m', 'N m')):
        ax.set(title=title, xlabel='Time (s)', ylabel=unit)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle('Same 0.4 s move, gains and fixtures; both joints move')
    fig.tight_layout()
    output = args.quintic.with_name('trajectory_profile_comparison')
    fig.savefig(output.with_suffix('.png'), dpi=140)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
