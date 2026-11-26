"""汇总保持实验前 40 ms 的内部接触与受控关节接触力矩。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-22'
    report = json.loads((directory / 'small_normal_hold_only.json').read_text())
    diagnostics = report['dynamics_diagnostics']
    snapshots = diagnostics['snapshots']
    assert len(snapshots) == report['controller']['settle_steps'] + 1
    assert snapshots[0]['time_s'] == 0
    joints = {joint['name']: joint for joint in diagnostics['joints']}
    first_contact = report['result']['first_task_contact_step']
    before_button = [snapshot for snapshot in snapshots if snapshot['step'] < first_contact]
    assert all(not any('button-0' in contact['geoms'] for contact in snapshot['contacts'])
               for snapshot in before_button)
    summary = dict(
        initial_control_note=diagnostics['initial_control_note'],
        initial_contacts=snapshots[0]['contacts'],
        first_button_contact_step=first_contact,
        pre_button_contact_torques_Nm={},
        max_abs_equality_residual_rad=max(abs(eq['residual_rad'])
                                        for snapshot in snapshots for eq in snapshot['equalities']),
        interpretation=('Internal contacts and contact torques precede button contact; '
                        'causal contribution still requires a collision-filter control experiment.'))
    times = np.array([snapshot['time_s'] * 1000 for snapshot in snapshots])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    pairs = sorted({tuple(contact['geoms']) for snapshot in snapshots
                    for contact in snapshot['contacts'] if 'button-0' not in contact['geoms']})
    for pair in pairs:
        values = [sum(contact['normal_force_N'] for contact in snapshot['contacts']
                      if tuple(contact['geoms']) == pair) for snapshot in snapshots]
        axes[0].plot(times, np.array(values) / 1000, label=' / '.join(pair))
    for name in ('shoulder_rot', 'elbow_flexion'):
        dof = joints[name]['dof_address']
        values = [snapshot['constraint_force_components']['contact'][dof] for snapshot in snapshots]
        axes[1].plot(times, values, label=name)
        summary['pre_button_contact_torques_Nm'][name] = dict(
            min=min(snapshot['constraint_force_components']['contact'][dof] for snapshot in before_button),
            max=max(snapshot['constraint_force_components']['contact'][dof] for snapshot in before_button))
    axes[0].set_title('Hold only: internal contacts exist before button contact')
    axes[0].set_ylabel('Internal normal force (kN)')
    axes[1].set_ylabel('Contact torque (Nm)')
    axes[1].set_xlabel('Simulation time (ms)')
    for ax in axes:
        ax.axvline(first_contact * report['timestep_s'] * 1000, color='gray', linestyle='--',
                   label='first button contact')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(directory / 'hold_dynamics.png', dpi=160)
    plt.close(fig)
    (directory / 'hold_dynamics_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
