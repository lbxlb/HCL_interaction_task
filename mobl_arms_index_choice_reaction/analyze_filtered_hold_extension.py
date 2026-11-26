"""分析过滤推进实验延长目标保持后的跟随误差与接触状态。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-22'
    short = json.loads((directory / 'small_normal_press_filtered.json').read_text())
    long = json.loads((directory / 'small_normal_press_filtered_hold200.json').read_text())
    assert long['mode'] == 'press'
    assert len(long['samples']) == long['controller']['steps'] == 270
    for key in ('initial_state', 'press_geometry', 'collision_filter', 'model_changes', 'timestep_s'):
        assert short[key] == long[key], key
    for key in ('kp', 'kd', 'torque_limit_Nm', 'settle_steps', 'ramp_steps'):
        assert short['controller'][key] == long['controller'][key], key
    shared_fields = ('step', 'time_s', 'phase', 'geom_distance_m', 'task_contact',
                     'button_normal_force_N', 'q_actual_rad', 'q_target_rad',
                     'applied_torque_Nm', 'button_contacts')
    for index, (a, b) in enumerate(zip(short['samples'], long['samples'])):
        for field in shared_fields:
            assert a[field] == b[field], (index, field)
    samples = long['samples']
    time_ms = np.array([s['time_s'] * 1000 for s in samples])
    position_error = np.degrees(np.array([s['q_error_rad'] for s in samples]))
    velocity = np.degrees(np.array([s['qvel_actual_rad_s'] for s in samples]))
    force = np.array([s['button_normal_force_N'] for s in samples])
    pair_distance = np.array([
        next((contact['distance_m'] * 1000 for contact in s['button_contacts']
              if contact['other_geom'] == 'hand_2distph'), np.nan)
        for s in samples])
    assert np.isfinite(position_error).all() and np.isfinite(velocity).all()
    assert np.isfinite(force).all() and np.isfinite(time_ms).all()
    assert np.all(np.diff(time_ms) > 0)
    assert not any(long['result']['warning_counts'])
    assert long['controller']['saturation_count'] == 0
    active = [s for s in samples if s['task_contact']]
    assert active and all(c['distance_m'] > 0 for s in active
                          for c in s['button_contacts'] if c['other_geom'] == 'hand_2distph')
    threshold_mm = long['press_geometry']['geom_margin_sum_mm']
    assert max(s['button_normal_force_N'] for s in samples[-25:]) - min(
        s['button_normal_force_N'] for s in samples[-25:]) < 0.001
    late_start = len(samples) - 25  # 最后 50 ms
    late = samples[late_start:]
    late_err = np.degrees(np.array([s['q_error_rad'] for s in late]))
    late_vel = np.degrees(np.array([s['qvel_actual_rad_s'] for s in late]))
    late_force = np.array([s['button_normal_force_N'] for s in late])
    late_distance = pair_distance[late_start:]
    summary = dict(
        first_100_physical_samples_identical_to_short_run=True,
        run=dict(steps=len(samples), duration_s=float(time_ms[-1] / 1000),
                 filtered_internal_pairs=len(long['collision_filter']['excluded_body_pairs']),
                 kp=long['controller']['kp'], kd=long['controller']['kd'],
                 torque_limit_Nm=long['controller']['torque_limit_Nm'],
                 saturation_count=long['controller']['saturation_count']),
        first_task_contact_time_s=long['result']['first_task_contact_time_s'],
        task_contact_persists_to_end=long['result']['final_task_contact'],
        contact_pair_distance_min_mm=float(np.nanmin(pair_distance)),
        contact_pair_distance_final_mm=float(pair_distance[-1]),
        summed_geom_margin_mm=threshold_mm,
        final_position_error_target_minus_actual_deg=position_error[-1].tolist(),
        late_50ms=dict(
            position_error_mean_deg=late_err.mean(axis=0).tolist(),
            position_error_range_deg=np.ptp(late_err, axis=0).tolist(),
            max_abs_joint_velocity_deg_s=np.max(np.abs(late_vel), axis=0).tolist(),
            normal_force_mean_N=float(late_force.mean()),
            normal_force_range_N=float(np.ptp(late_force)),
            contact_distance_mean_mm=float(np.nanmean(late_distance)),
            contact_distance_range_mm=float(np.nanmax(late_distance) - np.nanmin(late_distance))),
        interpretation=('Late-window signals are nearly stationary in this run, but no acceptance '
                        'tolerance is defined. Contact distance remains positive, so surface pressing '
                        'is not demonstrated.'))
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, position_error[:, 0], label='shoulder target - actual')
    axes[0].plot(time_ms, position_error[:, 1], label='elbow target - actual')
    axes[0].set_ylabel('Position error (deg)')
    axes[1].plot(time_ms, velocity[:, 0], label='shoulder velocity')
    axes[1].plot(time_ms, velocity[:, 1], label='elbow velocity')
    axes[1].set_ylabel('Velocity (deg/s)')
    axes[2].plot(time_ms, pair_distance, label='finger-button contact.dist')
    axes[2].axhline(threshold_mm, color='tab:red', linestyle='--', label='combined geom margin')
    axes[2].set_ylabel('Pair distance (mm)')
    axes[3].plot(time_ms, force, color='tab:orange', label='button normal force')
    axes[3].set_ylabel('Normal force (N)')
    axes[3].set_xlabel('Simulation time (ms)')
    for ax in axes:
        ax.axvline(long['controller']['settle_steps'] * long['timestep_s'] * 1000,
                   color='gray', linestyle=':', alpha=0.7)
        ax.axvline((long['controller']['settle_steps'] + long['controller']['ramp_steps'])
                   * long['timestep_s'] * 1000, color='gray', linestyle=':', alpha=0.7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_title('Filtered press: slow response settles with position error and margin contact')
    fig.savefig(directory / 'filtered_press_hold_extension.png', dpi=160)
    plt.close(fig)
    (directory / 'filtered_press_hold_extension.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
