"""Verify seeded, small initial-pose jitter and the scripted task baseline."""
import json
from pathlib import Path

import mujoco
import numpy as np

from button_slider_env import ButtonSliderEnv


def scripted_action(env):
    t = float(env.sim.data.time)
    if t < 1.0:
        phase = np.clip((t - 0.04) / 0.2, 0, 1)
        blend = 3 * phase**2 - 2 * phase**3
    else:
        phase = np.clip((t - 1.0) / 0.2, 0, 1)
        blend = 1 - (3 * phase**2 - 2 * phase**3)
    return np.full(2, 2 * blend - 1)


def reset_snapshot(env, seed):
    observation, info = env.reset(seed=seed)
    return {
        'observation': observation.copy(),
        'qpos': env.sim.data.qpos.copy(),
        'qvel': env.sim.data.qvel.copy(),
        'info': info,
        'row': env._row,
        'warnings': list(env.sim.last_warnings),
    }


def fixed_action_trajectory(env, seed, steps=40):
    observation, _ = env.reset(seed=seed)
    observations = [observation.copy()]
    rewards, events = [], []
    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step([-1.0, -1.0])
        observations.append(observation.copy())
        rewards.append(float(reward))
        events.extend(event['event'] for event in info['events'])
        if terminated or truncated:
            break
    return np.asarray(observations), np.asarray(rewards), events


def main():
    env = ButtonSliderEnv(max_episode_steps=150)
    env.reset(seed=4099)
    model = env.sim.model
    finger, button = env.sim.finger, env.sim.button
    checks = {}
    reset_samples = []
    try:
        first = reset_snapshot(env, 4101)
        same_seed = reset_snapshot(env, 4101)
        different_seed = reset_snapshot(env, 4102)
        checks['same_seed_same_observation'] = bool(np.array_equal(
            first['observation'], same_seed['observation']))
        checks['same_seed_same_initial_state'] = bool(np.array_equal(
            first['qpos'], same_seed['qpos']) and np.array_equal(first['qvel'], same_seed['qvel']))
        checks['different_seed_changes_initial_pose'] = bool(not np.array_equal(
            first['info']['initial_pose_offset_rad'], different_seed['info']['initial_pose_offset_rad']))
        trajectory_a = fixed_action_trajectory(env, 4103)
        trajectory_b = fixed_action_trajectory(env, 4103)
        checks['same_seed_same_action_same_trajectory'] = bool(
            np.array_equal(trajectory_a[0], trajectory_b[0])
            and np.array_equal(trajectory_a[1], trajectory_b[1])
            and trajectory_a[2] == trajectory_b[2])

        for seed in range(4200, 4220):
            sample = reset_snapshot(env, seed)
            q = sample['qpos'][env.sim.qadr]
            offset = np.asarray(sample['info']['initial_pose_offset_rad'])
            pair_contact = any(set(pair) == {
                model.geom(finger).name, model.geom(button).name
            } for pair in sample['row']['all_contact_pairs'])
            distance = float(mujoco.mj_geomDistance(
                model, env.sim.data, finger, button, 10, None))
            reset_samples.append({
                'seed': seed,
                'initial_joint_position_rad': q.astype(float).tolist(),
                'initial_joint_velocity_rad_s': sample['qvel'][env.sim.dofs].astype(float).tolist(),
                'initial_pose_offset_rad': offset.astype(float).tolist(),
                'finger_button_distance_query_m': distance,
                'distance_query_zero_without_contact': bool(distance == 0 and not pair_contact),
                'matching_contact': pair_contact,
                'warning_count': len(sample['warnings']),
            })
        checks['all_offsets_within_configured_envelope'] = all(
            np.all(np.abs(np.asarray(item['initial_pose_offset_rad']))
                   <= np.deg2rad(env.initial_pose_jitter_deg) + 1e-12)
            for item in reset_samples)
        checks['no_initial_fingertip_button_contact'] = all(
            not item['matching_contact'] for item in reset_samples)
        checks['no_initial_mujoco_warnings'] = all(
            item['warning_count'] == 0 for item in reset_samples)
        checks['zero_initial_shoulder_elbow_velocity'] = all(
            np.array_equal(np.asarray(item['initial_joint_velocity_rad_s']), np.zeros(2))
            for item in reset_samples)
        positive_distances = [item['finger_button_distance_query_m'] for item in reset_samples
                              if item['finger_button_distance_query_m'] > 0]
        checks['positive_distance_diagnostic_above_1mm'] = bool(
            positive_distances and min(positive_distances) > 0.001)

        scripted_episodes = []
        for seed in (4301, 4302, 4303):
            observation, _ = env.reset(seed=seed)
            events = []
            total_return = 0.0
            for _ in range(env.max_episode_steps):
                observation, reward, terminated, truncated, info = env.step(scripted_action(env))
                events.extend(event['event'] for event in info['events'])
                total_return += float(reward)
                if terminated or truncated:
                    break
            scripted_episodes.append({
                'seed': seed,
                'success': bool(info['is_success']),
                'events': events,
                'return': total_return,
                'max_slider_position_m': float(info['max_slider_position_m_episode']),
                'peak_normal_force_N': float(info['peak_normal_force_N_episode']),
                'failure_reason': info['failure_reason'],
            })
        checks['scripted_press_release_succeeds_for_all_seeds'] = all(
            item['success'] and item['events'] == ['pressed', 'released']
            and item['failure_reason'] is None for item in scripted_episodes)
    finally:
        env.close()

    result = {
        'checks': checks,
        'configured_jitter_deg_per_joint': 0.2,
        'velocity_initialization': 'zero velocity; only shoulder_rot and elbow_flexion qpos vary',
        'positive_geom_distance_min_m': min(positive_distances, default=None),
        'geom_distance_caveat': (
            'A zero geom-distance query without a matching contact is ambiguous and is not '
            'interpreted as contact or as a proven clearance.'
        ),
        'reset_samples': reset_samples,
        'scripted_episodes': scripted_episodes,
        'limitations': (
            'Small jitter about one validated temporary-model pose; not anatomical, hardware, '
            'or workspace safety validation.'
        ),
    }
    output = (Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-27/'
              'button_slider_randomized_reset.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    assert all(checks.values()), 'Randomized reset verification failed; inspect report'


if __name__ == '__main__':
    main()
