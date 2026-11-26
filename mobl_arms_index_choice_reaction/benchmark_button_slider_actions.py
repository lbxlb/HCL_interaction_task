"""Short deterministic and random action baselines; no policy training."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from button_slider_env import ButtonSliderEnv


def action_for(strategy, env, step, rng):
    if strategy == 'outer_hold':
        return np.array([-1.0, -1.0])
    if strategy == 'press_hold':
        return np.array([1.0, 1.0])
    if strategy == 'alternating':
        t = env.sim.data.time
        if t < 0.04:
            value = -1.0
        elif t < 0.4:
            value = np.clip(2 * (t - 0.04) / 0.2 - 1, -1, 1)
        elif t < 1.0:
            value = 1.0
        elif t < 1.2:
            value = np.clip(1 - 2 * (t - 1.0) / 0.2, -1, 1)
        else:
            value = -1.0
        return np.full(2, value)
    if strategy == 'random':
        return rng.uniform(-1.0, 1.0, size=2)
    raise ValueError(strategy)


def main():
    out = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-27'
    out.mkdir(parents=True, exist_ok=True)
    strategies = ('outer_hold', 'press_hold', 'alternating', 'random')
    env = ButtonSliderEnv(max_episode_steps=150)
    episodes = []
    observation_values = []
    for strategy_index, strategy in enumerate(strategies):
        for episode_index in range(3):
            # Pair every strategy with the same three reproducible initial poses.
            seed = 2700 + episode_index
            obs, reset_info = env.reset(seed=seed)
            observation_values.append(obs.copy())
            action_rng_seed = 100000 + strategy_index * 100 + episode_index
            rng = np.random.default_rng(action_rng_seed)
            events, rewards, positions, forces, saturation_channels, times = [], [], [], [], [], []
            for step in range(env.max_episode_steps):
                action = action_for(strategy, env, step, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                observation_values.append(obs.copy())
                rewards.append(float(reward))
                positions.append(float(info['slider_position_m']))
                forces.append(info['peak_normal_force_N_this_step'])
                times.append(float(info['time_s']))
                if info['events']:
                    events.extend(info['events'])
                saturation_channels.append(info['saturated_motor_channels_this_step'])
                if terminated or truncated:
                    break
            episodes.append(dict(
                strategy=strategy, episode=episode_index, seed=seed,
                initial_pose_offset_rad=reset_info['initial_pose_offset_rad'],
                action_rng_seed=action_rng_seed,
                steps=len(rewards), simulated_time_s=float(env.sim.data.time),
                return_sum=float(sum(rewards)), max_button_travel_m=max(positions),
                peak_button_force_N=max(forces), press_count=env.sim.task.press_count,
                release_count=env.sim.task.release_count, events=events,
                terminated=bool(terminated), truncated=bool(truncated),
                saturated_motor_samples=sum(n > 0 for n in saturation_channels),
                saturation_motor_channel_count=sum(saturation_channels),
                sample_time_s=times,
                positions_m=positions, forces_N=forces,
            ))
    obs = np.asarray(observation_values)
    observation_checks = dict(
        all_observations_finite=bool(np.isfinite(obs).all()),
        all_observations_inside_box=bool(np.all(obs >= env.observation_space.low)
                                         and np.all(obs <= env.observation_space.high)),
        box_has_finite_bounds=bool(np.isfinite(env.observation_space.low).all()
                                   and np.isfinite(env.observation_space.high).all()),
    )
    grouped = {}
    for strategy in strategies:
        rows = [e for e in episodes if e['strategy'] == strategy]
        grouped[strategy] = dict(
            episodes=len(rows), successes=sum(e['terminated'] for e in rows),
            timeouts=sum(e['truncated'] for e in rows),
            mean_return=float(np.mean([e['return_sum'] for e in rows])),
            max_observed_travel_m=max(e['max_button_travel_m'] for e in rows),
            max_force_N=max(e['peak_button_force_N'] for e in rows),
            saturated_episode_count=sum(e['saturated_motor_samples'] > 0 for e in rows),
            saturated_motor_channel_count=sum(e['saturation_motor_channel_count'] for e in rows),
        )
    paired_pose_checks = {}
    for episode_index in range(3):
        rows = [e for e in episodes if e['episode'] == episode_index]
        offsets = [np.asarray(e['initial_pose_offset_rad']) for e in rows]
        paired_pose_checks[str(episode_index)] = bool(
            len(rows) == len(strategies)
            and all(np.array_equal(offsets[0], offset) for offset in offsets[1:]))
    result = dict(
        seed_scheme=('Paired initial states: seed=2700+episode_index shared across strategies; '
                     'random action seed=100000+strategy_index*100+episode_index'),
        policy_interval_s=env.frame_skip * env.sim.model.opt.timestep,
        physics_interval_s=env.sim.model.opt.timestep,
        action_range=[-1, 1], action_dimension=2,
        observation_empirical_min=obs.min(axis=0).tolist(),
        observation_empirical_max=obs.max(axis=0).tolist(),
        observation_checks=observation_checks,
        summaries=grouped, paired_initial_pose_checks=paired_pose_checks, episodes=episodes,
        limitations=('Three paired, reproducible reset poses with +/-0.2 degree joint jitter per strategy. '
                     'Not an RL evaluation or generalization estimate; contact force is a sampled diagnostic.'),
    )
    path = out / 'button_slider_action_baselines.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756']
    labels = ['Outside hold', 'Press hold', 'Alternating', 'Random']
    for strategy, color, label in zip(strategies, colors, labels):
        row = next(e for e in episodes if e['strategy'] == strategy)
        axes[0, 0].plot(row['sample_time_s'], np.array(row['positions_m']) * 1000,
                        color=color, label=label)
        axes[0, 1].plot(row['sample_time_s'], row['forces_N'], color=color, label=label)
    x = np.arange(len(strategies))
    axes[1, 0].bar(x - 0.18, [grouped[s]['successes'] for s in strategies],
                   width=0.36, color=colors, label='Success')
    axes[1, 0].bar(x + 0.18, [grouped[s]['timeouts'] for s in strategies],
                   width=0.36, color=colors, alpha=0.35, label='Timeout')
    axes[1, 1].bar(x, [grouped[s]['mean_return'] for s in strategies], color=colors)
    axes[0, 0].set(ylabel='Button travel (mm)', xlabel='Simulation time (s)', title='Representative episode')
    axes[0, 1].set(ylabel='Button force (N)', xlabel='Simulation time (s)', title='Contact force')
    axes[1, 0].set(ylabel='Episodes (of 3)', title='Success and timeout counts', xticks=x, xticklabels=labels)
    axes[1, 1].set(ylabel='Mean return', title='Mean episode return', xticks=x, xticklabels=labels)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.25)
    axes[0, 0].legend(ncol=2)
    axes[0, 1].legend(ncol=2)
    axes[1, 0].legend()
    fig.tight_layout()
    fig.savefig(out / 'button_slider_action_baselines.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(summaries=grouped,
                          paired_initial_pose_checks=paired_pose_checks,
                          observation_min=result['observation_empirical_min'],
                          observation_max=result['observation_empirical_max'],
                          observation_checks=observation_checks), indent=2))
    env.close()
    assert all(observation_checks.values()), 'Observation normalization check failed'
    assert all(paired_pose_checks.values()), 'Strategies did not share identical reset poses'


if __name__ == '__main__':
    main()
