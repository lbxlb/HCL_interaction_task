"""Check Gymnasium API, deterministic reset, timeout and scripted task completion."""
import json
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from button_slider_env import ButtonSliderEnv


def main():
    env = ButtonSliderEnv(max_episode_steps=200)
    check_env(env, skip_render_check=True)
    checks = dict(gymnasium_checker=True,
                  finite_observation_bounds=bool(np.isfinite(env.observation_space.low).all()
                  and np.isfinite(env.observation_space.high).all()
                  and np.all(env.observation_space.low == -1)
                  and np.all(env.observation_space.high == 1)))
    initial, _ = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step([-1, -1])
    again, _ = env.reset(seed=42)
    checks['reset_repeatable'] = np.array_equal(initial, again)
    elapsed = env.sim.data.time
    try:
        env.step([np.nan, 0])
    except ValueError:
        checks['invalid_action_rejected_before_physics'] = env.sim.data.time == elapsed
    events, total, peak = [], 0.0, 0.0
    for step in range(env.max_episode_steps):
        # A hand-written smooth approach/withdrawal, not a learned policy.
        t = float(env.sim.data.time)
        phase = np.clip((t - 0.04) / 0.2, 0, 1)
        blend = 3 * phase**2 - 2 * phase**3
        if t >= 1.0:
            phase = np.clip((t - 1.0) / 0.2, 0, 1)
            blend = 1 - (3 * phase**2 - 2 * phase**3)
        action = np.full(2, 2 * blend - 1)
        obs, reward, terminated, truncated, info = env.step(action)
        events.extend(info['events'])
        total += reward
        peak = max(peak, info['slider_position_m'])
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
    checks['scripted_press_and_release_success'] = terminated and not truncated and info['is_success']
    checks['events_once_and_in_order'] = [e['event'] for e in events] == ['pressed', 'released']
    checks['finite_reward'] = np.isfinite(total).item()
    checks['no_warning'] = not any(w.number for w in env.sim.data.warning)
    terminal_info = info.copy()
    try:
        env.step([-1, -1])
    except RuntimeError:
        checks['step_after_done_rejected'] = True
    env.max_episode_steps = 2
    env.reset()
    for _ in range(2):
        _, _, term, trunc, _ = env.step([-1, -1])
    checks['outside_hold_times_out'] = trunc and not term
    result = dict(checks=checks, events=events, scripted_final_info=terminal_info,
                  scripted_return=total, sampled_peak_position_m=peak,
                  action_shape=[2], observation_shape=[15],
                  observation_fields=list(env.observation_fields),
                  observation_low=env.observation_space.low.tolist(),
                  observation_high=env.observation_space.high.tolist(),
                  physics_dt_s=0.002,
                  policy_dt_s=0.01,
                  initial_pose_jitter_deg=env.initial_pose_jitter_deg,
                  limitations=('Seeded shoulder/elbow jitter within +/-0.2 degrees on the temporary model; '
                               'scripted demonstration only, no trained-policy performance claim.'))
    path = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-27/button_slider_env_check.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    env.close()
    assert all(checks.values()), 'Environment validation failed; inspect report'


if __name__ == '__main__':
    main()
