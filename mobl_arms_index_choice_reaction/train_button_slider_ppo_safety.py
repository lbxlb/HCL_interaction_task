"""PPO diagnostic with explicit saturation and action-change costs."""
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from button_slider_env import ButtonSliderEnv
from button_slider_potential_shaping import ButtonSliderPotentialShaping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'ppo_safety_2026-11-08'
TIMESTEPS = 32768
SEEDS = tuple(range(3700, 3710))
SATURATION_PENALTY = 0.03
ACTION_RATE_PENALTY = 0.01

def make_env(jitter=0.2):
    base = ButtonSliderEnv(max_episode_steps=150, initial_pose_jitter_deg=jitter,
                           saturation_penalty=SATURATION_PENALTY,
                           action_rate_penalty=ACTION_RATE_PENALTY)
    return ButtonSliderPotentialShaping(base, weight=0.25, gamma=0.99, reach_scale_m=0.006)

def evaluate(model, env):
    rows = []
    for seed in SEEDS:
        obs, reset_info = env.reset(seed=seed)
        total = 0.0; done = truncated = False; info = {}; events = []; steps = 0
        components = {}
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            total += float(reward); steps += 1; events.extend(info.get('events', []))
            for key, value in info.get('reward_components', {}).items():
                components[key] = components.get(key, 0.0) + float(value)
        rows.append(dict(seed=seed, steps=steps, return_sum=total,
                         success=bool(info.get('is_success', False)),
                         terminated=bool(done), truncated=bool(truncated), events=events,
                         reward_components=components,
                         peak_normal_force_N=float(info['peak_normal_force_N_episode']),
                         max_slider_position_m=float(info['max_slider_position_m_episode']),
                         saturated_motor_channels=int(info['saturated_motor_channels_episode']),
                         initial_pose_offset_rad=reset_info['initial_pose_offset_rad']))
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    model = PPO('MlpPolicy', env, seed=2026, n_steps=64, batch_size=32, n_epochs=2,
                learning_rate=3e-4, gamma=0.99,
                policy_kwargs={'net_arch': {'pi': [32, 32], 'vf': [32, 32]}},
                device='cpu', verbose=0)
    model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
    path = OUT / 'ppo_safety_shaped'; model.save(str(path)); env.close()
    eval_env = make_env()
    try: rows = evaluate(PPO.load(str(path), env=eval_env, device='cpu'), eval_env)
    finally: eval_env.close()
    report = dict(purpose='Diagnostic PPO with explicit saturation/action-change costs; not final safety validation.',
                  requested_timesteps=TIMESTEPS, actual_timesteps=int(model.num_timesteps),
                  evaluation_jitter_deg=0.2, evaluation_seeds=list(SEEDS),
                  saturation_penalty=SATURATION_PENALTY, action_rate_penalty=ACTION_RATE_PENALTY,
                  model_file=path.with_suffix('.zip').relative_to(ROOT).as_posix(), evaluation=rows,
                  summary=dict(successes=sum(r['success'] for r in rows), episodes=len(rows),
                               success_rate=float(np.mean([r['success'] for r in rows])),
                               mean_return=float(np.mean([r['return_sum'] for r in rows])),
                               mean_saturated_channels=float(np.mean([r['saturated_motor_channels'] for r in rows])),
                               episodes_with_events=sum(bool(r['events']) for r in rows)))
    out = OUT / 'ppo_safety_report.json'; out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert int(model.num_timesteps) == TIMESTEPS

if __name__ == '__main__': main()
