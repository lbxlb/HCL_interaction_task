"""Multi-seed PPO training and valid-reset test evaluation."""
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from button_slider_env import ButtonSliderEnv
from button_slider_potential_shaping import ButtonSliderPotentialShaping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'ppo_multiseed_2026-11-08'
TRAINING_SEEDS = (2026, 2027, 2028)
TEST_SEEDS = tuple(range(5200, 5250))
TIMESTEPS = 32768

def make_env():
    return ButtonSliderPotentialShaping(
        ButtonSliderEnv(max_episode_steps=150, initial_pose_jitter_deg=0.5,
                        saturation_penalty=0.03, action_rate_penalty=0.01),
        weight=0.25, gamma=0.99, reach_scale_m=0.006)

def evaluate(model, env):
    rows=[]; invalid=[]
    for seed in TEST_SEEDS:
        try: obs, reset = env.reset(seed=seed)
        except RuntimeError as exc:
            invalid.append({'seed': seed, 'reason': str(exc)}); continue
        done=trunc=False; info={}; total=0.; events=[]; steps=0
        while not (done or trunc):
            action,_=model.predict(obs, deterministic=True)
            obs,reward,done,trunc,info=env.step(np.asarray(action,dtype=np.float32))
            total += float(reward); steps += 1; events.extend(info.get('events', []))
        rows.append(dict(seed=seed, steps=steps, return_sum=total, success=bool(info.get('is_success',False)),
                         terminated=bool(done), truncated=bool(trunc), events=events,
                         saturation_channels=int(info['saturated_motor_channels_episode']),
                         peak_force_N=float(info['peak_normal_force_N_episode']),
                         initial_pose_offset_rad=reset['initial_pose_offset_rad']))
    return rows, invalid

def make_model(env, seed):
    return PPO('MlpPolicy', env, seed=seed, n_steps=64, batch_size=32, n_epochs=2,
                learning_rate=3e-4, gamma=0.99,
                policy_kwargs={'net_arch': {'pi': [32,32], 'vf': [32,32]}},
                device='cpu', verbose=0)

def main():
    OUT.mkdir(parents=True, exist_ok=True); arms={}
    for seed in TRAINING_SEEDS:
        env=make_env()
        model=make_model(env, seed); model.learn(total_timesteps=TIMESTEPS, progress_bar=False)
        path=OUT / f'ppo_seed_{seed}'; model.save(str(path)); env.close()
        eval_env=make_env()
        try: rows, invalid=evaluate(PPO.load(str(path), env=eval_env, device='cpu'), eval_env)
        finally: eval_env.close()
        arms[str(seed)] = dict(model_file=path.with_suffix('.zip').relative_to(ROOT).as_posix(),
            actual_timesteps=int(model.num_timesteps), evaluation=rows, invalid_resets=invalid,
            summary={'valid_episodes':len(rows), 'invalid_resets':len(invalid),
                     'successes':sum(r['success'] for r in rows),
                     'success_rate':float(np.mean([r['success'] for r in rows])) if rows else 0.0,
                     'mean_return':float(np.mean([r['return_sum'] for r in rows])) if rows else 0.0,
                     'mean_saturation_channels':float(np.mean([r['saturation_channels'] for r in rows])) if rows else 0.0,
                     'mean_peak_force_N':float(np.mean([r['peak_force_N'] for r in rows])) if rows else 0.0})
    report={'purpose':'Three-seed PPO diagnostic on valid reset test seeds; not final performance or safety evaluation.',
            'training_seeds':list(TRAINING_SEEDS), 'test_seed_range':[5200,5249],
            'initial_pose_jitter_deg':0.5, 'timesteps_per_seed':TIMESTEPS,
            'quality_costs':{'saturation_penalty':0.03,'action_rate_penalty':0.01}, 'arms':arms}
    path=OUT/'ppo_multiseed_report.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':path.relative_to(ROOT).as_posix(), 'summaries':{k:v['summary'] for k,v in arms.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
