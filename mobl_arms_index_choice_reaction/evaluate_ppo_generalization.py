"""Evaluate a saved PPO policy over wider reset perturbations."""
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from button_slider_env import ButtonSliderEnv
from button_slider_potential_shaping import ButtonSliderPotentialShaping

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'artifacts/ppo_safety_2026-11-08/ppo_safety_shaped.zip'
OUT = ROOT / 'artifacts/ppo_safety_2026-11-08/ppo_generalization_report.json'

def make_env(jitter):
    return ButtonSliderPotentialShaping(
        ButtonSliderEnv(max_episode_steps=150, initial_pose_jitter_deg=jitter,
                        saturation_penalty=0.03, action_rate_penalty=0.01),
        weight=0.25, gamma=0.99, reach_scale_m=0.006)

def run(model, env, seeds):
    rows=[]; invalid=[]
    for seed in seeds:
        try:
            obs, reset = env.reset(seed=seed)
        except RuntimeError as exc:
            invalid.append(dict(seed=seed, reason=str(exc)))
            continue
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

def main():
    model=PPO.load(str(MODEL), device='cpu'); result={}
    for jitter in (0.2, 0.5, 1.0):
        env=make_env(jitter)
        try: rows, invalid=run(model,env,range(4700,4710))
        finally: env.close()
        result[str(jitter)] = dict(episodes=rows, invalid_resets=invalid, summary={
            'successes':sum(r['success'] for r in rows), 'episodes':len(rows),
            'invalid_resets':len(invalid),
            'success_rate':float(np.mean([r['success'] for r in rows])),
            'mean_saturation_channels':float(np.mean([r['saturation_channels'] for r in rows])),
            'mean_peak_force_N':float(np.mean([r['peak_force_N'] for r in rows]))})
    report={'purpose':'Generalization diagnostic, not a final performance estimate.',
            'model_file':MODEL.relative_to(ROOT).as_posix(), 'jitter_degrees': [0.2,0.5,1.0],
            'results':result}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
