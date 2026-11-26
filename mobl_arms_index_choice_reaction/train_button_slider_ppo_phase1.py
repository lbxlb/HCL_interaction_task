"""Longer PPO diagnostic for phase 1; not a final performance evaluation."""
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from button_slider_env import ButtonSliderEnv
from button_slider_potential_shaping import ButtonSliderPotentialShaping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ppo_phase1_2026-11-08"
TOTAL_TIMESTEPS = 32768
EVAL_SEEDS = tuple(range(2700, 2710))


def make_env():
    base = ButtonSliderEnv(max_episode_steps=150, initial_pose_jitter_deg=0.2)
    return ButtonSliderPotentialShaping(base, weight=0.25, gamma=0.99, reach_scale_m=0.006)


def evaluate(model, env):
    rows = []
    for seed in EVAL_SEEDS:
        obs, reset_info = env.reset(seed=seed)
        total = 0.0
        events = []
        done = False
        truncated = False
        info = {}
        steps = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            total += float(reward)
            events.extend(info.get("events", []))
            steps += 1
        rows.append({
            "seed": seed,
            "steps": steps,
            "return": total,
            "success": bool(info.get("is_success", False)),
            "terminated": bool(done),
            "truncated": bool(truncated),
            "events": events,
            "failure_reason": info.get("failure_reason"),
            "peak_normal_force_N": float(info["peak_normal_force_N_episode"]),
            "max_slider_position_m": float(info["max_slider_position_m_episode"]),
            "saturated_motor_channels": int(info["saturated_motor_channels_episode"]),
            "initial_pose_offset_rad": reset_info["initial_pose_offset_rad"],
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    model = PPO(
        "MlpPolicy", env, seed=2026, n_steps=64, batch_size=32, n_epochs=2,
        learning_rate=3e-4, gamma=0.99,
        policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
        device="cpu", verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    model_path = OUT / "ppo_phase1_shaped"
    model.save(str(model_path))
    env.close()
    eval_env = make_env()
    try:
        loaded = PPO.load(str(model_path), env=eval_env, device="cpu")
        rows = evaluate(loaded, eval_env)
    finally:
        eval_env.close()
    report = {
        "purpose": "Longer PPO diagnostic after smoke test; not a final performance or safety evaluation.",
        "algorithm": "Stable-Baselines3 PPO with potential-shaped reward",
        "training_seed": 2026,
        "requested_timesteps": TOTAL_TIMESTEPS,
        "actual_timesteps": int(model.num_timesteps),
        "evaluation_seeds": list(EVAL_SEEDS),
        "reward_shaping": {"weight": 0.25, "gamma": 0.99, "reach_scale_m": 0.006},
        "model_file": model_path.with_suffix(".zip").relative_to(ROOT).as_posix(),
        "evaluation": rows,
        "summary": {
            "successes": sum(r["success"] for r in rows),
            "episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "episodes_with_events": sum(bool(r["events"]) for r in rows),
            "episodes_with_failure": sum(r["failure_reason"] is not None for r in rows),
        },
    }
    path = OUT / "ppo_phase1_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert int(model.num_timesteps) == TOTAL_TIMESTEPS


if __name__ == "__main__":
    main()
