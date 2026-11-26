"""Run a tiny PPO integration test; this is not a task-performance experiment."""
import json
from importlib.metadata import version
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
import torch

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise SystemExit(
        "Stable-Baselines3/PyTorch is not installed in the active environment. "
        "Install mobl_arms_index_choice_reaction/requirements-rl.txt first."
    ) from exc

from button_slider_env import ButtonSliderEnv


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "ppo_smoke_2026-10-27"
MODEL_PATH = ARTIFACT_DIR / "ppo_button_slider_smoke"
TOTAL_TIMESTEPS = 256
ROLLOUT_STEPS = 64


def evaluate(model, env, seeds=(2700, 2701, 2702)):
    episodes = []
    for seed in seeds:
        observation, reset_info = env.reset(seed=seed)
        total_reward = 0.0
        clipped_actions = 0
        action_history = []
        terminated = truncated = False
        info = {}
        steps = 0
        while not (terminated or truncated):
            raw_action, _ = model.predict(observation, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
            action = np.clip(raw_action, -1.0, 1.0)
            clipped_actions += int(not np.array_equal(raw_action, action))
            action_history.append(action.astype(float))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if steps > env.max_episode_steps:
                raise RuntimeError("Evaluation exceeded the configured episode limit")
        episodes.append({
            "seed": int(seed),
            "initial_pose_offset_rad": reset_info["initial_pose_offset_rad"],
            "steps": steps,
            "return": total_reward,
            "success": bool(info.get("is_success", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": info.get("failure_reason"),
            "peak_normal_force_N": float(info["peak_normal_force_N_episode"]),
            "max_slider_position_m": float(info["max_slider_position_m_episode"]),
            "saturated_motor_channels": int(info["saturated_motor_channels_episode"]),
            "clipped_policy_actions": clipped_actions,
            "mean_action": np.mean(action_history, axis=0).tolist(),
            "min_action": np.min(action_history, axis=0).tolist(),
            "max_action": np.max(action_history, axis=0).tolist(),
        })
    return episodes


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_env = ButtonSliderEnv(max_episode_steps=150)
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=2026,
        n_steps=ROLLOUT_STEPS,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    actual_training_timesteps = int(model.num_timesteps)
    model.save(str(MODEL_PATH))
    train_env.close()

    eval_env = ButtonSliderEnv(max_episode_steps=150)
    try:
        loaded_model = PPO.load(str(MODEL_PATH), env=eval_env, device="cpu")
        episodes = evaluate(loaded_model, eval_env)
    finally:
        eval_env.close()

    report = {
        "purpose": "PPO integration smoke test only; not evidence that the policy learned the task.",
        "algorithm": "Stable-Baselines3 PPO with an MLP policy",
        "versions": {
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "pytorch": torch.__version__,
            "stable_baselines3": version("stable-baselines3"),
        },
        "seed": 2026,
        "initial_pose_jitter_deg_per_joint": train_env.initial_pose_jitter_deg,
        "requested_training_timesteps": TOTAL_TIMESTEPS,
        "actual_training_timesteps": actual_training_timesteps,
        "rollout_steps": ROLLOUT_STEPS,
        "batch_size": 32,
        "epochs_per_rollout": 2,
        "device": "cpu",
        "model_file": MODEL_PATH.with_suffix(".zip").relative_to(ROOT).as_posix(),
        "model_reloaded_for_evaluation": True,
        "evaluation": episodes,
        "evaluation_summary": {
            "successes": sum(item["success"] for item in episodes),
            "episodes": len(episodes),
            "mean_return": float(np.mean([item["return"] for item in episodes])),
            "all_truncated": all(item["truncated"] for item in episodes),
        },
        "evaluation_limitations": (
            "The current environment varies shoulder/elbow reset angles within only +/-0.2 degrees. "
            "The evaluation seeds match the three paired action-baseline start poses. Three episodes "
            "are a wiring check, not a performance or generalization estimate."
        ),
    }
    report_path = ARTIFACT_DIR / "ppo_smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    assert len(episodes) == 3
    assert actual_training_timesteps == TOTAL_TIMESTEPS
    assert all(item["terminated"] or item["truncated"] for item in episodes)
    assert all(item["failure_reason"] is None for item in episodes)


if __name__ == "__main__":
    main()
