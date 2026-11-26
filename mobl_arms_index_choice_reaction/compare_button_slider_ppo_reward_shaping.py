"""Small paired PPO pilot for base versus potential-shaped rewards.

This is a diagnostic comparison, not a convergence or safety evaluation.
Both arms use the same PPO configuration and random seed, then evaluate on
the same three deterministic initial-pose seeds.
"""
import json
from importlib.metadata import version
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
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
from button_slider_potential_shaping import ButtonSliderPotentialShaping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "ppo_reward_shaping_ab_2026-10-27"
TRAINING_SEED = 2026
EVALUATION_SEEDS = (2700, 2701, 2702)
TOTAL_TIMESTEPS = 2048
ROLLOUT_STEPS = 64
BATCH_SIZE = 32
EPOCHS_PER_ROLLOUT = 2
GAMMA = 0.99
SHAPING_WEIGHT = 0.25
REACH_SCALE_M = 0.006
MAX_EPISODE_STEPS = 150
INITIAL_POSE_JITTER_DEG = 0.2


def make_env(use_shaping):
    env = ButtonSliderEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        initial_pose_jitter_deg=INITIAL_POSE_JITTER_DEG,
    )
    if use_shaping:
        return ButtonSliderPotentialShaping(
            env,
            weight=SHAPING_WEIGHT,
            gamma=GAMMA,
            reach_scale_m=REACH_SCALE_M,
        )
    return env


def evaluate(model, env, seeds=EVALUATION_SEEDS):
    episodes = []
    for seed in seeds:
        observation, reset_info = env.reset(seed=seed)
        total_reward = 0.0
        base_return = 0.0
        shaping_return = 0.0
        clipped_actions = 0
        action_history = []
        events = []
        phase_counts = {}
        component_sums = {}
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
            base_reward = float(info.get("base_reward", reward))
            base_return += base_reward
            reward_shaping = info.get("reward_shaping", {})
            shaping_return += float(reward_shaping.get("shaping_reward", 0.0))
            phase = info["phase"]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            events.extend(dict(event=item["event"], time_s=float(item["time_s"]))
                          for item in info["events"])
            for name, value in info.get("reward_components", {}).items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value)
            steps += 1
            if steps > env.unwrapped.max_episode_steps:
                raise RuntimeError("Evaluation exceeded the configured episode limit")

        episodes.append({
            "seed": int(seed),
            "initial_pose_offset_rad": reset_info["initial_pose_offset_rad"],
            "steps": steps,
            "training_objective_return": total_reward,
            "base_task_return": base_return,
            "shaping_return": shaping_return,
            "success": bool(info.get("is_success", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": info.get("failure_reason"),
            "events": events,
            "phase_step_counts": phase_counts,
            "reward_component_sums": component_sums,
            "peak_normal_force_N": float(info["peak_normal_force_N_episode"]),
            "max_slider_position_m": float(info["max_slider_position_m_episode"]),
            "saturated_physics_samples": int(info["saturated_physics_samples_episode"]),
            "saturated_motor_channels": int(info["saturated_motor_channels_episode"]),
            "clipped_policy_actions": clipped_actions,
            "mean_action": np.mean(action_history, axis=0).tolist(),
            "min_action": np.min(action_history, axis=0).tolist(),
            "max_action": np.max(action_history, axis=0).tolist(),
        })
    return episodes


def make_model(env):
    return PPO(
        "MlpPolicy",
        env,
        seed=TRAINING_SEED,
        n_steps=ROLLOUT_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS_PER_ROLLOUT,
        learning_rate=3e-4,
        gamma=GAMMA,
        policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
        device="cpu",
        verbose=0,
    )


def summarize(episodes):
    return {
        "episodes": len(episodes),
        "successes": sum(item["success"] for item in episodes),
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "mean_base_task_return": float(np.mean(
            [item["base_task_return"] for item in episodes])),
        "mean_training_objective_return": float(np.mean(
            [item["training_objective_return"] for item in episodes])),
        "mean_shaping_return": float(np.mean(
            [item["shaping_return"] for item in episodes])),
        "mean_steps": float(np.mean([item["steps"] for item in episodes])),
        "mean_peak_normal_force_N": float(np.mean(
            [item["peak_normal_force_N"] for item in episodes])),
        "mean_max_slider_position_m": float(np.mean(
            [item["max_slider_position_m"] for item in episodes])),
        "total_saturated_physics_samples": sum(
            item["saturated_physics_samples"] for item in episodes),
        "total_saturated_motor_channels": sum(
            item["saturated_motor_channels"] for item in episodes),
        "episodes_with_failure": sum(item["failure_reason"] is not None
                                      for item in episodes),
    }


def save_plot(results, path):
    labels = [str(seed) for seed in EVALUATION_SEEDS]
    x = np.arange(len(labels))
    width = 0.36
    base = results["base"]["evaluation"]
    shaped = results["shaped"]["evaluation"]
    metrics = (
        ("base_task_return", "Undiscounted base-task return"),
        ("peak_normal_force_N", "Peak normal force (N)"),
        ("max_slider_position_m", "Maximum button travel (mm)"),
        ("success", "Task success (1 = yes)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, (key, title) in zip(axes.flat, metrics):
        base_values = [float(item[key]) for item in base]
        shaped_values = [float(item[key]) for item in shaped]
        if key == "max_slider_position_m":
            base_values = [value * 1000 for value in base_values]
            shaped_values = [value * 1000 for value in shaped_values]
        axis.bar(x - width / 2, base_values, width, label="Base reward", color="#4c78a8")
        axis.bar(x + width / 2, shaped_values, width,
                 label="Potential shaping", color="#f58518")
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Small PPO reward ablation (3 paired evaluation seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, use_shaping in (("base", False), ("shaped", True)):
        train_env = make_env(use_shaping)
        try:
            model = make_model(train_env)
            model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
            actual_timesteps = int(model.num_timesteps)
            model_path = OUTPUT_DIR / f"ppo_{name}_reward"
            model.save(str(model_path))
        finally:
            train_env.close()

        eval_env = make_env(use_shaping)
        try:
            loaded_model = PPO.load(str(model_path), env=eval_env, device="cpu")
            episodes = evaluate(loaded_model, eval_env)
        finally:
            eval_env.close()
        results[name] = {
            "requested_training_timesteps": TOTAL_TIMESTEPS,
            "actual_training_timesteps": actual_timesteps,
            "rollout_steps": ROLLOUT_STEPS,
            "ppo_updates": actual_timesteps // ROLLOUT_STEPS,
            "model_file": model_path.with_suffix(".zip").relative_to(ROOT).as_posix(),
            "evaluation": episodes,
            "summary": summarize(episodes),
        }

    for base_episode, shaped_episode in zip(
            results["base"]["evaluation"], results["shaped"]["evaluation"]):
        assert base_episode["seed"] == shaped_episode["seed"]
        np.testing.assert_array_equal(
            base_episode["initial_pose_offset_rad"],
            shaped_episode["initial_pose_offset_rad"],
        )
    assert all(result["actual_training_timesteps"] == TOTAL_TIMESTEPS
               for result in results.values())

    report = {
        "purpose": (
            "Budget-limited paired PPO diagnostic for reward plumbing and early policy behavior; "
            "not evidence of convergence, generalization, or physical safety."
        ),
        "algorithm": "Stable-Baselines3 PPO with an MLP policy",
        "versions": {
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "pytorch": torch.__version__,
            "stable_baselines3": version("stable-baselines3"),
        },
        "shared_training_seed": TRAINING_SEED,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "initial_pose_jitter_deg_per_joint": INITIAL_POSE_JITTER_DEG,
        "gamma": GAMMA,
        "total_timesteps_per_arm": TOTAL_TIMESTEPS,
        "rollout_steps": ROLLOUT_STEPS,
        "batch_size": BATCH_SIZE,
        "epochs_per_rollout": EPOCHS_PER_ROLLOUT,
        "device": "cpu",
        "shaping": {
            "weight": SHAPING_WEIGHT,
            "reach_scale_m": REACH_SCALE_M,
            "formula": "weight * (gamma * Phi(next_state) - Phi(current_state))",
        },
        "paired_evaluation_initial_states_match": True,
        "arms": results,
        "limitations": (
            "Only 2,048 training transitions and three paired evaluation episodes per arm. "
            "This is a short diagnostic pilot, not a statistically reliable performance estimate. "
            "The shaping coefficient and reach scale are experimental settings, not safety limits."
        ),
    }
    report_path = OUTPUT_DIR / "ppo_reward_shaping_ab_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")
    image_path = OUTPUT_DIR / "ppo_reward_shaping_ab.png"
    save_plot(results, image_path)
    print(json.dumps({
        "report": report_path.relative_to(ROOT).as_posix(),
        "plot": image_path.relative_to(ROOT).as_posix(),
        "arms": {name: result["summary"] for name, result in results.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
