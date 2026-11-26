"""Audit proximity signals without changing the task reward or physics."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from benchmark_button_slider_actions import action_for
from button_slider_env import ButtonSliderEnv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "trajectory_check_2026-10-27"
PPO_MODEL = ROOT / "artifacts" / "ppo_smoke_2026-10-27" / "ppo_button_slider_smoke.zip"
SEED = 2700


def proximity_metrics(env):
    """Return geometric diagnostics; site-to-box distance is not contact force."""
    sim = env.sim
    model, data = sim.model, sim.data
    site_id = model.site("fingertip").id
    finger_point = np.asarray(data.site_xpos[site_id], dtype=float)
    button_center = np.asarray(data.geom_xpos[sim.button], dtype=float)
    button_rotation = np.asarray(data.geom_xmat[sim.button], dtype=float).reshape(3, 3)
    half_size = np.asarray(model.geom_size[sim.button], dtype=float)

    # Transform the fingertip site into the button box's local frame, then use
    # the standard signed distance to an axis-aligned box in that frame.
    local = button_rotation.T @ (finger_point - button_center)
    q = np.abs(local) - half_size
    outside_distance = float(np.linalg.norm(np.maximum(q, 0.0)))
    inside_distance = float(min(float(np.max(q)), 0.0))
    signed_site_box_distance = outside_distance + inside_distance

    return dict(
        time_s=float(data.time),
        fingertip_button_center_distance_m=float(np.linalg.norm(finger_point - button_center)),
        fingertip_local_button_xyz_m=local.tolist(),
        fingertip_site_to_box_signed_distance_m=signed_site_box_distance,
        geom_distance_m=float(env._row["finger_button_distance_m"]),
        contact_loaded=bool(env._row["press_task"]["contact_loaded"]),
        slider_position_m=float(env._row["slider_position_m"]),
    )


def run_episode(policy, seed=SEED):
    env = ButtonSliderEnv(max_episode_steps=150)
    observation, _ = env.reset(seed=seed)
    rows = [proximity_metrics(env)]
    events = []
    total_return = 0.0
    for step in range(env.max_episode_steps):
        if policy == "alternating":
            action = action_for("alternating", env, step, np.random.default_rng(0))
        else:
            action, _ = policy.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float64).reshape(2)
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        row = proximity_metrics(env)
        row["reward"] = float(reward)
        row["reward_components"] = info["reward_components"]
        rows.append(row)
        events.extend(info["events"])
        if terminated or truncated:
            break
    result = dict(
        policy="manual_alternating" if policy == "alternating" else "ppo_smoke",
        seed=seed,
        rows=rows,
        events=events,
        success=bool(info.get("is_success", False)),
        total_return=total_return,
        steps=step + 1,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )
    env.close()
    return result


def summarize(episode):
    rows = episode["rows"]
    loaded = [row for row in rows if row["contact_loaded"]]
    return dict(
        samples=len(rows),
        success=episode["success"],
        events=[(round(event["time_s"], 3), event["event"]) for event in episode["events"]],
        loaded_contact_samples=len(loaded),
        minimum_site_box_signed_distance_m=min(
            row["fingertip_site_to_box_signed_distance_m"] for row in rows),
        minimum_site_box_distance_during_loaded_contact_m=(
            min(row["fingertip_site_to_box_signed_distance_m"] for row in loaded)
            if loaded else None),
        geom_distance_zero_without_loaded_contact_samples=sum(
            row["geom_distance_m"] == 0.0 and not row["contact_loaded"] for row in rows),
        maximum_slider_position_m=max(row["slider_position_m"] for row in rows),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(PPO_MODEL), device="cpu")
    episodes = [run_episode("alternating"), run_episode(model)]
    report = dict(
        purpose="Offline proximity metric audit only; no reward, observation, or physics changes.",
        seed=SEED,
        metrics=("MuJoCo geom distance", "fingertip site to button-box signed distance",
                 "button slider position", "loaded contact flag"),
        distance_limitations=(
            "geom distance can report zero without a matching loaded contact. The site-to-box "
            "signed distance treats the fingertip site as a point and the button geom as a box; "
            "it is a shaping candidate diagnostic, not an exact fingertip-mesh contact gap."),
        summaries={episode["policy"]: summarize(episode) for episode in episodes},
        episodes={episode["policy"]: episode for episode in episodes},
    )
    json_path = OUTPUT_DIR / "button_slider_proximity_metric_audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = {"manual_alternating": "#54a24b", "ppo_smoke": "#4c78a8"}
    for episode in episodes:
        name = episode["policy"]
        rows = episode["rows"]
        times = [row["time_s"] for row in rows]
        label = "Fixed successful action" if name == "manual_alternating" else "PPO smoke policy"
        axes[0].plot(
            times,
            [row["fingertip_site_to_box_signed_distance_m"] * 1000 for row in rows],
            color=colors[name], label=label)
        axes[1].plot(times, [row["geom_distance_m"] * 1000 for row in rows],
                     color=colors[name], label=label)
        loaded_times = [row["time_s"] for row in rows if row["contact_loaded"]]
        if loaded_times:
            axes[0].scatter(loaded_times,
                            [next(row["fingertip_site_to_box_signed_distance_m"] * 1000
                                  for row in rows if row["time_s"] == time)
                             for time in loaded_times],
                            color=colors[name], s=8, alpha=0.6)
    axes[0].set_ylabel("Site-to-box signed distance (mm)")
    axes[0].set_title("Point-to-box proximity and loaded contact")
    axes[1].set_ylabel("mj_geomDistance (mm)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_title("MuJoCo geometry-distance query (diagnostic only)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    image_path = OUTPUT_DIR / "button_slider_proximity_metric_audit.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print(json.dumps(dict(json=str(json_path), image=str(image_path),
                          summaries=report["summaries"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
