"""Compare an optional shaping wrapper with unchanged base rewards."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from button_slider_env import ButtonSliderEnv
from button_slider_potential_shaping import ButtonSliderPotentialShaping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "trajectory_check_2026-10-27"
SEED = 2700
GAMMA = 0.99  # Stable-Baselines3 PPO default unless explicitly overridden.
WEIGHT = 0.25  # Small experiment coefficient; not a safety parameter.


def manual_action(simulation_time_s):
    """The previously validated fixed press/hold/retract baseline."""
    t = simulation_time_s
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
    return np.full(2, value, dtype=np.float64)


def run_trial():
    base_env = ButtonSliderEnv(max_episode_steps=150)
    shaped_env = ButtonSliderPotentialShaping(
        ButtonSliderEnv(max_episode_steps=150),
        weight=WEIGHT,
        gamma=GAMMA,
        reach_scale_m=0.006,
    )
    base_observation, _ = base_env.reset(seed=SEED)
    shaped_observation, _ = shaped_env.reset(seed=SEED)
    np.testing.assert_array_equal(base_observation, shaped_observation)

    rows = []
    base_return = shaped_return = shaping_return = 0.0
    for _ in range(base_env.max_episode_steps):
        action = manual_action(base_env.sim.data.time)
        base_observation, base_reward, base_terminated, base_truncated, base_info = base_env.step(action)
        shaped_observation, shaped_reward, terminated, truncated, shaped_info = shaped_env.step(action)

        np.testing.assert_array_equal(base_observation, shaped_observation)
        assert base_terminated == terminated and base_truncated == truncated
        assert base_info["events"] == shaped_info["events"]
        assert base_info["reward_components"] == {
            name: value for name, value in shaped_info["reward_components"].items()
            if name != "potential_shaping"
        }
        shaping_reward = shaped_info["reward_shaping"]["shaping_reward"]
        assert np.isclose(shaped_reward, base_reward + shaping_reward, rtol=0, atol=1e-12)

        base_return += float(base_reward)
        shaped_return += float(shaped_reward)
        shaping_return += float(shaping_reward)
        events = shaped_info["events"]
        rows.append(dict(
            time_s=float(shaped_info["time_s"]),
            phase_before=shaped_info["reward_shaping"]["phase_before"],
            phase_after=shaped_info["reward_shaping"]["phase_after"],
            base_reward=float(base_reward),
            shaping_reward=float(shaping_reward),
            shaped_reward=float(shaped_reward),
            potential_before=float(shaped_info["reward_shaping"]["potential_before"]),
            potential_after=float(shaped_info["reward_shaping"]["potential_after"]),
            slider_position_m=float(shaped_info["slider_position_m"]),
            contact_loaded=bool(shaped_env.unwrapped._row["press_task"]["contact_loaded"]),
            events=events,
        ))
        if terminated or truncated:
            break

    success = bool(shaped_info.get("is_success", False))
    assert success, "The fixed successful baseline must remain successful under reward-only shaping"
    assert base_return == sum(row["base_reward"] for row in rows)
    assert abs(shaping_return) > 0

    discounted_base_return = sum(
        (GAMMA ** index) * row["base_reward"] for index, row in enumerate(rows))
    discounted_shaped_return = sum(
        (GAMMA ** index) * row["shaped_reward"] for index, row in enumerate(rows))
    discounted_shaping_return = sum(
        (GAMMA ** index) * row["shaping_reward"] for index, row in enumerate(rows))
    initial_potential = rows[0]["potential_before"]
    assert np.isclose(discounted_shaping_return, -WEIGHT * initial_potential,
                      rtol=1e-10, atol=1e-10)

    # For a stationary nonnegative potential, PBRS gives a non-positive
    # per-step increment, so idling cannot collect positive progress forever.
    stationary_deltas = [WEIGHT * (GAMMA * value - value)
                         for value in np.linspace(0.0, 4.0, 17)]
    assert all(delta <= 1e-12 for delta in stationary_deltas)

    base_env.close()
    shaped_env.close()
    return dict(rows=rows, base_return=base_return, shaped_return=shaped_return,
                shaping_return=shaping_return, success=success,
                discounted_base_return=discounted_base_return,
                discounted_shaped_return=discounted_shaped_return,
                discounted_shaping_return=discounted_shaping_return,
                initial_potential=initial_potential,
                stationary_potential_deltas=stationary_deltas)


def run_press_hold_negative_control():
    """Hold the arm at the press target and verify it earns no release progress."""
    env = ButtonSliderPotentialShaping(
        ButtonSliderEnv(max_episode_steps=150),
        weight=WEIGHT,
        gamma=GAMMA,
        reach_scale_m=0.006,
    )
    observation, _ = env.reset(seed=SEED)
    pressed_loaded_rewards = []
    events = []
    was_truncated = False
    final_potential = None
    for _ in range(env.unwrapped.max_episode_steps):
        observation, reward, terminated, truncated, info = env.step(np.ones(2))
        was_truncated = bool(truncated)
        final_potential = float(info["reward_shaping"]["potential_after"])
        events.extend(event["event"] for event in info["events"])
        shaping = info["reward_shaping"]
        loaded = bool(env.unwrapped._row["press_task"]["contact_loaded"])
        if shaping["phase_before"] == "pressed" and loaded:
            pressed_loaded_rewards.append(shaping["shaping_reward"])
        if terminated or truncated:
            break
    release_count = env.unwrapped.sim.task.release_count
    env.close()
    assert release_count == 0 and "released" not in events
    assert pressed_loaded_rewards, "Negative control never reached loaded pressed phase"
    assert all(value <= 1e-12 for value in pressed_loaded_rewards)
    assert was_truncated and final_potential > 0, "Time-limit truncation must retain bootstrap potential"
    return dict(
        events=events,
        release_count=release_count,
        loaded_pressed_samples=len(pressed_loaded_rewards),
        max_loaded_pressed_shaping_reward=max(pressed_loaded_rewards),
        min_loaded_pressed_shaping_reward=min(pressed_loaded_rewards),
        earns_positive_release_progress=False,
        truncated=was_truncated,
        truncation_bootstrap_potential_retained=final_potential,
    )


def run_default_zero_weight_equivalence():
    base = ButtonSliderEnv(max_episode_steps=150)
    wrapped = ButtonSliderPotentialShaping(ButtonSliderEnv(max_episode_steps=150))
    base_observation, _ = base.reset(seed=SEED)
    wrapped_observation, _ = wrapped.reset(seed=SEED)
    np.testing.assert_array_equal(base_observation, wrapped_observation)
    base_return = wrapped_return = 0.0
    steps = 0
    for _ in range(base.max_episode_steps):
        action = manual_action(base.sim.data.time)
        base_observation, base_reward, base_terminated, base_truncated, base_info = base.step(action)
        wrapped_observation, wrapped_reward, terminated, truncated, info = wrapped.step(action)
        np.testing.assert_array_equal(base_observation, wrapped_observation)
        assert base_reward == wrapped_reward
        assert base_terminated == terminated and base_truncated == truncated
        assert base_info["events"] == info["events"]
        base_return += float(base_reward)
        wrapped_return += float(wrapped_reward)
        steps += 1
        if terminated or truncated:
            break
    base.close()
    wrapped.close()
    assert base_return == wrapped_return
    return dict(default_weight=wrapped.weight, steps=steps,
                base_return=base_return, wrapped_return=wrapped_return,
                exact_reward_equivalence=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_trial()
    press_hold_control = run_press_hold_negative_control()
    zero_weight_check = run_default_zero_weight_equivalence()
    rows = result["rows"]
    phases = ("ready", "press_pending", "pressed", "release_pending")
    phase_colors = dict(ready="#4c78a8", press_pending="#f58518",
                        pressed="#54a24b", release_pending="#b279a2")
    phase_summary = {}
    for phase in phases:
        phase_rows = [row for row in rows if row["phase_before"] == phase]
        phase_summary[phase] = dict(
            samples=len(phase_rows),
            positive_shaping_steps=sum(row["shaping_reward"] > 0 for row in phase_rows),
            negative_shaping_steps=sum(row["shaping_reward"] < 0 for row in phase_rows),
            shaping_reward_sum=float(sum(row["shaping_reward"] for row in phase_rows)),
        )
    report = dict(
        purpose="Optional phase-aware potential shaping replay; the base environment is unchanged.",
        seed=SEED,
        gamma=GAMMA,
        weight=WEIGHT,
        reach_scale_m=0.006,
        potential_stages={
            "ready": "1 - clip(max(site_to_box_signed_distance,0)/reach_scale, 0, 1)",
            "press_pending": "1 + clip(pending_elapsed/press_hold, 0, 1)",
            "pressed": "2 + clip((press_position-slider_position)/(press_position-release_position), 0, 1)",
            "release_pending": "3 + clip(pending_elapsed/release_hold, 0, 1)",
            "terminal": "0",
        },
        shaping_formula="weight * (gamma * Phi(next_state) - Phi(current_state))",
        base_return=result["base_return"],
        shaped_return=result["shaped_return"],
        undiscounted_shaping_sum=result["shaping_return"],
        discounted_base_return=result["discounted_base_return"],
        discounted_shaped_return=result["discounted_shaped_return"],
        discounted_shaping_return=result["discounted_shaping_return"],
        discounted_shaping_expected_offset=-WEIGHT * result["initial_potential"],
        success=result["success"],
        same_observations_and_events_as_base_env=True,
        stationary_potential_shaping_never_positive=True,
        press_hold_negative_control=press_hold_control,
        default_zero_weight_equivalence=zero_weight_check,
        phase_summary=phase_summary,
        episodes={"manual_successful_action": rows},
        limitations=(
            "One paired fixed-action episode validates reward plumbing, not policy learning or reward quality. "
            "Reach scale and coefficient are experiment values for this temporary model, not physical limits. "
            "The fingertip site-to-box distance is a shaping proxy, not mesh contact geometry."),
    )
    json_path = OUTPUT_DIR / "button_slider_potential_shaping_trial.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    times = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, [row["slider_position_m"] * 1000 for row in rows], color="#4c78a8")
    axes[0].set_ylabel("Button travel (mm)")
    axes[0].set_title("Fixed-action replay with optional phase-aware shaping")
    for phase, color in phase_colors.items():
        phase_rows = [row for row in rows if row["phase_before"] == phase]
        axes[1].scatter([row["time_s"] for row in phase_rows],
                        [row["shaping_reward"] for row in phase_rows],
                        s=20, color=color, label=phase)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Shaping reward / policy step")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].legend(ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    image_path = OUTPUT_DIR / "button_slider_potential_shaping_trial.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print(json.dumps(dict(
        json=str(json_path), image=str(image_path),
        base_return=result["base_return"], shaped_return=result["shaped_return"],
        shaping_return=result["shaping_return"],
        discounted_base_return=result["discounted_base_return"],
        discounted_shaped_return=result["discounted_shaped_return"],
        discounted_shaping_return=result["discounted_shaping_return"],
        success=result["success"],
        press_hold_negative_control=press_hold_control,
        default_zero_weight_equivalence=zero_weight_check,
        phase_summary=phase_summary,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
