"""Optional phase-aware potential shaping for the temporary slider task.

The wrapper leaves the base environment and its task criteria unchanged. A
nonzero shaping weight is an experiment setting, not a validated reward.
"""
import numpy as np
import gymnasium as gym
import mujoco

from button_slider_env import ButtonSliderEnv


def fingertip_site_to_button_box_signed_distance(env):
    """Signed point-to-box distance for the fingertip site and target button.

    This is a continuous proximity proxy. The fingertip collision geometry is
    a mesh, so this value is not the physical mesh gap or contact condition.
    """
    base = env.unwrapped
    if not isinstance(base, ButtonSliderEnv):
        raise TypeError("Expected ButtonSliderEnv or a wrapper around it")
    model, data, sim = base.sim.model, base.sim.observation, base.sim
    button_geom = sim.button
    if int(model.geom_type[button_geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("The signed point-to-box potential requires a box button geom")

    site_id = model.site("fingertip").id
    site_world = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    center_world = np.asarray(data.geom_xpos[button_geom], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[button_geom], dtype=np.float64).reshape(3, 3)
    half_size = np.asarray(model.geom_size[button_geom], dtype=np.float64)
    local = rotation.T @ (site_world - center_world)
    excess = np.abs(local) - half_size
    outside_distance = np.linalg.norm(np.maximum(excess, 0.0))
    inside_distance = min(float(np.max(excess)), 0.0)
    return float(outside_distance + inside_distance)


def phase_aware_potential(env, reach_scale_m=0.006):
    """Compute a bounded progress potential from task stages.

    Stage ranges are [0,1] for approach, [1,2] for press confirmation,
    [2,3] for retraction, and [3,4] for release confirmation.
    """
    base = env.unwrapped
    if not isinstance(base, ButtonSliderEnv):
        raise TypeError("Expected ButtonSliderEnv or a wrapper around it")
    if not np.isfinite(reach_scale_m) or reach_scale_m <= 0:
        raise ValueError("reach_scale_m must be finite and positive")

    task = base.sim.task
    phase = task.phase
    if phase == "ready":
        distance = max(0.0, fingertip_site_to_button_box_signed_distance(base))
        return float(1.0 - np.clip(distance / reach_scale_m, 0.0, 1.0))
    if phase == "press_pending":
        fraction = np.clip(task.pending_elapsed_s / task.criteria.press_hold_s, 0.0, 1.0)
        return float(1.0 + fraction)
    if phase == "pressed":
        # A spring-returning slider is not evidence that the arm has released
        # the button while loaded fingertip contact remains.
        if bool(base._row["press_task"]["contact_loaded"]):
            return 2.0
        press_position = task.criteria.press_position_m
        release_position = task.criteria.release_position_m
        slider_position = float(base._row["slider_position_m"])
        release_progress = np.clip(
            (press_position - slider_position) / (press_position - release_position), 0.0, 1.0)
        return float(2.0 + release_progress)
    if phase == "release_pending":
        fraction = np.clip(task.pending_elapsed_s / task.criteria.release_hold_s, 0.0, 1.0)
        return float(3.0 + fraction)
    raise RuntimeError(f"Unexpected button task phase: {phase!r}")


class ButtonSliderPotentialShaping(gym.Wrapper):
    """Add gamma*Phi(next)-Phi(current) without modifying the base task.

    Set ``weight=0`` to preserve base rewards. ``gamma`` must match the RL
    learner's discount factor for the standard potential-based argument.
    On truncation, the next-state potential is retained for time-limit
    bootstrapping. Every true terminal state uses zero potential, as required
    for the discounted shaping terms to telescope to a policy-independent
    offset.
    """

    def __init__(self, env, *, weight=0.0, gamma=0.99, reach_scale_m=0.006):
        super().__init__(env)
        if not isinstance(env.unwrapped, ButtonSliderEnv):
            raise TypeError("Expected ButtonSliderEnv or a wrapper around it")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and non-negative")
        if not np.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not np.isfinite(reach_scale_m) or reach_scale_m <= 0:
            raise ValueError("reach_scale_m must be finite and positive")
        self.weight = float(weight)
        self.gamma = float(gamma)
        self.reach_scale_m = float(reach_scale_m)
        self._potential = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._potential = phase_aware_potential(self, self.reach_scale_m)
        self._phase = self.unwrapped.sim.task.phase
        info = dict(info)
        info["reward_shaping"] = dict(
            potential_before=self._potential,
            potential_after=self._potential,
            phase_before=self._phase,
            phase_after=self._phase,
            shaping_reward=0.0,
            weight=self.weight,
            gamma=self.gamma,
        )
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        previous_potential = self._potential
        previous_phase = self._phase
        next_potential = (0.0 if terminated else
                          phase_aware_potential(self, self.reach_scale_m))
        next_phase = info["phase"]
        shaping_reward = self.weight * (
            self.gamma * next_potential - previous_potential)
        self._potential = next_potential
        self._phase = next_phase

        info = dict(info)
        components = dict(info.get("reward_components", {}))
        components["potential_shaping"] = float(shaping_reward)
        info["reward_components"] = components
        info["base_reward"] = float(base_reward)
        info["reward_shaping"] = dict(
            potential_before=float(previous_potential),
            potential_after=float(next_potential),
            phase_before=previous_phase,
            phase_after=next_phase,
            shaping_reward=float(shaping_reward),
            weight=self.weight,
            gamma=self.gamma,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        return observation, float(base_reward + shaping_reward), terminated, truncated, info
