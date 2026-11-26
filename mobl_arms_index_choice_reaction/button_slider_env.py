"""Minimal Gymnasium task around the validated temporary MuJoCo slider."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from button_slider_simulation import ButtonSliderSimulation, SimulationFault


class ButtonSliderEnv(gym.Env):
    metadata = {'render_modes': []}
    observation_fields = (
        'shoulder_position_normalized', 'elbow_position_normalized',
        'shoulder_velocity_tanh_per_1_rad_s', 'elbow_velocity_tanh_per_1_rad_s',
        'slider_position_normalized', 'slider_velocity_tanh_per_0.05_m_s',
        'finger_button_loaded', 'phase_ready', 'phase_press_pending', 'phase_pressed',
        'phase_release_pending', 'pending_time_fraction', 'episode_progress',
        'previous_shoulder_target_normalized', 'previous_elbow_target_normalized',
    )

    def __init__(self, max_episode_steps=150, fault_penalty=-11.0,
                 initial_pose_jitter_deg=0.2, saturation_penalty=0.0,
                 action_rate_penalty=0.0, resample_invalid_reset=True,
                 max_reset_attempts=100):
        if type(max_episode_steps) is not int or max_episode_steps <= 0:
            raise ValueError('max_episode_steps must be a positive integer')
        if not np.isfinite(fault_penalty) or fault_penalty >= 0:
            raise ValueError('fault_penalty must be finite and negative')
        if (not np.isfinite(initial_pose_jitter_deg) or initial_pose_jitter_deg < 0
                or initial_pose_jitter_deg > 1.0):
            raise ValueError('initial_pose_jitter_deg must be within [0, 1.0]')
        for name, value in (('saturation_penalty', saturation_penalty),
                            ('action_rate_penalty', action_rate_penalty)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        if type(max_reset_attempts) is not int or max_reset_attempts <= 0:
            raise ValueError('max_reset_attempts must be a positive integer')
        self.sim = ButtonSliderSimulation()
        self.max_episode_steps = max_episode_steps
        self.fault_penalty = float(fault_penalty)
        self.initial_pose_jitter_deg = float(initial_pose_jitter_deg)
        self.saturation_penalty = float(saturation_penalty)
        self.action_rate_penalty = float(action_rate_penalty)
        self.resample_invalid_reset = bool(resample_invalid_reset)
        self.max_reset_attempts = max_reset_attempts
        self.frame_skip = 5  # 2ms physics, 10ms policy interval.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # All policy features are smoothly mapped to [-1, 1]. Scales are
        # numerical normalization scales; they are not safety limits.
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32)
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Reset options are not supported yet')
        reset_attempts = 0
        while True:
            reset_attempts += 1
            if self.initial_pose_jitter_deg:
                offset_rad = np.deg2rad(self.np_random.uniform(
                    -self.initial_pose_jitter_deg, self.initial_pose_jitter_deg, size=2))
            else:
                offset_rad = np.zeros(2, dtype=float)
            self._row = self.sim.reset(initial_offset_rad=offset_rad)
            finger_name = self.sim.model.geom(self.sim.finger).name
            button_name = self.sim.model.geom(self.sim.button).name
            invalid = any(set(pair) == {finger_name, button_name}
                          for pair in self._row['all_contact_pairs'])
            if not invalid or not self.resample_invalid_reset:
                break
            if reset_attempts >= self.max_reset_attempts:
                raise RuntimeError('Could not sample a non-contact initial state')
        self._initial_pose_offset_rad = offset_rad.copy()
        self._reset_attempts = reset_attempts
        self._reset_seed = int(self.np_random_seed)
        if invalid:
            raise RuntimeError('Randomized reset started with fingertip-button contact')
        # The first PD command should start from the actual randomized pose,
        # otherwise the inferred target velocity ignores the reset offset.
        self._last_target = self.sim.data.qpos[self.sim.qadr].copy()
        self._steps = 0
        self._peak_force_episode = 0.0
        self._max_position_episode = float(self._row['slider_position_m'])
        self._saturated_physics_samples_episode = 0
        self._saturated_motor_channels_episode = 0
        self._peak_force_this_step = 0.0
        self._saturated_channels_this_step = 0
        self._failure_reason = None
        self._failure_details = None
        self._done = False
        self._last_valid_observation = self._observation().copy()
        self._last_valid_observation_time_s = float(self._row['time_s'])
        self._cache_valid_raw_state()
        return self._last_valid_observation.copy(), self._info([], {})

    def _cache_valid_raw_state(self):
        self._last_valid_raw_state = dict(
            time_s=float(self._row['time_s']),
            joint_position_rad=self.sim.data.qpos[self.sim.qadr].astype(float).tolist(),
            joint_velocity_rad_s=self.sim.data.qvel[self.sim.dofs].astype(float).tolist(),
            slider_position_m=float(self._row['slider_position_m']),
            slider_velocity_m_s=float(self._row['slider_velocity_m_s']),
        )

    def _observation(self):
        phase = self.sim.task.phase
        model, data = self.sim.model, self.sim.data
        q_bounds = np.asarray((model.jnt_range[model.joint('shoulder_rot').id],
                               model.jnt_range[model.joint('elbow_flexion').id]))
        q_center = np.mean(q_bounds, axis=1)
        q_scale = (q_bounds[:, 1] - q_bounds[:, 0]) / (2 * np.arctanh(0.95))
        slider_joint = model.joint('probe_button_slide').id
        slider_low, slider_high = model.jnt_range[slider_joint]
        slider_center = (slider_low + slider_high) / 2
        slider_scale = (slider_high - slider_low) / (2 * np.arctanh(0.95))
        target_fraction = ((self._last_target - self.sim.q_out)
                           / (self.sim.q_press - self.sim.q_out))
        values = np.concatenate([
            np.tanh((data.qpos[self.sim.qadr] - q_center) / q_scale),
            np.tanh(data.qvel[self.sim.dofs] / 1.0),
            [np.tanh((self._row['slider_position_m'] - slider_center) / slider_scale),
             np.tanh(self._row['slider_velocity_m_s'] / 0.05),
             float(self._row['press_task']['contact_loaded'])],
            [float(phase == p) for p in ('ready', 'press_pending', 'pressed', 'release_pending')],
            [np.clip(self.sim.task.pending_elapsed_s / 0.02, 0, 1),
             np.clip(self._steps / self.max_episode_steps, 0, 1)],
            2 * target_fraction - 1,
        ])
        return np.clip(values, -1, 1).astype(np.float32)

    def _info(self, events, components):
        raw = (getattr(self, '_last_valid_raw_state', None)
               if getattr(self, '_failure_reason', None) else None)
        if raw is None:
            raw = dict(joint_position_rad=self.sim.data.qpos[self.sim.qadr].astype(float).tolist(),
                       joint_velocity_rad_s=self.sim.data.qvel[self.sim.dofs].astype(float).tolist(),
                       slider_position_m=float(self._row['slider_position_m']),
                       slider_velocity_m_s=float(self._row['slider_velocity_m_s']))
        return dict(time_s=raw.get('time_s', self._row['time_s']), phase=self.sim.task.phase,
                    events=events, press_count=self.sim.task.press_count,
                    release_count=self.sim.task.release_count,
                    slider_position_m=raw['slider_position_m'],
                    joint_position_rad=raw['joint_position_rad'],
                    joint_velocity_rad_s=raw['joint_velocity_rad_s'],
                    slider_velocity_m_s=raw['slider_velocity_m_s'],
                    last_valid_observation_time_s=getattr(
                        self, '_last_valid_observation_time_s', float(self._row['time_s'])),
                    peak_normal_force_N_this_step=self._peak_force_this_step,
                    saturated_motor_channels_this_step=self._saturated_channels_this_step,
                    peak_normal_force_N_episode=self._peak_force_episode,
                    max_slider_position_m_episode=self._max_position_episode,
                    saturated_physics_samples_episode=self._saturated_physics_samples_episode,
                    saturated_motor_channels_episode=self._saturated_motor_channels_episode,
                    failure_reason=getattr(self, '_failure_reason', None),
                    failure_details=getattr(self, '_failure_details', None),
                    reset_seed=getattr(self, '_reset_seed', None),
                    initial_pose_offset_rad=getattr(
                        self, '_initial_pose_offset_rad', np.zeros(2)).astype(float).tolist(),
                    reset_attempts=getattr(self, '_reset_attempts', 1),
                    reward_components=components,
                    is_success=self.sim.task.release_count > 0)

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset() before stepping or after episode end')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Action must be a finite (2,) vector within [-1, 1]')
        target = self.sim.q_out + (action + 1) / 2 * (self.sim.q_press - self.sim.q_out)
        start = self._last_target.copy()
        target_velocity = (target - start) / (self.frame_skip * self.sim.model.opt.timestep)
        events = []
        components = dict(press=0.0, release=0.0, effort=0.0, saturation=0.0,
                          action_rate=0.0, fault=0.0)
        saturated_channels = 0
        peak_force = 0.0
        for substep in range(self.frame_skip):
            interpolated = start + (substep + 1) / self.frame_skip * (target - start)
            saturation_before = self.sim.saturation_count
            try:
                self._row = self.sim.step(interpolated, target_velocity)
            except SimulationFault as fault:
                self._failure_reason = fault.reason
                self._failure_details = dict(fault.details,
                                             last_valid_time_s=self._last_valid_observation_time_s)
                components['fault'] = self.fault_penalty
                self._steps += 1
                self._peak_force_this_step = peak_force
                self._saturated_channels_this_step = saturated_channels
                self._peak_force_episode = max(self._peak_force_episode, peak_force)
                self._saturated_motor_channels_episode += saturated_channels
                self._done = True
                return (self._last_valid_observation.copy(), sum(components.values()),
                        True, False, self._info(events, components))
            saturated_now = self.sim.saturation_count - saturation_before
            saturated_channels += saturated_now
            components['saturation'] -= self.saturation_penalty * saturated_now
            if saturated_now:
                self._saturated_physics_samples_episode += 1
            peak_force = max(peak_force, sum(c['normal_force_N'] for c in self._row['button_contacts']))
            self._max_position_episode = max(self._max_position_episode, self._row['slider_position_m'])
            self._last_target = interpolated.copy()
            event = self._row['press_task']['event']
            if event:
                events.append(dict(event=event, time_s=self._row['time_s']))
                components['press' if event == 'pressed' else 'release'] += 1.0 if event == 'pressed' else 10.0
            # Integral of squared normalized torque, independent of frame_skip.
            components['effort'] -= 0.001 * float(np.sum(
                (np.asarray(self._row['motor_torque_Nm']) / 10) ** 2)) * self.sim.model.opt.timestep
            if event == 'released':
                break
        action_delta = (target - start) / (self.sim.q_press - self.sim.q_out)
        components['action_rate'] -= self.action_rate_penalty * float(np.sum(action_delta ** 2))
        self._steps += 1
        self._peak_force_this_step = peak_force
        self._saturated_channels_this_step = saturated_channels
        self._peak_force_episode = max(self._peak_force_episode, peak_force)
        self._saturated_motor_channels_episode += saturated_channels
        terminated = self.sim.task.release_count > 0
        if getattr(self, '_failure_reason', None):
            terminated = True
        truncated = self._steps >= self.max_episode_steps and not terminated
        self._done = terminated or truncated
        if self._failure_reason:
            self._done = True
            return (self._last_valid_observation.copy(), sum(components.values()),
                    True, False, self._info(events, components))
        observation = self._observation()
        self._last_valid_observation = observation.copy()
        self._last_valid_observation_time_s = float(self._row['time_s'])
        self._cache_valid_raw_state()
        return observation.copy(), sum(components.values()), terminated, truncated, self._info(events, components)
