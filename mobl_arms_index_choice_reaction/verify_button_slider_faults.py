"""Inject non-finite state and MuJoCo warnings to check fault termination."""
from pathlib import Path

import numpy as np

from button_slider_env import ButtonSliderEnv
import button_slider_simulation as simulation_module


def main():
    results = {}
    env = ButtonSliderEnv()
    try:
        env.reset(seed=12)
        last_valid = env._last_valid_observation.copy()
        original_mj_step = simulation_module.mujoco.mj_step

        def nonfinite(model, data):
            original_mj_step(model, data)
            data.qpos[0] = np.nan

        simulation_module.mujoco.mj_step = nonfinite
        try:
            obs, reward, terminated, truncated, info = env.step([0, 0])
        finally:
            simulation_module.mujoco.mj_step = original_mj_step
        results['nonfinite_fault_terminated'] = terminated and not truncated
        results['nonfinite_fault_reason'] = info['failure_reason'] == 'non_finite_state'
        results['nonfinite_returns_last_valid_observation'] = np.array_equal(obs, last_valid)
        results['nonfinite_penalty_reported'] = info['reward_components']['fault'] == env.fault_penalty
        results['nonfinite_raw_info_finite'] = bool(np.isfinite(
            info['joint_position_rad'] + info['joint_velocity_rad_s'] +
            [info['slider_position_m'], info['slider_velocity_m_s']]).all())
        results['nonfinite_raw_info_is_last_valid'] = info['joint_position_rad'] == env._last_valid_raw_state['joint_position_rad']
        try:
            env.step([0, 0])
            results['post_fault_step_rejected'] = False
        except RuntimeError:
            results['post_fault_step_rejected'] = True
        env.reset(seed=12)
        last_valid = env._last_valid_observation.copy()

        def warning(model, data):
            original_mj_step(model, data)
            data.warning[0].number = 1

        simulation_module.mujoco.mj_step = warning
        try:
            obs, reward, terminated, truncated, info = env.step([-1, -1])
        finally:
            simulation_module.mujoco.mj_step = original_mj_step
        results['warning_fault_terminated'] = terminated and not truncated
        results['warning_fault_reason'] = info['failure_reason'] == 'mujoco_warning'
        results['warning_details_reported'] = bool(info['failure_details']['warnings'])
        results['warning_penalty_reported'] = info['reward_components']['fault'] == env.fault_penalty
        results['warning_returns_last_valid_observation'] = np.array_equal(obs, last_valid)
        results['warning_raw_info_is_last_valid'] = info['joint_position_rad'] == env._last_valid_raw_state['joint_position_rad']
        results['warning_info_time_is_last_valid'] = info['time_s'] == info['last_valid_observation_time_s']
    finally:
        env.close()
    report = dict(checks=results, fault_penalty=-11.0,
                  penalty_basis='Magnitude equals the +1 press and +10 release bonuses combined; '
                  'fault still terminates separately. This is task reward design, not a physical safety limit.',
                  limitations='Faults are injected at the environment boundary; no real MuJoCo warning '
                  'or unstable physical state was induced.')
    path = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-27/button_slider_fault_checks.json'
    path.write_text(__import__('json').dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(__import__('json').dumps(report, ensure_ascii=False, indent=2))
    assert all(results.values()), 'Fault handling checks failed'


if __name__ == '__main__':
    main()
