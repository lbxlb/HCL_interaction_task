"""无窗口诊断单轮演示轨迹；不自动重置、不改变模型或控制参数。"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from simulator import IndexSimulator


def diagnose(max_steps, hold_reference_on_contact=False):
    base = Path(__file__).resolve().parent
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', base / 'simulation.xml')
        sim.reset()
    finger = sim.model.geom('hand_2distph').id
    tip = sim.model.site('fingertip').id
    buttons = [sim.model.geom(f'button-{i}').id for i in range(4)]
    target = sim.config['target_button']
    initial_distances = [float(mujoco.mj_geomDistance(sim.model, sim.data, finger, b, 10, None))
                         for b in buttons]
    result = {
        'steps': max_steps,
        'dt_seconds': float(sim.model.opt.timestep),
        'auto_reset': False,
        'reference_description': '只按参考关节位置重算几何，其余关节固定在重置位置；不积分动力学',
        'buttons': [dict(button=i, initial_geom_distance_m=d, min_geom_distance_m=d,
                         min_distance_step=0, first_contact_step=None,
                         reference_min_geom_distance_m=d, reference_min_distance_step=0,
                         reference_first_contact_step=None)
                    for i, d in enumerate(initial_distances)],
        'joints': {},
    }
    held_qpos = None
    if hold_reference_on_contact:
        result['reference_stop'] = dict(
            mode='hold_on_first_target_contact', target_button=target,
            stopped=False, stop_step=None, stop_time_seconds=None,
            held_steps=0, held_target_contact_steps=0,
            max_hold_position_drift_rad=0.0,
            description='仅冻结诊断用参考姿态；不控制实际模型，不验证接触力')
    reference_data = mujoco.MjData(sim.model)
    for name in sim.key_joints:
        result['joints'][name] = dict(max_post_integration_change_rad=0.0,
                                    max_reference_error_rad=0.0)
    try:
        for step in range(1, max_steps + 1):
            commanded = {}
            for name, info in sim.key_joints.items():
                lo, hi = info['range']
                delta = info['speed'] * sim.model.opt.timestep
                commanded[name] = float(np.clip(
                    sim.data.qpos[info['qpos_adr']] + info['direction'] * delta, lo, hi))
            with contextlib.redirect_stdout(io.StringIO()):
                sim.step()
            if not np.isfinite(sim.data.qpos).all():
                raise RuntimeError(f'第 {step} 步出现非有限位置值')
            reference_data.qpos[:] = sim.reference_qpos if held_qpos is None else held_qpos
            mujoco.mj_forward(sim.model, reference_data)
            if hold_reference_on_contact:
                stop = result['reference_stop']
                touching_target = any({int(c.geom1), int(c.geom2)} == {finger, buttons[target]}
                                      for c in reference_data.contact)
                if held_qpos is None and touching_target:
                    held_qpos = reference_data.qpos.copy()
                    stop.update(stopped=True, stop_step=step,
                                stop_time_seconds=float(sim.data.time),
                                stop_geom_distance_m=float(mujoco.mj_geomDistance(
                                    sim.model, reference_data, finger, buttons[target], 10, None)),
                                joint_positions_rad={name: float(held_qpos[info['qpos_adr']])
                                                     for name, info in sim.key_joints.items()})
                elif held_qpos is not None:
                    stop['held_steps'] += 1
                    stop['held_target_contact_steps'] += int(touching_target)
                    stop['max_hold_position_drift_rad'] = max(
                        stop['max_hold_position_drift_rad'],
                        float(np.max(np.abs(reference_data.qpos - held_qpos))))
            for name, info in sim.key_joints.items():
                actual = float(sim.data.qpos[info['qpos_adr']])
                item = result['joints'][name]
                item['max_post_integration_change_rad'] = max(
                    item['max_post_integration_change_rad'], abs(actual - commanded[name]))
                item['max_reference_error_rad'] = max(
                    item['max_reference_error_rad'], abs(actual - reference_data.qpos[info['qpos_adr']]))
                item['final_actual_rad'] = actual
                item['final_reference_rad'] = float(reference_data.qpos[info['qpos_adr']])
            for item, button in zip(result['buttons'], buttons):
                distance = float(mujoco.mj_geomDistance(sim.model, sim.data, finger, button, 10, None))
                if distance < item['min_geom_distance_m']:
                    item['min_geom_distance_m'] = distance
                    item['min_distance_step'] = step
                item['final_geom_distance_m'] = distance
                item['final_tip_to_button_center_m'] = float(np.linalg.norm(
                    sim.data.site_xpos[tip] - sim.data.geom_xpos[button]))
                touched = any({int(c.geom1), int(c.geom2)} == {finger, button}
                              for c in sim.data.contact)
                if touched and item['first_contact_step'] is None:
                    item['first_contact_step'] = step
                reference_distance = float(mujoco.mj_geomDistance(
                    sim.model, reference_data, finger, button, 10, None))
                if reference_distance < item['reference_min_geom_distance_m']:
                    item['reference_min_geom_distance_m'] = reference_distance
                    item['reference_min_distance_step'] = step
                item['reference_final_geom_distance_m'] = reference_distance
                reference_touched = any({int(c.geom1), int(c.geom2)} == {finger, button}
                                        for c in reference_data.contact)
                if reference_touched and item['reference_first_contact_step'] is None:
                    item['reference_first_contact_step'] = step
        result['elapsed_sim_seconds'] = float(sim.data.time)
        return result
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--hold-reference-on-contact', action='store_true',
                        help='首次检测到目标接触后保持诊断用参考姿态，不改变实际运动')
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error('--max-steps 必须是正整数')
    print(json.dumps(diagnose(args.max_steps, args.hold_reference_on_contact),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
