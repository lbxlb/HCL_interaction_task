"""定点触碰闭环控制实验：寻找到达 button-0 的目标关节角，并通过高增益 PD 控制驱动指尖触碰按钮。"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from diagnose_collision_filter import EXCLUDED_BODY_PAIRS
from simulator import IndexSimulator
from task import ChoicePanelTask

def main():
    base = Path(__file__).resolve().parent
    config_path = base / 'config.yaml'
    source = base / 'simulation.xml'
    donor = base.parent / 'mobl_arms_index_pointing/simulation.xml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # Force target button to 0
    config['target_button'] = 0

    original, donor_bytes = source.read_bytes(), donor.read_bytes()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(config_path, source)
        sim.reset()

    try:
        initial = sim.data.qpos.copy()
        dt = sim.model.opt.timestep
        positions = {sim.model.joint(j).name: float(initial[sim.model.jnt_qposadr[j]])
                     for j in range(sim.model.njnt)}
    finally:
        sim.close()

    tree = ET.fromstring(original)
    equality = ET.SubElement(tree, 'equality')
    for relation in ET.fromstring(donor_bytes).findall('./equality/joint'):
        equality.append(copy.deepcopy(relation))
    fixtures = ['head_rot', 'elv_angle', 'shoulder_elv', 'shoulder_rot', 'pro_sup']
    for name in fixtures:
        ET.SubElement(equality, 'joint', name=f'fixture_{name}', joint1=name,
                      polycoef=f'{positions[name]} 0 0 0 0', solref='0.004 1',
                      solimp='0.9999 0.9999 0.001 0.5 2')
    contact = ET.SubElement(tree, 'contact')
    for a, b in EXCLUDED_BODY_PAIRS:
        ET.SubElement(contact, 'exclude', body1=a, body2=b)
    actuator = ET.SubElement(tree, 'actuator')
    ET.SubElement(actuator, 'motor', name='elbow_test_motor', joint='elbow_flexion',
                  gear='1', ctrllimited='true', ctrlrange='-10 10')
    ET.SubElement(actuator, 'motor', name='shoulder_test_motor', joint='shoulder_rot',
                  gear='1', ctrllimited='true', ctrlrange='-10 10')
    for asset in tree.findall('./asset/*'):
        if 'file' in asset.attrib:
            asset.set('file', str((base / asset.get('file')).resolve()))

    with tempfile.TemporaryDirectory(prefix='panel_reaching_') as directory:
        path = Path(directory) / 'experiment.xml'
        ET.ElementTree(tree).write(path, encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(path))
        model.opt.timestep = dt
        data = mujoco.MjData(model)
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)

        shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'shoulder_rot')
        elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'elbow_flexion')
        s_range = model.jnt_range[shoulder_id]
        e_range = model.jnt_range[elbow_id]
        s_adr = int(model.jnt_qposadr[shoulder_id])
        e_adr = int(model.jnt_qposadr[elbow_id])
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'hand_2distph')
        target_button = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'button-0')

        # Find best target joint configuration for button-0
        best_s, best_e = initial[s_adr], initial[e_adr]
        min_d = float('inf')
        for s_val in np.linspace(s_range[0], s_range[1], 30):
            for e_val in np.linspace(e_range[0], e_range[1], 30):
                data.qpos[s_adr] = s_val
                data.qpos[e_adr] = e_val
                mujoco.mj_forward(model, data)
                d = float(mujoco.mj_geomDistance(model, data, finger, target_button, 10, None))
                if d < min_d:
                    min_d = d
                    best_s, best_e = s_val, e_val

        print(f"🎯 寻找到达 button-0 的最优关节目标：shoulder_rot={best_s:.4f} rad, elbow_flexion={best_e:.4f} rad (最小距离: {min_d:.4f} m)")

        # Reset data to initial posture for trajectory execution
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)

        class ClosedLoopSimulator:
            def __init__(self, model, data, config):
                self.model = model
                self.data = data
                self.config = config
                self.finger_geom_name = "hand_2distph"
                self.is_running = True

        sim_wrapper = ClosedLoopSimulator(model, data, config)
        task = ChoicePanelTask(config, sim_wrapper)
        task.reset()

        shoulder_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'shoulder_test_motor')
        elbow_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'elbow_test_motor')
        shoulder_dof = int(model.joint('shoulder_rot').dofadr[0])
        elbow_dof = int(model.joint('elbow_flexion').dofadr[0])

        print("🚀 开始执行到达 button-0 的闭环控制仿真（上限 500 步）...")
        reach_steps = 400
        for step in range(reach_steps):
            t = step * dt
            # Quintic or cosine trajectory from initial to best target over 0.8 seconds
            duration = 0.8
            phase = np.clip(t / duration, 0.0, 1.0)
            blend = 10 * phase ** 3 - 15 * phase ** 4 + 6 * phase ** 5 if t < duration else 1.0
            vel_blend = (30 * phase ** 2 * (1 - phase) ** 2) / duration if t < duration else 0.0

            target_s = initial[s_adr] + (best_s - initial[s_adr]) * blend
            target_e = initial[e_adr] + (best_e - initial[e_adr]) * blend
            vel_s = (best_s - initial[s_adr]) * vel_blend
            vel_e = (best_e - initial[e_adr]) * vel_blend

            kp, kd = 50.0, 8.0
            shoulder_raw = kp * (target_s - data.qpos[s_adr]) + kd * (vel_s - data.qvel[shoulder_dof]) + data.qfrc_bias[shoulder_dof] - data.qfrc_passive[shoulder_dof]
            elbow_raw = kp * (target_e - data.qpos[e_adr]) + kd * (vel_e - data.qvel[elbow_dof]) + data.qfrc_bias[elbow_dof] - data.qfrc_passive[elbow_dof]

            data.ctrl[shoulder_motor] = float(np.clip(shoulder_raw, -10, 10))
            data.ctrl[elbow_motor] = float(np.clip(elbow_raw, -10, 10))

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            status = task.update()
            if status['done']:
                print(f"🎉 成功在第 {step} 步触碰 button-0！反应时间：{status['reaction_time']:.4f} 秒")
                break

        print("✅ 定点触碰闭环控制实验结束！")

if __name__ == '__main__':
    main()
