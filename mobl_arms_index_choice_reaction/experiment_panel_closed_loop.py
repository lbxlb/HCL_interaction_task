"""面板任务闭环 PD 控制原型实验：动态注入等式约束与力矩电机，测试主动到达目标按钮。"""
import argparse
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

    with tempfile.TemporaryDirectory(prefix='panel_closed_loop_') as directory:
        path = Path(directory) / 'experiment.xml'
        ET.ElementTree(tree).write(path, encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(path))
        model.opt.timestep = dt
        data = mujoco.MjData(model)
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)

        # Create a lightweight simulator wrapper for task
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
        shoulder_adr = int(model.joint('shoulder_rot').qposadr[0])
        elbow_adr = int(model.joint('elbow_flexion').qposadr[0])

        print("🚀 开始闭环面板控制原型测试（运行 200 步）...")
        for step in range(200):
            # Target offset to reach button
            t = step * dt
            phase = np.clip(t / 0.5, 0.0, 1.0)
            desired_elbow = initial[elbow_adr] + 0.15 * 0.5 * (1 - np.cos(np.pi * phase))
            desired_shoulder = initial[shoulder_adr] + 0.05 * 0.5 * (1 - np.cos(np.pi * phase))
            vel_elbow = 0.15 * 0.5 * np.pi / 0.5 * np.sin(np.pi * phase) if t < 0.5 else 0.0
            vel_shoulder = 0.05 * 0.5 * np.pi / 0.5 * np.sin(np.pi * phase) if t < 0.5 else 0.0

            # PD control + bias compensation
            kp, kd = 50.0, 8.0
            elbow_raw = kp * (desired_elbow - data.qpos[elbow_adr]) + kd * (vel_elbow - data.qvel[elbow_dof]) + data.qfrc_bias[elbow_dof] - data.qfrc_passive[elbow_dof]
            shoulder_raw = kp * (desired_shoulder - data.qpos[shoulder_adr]) + kd * (vel_shoulder - data.qvel[shoulder_dof]) + data.qfrc_bias[shoulder_dof] - data.qfrc_passive[shoulder_dof]

            data.ctrl[elbow_motor] = float(np.clip(elbow_raw, -10, 10))
            data.ctrl[shoulder_motor] = float(np.clip(shoulder_raw, -10, 10))

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            status = task.update()
            if status['done']:
                print(f"🎉 任务在第 {step} 步达成！结果：{status['outcome']}")
                break

        print("✅ 闭环面板控制原型测试正常结束！")

if __name__ == '__main__':
    main()
