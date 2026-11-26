"""对比分析运动学位置驱动（直接修改 qpos）与动力学力矩控制（PD + 补偿 + mj_step）的边界与差异。"""
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

    # 1. Run kinematic position drive (standard IndexSimulator step)
    with contextlib.redirect_stdout(io.StringIO()):
        sim_kinematic = IndexSimulator(config_path, source)
        sim_kinematic.reset()
    
    kinematic_traces = []
    for _ in range(100):
        sim_kinematic.step()
        kinematic_traces.append(sim_kinematic.data.qpos.copy())
    sim_kinematic.close()

    # 2. Setup dynamic torque control model
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

    with tempfile.TemporaryDirectory(prefix='control_boundary_') as directory:
        path = Path(directory) / 'experiment.xml'
        ET.ElementTree(tree).write(path, encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(path))
        model.opt.timestep = dt
        data = mujoco.MjData(model)
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)

        shoulder_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'shoulder_test_motor')
        elbow_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'elbow_test_motor')
        shoulder_dof = int(model.joint('shoulder_rot').dofadr[0])
        elbow_dof = int(model.joint('elbow_flexion').dofadr[0])
        shoulder_adr = int(model.joint('shoulder_rot').qposadr[0])
        elbow_adr = int(model.joint('elbow_flexion').qposadr[0])

        dynamic_traces = []
        for step in range(100):
            t = step * dt
            # Match target trajectory used in kinematic simulator or reference
            target_elbow = initial[elbow_adr] + 0.05 * np.sin(t)
            target_shoulder = initial[shoulder_adr] + 0.02 * np.sin(t)
            
            kp, kd = 50.0, 8.0
            shoulder_raw = kp * (target_shoulder - data.qpos[shoulder_adr]) + kd * (0.0 - data.qvel[shoulder_dof]) + data.qfrc_bias[shoulder_dof] - data.qfrc_passive[shoulder_dof]
            elbow_raw = kp * (target_elbow - data.qpos[elbow_adr]) + kd * (0.0 - data.qvel[elbow_dof]) + data.qfrc_bias[elbow_dof] - data.qfrc_passive[elbow_dof]

            data.ctrl[shoulder_motor] = float(np.clip(shoulder_raw, -10, 10))
            data.ctrl[elbow_motor] = float(np.clip(elbow_raw, -10, 10))

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            dynamic_traces.append(data.qpos.copy())

    report = {
        'description': '对比运动学位置驱动与动力学力矩控制的边界',
        'steps': 100,
        'dt': dt,
        'kinematic_final_elbow': float(kinematic_traces[-1][elbow_adr]),
        'dynamic_final_elbow': float(dynamic_traces[-1][elbow_adr])
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
