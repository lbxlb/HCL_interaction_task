"""固定姿态、零速度下核对被动力与重力项；不修改 XML 或运行控制器。"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from simulator import IndexSimulator


def audit(reference_steps, model_path=None):
    base = Path(__file__).resolve().parent
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', model_path or base / 'simulation.xml')
        sim.reset()
    model = sim.model
    gravity = model.opt.gravity.copy()
    reference = sim.data.qpos.copy()
    directions = {name: 1 for name in sim.key_joints}
    for _ in range(reference_steps):
        for name, info in sim.key_joints.items():
            adr = info['qpos_adr']
            reference[adr], directions[name] = sim._advance_position(
                reference[adr], directions[name], info)
    result = dict(
        description='固定 qpos、qvel=0、无演示位置增量；对比同一姿态下开启/关闭重力的前向计算',
        njnt=model.njnt, nv=model.nv, nu=model.nu, ntendon=model.ntendon,
        neq=model.neq, gravity_m_s2=gravity.tolist(),
        reference_steps=reference_steps, joints={}, poses={},
        inertia_review_candidates=[])
    for name, info in sim.key_joints.items():
        jid = info['joint_id']
        dof = int(model.jnt_dofadr[jid])
        body = int(model.jnt_bodyid[jid])
        result['joints'][name] = dict(
            joint_id=jid, qpos_adr=info['qpos_adr'], dof_adr=dof,
            range_rad=list(info['range']), stiffness_Nm_per_rad=float(model.jnt_stiffness[jid]),
            spring_reference_rad=float(model.qpos_spring[info['qpos_adr']]),
            damping_Nm_s_per_rad=float(model.dof_damping[dof]),
            armature_kg_m2=float(model.dof_armature[dof]),
            body_name=model.body(body).name, body_mass_kg=float(model.body_mass[body]),
            body_principal_inertia_kg_m2=model.body_inertia[body].tolist())
    for body in range(1, model.nbody):
        if model.body_mass[body] <= 0.00101 and np.max(model.body_inertia[body]) >= 1:
            result['inertia_review_candidates'].append(dict(
                body_name=model.body(body).name, mass_kg=float(model.body_mass[body]),
                principal_inertia_kg_m2=model.body_inertia[body].tolist()))
    try:
        for label, qpos in [('reset', sim.data.qpos.copy()), ('reference', reference)]:
            states = []
            for gravity_on in [True, False]:
                model.opt.gravity[:] = gravity if gravity_on else 0
                data = mujoco.MjData(model)
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                mass = np.empty((model.nv, model.nv))
                mujoco.mj_fullM(model, mass, data.qM)
                residual = mass @ data.qacc + data.qfrc_bias - (
                    data.qfrc_passive + data.qfrc_actuator + data.qfrc_applied + data.qfrc_constraint)
                assert np.max(np.abs(residual)) < 1e-8, '动力学方程残差异常'
                states.append((data, mass))
            on, mass = states[0]
            off, _ = states[1]
            residual = mass @ on.qacc + on.qfrc_bias - (
                on.qfrc_passive + on.qfrc_actuator + on.qfrc_applied + on.qfrc_constraint)
            pose = dict(contact_count=on.ncon,
                        max_dynamics_residual_Nm=float(np.max(np.abs(residual))),
                        contacts=[dict(geom1=model.geom(int(c.geom1)).name,
                                       geom2=model.geom(int(c.geom2)).name,
                                       distance_m=float(c.dist)) for c in on.contact], joints={})
            for name, info in sim.key_joints.items():
                dof = result['joints'][name]['dof_adr']
                pose['joints'][name] = dict(
                    position_rad=float(qpos[info['qpos_adr']]),
                    bias_with_gravity_Nm=float(on.qfrc_bias[dof]),
                    bias_without_gravity_Nm=float(off.qfrc_bias[dof]),
                    gravity_bias_difference_Nm=float(on.qfrc_bias[dof] - off.qfrc_bias[dof]),
                    passive_at_zero_velocity_Nm=float(on.qfrc_passive[dof]),
                    actuator_Nm=float(on.qfrc_actuator[dof]),
                    constraint_Nm=float(on.qfrc_constraint[dof]),
                    acceleration_with_gravity_rad_s2=float(on.qacc[dof]),
                    acceleration_without_gravity_rad_s2=float(off.qacc[dof]),
                    generalized_mass_diagonal_kg_m2=float(mass[dof, dof]))
            result['poses'][label] = pose
        return result
    finally:
        model.opt.gravity[:] = gravity
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-steps', type=int, default=396)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.reference_steps < 0:
        parser.error('--reference-steps 必须为非负整数')
    content = json.dumps(audit(args.reference_steps), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
    print(content, end='')


if __name__ == '__main__':
    main()
