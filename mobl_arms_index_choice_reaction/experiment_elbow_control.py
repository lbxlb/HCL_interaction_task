"""临时单肘关节力矩控制实验；其余独立关节由软等式约束保持。"""
import contextlib
import argparse
import copy
from functools import partial
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnose_collision_filter import EXCLUDED_BODY_PAIRS
from simulator import IndexSimulator


def reference_at(t, initial_angle, offsets, segment_seconds, profile='cosine', approach_seconds=0.4):
    index = min(int(t / segment_seconds), len(offsets) - 1)
    local_time = t - index * segment_seconds
    start = offsets[index - 1] if index else 0.0
    delta = offsets[index] - start
    phase = np.clip(local_time / approach_seconds, 0.0, 1.0)
    if profile == 'quintic':
        blend = 10 * phase ** 3 - 15 * phase ** 4 + 6 * phase ** 5
        desired = initial_angle + start + delta * blend
        velocity = delta / approach_seconds * 30 * phase ** 2 * (1 - phase) ** 2 if local_time < approach_seconds else 0.0
    elif profile == 'cosine':
        desired = initial_angle + start + delta * 0.5 * (1 - np.cos(np.pi * phase))
        velocity = delta * 0.5 * np.pi / approach_seconds * np.sin(np.pi * phase) if local_time < approach_seconds else 0.0
    else:
        raise ValueError(f'Unknown trajectory profile: {profile}')
    return desired, velocity, index


def run(model, initial, mode, target_offset=0.1, sequence=None, release_shoulder=False,
        hold_shoulder=False, shoulder_sequence=None, observer=None, trajectory=reference_at,
        kp=30.0, kd=4.0, sequence_segment_seconds=1.0):
    data = mujoco.MjData(model)
    data.qpos[:] = initial
    if release_shoulder:
        data.eq_active[model.equality('fixture_shoulder_rot').id] = False
    mujoco.mj_forward(model, data)
    shoulder_adr = int(model.joint('shoulder_rot').qposadr[0])
    shoulder_dof = int(model.joint('shoulder_rot').dofadr[0])
    shoulder_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'shoulder_test_motor')
    assert not hold_shoulder or (release_shoulder and shoulder_motor >= 0)
    shoulder_saturations = 0
    joint = model.joint('elbow_flexion')
    adr, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
    motor = model.actuator('elbow_test_motor').id
    mask = np.arange(model.nq) != adr
    dt = model.opt.timestep
    offsets = tuple(sequence) if sequence is not None else (target_offset,)
    segment_seconds = sequence_segment_seconds if sequence is not None else 2.0
    duration = len(offsets) * segment_seconds
    shoulder_offsets = tuple(shoulder_sequence) if shoulder_sequence is not None else (0.0,) * len(offsets)
    assert len(shoulder_offsets) == len(offsets)
    assert shoulder_sequence is None or hold_shoulder
    shoulder_joint = model.joint('shoulder_rot')
    assert all(shoulder_joint.range[0] <= initial[shoulder_adr] + v <= shoulder_joint.range[1]
               for v in shoulder_offsets)
    final_target = initial[adr] + offsets[-1]
    assert all(joint.range[0] <= initial[adr] + offset <= joint.range[1] for offset in offsets)
    steps = round(duration / dt)
    rows = []
    max_other_drift = 0.0
    max_final_error = 0.0
    saturations = 0
    max_joint_drift = np.zeros(model.nq)
    if observer is not None:
        observer(0, model, data)
    for step in range(steps):
        # 按参考轨迹接近目标偏移量，之后保持；不直接修改动力学位置。
        t = step * dt
        desired, velocity, _ = trajectory(t, initial[adr], offsets, segment_seconds)
        shoulder_desired, shoulder_velocity, _ = trajectory(
            t, initial[shoulder_adr], shoulder_offsets, segment_seconds)
        feedback = kp * (desired - data.qpos[adr]) + kd * (velocity - data.qvel[dof])
        # 单轴模型补偿只抵消本轴 bias 与 passive，不包含约束力。
        compensation = data.qfrc_bias[dof] - data.qfrc_passive[dof]
        raw = 0.0 if mode == 'unpowered' else feedback
        if mode == 'pd_compensated':
            raw += compensation
        torque = float(np.clip(raw, -10, 10))
        saturations += int(abs(raw) > 10)
        data.ctrl[motor] = torque
        shoulder_torque = 0.0
        if hold_shoulder:
            shoulder_raw = (kp * (shoulder_desired - data.qpos[shoulder_adr])
                            + kd * (shoulder_velocity - data.qvel[shoulder_dof])
                            + data.qfrc_bias[shoulder_dof] - data.qfrc_passive[shoulder_dof])
            shoulder_torque = float(np.clip(shoulder_raw, -10, 10))
            shoulder_saturations += int(abs(shoulder_raw) > 10)
        if shoulder_motor >= 0:
            data.ctrl[shoulder_motor] = shoulder_torque
        rows.append(dict(time_s=float(t), desired_rad=float(desired),
                         actual_rad=float(data.qpos[adr]), torque_Nm=torque,
                         shoulder_torque_Nm=shoulder_torque,
                         shoulder_desired_offset_rad=float(shoulder_desired - initial[shoulder_adr]),
                         shoulder_offset_rad=float(data.qpos[shoulder_adr] - initial[shoulder_adr])))
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc))
        assert np.isclose(data.time, (step + 1) * dt, atol=1e-12, rtol=0)
        assert np.isclose(data.qfrc_actuator[dof], torque, atol=1e-12, rtol=0)
        assert np.isclose(data.qfrc_actuator[shoulder_dof], shoulder_torque, atol=1e-12, rtol=0)
        max_other_drift = max(max_other_drift, float(np.max(np.abs(data.qpos[mask] - initial[mask]))))
        max_joint_drift = np.maximum(max_joint_drift, np.abs(data.qpos - initial))
        if observer is not None:
            observer(step + 1, model, data)
        if data.time >= duration - 0.5:
            max_final_error = max(max_final_error,
                                  abs(final_target - data.qpos[adr]))
    rows.append(dict(time_s=float(data.time), desired_rad=float(final_target),
                     actual_rad=float(data.qpos[adr]), torque_Nm=float(data.ctrl[motor]),
                     shoulder_torque_Nm=float(data.ctrl[shoulder_motor]) if shoulder_motor >= 0 else 0.0,
                     shoulder_desired_offset_rad=float(shoulder_offsets[-1]),
                     shoulder_offset_rad=float(data.qpos[shoulder_adr] - initial[shoulder_adr])))
    tracking_errors = [row['desired_rad'] - row['actual_rad'] for row in rows]
    shoulder_errors = [s['shoulder_desired_offset_rad'] - s['shoulder_offset_rad'] for s in rows]
    result = dict(mode=mode, steps=steps, elapsed_seconds=float(data.time),
                target_offset_rad=offsets[-1],
                final_error_rad=float(final_target - data.qpos[adr]),
                max_abs_tracking_error_rad=float(max(map(abs, tracking_errors))),
                max_abs_error_last_half_second_rad=float(max_final_error),
                rms_tracking_error_rad=float(np.sqrt(np.mean(np.square(tracking_errors)))),
                max_other_joint_drift_rad=max_other_drift, saturation_steps=saturations,
                peak_abs_torque_Nm=max(abs(row['torque_Nm']) for row in rows),
                final_velocity_rad_s=float(data.qvel[dof]),
                warning_counts=[int(w.number) for w in data.warning], samples=rows)
    result.update(shoulder_fixture_active=not release_shoulder,
                  shoulder_hold_control=hold_shoulder,
                  shoulder_target_sequence_offsets_rad=list(shoulder_offsets),
                  shoulder_max_abs_tracking_error_rad=float(max(map(abs, shoulder_errors))),
                  shoulder_rms_tracking_error_rad=float(np.sqrt(np.mean(np.square(shoulder_errors)))),
                  shoulder_final_error_rad=float(shoulder_errors[-1]),
                  shoulder_saturation_steps=shoulder_saturations,
                  shoulder_peak_abs_torque_Nm=max(abs(s['shoulder_torque_Nm']) for s in rows),
                  shoulder_final_velocity_rad_s=float(data.qvel[shoulder_dof]),
                  max_remaining_joint_drift_rad=float(np.max(max_joint_drift[
                      (np.arange(model.nq) != adr) & (np.arange(model.nq) != shoulder_adr)])),
                  active_equality_count=int(np.count_nonzero(data.eq_active)),
                  max_joint_drift_rad={model.joint(j).name: float(max_joint_drift[model.jnt_qposadr[j]])
                                       for j in range(model.njnt)})
    if sequence is not None:
        segments = []
        per_segment = round(segment_seconds / dt)
        for index, offset in enumerate(offsets):
            # 包含段末边界状态；下一段的初始目标与本段末目标连续。
            part = rows[index * per_segment:(index + 1) * per_segment + 1]
            tail = part[-round(0.2 / dt) - 1:]
            segments.append(dict(target_offset_rad=offset,
                max_abs_tracking_error_rad=max(abs(s['desired_rad'] - s['actual_rad']) for s in part),
                max_abs_error_last_0_2_seconds_rad=max(abs(initial[adr] + offset - s['actual_rad']) for s in tail),
                endpoint_error_rad=float(initial[adr] + offset - part[-1]['actual_rad'])))
            segments[-1].update(
                shoulder_max_abs_tracking_error_rad=max(abs(s['shoulder_desired_offset_rad'] - s['shoulder_offset_rad']) for s in part),
                shoulder_max_abs_error_last_0_2_seconds_rad=max(abs(shoulder_offsets[index] - s['shoulder_offset_rad']) for s in tail))
        result.update(target_sequence_offsets_rad=list(offsets), segment_seconds=segment_seconds,
                      segments=segments)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument('--sweep', action='store_true', help='测试正反方向的四个小幅目标')
    experiments.add_argument('--sequence', action='store_true', help='不重置状态，连续往返四段目标')
    experiments.add_argument('--release-shoulder', action='store_true',
                             help='连续往返中对比肩部旋转夹具启用/停用；肩部无执行器')
    experiments.add_argument('--hold-shoulder', action='store_true',
                             help='肩部夹具停用，对比肩部电机零力矩与补偿 PD 保持')
    experiments.add_argument('--dual-motion', action='store_true', help='肩肘同时小幅往返，与肩部保持作对照')
    parser.add_argument('--output', type=Path, help='JSON 输出路径')
    parser.add_argument('--render-dual', action='store_true', help='配合 --dual-motion 输出实际运动 GIF')
    parser.add_argument('--measure-tip', action='store_true', help='配合 --dual-motion 测量指尖世界坐标误差')
    parser.add_argument('--profile', choices=('cosine', 'quintic'), default='cosine',
                        help='双轴实验的参考轨迹形状，默认保持原半余弦')
    parser.add_argument('--kp', type=float, default=30.0, help='PD比例增益')
    parser.add_argument('--kd', type=float, default=4.0, help='PD微分增益')
    parser.add_argument('--approach-seconds', type=float, default=0.4,
                        help='双轴每段移动时长，取值 (0, 0.8]，每段仍为 1 秒')
    parser.add_argument('--segment-seconds', type=float, default=1.0,
                        help='双轴序列每段总时长，默认 1 秒')
    args = parser.parse_args()
    if args.render_dual and not args.dual_motion:
        parser.error('--render-dual 需要 --dual-motion')
    if args.measure_tip and not args.dual_motion:
        parser.error('--measure-tip 需要 --dual-motion')
    if args.profile != 'cosine' and not args.dual_motion:
        parser.error('--profile quintic 需要 --dual-motion')
    if not 0 < args.approach_seconds < args.segment_seconds:
        parser.error('--approach-seconds 必须大于零且小于 --segment-seconds')
    if args.segment_seconds - args.approach_seconds < 0.2:
        parser.error('每段至少保留 0.2 秒保持时间，以计算段末误差')
    if args.approach_seconds != 0.4 and not args.dual_motion:
        parser.error('自定义 --approach-seconds 需要 --dual-motion')
    if args.segment_seconds != 1.0 and not args.dual_motion:
        parser.error('自定义 --segment-seconds 需要 --dual-motion')
    trajectory = partial(reference_at, profile=args.profile, approach_seconds=args.approach_seconds)
    base = Path(__file__).resolve().parent
    source = base / 'simulation.xml'
    donor = base.parent / 'mobl_arms_index_pointing/simulation.xml'
    original, donor_bytes = source.read_bytes(), donor.read_bytes()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', source)
        sim.reset()
    try:
        initial = sim.data.qpos.copy()
        dt = sim.model.opt.timestep
        positions = {sim.model.joint(j).name: float(initial[sim.model.jnt_qposadr[j]])
                     for j in range(sim.model.njnt)}
    finally:
        sim.close()
    tree = ET.fromstring(original)
    assert tree.find('equality') is None and tree.find('actuator') is None
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
    if args.hold_shoulder or args.dual_motion:
        ET.SubElement(actuator, 'motor', name='shoulder_test_motor', joint='shoulder_rot',
                      gear='1', ctrllimited='true', ctrlrange='-10 10')
    for asset in tree.findall('./asset/*'):
        if 'file' in asset.attrib:
            asset.set('file', str((base / asset.get('file')).resolve()))
    with tempfile.TemporaryDirectory(prefix='panel_elbow_') as directory:
        path = Path(directory) / 'experiment.xml'
        ET.ElementTree(tree).write(path, encoding='utf-8')
        model = mujoco.MjModel.from_xml_path(str(path))
        model.opt.timestep = dt
        assert model.nu == (2 if args.hold_shoulder or args.dual_motion else 1) and model.neq == 17
        sweep_offsets = (-0.1, -0.05, 0.05, 0.1)
        if args.dual_motion:
            snapshots = []
            def capture(step, model, data):
                if step % 25 == 0:
                    snapshots.append((float(data.time), data.qpos.copy()))
            runs = []
            for targets in (None, (-0.05, 0.05, -0.05, 0.0)):
                measurement = None
                if args.measure_tip:
                    from measure_fingertip import FingertipMeasurement
                    measurement = FingertipMeasurement(model, initial, (0.1, -0.1, 0.1, 0.0),
                                                       targets or (0.0,) * 4, trajectory,
                                                       args.segment_seconds)
                def observe(step, model, data):
                    if measurement is not None:
                        measurement.observe(step, model, data)
                    if args.render_dual and targets is not None:
                        capture(step, model, data)
                item = run(model, initial, 'pd_compensated', sequence=(0.1, -0.1, 0.1, 0.0),
                           release_shoulder=True, hold_shoulder=True, shoulder_sequence=targets,
                           observer=observe, trajectory=trajectory, kp=args.kp, kd=args.kd,
                           sequence_segment_seconds=args.segment_seconds)
                if measurement is not None:
                    assert len(measurement.samples) == len(item['samples'])
                    item['fingertip_tracking'] = measurement.report()
                runs.append(item)
            if args.render_dual:
                from render_dual_motion import render
                render_dir = base.parent / 'artifacts/trajectory_check_2026-10-21'
                if args.profile != 'cosine':
                    render_dir = render_dir / args.profile
                if args.approach_seconds != 0.4:
                    render_dir = render_dir / f'move_{args.approach_seconds:g}s'
                if args.segment_seconds != 1.0:
                    render_dir = render_dir / f'segment_{args.segment_seconds:g}s'
                render(model, snapshots, render_dir)
        elif args.hold_shoulder:
            runs = [run(model, initial, 'pd_compensated', sequence=(0.1, -0.1, 0.1, 0.0),
                        release_shoulder=True, hold_shoulder=hold, kp=args.kp, kd=args.kd) for hold in (False, True)]
        elif args.release_shoulder:
            runs = [run(model, initial, 'pd_compensated', sequence=(0.1, -0.1, 0.1, 0.0),
                        release_shoulder=release, kp=args.kp, kd=args.kd) for release in (False, True)]
        elif args.sequence:
            runs = [run(model, initial, 'pd_compensated', sequence=(0.1, -0.1, 0.1, 0.0), kp=args.kp, kd=args.kd)]
        elif args.sweep:
            runs = [run(model, initial, 'pd_compensated', target_offset=offset, kp=args.kp, kd=args.kd)
                    for offset in sweep_offsets]
        else:
            runs = [run(model, initial, mode, kp=args.kp, kd=args.kd) for mode in ('unpowered', 'pd', 'pd_compensated')]
    assert source.read_bytes() == original and donor.read_bytes() == donor_bytes
    result = dict(description='软约束夹具下的单肘关节隔离实验，不代表完整上肢控制',
                  target_offset_rad=0.1, approach_seconds=args.approach_seconds, kp=args.kp, kd=args.kd,
                  torque_limit_Nm=10.0, fixture_joints=fixtures, runs=runs)
    if args.sweep:
        result.pop('target_offset_rad')
        result['target_sweep_offsets_rad'] = list(sweep_offsets)
    if args.sequence or args.release_shoulder or args.hold_shoulder or args.dual_motion:
        result.pop('target_offset_rad')
        result.update(target_sequence_offsets_rad=[0.1, -0.1, 0.1, 0.0],
                      segment_seconds=args.segment_seconds, resets_between_segments=False)
    default_output = ('artifacts/trajectory_check_2026-10-21/elbow_target_sweep.json'
                      if args.sweep else 'artifacts/trajectory_check_2026-10-20/elbow_control.json')
    if args.sequence:
        default_output = 'artifacts/trajectory_check_2026-10-21/elbow_continuous_sequence.json'
    if args.release_shoulder:
        result.update(experiment='shoulder_rot_fixture_on_off',
                      released_joint='shoulder_rot', released_joint_actuated=False)
        default_output = 'artifacts/trajectory_check_2026-10-21/elbow_shoulder_release.json'
    if args.hold_shoulder:
        result.update(experiment='shoulder_hold_on_off', shoulder_kp=30.0, shoulder_kd=4.0,
                      shoulder_torque_limit_Nm=10.0, shoulder_target='initial_position',
                      shoulder_fixture_active=False, actuator_count=2)
        default_output = 'artifacts/trajectory_check_2026-10-21/elbow_shoulder_hold.json'
    if args.dual_motion:
        result.update(experiment='shoulder_elbow_dual_motion', shoulder_kp=30.0, shoulder_kd=4.0,
                      shoulder_torque_limit_Nm=10.0, shoulder_fixture_active=False, actuator_count=2)
        default_output = 'artifacts/trajectory_check_2026-10-21/shoulder_elbow_dual_motion.json'
        if args.measure_tip:
            default_output = 'artifacts/trajectory_check_2026-10-21/shoulder_elbow_fingertip.json'
        if args.profile != 'cosine':
            result['trajectory_profile'] = args.profile
            default_output = str(Path(default_output).with_stem(Path(default_output).stem + '_' + args.profile))
        if args.approach_seconds != 0.4:
            default_output = str(Path(default_output).with_stem(
                Path(default_output).stem + f'_move_{args.approach_seconds:g}s'))
        if args.segment_seconds != 1.0:
            default_output = str(Path(default_output).with_stem(
                Path(default_output).stem + f'_segment_{args.segment_seconds:g}s'))
    out = args.output if args.output is not None else base.parent / default_output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for item in runs:
        summary = {k: v for k, v in item.items() if k not in ('samples', 'fingertip_tracking')}
        if 'fingertip_tracking' in item:
            summary['fingertip_tracking'] = {k: v for k, v in item['fingertip_tracking'].items()
                                             if k != 'samples'}
        print(json.dumps(summary, ensure_ascii=False))
    print(out)


if __name__ == '__main__':
    main()
