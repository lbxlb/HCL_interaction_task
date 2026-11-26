"""在临时模型中排除指定骨骼 body 对，进行静态及可选短程运动对照。"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnose_dynamics import audit
from simulator import IndexSimulator


EXCLUDED_BODY_PAIRS = [
    ('thorax', 'scapula'), ('thorax', 'humerus'),
    ('clavicle', 'scapula'), ('scapula', 'humerus'), ('humerus', 'radius'),
]


def rollout(model_path, steps, observer=None):
    """沿用现有演示增量和动力学；参考副本只做前向计算。"""
    base = Path(__file__).resolve().parent
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', model_path)
        sim.reset()
    initial = sim.data.qpos.copy()
    reference = mujoco.MjData(sim.model)
    finger = sim.model.geom('hand_2distph').id
    target = sim.model.geom(f"button-{sim.config['target_button']}").id
    contacts = {}
    max_drift = np.zeros(sim.model.nq)
    max_error = np.zeros(sim.model.nq)
    max_speed = 0.0
    actual_touch = reference_touch = None
    initial_pairs = {tuple(sorted((sim.model.geom(int(c.geom1)).name,
                                  sim.model.geom(int(c.geom2)).name))) for c in sim.data.contact}
    try:
        for step in range(steps + 1):
            if step:
                with contextlib.redirect_stdout(io.StringIO()):
                    sim.step()
            for array in (sim.data.qpos, sim.data.qvel, sim.data.qacc):
                if not np.isfinite(array).all():
                    raise RuntimeError(f'第 {step} 步出现非有限状态')
            if not np.isclose(sim.data.time, step * sim.model.opt.timestep, atol=1e-12, rtol=0):
                raise RuntimeError('仿真时间异常，可能发生状态重置')
            if observer is not None:
                observer(sim.model, sim.data, step)
            max_drift = np.maximum(max_drift, np.abs(sim.data.qpos - initial))
            max_error = np.maximum(max_error, np.abs(sim.data.qpos - sim.reference_qpos))
            max_speed = max(max_speed, float(np.max(np.abs(sim.data.qvel))))
            for c in sim.data.contact:
                pair = tuple(sorted((sim.model.geom(int(c.geom1)).name,
                                     sim.model.geom(int(c.geom2)).name)))
                if pair not in contacts:
                    contacts[pair] = dict(geoms=list(pair), first_step=step,
                                         new_since_reset=pair not in initial_pairs,
                                         min_distance_m=float(c.dist))
                contacts[pair]['min_distance_m'] = min(contacts[pair]['min_distance_m'], float(c.dist))
            if actual_touch is None and any({int(c.geom1), int(c.geom2)} == {finger, target}
                                             for c in sim.data.contact):
                actual_touch = step
            reference.qpos[:] = sim.reference_qpos
            mujoco.mj_forward(sim.model, reference)
            if reference_touch is None and any({int(c.geom1), int(c.geom2)} == {finger, target}
                                                for c in reference.contact):
                reference_touch = step
        return dict(steps=steps, elapsed_seconds=float(sim.data.time),
                    all_states_finite=True, max_abs_velocity_rad_s=max_speed,
                    actual_first_target_contact_step=actual_touch,
                    reference_first_target_contact_step=reference_touch,
                    contacts=list(contacts.values()),
                    warning_counts=[int(w.number) for w in sim.data.warning],
                    joints={sim.model.joint(j).name: dict(
                        max_displacement_from_reset_rad=float(max_drift[sim.model.jnt_qposadr[j]]),
                        max_reference_error_rad=float(max_error[sim.model.jnt_qposadr[j]]),
                        final_actual_rad=float(sim.data.qpos[sim.model.jnt_qposadr[j]]))
                        for j in range(sim.model.njnt)})
    finally:
        sim.close()


def compare(rollout_steps=0):
    source = Path(__file__).resolve().parent / 'simulation.xml'
    original = source.read_bytes()
    model = mujoco.MjModel.from_xml_path(str(source))
    collision_settings = {}
    for name in sorted({name for pair in EXCLUDED_BODY_PAIRS for name in pair}):
        geom = model.geom(name)
        body = model.body(int(model.geom_bodyid[geom.id])).name
        assert body == name, '本实验假设待排除 geom 与所属 body 同名'
        collision_settings[name] = dict(body=body,
            contype=int(model.geom_contype[geom.id]),
            conaffinity=int(model.geom_conaffinity[geom.id]),
            margin_m=float(model.geom_margin[geom.id]))
    baseline = audit(396)
    tree = ET.fromstring(original)
    # 临时 XML 位于 /tmp；将资源路径显式指回原模型目录。
    for element in tree.findall('./asset/*'):
        if 'file' in element.attrib:
            element.set('file', str((source.parent / element.get('file')).resolve()))
    contact = tree.find('contact')
    if contact is None:
        contact = ET.SubElement(tree, 'contact')
    for a, b in EXCLUDED_BODY_PAIRS:
        ET.SubElement(contact, 'exclude', body1=a, body2=b)
    with tempfile.TemporaryDirectory(prefix='panel_collision_') as directory:
        model_path = Path(directory) / 'filtered.xml'
        ET.ElementTree(tree).write(model_path, encoding='utf-8')
        filtered = audit(396, model_path)
        trajectories = None
        if rollout_steps:
            trajectories = dict(baseline=rollout(source, rollout_steps),
                                filtered=rollout(model_path, rollout_steps))
    assert source.read_bytes() == original, '正式模型发生变化'
    assert baseline['joints'] == filtered['joints'], '关节参数发生变化'
    excluded = {frozenset(pair) for pair in EXCLUDED_BODY_PAIRS}
    summary = {}
    for name, before in baseline['poses'].items():
        after = filtered['poses'][name]
        expected = [c for c in before['contacts']
                    if frozenset((c['geom1'], c['geom2'])) not in excluded]
        assert after['contacts'] == expected, '接触变化超出指定范围'
        summary[name] = dict(contacts_before=before['contact_count'],
                             contacts_after=after['contact_count'], joints={})
        for joint, old in before['joints'].items():
            new = after['joints'][joint]
            for key in ['position_rad', 'bias_with_gravity_Nm', 'bias_without_gravity_Nm',
                        'passive_at_zero_velocity_Nm', 'generalized_mass_diagonal_kg_m2']:
                assert np.isclose(old[key], new[key], rtol=0, atol=1e-12), key
            summary[name]['joints'][joint] = dict(
                constraint_before_Nm=old['constraint_Nm'], constraint_after_Nm=new['constraint_Nm'],
                acceleration_before_rad_s2=old['acceleration_with_gravity_rad_s2'],
                acceleration_after_rad_s2=new['acceleration_with_gravity_rad_s2'])
    assert any({c['geom1'], c['geom2']} == {'button-0', 'hand_2distph'}
               for c in filtered['poses']['reference']['contacts']), '目标接触丢失'
    result = dict(description='仅临时排除五对 body 的碰撞；有限时长诊断，不是正式碰撞策略',
                original_xml_sha256=hashlib.sha256(original).hexdigest(),
                original_collision_settings=collision_settings,
                excluded_body_pairs=EXCLUDED_BODY_PAIRS, summary=summary,
                baseline=baseline, filtered=filtered)
    if trajectories is not None:
        result['rollout'] = trajectories
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout-steps', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'artifacts/trajectory_check_2026-10-20/collision_filter_comparison.json')
    args = parser.parse_args()
    if args.rollout_steps < 0:
        parser.error('--rollout-steps 必须为非负整数')
    result = compare(args.rollout_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    if 'rollout' in result:
        for name, run in result['rollout'].items():
            print(f"{name}: {run['steps']} 步，接触对种类 {len(run['contacts'])}，"
                  f"实际目标接触步 {run['actual_first_target_contact_step']}，"
                  f"参考目标接触步 {run['reference_first_target_contact_step']}")
    print(f'验证通过，完整结果：{args.output}')


if __name__ == '__main__':
    main()
