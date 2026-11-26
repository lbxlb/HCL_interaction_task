"""临时恢复本地指向模型的关节等式约束，保留面板碰撞设置。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from diagnose_collision_filter import EXCLUDED_BODY_PAIRS, rollout


def measure(model_path, steps, constraints, restored):
    errors = {c['name']: dict(max_abs_error_rad=0.0) for c in constraints}

    def observe(model, data, step):
        for c in constraints:
            a = model.joint(c['joint1']).qposadr[0]
            b = model.joint(c['joint2']).qposadr[0] if 'joint2' in c else None
            x = 0.0 if b is None else data.qpos[b] - model.qpos0[b]
            coefficients = [float(v) for v in c['polycoef'].split()]
            error = float(data.qpos[a] - model.qpos0[a] - np.polynomial.polynomial.polyval(x, coefficients))
            item = errors[c['name']]
            if step == 0:
                item['initial_error_rad'] = error
            item['max_abs_error_rad'] = max(item['max_abs_error_rad'], abs(error))
            item['final_error_rad'] = error
            if restored:
                eqid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, c['name'])
                rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                      & (data.efc_id == eqid))
                assert len(rows) == 1, c['name']
                assert np.isclose(data.efc_pos[rows[0]], error, atol=1e-12, rtol=0), c['name']

    run = rollout(model_path, steps, observer=observe)
    run['candidate_relation_errors'] = errors
    return run


def compare(steps, include_filtered=False):
    base = Path(__file__).resolve().parent
    source = base / 'simulation.xml'
    donor = base.parent / 'mobl_arms_index_pointing/simulation.xml'
    original, donor_bytes = source.read_bytes(), donor.read_bytes()
    tree = ET.fromstring(original)
    assert tree.find('equality') is None, '正式模型已有约束，请重新审查实验'
    candidates = ET.fromstring(donor_bytes).findall('./equality/joint')
    assert len(candidates) == 12
    constraints = [dict(c.attrib) for c in candidates]
    eq = ET.SubElement(tree, 'equality')
    for c in candidates:
        eq.append(copy.deepcopy(c))
    for element in tree.findall('./asset/*'):
        if 'file' in element.attrib:
            element.set('file', str((base / element.get('file')).resolve()))
    with tempfile.TemporaryDirectory(prefix='panel_constraints_') as directory:
        path = Path(directory) / 'restored.xml'
        ET.ElementTree(tree).write(path, encoding='utf-8')
        before = mujoco.MjModel.from_xml_path(str(source))
        after = mujoco.MjModel.from_xml_path(str(path))
        assert before.neq == 0 and after.neq == 12
        assert np.all(after.eq_active0)
        for name in ['body_mass', 'body_inertia', 'jnt_range', 'jnt_stiffness',
                     'dof_damping', 'dof_armature', 'geom_contype', 'geom_conaffinity',
                     'geom_margin', 'exclude_signature', 'qpos0']:
            np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
        runs = dict(baseline=measure(source, steps, constraints, False),
                    restored=measure(path, steps, constraints, True))
        if include_filtered:
            for label, restore in [('filtered', False), ('combined', True)]:
                variant = copy.deepcopy(tree)
                if not restore:
                    variant.remove(variant.find('equality'))
                contact = variant.find('contact')
                if contact is None:
                    contact = ET.SubElement(variant, 'contact')
                for a, b in EXCLUDED_BODY_PAIRS:
                    ET.SubElement(contact, 'exclude', body1=a, body2=b)
                variant_path = Path(directory) / f'{label}.xml'
                ET.ElementTree(variant).write(variant_path, encoding='utf-8')
                compiled = mujoco.MjModel.from_xml_path(str(variant_path))
                assert compiled.neq == (12 if restore else 0)
                assert compiled.nexclude == before.nexclude + 5
                for name in ['body_mass', 'body_inertia', 'jnt_range', 'jnt_stiffness',
                             'dof_damping', 'dof_armature', 'geom_contype', 'geom_conaffinity',
                             'geom_margin', 'qpos0']:
                    np.testing.assert_array_equal(getattr(before, name), getattr(compiled, name))
                runs[label] = measure(variant_path, steps, constraints, restore)
    assert source.read_bytes() == original and donor.read_bytes() == donor_bytes
    return dict(description=('十二条关节关系与五对碰撞过滤的四组对照' if include_filtered else
                             '仅恢复十二条关节关系；未加入碰撞过滤、执行器或 inactive weld'),
                original_xml_sha256=hashlib.sha256(original).hexdigest(),
                donor_xml_sha256=hashlib.sha256(donor_bytes).hexdigest(),
                candidate_constraints=constraints, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--include-filtered', action='store_true', help='补齐只过滤及约束加过滤的两组对照')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'artifacts/trajectory_check_2026-10-20/joint_constraints_600.json')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps 必须为正整数')
    if args.include_filtered and args.output.name == 'joint_constraints_600.json':
        args.output = args.output.with_name('joint_constraints_collision_four_way.json')
    result = compare(args.steps, args.include_filtered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, run in result['runs'].items():
        print(name, json.dumps(dict(
            max_relation_error_rad=max(c['max_abs_error_rad'] for c in run['candidate_relation_errors'].values()),
            shoulder_girdle=run['joints']['sternoclavicular_r3'],
            shoulder_elevation=run['joints']['shoulder_elv'],
            elbow=run['joints']['elbow_flexion'], warning_counts=run['warning_counts'],
            actual_contact=run['actual_first_target_contact_step'],
            reference_contact=run['reference_first_target_contact_step']), ensure_ascii=False))
    print(f'完整结果：{args.output}')


if __name__ == '__main__':
    main()
