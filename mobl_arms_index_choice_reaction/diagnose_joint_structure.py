"""列出面板模型关节层级，并与本地指向模型的等式约束核对。"""
import contextlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from simulator import IndexSimulator


def main():
    base = Path(__file__).resolve().parent
    source = base.parent / 'mobl_arms_index_pointing/simulation.xml'
    panel_tree = ET.parse(base / 'simulation.xml')
    pointing_tree = ET.parse(source)
    panel_joints = {j.get('name'): j for j in panel_tree.findall('.//worldbody//joint')}
    pointing_joints = {j.get('name'): j for j in pointing_tree.findall('.//worldbody//joint')}
    constraints = [dict(j.attrib) for j in pointing_tree.findall('./equality/joint')]
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', base / 'simulation.xml')
        sim.reset()
    rows = []
    try:
        for jid in range(sim.model.njnt):
            name = sim.model.joint(jid).name
            body = int(sim.model.jnt_bodyid[jid])
            parent = int(sim.model.body_parentid[body])
            related = [c for c in constraints if name in (c.get('joint1'), c.get('joint2'))]
            rows.append(dict(name=name, body=sim.model.body(body).name,
                             parent_body=sim.model.body(parent).name,
                             joint_type=int(sim.model.jnt_type[jid]),
                             qpos_adr=int(sim.model.jnt_qposadr[jid]),
                             dof_adr=int(sim.model.jnt_dofadr[jid]),
                             limited=bool(sim.model.jnt_limited[jid]),
                             configured_range=sim.model.jnt_range[jid].tolist(),
                             reset_rad=float(sim.data.qpos[sim.model.jnt_qposadr[jid]]),
                             demo_driven=name in sim.key_joints,
                             matching_joint_attributes_in_local_pointing=(name in pointing_joints and
                                 panel_joints[name].attrib == pointing_joints[name].attrib),
                             local_pointing_constraints=related))
        for constraint in constraints:
            for key in ('joint1', 'joint2'):
                if key in constraint:
                    assert constraint[key] in panel_joints, constraint
        assert len(rows) == 18 and sim.model.neq == 0
        assert all(r['joint_type'] == int(mujoco.mjtJoint.mjJNT_HINGE) for r in rows)
        dependent = {c['joint1'] for c in constraints}
        result = dict(description='本地模型对照，不代表已核实上游原始设计；未修改模型',
                      panel_neq=sim.model.neq, local_pointing_joint_equalities=len(constraints),
                      independent_candidates_if_restored=[r['name'] for r in rows if r['name'] not in dependent],
                      joints=rows, candidate_constraints=constraints)
        out = base.parent / 'artifacts/trajectory_check_2026-10-20/joint_structure.json'
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        lines = ['# 面板模型关节结构核对', '',
                 '来源：当前面板 XML 与本地指向 XML。未修改正式模型；本地指向模型不能视为已验证的上游原版。', '',
                 '| 关节 | 所属 body ← 父 body | 有限位 | 演示增量驱动 | 与指向模型关节属性相同 |',
                 '|---|---|---|---|---|']
        for r in rows:
            lines.append(f"| {r['name']} | {r['body']} ← {r['parent_body']} | {r['limited']} | {r['demo_driven']} | {r['matching_joint_attributes_in_local_pointing']} |")
        lines += ['', '## 本地指向模型中存在、面板模型中缺少的约束', '',
                  '下表是 XML 的 joint1、joint2 与 polycoef 原值；不能仅凭名称推断解剖学有效性。', '',
                  '| joint1 | joint2（无则约束单关节） | polycoef |', '|---|---|---|']
        for c in constraints:
            lines.append(f"| {c['joint1']} | {c.get('joint2', '无')} | {c['polycoef']} |")
        lines += ['', '若这些关系恢复且有效，按依赖关系计数剩余的独立候选关节：', '',
                  ', '.join(result['independent_candidates_if_restored']), '',
                  '该计数不是约束雅可比秩验证，也不表示已经实现六自由度控制。']
        out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k not in ('joints', 'candidate_constraints')}, ensure_ascii=False, indent=2))
        print('关节属性不一致或在本地指向模型缺失：', [r['name'] for r in rows if not r['matching_joint_attributes_in_local_pointing']])
        print(out.with_suffix('.md'))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
