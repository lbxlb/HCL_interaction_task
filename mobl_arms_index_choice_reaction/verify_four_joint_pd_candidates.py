"""Four-joint PD tracking precheck without the two-joint fixture model."""
import json, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco, numpy as np

BASE=Path(__file__).resolve().parent; ROOT=BASE.parent
CAND=ROOT/'artifacts/trajectory_check_2026-11-08/button_reachability_multidof.json'
OUT=ROOT/'artifacts/trajectory_check_2026-11-08/four_joint_pd_candidates.json'
JOINTS=('elv_angle','shoulder_elv','shoulder_rot','elbow_flexion')

def main():
    source=ET.fromstring((BASE/'simulation.xml').read_bytes())
    actuator=ET.SubElement(source,'actuator')
    for name,joint in zip(('elv_motor','elev_motor','shoulder_motor','elbow_motor'),JOINTS):
        ET.SubElement(actuator,'motor',name=name,joint=joint,gear='1',ctrllimited='true',ctrlrange='-10 10')
    asset=source.find('asset')
    for node in asset.findall('*'):
        if 'file' in node.attrib: node.set('file',str((BASE/node.get('file')).resolve()))
    scan=json.loads(CAND.read_text()); results={}
    with tempfile.TemporaryDirectory() as td:
        xml=Path(td)/'four.xml'; ET.ElementTree(source).write(xml,encoding='utf-8')
        model=mujoco.MjModel.from_xml_path(str(xml)); model.opt.timestep=.002
        for bi in range(4):
            data=mujoco.MjData(model); qadr=[int(model.jnt_qposadr[model.joint(j).id]) for j in JOINTS]
            q0=np.zeros(4); target=np.asarray(scan['argmin_joint_values_rad'][str(bi)],float)
            data.qpos[qadr]=q0; mujoco.mj_forward(model,data); maxerr=0.; sat=0; finite=True
            for step in range(1000):
                alpha=min(1.,(step+1)/500); desired=q0+alpha*(target-q0)
                raw=50*(desired-data.qpos[qadr])-8*data.qvel[[model.jnt_dofadr[model.joint(j).id] for j in JOINTS]]
                sat += int(np.count_nonzero(np.abs(raw)>10)); data.ctrl[:]=np.clip(raw,-10,10)
                mujoco.mj_step(model,data); finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                maxerr=max(maxerr,float(np.max(np.abs(target-data.qpos[qadr]))))
            results[str(bi)]={'target_rad':target.tolist(),'final_rad':data.qpos[qadr].tolist(),
                              'max_final_error_rad':float(np.max(np.abs(target-data.qpos[qadr]))),
                              'max_error_rad':maxerr,'saturation_samples':sat,'finite':finite}
    report={'purpose':'Four-joint PD tracking precheck; no button slider or task success claim.',
            'joint_order':list(JOINTS),'results':results,
            'limitations':'Uses direct panel XML with four temporary motors, no equality fixtures, and no contact task.'}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
