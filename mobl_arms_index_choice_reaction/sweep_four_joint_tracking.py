"""No-contact four-joint tracking sweep for candidate poses."""
import json, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco, numpy as np
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent
CAND=ROOT/'artifacts/trajectory_check_2026-11-08/button_reachability_multidof.json'; OUT=ROOT/'artifacts/trajectory_check_2026-11-08/four_joint_tracking_sweep.json'
JOINTS=('elv_angle','shoulder_elv','shoulder_rot','elbow_flexion')
def make_xml(path):
 root=ET.fromstring((BASE/'simulation.xml').read_bytes()); a=ET.SubElement(root,'actuator')
 for n,j in zip(('m0','m1','m2','m3'),JOINTS): ET.SubElement(a,'motor',name=n,joint=j,gear='1',ctrllimited='true',ctrlrange='-10 10')
 for node in root.find('asset').findall('*'):
  if 'file' in node.attrib: node.set('file',str((BASE/node.get('file')).resolve()))
 ET.ElementTree(root).write(path,encoding='utf-8')
def run(model,target,kp,kd):
 data=mujoco.MjData(model); qa=np.array([model.jnt_qposadr[model.joint(j).id] for j in JOINTS],int); da=np.array([model.jnt_dofadr[model.joint(j).id] for j in JOINTS],int); data.qpos[qa]=0; mujoco.mj_forward(model,data); sat=0; maxerr=0
 for step in range(3000):
  alpha=min(1.,(step+1)/1500); desired=alpha*target; vel=target/(1500*model.opt.timestep) if step<1500 else np.zeros(4)
  raw=kp*(desired-data.qpos[qa])+kd*(vel-data.qvel[da])+data.qfrc_bias[da]-data.qfrc_passive[da]; sat+=int(np.count_nonzero(np.abs(raw)>10)); data.ctrl[:]=np.clip(raw,-10,10); mujoco.mj_step(model,data); maxerr=max(maxerr,float(np.max(np.abs(target-data.qpos[qa]))))
 return {'final_error_rad':float(np.max(np.abs(target-data.qpos[qa]))),'max_error_rad':maxerr,'saturation_samples':sat,'finite':bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())}
def main():
 targets=json.loads(CAND.read_text())['argmin_joint_values_rad']; result={}
 with tempfile.TemporaryDirectory() as td:
  xml=Path(td)/'track.xml'; make_xml(xml); model=mujoco.MjModel.from_xml_path(str(xml)); model.opt.timestep=.002
  for kp,kd in ((20,6),(50,8),(100,15)):
   result[f'{kp}/{kd}']={str(i):run(model,np.asarray(targets[str(i)],float),kp,kd) for i in range(4)}
 report={'purpose':'No-contact four-joint tracking sweep; not task or safety validation.','controller_sweep':list(result),'results':result,'limitations':'Temporary motors on formal panel copy; torque limit is an experiment setting.'}; OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
