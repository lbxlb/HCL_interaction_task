"""Audit candidate, outward, and inward geometric clearances."""
import json
from pathlib import Path
import mujoco, numpy as np
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent
CAND=ROOT/'artifacts/trajectory_check_2026-11-08/button_reachability_multidof.json'; OUT=ROOT/'artifacts/trajectory_check_2026-11-08/button_clearance_audit.json'
JOINTS=('elv_angle','shoulder_elv','shoulder_rot','elbow_flexion')
def main():
 model=mujoco.MjModel.from_xml_path(str(BASE/'simulation.xml')); data=mujoco.MjData(model); scan=json.loads(CAND.read_text()); qa=np.array([model.jnt_qposadr[model.joint(j).id] for j in JOINTS],int); da=np.array([model.jnt_dofadr[model.joint(j).id] for j in JOINTS],int); site=model.site('fingertip').id; finger=model.geom('hand_2distph').id; result={}
 for i in range(4):
  bid=model.geom(f'button-{i}').id; target=np.asarray(scan['argmin_joint_values_rad'][str(i)],float); data.qpos[qa]=target; mujoco.mj_forward(model,data); normal=data.geom_xmat[bid].reshape(3,3)[:,2].copy(); jac=np.zeros((3,model.nv)); mujoco.mj_jacSite(model,data,jac,None,site)
  out_delta=np.linalg.lstsq(jac[:,da],normal*.002,rcond=None)[0]; in_delta=np.linalg.lstsq(jac[:,da],normal*-.0004,rcond=None)[0]
  poses={'candidate':target,'outward_2mm':target+out_delta,'inward_0.4mm':target+in_delta}; distances={}
  for name,q in poses.items(): data.qpos[qa]=q; mujoco.mj_forward(model,data); distances[name]=float(mujoco.mj_geomDistance(model,data,finger,bid,10,None))
  path=[]
  for step in range(101):
   a=step/100; data.qpos[qa]=(1-a)*poses['outward_2mm']+a*poses['candidate']; mujoco.mj_forward(model,data); d=float(mujoco.mj_geomDistance(model,data,finger,bid,10,None));
   if d<=0 and not path: path=[step,d]
  result[str(i)]={'distances_m':distances,'first_contact_on_outward_to_candidate':path or None,'target_joint_values_rad':target.tolist()}
 report={'purpose':'Geometric clearance audit only; no dynamics or safety claim.','results':result,'limitations':'Point/geom distance and Jacobian offsets are diagnostics; contact margin and button travel are separate quantities.'}; OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
