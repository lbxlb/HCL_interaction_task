"""Select near-surface, low-penetration candidates instead of deepest contacts."""
import json
from pathlib import Path
import mujoco, numpy as np
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent; OUT=ROOT/'artifacts/trajectory_check_2026-11-08/near_surface_candidates.json'
JOINTS=('elv_angle','shoulder_elv','shoulder_rot','elbow_flexion')
def main():
 model=mujoco.MjModel.from_xml_path(str(BASE/'simulation.xml')); data=mujoco.MjData(model); finger=model.geom('hand_2distph').id; jids=[model.joint(j).id for j in JOINTS]; qa=[int(model.jnt_qposadr[j]) for j in jids]; rng=np.random.default_rng(20261109); best={str(i):None for i in range(4)}
 for _ in range(300000):
  q=np.array([rng.uniform(model.jnt_range[j,0],model.jnt_range[j,1]) for j in jids]); data.qpos[qa]=q; mujoco.mj_forward(model,data)
  for i in range(4):
   d=float(mujoco.mj_geomDistance(model,data,finger,model.geom(f'button-{i}').id,10,None)); score=abs(d) if d>=0 else abs(d)+0.02
   if best[str(i)] is None or score<best[str(i)]['score']: best[str(i)]={'score':score,'distance_m':d,'joint_values_rad':q.tolist()}
 report={'purpose':'Near-surface geometric candidate selection; no dynamic or task-success claim.','samples':300000,'joint_order':list(JOINTS),'candidates':best,'selection':'minimize absolute distance while penalizing penetration by 20 mm','limitations':'Random search resolution and geom distance do not establish a valid force-free approach trajectory.'}; OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
