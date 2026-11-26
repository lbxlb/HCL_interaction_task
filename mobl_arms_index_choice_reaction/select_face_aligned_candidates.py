"""Select candidates near the button front face with normal alignment."""
import json
from pathlib import Path
import mujoco, numpy as np
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent; OUT=ROOT/'artifacts/trajectory_check_2026-11-08/face_aligned_candidates.json'
JOINTS=('elv_angle','shoulder_elv','shoulder_rot','elbow_flexion')
def main():
 model=mujoco.MjModel.from_xml_path(str(BASE/'simulation.xml')); data=mujoco.MjData(model); finger=model.geom('hand_2distph').id; site=model.site('fingertip').id; jids=[model.joint(j).id for j in JOINTS]; qa=[int(model.jnt_qposadr[j]) for j in jids]; rng=np.random.default_rng(20261110); best={str(i):None for i in range(4)}; accepted={str(i):0 for i in range(4)}
 for _ in range(500000):
  q=np.array([rng.uniform(model.jnt_range[j,0],model.jnt_range[j,1]) for j in jids]); data.qpos[qa]=q; mujoco.mj_forward(model,data)
  for i in range(4):
   bid=model.geom(f'button-{i}').id; rot=data.geom_xmat[bid].reshape(3,3); local=rot.T@(data.site_xpos[site]-data.geom_xpos[bid]); half=model.geom_size[bid,:3]; d=float(mujoco.mj_geomDistance(model,data,finger,bid,10,None));
   face_margin=float(half[0]-abs(local[0])); vertical_margin=float(half[1]-abs(local[1])); front=float(local[2]+half[2]);
   if abs(d)<0.001 and face_margin>=0 and vertical_margin>=0 and 0<=front<=0.003:
    accepted[str(i)]+=1; score=abs(d)+abs(front-0.001)
    if best[str(i)] is None or score<best[str(i)]['score']: best[str(i)]={'score':score,'distance_m':d,'local_site_m':local.tolist(),'joint_values_rad':q.tolist(),'face_margin_m':face_margin,'vertical_margin_m':vertical_margin}
 report={'purpose':'Face-aligned geometric candidate selection; no dynamics or task-success claim.','samples':500000,'joint_order':list(JOINTS),'accepted_counts':accepted,'candidates':best,'criteria':'|geom distance|<1 mm, fingertip projection inside front face, local front depth 0..3 mm','limitations':'Random search and local box test do not establish force direction or dynamic reachability.'}; OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
