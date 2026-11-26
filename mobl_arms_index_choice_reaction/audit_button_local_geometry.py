"""Inspect local button coordinates around near-surface candidates."""
import json
from pathlib import Path
import mujoco, numpy as np
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent; CAND=ROOT/'artifacts/trajectory_check_2026-11-08/near_surface_candidates.json'; OUT=ROOT/'artifacts/trajectory_check_2026-11-08/button_local_geometry_audit.json'
JOINTS=('elv_angle','shoulder_elv','shoulder_rot','elbow_flexion')
def main():
 model=mujoco.MjModel.from_xml_path(str(BASE/'simulation.xml')); data=mujoco.MjData(model); site=model.site('fingertip').id; finger=model.geom('hand_2distph').id; candidates=json.loads(CAND.read_text())['candidates']; result={}; qa=[int(model.jnt_qposadr[model.joint(j).id]) for j in JOINTS]
 for i in range(4):
  q=np.asarray(candidates[str(i)]['joint_values_rad']); bid=model.geom(f'button-{i}').id; data.qpos[qa]=q; mujoco.mj_forward(model,data); rot=data.geom_xmat[bid].reshape(3,3); local=rot.T@(data.site_xpos[site]-data.geom_xpos[bid]); half=model.geom_size[bid,:3]; d=float(mujoco.mj_geomDistance(model,data,finger,bid,10,None)); normal=rot[:,2];
  result[str(i)]={'joint_values_rad':q.tolist(),'button_center_world_m':data.geom_xpos[bid].tolist(),'button_rotation_columns_world':rot.tolist(),'button_half_size_m':half.tolist(),'fingertip_site_world_m':data.site_xpos[site].tolist(),'fingertip_site_local_m':local.tolist(),'geom_distance_m':d,'front_axis_dot_world':normal.tolist(),'local_inside_xy':bool(abs(local[0])<=half[0] and abs(local[1])<=half[1])}
 report={'purpose':'Local geometry audit; no dynamics or task-success claim.','results':result,'limitations':'Site position is not the full fingertip mesh; local box coordinates do not replace contact manifold analysis.'}; OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
