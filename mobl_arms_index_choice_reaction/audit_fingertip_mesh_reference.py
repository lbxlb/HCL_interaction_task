"""Compare fingertip site and hand_2distph mesh local geometry."""
import json
from pathlib import Path
import mujoco, numpy as np
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent; OUT=ROOT/'artifacts/trajectory_check_2026-11-08/fingertip_mesh_reference.json'
def main():
 model=mujoco.MjModel.from_xml_path(str(BASE/'simulation.xml')); data=mujoco.MjData(model); mujoco.mj_forward(model,data); gid=model.geom('hand_2distph').id; sid=model.site('fingertip').id; mid=int(model.geom_dataid[gid]); start=int(model.mesh_vertadr[mid]); count=int(model.mesh_vertnum[mid]); verts=model.mesh_vert[start:start+count].copy(); bbox={'min':verts.min(0).tolist(),'max':verts.max(0).tolist(),'center':verts.mean(0).tolist()}; body=int(model.geom_bodyid[gid]); site_body=int(model.site_bodyid[sid]); site_body_local=model.site_pos[sid].tolist(); report={'purpose':'Reference-point audit; no dynamics or task-success claim.','geom':'hand_2distph','mesh_vertex_count':count,'mesh_local_bbox':bbox,'geom_local_pos':model.geom_pos[gid].tolist(),'geom_body':model.body(body).name,'site_body':model.body(site_body).name,'site_pos_in_body_local':site_body_local,'site_world_m':data.site_xpos[sid].tolist(),'geom_world_m':data.geom_xpos[gid].tolist(),'limitations':'Mesh vertices and site use different local frames if attached to different bodies; closest mesh point requires contact data.'}; OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
