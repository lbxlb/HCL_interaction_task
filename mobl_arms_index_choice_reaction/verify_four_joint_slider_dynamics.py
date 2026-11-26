"""Four-joint temporary slider dynamics, one model per target button."""
import json, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco, numpy as np

BASE=Path(__file__).resolve().parent; ROOT=BASE.parent
CAND=ROOT/'artifacts/trajectory_check_2026-11-08/near_surface_candidates.json'
OUT=ROOT/'artifacts/trajectory_check_2026-11-08/four_joint_slider_dynamics.json'
OUT_SLOW=ROOT/'artifacts/trajectory_check_2026-11-08/four_joint_slider_dynamics_slow_pd.json'
JOINTS=('elv_angle','shoulder_elv','shoulder_rot','elbow_flexion')

def make_xml(button_name, path):
    root=ET.fromstring((BASE/'simulation.xml').read_bytes())
    contact=root.find('contact')
    if contact is None: contact=ET.SubElement(root,'contact')
    for first,second in (('thorax','scapula'),('thorax','humerus'),('clavicle','scapula'),('scapula','humerus'),('humerus','radius')):
        ET.SubElement(contact,'exclude',body1=first,body2=second)
    body=root.find(f"./worldbody/body[@name='{button_name}']")
    ET.SubElement(body,'joint',name='probe_button_slide',type='slide',axis='0 0 -1',limited='true',range='0 0.005',stiffness='1000',damping='10',springref='0',solreflimit='0.004 1')
    actuator=ET.SubElement(root,'actuator')
    for name,joint in zip(('m0','m1','m2','m3'),JOINTS): ET.SubElement(actuator,'motor',name=name,joint=joint,gear='1',ctrllimited='true',ctrlrange='-10 10')
    for node in root.find('asset').findall('*'):
        if 'file' in node.attrib: node.set('file',str((BASE/node.get('file')).resolve()))
    ET.ElementTree(root).write(path,encoding='utf-8')

def run(button, candidate, xml, kp=50., kd=8., approach_steps=100, press_steps=150,
        outward_m=0.006, inward_m=-0.0008, smooth_profile=False):
    model=mujoco.MjModel.from_xml_path(str(xml)); model.opt.timestep=.002; model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data=mujoco.MjData(model); qids=[model.joint(j).id for j in JOINTS]; qa=np.array([model.jnt_qposadr[j] for j in qids],int); da=np.array([model.jnt_dofadr[j] for j in qids],int)
    finger=model.geom('hand_2distph').id; bid=model.geom(f'button-{button}').id; site=model.site('fingertip').id; slider=model.joint('probe_button_slide').id; sq=int(model.jnt_qposadr[slider]); sd=int(model.jnt_dofadr[slider]); motors=np.arange(4)
    target=np.asarray(candidate,float); data.qpos[qa]=target; mujoco.mj_forward(model,data)
    normal=data.geom_xmat[bid].reshape(3,3)[:,2].copy(); jac=np.zeros((3,model.nv)); mujoco.mj_jacSite(model,data,jac,None,site)
    inward=np.linalg.lstsq(jac[:,da],normal*inward_m,rcond=None)[0]; qpress=target+inward
    outward=np.linalg.lstsq(jac[:,da],normal*outward_m,rcond=None)[0]; qout=target+outward
    data.qpos[qa]=qout; data.qvel[:]=0; mujoco.mj_forward(model,data); data.qpos[sq]=0; data.qvel[sd]=0; gravity=-float(data.qfrc_bias[sd]); model.qpos_spring[sq]=-gravity/1000; mujoco.mj_forward(model,data)
    rows=[]; events=[]; sat=0; finite=True
    for phase,target_q,count in (('approach',qout,approach_steps),('press',qpress,press_steps),('hold',qpress,press_steps),('release',qout,press_steps),('outside',qout,press_steps)):
        start=data.qpos[qa].copy()
        for k in range(count):
            u=(k+1)/count
            if smooth_profile:
                blend=0.5-0.5*np.cos(np.pi*u)
                blend_rate=0.5*np.pi*np.sin(np.pi*u)/(count*model.opt.timestep)
            else:
                blend=u; blend_rate=1/(count*model.opt.timestep)
            desired=start+blend*(target_q-start); vel=blend_rate*(target_q-start)
            raw=kp*(desired-data.qpos[qa])+kd*(vel-data.qvel[da])+data.qfrc_bias[da]-data.qfrc_passive[da]
            sat+=int(np.count_nonzero(np.abs(raw)>10)); data.ctrl[:]=np.clip(raw,-10,10); mujoco.mj_step(model,data)
            force=np.zeros(6); contacts=[]
            for ci in range(data.ncon):
                c=data.contact[ci]
                if bid in (c.geom1,c.geom2): mujoco.mj_contactForce(model,data,ci,force); contacts.append(float(force[0]))
            dist=float(mujoco.mj_geomDistance(model,data,finger,bid,10,None)); loaded=bool(contacts and dist<=0 and max(contacts)>0)
            rows.append((phase,float(data.time),float(data.qpos[sq]),dist,max(contacts,default=0.),loaded,
                         float(np.max(np.abs(target_q-data.qpos[qa]))),
                         float(np.max(np.abs(data.qvel[da]))))); finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    phases={}
    for p in ('approach','press','hold','release','outside'):
        subset=[r for r in rows if r[0]==p]
        phases[p]={'min_distance_m':min(r[3] for r in subset),'max_position_m':max(r[2] for r in subset),
                   'loaded_samples':sum(r[5] for r in subset),'peak_force_N':max(r[4] for r in subset),
                   'max_joint_error_rad':max(r[6] for r in subset),'max_joint_speed_rad_s':max(r[7] for r in subset)}
    return {'button':button,'finite':finite,'min_distance_m':min(r[3] for r in rows),'max_slider_position_m':max(r[2] for r in rows),'peak_force_N':max(r[4] for r in rows),'loaded_samples':sum(r[5] for r in rows),'saturation_samples':sat,'final_slider_position_m':rows[-1][2],'phases':phases}

def main():
    candidates={k:v['joint_values_rad'] for k,v in json.loads(CAND.read_text())['candidates'].items()}; results=[]
    with tempfile.TemporaryDirectory() as td:
        for button in range(4):
            xml=Path(td)/f'b{button}.xml'; make_xml(f'button-{button}',xml); results.append(run(button,candidates[str(button)],xml))
    report={'purpose':'Four-joint temporary slider dynamics; diagnostic only, not final task or hardware validation.','joint_order':list(JOINTS),'controller':{'kp':50,'kd':8,'approach_steps':100,'press_steps':150},'results':results,'limitations':'Uses temporary slider, approximate PD and excluded internal pairs; thresholds are not hardware specifications.'}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
    slow=[]
    with tempfile.TemporaryDirectory() as td:
        for button in range(4):
            xml=Path(td)/f's{button}.xml'; make_xml(f'button-{button}',xml)
            slow.append(run(button,candidates[str(button)],xml,kp=20.,kd=6.,approach_steps=300,press_steps=300,outward_m=0.002,inward_m=-0.0004,smooth_profile=True))
    slow_report={'purpose':'Slower/lower-gain smooth-profile comparison; diagnostic only.', 'joint_order':list(JOINTS),
                 'controller':{'kp':20,'kd':6,'approach_steps':300,'press_steps':300,'outward_m':0.002,'inward_m':-0.0004,'smooth_profile':True}, 'results':slow,
                 'limitations':'Temporary slider and approximate PD; no hardware safety interpretation.'}
    OUT_SLOW.write_text(json.dumps(slow_report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'slow_pd_summary':[{'button':r['button'],'max_slider_position_m':r['max_slider_position_m'],'peak_force_N':r['peak_force_N'],'loaded_samples':r['loaded_samples'],'saturation_samples':r['saturation_samples']} for r in slow]},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
