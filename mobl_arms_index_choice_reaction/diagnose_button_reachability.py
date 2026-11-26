"""诊断四个按钮的几何位置、初始距离及多关节活动范围内的可达性与穿透情况。"""
import json
import mujoco
import numpy as np
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent
    model = mujoco.MjModel.from_xml_path(str(base / 'simulation.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'hand_2distph')
    buttons = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'button-{i}') for i in range(4)]
    
    report = {
        'buttons_world_pos': {f'button-{i}': list(data.geom_xpos[b]) for i, b in enumerate(buttons)},
        'initial_fingertip_pos': list(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'fingertip')]),
        'reachability_scan': []
    }

    # Scan joint range combinations for shoulder_rot, elbow_flexion, pro_sup
    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'shoulder_rot')
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'elbow_flexion')
    
    s_range = model.jnt_range[shoulder_id]
    e_range = model.jnt_range[elbow_id]
    
    s_adr = int(model.jnt_qposadr[shoulder_id])
    e_adr = int(model.jnt_qposadr[elbow_id])

    min_distances = {i: float('inf') for i in range(4)}
    
    # Coarse grid scan over shoulder and elbow joint limits
    for s_val in np.linspace(s_range[0], s_range[1], 15):
        for e_val in np.linspace(e_range[0], e_range[1], 15):
            data.qpos[s_adr] = s_val
            data.qpos[e_adr] = e_val
            mujoco.mj_forward(model, data)
            
            for i, b in enumerate(buttons):
                dist = float(mujoco.mj_geomDistance(model, data, finger, b, 10, None))
                if dist < min_distances[i]:
                    min_distances[i] = dist

    report['min_geom_distances_across_range_m'] = min_distances
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
