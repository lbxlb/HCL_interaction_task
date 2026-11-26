"""Coarse four-DOF reachability scan including shoulder elevation joints."""
import json
from pathlib import Path
import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / 'artifacts/trajectory_check_2026-11-08'
JOINTS = ('elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(BASE / 'simulation.xml'))
    data = mujoco.MjData(model)
    finger = model.geom('hand_2distph').id
    buttons = [model.geom(f'button-{i}').id for i in range(4)]
    jids = [model.joint(name).id for name in JOINTS]
    qadr = [int(model.jnt_qposadr[j]) for j in jids]
    ranges = {name: model.jnt_range[j].tolist() for name, j in zip(JOINTS, jids)}
    mins = {str(i): float('inf') for i in range(4)}
    argmins = {}
    contacts = {str(i): 0 for i in range(4)}
    grids = [np.linspace(model.jnt_range[j, 0], model.jnt_range[j, 1], 9) for j in jids]
    samples = 0
    for values in np.ndindex(*(len(g) for g in grids)):
        for adr, grid, idx in zip(qadr, grids, values): data.qpos[adr] = grid[idx]
        mujoco.mj_forward(model, data); samples += 1
        for i, button in enumerate(buttons):
            dist = float(mujoco.mj_geomDistance(model, data, finger, button, 10, None))
            if dist < mins[str(i)]:
                mins[str(i)] = dist
                argmins[str(i)] = [float(v) for v in (grids[k][values[k]] for k in range(4))]
            if dist <= 0: contacts[str(i)] += 1
    rng = np.random.default_rng(20261108)
    random_samples = 100000
    for _ in range(random_samples):
        values = [rng.uniform(model.jnt_range[j, 0], model.jnt_range[j, 1]) for j in jids]
        for adr, value in zip(qadr, values): data.qpos[adr] = value
        mujoco.mj_forward(model, data); samples += 1
        for i, button in enumerate(buttons):
            dist = float(mujoco.mj_geomDistance(model, data, finger, button, 10, None))
            if dist < mins[str(i)]:
                mins[str(i)] = dist; argmins[str(i)] = [float(v) for v in values]
            if dist <= 0: contacts[str(i)] += 1
    report = {
        'purpose': 'Coarse geometry scan only; no dynamics, trajectory, or task-success claim.',
        'joints': list(JOINTS), 'grid_points_per_joint': 9, 'random_samples': random_samples, 'samples': samples,
        'joint_ranges_rad': ranges, 'min_geom_distances_m': mins,
        'argmin_joint_values_rad': argmins, 'nonpositive_distance_sample_counts': contacts,
        'limitations': 'Grid resolution is coarse; positive/negative geom distance is a geometric diagnostic, not a valid press trajectory.'
    }
    path = OUT / 'button_reachability_multidof.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
