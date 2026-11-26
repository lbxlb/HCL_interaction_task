"""Kinematic candidate trajectory check for all four panel buttons."""
import json
from pathlib import Path
import mujoco
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'artifacts/trajectory_check_2026-11-08/button_reachability_multidof.json'
OUT = ROOT / 'artifacts/trajectory_check_2026-11-08/button_candidate_trajectory_checks.json'
JOINTS = ('elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion')

def main():
    model = mujoco.MjModel.from_xml_path(str(BASE / 'simulation.xml'))
    data = mujoco.MjData(model)
    finger = model.geom('hand_2distph').id
    report = json.loads(REPORT.read_text())
    candidates = report['argmin_joint_values_rad']
    addresses = [int(model.jnt_qposadr[model.joint(name).id]) for name in JOINTS]
    initial = data.qpos[addresses].copy()
    results = {}
    for button_index in range(4):
        button = model.geom(f'button-{button_index}').id
        target = np.asarray(candidates[str(button_index)], dtype=float)
        distances = []; contact_steps = []; finite = True
        for step in range(101):
            alpha = step / 100.0
            data.qpos[addresses] = (1 - alpha) * initial + alpha * target
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            distance = float(mujoco.mj_geomDistance(model, data, finger, button, 10, None))
            distances.append(distance)
            if distance <= 0: contact_steps.append(step)
            finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        results[str(button_index)] = {
            'target_joint_values_rad': target.tolist(),
            'initial_joint_values_rad': initial.tolist(),
            'min_distance_m': float(min(distances)),
            'first_nonpositive_distance_step': contact_steps[0] if contact_steps else None,
            'nonpositive_distance_samples': len(contact_steps),
            'finite_state_all_steps': finite,
            'trajectory_type': 'direct qpos interpolation; kinematic reachability check only',
        }
    output = {'purpose': 'Candidate trajectory precheck; not dynamic control or button press validation.',
              'steps_per_trajectory': 101, 'joint_order': list(JOINTS), 'results': results,
              'limitations': 'Formal model has no slider actuator in this check; contact distance is geometric and qpos is directly assigned.'}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
