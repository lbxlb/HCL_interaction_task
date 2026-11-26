"""固定其他关节，网格扫描肩旋与肘屈的指尖到目标按钮盒面距离。"""
import contextlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from simulator import IndexSimulator


def point_to_box_signed_distance(point, model, data, geom_id):
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    local = rotation.T @ (point - data.geom_xpos[geom_id])
    offset = np.abs(local) - model.geom_size[geom_id, :3]
    outside = np.maximum(offset, 0.0)
    return float(np.linalg.norm(outside) + min(float(np.max(offset)), 0.0))


def scan():
    base = Path(__file__).resolve().parent
    xml_path = base / 'simulation.xml'
    xml_before = xml_path.read_bytes()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', xml_path)
        sim.reset()
    try:
        model, data = sim.model, sim.data
        initial = data.qpos.copy()
        tip_id = model.site('fingertip').id
        target_id = model.geom(f"button-{sim.config['target_button']}").id
        axes = []
        for name in ('shoulder_rot', 'elbow_flexion'):
            joint = model.joint(name)
            axes.append(dict(name=name, adr=int(joint.qposadr[0]),
                             range_rad=joint.range.copy(), initial_rad=float(initial[joint.qposadr[0]])))
        finger_id = model.geom('hand_2distph').id
        def evaluate(shoulder, elbow):
            data.qpos[:] = initial
            data.qpos[axes[0]['adr']] = shoulder
            data.qpos[axes[1]['adr']] = elbow
            mujoco.mj_kinematics(model, data)
            point = data.site_xpos[tip_id].copy()
            distance = point_to_box_signed_distance(point, model, data, target_id)
            return distance, point
        def finger_clearance(shoulder, elbow):
            data.qpos[:] = initial
            data.qpos[axes[0]['adr']] = shoulder
            data.qpos[axes[1]['adr']] = elbow
            mujoco.mj_kinematics(model, data)
            return float(mujoco.mj_geomDistance(model, data, finger_id, target_id, 10.0, None))

        coarse_axes = [np.linspace(*axis['range_rad'],
                                   int(np.ceil(np.degrees(np.diff(axis['range_rad'])[0]))) + 1)
                       for axis in axes]
        coarse = np.empty((len(coarse_axes[0]), len(coarse_axes[1])))
        for i, shoulder in enumerate(coarse_axes[0]):
            for j, elbow in enumerate(coarse_axes[1]):
                coarse[i, j] = evaluate(shoulder, elbow)[0]
        ci, cj = np.unravel_index(np.argmin(np.abs(coarse)), coarse.shape)
        surface_coarse_best = (coarse_axes[0][ci], coarse_axes[1][cj])
        fine_axes = [np.linspace(max(axis['range_rad'][0], center - np.radians(1)),
                                 min(axis['range_rad'][1], center + np.radians(1)), 21)
                     for axis, center in zip(axes, surface_coarse_best)]
        fine = np.empty((len(fine_axes[0]), len(fine_axes[1])))
        for i, shoulder in enumerate(fine_axes[0]):
            for j, elbow in enumerate(fine_axes[1]):
                fine[i, j] = evaluate(shoulder, elbow)[0]
        fi, fj = np.unravel_index(np.argmin(np.abs(fine)), fine.shape)
        surface_shoulder, surface_elbow = fine_axes[0][fi], fine_axes[1][fj]
        surface_distance, surface_tip = evaluate(surface_shoulder, surface_elbow)
        di, dj = np.unravel_index(np.argmin(coarse), coarse.shape)
        depth_coarse_best = (coarse_axes[0][di], coarse_axes[1][dj])
        depth_axes = [np.linspace(max(axis['range_rad'][0], center - np.radians(1)),
                                  min(axis['range_rad'][1], center + np.radians(1)), 21)
                      for axis, center in zip(axes, depth_coarse_best)]
        depth_grid = np.empty((len(depth_axes[0]), len(depth_axes[1])))
        for i, shoulder in enumerate(depth_axes[0]):
            for j, elbow in enumerate(depth_axes[1]):
                depth_grid[i, j] = evaluate(shoulder, elbow)[0]
        di, dj = np.unravel_index(np.argmin(depth_grid), depth_grid.shape)
        deep_shoulder, deep_elbow = depth_axes[0][di], depth_axes[1][dj]
        deepest, deep_tip = evaluate(deep_shoulder, deep_elbow)
        initial_distance, initial_tip = evaluate(axes[0]['initial_rad'], axes[1]['initial_rad'])
        initial_finger_clearance = finger_clearance(axes[0]['initial_rad'], axes[1]['initial_rad'])
        assert np.isfinite(coarse).all() and np.isfinite(fine).all()
        assert xml_path.read_bytes() == xml_before
        return dict(method='two-stage grid; coarse 1 degree, local refinement 0.1 degree',
                    simulation_steps=0, other_joints='fixed at reset qpos',
                    constraints='joint equalities are not enforced during this FK scan',
                    target_button=int(sim.config['target_button']),
                    target_box_half_size_m=model.geom_size[target_id, :3].tolist(),
                    axes=[dict(name=a['name'], range_rad=a['range_rad'].tolist(),
                               range_deg=np.degrees(a['range_rad']).tolist(),
                               initial_rad=a['initial_rad'],
                               initial_deg=float(np.degrees(a['initial_rad']))) for a in axes],
                    coarse_grid_shape=list(coarse.shape), fine_grid_shape=list(fine.shape),
                    distance_sign='positive outside button box, zero on surface, negative inside',
                    initial=dict(signed_distance_to_surface_m=initial_distance,
                                 signed_distance_to_surface_mm=initial_distance * 1000,
                                 finger_geom_clearance_to_button_m=initial_finger_clearance,
                                 fingertip_world_m=initial_tip.tolist()),
                    nearest_surface=dict(signed_distance_m=surface_distance,
                        absolute_surface_error_mm=abs(surface_distance) * 1000,
                        finger_geom_clearance_to_button_m=finger_clearance(surface_shoulder, surface_elbow),
                        shoulder_rot_rad=float(surface_shoulder),
                        shoulder_rot_deg=float(np.degrees(surface_shoulder)),
                        elbow_flexion_rad=float(surface_elbow),
                        elbow_flexion_deg=float(np.degrees(surface_elbow)),
                        fingertip_world_m=surface_tip.tolist(), grid_resolution_deg=0.1),
                    deepest_grid_point=dict(signed_distance_m=deepest,
                        penetration_depth_mm=max(0.0, -deepest) * 1000,
                        finger_geom_clearance_to_button_m=finger_clearance(deep_shoulder, deep_elbow),
                        shoulder_rot_deg=float(np.degrees(deep_shoulder)),
                        elbow_flexion_deg=float(np.degrees(deep_elbow)),
                        fingertip_world_m=deep_tip.tolist(), grid_resolution_deg=0.1),
                    coarse_nearest_surface_deg=dict(shoulder=float(np.degrees(surface_coarse_best[0])),
                                                   elbow=float(np.degrees(surface_coarse_best[1]))),
                    grids=dict(coarse_shoulder_deg=np.degrees(coarse_axes[0]).tolist(),
                               coarse_elbow_deg=np.degrees(coarse_axes[1]).tolist(),
                               coarse_signed_distance_m=coarse.tolist(),
                               refinement_shoulder_deg=np.degrees(fine_axes[0]).tolist(),
                               refinement_elbow_deg=np.degrees(fine_axes[1]).tolist(),
                               refinement_signed_distance_m=fine.tolist()))
    finally:
        sim.close()


def main():
    base = Path(__file__).resolve().parent
    out = base.parent / 'artifacts/trajectory_check_2026-10-21/shoulder_elbow_reachability.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    report = scan()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shoulder = report['grids']['coarse_shoulder_deg']
    elbow = report['grids']['coarse_elbow_deg']
    distance = np.array(report['grids']['coarse_signed_distance_m']) * 1000
    fig, ax = plt.subplots(figsize=(8, 5.5))
    limit = max(abs(float(np.min(distance))), abs(float(np.max(distance))))
    contour = ax.contourf(elbow, shoulder, distance, levels=25,
                          cmap='coolwarm', extend='both')
    ax.contour(elbow, shoulder, distance, levels=[0], colors='black', linewidths=1.2)
    fig.colorbar(contour, ax=ax, label='Signed distance to button surface (mm)')
    ax.scatter(report['nearest_surface']['elbow_flexion_deg'],
               report['nearest_surface']['shoulder_rot_deg'],
               color='yellow', edgecolor='black', marker='*', s=130, label='Nearest grid to surface')
    ax.scatter(report['axes'][1]['initial_deg'], report['axes'][0]['initial_deg'],
               color='white', edgecolor='black', marker='o', s=55, label='Reset pose')
    ax.set(xlabel='Elbow flexion (deg)', ylabel='Shoulder rotation (deg)',
           title='Static FK scan; black line is fingertip on button surface')
    ax.legend()
    fig.tight_layout()
    figure = out.with_suffix('.png')
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'grids'}, ensure_ascii=False, indent=2))
    print(out)
    print(figure)


if __name__ == '__main__':
    main()
