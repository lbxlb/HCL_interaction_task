"""静态测量重置姿态下指尖与面板按钮的几何距离。"""
import contextlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from simulator import IndexSimulator


def point_to_box_surface_distance(point, model, data, geom_id):
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    local = rotation.T @ (point - data.geom_xpos[geom_id])
    outside = np.maximum(np.abs(local) - model.geom_size[geom_id, :3], 0.0)
    return float(np.linalg.norm(outside))


def measure():
    base = Path(__file__).resolve().parent
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', base / 'simulation.xml')
        sim.reset()
    try:
        model, data = sim.model, sim.data
        mujoco.mj_forward(model, data)
        finger = model.geom('hand_2distph').id
        tip = data.site_xpos[model.site('fingertip').id].copy()
        buttons = []
        for index in range(4):
            geom_id = model.geom(f'button-{index}').id
            center = data.geom_xpos[geom_id].copy()
            buttons.append(dict(button=index, is_configured_target=index == sim.config['target_button'],
                geom_surface_distance_m=float(mujoco.mj_geomDistance(
                    model, data, finger, geom_id, 10.0, None)),
                fingertip_to_geom_center_m=float(np.linalg.norm(tip - center)),
                fingertip_to_button_surface_m=point_to_box_surface_distance(tip, model, data, geom_id),
                button_geom_size_m=model.geom_size[geom_id, :3].tolist(),
                fingertip_world_m=tip.tolist(), button_center_world_m=center.tolist()))
        return dict(measurement='reset_pose_static_geometry_only', sim_steps=0,
                    timestep_seconds=float(model.opt.timestep), target_button=sim.config['target_button'],
                    fingertip_site='fingertip', finger_collision_geom='hand_2distph',
                    button_shape='oriented boxes; reported point-to-surface distance is unsigned',
                    buttons=buttons)
    finally:
        sim.close()


def main():
    base = Path(__file__).resolve().parent
    output = base.parent / 'artifacts/trajectory_check_2026-10-21/initial_panel_distances.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = measure()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    buttons = report['buttons']
    x = np.arange(len(buttons))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, [b['geom_surface_distance_m'] * 1000 for b in buttons],
           width, label='Finger geom to button surface')
    ax.bar(x + width / 2, [b['fingertip_to_button_surface_m'] * 1000 for b in buttons],
           width, label='Fingertip point to button surface')
    ax.set_xticks(x, [f"Button {b['button']}\n(target)" if b['is_configured_target']
                     else f"Button {b['button']}" for b in buttons])
    ax.set_ylabel('Unsigned separation (mm)')
    ax.set_title('Reset pose: static distances, no simulation steps')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figure = output.with_suffix('.png')
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(output)
    print(figure)


if __name__ == '__main__':
    main()
