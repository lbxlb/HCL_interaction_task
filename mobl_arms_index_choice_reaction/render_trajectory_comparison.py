"""离屏保存实际/参考姿态对照；参考图不代表动力学控制成功。"""
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import contextlib
import io
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from simulator import IndexSimulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, nargs='+', default=[396, 424])
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'artifacts' / 'trajectory_comparison')
    args = parser.parse_args()
    if min(args.steps) <= 0:
        parser.error('--steps 必须均为正整数')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent
    with contextlib.redirect_stdout(io.StringIO()):
        sim = IndexSimulator(base / 'config.yaml', base / 'simulation.xml')
        sim.reset()
    reference = mujoco.MjData(sim.model)
    button = sim.model.geom('button-0').id
    finger = sim.model.geom('hand_2distph').id
    records = []
    renderer = None
    try:
        # 与任务重置时一致，使屏幕颜色对应目标；不更新任务或重置轨迹。
        target = sim.model.geom(f"button-{sim.config['target_button']}").id
        sim.model.geom('screen').rgba[:] = sim.model.geom_rgba[target]
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        for step in range(1, max(args.steps) + 1):
            with contextlib.redirect_stdout(io.StringIO()):
                sim.step()
            if step not in args.steps:
                continue
            reference.qpos[:] = sim.reference_qpos
            mujoco.mj_forward(sim.model, reference)
            views = []
            record = {'step': step, 'time_seconds': float(sim.data.time), 'states': {}}
            for label, data in [('ACTUAL (dynamics)', sim.data), ('REFERENCE (kinematics)', reference)]:
                images = []
                for detail in [False, True]:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = sim.data.geom_xpos[button] if detail else [0.25, -0.1, 0.85]
                    camera.distance = 0.32 if detail else 1.35
                    camera.azimuth = 135
                    camera.elevation = -15
                    renderer.update_scene(data, camera=camera)
                    frame = renderer.render().copy()
                    cv2.rectangle(frame, (0, 0), (640, 34), (0, 0, 0), -1)
                    cv2.putText(frame, f'{label} | {"target close-up" if detail else "overview"}',
                                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
                    images.append(frame)
                views.append(np.concatenate(images, axis=1))
                contacts = [dict(geom1=sim.model.geom(int(c.geom1)).name,
                                 geom2=sim.model.geom(int(c.geom2)).name,
                                 distance_m=float(c.dist))
                            for c in data.contact if finger in (c.geom1, c.geom2)]
                record['states'][label] = {
                    'finger_button0_distance_m': float(mujoco.mj_geomDistance(
                        sim.model, data, finger, button, 10, None)),
                    'finger_contacts': contacts,
                }
            path = args.output_dir / f'step_{step:04d}.png'
            if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(views, axis=0), cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f'图片保存失败：{path}')
            record['image'] = path.name
            records.append(record)
        (args.output_dir / 'comparison.json').write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(records, ensure_ascii=False, indent=2))
    finally:
        if renderer is not None:
            renderer.close()
        sim.close()


if __name__ == '__main__':
    main()
