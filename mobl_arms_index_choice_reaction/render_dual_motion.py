"""重放实际积分所得 qpos 快照，生成固定镜头双视图 GIF。"""
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image


def render(model, snapshots, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    data.eq_active[model.equality('fixture_shoulder_rot').id] = False
    data.qpos[:] = snapshots[0][1]
    mujoco.mj_forward(model, data)
    finger = data.geom_xpos[model.geom('hand_2distph').id].copy()
    cameras = []
    for lookat, distance in (([0.25, -0.1, 0.85], 1.35), (finger, 0.65)):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = 135
        camera.elevation = -15
        cameras.append(camera)
    frames = []
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        for t, qpos in snapshots:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            views = []
            for index, camera in enumerate(cameras):
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                cv2.rectangle(frame, (0, 0), (480, 32), (0, 0, 0), -1)
                cv2.putText(frame, f'ACTUAL | {"overview" if index == 0 else "arm detail"} | t={t:.2f}s',
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                views.append(frame)
            frames.append(Image.fromarray(np.concatenate(views, axis=1)))
    path = output_dir / 'shoulder_elbow_dual_motion.gif'
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    for index in (0, 8, 28, 48, 80):
        frames[index].save(output_dir / f'dual_motion_frame_{index:03d}.png')
    metadata = dict(description='Replay of actual integrated qpos, no target-pose substitution',
                    frames=len(frames), sample_interval_seconds=0.05, frame_duration_ms=50,
                    first_time_seconds=snapshots[0][0], last_time_seconds=snapshots[-1][0],
                    size_pixels=[960, 360], overview_camera_fixed=True, detail_camera_fixed=True)
    (output_dir / 'dual_motion_render.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(path)
