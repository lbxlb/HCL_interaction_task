"""双轴实验的只读末端测量：独立数据中计算目标姿态的正向运动学。"""
import mujoco
import numpy as np


class FingertipMeasurement:
    def __init__(self, model, initial, elbow_offsets, shoulder_offsets, reference_at, segment_seconds=1.0):
        self.reference = mujoco.MjData(model)
        self.initial = initial.copy()
        self.tip = model.site('fingertip').id
        self.elbow = int(model.joint('elbow_flexion').qposadr[0])
        self.shoulder = int(model.joint('shoulder_rot').qposadr[0])
        self.elbow_offsets = elbow_offsets
        self.shoulder_offsets = shoulder_offsets
        self.reference_at = reference_at
        self.segment_seconds = segment_seconds
        self.samples = []

    def observe(self, step, model, actual):
        # 其他关节固定在初态，避免参考姿态跟随实际漂移而掩盖误差。
        self.reference.qpos[:] = self.initial
        for adr, offsets in ((self.elbow, self.elbow_offsets),
                             (self.shoulder, self.shoulder_offsets)):
            self.reference.qpos[adr] = self.reference_at(
                actual.time, self.initial[adr], offsets, self.segment_seconds)[0]
        # 只更新运动学，不积分、不把参考数据写回实际控制状态。
        mujoco.mj_kinematics(model, self.reference)
        desired = self.reference.site_xpos[self.tip].copy()
        position = actual.site_xpos[self.tip].copy()
        error = desired - position
        assert np.isfinite([desired, position, error]).all()
        if step == 0:
            assert np.allclose(error, 0, atol=1e-12, rtol=0)
        self.samples.append(dict(time_s=float(actual.time),
                                 desired_world_m=desired.tolist(),
                                 actual_world_m=position.tolist(),
                                 error_world_m=error.tolist(),
                                 error_norm_m=float(np.linalg.norm(error))))

    def report(self):
        norms = np.array([s['error_norm_m'] for s in self.samples])
        peak = int(np.argmax(norms))
        segments = []
        for index in range(len(self.elbow_offsets)):
            tail = [s['error_norm_m'] for s in self.samples
                    if (index * self.segment_seconds + self.segment_seconds - 0.2 - 1e-10
                            <= s['time_s'] <= (index + 1) * self.segment_seconds + 1e-10)]
            segments.append(dict(segment=index + 1, max_error_last_0_2_seconds_m=max(tail)))
        return dict(site='fingertip', coordinate_frame='world',
                    reference_definition='initial qpos with shoulder_rot and elbow_flexion replaced by their targets; all other joints fixed at initial positions',
                    error_definition='desired minus actual; Euclidean norm',
                    segment_seconds=self.segment_seconds,
                    max_error_m=float(norms[peak]), peak_time_s=self.samples[peak]['time_s'],
                    rms_error_m=float(np.sqrt(np.mean(norms ** 2))),
                    final_error_m=float(norms[-1]), segments=segments, samples=self.samples)


def plot(report, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for run in report['runs']:
        moving = any(run['shoulder_target_sequence_offsets_rad'])
        label = 'Both joints move' if moving else 'Shoulder holds'
        samples = run['fingertip_tracking']['samples']
        t = [s['time_s'] for s in samples]
        errors = np.array([s['error_world_m'] for s in samples]) * 1000
        axes[0, 0].plot(t, [s['error_norm_m'] * 1000 for s in samples], label=label)
        for index, ax in enumerate(list(axes.flat)[1:]):
            ax.plot(t, errors[:, index], label=label)
    for ax, title in zip(axes.flat, ('Position error norm', 'World X error',
                                     'World Y error', 'World Z error')):
        ax.set_title(title)
        ax.set_ylabel('Error (mm)')
        ax.set_xlabel('Time (s)')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle('Fingertip tracking; reference keeps other joints at initial positions')
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
