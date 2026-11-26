"""隔离验证临时按钮滑块的重力平衡和卸载回弹，不加载手臂或正式 XML。"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


DT = 0.002
BUTTON_POS = "0.392445 -0.24 0.843"
BUTTON_QUAT = "0.922997 0 -0.384808 0"
SLIDER_RANGE_M = (0.0, 0.005)
STIFFNESS_N_M = 1000.0
DAMPING_NS_M = 10.0
DIAGNOSTIC_FORCE_N = 1.0
STEPS = (("gravity_settle", 100, 0.0),
         ("diagnostic_force", 50, DIAGNOSTIC_FORCE_N),
         ("release_rebound", 150, 0.0))


def make_model():
    # The geometry and orientation match button-0 in simulation.xml. This model
    # intentionally contains no hand, contact partner, actuator, or task code.
    xml = f'''<mujoco model="isolated_button_slider_probe">
      <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody>
        <body name="button-0" pos="{BUTTON_POS}" quat="{BUTTON_QUAT}">
          <joint name="probe_button_slide" type="slide" axis="0 0 -1"
                 limited="true" range="{SLIDER_RANGE_M[0]} {SLIDER_RANGE_M[1]}"
                 stiffness="{STIFFNESS_N_M}" springref="0" damping="{DAMPING_NS_M}"/>
          <geom name="button-0" type="box" size="0.025 0.025 0.01" density="1000"/>
        </body>
      </worldbody>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "artifacts/trajectory_check_2026-10-22"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint_id = model.joint("probe_button_slide").id
    qadr = int(model.jnt_qposadr[joint_id])
    dof = int(model.jnt_dofadr[joint_id])
    q_ref = float(model.qpos_spring[qadr])
    mass = float(model.body_mass[model.body("button-0").id])

    rows = []
    step_index = 0
    for phase, count, force in STEPS:
        for _ in range(count):
            data.qfrc_applied[dof] = force
            mujoco.mj_step(model, data)
            q = float(data.qpos[qadr])
            qvel = float(data.qvel[dof])
            # qfrc_bias is the generalized force required to balance gravity;
            # the physical gravity contribution is its negative.
            gravity_force = -float(data.qfrc_bias[dof])
            spring_force = -STIFFNESS_N_M * (q - q_ref)
            damping_force = -DAMPING_NS_M * qvel
            rows.append(dict(step=step_index, time_s=float(data.time), phase=phase,
                             slide_position_m=q, slide_velocity_m_s=qvel,
                             slide_acceleration_m_s2=float(data.qacc[dof]),
                             gravity_force_N=gravity_force,
                             spring_force_N=spring_force,
                             damping_force_N=damping_force,
                             diagnostic_force_N=force,
                             lower_limit_near=bool(q <= SLIDER_RANGE_M[0] + 1e-7),
                             upper_limit_near=bool(q >= SLIDER_RANGE_M[1] - 1e-7),
                             contact_count=int(data.ncon)))
            step_index += 1
    data.qfrc_applied[dof] = 0.0

    warnings = [dict(name=mujoco.mjtWarning(i).name,
                     count=int(data.warning[i].number))
                for i in range(mujoco.mjtWarning.mjNWARNING)
                if int(data.warning[i].number) > 0]
    settled = [r for r in rows if r["phase"] == "gravity_settle"][-20:]
    rebound = rows[-20:]
    equilibrium_estimate = (mass * 9.81 * abs(float(data.xmat[model.body("button-0").id].reshape(3, 3)[2, 2]))
                            / STIFFNESS_N_M)
    # The generalized gravity term is the more direct projected-force check.
    initial_gravity = rows[0]["gravity_force_N"]
    report = dict(
        experiment="isolated slider, no hand contact; one diagnostic force pulse",
        mujoco_version=mujoco.__version__, timestep_s=DT,
        button_mass_kg=mass, slider_axis_local=[0, 0, -1],
        slider_range_m=list(SLIDER_RANGE_M), spring_stiffness_N_m=STIFFNESS_N_M,
        damping_Ns_m=DAMPING_NS_M, qpos_spring_reference_m=q_ref,
        phase_steps={name: count for name, count, _ in STEPS},
        diagnostic_force_N=DIAGNOSTIC_FORCE_N,
        estimated_static_offset_from_initial_gravity_m=initial_gravity / STIFFNESS_N_M,
        geometry_based_offset_sanity_check_m=equilibrium_estimate,
        gravity_settle_last_20=dict(
            mean_position_m=float(np.mean([r["slide_position_m"] for r in settled])),
            mean_velocity_m_s=float(np.mean([r["slide_velocity_m_s"] for r in settled])),
            mean_gravity_force_N=float(np.mean([r["gravity_force_N"] for r in settled])),
            mean_spring_force_N=float(np.mean([r["spring_force_N"] for r in settled]))),
        rebound_last_20=dict(
            mean_position_m=float(np.mean([r["slide_position_m"] for r in rebound])),
            mean_velocity_m_s=float(np.mean([r["slide_velocity_m_s"] for r in rebound]))),
        max_position_m=max(r["slide_position_m"] for r in rows),
        max_speed_m_s=max(abs(r["slide_velocity_m_s"]) for r in rows),
        lower_limit_touched=any(r["lower_limit_near"] for r in rows),
        upper_limit_touched=any(r["upper_limit_near"] for r in rows),
        contact_count_max=max(r["contact_count"] for r in rows),
        warnings=warnings,
        samples=rows,
        limitations=("The 1 N pulse is a diagnostic generalized force, not a modeled finger press. "
                     "Spring and damping values are provisional assumptions, not measured hardware parameters."))
    json_path = output_dir / "button_slider_isolated_baseline.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    time = np.array([r["time_s"] for r in rows])
    phase = np.array([r["phase"] for r in rows])
    position_mm = np.array([r["slide_position_m"] for r in rows]) * 1000
    velocity_mm_s = np.array([r["slide_velocity_m_s"] for r in rows]) * 1000
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, position_mm, color="#1464a5", label="slider position")
    axes[0].axhline(SLIDER_RANGE_M[1] * 1000, color="gray", ls="--", label="upper travel limit")
    axes[0].set_ylabel("position (mm)")
    axes[0].legend(loc="best")
    axes[1].plot(time, velocity_mm_s, color="#d17b0f")
    axes[1].set_ylabel("velocity (mm/s)")
    axes[2].plot(time, [r["gravity_force_N"] for r in rows], label="gravity", color="#4c956c")
    axes[2].plot(time, [r["spring_force_N"] for r in rows], label="spring", color="#a23b72")
    axes[2].plot(time, [r["diagnostic_force_N"] for r in rows], label="diagnostic input", color="#666")
    axes[2].set_ylabel("force (N)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="best")
    for ax in axes:
        for boundary in (0.2, 0.3):
            ax.axvline(boundary, color="black", alpha=0.18, lw=1)
        ax.grid(alpha=0.25)
    fig.suptitle("Isolated button slider: gravity, diagnostic pulse, rebound")
    fig.tight_layout()
    image_path = output_dir / "button_slider_isolated_baseline.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"},
                     ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Plot: {image_path}")


if __name__ == "__main__":
    main()
