"""对比临时按钮滑块的原始弹簧零位与重力补偿零位。"""
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
PHASES = (("settle", 100, 0.0),
          ("diagnostic_force", 50, DIAGNOSTIC_FORCE_N),
          ("release", 150, 0.0))
LIMIT_CASES = (
    ("default", None, None),
    ("shorter_time_constant", "0.004 1", None),
    ("higher_impedance", None, "0.9999 0.9999 0.001 0.5 2"),
    ("shorter_time_constant_high_impedance", "0.004 1",
     "0.9999 0.9999 0.001 0.5 2"),
)


def make_model(spring_reference_m, solref_limit=None, solimp_limit=None):
    limit_attrs = ""
    if solref_limit is not None:
        limit_attrs += f' solreflimit="{solref_limit}"'
    if solimp_limit is not None:
        limit_attrs += f' solimplimit="{solimp_limit}"'
    xml = f'''<mujoco model="button_slider_gravity_compensation_probe">
      <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody>
        <body name="button-0" pos="{BUTTON_POS}" quat="{BUTTON_QUAT}">
          <joint name="probe_button_slide" type="slide" axis="0 0 -1"
                 limited="true" range="{SLIDER_RANGE_M[0]} {SLIDER_RANGE_M[1]}"
                 stiffness="{STIFFNESS_N_M}" springref="{spring_reference_m:.17g}"
                 damping="{DAMPING_NS_M}"{limit_attrs}/>
          <geom name="button-0" type="box" size="0.025 0.025 0.01" density="1000"/>
        </body>
      </worldbody>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def simulate(case_name, spring_reference_m, solref_limit=None, solimp_limit=None):
    model = make_model(spring_reference_m, solref_limit, solimp_limit)
    data = mujoco.MjData(model)
    joint_id = model.joint("probe_button_slide").id
    qadr = int(model.jnt_qposadr[joint_id])
    dof = int(model.jnt_dofadr[joint_id])
    mujoco.mj_forward(model, data)
    initial_gravity_force = -float(data.qfrc_bias[dof])
    mass = float(model.body_mass[model.body("button-0").id])
    rows = []
    index = 0
    for phase, count, force in PHASES:
        for _ in range(count):
            data.qfrc_applied[dof] = force
            mujoco.mj_step(model, data)
            q = float(data.qpos[qadr])
            qvel = float(data.qvel[dof])
            rows.append(dict(case=case_name, step=index, time_s=float(data.time), phase=phase,
                             position_m=q, velocity_m_s=qvel,
                             acceleration_m_s2=float(data.qacc[dof]),
                             acceleration_evaluation_time_s=float(data.time - DT),
                             gravity_force_N=-float(data.qfrc_bias[dof]),
                             spring_force_N=-STIFFNESS_N_M *
                             (q - float(model.qpos_spring[qadr])),
                             damping_force_N=-DAMPING_NS_M * qvel,
                             generalized_constraint_force_N=float(data.qfrc_constraint[dof]),
                             diagnostic_force_N=force,
                             lower_bound_near=bool(q <= SLIDER_RANGE_M[0] + 1e-7),
                             upper_bound_near=bool(q >= SLIDER_RANGE_M[1] - 1e-7),
                             contact_count=int(data.ncon)))
            index += 1
    warnings = [dict(name=mujoco.mjtWarning(i).name,
                     count=int(data.warning[i].number))
                for i in range(mujoco.mjtWarning.mjNWARNING)
                if int(data.warning[i].number) > 0]
    settle = [r for r in rows if r["phase"] == "settle"][-20:]
    release = rows[-20:]
    return dict(case=case_name, spring_reference_m=float(model.qpos_spring[qadr]),
                requested_solreflimit=solref_limit, requested_solimplimit=solimp_limit,
                compiled_solreflimit=model.jnt_solref[joint_id].tolist(),
                compiled_solimplimit=model.jnt_solimp[joint_id].tolist(),
                initial_gravity_force_N=initial_gravity_force,
                button_mass_kg=mass, phase_steps={p: n for p, n, _ in PHASES},
                diagnostic_force_N=DIAGNOSTIC_FORCE_N, samples=rows,
                settle_last_20=dict(
                    mean_position_m=float(np.mean([r["position_m"] for r in settle])),
                    mean_velocity_m_s=float(np.mean([r["velocity_m_s"] for r in settle]))),
                release_last_20=dict(
                    mean_position_m=float(np.mean([r["position_m"] for r in release])),
                    mean_velocity_m_s=float(np.mean([r["velocity_m_s"] for r in release]))),
                max_position_m=max(r["position_m"] for r in rows),
                min_position_m=min(r["position_m"] for r in rows),
                max_lower_limit_violation_m=max(
                    0.0, SLIDER_RANGE_M[0] - min(r["position_m"] for r in rows)),
                max_speed_m_s=max(abs(r["velocity_m_s"]) for r in rows),
                lower_bound_near=any(r["lower_bound_near"] for r in rows),
                upper_bound_near=any(r["upper_bound_near"] for r in rows),
                contact_count_max=max(r["contact_count"] for r in rows),
                warnings=warnings)


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "artifacts/trajectory_check_2026-10-22"
    output_dir.mkdir(parents=True, exist_ok=True)

    # At q=0 with springref=0, the spring contributes no force. Measure the
    # gravity generalized force there, then preload the spring oppositely so
    # gravity + spring force is zero at the intended unpressed coordinate.
    probe_model = make_model(0.0)
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_forward(probe_model, probe_data)
    probe_dof = int(probe_model.jnt_dofadr[probe_model.joint("probe_button_slide").id])
    gravity_force = -float(probe_data.qfrc_bias[probe_dof])
    compensated_reference = -gravity_force / STIFFNESS_N_M

    original = simulate("original_reference_q0", 0.0)
    compensated_cases = [
        simulate(f"gravity_compensated_{name}", compensated_reference, solref, solimp)
        for name, solref, solimp in LIMIT_CASES
    ]
    report = dict(
        experiment="isolated slider comparison; no hand contact",
        mujoco_version=mujoco.__version__, timestep_s=DT,
        slider_range_m=list(SLIDER_RANGE_M), spring_stiffness_N_m=STIFFNESS_N_M,
        damping_Ns_m=DAMPING_NS_M, measured_gravity_force_at_q0_N=gravity_force,
        compensation_rule="springref = -gravity_generalized_force / stiffness",
        limit_case_definitions=[dict(name=n, solreflimit=r, solimplimit=i)
                                for n, r, i in LIMIT_CASES],
        sampling_note="Position/velocity are post-step; acceleration is the step's pre-integration value. Lower-bound proximity includes resting at q=0; violation is reported separately.",
        cases=[original, *compensated_cases],
        limitations=("The 1 N input is a diagnostic generalized force, not a finger model. "
                     "Spring stiffness/damping are provisional, and this tests only the isolated slider."))
    json_path = output_dir / "button_slider_gravity_compensation.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False,
                             gridspec_kw={"height_ratios": [1.1, 0.8, 1.0]})
    colors = {"original_reference_q0": "#999999",
              "gravity_compensated_default": "#d17b0f",
              "gravity_compensated_shorter_time_constant": "#1464a5",
              "gravity_compensated_higher_impedance": "#4c956c",
              "gravity_compensated_shorter_time_constant_high_impedance": "#a23b72"}
    labels = {"original_reference_q0": "original reference, q=0",
              "gravity_compensated_default": "compensated: default limit",
              "gravity_compensated_shorter_time_constant": "compensated: shorter limit time constant",
              "gravity_compensated_higher_impedance": "compensated: higher limit impedance",
              "gravity_compensated_shorter_time_constant_high_impedance": "compensated: both changes"}
    for case in report["cases"]:
        rows = case["samples"]
        times = [r["time_s"] for r in rows]
        positions = [r["position_m"] * 1000 for r in rows]
        velocities = [r["velocity_m_s"] * 1000 for r in rows]
        axes[0].plot(times, positions, color=colors[case["case"]], label=labels[case["case"]])
        axes[2].plot(times, velocities, color=colors[case["case"]], label=labels[case["case"]])
        if case["case"] != "original_reference_q0":
            axes[1].plot(times, positions, color=colors[case["case"]],
                         label=labels[case["case"]])
    axes[0].set_ylabel("slider position (mm)")
    axes[1].set_ylabel("position near lower limit (mm)")
    axes[1].set_xlim(0.29, 0.36)
    axes[1].set_ylim(-0.08, 0.025)
    axes[1].axhline(0.0, color="black", ls="--", lw=1, label="q=0 lower bound")
    axes[2].set_ylabel("slider velocity (mm/s)")
    axes[2].set_xlabel("time (s)")
    for ax in (axes[0], axes[2]):
        for boundary in (0.2, 0.3):
            ax.axvline(boundary, color="black", alpha=0.18, lw=1)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Gravity compensation and joint-limit response")
    fig.tight_layout()
    image_path = output_dir / "button_slider_gravity_compensation.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print(json.dumps({k: [{x: y for x, y in c.items() if x != "samples"}
                          for c in v] if k == "cases" else v
                      for k, v in report.items()}, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Plot: {image_path}")


if __name__ == "__main__":
    main()
