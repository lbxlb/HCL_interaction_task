"""比较临时按钮滑块的时间步长与限位响应；不加载手臂、不改正式 XML。"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


BUTTON_POS = "0.392445 -0.24 0.843"
BUTTON_QUAT = "0.922997 0 -0.384808 0"
SLIDER_RANGE_M = (0.0, 0.005)
STIFFNESS_N_M = 1000.0
DAMPING_NS_M = 10.0
DIAGNOSTIC_FORCE_N = 1.0
PHASE_DURATIONS_S = (("settle", 0.2, 0.0),
                     ("diagnostic_force", 0.1, DIAGNOSTIC_FORCE_N),
                     ("release", 0.3, 0.0))
LIMIT_CASES = (("default", None), ("shorter_time_constant", "0.004 1"))


def make_model(timestep_s, spring_reference_m, solref_limit):
    solref_attr = "" if solref_limit is None else f' solreflimit="{solref_limit}"'
    xml = f'''<mujoco model="button_slider_limit_timestep_probe">
      <option timestep="{timestep_s:.17g}" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody>
        <body name="button-0" pos="{BUTTON_POS}" quat="{BUTTON_QUAT}">
          <joint name="probe_button_slide" type="slide" axis="0 0 -1"
                 limited="true" range="{SLIDER_RANGE_M[0]} {SLIDER_RANGE_M[1]}"
                 stiffness="{STIFFNESS_N_M}" springref="{spring_reference_m:.17g}"
                 damping="{DAMPING_NS_M}"{solref_attr}/>
          <geom name="button-0" type="box" size="0.025 0.025 0.01" density="1000"/>
        </body>
      </worldbody>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def simulate(case_name, timestep_s, spring_reference_m, solref_name, solref_limit):
    model = make_model(timestep_s, spring_reference_m, solref_limit)
    data = mujoco.MjData(model)
    joint_id = model.joint("probe_button_slide").id
    qadr = int(model.jnt_qposadr[joint_id])
    dof = int(model.jnt_dofadr[joint_id])
    rows = []
    step_index = 0
    for phase, duration_s, force in PHASE_DURATIONS_S:
        count = int(round(duration_s / timestep_s))
        for _ in range(count):
            data.qfrc_applied[dof] = force
            mujoco.mj_step(model, data)
            q = float(data.qpos[qadr])
            rows.append(dict(step=step_index, time_s=float(data.time), phase=phase,
                             position_m=q, velocity_m_s=float(data.qvel[dof]),
                             acceleration_m_s2=float(data.qacc[dof]),
                             gravity_force_N=-float(data.qfrc_bias[dof]),
                             spring_force_N=-STIFFNESS_N_M *
                             (q - float(model.qpos_spring[qadr])),
                             constraint_force_N=float(data.qfrc_constraint[dof]),
                             diagnostic_force_N=force,
                             lower_bound_near=bool(q <= SLIDER_RANGE_M[0] + 1e-7),
                             upper_bound_near=bool(q >= SLIDER_RANGE_M[1] - 1e-7),
                             contact_count=int(data.ncon)))
            step_index += 1
    warnings = [dict(name=mujoco.mjtWarning(i).name,
                     count=int(data.warning[i].number))
                for i in range(mujoco.mjtWarning.mjNWARNING)
                if int(data.warning[i].number) > 0]
    settle = [r for r in rows if r["phase"] == "settle"][-int(round(0.04 / timestep_s)):]
    release = rows[-int(round(0.04 / timestep_s)):]
    return dict(case=case_name, timestep_s=timestep_s, solref_name=solref_name,
                requested_solreflimit=solref_limit,
                compiled_solreflimit=model.jnt_solref[joint_id].tolist(),
                spring_reference_m=float(model.qpos_spring[qadr]),
                phase_steps={p: int(round(d / timestep_s)) for p, d, _ in PHASE_DURATIONS_S},
                diagnostic_force_N=DIAGNOSTIC_FORCE_N, samples=rows,
                settle_last_40ms=dict(
                    mean_position_m=float(np.mean([r["position_m"] for r in settle])),
                    mean_velocity_m_s=float(np.mean([r["velocity_m_s"] for r in settle]))),
                release_last_40ms=dict(
                    mean_position_m=float(np.mean([r["position_m"] for r in release])),
                    mean_velocity_m_s=float(np.mean([r["velocity_m_s"] for r in release]))),
                min_position_m=min(r["position_m"] for r in rows),
                max_position_m=max(r["position_m"] for r in rows),
                max_lower_limit_violation_m=max(
                    0.0, SLIDER_RANGE_M[0] - min(r["position_m"] for r in rows)),
                max_speed_m_s=max(abs(r["velocity_m_s"]) for r in rows),
                max_abs_constraint_force_N=max(abs(r["constraint_force_N"]) for r in rows),
                lower_bound_near=any(r["lower_bound_near"] for r in rows),
                upper_bound_near=any(r["upper_bound_near"] for r in rows),
                contact_count_max=max(r["contact_count"] for r in rows),
                warnings=warnings)


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "artifacts/trajectory_check_2026-10-22"
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_model = make_model(0.002, 0.0, None)
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_forward(probe_model, probe_data)
    probe_dof = int(probe_model.jnt_dofadr[probe_model.joint("probe_button_slide").id])
    gravity_force = -float(probe_data.qfrc_bias[probe_dof])
    compensated_reference = -gravity_force / STIFFNESS_N_M

    cases = []
    for timestep_s in (0.002, 0.001):
        for name, solref_limit in LIMIT_CASES:
            case_name = f"dt_{int(timestep_s * 1000)}ms_{name}"
            cases.append(simulate(case_name, timestep_s, compensated_reference,
                                  name, solref_limit))
    report = dict(
        experiment="isolated gravity-compensated slider; timestep sensitivity",
        mujoco_version=mujoco.__version__, gravity_force_at_q0_N=gravity_force,
        spring_reference_m=compensated_reference, slider_range_m=list(SLIDER_RANGE_M),
        spring_stiffness_N_m=STIFFNESS_N_M, damping_Ns_m=DAMPING_NS_M,
        phase_durations_s={p: d for p, d, _ in PHASE_DURATIONS_S},
        limit_case_definitions=[dict(name=n, solreflimit=r) for n, r in LIMIT_CASES],
        cases=cases,
        limitations=("The 1 N input is a diagnostic generalized force, not a finger model. "
                     "Spring/damping values are provisional; this remains an isolated slider test."))
    json_path = output_dir / "button_slider_limit_timestep.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    colors = {"dt_2ms_default": "#d17b0f", "dt_2ms_shorter_time_constant": "#a23b72",
              "dt_1ms_default": "#4c956c", "dt_1ms_shorter_time_constant": "#1464a5"}
    labels = {"dt_2ms_default": "2 ms, default limit", "dt_2ms_shorter_time_constant": "2 ms, solref 0.004",
              "dt_1ms_default": "1 ms, default limit", "dt_1ms_shorter_time_constant": "1 ms, solref 0.004"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1.1, 0.8, 1.0]})
    for case in cases:
        rows = case["samples"]
        times = [r["time_s"] for r in rows]
        positions = [r["position_m"] * 1000 for r in rows]
        velocities = [r["velocity_m_s"] * 1000 for r in rows]
        axes[0].plot(times, positions, color=colors[case["case"]], label=labels[case["case"]])
        axes[1].plot(times, positions, color=colors[case["case"]], label=labels[case["case"]])
        axes[2].plot(times, velocities, color=colors[case["case"]], label=labels[case["case"]])
    axes[0].set_ylabel("slider position (mm)")
    axes[1].set_ylabel("position near lower limit (mm)")
    axes[1].set_xlim(0.29, 0.36)
    axes[1].set_ylim(-0.08, 0.025)
    axes[1].axhline(0.0, color="black", ls="--", lw=1, label="q=0 lower bound")
    axes[2].set_ylabel("slider velocity (mm/s)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Timestep sensitivity of the compensated button slider limit")
    fig.tight_layout()
    image_path = output_dir / "button_slider_limit_timestep.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    print(json.dumps({k: [{x: y for x, y in c.items() if x != "samples"}
                          for c in v] if k == "cases" else v
                      for k, v in report.items()}, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Plot: {image_path}")


if __name__ == "__main__":
    main()
