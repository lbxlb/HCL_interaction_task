"""在多个步长下配对比较默认限位与 solref_limit=2*dt。"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from test_button_slider_limit_timestep import (
    DIAGNOSTIC_FORCE_N,
    PHASE_DURATIONS_S,
    STIFFNESS_N_M,
    make_model,
    simulate,
)
import mujoco


TIMESTEPS_S = (0.002, 0.001, 0.0005)


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "artifacts/trajectory_check_2026-10-22"
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_model = make_model(TIMESTEPS_S[0], 0.0, None)
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_forward(probe_model, probe_data)
    probe_dof = int(probe_model.jnt_dofadr[probe_model.joint("probe_button_slide").id])
    gravity_force = -float(probe_data.qfrc_bias[probe_dof])
    compensated_reference = -gravity_force / STIFFNESS_N_M

    cases = []
    for timestep_s in TIMESTEPS_S:
        short_ms = f"{timestep_s * 1000:g}ms"
        for name, solref_limit in (
            ("default", None),
            ("solref_2dt", f"{2 * timestep_s:.17g} 1"),
        ):
            case_name = f"{short_ms}_{name}"
            result = simulate(case_name, timestep_s, compensated_reference,
                              name, solref_limit)
            for row in result["samples"]:
                row["solver_evaluation_time_s"] = row["time_s"] - timestep_s
            result["refsafe_enabled"] = not bool(
                make_model(timestep_s, compensated_reference, solref_limit)
                .opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
            result["time_constant_to_timestep_ratio"] = (
                None if solref_limit is None else 2.0)
            cases.append(result)

    report = dict(
        experiment="isolated slider; timestep-scaled joint-limit solref comparison",
        mujoco_version=mujoco.__version__, gravity_force_at_q0_N=gravity_force,
        spring_reference_m=compensated_reference,
        slider_range_m=[0.0, 0.005], spring_stiffness_N_m=STIFFNESS_N_M,
        damping_Ns_m=10.0, diagnostic_force_N=DIAGNOSTIC_FORCE_N,
        phase_durations_s={p: d for p, d, _ in PHASE_DURATIONS_S},
        limit_comparison=("For each timestep, compare MuJoCo defaults with requested "
                          "solreflimit=[2*timestep_s, 1]. Verify the compiled value and refsafe flag."),
        sampling_note=("Every integration step is recorded. time_s labels post-step position/velocity; "
                       "solver_evaluation_time_s labels pre-integration acceleration, gravity and "
                       "constraint force. Spring force is computed from post-step position. "
                       "Do not combine these fields as a simultaneous force balance."),
        cases=cases,
        limitations=("The diagnostic generalized force is not a finger model. "
                     "Spring/damping values are provisional; this remains an isolated slider test."))
    json_path = output_dir / "button_slider_refsafe_scaling.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    colors = {
        "2ms_default": "#d17b0f", "2ms_solref_2dt": "#a23b72",
        "1ms_default": "#4c956c", "1ms_solref_2dt": "#1464a5",
        "0.5ms_default": "#777777", "0.5ms_solref_2dt": "#17a2b8",
    }
    labels = {
        "2ms_default": "2 ms, default", "2ms_solref_2dt": "2 ms, solref=4 ms",
        "1ms_default": "1 ms, default", "1ms_solref_2dt": "1 ms, solref=2 ms",
        "0.5ms_default": "0.5 ms, default", "0.5ms_solref_2dt": "0.5 ms, solref=1 ms",
    }
    fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [1.1, 0.8, 1.0]})
    for case in cases:
        rows = case["samples"]
        times = [r["time_s"] for r in rows]
        positions = [r["position_m"] * 1000 for r in rows]
        velocities = [r["velocity_m_s"] * 1000 for r in rows]
        axes[0].plot(times, positions, color=colors[case["case"]],
                     label=labels[case["case"]])
        axes[1].plot(times, positions, color=colors[case["case"]],
                     label=labels[case["case"]])
        axes[2].plot(times, velocities, color=colors[case["case"]],
                     label=labels[case["case"]])
    axes[0].set_ylabel("slider position (mm)")
    axes[1].set_ylabel("position near lower limit (mm)")
    axes[1].set_xlim(0.29, 0.36)
    axes[1].set_ylim(-0.08, 0.025)
    axes[1].axhline(0, color="black", ls="--", lw=1, label="q=0 lower bound")
    axes[2].set_ylabel("slider velocity (mm/s)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncol=2)
    fig.suptitle("Joint-limit response with timestep-scaled solref")
    fig.tight_layout()
    image_path = output_dir / "button_slider_refsafe_scaling.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)

    summary = [{k: v for k, v in case.items() if k != "samples"} for case in cases]
    print(json.dumps(dict(experiment=report["experiment"], cases=summary),
                     ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Plot: {image_path}")


if __name__ == "__main__":
    main()
