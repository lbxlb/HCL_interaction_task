"""Render the single-button event trace from the archived physics samples."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/trajectory_check_2026-10-22/button_press_state_online.json"
FIGURE = ROOT / "docs/figures/press-cycle.png"


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = report["samples"]
    times = [row["time_s"] for row in rows]
    travel = [row["slider_position_m"] * 1000 for row in rows]
    force = [
        sum(contact["normal_force_N"] for contact in row["button_contacts"]
            if set(contact["geoms"]) == {"hand_2distph", report["target_button"]})
        for row in rows
    ]
    fig, (position, load) = plt.subplots(2, 1, figsize=(8.4, 4.6), sharex=True,
                                         layout="constrained")
    position.plot(times, travel, color="#17656c", linewidth=2)
    load.plot(times, force, color="#80532c", linewidth=2)
    position.set_ylabel("Slider travel (mm)")
    load.set_ylabel("Normal force (N)")
    load.set_xlabel("Simulation time (s)")
    position.set_title("Single-button press and release", loc="left", fontweight="bold", pad=14)
    criteria = report["online_task"]["criteria"]
    for event in report["online_task"]["events"]:
        when = event["time_s"]
        for axis in (position, load):
            axis.axvline(when, color="#687580", linestyle=":", linewidth=1)
        label = event["event"]
        position.text(when + .009, .97, f"{label.capitalize()}\n{when:.3f} s",
                      transform=position.get_xaxis_transform(), va="top", fontsize=9,
                      bbox=dict(facecolor="white", alpha=.88, edgecolor="none", pad=1.8))
    for key, color in (("press", "#486477"), ("release", "#857666")):
        value = criteria[f"{key}_position_m"] * 1000
        position.axhline(value, color=color, linestyle="--", linewidth=1,
                         label=f"{key.capitalize()} threshold ({value:g} mm)")
    position.legend(frameon=False, loc="upper right", fontsize=8)
    for axis in (position, load):
        axis.set_xlim(0, times[-1])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.16)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
