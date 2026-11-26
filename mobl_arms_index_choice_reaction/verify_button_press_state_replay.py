"""Replay saved physics observations through independent task logic."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from button_press_state import ButtonPressState


def main():
    base = Path(__file__).resolve().parent
    out = base.parent / 'artifacts/trajectory_check_2026-10-22'
    source = out / 'button_slider_press_release.json'
    original = source.read_bytes()
    report = json.loads(original)
    state = ButtonPressState()
    rows, events = [], []
    for sample in report['samples']:
        loaded = any(set(c['geoms']) == {report['target_button'], 'hand_2distph'}
                     and c['distance_m'] <= 0 and c['normal_force_N'] > 0
                     for c in sample['button_contacts'])
        status = state.update(sample['time_s'], sample['slider_position_m'], loaded)
        row = dict(time_s=sample['time_s'], position_m=sample['slider_position_m'],
                   contact_loaded=loaded, **status)
        rows.append(row)
        if status['event']:
            events.append(row.copy())
    checks = dict(one_press=state.press_count == 1, one_release=state.release_count == 1,
                  events_in_order=[e['event'] for e in events] == ['pressed', 'released'],
                  final_ready=state.phase == 'ready', source_unchanged=source.read_bytes() == original)
    result = dict(criteria=asdict(state.criteria), checks=checks, events=events, samples=rows,
                  source=source.name, source_sha256=hashlib.sha256(original).hexdigest(),
                  limitations='Offline replay of one temporary-model trajectory; demonstration thresholds. '
                  'No formal task integration, trained policy or hardware validation.')
    (out / 'button_press_state_replay.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([r['time_s'] for r in rows], [r['position_m'] * 1000 for r in rows], label='Button travel')
    ax.axhline(state.criteria.press_position_m * 1000, linestyle='--', label='Press threshold')
    ax.axhline(state.criteria.release_position_m * 1000, linestyle=':', label='Release threshold')
    for e in events:
        ax.axvline(e['time_s'], color='gray', linestyle=':')
        ax.annotate(f"{e['event']} {e['time_s']:.3f}s", (e['time_s'], e['position_m'] * 1000),
                    xytext=(20, -35) if e['event'] == 'pressed' else (8, 25),
                    textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    ax.set(xlabel='Simulation time (s)', ylabel='Button travel (mm)', title='Offline task events: 20ms persistence and hysteresis')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'button_press_state_replay.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(checks=checks, events=events), indent=2))
    assert all(checks.values()), 'Task replay failed; inspect saved report'


if __name__ == '__main__':
    main()
