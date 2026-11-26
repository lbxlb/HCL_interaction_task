"""Run task logic at every physics observation and compare with saved replay."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from button_press_state import ButtonPressState
from test_button_slider_full_baseline import main as run_experiment


def main():
    out = Path(__file__).resolve().parent.parent / 'artifacts/trajectory_check_2026-10-22'
    reference_paths = [out / 'button_slider_press_release.json', out / 'button_press_state_replay.json']
    reference_bytes = [p.read_bytes() for p in reference_paths]
    physics, replay = [json.loads(b) for b in reference_bytes]
    task = ButtonPressState()
    task_rows, events = [], []

    def observe(model, data, finger, button, row):
        # The experiment refreshes contacts on a separate, synchronized MjData.
        # This callback consumes that observation and never writes model/data.
        loaded = any(set(c['geoms']) == {model.geom(finger).name, model.geom(button).name}
                     and c['distance_m'] <= 0 and c['normal_force_N'] > 0
                     for c in row['button_contacts'])
        status = task.update(row['time_s'], row['slider_position_m'], loaded)
        task_row = dict(time_s=row['time_s'], position_m=row['slider_position_m'],
                        contact_loaded=loaded, **status)
        task_rows.append(task_row)
        row['press_task'] = task_row.copy()
        if status['event']:
            events.append(task_row.copy())

    stem = 'button_press_state_online'
    run_experiment(observation_hook=observe, output_stem=stem, press=True, release=True)
    path = out / f'{stem}.json'
    report = json.loads(path.read_text())
    # The baseline's observer_* fields belong to its independent geometry model.
    # All physics fields must still be present and exactly equal, without tolerance.
    physics_keys = set(physics['samples'][0]) - {
        k for k in physics['samples'][0] if k.startswith('observer_')}
    physics_equal = len(report['samples']) == len(physics['samples']) and all(
        all(key in new and new[key] == old[key] for key in physics_keys)
        for old, new in zip(physics['samples'], report['samples']))
    checks = dict(
        all_501_observations_processed=len(task_rows) == len(report['samples']) == 501,
        criteria_match_replay=asdict(task.criteria) == replay['criteria'],
        all_task_states_match_replay=task_rows == replay['samples'],
        events_match_replay=events == replay['events'],
        physics_samples_unchanged=physics_equal,
        one_press_one_release=task.press_count == task.release_count == 1,
        final_ready=task.phase == 'ready',
        references_unchanged=[p.read_bytes() for p in reference_paths] == reference_bytes,
        formal_xml_unchanged=report['formal_xml_unchanged'],
    )
    report['online_task'] = dict(
        criteria=asdict(task.criteria), events=events, checks=checks,
        physics_fields_compared=sorted(physics_keys),
        reference_sha256={p.name: hashlib.sha256(b).hexdigest()
                          for p, b in zip(reference_paths, reference_bytes)},
        limitations='One temporary-model trajectory. Task logic only observes; no feedback, '
                    'formal task replacement, reward/termination wiring or trained policy.',
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(checks=checks, events=events), indent=2))
    assert all(checks.values()), 'Online task checks failed; inspect saved report'


if __name__ == '__main__':
    main()
