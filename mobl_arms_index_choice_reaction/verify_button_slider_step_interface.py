"""Compare interactive stepping and repeated reset against the saved online run."""
import hashlib
import json
from pathlib import Path

from button_slider_simulation import ButtonSliderSimulation


def main():
    root = Path(__file__).resolve().parent.parent
    out = root / 'artifacts/trajectory_check_2026-10-27'
    reference_path = root / 'artifacts/trajectory_check_2026-10-22/button_press_state_online.json'
    paths = [reference_path,
             root / 'mobl_arms_index_choice_reaction/simulation.xml',
             root / 'mobl_arms_index_pointing/simulation.xml']
    originals = [p.read_bytes() for p in paths]
    reference = json.loads(originals[0])
    sim = ButtonSliderSimulation()
    checks = {}
    try:
        sim.step([0, 0])
    except RuntimeError:
        checks['step_before_reset_rejected'] = True
    first = None
    for episode in range(2):
        rows = [sim.reset()]
        for i in range(500):
            rows.append(sim.step(*sim.reference_target(i)))
        checks[f'episode_{episode}_matches_reference'] = rows == reference['samples']
        checks[f'episode_{episode}_no_warning'] = not any(w.number for w in sim.data.warning)
        checks[f'episode_{episode}_no_saturation'] = sim.saturation_count == 0
        if first is None:
            first = rows
        else:
            checks['reset_reproduces_entire_episode'] = rows == first
    prior_time = float(sim.data.time)
    rejected = 0
    for target in ([0], [float('nan'), 0]):
        try:
            sim.step(target)
        except ValueError:
            rejected += 1
    checks['invalid_targets_rejected_without_stepping'] = rejected == 2 and float(sim.data.time) == prior_time
    checks['references_and_xml_unchanged'] = [p.read_bytes() for p in paths] == originals
    result = dict(checks=checks, samples=first,
                  reference_sha256=hashlib.sha256(originals[0]).hexdigest(),
                  limitations='Temporary simulation core; fixed model and deterministic reset. '
                  'No Gymnasium wrapper, reward, termination, randomized reset or learned policy.')
    (out / 'button_slider_step_interface.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(checks, indent=2))
    assert all(checks.values()), 'Step interface regression failed'


if __name__ == '__main__':
    main()
