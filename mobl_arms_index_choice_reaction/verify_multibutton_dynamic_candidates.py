"""Run short press/release dynamics for the four candidate button models."""
import json
from pathlib import Path
import numpy as np
from button_slider_simulation import ButtonSliderSimulation, SimulationFault

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/trajectory_check_2026-11-08/multibutton_dynamic_candidates.json'

def run(button):
    sim = ButtonSliderSimulation(target_button=button)
    sim.reset(); rows=[]; events=[]
    try:
        for phase in ('press', 'hold', 'release', 'outside'):
            target = sim.q_press if phase in ('press','hold') else sim.q_out
            count = 100 if phase in ('press','release') else 200
            for _ in range(count):
                row = sim.step(target, np.zeros(2)); rows.append(row)
                if row['press_task']['event']:
                    events.append({'event': row['press_task']['event'], 'time_s': row['time_s']})
    except SimulationFault as exc:
        return {'button': button, 'status': 'simulation_fault', 'reason': exc.reason, 'details': exc.details}
    finally:
        pass
    return {'button': button, 'status': 'completed', 'events': events,
            'press_count': rows[-1]['press_task']['press_count'],
            'release_count': rows[-1]['press_task']['release_count'],
            'max_slider_position_m': max(r['slider_position_m'] for r in rows),
            'min_slider_position_m': min(r['slider_position_m'] for r in rows),
            'peak_normal_force_N': max((sum(c['normal_force_N'] for c in r['button_contacts']) for r in rows), default=0.0),
            'final_slider_position_m': rows[-1]['slider_position_m'],
            'final_slider_velocity_m_s': rows[-1]['slider_velocity_m_s'],
            'saturation_count': sim.saturation_count}

def main():
    result = {'purpose': 'Temporary per-button dynamics check; not a final four-button task or hardware validation.',
              'buttons': [run(i) for i in range(4)],
              'limitations': 'Uses candidate-derived temporary sliders and two-joint PD control while other joints remain fixed.'}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
