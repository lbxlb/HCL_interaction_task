"""Measure valid versus initially-contacting reset states."""
import json
from pathlib import Path
from button_slider_env import ButtonSliderEnv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'ppo_multiseed_2026-11-08' / 'initial_state_sampling.json'

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {'purpose': 'Reset validity protocol; invalid contact starts are excluded, not task failures.', 'ranges': {}}
    for jitter in (0.2, 0.5, 1.0):
        env = ButtonSliderEnv(max_episode_steps=150, initial_pose_jitter_deg=jitter,
                              resample_invalid_reset=False)
        valid, invalid = [], []
        try:
            for seed in range(5000, 5100):
                try:
                    _, info = env.reset(seed=seed)
                except RuntimeError as exc:
                    invalid.append({'seed': seed, 'reason': str(exc)})
                else:
                    valid.append({'seed': seed, 'initial_pose_offset_rad': info['initial_pose_offset_rad']})
        finally:
            env.close()
        result['ranges'][str(jitter)] = {
            'requested': 100, 'valid': len(valid), 'invalid_initial_contact': len(invalid),
            'valid_seeds': valid, 'invalid': invalid,
        }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: {'valid': v['valid'], 'invalid_initial_contact': v['invalid_initial_contact']}
                      for k, v in result['ranges'].items()}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
