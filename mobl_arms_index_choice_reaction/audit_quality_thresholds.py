"""Audit candidate quality thresholds without declaring them safety limits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'artifacts/ppo_multiseed_2026-11-08/ppo_multiseed_report.json'
out = ROOT / 'artifacts/ppo_multiseed_2026-11-08/quality_threshold_sensitivity.json'
d = json.loads(src.read_text())
force_thresholds = (0.25, 0.30, 0.40, 0.50, 0.75, 1.0)
sat_thresholds = (0, 5, 10, 15)
result = {'purpose': 'Sensitivity audit only; thresholds are hypothetical analysis values, not hardware limits.', 'arms': {}}
for seed, arm in d['arms'].items():
    rows = arm['evaluation']
    result['arms'][seed] = {
        'episodes': len(rows),
        'force_thresholds_N': {str(t): sum(r['peak_force_N'] > t for r in rows) for t in force_thresholds},
        'saturation_thresholds_channels': {str(t): sum(r['saturation_channels'] > t for r in rows) for t in sat_thresholds},
        'observed_peak_force_range_N': [min(r['peak_force_N'] for r in rows), max(r['peak_force_N'] for r in rows)],
        'observed_saturation_range_channels': [min(r['saturation_channels'] for r in rows), max(r['saturation_channels'] for r in rows)],
    }
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
