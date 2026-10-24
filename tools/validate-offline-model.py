"""Evaluate recovered weights against retained references, without inherited claims."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from graphwave.evaluate import evaluate_case
from offline.domain import model_info, digest

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--predictions', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
dataset = json.loads(args.dataset.read_text())
rows = []
for case in dataset['cases']:
    if case['split'] != 'test':
        continue
    row = evaluate_case(case, args.predictions / case['case_id'], dataset['branch_scales_in_100nm'])
    rows.append(row)
    print(json.dumps({'case_id': row['case_id'], 'full': row['field_checks']['full_vector']['relative_l2'],
                      'material': row['field_checks']['material_vector']['relative_l2'], 'all_gates': row['all_gates_passed']}), flush=True)
report = {'schema': 'workai.offline.validation.v1', 'weights_sha256': model_info()['weights_sha256'],
          'source_dataset_sha256': digest(args.dataset), 'threshold': .05, 'reporting_floor_m': 1e-13,
          'reference_geometries': 'Four retained historical test geometries; no new eight-geometry campaign.',
          'original_eight_case_acceptance_inherited': False,
          'reference_comparison': 'Same reference mesh, not a new refined-mesh convergence campaign.',
          'cases': rows, 'all_checks_passed': len(rows) == 4 and all(r['all_gates_passed'] for r in rows)}
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
