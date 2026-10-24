"""Evaluate the fixed eight-case plan as independent jobs become available."""
import argparse
import json
import shutil
import time
from pathlib import Path

from .continued_validation import reference_quality, prediction_metrics
from spectral3d.campaign import atomic, now
from spectral3d.compare import digest


def maximum(row):
    values = list(row['field_checks'].values())
    values += [v for p in row['probes'] for v in p['checks'].values()]
    return max(v['relative_l2'] for v in values if v['relative_l2'] is not None)


def run(args):
    root = args.campaign
    plan = json.loads((root / 'planned-inputs.json').read_text())
    prediction_folder = 'predictions-buffered' if args.engine == 'buffered' else 'predictions'
    out = root / ('metrics-buffered' if args.engine == 'buffered' else 'metrics')
    out.mkdir(exist_ok=True)
    source = out / 'evaluation-source'
    source.mkdir(exist_ok=True)
    sources = {}
    for name in ['continued_validation.py', 'evaluate_continuation.py']:
        path = Path(__file__).parent / name
        if (source / name).exists():
            assert digest(path) == digest(source / name)
        else:
            shutil.copyfile(path, source / name)
        sources[name] = digest(path)
    policy = {'created_at_utc': now(), 'threshold': .05, 'reporting_floor_m': 1e-13,
              'require_all_eight_case_reference_and_prediction_gates': True,
              'require_worst_reported_relative_error_not_worse_than_parent': True,
              'require_max_full_vector_error_not_worse_than_parent': True,
              'weights_updated_after_testing': False}
    policy_path = root / 'promotion-policy.json'
    if not policy_path.exists():
        atomic(policy_path, policy)
    rows = []
    pending = list(plan['cases'])
    while pending:
        progress = False
        refs = json.loads((root / 'references-status.json').read_text())
        preds = json.loads((root / (prediction_folder + '-status.json')).read_text())
        for case in list(pending):
            cid = case['case_id']
            r = next(c for c in refs['cases'] if c['case_id'] == cid)
            p = next(c for c in preds['cases'] if c['case_id'] == cid)
            if r['phase'] == 'failed' or p['phase'] == 'failed':
                rows.append({'case_id': cid, 'passed': False, 'execution_failed': True, 'reference': r, 'prediction': p})
                pending.remove(case)
                progress = True
                continue
            if r['phase'] != 'completed' or p['phase'] != 'completed':
                continue
            directory = out / cid
            directory.mkdir(exist_ok=True)
            print(json.dumps({'phase': 'evaluating', 'case_id': cid}), flush=True)
            q_path = directory / 'reference-quality.json'
            quality = json.loads(q_path.read_text()) if q_path.exists() else reference_quality(root / 'references' / cid, cid, directory)
            metrics = {}
            for label in ['parent', 'continued']:
                path = directory / (label + '-metrics.json')
                metrics[label] = json.loads(path.read_text()) if path.exists() else prediction_metrics(
                    root / prediction_folder / cid / label, root / 'references' / cid / 'space/manifest.json', directory, label)
            row = {'case_id': cid, 'reference_quality_passed': quality['passed'],
                   'reference_quality_sha256': digest(q_path), 'parent': metrics['parent'], 'continued': metrics['continued'],
                   'parent_worst_relative_error': maximum(metrics['parent']),
                   'continued_worst_relative_error': maximum(metrics['continued']),
                   'passed': quality['passed'] and metrics['continued']['passed']}
            atomic(directory / 'comparison.json', row)
            rows.append(row)
            pending.remove(case)
            progress = True
            print(json.dumps({'case_id': cid, 'reference_passed': quality['passed'],
                'parent_max_error': row['parent_worst_relative_error'], 'continued_max_error': row['continued_worst_relative_error'],
                'continued_passed': metrics['continued']['passed'],
                'parent_full': metrics['parent']['field_checks']['full_vector']['relative_l2'],
                'continued_full': metrics['continued']['field_checks']['full_vector']['relative_l2']}), flush=True)
        atomic(out / 'partial.json', {'phase': 'running' if pending else 'completed', 'cases': rows,
                                    'pending': [c['case_id'] for c in pending], 'updated_at_utc': now()})
        if not progress and pending:
            time.sleep(10)
    complete = len(rows) == 8 and not any(r.get('execution_failed') for r in rows)
    all_passed = complete and all(r['passed'] for r in rows)
    maximums = {}
    if complete:
        for label in ['parent', 'continued']:
            maximums[label] = {'worst_reported_relative_error': max(maximum(r[label]) for r in rows),
                              **{part: max(r[label]['field_checks'][part]['relative_l2'] for r in rows)
                                 for part in ['full_vector', 'material_vector', 'geometric_vector', 'material_2f_vector']}}
    improves = (complete and maximums['continued']['worst_reported_relative_error'] <= maximums['parent']['worst_reported_relative_error']
                and maximums['continued']['full_vector'] <= maximums['parent']['full_vector'])
    report = {'schema': 'workai.graphwave_continuation_validation.v1', 'execution_engine': args.engine, 'planned_inputs_sha256': digest(root / 'planned-inputs.json'),
              'weights_sha256': plan['weights_sha256'], 'parent_weights_sha256': plan['parent_weights_sha256'],
              'threshold': .05, 'reporting_floor_m': 1e-13, 'test_used_for_weight_selection': False,
              'evaluation_source_sha256': sources, 'promotion_policy_sha256': digest(policy_path),
              'case_count': len(rows), 'receivers': 64, 'all_reference_and_prediction_checks_passed': all_passed,
              'maximums': maximums, 'improves_predeclared_metrics': improves,
              'eligible_for_further_release_checks': all_passed and improves,
              'accepted_for_offline_release': False, 'cases': rows, 'completed_at_utc': now(),
              'scope': 'P4 parent versus trained P5 extension at equal element size; increased compute cost. Reference convergence is separately checked. No experimental or pointwise universal guarantee.'}
    atomic(out / 'metrics.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--engine', choices=['original', 'buffered'], default='original')
    run(parser.parse_args())
