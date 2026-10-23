"""Collect finished numerical/training evidence without solving or training."""
import argparse, hashlib, json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root = args.root
    admission_path = root / 'spectral3d/admissions/aluminum16-20261023-v3/manifest.json'
    admission = json.loads(admission_path.read_text())
    run = root / 'spectralop/runs/aluminum-spectral-20261023-v1'
    status = json.loads((run / 'status.json').read_text())
    metrics = json.loads((run / 'metrics.json').read_text())
    freeze = json.loads((run / 'checkpoint-freeze.json').read_text())
    assert admission['phase'] == 'completed' and admission['all_reference_checks_passed']
    assert status['status'] == 'completed_experimental' and status['step'] == 8000
    assert sha(run / 'best.pt') == metrics['checkpoint_sha256'] == freeze['checkpoint_sha256']
    assert not metrics['accepted_for_solver_routing'] and not metrics['test_used_for_checkpoint_selection']
    groups = [
        'datasets/aluminum16-20261023',
        'admissions/aluminum16-20261023',
        'admissions/aluminum16-20261023-v2',
        'admissions/degree-candidates-20261023',
        'admissions/aluminum16-20261023-v3',
        'campaigns/curved-p3-h20-nm4-20261023',
        'campaigns/curved-p3-h14-nm4-20261023',
        'campaigns/curved-p4-h20-nm4-20261023',
        'campaigns/optimized-newmark-check-20261023',
    ]
    counts = []; seen = set()
    for group in groups:
        states = {}; manifests = []
        for path in sorted((root / 'spectral3d' / group).rglob('manifest.json')):
            record = json.loads(path.read_text())
            if record.get('schema') != 'workai.spectral_reference_campaign.v1':
                continue
            manifests.append({'path': str(path.relative_to(root)), 'sha256': sha(path)})
            for case in record['cases']:
                for control in case.get('controls', []):
                    directory = path.parent / control['directory']
                    assert directory not in seen
                    seen.add(directory)
                    phase = control['phase']
                    assert phase in ['completed', 'interrupted'], (directory, phase)
                    states[phase] = states.get(phase, 0) + 1
        counts.append({'group': group, 'controls': states, 'manifests': manifests})
    quality = []
    for case in admission['cases']:
        path = Path(case['quality_path']); assert sha(path) == case['quality_sha256']
        checks = json.loads(path.read_text()); field = checks['field']['cases'][0]['checks']
        quality.append({'case_id': case['case_id'], 'base_manifest': case['base_manifest'],
                        'base_manifest_sha256': case['base_manifest_sha256'], 'quality_sha256': sha(path),
                        'reference_checks_passed': case['reference_checks_passed'],
                        'full_field_h_relative_l2': field['full_vector']['relative_l2'],
                        'material_field_h_relative_l2': field['material_vector']['relative_l2']})
    tests = [{'case_id': c['case_id'],
              'full_field_relative_l2': c['field_checks']['full_vector']['relative_l2'],
              'material_field_relative_l2': c['field_checks']['material_vector']['relative_l2'],
              'probe_checks_pass': c['probe_checks_pass']} for c in metrics['cases'] if c['split'] == 'test']
    record = {'schema': 'workai.spectral_iteration_review.v1', 'physical_new_geometries': 16,
              'split_counts': status['split_counts'], 'completed_reference_controls': sum(c['controls'].get('completed', 0) for c in counts),
              'interrupted_reference_controls': sum(c['controls'].get('interrupted', 0) for c in counts),
              'reference_groups': counts, 'admission_sha256': sha(admission_path), 'reference_quality': quality,
              'new_optimization_steps': status['step'], 'selected_step': metrics['selected_step'],
              'checkpoint_sha256': metrics['checkpoint_sha256'], 'metrics_sha256': sha(run / 'metrics.json'),
              'independent_tests': tests, 'acceptance': metrics['acceptance'],
              'accepted_for_solver_routing': False, 'general_ai_goal_achieved': False,
              'scope': 'Only 16 newly sampled parameterized aluminum geometries. Repeated controls and refinement levels are not independent geometries. No continuous-domain error guarantee.'}
    with args.output.open('x') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k not in ['reference_groups', 'reference_quality']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
