"""Diagnose complete-record linear phase errors of frozen enriched graph models.

This calculation isolates the linear control. It is not a nonlinear prediction,
and COMSOL eigenpair postprocessing is not a native nonlinear transient.
Candidate traces are saved before reference arrays are loaded.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, physical_sampling, read_modes, save
from graphwave.modal_forcing import driven_states
from graphwave.quarter_symmetry import make_quarter_mesh
from spectral3d.mesh import boundary_data
from tools.comsol_modal_response import response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    source_hashes = dict(plan['source_sha256'])
    source_hashes['tools/diagnose-enriched-linear-matrix.py'] = digest(__file__)
    for name, expected in source_hashes.items():
        if digest(repo / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    labels = [path.name for path in args.model]
    if len(set(labels)) != len(labels):
        parser.error('Model directory names must be distinct')
    case_ids = [case['id'] for case in plan['cases']]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('Duplicate case identifiers')
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        'schema': 'workai.enriched_linear_diagnosis.v1',
        'plan_sha256': digest(args.plan), 'source_sha256': source_hashes,
        'models': {}, 'case_ids': case_ids,
        'nonlinear_prediction': False, 'native_comsol_transient': False,
        'scientific_accuracy_certified': False,
        'scope': 'Linear phase development diagnosis; all nonlinear and reference convergence checks remain required',
    }
    for path in args.model:
        config = json.loads((path / 'model.json').read_text())
        for name, key in [('basis.npz', 'basis_sha256'), ('weights.npz', 'weights_sha256')]:
            if digest(path / name) != config[key]:
                raise ValueError('Model artifact changed: ' + str(path / name))
        if not config.get('quarter_domain'):
            raise ValueError('An explicit quarter-domain model is required')
        identity['models'][path.name] = {
            'model_sha256': digest(path / 'model.json'),
            'basis_sha256': config['basis_sha256'],
            'weights_sha256': config['weights_sha256'],
            'modes': config['exported_modes'],
        }
        folder = args.output / path.name
        folder.mkdir()
        with np.load(path / 'basis.npz', allow_pickle=False) as archive:
            frequency, fields = archive['frequency_hz'], archive['fields']
        if len(frequency) != config['exported_modes']:
            raise ValueError('The complete trained space must be used')
        mesh = make_quarter_mesh(config['scene'], config['degree'], config['h_mm'],
                                 config['grading_ratios'], config['axis_elements'])
        for case in plan['cases']:
            scene = case['scene']
            for key in ['geometry', 'material', 'boundary', 'constitutive']:
                if scene[key] != config['scene'][key]:
                    raise ValueError('Case changes compiled physics: ' + key)
            acq = scene['acquisition']
            times = np.arange(round(acq['duration_us'] / acq['sample_step_us']) + 1) * acq['sample_step_us']
            if len(times) != 25001 or times[-1] != 5000:
                raise ValueError('Complete 5 ms records are required')
            load, _, _ = boundary_data(mesh, scene)
            coupling = np.einsum('mnc,nc->m', fields, load)
            points = np.asarray([row['point_mm'] for row in scene['receivers']]) * 1e-3
            receivers = physical_sampling(mesh, points)(fields)
            coordinates, velocity, method = driven_states(scene, frequency, times)
            del velocity
            traces = np.einsum('tm,m,mrc->trc', coordinates, coupling, receivers, optimize=True)
            del coordinates
            target = folder / (case['id'] + '.npz')
            np.savez_compressed(target, times_us=times, traces=traces)
            save(target.with_suffix('.json'), {
                'case_id': case['id'], 'duration_us': 5000, 'frames': len(times),
                'method': method, 'prediction_sha256': digest(target),
                'reference_arrays_loaded_before_prediction': False,
            })
            print(json.dumps({'phase': 'candidate_linear_record', 'model': path.name,
                              'case': case['id'], 'elapsed_s': time.monotonic() - started}), flush=True)
        del mesh, fields
    save(args.output / 'identity.json', identity)

    # References enter only after every candidate-only record is on disk.
    reference = read_modes(args.reference)
    spec = importlib.util.spec_from_file_location('linear_check', repo / 'tools/validate-comsol-linear-matrix.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    rows = []
    for case in plan['cases']:
        times, expected, method = response(reference, case['scene'])
        actual = {}
        row = {'case_id': case['id'], 'models': {}, 'model_pairs': {}}
        for label in labels:
            path = args.output / label / (case['id'] + '.npz')
            record = json.loads(path.with_suffix('.json').read_text())
            if digest(path) != record['prediction_sha256']:
                raise ValueError('Saved candidate trace changed')
            with np.load(path, allow_pickle=False) as archive:
                if not np.array_equal(times, archive['times_us']):
                    raise ValueError('Candidate/reference time grid mismatch')
                actual[label] = archive['traces']
            row['models'][label] = comparison.check(actual[label], expected, case['scene'])
        for index, left in enumerate(labels):
            for right in labels[index + 1:]:
                row['model_pairs'][left + '_vs_' + right] = comparison.check(actual[left], actual[right], case['scene'])
        target = args.output / ('reference-' + case['id'] + '.npz')
        np.savez_compressed(target, times_us=times, traces=expected)
        row.update(reference_sha256=digest(target), reference_method=method)
        save(args.output / (case['id'] + '.json'), row)
        rows.append(row)
        print(json.dumps({'phase': 'linear_diagnosis', 'case': case['id'], 'models': {
            key: {name: value[name] for name in ['maximum_relative_l2', 'maximum_vector_waveform_relative_l2', 'passed']}
            for key, value in row['models'].items()}}), flush=True)
    for name, expected in source_hashes.items():
        if digest(repo / name) != expected:
            raise ValueError('Source changed during diagnosis: ' + name)
    summary = dict(identity, reference_identity=reference['identity'], results=rows,
                   elapsed_s=time.monotonic() - started)
    summary['totals'] = {label: {
        'all_component_band_passes': sum(row['models'][label]['passed'] for row in rows),
        'vector_waveform_passes': sum(row['models'][label]['maximum_vector_waveform_relative_l2'] < .05 for row in rows),
        'maximum_relative_l2': max(row['models'][label]['maximum_relative_l2'] for row in rows),
        'cases': len(rows),
    } for label in labels}
    save(args.output / 'summary.json', summary)
    print(json.dumps({'phase': 'completed', 'totals': summary['totals']}), flush=True)


if __name__ == '__main__':
    main()
