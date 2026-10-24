"""Independent momentum-balance checks on saved continued predictions."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from .physics_checks import physics_checks
from spectralop.data import tensors
from .continue_training import atomic, now
from spectral3d.compare import digest


def run(args):
    torch.set_num_threads(1)
    plan = json.loads((args.campaign / 'planned-inputs.json').read_text())
    development = json.loads(args.development.read_text())
    assert development['normalization_from_training_only'] is True
    scales = torch.as_tensor(development['branch_scales_in_100nm'], dtype=torch.float64)
    output = args.campaign / 'physics-buffered'
    output.mkdir(exist_ok=True)
    rows = []
    pending = list(plan['cases'])
    while pending:
        progress = False
        for case in list(pending):
            cid = case['case_id']
            directory = args.campaign / 'predictions-buffered' / cid / 'continued'
            if not (directory / 'prediction.json').exists():
                continue
            receipt = json.loads((directory / 'prediction.json').read_text())
            assert receipt['weights_sha256'] == plan['weights_sha256']
            assert digest(directory / 'fields.npz') == receipt['field_sha256']
            path = output / (cid + '.json')
            if path.exists():
                row = json.loads(path.read_text())
                assert row['prediction_sha256'] == digest(directory / 'prediction.json')
            else:
                with np.load(directory / 'mesh.npz', allow_pickle=False) as saved:
                    mesh = dict(saved)
                with np.load(directory / 'fields.npz', allow_pickle=False) as saved:
                    values = [saved[k] for k in ['linear', 'stvk', 'murnaghan']]
                    times = saved['times_us']
                parts = np.stack([values[0], values[1]-values[0], values[2]-values[1]], 2)
                data = tensors(case['scene'], mesh)
                data['times'] = times
                physical = physics_checks(data, parts, scales)
                row = {'case_id': cid, 'prediction_sha256': digest(directory / 'prediction.json'),
                       'weights_sha256': plan['weights_sha256'], 'physics': physical, 'passed': physical['passed']}
                atomic(path, row)
                del values, mesh, parts, data
                gc.collect()
            rows.append(row)
            pending.remove(case)
            progress = True
            print(json.dumps({'case_id': cid, 'physics_passed': row['passed'],
                'maximum_normalized_residual': max(max(r['interior_branch_rms'] + r['boundary_branch_rms']) for r in row['physics']['checks'])}), flush=True)
        atomic(output / 'partial.json', {'cases': rows, 'pending': [c['case_id'] for c in pending], 'updated_at_utc': now()})
        if pending and not progress:
            status = json.loads((args.campaign / 'predictions-buffered-status.json').read_text())
            failed = {c['case_id'] for c in status['cases'] if c['phase'] == 'failed'}
            if failed.intersection(c['case_id'] for c in pending):
                raise RuntimeError('Prediction failed; physics validation cannot complete.')
            time.sleep(10)
    atomic(output / 'metrics.json', {'weights_sha256': plan['weights_sha256'], 'cases': rows,
           'all_checks_passed': len(rows) == 8 and all(r['passed'] for r in rows),
           'normalization_dataset_sha256': digest(args.development), 'normalization_from_prior_training_only': True,
           'completed_at_utc': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    run(parser.parse_args())
