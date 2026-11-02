"""Add explicitly vector-normalized STFT checks to the frozen chirp evidence.

The original band metric is RMS per complex coefficient, including the three
components in its denominator. This supplementary metric uses vector magnitude
per time/frequency bin, retaining the original 5% / 1e-13 m decision rule.
Original reports and neural trajectories are not rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import stft


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit = json.loads(args.audit.read_text()); cases = []
    for row in audit['rows']:
        if not row['case_id'].startswith('chirp-'):
            continue
        case = row['case_id']
        choices = [args.input / case / case, args.input / ('comsol-linear-' + case) / case,
                   args.input / case]
        present = [path for path in choices if (path / 'metrics.json').exists()]
        if len(present) != 1:
            raise ValueError('Missing or ambiguous completed case: ' + case)
        folder = present[0]
        for name, key in [('identity.json', 'identity_sha256'), ('metrics.json', 'metrics_sha256'),
                          ('comsol-modal-reference.npz', 'comsol_modal_reference_sha256')]:
            if sha(folder / name) != row[key]:
                raise ValueError('Archived data differ from independent audit')
        identity = json.loads((folder / 'identity.json').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        if identity['weights_sha256'] != audit['weights_sha256']:
            raise ValueError('Wrong model')
        blocks = []; first = 0
        for block in metrics['prediction']['probe_blocks']:
            name = block['file']
            if Path(name).name != name or block['first_frame'] != first or sha(folder / name) != block['sha256']:
                raise ValueError('Probe block changed or missing')
            with np.load(folder / name, allow_pickle=False) as saved:
                blocks.append(saved['predicted'][0])
            first = block['last_frame'] + 1
        actual = np.concatenate(blocks)
        with np.load(folder / 'comsol-modal-reference.npz', allow_pickle=False) as saved:
            expected = saved['linear']
        if first != 25001 or actual.shape != (25001, 8, 3) or expected.shape != actual.shape:
            raise ValueError('Incomplete 5 ms record')
        scene = identity['case']['scene']; low, high = sorted([scene['source']['frequency_mhz'], scene['signal']['end_frequency_mhz']])
        records = []
        for receiver in range(8):
            frequency, _, a = stft(actual[:, receiver], fs=1/scene['acquisition']['sample_step_us'], nperseg=512, noverlap=384, axis=0, boundary=None)
            _, _, b = stft(expected[:, receiver], fs=1/scene['acquisition']['sample_step_us'], nperseg=512, noverlap=384, axis=0, boundary=None)
            for band, order in [('drive_band', 1), ('second_band', 2)]:
                mask = (frequency >= order*.9*low) & (frequency <= order*(high+.02))
                count = int(np.count_nonzero(mask)*a.shape[-1])
                if not count:
                    raise ValueError('Empty frequency band')
                error = float(np.sqrt(np.sum(abs(a[mask]-b[mask])**2)/count))
                scale = float(np.sqrt(np.sum(abs(b[mask])**2)/count))
                legacy = metrics['linear_comparison']['spectral'][receiver][band]
                # Verify the definition change, not a hidden trajectory change.
                np.testing.assert_allclose([error, scale], np.sqrt(3)*np.array([legacy['error_rms_m'], legacy['reference_rms_m']]), rtol=1e-7, atol=1e-23)
                passed = error <= (.05*scale if scale > 1e-13 else 1e-13)
                records.append({'receiver': receiver, 'band': band, 'complex_bins': count,
                                'vector_error_rms_m': error, 'vector_reference_rms_m': scale,
                                'relative_l2': error/scale if scale > 1e-13 else None,
                                'passed': bool(passed), 'legacy_per_coefficient_passed': legacy['passed'],
                                'legacy_was_weak': legacy['reference_rms_m'] <= 1e-13,
                                'vector_is_weak': scale <= 1e-13})
        cases.append({'case_id': case, 'metrics_sha256': row['metrics_sha256'], 'checks': records,
                      'supplemental_band_checks_passed': all(r['passed'] for r in records),
                      'original_full_case_passed': row['linear_comparison_passed'],
                      'full_case_with_supplement_passed': row['linear_comparison_passed'] and all(r['passed'] for r in records)})
    if len(cases) != 5:
        raise ValueError('All five chirp cycle counts are required')
    result = {'schema': 'workai.comsol_chirp_vector_band_supplement.v1',
              'scope': 'Linear response at eight receivers for all five chirp cycle counts; complete 5 ms records',
              'reference_kind': 'COMSOL-sourced truncated linear modal response, not native nonlinear transient',
              'duration_us': 5000, 'samples_per_case': 25001,
              'source_audit_sha256': sha(args.audit), 'weights_sha256': audit['weights_sha256'],
              'normalization': 'sqrt(sum over XYZ, selected frequencies and windows of squared complex coefficients / (frequency bins * windows))',
              'original_normalization': 'RMS per complex coefficient, with XYZ included in the denominator',
              'relative_limit': .05, 'absolute_floor_m': 1e-13,
              'completed_cases': len(cases), 'completed_band_checks': sum(len(c['checks']) for c in cases),
              'changed_band_verdicts': sum(r['passed'] != r['legacy_per_coefficient_passed'] for c in cases for r in c['checks']),
              'changed_weak_classifications': sum(r['legacy_was_weak'] != r['vector_is_weak'] for c in cases for r in c['checks']),
              'original_reports_preserved': True, 'predictions_unchanged': True,
              'reference_convergence_verified': False, 'scientific_accuracy_certified': False, 'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
