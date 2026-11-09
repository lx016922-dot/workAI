"""Independent analysis of a P1 COMSOL delivery: spectral labels against COMSOL, and old against new linear runs.

    python research/comsol-p1-20261024/compare_baselines.py <delivery data dir> \
        --labels <labels-a-top-gaussian> <labels-b-end-axial> <labels-c-two-patch> \
        [--old-linear research/comsol-p1-20261024/recovery-rebuild-20261029/data] --output <json>

COMSOL is the reference. For each case with linear and murnaghan base runs:
- spectral vs COMSOL (coarsegraph.evaluate.case_metrics): projected waveform, f, 2f of the full response, and the
  2f of the nonlinear increment (murnaghan − linear), with the 10 % relative reporting floor;
- old vs new linear (when an older linear run exists): the difference against the size of the nonlinear increment,
  overall and in the f and 2f bands, i.e. how much of the increment a run-to-run difference of the linear model could fake.
Labels: spectral GLL, degree 4, h 1.5 mm, from `python -m sweep run --spec coarsegraph/specs/labels-*.json`.
"""
import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from coarsegraph.comsol import load_case, read_csv, projected, import_data, compare, _pairs
from coarsegraph.evaluate import _features

HERE = Path(__file__).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reproduction(cid, new_dir, old_dir):
    case = load_case(cid); f = case['sources'][0]['frequency_mhz']
    t, new = read_csv(new_dir / f'{cid}_linear.csv', case); _, old = read_csv(old_dir / f'{cid}_linear.csv', case); _, full = read_csv(new_dir / f'{cid}_murnaghan.csv', case)
    L, Lo, N = projected(case, new), projected(case, old), projected(case, full)
    diff, inc = L - Lo, N - L
    feats = lambda X: np.array([_features(t, X[:, r], f) for r in range(X.shape[1])])  # rms, |f|, |2f| per receiver
    fd, fi, fl = feats(diff), feats(inc), feats(L)
    keep = fi[:, 2] >= .1 * fi[:, 2].max()
    return {'linear_difference_max_over_linear_peak_projected': float(np.max(np.abs(diff)) / np.max(np.abs(L))),
            'linear_difference_max_over_linear_peak_xyz': float(np.max(np.abs(new - old)) / np.max(np.abs(old))),
            'increment_max_over_linear_peak': float(np.max(np.abs(inc)) / np.max(np.abs(L))),
            'linear_difference_rms_over_increment_rms': float(np.linalg.norm(diff) / np.linalg.norm(inc)),
            'linear_difference_f_over_linear_f_max': float(np.max(fd[:, 1] / fl[:, 1])),
            'linear_difference_2f_over_increment_2f_worst': float(np.max(fd[keep, 2] / fi[keep, 2])),
            'linear_difference_2f_over_increment_2f_median': float(np.median(fd[keep, 2] / fi[keep, 2])),
            'increment_2f_max_m': float(fi[:, 2].max()), 'linear_record_2f_max_m': float(fl[:, 2].max()),
            'increment_2f_over_linear_record_2f': float(fi[:, 2].max() / fl[:, 2].max()),
            'old_csv_sha256': digest(old_dir / f'{cid}_linear.csv'), 'new_csv_sha256': digest(new_dir / f'{cid}_linear.csv')}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('data', type=Path); p.add_argument('--labels', type=Path, nargs='+', required=True)
    p.add_argument('--old-linear', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    labels = {}
    for d in a.labels: labels.update(_pairs(d))
    with tempfile.TemporaryDirectory() as tmp:
        import_data(a.data, Path(tmp) / 'imported'); rows = compare(Path(tmp) / 'imported', a.labels)
    for row in rows:
        cid = row['case_id']
        if cid in labels:
            row['spectral_label_result_sha256'] = {law: digest(Path(labels[cid][law]) / 'out' / 'result.json') for law in ('linear', 'murnaghan')}
        if a.old_linear and (a.old_linear / f'{cid}_linear.csv').exists():
            row['old_vs_new_linear'] = reproduction(cid, a.data, a.old_linear)
        row['comsol_csv_sha256'] = {law: digest(a.data / f'{cid}_{law}.csv') for law in ('linear', 'murnaghan')}
    out = {'schema': 'workai.comsol.p1.baseline_comparison.v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'reference': 'COMSOL (delivery in the data directory); spectral GLL labels degree 4, h 1.5 mm',
           'metric': 'coarsegraph.evaluate.case_metrics, 10 % relative reporting floor, 0–30 μs; increment = murnaghan − linear of the same source',
           'tool_sha256': digest(HERE), 'rows': rows}
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for row in rows:
        s = row.get('spectral_vs_comsol', {}); r = row.get('old_vs_new_linear', {})
        print(f"{row['case_id']}: waveform {s.get('worst_waveform_l2', float('nan')):.2e}  increment 2f worst {s.get('worst_increment_a2', float('nan')):.2e}"
              f"  vector {s.get('increment_a2_vector_relative', float('nan')):.2e}  old/new linear 2f vs increment {r.get('linear_difference_2f_over_increment_2f_worst', float('nan')):.2e}")


if __name__ == '__main__':
    main()
