"""Independent field comparison for continued-model experiments.

The accelerated transfer implements the existing high-order interpolant and
checks it against the original implementation. It is used only for comparing
saved results, never for modifying predictions or training weights.
"""
import gc
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, load_npz, save_npz
from spectral3d.mesh import interpolation
from spectral3d.compare import compare, spectral, complex_coefficient, digest
from spectral3d.field_compare import weighted, field
from spectral3d.campaign import atomic


@njit(cache=True)
def _shape(nodes, local):
    n = len(nodes)
    values = np.ones((3, n))
    derivative = np.zeros((3, n))
    for axis in range(3):
        for i in range(n):
            denominator = 1.
            product = 1.
            for j in range(n):
                if j != i:
                    denominator *= nodes[i] - nodes[j]
                    product *= local[axis] - nodes[j]
            values[axis, i] = product / denominator
            for skip in range(n):
                if skip == i:
                    continue
                product = 1.
                for j in range(n):
                    if j != i and j != skip:
                        product *= local[axis] - nodes[j]
                derivative[axis, i] += product / denominator
    weights = np.empty(n**3)
    gradients = np.empty((n**3, 3))
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                weights[count] = values[0, i] * values[1, j] * values[2, k]
                gradients[count, 0] = derivative[0, i] * values[1, j] * values[2, k]
                gradients[count, 1] = values[0, i] * derivative[1, j] * values[2, k]
                gradients[count, 2] = values[0, i] * values[1, j] * derivative[2, k]
                count += 1
    return weights, gradients


@njit(cache=True)
def _locate(points, cells, nodes, lower, upper, queries, tolerance):
    owner = np.full(len(queries), -1, dtype=np.int32)
    coordinates = np.zeros((len(queries), 3))
    for q in range(len(queries)):
        point = queries[q]
        for e in range(len(cells)):
            inside = True
            for axis in range(3):
                margin = .03 * (upper[e, axis] - lower[e, axis]) + 1e-12
                if point[axis] < lower[e, axis] - margin or point[axis] > upper[e, axis] + margin:
                    inside = False
            if not inside:
                continue
            local = np.zeros(3)
            for _ in range(18):
                weights, gradients = _shape(nodes, local)
                residual = -point.copy()
                jac = np.zeros((3, 3))
                for a in range(len(cells[e])):
                    for i in range(3):
                        residual[i] += weights[a] * points[cells[e, a], i]
                        for j in range(3):
                            jac[i, j] += points[cells[e, a], i] * gradients[a, j]
                if np.linalg.norm(residual) < 5e-16:
                    if np.max(np.abs(local)) <= 1 + tolerance:
                        owner[q] = e
                        coordinates[q] = local
                    break
                try:
                    step = np.linalg.solve(jac, residual)
                except Exception:
                    break
                local -= step
                if np.max(np.abs(local)) > 4:
                    break
            if owner[q] >= 0:
                break
    return owner, coordinates


@njit(cache=True)
def _entries(cells, nodes, owners, coordinates):
    indices = np.empty((len(owners), cells.shape[1]), dtype=np.int32)
    values = np.empty((len(owners), cells.shape[1]))
    for q in range(len(owners)):
        indices[q] = cells[owners[q]]
        values[q] = _shape(nodes, coordinates[q])[0]
    return indices, values


def transfer(coarse, fine, path):
    path = Path(path)
    if path.exists():
        matrix = load_npz(path)
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == receipt['sha256']
        return matrix, receipt
    cells = coarse['elements'].reshape(len(coarse['elements']), -1)
    lower = coarse['bounds_min'] if 'bounds_min' in coarse else coarse['origin']
    upper = coarse['bounds_max'] if 'bounds_max' in coarse else coarse['origin'] + coarse['size']
    owners, local = _locate(coarse['points'], cells, coarse['gll'], lower, upper, fine['points'], 1e-8)
    missing = np.flatnonzero(owners < 0)
    if len(missing):
        assert str(coarse.get('mapping_kind', '')) == 'isoparametric', 'Unmapped Cartesian point'
        second, coords = _locate(coarse['points'], cells, coarse['gll'], lower, upper, fine['points'][missing], .002)
        assert np.all(second >= 0), 'Comparison point exceeds the unchanged .002 local-coordinate tolerance'
        owners[missing], local[missing] = second, coords
    ids, values = _entries(cells, coarse['gll'], owners, local)
    rows = np.broadcast_to(np.arange(len(owners))[:, None], ids.shape)
    keep = np.abs(values) > 1e-15
    matrix = csr_matrix((values[keep], (rows[keep], ids[keep])), shape=(len(fine['points']), len(coarse['points'])))
    np.testing.assert_allclose(matrix @ coarse['points'], fine['points'], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.asarray(matrix.sum(1)).ravel(), 1., rtol=0, atol=2e-12)
    # Test against the original implementation using an independent irregular
    # field, including every extrapolated node when the set is small.
    selected = np.unique(np.r_[np.linspace(0, len(owners)-1, 48, dtype=int), missing[:48]])
    values = np.random.default_rng(2026102421).normal(size=(len(coarse['points']), 3))
    max_error = 0.
    for i in selected:
        try:
            nodes, weight = interpolation(coarse, fine['points'][i] * 1000)
        except ValueError:
            from spectral3d.curved_mesh import interpolation as curved_interpolation
            nodes, weight = curved_interpolation(coarse, fine['points'][i] * 1000, local_tolerance=.002)
        expected = weight @ values[nodes]
        actual = np.asarray(matrix[i] @ values).ravel()
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
        max_error = max(max_error, float(np.max(abs(actual - expected))))
    save_npz(path, matrix)
    receipt = {'sha256': digest(path), 'coarse_nodes': len(coarse['points']), 'fine_nodes': len(fine['points']),
               'extrapolated_nodes': len(missing), 'maximum_reference_coordinate_excess': .002,
               'checked_against_original_interpolation': len(selected), 'interpolation_max_abs_difference': max_error}
    atomic(path.with_suffix('.json'), receipt)
    return matrix, receipt


def map_field(matrix, values):
    mapped = np.empty((len(values), matrix.shape[0], 3), dtype=values.dtype)
    for start in range(0, len(values), 16):
        part = values[start:start+16]
        mapped[start:start+len(part)] = (matrix @ part.transpose(1, 0, 2).reshape(matrix.shape[1], -1)).reshape(matrix.shape[0], len(part), 3).transpose(1, 0, 2)
    return mapped


def field_checks(left, right, matrix, mass, times, frequency):
    checks = {}
    for name in ['linear', 'stvk', 'full', 'geometric', 'material']:
        if name in ['linear', 'stvk', 'full']:
            index = ['linear', 'stvk', 'full'].index(name)
            a, b = left[index], right[index]
        else:
            index = 1 if name == 'geometric' else 2
            a, b = left[index] - left[index - 1], right[index] - right[index - 1]
        mapped = map_field(matrix, a)
        checks[name + '_vector'] = weighted(mapped, b, mass)
        for axis, label in enumerate('xyz'):
            checks[name + '_' + label] = weighted(mapped[:, :, axis:axis+1], b[:, :, axis:axis+1], mass)
        if name == 'material':
            ca = complex_coefficient(times, mapped, 2 * frequency)
            cb = complex_coefficient(times, b, 2 * frequency)
            checks['material_2f_vector'] = weighted(ca[None], cb[None], mass)
            for axis, label in enumerate('xyz'):
                checks['material_2f_' + label] = weighted(ca[None, :, axis:axis+1], cb[None, :, axis:axis+1], mass)
        del mapped
    return checks


def reference_data(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    assert manifest['phase'] == 'completed'
    case = manifest['cases'][0]
    assert digest(path.parent / case['mesh_file']) == case['mesh_sha256']
    with np.load(path.parent / case['mesh_file'], allow_pickle=False) as saved:
        mesh = dict(saved)
    values = [field(path.parent, case, law) for law in ['linear', 'stvk', 'murnaghan']]
    return case, mesh, values


def reference_quality(root, case_id, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: root / key / 'manifest.json' for key in ['base', 'space', 'degree', 'time']}
    summaries = {key: spectral(path) for key, path in paths.items()}
    checks = {key: compare(summaries['base'], summaries[key]) for key in ['space', 'degree', 'time']}
    case, coarse, left = reference_data(paths['base'])
    _, fine, right = reference_data(paths['space'])
    assert case['case_id'] == case_id
    matrix, receipt = transfer(coarse, fine, output / 'reference-transfer.npz')
    times = summaries['base'][case_id]['times']
    fields = field_checks(left, right, matrix, fine['mass'], times, case['scene']['source']['frequency_mhz'])
    guards = all(d['energy_balance_within_1_percent'] and d['max_strain'] <= .001 and d['min_J'] > 0
                 for summary in summaries.values() for d in summary[case_id]['diagnostics'])
    result = {'schema': 'workai.continued_reference_quality.v1', 'case_id': case_id, 'threshold': .05,
              'reporting_floor_m': 1e-13, 'comparisons': checks, 'fields': fields, 'guards_passed': guards,
              'transfer': receipt, 'levels': {k: {'manifest': str(v), 'sha256': digest(v)} for k, v in paths.items()},
              'passed': guards and all(v['pass'] for v in fields.values()) and all(v['all_probe_checks_pass'] for v in checks.values())}
    atomic(output / 'reference-quality.json', result)
    del left, right, matrix
    gc.collect()
    return result


def prediction_metrics(directory, reference, output, label):
    directory, reference, output = Path(directory), Path(reference), Path(output)
    receipt = json.loads((directory / 'prediction.json').read_text())
    for name, key in [('fields.npz', 'field_sha256'), ('mesh.npz', 'mesh_sha256'), ('probes.npz', 'probe_sha256')]:
        assert digest(directory / name) == receipt[key]
    assert receipt['neural_forward_only'] and not receipt['numerical_correction'] and not receipt['reference_responses_read_for_inference']
    case, fine, true = reference_data(reference)
    with np.load(directory / 'mesh.npz', allow_pickle=False) as saved:
        coarse = dict(saved)
    with np.load(directory / 'fields.npz', allow_pickle=False) as saved:
        predicted = [saved[k] for k in ['linear', 'stvk', 'murnaghan']]
        times = saved['times_us']
    matrix, transfer_receipt = transfer(coarse, fine, output / (label + '-transfer.npz'))
    checks = field_checks(predicted, true, matrix, fine['mass'], times, case['scene']['source']['frequency_mhz'])
    with np.load(directory / 'probes.npz', allow_pickle=False) as saved:
        controls = saved['controls']
    parts = np.stack([controls[:, :, 0], controls[:, :, 1] - controls[:, :, 0], controls[:, :, 2] - controls[:, :, 1]], 2)
    item = {case['case_id']: {'scene': case['scene'], 'times': times, 'parts': parts, 'diagnostics': receipt['diagnostics']}}
    probes = compare(item, spectral(reference))['cases'][0]
    from spectral3d.mesh import boundary_data
    fixed = boundary_data(coarse, case['scene'])[1]
    exact = all(np.all(v[0] == 0) and np.all(v[:, fixed] == 0) for v in predicted)
    guards = all(d['energy_balance_within_1_percent'] and d['max_strain'] <= .001 and d['min_J'] > 0 for d in receipt['diagnostics'])
    result = {'case_id': case['case_id'], 'label': label, 'weights_sha256': receipt['weights_sha256'],
              'prediction_sha256': digest(directory / 'prediction.json'), 'fine_reference_sha256': digest(reference),
              'field_checks': checks, 'probes': probes['probes'], 'probe_checks_passed': probes['all_probe_checks_pass'],
              'energy_strain_J_passed': guards, 'exact_initial_and_fixed': exact, 'transfer': transfer_receipt,
              'elapsed_s': sum(d['elapsed_s'] for d in receipt['diagnostics']),
              'passed': all(v['pass'] for v in checks.values()) and probes['all_probe_checks_pass'] and guards and exact}
    atomic(output / (label + '-metrics.json'), result)
    del predicted, true, matrix, parts, controls
    gc.collect()
    return result
