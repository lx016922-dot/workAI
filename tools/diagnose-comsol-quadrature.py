"""Separate mass/strain integration from spatial error using actual COMSOL modes.

This is an offline diagnostic, not an inference implementation or a learned
correction. Every quadrature variant uses the same mesh and material tensors.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral3d.mesh import make_mesh, boundary_data
from graphwave.modal_calibration import stiffness, read_modes, modal_sampling, match, digest


def exact_operators(mesh, quadratic, density):
    n = mesh['degree'] + 1
    q = n + 1
    points, weights = np.polynomial.legendre.leggauss(q)
    values, derivatives = [], []
    for i, node in enumerate(mesh['gll']):
        coefficients = np.polynomial.polynomial.polyfromroots(np.delete(mesh['gll'], i))
        coefficients /= np.polynomial.polynomial.polyval(node, coefficients)
        values.append(np.polynomial.polynomial.polyval(points, coefficients))
        derivatives.append(np.polynomial.polynomial.polyval(points, np.polynomial.polynomial.polyder(coefficients)))
    v = np.array(values).T
    d = np.array(derivatives).T
    interpolation = np.einsum('ia,jb,kc->ijkabc', v, v, v).reshape(q**3, n**3)
    gradient = np.stack([np.einsum('ia,jb,kc->ijkabc', d, v, v),
                         np.einsum('ia,jb,kc->ijkabc', v, d, v),
                         np.einsum('ia,jb,kc->ijkabc', v, v, d)], -1).reshape(q**3, n**3, 3)
    quadrature = np.einsum('i,j,k->ijk', weights, weights, weights).ravel()
    rows, columns, stiffness_values, mass_values = [], [], [], []
    for element, size in zip(mesh['elements'], mesh['size']):
        grad = gradient * (2 / size)
        volume = quadrature * np.prod(size) / 8
        strain = np.zeros((q**3, 6, n**3 * 3))
        strain[:, 0, 0::3] = grad[:, :, 0]
        strain[:, 1, 1::3] = grad[:, :, 1]
        strain[:, 2, 2::3] = grad[:, :, 2]
        for row, a, b in [(3, 1, 2), (4, 0, 2), (5, 0, 1)]:
            strain[:, row, a::3] = grad[:, :, b] / np.sqrt(2)
            strain[:, row, b::3] = grad[:, :, a] / np.sqrt(2)
        block = np.einsum('qra,rs,qsb,q->ab', strain, 2e11 * quadratic, strain, volume, optimize=True)
        scalar_mass = np.einsum('qa,qb,q->ab', interpolation, interpolation, volume * density)
        dofs = (element.ravel()[:, None] * 3 + np.arange(3)).ravel()
        rows.append(np.repeat(dofs, len(dofs)))
        columns.append(np.tile(dofs, len(dofs)))
        stiffness_values.append(block.ravel())
        mass_values.append(np.kron(scalar_mass, np.eye(3)).ravel())
    shape = (mesh['points'].size, mesh['points'].size)
    indices = (np.concatenate(rows), np.concatenate(columns))
    return tuple(coo_matrix((np.concatenate(values), indices), shape=shape).tocsr()
                 for values in [stiffness_values, mass_values])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=Path('offline/model/continued-weights.npz'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reference = read_modes(args.reference)
    degree = int(reference['receipt']['element_order'])
    h = float(reference['receipt']['mesh_h_mm'])
    if degree != 4 or reference['receipt']['mesh_type'] != 'hex':
        raise ValueError('This diagnostic requires a matched P4 Cartesian COMSOL mesh')
    scene = reference['scene']
    mesh = make_mesh(scene, degree, h)
    if len(mesh['elements']) != int(reference['receipt']['elements']):
        raise ValueError('COMSOL and local mesh topology differ')
    with np.load(args.weights, allow_pickle=False) as file:
        weights = dict(file)
    exact_k, exact_m = exact_operators(mesh, weights['quadratic_tensor'], scene['material']['rho_kg_m3'])
    gll_k = stiffness(mesh, mesh['derivative'], weights['quadratic_tensor'])
    gll_m = diags(np.repeat(mesh['mass'], 3)).tocsr()
    load, fixed, _ = boundary_data(mesh, scene)
    free = np.ones(mesh['points'].size, bool)
    free[(fixed[:, None] * 3 + np.arange(3)).ravel()] = False
    label_mesh = make_mesh(scene, 4, 2.5)
    if not np.allclose(label_mesh['points'], reference['coordinates'], rtol=0, atol=1e-14):
        raise ValueError('Unexpected COMSOL export coordinates')
    sample = modal_sampling(mesh, reference['coordinates'])
    count = len(reference['frequency_hz'])
    records = []
    for name, k, m in [('gll_K_gll_M', gll_k, gll_m), ('exact_K_gll_M', exact_k, gll_m),
                       ('gll_K_exact_M', gll_k, exact_m), ('exact_K_exact_M', exact_k, exact_m)]:
        start = time.monotonic()
        values, vectors = eigsh(k[free][:, free], M=m[free][:, free], k=count, sigma=0,
                               which='LM', tol=1e-11, v0=np.ones(free.sum()))
        order = np.argsort(values)
        frequencies = np.sqrt(values[order]) / (2 * np.pi)
        fields = np.zeros((count, mesh['points'].size))
        fields[:, free] = vectors[:, order].T
        mapping, mac = match(reference['fields'], sample(fields.reshape(count, -1, 3)), label_mesh['mass'])
        participation = abs(load.ravel()[free] @ vectors[:, order])
        primary = int(np.argmax(participation))
        primary_reference = int(np.flatnonzero(mapping == primary)[0])
        difference = frequencies[mapping] - reference['frequency_hz']
        records.append({'method': name, 'primary_frequency_hz': float(frequencies[primary]),
                        'primary_comsol_frequency_hz': float(reference['frequency_hz'][primary_reference]),
                        'primary_difference_hz': float(difference[primary_reference]),
                        'maximum_frequency_difference_hz': float(np.max(abs(difference))),
                        'minimum_mac': float(mac.min()), 'frequencies_hz': frequencies[mapping].tolist(),
                        'frequency_differences_hz': difference.tolist(), 'seconds': time.monotonic() - start})
        print(json.dumps({k: v for k, v in records[-1].items() if not isinstance(v, list)}), flush=True)
    result = {'schema': 'workai.comsol_quadrature_diagnostic.v1', 'degree': degree, 'h_mm': h,
              'nodes': len(mesh['points']), 'elements': len(mesh['elements']), 'mode_count': count,
              'reference_identity': reference['identity'], 'weights_sha256': digest(args.weights),
              'diagnostic_source_sha256': digest(__file__), 'cases': records,
              'scope': 'Offline mass/stiffness integration diagnosis, using analytic derivatives and frozen material tensors; not a new neural predictor',
              'scientific_accuracy_certified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
