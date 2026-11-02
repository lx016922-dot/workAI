"""Independent spatial reconstruction of native COMSOL Q4 eigenfields.

This is a research reference builder, not candidate inference. It preserves the
native COMSOL eigenfrequencies, modal masses, traction integrals, and physical
receiver values. It uses analytic Murnaghan coefficients rather than learned
energy tensors. A built basis alone is not a nonlinear transient certificate.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from numba import njit
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.comsol_artifacts import numeric_csv, properties
from spectral3d.long_reference_force import activation


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def q4_interpolation(points):
    nodes = np.array([-1., -np.sqrt(3 / 7), 0., np.sqrt(3 / 7), 1.])
    values, derivatives = [], []
    for index in range(5):
        others = np.delete(nodes, index)
        polynomial = Polynomial.fromroots(others) / np.prod(nodes[index] - others)
        values.append(polynomial(points))
        derivatives.append(polynomial.deriv()(points))
    return np.array(values).T, np.array(derivatives).T


def cell_fields(fields, sizes, points):
    """Evaluate Q4 fields and physical derivatives in one actual tensor cell."""
    if fields.ndim != 5 or fields.shape[1:] != (5, 5, 5, 3):
        raise ValueError('Three-component Q4 nodal fields are required')
    sizes = np.asarray(sizes, dtype=float)
    if sizes.shape != (3,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('Finite positive physical element sizes required')
    interpolation, derivative = q4_interpolation(points)
    def evaluate(matrices):
        return np.einsum('ia,jb,kc,mabcd->mijkd', *matrices, fields, optimize=True)
    values = evaluate([interpolation] * 3)
    gradients = []
    for axis in range(3):
        matrices = [interpolation] * 3
        matrices[axis] = derivative * (2 / sizes[axis])
        gradients.append(evaluate(matrices))
    return values, np.stack(gradients, axis=-1)


def read_nodal(folder):
    collection = json.loads((folder / 'collection.json').read_text())
    if collection['phase'] != 'completed':
        raise ValueError('Unfinished COMSOL result')
    identity = collection['identity']
    for item in collection['artifacts']:
        path = folder / item['name']
        if (Path(item['name']).name != item['name'] or path.is_symlink()
                or not item.get('verified_against_signed_manifest')
                or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']):
            raise ValueError('Unverified or altered COMSOL artifact')
    receipt = properties(folder / 'completed.properties')
    for key in ['job_id', 'source_sha256', 'scene_sha256', 'parameters_sha256']:
        if receipt[key] != identity[key]:
            raise ValueError('COMSOL identity mismatch')
    if digest(folder / 'scene.json') != identity['scene_sha256']:
        raise ValueError('Changed scene')
    if (receipt['method'] != 'comsol_3d_linear_eigenfrequency_weakform'
            or receipt.get('eigenfield_sampling') != 'quarter_mesh_q4_gll_nodes'
            or receipt.get('nodal_csv_columns') != 'mode,point,u_m,v_m,w_m'
            or receipt.get('actual_mesh_grading_verified_before_solve') != 'true'
            or receipt.get('domain_symmetry') != 'quarter_yz'
            or int(receipt['physical_domain_multiplicity']) != 4):
        raise ValueError('Wrong reference type or missing actual-mesh admission')
    raw = json.loads((folder / 'scene.json').read_text())
    scene, execution = raw['scene'], raw['execution']
    dimensions = [execution['field_n' + axis] for axis in 'xyz']
    edges = [np.array(json.loads(receipt[f'sampling_element_edges_axis_{axis}_m'])) for axis in range(3)]
    sizes = np.array([scene['geometry'][key] for key in ['length_mm', 'width_mm', 'thickness_mm']]) * 1e-3
    sizes[1:] *= .5
    for axis, values in enumerate(edges):
        step = np.diff(values)
        size_key = 'coarse_h_mm' if receipt['level'] == 'coarse' else 'fine_h_mm'
        expected = int(np.ceil(sizes[axis] / (execution[size_key] * 1e-3)))
        ratio = execution.get('grading_ratio' if axis == 0 else 'cross_section_grading_ratio', 1.)
        if (values.shape != (expected + 1,) or dimensions[axis] != 4 * expected + 1
                or not np.isfinite(values).all() or np.any(step <= 0)
                or not np.allclose(values[[0, -1]], [0, sizes[axis]], rtol=0, atol=1e-12)):
            raise ValueError('Actual nodal tensor grid differs from the signed execution')
        if len(step) > 1:
            if not np.isclose(step.max() / step.min(), ratio, rtol=1e-6, atol=1e-7):
                raise ValueError('Actual grading ratio differs')
            decreasing = axis == 0 and scene['boundary']['fixed_face'] == 'xmax'
            if ratio > 1 and not np.all(np.diff(step) < 0 if decreasing else np.diff(step) > 0):
                raise ValueError('Actual grading direction differs')
    grid_points, modes = int(np.prod(dimensions)), int(receipt['modes'])
    points = grid_points + len(scene['receivers'])
    if (grid_points != int(receipt['grid_sample_points']) or points != int(receipt['sample_points'])
            or points * modes > 2_000_000):
        raise ValueError('Nodal export count mismatch or original record budget exceeded')
    table = np.loadtxt(numeric_csv(folder / (receipt['level'] + '.csv')), delimiter=',', skiprows=1)
    if table.shape != (points * modes, 5) or not np.isfinite(table).all():
        raise ValueError('Incomplete/nonfinite native nodal export')
    if (not np.array_equal(table[:, 0], np.repeat(np.arange(modes), points))
            or not np.array_equal(table[:, 1], np.tile(np.arange(points), modes))):
        raise ValueError('Repeated, missing or shuffled native nodal records')
    fields = table[:, 2:].reshape(modes, points, 3).copy()
    integral = np.genfromtxt(numeric_csv(folder / (receipt['level'] + '.field.csv')), delimiter=',', names=True)
    frequency = np.array(json.loads(receipt['modal_frequencies_hz']))
    if (frequency.shape != (modes,) or len(integral) != modes or np.any(frequency <= 0)
            or not np.isfinite(frequency).all() or not np.array_equal(integral['mode'], np.arange(modes))):
        raise ValueError('Incomplete native modal frequencies/integrals')
    for key in integral.dtype.names:
        if not np.isfinite(integral[key]).all():
            raise ValueError('Nonfinite modal integral')
    if np.any(integral['mass_integral'] <= 0):
        raise ValueError('Nonpositive native modal mass')
    np.testing.assert_allclose(integral['stiffness_integral'] / integral['mass_integral'], (2 * np.pi * frequency)**2, rtol=1e-8, atol=0)
    normalized = fields / np.sqrt(integral['mass_integral'])[:, None, None]
    direction = np.array(scene['source']['direction'], dtype=float)
    direction /= np.linalg.norm(direction)
    force = np.stack([integral['force_' + axis + '_integral'] for axis in 'xyz'], axis=-1) @ direction
    return {'scene': scene, 'execution': execution, 'receipt': receipt, 'identity': identity,
            'folder': str(folder), 'edges': edges, 'dimensions': dimensions, 'frequency_hz': frequency,
            'fields': normalized[:, :grid_points].reshape(modes, *dimensions, 3),
            'receiver_fields': normalized[:, grid_points:], 'load': force / np.sqrt(integral['mass_integral'])}


def assemble(reference, quadrature=7, count=None):
    if quadrature not in [5, 7, 9, 13]:
        raise ValueError('Use a declared spatial quadrature refinement')
    count = len(reference['frequency_hz']) if count is None else count
    if not 8 <= count <= len(reference['frequency_hz']):
        raise ValueError('Native modal count outside exported basis')
    edges, scene = reference['edges'], reference['scene']
    material = scene['material']
    rho, lam, mu = [material[key] for key in ['rho_kg_m3', 'lambda_pa', 'mu_pa']]
    constants = np.array([material[key] for key in ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']])
    nodes, weights = leggauss(quadrature)
    local_weights = np.einsum('i,j,k->ijk', weights, weights, weights).ravel()
    cells = int(np.prod([len(axis) - 1 for axis in edges])); per_cell = quadrature**3
    if count * cells * per_cell * 9 * 8 > 6 * 1024**3:
        raise ValueError('Declared reference reconstruction exceeds its 6 GiB gradient memory budget')
    gradient = np.empty((count, cells * per_cell, 3, 3))
    volume = np.empty(cells * per_cell)
    gram, tangent = np.zeros((count, count)), np.zeros((count, count))
    for element, cell in enumerate(itertools.product(*[range(len(axis) - 1) for axis in edges])):
        sizes = np.array([edges[axis][i + 1] - edges[axis][i] for axis, i in enumerate(cell)])
        selection = (slice(0, count),) + tuple(slice(4 * i, 4 * i + 5) for i in cell) + (slice(None),)
        values, gradients = cell_fields(reference['fields'][selection], sizes, nodes)
        values = values.reshape(count, per_cell, 3)
        gradients = gradients.reshape(count, per_cell, 3, 3)
        weighted = local_weights * np.prod(sizes) / 8
        flat = values.reshape(count, -1)
        gram += (values * (weighted * rho)[None, :, None]).reshape(count, -1) @ flat.T
        strain = .5 * (gradients + gradients.swapaxes(-1, -2))
        stress = 2 * mu * strain + lam * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
        tangent += (stress * weighted[None, :, None, None]).reshape(count, -1) @ gradients.reshape(count, -1).T
        indices = slice(element * per_cell, (element + 1) * per_cell)
        gradient[:, indices] = gradients; volume[indices] = weighted
    omega = 2 * np.pi * reference['frequency_hz'][:count]
    mass_error = float(np.max(abs(gram - np.eye(count))))
    stiffness_error = float(np.max(abs(tangent / omega[:, None] / omega[None, :] - np.eye(count))))
    if mass_error > 1e-6 or stiffness_error > 1e-6:
        raise ValueError(f'Reconstructed native eigenfields fail mass/stiffness admission: {mass_error}, {stiffness_error}')
    return {'frequency_hz': reference['frequency_hz'][:count], 'omega': omega,
            'gradient': np.ascontiguousarray(gradient.reshape(count, -1)), 'volume': volume,
            'load': reference['load'][:count], 'receiver_fields': reference['receiver_fields'][:count],
            'material_constants': constants, 'mass_gram_error': mass_error,
            'native_tangent_reconstruction_error': stiffness_error, 'quadrature_points_per_axis': quadrature}


@njit(cache=True)
def stresses(gradients, volume, constants, laws):
    residual = np.empty_like(gradients)
    energies = np.zeros(len(laws)); maximum = np.zeros(len(laws)); minimum = np.ones(len(laws))
    strain, second, linear, full = [np.empty((3, 3)) for _ in range(4)]
    for row in range(len(laws)):
        for point in range(len(volume)):
            g = gradients[row, point]
            base, _, _ = activation(g, constants, 0, strain, second, linear)
            value, norm, jac = activation(g, constants, laws[row], strain, second, full)
            residual[row, point] = (full - linear) * volume[point]
            energies[row] += (value - base) * volume[point]
            maximum[row] = max(maximum[row], norm); minimum[row] = min(minimum[row], jac)
    return residual, energies, maximum, minimum


def nonlinear_force(model, coordinates, laws):
    coordinates = np.atleast_2d(coordinates)
    laws = np.asarray(laws, dtype=np.int64)
    if coordinates.shape != (len(laws), len(model['omega'])) or np.any((laws < 0) | (laws > 2)):
        raise ValueError('Matching reference modal states and constitutive controls required')
    gradients = (coordinates @ model['gradient']).reshape(len(laws), -1, 3, 3)
    residual, energy, strain, jac = stresses(gradients, model['volume'], model['material_constants'], laws)
    force = residual.reshape(len(laws), -1) @ model['gradient'].T
    if (not np.isfinite(force).all() or not np.isfinite(energy).all()
            or np.any(strain > .001) or np.any(jac <= 0)):
        raise ValueError('Independent reference violates original strain/J/finite-energy guard')
    return force, energy, strain, jac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--quadrature', type=int, default=7)
    parser.add_argument('--modes', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = read_nodal(args.reference)
    model = assemble(reference, args.quadrature, args.modes)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'basis.npz', **model)
    report = {'schema': 'workai.native_comsol_nodal_reference.v1',
              'reference_identity': reference['identity'], 'scene': reference['scene'],
              'builder_sha256': digest(__file__), 'force_source_sha256': digest(Path(__file__).resolve().parents[1] / 'spectral3d/long_reference_force.py'),
              'basis_sha256': digest(args.output / 'basis.npz'), 'modes': len(model['omega']),
              'mass_gram_error': model['mass_gram_error'],
              'native_tangent_reconstruction_error': model['native_tangent_reconstruction_error'],
              'quadrature_points_per_axis': args.quadrature,
              'modal_load_units': 'generalized acceleration per pascal of nominal traction',
              'learned_tensors_used': False, 'native_nonlinear_transient': False,
              'nonlinear_reference_convergence_verified': False, 'scientific_accuracy_certified': False}
    (args.output / 'model.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
