"""Train a conservative reduced space from candidate-only static responses.

No reference trajectory, fitted constitutive coefficient, sparse quadrature or
online correction is used. Coordinates use the original diagonal nodal mass.
"""
import numpy as np
from scipy.linalg import eigh


def fit_space(base, snapshots, added):
    """Mass-normalized POD of equally weighted independent force responses."""
    if (base.ndim != 2 or snapshots.ndim != 2 or base.shape[0] != snapshots.shape[0]
            or not 0 < added < snapshots.shape[1] or
            not np.isfinite(base).all() or not np.isfinite(snapshots).all()):
        raise ValueError('Finite compatible bases and more snapshots than new modes required')
    np.testing.assert_allclose(base.T @ base, np.eye(base.shape[1]), rtol=0, atol=1e-8)
    residual = snapshots.copy()
    for _ in range(2):
        residual -= base @ (base.T @ residual)
    norms = np.linalg.norm(residual, axis=0)
    if np.any(norms <= np.finfo(float).eps * np.linalg.norm(snapshots, axis=0)):
        raise ValueError('A training response has no resolved complementary component')
    residual /= norms
    values, vectors = eigh(residual.T @ residual)
    order = np.argsort(values)[::-1][:added]
    if values[order[-1]] <= 1e-12 * values[order[0]]:
        raise ValueError('Requested enrichment exceeds the independent training rank')
    directions = residual @ (vectors[:, order] / np.sqrt(values[order]))
    for _ in range(2):
        directions -= base @ (base.T @ directions)
    directions = np.linalg.qr(directions, mode='reduced')[0]
    space = np.column_stack([base, directions])
    np.testing.assert_allclose(space.T @ space, np.eye(space.shape[1]), rtol=0, atol=1e-8)
    return space, values[::-1]


def diagonalize(space, normalized_stiffness):
    """Rayleigh-Ritz rotation preserves the trained space and full quadratic energy."""
    tangent = space.T @ (normalized_stiffness @ space)
    skew = np.linalg.norm(tangent - tangent.T) / np.linalg.norm(tangent)
    if skew > 1e-11:
        raise ValueError('Candidate tangent is not conservative')
    values, rotation = eigh(.5 * (tangent + tangent.T))
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Nonpositive enriched stiffness')
    return np.sqrt(values), space @ rotation, float(skew)


def static_metrics(basis, omega, stiffness, rhs, exact, component):
    """All three equations contribute to each coupled Galerkin response."""
    approximate = basis @ ((basis.T @ rhs) / omega[:, None]**2)
    error = approximate - exact
    energy = np.sum(exact * (stiffness @ exact), axis=0)
    error_energy = np.maximum(np.sum(error * (stiffness @ error), axis=0), 0)
    result = {'energy_norm_relative': np.sqrt(error_energy / energy).tolist()}
    for axis, label in enumerate(['x', 'y', 'z', 'vector']):
        selected = component == axis if axis < 3 else np.ones(len(component), dtype=bool)
        result[label + '_mass_relative'] = (np.linalg.norm(error[selected], axis=0) /
                                            np.maximum(np.linalg.norm(exact[selected], axis=0), 1e-30)).tolist()
    return result
