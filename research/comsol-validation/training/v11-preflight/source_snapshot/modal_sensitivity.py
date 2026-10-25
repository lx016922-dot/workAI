"""Exact eigenfrequency sensitivities of the conservative learned graph tangent.

Only the offline training gradient is changed. The force, mass and constitutive
energy are the same ones evaluated by the deployed graph message function.
"""
import numpy as np

from .inference import inverse_mapping


def modal_gradients(mesh, derivative, fields):
    """Physical displacement gradients at every element quadrature node."""
    values = fields[:, mesh['elements']]
    reference = np.stack([
        np.einsum('ia,meajkc->meijkc', derivative, values, optimize=True),
        np.einsum('ja,meiakc->meijkc', derivative, values, optimize=True),
        np.einsum('ka,meijac->meijkc', derivative, values, optimize=True),
    ], axis=-1)
    return np.einsum('meijkcA,eijkAb->meijkcb', reference, inverse_mapping(mesh), optimize=True)


def mandel_strain(gradient):
    return np.stack([gradient[..., 0, 0], gradient[..., 1, 1], gradient[..., 2, 2],
                     (gradient[..., 1, 2]+gradient[..., 2, 1])/np.sqrt(2),
                     (gradient[..., 0, 2]+gradient[..., 2, 0])/np.sqrt(2),
                     (gradient[..., 0, 1]+gradient[..., 1, 0])/np.sqrt(2)], axis=-1)


def frequency_jacobian(mesh, derivative, quadratic, frequency, fields, basis):
    """Hellmann--Feynman derivative; valid for isolated mass-orthogonal modes.

    Mode crossings/near degeneracies need subspace tracking by the caller. No
    COMSOL field or frequency enters this calculation.
    """
    frequency = np.asarray(frequency)
    if fields.shape != (len(frequency), len(mesh['points']), 3) or np.any(frequency <= 0):
        raise ValueError('Positive frequencies and matching nodal eigenvectors required')
    strain = mandel_strain(modal_gradients(mesh, derivative, fields))
    elasticity = 2e11*(quadratic+quadratic.T)
    stress = np.einsum('...r,rs->...s', strain, elasticity, optimize=True)
    norm = np.einsum('mnc,mnc,n->m', fields, fields, mesh['mass'])
    if not np.all(norm > 0):
        raise ValueError('Positive modal mass required')
    result = []
    for direction in basis:
        changed = mandel_strain(modal_gradients(mesh, direction, fields))
        eigenvalue_derivative = np.einsum('meijkr,meijkr,eijk->m', stress, changed,
                                         mesh['volume_weight'], optimize=True)/norm
        result.append(eigenvalue_derivative/(8*np.pi**2*frequency))
    return np.stack(result, axis=-1)
