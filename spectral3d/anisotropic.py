"""GLL internal force for any hyperelastic material given by its second- and third-order constants.

W = ½ c2:E:E + ⅙ c3:E:E:E in Mandel coordinates (physics.materials), Green–Lagrange strain, P = F·S.
law 0: small strain, P = S, quadratic energy only; law 1: Green strain, quadratic energy (StVK form);
law 2: Green strain, quadratic and cubic energy. Written independently of the network's energy activation
so the two can be checked against each other.
"""
import math
import numpy as np
from numba import njit

R2 = math.sqrt(2.)


@njit(cache=True)
def constitutive_mandel(g, c2, c3, law):
    f = g + np.eye(3); e = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            e[i, j] = .5 * (g[i, j] + g[j, i])
            if law:
                for k in range(3): e[i, j] += .5 * g[k, i] * g[k, j]
    z = np.array([e[0, 0], e[1, 1], e[2, 2], R2 * e[1, 2], R2 * e[0, 2], R2 * e[0, 1]])
    s = c2 @ z; energy = .5 * np.dot(z, s)
    if law == 2:
        q = np.zeros(6)
        for a in range(6):
            for b in range(6):
                for c in range(6): q[a] += c3[a, b, c] * z[b] * z[c]
        s = s + .5 * q; energy += np.dot(z, q) / 6.
    S = np.array([[s[0], s[5] / R2, s[4] / R2], [s[5] / R2, s[1], s[3] / R2], [s[4] / R2, s[3] / R2, s[2]]])
    p = S if law == 0 else f @ S
    jac = (f[0, 0] * (f[1, 1] * f[2, 2] - f[1, 2] * f[2, 1]) - f[0, 1] * (f[1, 0] * f[2, 2] - f[1, 2] * f[2, 0]) + f[0, 2] * (f[1, 0] * f[2, 1] - f[1, 1] * f[2, 0]))
    strain = 0.
    for i in range(3):
        for j in range(3): strain += e[i, j] * e[i, j]
    return p, energy, math.sqrt(strain), jac


@njit(cache=True)
def internal_force(u, elements, derivative, sizes, volume_weight, c2, c3, law):
    """Axis-aligned elements (plates, notches)."""
    force = np.zeros_like(u); energy = 0.; max_strain = 0.; min_j = 1e100; n = elements.shape[1]
    stress = np.empty((n, n, n, 3, 3)); local = np.empty((n, n, n, 3)); g = np.empty((3, 3))
    for e in range(len(elements)):
        ids = elements[e]
        for i in range(n):
            for j in range(n):
                for k in range(n): local[i, j, k] = u[ids[i, j, k]]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    g[:] = 0.
                    for a in range(n):
                        for c in range(3):
                            g[c, 0] += derivative[i, a] * local[a, j, k, c] * 2 / sizes[e, 0]
                            g[c, 1] += derivative[j, a] * local[i, a, k, c] * 2 / sizes[e, 1]
                            g[c, 2] += derivative[k, a] * local[i, j, a, c] * 2 / sizes[e, 2]
                    p, w, strain, jac = constitutive_mandel(g, c2, c3, law)
                    energy += w * volume_weight[e, i, j, k]; max_strain = max(max_strain, strain); min_j = min(min_j, jac)
                    stress[i, j, k] = p * volume_weight[e, i, j, k]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    node = ids[i, j, k]
                    for a in range(n):
                        for c in range(3):
                            force[node, c] += derivative[a, i] * stress[a, j, k, c, 0] * 2 / sizes[e, 0]
                            force[node, c] += derivative[a, j] * stress[i, a, k, c, 1] * 2 / sizes[e, 1]
                            force[node, c] += derivative[a, k] * stress[i, j, a, c, 2] * 2 / sizes[e, 2]
    return force, energy, max_strain, min_j


@njit(cache=True)
def curved_internal_force(u, elements, derivative, inverse, volume_weight, c2, c3, law):
    """Isoparametric elements (pipes, plates with holes)."""
    force = np.zeros_like(u); energy = 0.; max_strain = 0.; min_j = 1e100; n = elements.shape[1]
    flux = np.empty((n, n, n, 3, 3)); local = np.empty((n, n, n, 3)); reference_gradient = np.empty((3, 3))
    for e in range(len(elements)):
        ids = elements[e]
        for i in range(n):
            for j in range(n):
                for k in range(n): local[i, j, k] = u[ids[i, j, k]]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    reference_gradient[:] = 0.
                    for a in range(n):
                        for c in range(3):
                            reference_gradient[c, 0] += derivative[i, a] * local[a, j, k, c]
                            reference_gradient[c, 1] += derivative[j, a] * local[i, a, k, c]
                            reference_gradient[c, 2] += derivative[k, a] * local[i, j, a, c]
                    inv = inverse[e, i, j, k]; g = reference_gradient @ inv; p, w, strain, jac = constitutive_mandel(g, c2, c3, law)
                    energy += w * volume_weight[e, i, j, k]; max_strain = max(max_strain, strain); min_j = min(min_j, jac)
                    flux[i, j, k] = (p @ inv.T) * volume_weight[e, i, j, k]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    node = ids[i, j, k]
                    for a in range(n):
                        for c in range(3):
                            force[node, c] += derivative[a, i] * flux[a, j, k, c, 0] + derivative[a, j] * flux[i, a, k, c, 1] + derivative[a, k] * flux[i, j, a, c, 2]
    return force, energy, max_strain, min_j
