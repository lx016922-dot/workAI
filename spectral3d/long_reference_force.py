"""Analytic validation force, checked against the original spectral solver.

Tensor-product assembly uses reusable buffers. The constitutive coefficients
are material constants, never learned network tensors.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True, inline='always')
def activation(g, constants, law, e, second, stress):
    lam, mu, l, m, n = constants
    tr, tr2, tr3 = 0., 0., 0.
    for i in range(3):
        for j in range(3):
            e[i,j] = .5*(g[i,j]+g[j,i])
            if law:
                for k in range(3): e[i,j] += .5*g[k,i]*g[k,j]
            tr2 += e[i,j]*e[i,j]
        tr += e[i,i]
    a = n if law == 2 else 0.
    b = m-n/2 if law == 2 else 0.
    c = l-m+n/2 if law == 2 else 0.
    for i in range(3):
        for j in range(3):
            squared = 0.
            for k in range(3): squared += e[i,k]*e[k,j]
            tr3 += squared*e[j,i]
            second[i,j] = (2*mu+2*b*tr)*e[i,j]+a*squared
            if i == j: second[i,j] += lam*tr+b*tr2+c*tr*tr
    for i in range(3):
        for j in range(3):
            stress[i,j] = second[i,j]
            if law:
                for k in range(3): stress[i,j] += g[i,k]*second[k,j]
    energy = .5*lam*tr*tr+mu*tr2+a/3*tr3+b*tr*tr2+c/3*tr**3
    jac = ((1+g[0,0])*((1+g[1,1])*(1+g[2,2])-g[1,2]*g[2,1])
           -g[0,1]*(g[1,0]*(1+g[2,2])-g[1,2]*g[2,0])
           +g[0,2]*(g[1,0]*g[2,1]-(1+g[1,1])*g[2,0]))
    return energy, math.sqrt(tr2), jac

@njit(cache=True)
def internal_force(current, cells, derivative, inverse, volume, constants, law):
    message = np.zeros_like(current)
    potential, strain_max, jac_min = 0., 0., 1.
    n = cells.shape[1]
    local, flux = np.empty((n, n, n, 3)), np.empty((n, n, n, 3, 3))
    ref, g = np.empty((3, 3)), np.empty((3, 3))
    stress, strain, second = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3))
    z, linear, nonlinear = np.empty(6), np.empty(6), np.empty(6)
    for element in range(len(cells)):
        ids = cells[element]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for axis in range(3):
                        local[i, j, k, axis] = current[ids[i, j, k], axis]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ref[:] = 0.
                    for neighbor in range(n):
                        for axis in range(3):
                            ref[axis, 0] += derivative[i, neighbor] * local[neighbor, j, k, axis]
                            ref[axis, 1] += derivative[j, neighbor] * local[i, neighbor, k, axis]
                            ref[axis, 2] += derivative[k, neighbor] * local[i, j, neighbor, axis]
                    transform = inverse[element, i, j, k]
                    for a in range(3):
                        for b in range(3):
                            g[a, b] = 0.
                            for c in range(3):
                                g[a, b] += ref[a, c] * transform[c, b]
                    energy, norm, jac = activation(g, constants, law, strain, second, stress)
                    for a in range(3):
                        for b in range(3):
                            value = 0.
                            for c in range(3):
                                value += stress[a, c] * transform[b, c]
                            flux[i, j, k, a, b] = value * volume[element, i, j, k]
                    potential += energy * volume[element, i, j, k]
                    strain_max = max(strain_max, norm)
                    jac_min = min(jac_min, jac)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    node = ids[i, j, k]
                    for neighbor in range(n):
                        for axis in range(3):
                            message[node, axis] += (derivative[neighbor, i] * flux[neighbor, j, k, axis, 0]
                                + derivative[neighbor, j] * flux[i, neighbor, k, axis, 1]
                                + derivative[neighbor, k] * flux[i, j, neighbor, axis, 2])
    return message, potential, strain_max, jac_min


