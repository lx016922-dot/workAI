"""The same learned graph operations with reusable pointwise work buffers.

All learned tensor entries participate. No coefficients are thresholded, no
constitutive teacher is substituted, and the recurrence is unchanged.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def _activation(g, quadratic, cubic, law, strain, z, linear, nonlinear, second, stress):
    norm = 0.
    for i in range(3):
        for j in range(3):
            value = .5 * (g[i, j] + g[j, i])
            if law:
                product = 0.
                for k in range(3):
                    product += g[k, i] * g[k, j]
                value += .5 * product
            strain[i, j] = value
            norm += value * value
    z[0] = strain[0, 0] / 1e-4
    z[1] = strain[1, 1] / 1e-4
    z[2] = strain[2, 2] / 1e-4
    z[3] = math.sqrt(2) * strain[1, 2] / 1e-4
    z[4] = math.sqrt(2) * strain[0, 2] / 1e-4
    z[5] = math.sqrt(2) * strain[0, 1] / 1e-4
    for a in range(6):
        linear[a] = 0.
        nonlinear[a] = 0.
        for b in range(6):
            linear[a] += quadratic[a, b] * z[b]
        linear[a] *= 2
        if law == 2:
            for b in range(6):
                for c in range(6):
                    nonlinear[a] += 3 * cubic[a, b, c] * z[b] * z[c]
    second[0, 0] = 1e7 * (linear[0] + 1e-4 * nonlinear[0])
    second[1, 1] = 1e7 * (linear[1] + 1e-4 * nonlinear[1])
    second[2, 2] = 1e7 * (linear[2] + 1e-4 * nonlinear[2])
    second[1, 2] = second[2, 1] = 1e7 * (linear[3] + 1e-4 * nonlinear[3]) / math.sqrt(2)
    second[0, 2] = second[2, 0] = 1e7 * (linear[4] + 1e-4 * nonlinear[4]) / math.sqrt(2)
    second[0, 1] = second[1, 0] = 1e7 * (linear[5] + 1e-4 * nonlinear[5]) / math.sqrt(2)
    for i in range(3):
        for j in range(3):
            stress[i, j] = second[i, j]
            if law:
                stress[i, j] = 0.
                for k in range(3):
                    stress[i, j] += (g[i, k] + (1. if i == k else 0.)) * second[k, j]
    a, b = 0., 0.
    for i in range(6):
        a += z[i] * linear[i]
        b += z[i] * nonlinear[i]
    energy = 1e3 * (.5 * a + 1e-4 / 3 * b)
    jac = ((1 + g[0, 0]) * ((1 + g[1, 1]) * (1 + g[2, 2]) - g[1, 2] * g[2, 1])
           - g[0, 1] * (g[1, 0] * (1 + g[2, 2]) - g[1, 2] * g[2, 0])
           + g[0, 2] * (g[1, 0] * g[2, 1] - (1 + g[1, 1]) * g[2, 0]))
    return energy, math.sqrt(norm), jac


@njit(cache=True)
def energy_activation(gradient, quadratic, cubic, law):
    stress, strain, second = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3))
    energy, norm, jac = _activation(gradient, quadratic, cubic, law, strain, np.empty(6), np.empty(6), np.empty(6), second, stress)
    return stress, energy, norm, jac


@njit(cache=True)
def graph_message(current, cells, derivative, inverse, volume, quadratic, cubic, law):
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
                    energy, norm, jac = _activation(g, quadratic, cubic, law, strain, z, linear, nonlinear, second, stress)
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


@njit(cache=True)
def recurrent_forward(cells, derivative, inverse, volume, mass, load, fixed, quadratic, cubic, state_weights,
                      law, dt, subdivisions, frames, pulse_values):
    current, previous, last_external = np.zeros_like(load), np.zeros_like(load), np.zeros_like(load)
    fields = np.zeros((frames, len(mass), 3))
    energy_log = np.zeros((frames, 4))
    work, max_strain, min_j = 0., 0., 1.
    total_steps = (frames - 1) * subdivisions
    for step in range(total_steps + 1):
        message, potential, strain, jac = graph_message(current, cells, derivative, inverse, volume, quadratic, cubic, law)
        if not np.isfinite(potential) or strain > .001 or jac <= 0:
            raise ValueError('Graph recurrence violates strain/J/finite-energy guard.')
        max_strain, min_j = max(max_strain, strain), min(min_j, jac)
        external = load * pulse_values[step]
        scaled = (external - message) / mass[:, None] * dt * dt
        scaled[fixed] = 0.
        work += np.sum(.5 * (external + last_external) * (current - previous))
        if step % subdivisions == 0:
            frame = step // subdivisions
            fields[frame] = current
            velocity = (current - previous + .5 * scaled) / dt
            velocity[fixed] = 0.
            kinetic = np.sum(.5 * mass[:, None] * velocity * velocity)
            energy_log[frame] = np.array([step * dt * 1e6, kinetic, potential, work])
        if step < total_steps:
            next_state = current + state_weights[0] * (current - previous) + state_weights[1] * scaled
            next_state[fixed] = 0.
            previous, current, last_external = current, next_state, external
    return fields, energy_log, max_strain, min_j
