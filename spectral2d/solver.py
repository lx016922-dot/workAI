"""Matrix-free GLL plane-strain dynamics with the three-dimensional constitutive law.

u = (u_x, u_z)(x, z, t); the displacement gradient is embedded in a 3x3 matrix
with zero y row and column, so Green strain, Murnaghan stress, energy and J are
exactly those of `spectral3d.solver.constitutive` under plane strain. Explicit
velocity Verlet with the same version-2 excitation model (several sources,
arbitrary signals, symmetry constraints, velocity-implicit absorbing damping).
"""
import math
import time
import numpy as np
from numba import njit
from spectral3d.solver import constitutive
from physics.signals import evaluate
from .mesh import interpolation


@njit(cache=True)
def internal_force(u, elements, derivative, sizes, area_weight, constants, law):
    force = np.zeros_like(u); energy = 0.; max_strain = 0.; min_j = 1e100; n = elements.shape[1]
    stress = np.empty((n, n, 3, 3)); local = np.empty((n, n, 2)); g = np.zeros((3, 3))
    for e in range(len(elements)):
        ids = elements[e]
        for i in range(n):
            for j in range(n):
                local[i, j, 0] = u[ids[i, j], 0]; local[i, j, 1] = u[ids[i, j], 1]
        for i in range(n):
            for j in range(n):
                for r in range(3):
                    for c in range(3): g[r, c] = 0.
                for a in range(n):
                    for c in range(2):
                        row = 2 * c   # component 0 -> x (row 0), component 1 -> z (row 2)
                        g[row, 0] += derivative[i, a] * local[a, j, c] * 2 / sizes[e, 0]
                        g[row, 2] += derivative[j, a] * local[i, a, c] * 2 / sizes[e, 1]
                p, w, strain, jac = constitutive(g, constants, law)
                energy += w * area_weight[e, i, j]; max_strain = max(max_strain, strain); min_j = min(min_j, jac)
                for r in range(3):
                    for c in range(3): stress[i, j, r, c] = p[r, c] * area_weight[e, i, j]
        for i in range(n):
            for j in range(n):
                node = ids[i, j]
                for a in range(n):
                    for c in range(2):
                        row = 2 * c
                        force[node, c] += derivative[a, i] * stress[a, j, row, 0] * 2 / sizes[e, 0]
                        force[node, c] += derivative[a, j] * stress[i, a, row, 2] * 2 / sizes[e, 1]
    return force, energy, max_strain, min_j


def solve(scene, mesh, excitation, receivers, cfl=.3, keep_fields=False, callback=None):
    """Per-unit-width plane-strain transient; `receivers` are dicts with point_mm and direction."""
    if not .05 <= cfl <= .35: raise ValueError('Use CFL 0.05–0.35.')
    start = time.monotonic(); material = scene['material']; mass = mesh['mass']
    constants = np.array([material[k] for k in ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']]); law = {'linear': 0, 'stvk': 1, 'murnaghan': 2}[scene['constitutive']]
    speed = math.sqrt((material['lambda_pa'] + 2 * material['mu_pa']) / material['rho_kg_m3'])
    sample = scene['acquisition']['sample_step_us'] * 1e-6; subdivisions = math.ceil(sample / (cfl * mesh['min_node_spacing'] / (speed * 1.05))); dt = sample / subdivisions
    frames = round(scene['acquisition']['duration_us'] / scene['acquisition']['sample_step_us']) + 1; steps = (frames - 1) * subdivisions
    if steps > 500000 or steps * mesh['elements'].size > 5_000_000_000: raise ValueError('Plane-strain compute budget exceeded; coarsen the mesh or shorten the record.')
    if keep_fields and frames * len(mass) * 16 > 1024**3: raise ValueError('Field exceeds 1 GiB; store receivers or lines instead.')
    loads = excitation['loads']; constrained = excitation['constrained']; alpha = excitation['alpha']
    times = np.arange(steps + 1) * dt * 1e6; values = [evaluate(src, times) for _, src in loads]
    damping = None if alpha is None else mass * alpha
    fields = np.zeros((frames, len(mass), 2)) if keep_fields else None
    taps = [interpolation(mesh, rx['point_mm']) for rx in receivers]; probes = np.zeros((frames, len(taps), 2))
    u = np.zeros((len(mass), 2)); v = u.copy(); acc = u.copy(); previous_ext = u.copy(); work = 0.; dissipated = 0.; energies = []; max_strain = 0.; min_j = 1.
    for step in range(1, steps + 1):
        next_u = u + dt * v + .5 * dt * dt * acc; next_u[constrained] = 0
        fint, energy, strain, jac = internal_force(next_u, mesh['elements'], mesh['derivative'], mesh['size'], mesh['area_weight'], constants, law)
        if not np.isfinite(energy) or strain > .001 or jac <= 0: raise ValueError(f'Elastic guard failed: strain={strain}, J={jac}')
        max_strain = max(max_strain, strain); min_j = min(min_j, jac)
        ext = loads[0][0] * values[0][step]
        for k in range(1, len(loads)): ext = ext + loads[k][0] * values[k][step]
        if damping is None:
            next_acc = (ext - fint) / mass[:, None]; next_acc[constrained] = 0
            next_v = v + .5 * dt * (acc + next_acc); next_v[constrained] = 0
        else:
            undamped = (ext - fint) / mass[:, None]
            next_v = (v + .5 * dt * (acc + undamped)) / (1 + .5 * dt * alpha)[:, None]; next_v[constrained] = 0
            next_acc = undamped - alpha[:, None] * next_v; next_acc[constrained] = 0
            dissipated += float(np.sum(damping[:, None] * .5 * (v + next_v) * (next_u - u)))
        work += float(np.sum(.5 * (ext + previous_ext) * (next_u - u))); u, v, acc = next_u, next_v, next_acc; previous_ext = ext
        if step % subdivisions == 0:
            frame = step // subdivisions
            if fields is not None: fields[frame] = u
            for i, (ids, w) in enumerate(taps): probes[frame, i] = w @ u[ids]
            energies.append([step * dt * 1e6, float(np.sum(.5 * mass[:, None] * v * v)), energy, work] + ([dissipated] if damping is not None else []))
            if callback and (frame % 40 == 0 or frame == frames - 1): callback({'frame': frame, 'frames': frames, 'elapsed_s': time.monotonic() - start})
    energies = np.array(energies); stored = energies[:, 1] + energies[:, 2] + (energies[:, 4] if damping is not None else 0.)
    scale = max(np.max(abs(energies[:, 3])), np.max(abs(stored)), 1e-30); balance = float(np.max(abs(stored - energies[:, 3])) / scale)
    diagnostics = {'method': 'cartesian_gll_plane_strain_total_lagrangian_velocity_verlet_v1', 'dimension': '2d_plane_strain', 'units': 'forces, masses and energies per metre of width',
                   'degree': mesh['degree'], 'h_mm': mesh['h_mm'], 'nodes': len(mass), 'elements': len(mesh['elements']), 'quadrature_points': int(mesh['elements'].size),
                   'internal_steps': steps, 'internal_step_us': dt * 1e6, 'min_node_spacing_mm': mesh['min_node_spacing'] * 1000, 'loaded_length_mm': excitation['length_m'] * 1e3,
                   'section_area_mm2': float(mesh['area_weight'].sum() * 1e6), 'max_strain': max_strain, 'min_J': min_j, 'external_work_balance_relative': balance,
                   'energy_balance_within_1_percent': balance <= .01, 'dissipated_energy_j_per_m': dissipated,
                   'energy_columns': ['time_us', 'kinetic_j_per_m', 'strain_j_per_m', 'external_work_j_per_m'] + (['dissipated_j_per_m'] if damping is not None else []),
                   'signal_peaks': [float(np.max(abs(x))) for x in values], 'elapsed_s': time.monotonic() - start, 'calculation_dtype': 'float64', 'output_dtype': 'float64',
                   'accepted_for_solver_routing': False, 'spatial_convergence_verified': False, 'temporal_convergence_verified': False, 'energy_samples': energies.tolist()}
    diagnostics.update(excitation['summary'])
    return {'probes_m': probes, 'fields_m': fields, 'times_us': np.arange(frames) * scene['acquisition']['sample_step_us'], 'diagnostics': diagnostics}
