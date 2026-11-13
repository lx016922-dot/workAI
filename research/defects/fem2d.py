"""Independent 2D plane-strain reference for the defect cases: no code shared with the GLL network.

Discretization: bilinear (Q1) quadrilaterals on a uniform grid, 2 × 2 Gauss stiffness, row-sum lumped mass,
explicit central differences. Small-strain linear elasticity (the cases compare the linear control).

* Crack / delamination: the grid nodes on the defect face are duplicated (the + side elements take the copies);
  front nodes inside the material stay shared, edges on an outer face open.
* Contact (closed defects): exact frictionless unilateral constraint by the forward-increment Lagrange-multiplier
  method (Carpenter, Taylor & Katona 1991): after each predictor step, a node pair that would interpenetrate is
  moved apart along the normal with momentum conserved, so the gap is exactly zero and the contact force is
  whatever it takes. There is no penalty stiffness, unlike the network's interface spring.
* Excitation: uniform traction on the x = 0 end face (consistent nodal forces), any direction in the x–z plane.
* Absorbing layer: the same mass-proportional damping field as the scene (cubic ramp, design attenuation),
  m ü + m α u̇ = f, central differences.

Units: SI internally; mm, μs, MHz, MPa at the interface.
"""
import math
import numpy as np
from numba import njit


def q1_stiffness(hx, hz, lam, mu):
    """8 × 8 plane-strain stiffness of an hx × hz rectangle (dofs u0x, u0z, u1x, u1z, …; nodes counter-clockwise from (0,0))."""
    D = np.array([[lam + 2 * mu, lam, 0.], [lam, lam + 2 * mu, 0.], [0., 0., mu]])
    K = np.zeros((8, 8)); g = 1 / math.sqrt(3)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for xi in (-g, g):
        for eta in (-g, g):
            B = np.zeros((3, 8))
            for a, (sa, ta) in enumerate(corners):
                dx = sa * (1 + ta * eta) / 4 * 2 / hx; dz = ta * (1 + sa * xi) / 4 * 2 / hz
                B[0, 2 * a] = dx; B[1, 2 * a + 1] = dz; B[2, 2 * a] = dz; B[2, 2 * a + 1] = dx
            K += B.T @ D @ B * hx * hz / 4
    return K


def build(L, T, h, splits):
    """Grid nodes, elements and lumped masses per unit density; splits = [{'axis': 'x'|'z', 'at': mm, 'from': mm, 'to': mm, 'contact': bool}]."""
    nx, nz = round(L / h), round(T / h)
    assert abs(nx * h - L) < 1e-9 and abs(nz * h - T) < 1e-9, 'grid must fit the plate'
    node = lambda i, j: i * (nz + 1) + j
    xs, zs = np.meshgrid(np.arange(nx + 1) * h, np.arange(nz + 1) * h, indexing='ij')
    points = list(np.stack([xs.ravel(), zs.ravel()], axis=1))
    elements = np.array([[node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)] for i in range(nx) for j in range(nz)], dtype=np.int64)
    cell = lambda i, j: i * nz + j
    pairs_a, pairs_b, normals = [], [], []
    for s in splits:
        k = lambda v: round(v / h)
        if s['axis'] == 'x':   # vertical face x = at, z from .. to
            i, j0, j1 = k(s['at']), k(s['from']), k(s['to'])
            line = [(i, j) for j in range(j0, j1 + 1) if j0 < j < j1 or (j == j0 and j0 == 0) or (j == j1 and j1 == nz)]
            plus = [cell(i, j) for j in range(j0, j1)]; normal = (1., 0.)
        else:                  # horizontal face z = at, x from .. to
            j, i0, i1 = k(s['at']), k(s['from']), k(s['to'])
            line = [(i, j) for i in range(i0, i1 + 1) if i0 < i < i1 or (i == i0 and i0 == 0) or (i == i1 and i1 == nx)]
            plus = [cell(i, j) for i in range(i0, i1)]; normal = (0., 1.)
        copies = {}
        for i, j in line:
            old = node(i, j); copies[old] = len(points); points.append(points[old].copy())
        for c in plus:
            e = elements[c]
            for old, new in copies.items(): e[e == old] = new
        if s.get('contact'):
            for old, new in copies.items(): pairs_a.append(old); pairs_b.append(new); normals.append(normal)
    points = np.asarray(points) * 1e-3
    mass = np.zeros(len(points)); np.add.at(mass, elements.ravel(), (h * 1e-3) ** 2 / 4)
    return {'points': points, 'elements': elements, 'mass_per_rho': mass, 'nx': nx, 'nz': nz, 'h': h,
            'contact_a': np.asarray(pairs_a, dtype=np.int64), 'contact_b': np.asarray(pairs_b, dtype=np.int64), 'contact_n': np.asarray(normals, dtype=np.float64).reshape(-1, 2)}


@njit(cache=True)
def run(elements, K, mass, alpha, load, signal, dt, steps, every, probes, ca, cb, cn):
    """Central differences; returns probe displacements (frames × probes × 2) and the energy log (frames × 3: kinetic, strain, work)."""
    n = len(mass); u = np.zeros((n, 2)); prev = np.zeros((n, 2)); f = np.zeros((n, 2))
    frames = steps // every + 1; out = np.zeros((frames, len(probes), 2)); energy = np.zeros((frames, 3)); work = 0.; last_ext = np.zeros((n, 2))
    for step in range(steps + 1):
        f[:] = 0.; strain = 0.
        for e in range(len(elements)):
            for a in range(4):
                na = elements[e, a]
                for b in range(4):
                    nb = elements[e, b]
                    for p in range(2):
                        for q in range(2):
                            f[na, p] += K[2 * a + p, 2 * b + q] * u[nb, q]
        for i in range(n): strain += .5 * (f[i, 0] * u[i, 0] + f[i, 1] * u[i, 1])
        ext = load * signal[step]
        for i in range(n): work += .5 * ((ext[i, 0] + last_ext[i, 0]) * (u[i, 0] - prev[i, 0]) + (ext[i, 1] + last_ext[i, 1]) * (u[i, 1] - prev[i, 1]))
        if step % every == 0:
            kin = 0.
            for i in range(n):
                for p in range(2):
                    v = (u[i, p] - prev[i, p]) / dt; kin += .5 * mass[i] * v * v
            k = step // every; energy[k, 0] = kin; energy[k, 1] = strain; energy[k, 2] = work
            for r in range(len(probes)):
                out[k, r, 0] = u[probes[r], 0]; out[k, r, 1] = u[probes[r], 1]
        if step == steps: break
        new = np.empty((n, 2))
        for i in range(n):
            c = .5 * alpha[i] * dt
            for p in range(2):
                new[i, p] = (2 * u[i, p] - (1 - c) * prev[i, p] + dt * dt * (ext[i, p] - f[i, p]) / mass[i]) / (1 + c)
        for k in range(len(ca)):   # forward-increment Lagrange multiplier: exact, frictionless, momentum conserving
            a = ca[k]; b = cb[k]; gap = (new[b, 0] - new[a, 0]) * cn[k, 0] + (new[b, 1] - new[a, 1]) * cn[k, 1]
            if gap < 0.:
                total = mass[a] + mass[b]
                for p in range(2):
                    new[a, p] += gap * mass[b] / total * cn[k, p]; new[b, p] -= gap * mass[a] / total * cn[k, p]
        prev = u; u = new; last_ext = ext
    return out, energy


def simulate(case, h, cfl=.35):
    """case: {'L','T','rho','lam','mu','direction','amplitude_mpa','f_mhz','cycles','duration_us','sample_us','absorber_mm','splits','receivers' [(name,x,z)]}."""
    m = build(case['L'], case['T'], h, case['splits'])
    lam, mu, rho = case['lam'], case['mu'], case['rho']; cl = math.sqrt((lam + 2 * mu) / rho)
    K = q1_stiffness(h * 1e-3, h * 1e-3, lam, mu); mass = m['mass_per_rho'] * rho
    sample = case['sample_us'] * 1e-6; every = math.ceil(sample / (cfl * h * 1e-3 / cl)); dt = sample / every
    steps = round(case['duration_us'] / case['sample_us']) * every
    # end-face traction: consistent nodal forces of a uniform traction on x = 0 (per unit depth)
    load = np.zeros((len(mass), 2)); d = np.asarray(case['direction'], dtype=float); d = d / np.linalg.norm(d)
    for j in range(m['nz'] + 1):
        w = h * 1e-3 * (.5 if j in (0, m['nz']) else 1.); load[j] = d * case['amplitude_mpa'] * 1e6 * w
    t = np.arange(steps + 1) * dt * 1e6; f = case['f_mhz']; span = case['cycles'] / f
    signal = np.where(t <= span, .5 * (1 - np.cos(2 * np.pi * f * t / case['cycles'])) * np.sin(2 * np.pi * f * t), 0.)
    alpha = np.zeros(len(mass)); x = m['points'][:, 0] * 1e3; Lab = case['absorber_mm']
    if Lab:
        peak = 4 * cl * case.get('attenuation_db', 60.) * math.log(10) / (20 * Lab * 1e-3)
        alpha = peak * np.clip((Lab - (case['L'] - x)) / Lab, 0, 1) ** 3
    probes = np.array([round(rx / h) * (m['nz'] + 1) + round(rz / h) for _, rx, rz in case['receivers']], dtype=np.int64)
    out, energy = run(m['elements'], K, mass, alpha, load, signal, dt, steps, every, probes, m['contact_a'], m['contact_b'], m['contact_n'])
    return {'times_us': np.arange(out.shape[0]) * case['sample_us'], 'probes_m': out, 'energy': energy, 'dt_s': dt, 'steps': steps, 'nodes': len(mass),
            'elements': len(m['elements']), 'contact_pairs': len(m['contact_a'])}
