"""Rayleigh–Lamb dispersion of a traction-free isotropic plate.

Characteristic functions (half-thickness h, p^2 = w^2/cL^2 - k^2, q^2 = w^2/cT^2 - k^2):
  symmetric      D_S = (q^2 - k^2)^2 cos(ph) sin(qh) + 4 k^2 p q sin(ph) cos(qh)
  antisymmetric  D_A = (q^2 - k^2)^2 sin(ph) cos(qh) + 4 k^2 p q cos(ph) sin(qh)
Evaluated with complex p, q so every phase-velocity regime uses one formula; D_S / q and
D_A / p are real for real k and have no spurious zero at q = 0 or p = 0. Roots are
bracketed on a slowness grid and refined with Brent's method, then tracked across
frequency. Group velocity is dw/dk along each track.
"""
import math
import numpy as np
from scipy.optimize import brentq

FAMILIES = ('S', 'A')


def bulk_speeds(material):
    rho = material['rho_kg_m3']; lam = material['lambda_pa']; mu = material['mu_pa']
    return math.sqrt((lam + 2 * mu) / rho), math.sqrt(mu / rho)


def rayleigh_speed(cL, cT):
    f = lambda c: (2 - c * c / cT**2)**2 - 4 * math.sqrt(1 - c * c / cL**2) * math.sqrt(1 - c * c / cT**2)
    return brentq(f, .5 * cT, cT * (1 - 1e-12), xtol=1e-12)


def _scaled_trig(w):
    """cos(w)·exp(-|Im w|) and sin(w)·exp(-|Im w|) for complex arrays, without overflow."""
    a = w.real; b = w.imag; decay = np.exp(-2 * np.abs(b)); cosh = (1 + decay) / 2; sinh = np.sign(b) * (1 - decay) / 2
    return np.cos(a) * cosh - 1j * np.sin(a) * sinh, np.sin(a) * cosh + 1j * np.cos(a) * sinh


def characteristic(omega, k, half, cL, cT, family):
    """Real-valued characteristic function for arrays of wavenumber k (rad/m).

    Each product carries exactly one factor in ph and one in qh, so scaling both by
    exp(-|Im|) multiplies the function by a positive number and keeps every zero."""
    k = np.asarray(k, dtype=np.complex128)
    p = np.sqrt(omega**2 / cL**2 - k * k); q = np.sqrt(omega**2 / cT**2 - k * k)
    cos_p, sin_p = _scaled_trig(p * half); cos_q, sin_q = _scaled_trig(q * half)
    if family == 'S': value = ((q * q - k * k)**2 * cos_p * sin_q + 4 * k * k * p * q * sin_p * cos_q) / np.where(q == 0, 1, q)
    elif family == 'A': value = ((q * q - k * k)**2 * sin_p * cos_q + 4 * k * k * p * q * cos_p * sin_q) / np.where(p == 0, 1, p)
    else: raise ValueError('family must be S or A')
    return np.real(value)


def phase_velocities(frequency_hz, thickness_m, cL, cT, family, cp_min=30., cp_max=None, samples=6000):
    """All phase velocities (m/s) of one family at one frequency, sorted ascending."""
    omega = 2 * math.pi * frequency_hz; half = thickness_m / 2; cp_max = cp_max or 60 * cL
    slowness = np.linspace(1 / cp_max, 1 / cp_min, samples); values = characteristic(omega, omega * slowness, half, cL, cT, family)
    scalar = lambda s: float(characteristic(omega, np.array([omega * s]), half, cL, cT, family)[0])
    roots = []
    for i in np.flatnonzero(values[:-1] * values[1:] < 0):
        try: roots.append(1 / brentq(scalar, slowness[i], slowness[i + 1], xtol=1e-14, rtol=1e-12, maxiter=200))
        except ValueError: continue
    return sorted(roots)


def dispersion_curves(thickness_mm, material, fd_max_mhz_mm=6., fd_min_mhz_mm=.05, points=240, families=FAMILIES, cp_min=30.):
    """Tracked modes with fd, frequency, phase velocity, wavenumber and group velocity."""
    cL, cT = bulk_speeds(material); d = thickness_mm * 1e-3; fds = np.linspace(fd_min_mhz_mm, fd_max_mhz_mm, points)
    modes = []
    for family in families:
        tracks = []
        for fd in fds:
            f = fd * 1e6 / thickness_mm; omega = 2 * math.pi * f
            roots = phase_velocities(f, d, cL, cT, family, cp_min=cp_min)
            slowness = [1 / cp for cp in roots]; used = [False] * len(roots)
            # Predict each live track in slowness, match greedily by distance.
            candidates = []
            for t in tracks:
                if t['missed'] > 2: continue
                s_last = 1 / t['cp'][-1]; slope = (s_last - 1 / t['cp'][-2]) / (t['fd'][-1] - t['fd'][-2]) if len(t['fd']) > 1 else 0.
                predicted = s_last + slope * (fd - t['fd'][-1])
                # Relative tolerance: slowness of A0 changes by tens of percent per step at low fd.
                tolerance = max((.3 if len(t['fd']) > 1 else .6) * abs(predicted), .05 / cT)
                for r, s in enumerate(slowness): candidates.append((abs(s - predicted) / tolerance, t, r))
            candidates.sort(key=lambda c: c[0]); assigned = set()
            for distance, t, r in candidates:
                if used[r] or id(t) in assigned or distance > 1.: continue
                t['fd'].append(fd); t['cp'].append(roots[r]); t['missed'] = 0; used[r] = True; assigned.add(id(t))
            for t in tracks:
                if id(t) not in assigned: t['missed'] += 1
            for r, cp in enumerate(roots):
                if not used[r]: tracks.append({'fd': [fd], 'cp': [cp], 'missed': 0})
        tracks = [t for t in tracks if len(t['fd']) >= 3]; tracks.sort(key=lambda t: (t['fd'][0], t['cp'][0]))
        step = fds[1] - fds[0]; groups = []
        for t in tracks:
            # Two branches born at the same fd with nearly the same phase velocity meet at a zero-group-velocity point.
            if groups and len(groups[-1]) == 1 and abs(t['fd'][0] - groups[-1][0]['fd'][0]) <= 1.5 * step and abs(t['cp'][0] - groups[-1][0]['cp'][0]) <= .2 * t['cp'][0]: groups[-1].append(t)
            else: groups.append([t])
        for order, group in enumerate(groups):
            for t in group:
                fd = np.array(t['fd']); cp = np.array(t['cp']); freq = fd * 1e6 / thickness_mm; omega = 2 * math.pi * freq; k = omega / cp
                group_velocity = np.gradient(omega, k) if len(k) > 2 else np.full(len(k), np.nan)
                backward = len(group) > 1 and np.nanmedian(group_velocity) < 0
                modes.append({'family': family, 'order': order, 'name': f'{family}{order}' + ('b' if backward else ''), 'backward_branch': bool(backward),
                              'fd_mhz_mm': fd.tolist(), 'frequency_mhz': (freq * 1e-6).tolist(), 'phase_velocity_m_s': cp.tolist(),
                              'wavenumber_rad_per_mm': (k * 1e-3).tolist(), 'group_velocity_m_s': group_velocity.tolist(),
                              'cutoff_fd_mhz_mm': float(fd[0]) if order > 0 else 0., 'zero_group_velocity_point_fd_mhz_mm': float(fd[0]) if len(group) > 1 else None})
    return {'schema': 'workai.lamb_dispersion.v1', 'thickness_mm': thickness_mm, 'material': material, 'longitudinal_speed_m_s': cL, 'shear_speed_m_s': cT,
            'rayleigh_speed_m_s': rayleigh_speed(cL, cT), 'plate_speed_m_s': 2 * cT * math.sqrt(1 - cT**2 / cL**2), 'fd_range_mhz_mm': [fd_min_mhz_mm, fd_max_mhz_mm], 'modes': modes}


def mode_at(curves, name, fd_mhz_mm):
    """Interpolated phase and group velocity of a named mode at fd, or None outside its track."""
    mode = next((m for m in curves['modes'] if m['name'] == name), None)
    if mode is None: raise ValueError(f'No mode named {name}.')
    fd = np.array(mode['fd_mhz_mm'])
    if not fd[0] <= fd_mhz_mm <= fd[-1]: return None
    return {'name': name, 'fd_mhz_mm': fd_mhz_mm, 'phase_velocity_m_s': float(np.interp(fd_mhz_mm, fd, mode['phase_velocity_m_s'])),
            'group_velocity_m_s': float(np.interp(fd_mhz_mm, fd, mode['group_velocity_m_s']))}


def mode_shape(frequency_hz, cp, thickness_m, cL, cT, family, z_m):
    """Through-thickness displacement shape (z from the mid-plane).

    Complex shapes follow the exp(+i w t) time convention of `dispersion.analysis.harmonic_coefficients`
    (a wave towards +x is U(z) exp(-i k x) exp(+i w t)), so measured spectral coefficients can be
    decomposed on them directly. `ux_quadrature` and `uz_inphase` are the real profiles."""
    omega = 2 * math.pi * frequency_hz; k = omega / cp; h = thickness_m / 2; z = np.asarray(z_m, dtype=float)
    p = np.sqrt(complex(omega**2 / cL**2 - k * k)); q = np.sqrt(complex(omega**2 / cT**2 - k * k))
    if family == 'S':
        denominator = (k * k - q * q) * np.sin(q * h); numerator = 2j * k * p * np.sin(p * h)
        a, b = (1., numerator / denominator) if abs(denominator) > 1e-9 * max(abs(numerator), 1e-300) else (0., 1.)
        ux = 1j * k * a * np.cos(p * z) + q * b * np.cos(q * z); uz = -p * a * np.sin(p * z) - 1j * k * b * np.sin(q * z)
    elif family == 'A':
        denominator = (k * k - q * q) * np.cos(q * h); numerator = -2j * k * p * np.cos(p * h)
        a, b = (1., numerator / denominator) if abs(denominator) > 1e-9 * max(abs(numerator), 1e-300) else (0., 1.)
        ux = 1j * k * a * np.sin(p * z) - q * b * np.sin(q * z); uz = p * a * np.cos(p * z) - 1j * k * b * np.cos(q * z)
    else: raise ValueError('family must be S or A')
    # The derivation above uses exp(i(kx - wt)); conjugate for the exp(+iwt) convention of the analysis module.
    ux = np.conj(ux); uz = np.conj(uz)
    # Lamb modes have u_x and u_z in quadrature; rotate so u_z is real (or u_x imaginary) and normalize.
    reference = uz[np.argmax(np.abs(uz))] if np.max(np.abs(uz)) >= np.max(np.abs(ux)) else ux[np.argmax(np.abs(ux))] / 1j
    phase = np.exp(-1j * np.angle(reference)); ux = ux * phase; uz = uz * phase
    scale = max(np.max(np.abs(ux)), np.max(np.abs(uz))); ux = ux / scale; uz = uz / scale
    residual = float(max(np.max(np.abs(ux.real)), np.max(np.abs(uz.imag))))
    return {'z_m': z.tolist(), 'ux_quadrature': ux.imag.tolist(), 'uz_inphase': uz.real.tolist(), 'complex_ux': ux, 'complex_uz': uz, 'quadrature_residual': residual,
            'wavenumber_rad_per_m': k, 'family': family, 'time_convention': 'exp(+i omega t), wave towards +x is exp(-i k x)'}


def main():
    import argparse, json
    from solid.schema import MATERIAL
    parser = argparse.ArgumentParser(description='Rayleigh–Lamb dispersion curves of an aluminum (or given) plate.')
    parser.add_argument('--thickness-mm', type=float, required=True); parser.add_argument('--fd-max', type=float, default=6.); parser.add_argument('--points', type=int, default=240)
    parser.add_argument('--material-json', type=str, help='JSON file with rho_kg_m3, lambda_pa, mu_pa (defaults to the aluminum card)')
    parser.add_argument('--output', type=str, required=True); parser.add_argument('--csv', type=str)
    args = parser.parse_args()
    material = json.load(open(args.material_json)) if args.material_json else MATERIAL
    curves = dispersion_curves(args.thickness_mm, material, fd_max_mhz_mm=args.fd_max, points=args.points)
    open(args.output, 'w', encoding='utf-8').write(json.dumps(curves, indent=1) + '\n')
    if args.csv:
        with open(args.csv, 'w', encoding='utf-8') as handle:
            handle.write('mode,fd_mhz_mm,frequency_mhz,phase_velocity_m_s,group_velocity_m_s,wavenumber_rad_per_mm\n')
            for m in curves['modes']:
                for row in zip(m['fd_mhz_mm'], m['frequency_mhz'], m['phase_velocity_m_s'], m['group_velocity_m_s'], m['wavenumber_rad_per_mm']): handle.write(m['name'] + ',' + ','.join(f'{v:.9g}' for v in row) + '\n')
    print(json.dumps({'modes': [m['name'] for m in curves['modes']], 'rayleigh_speed_m_s': curves['rayleigh_speed_m_s'], 'plate_speed_m_s': curves['plate_speed_m_s']}))


if __name__ == '__main__': main()
