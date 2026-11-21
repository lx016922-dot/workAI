"""A synchronous pair of Lamb modes that is not on the cp = cL family: found numerically, with the growth rate of its second harmonic.

`theory_shg_lamb.pair` knows where its pairs are (cp = cL exactly). A pair like an antisymmetric primary with a symmetric
second harmonic sits where two branches happen to meet the condition ω₂(2k) = 2ω₁(k); `synchronous` finds that point from
a first guess and returns the same quantities as `pair`, `table` the same as `burst_table`, so the analysis of the first pair
(`analyze_s1s2.analyse`) applies unchanged.

    python theory_pair.py        # the pair used in pair_a_s.py and its self-checks
"""
import math
import numpy as np
from theory_shg_lamb import Plate


def parity(plate, U):
    """+1 for a symmetric Lamb mode (uₓ even in z), −1 for an antisymmetric one."""
    ux = U[0::2]; return 1 if abs(ux[-1] + ux[0]) > abs(ux[-1] - ux[0]) else -1


def branch(plate, k, omega, sign):
    """The mode of the given parity at wavenumber k nearest to ω: (ω, shape, group velocity)."""
    w, U = plate.modes(k); best = None
    for i in np.argsort(np.abs(w - omega))[:12]:
        if parity(plate, U[:, i]) == sign: best = int(i); break
    u = U[:, best]; _, dK = plate.stiffness(k)
    return float(w[best]), u, float(np.real(u.conj() @ dK @ u) / (2 * w[best] * np.real(u.conj() @ plate.M @ u)))


def synchronous(thickness, rho, lam, mu, l, m, n, f_guess, cp_guess, primary=-1, elements=24, degree=8):
    """The pair primary(f) → symmetric(2f) near a guess: frequency, wavenumber, velocities and the growth κ₂ of the surface uₓ."""
    plate = Plate(thickness, rho, lam, mu, elements, degree); omega = 2 * math.pi * f_guess; k = omega / cp_guess
    def mismatch(k, omega):
        w1, _, _ = branch(plate, k, omega, primary); w2, _, _ = branch(plate, 2 * k, 2 * w1, 1); return w2 - 2 * w1, w1
    step = 1e-4 * k; d0, w = mismatch(k, omega)
    for _ in range(60):
        d1, w = mismatch(k + step, w); slope = (d1 - d0) / step; move = -d1 / slope; k += step + move; d0, w = mismatch(k, w); step = max(min(abs(move), 1e-3 * k), 1e-9 * k) * (1 if move >= 0 else -1) or 1e-9 * k
        if abs(d0) < 1e-11 * w: break
    w1, U1, g1 = branch(plate, k, w, primary); w2, U2, g2 = branch(plate, 2 * k, 2 * w1, 1); _, dK = plate.stiffness(2 * k)
    out = {'f_hz': w1 / (2 * math.pi), 'k_rad_m': k, 'phase_velocity': w1 / k, 'cl': math.sqrt((lam + 2 * mu) / rho), 'ct': math.sqrt(mu / rho), 'omega_error': 0., 'omega2_error': w2 / (2 * w1) - 1, 'group_1': g1, 'group_2': g2,
           'growth': {}, 'surface_uz_over_ux_1': abs(plate.surface(U1)[1] / plate.surface(U1)[0]), 'surface_uz_over_ux_2': abs(plate.surface(U2)[1] / plate.surface(U2)[0]), 'z': plate.z, 'U1': U1, 'U2': U2,
           'parity_1': parity(plate, U1), 'parity_2': parity(plate, U2)}
    for part, constants in (('geometric', (0., 0., 0.)), ('total', (l, m, n))):
        rate = 1j * (U2.conj() @ plate.forcing(U1, k, *constants)) / (U2.conj() @ dK @ U2); out['growth'][part] = complex(rate * plate.surface(U2)[0] / plate.surface(U1)[0] ** 2)
    out['growth']['material'] = out['growth']['total'] - out['growth']['geometric']
    return out


def table(thickness, rho, lam, mu, l, m, n, f_guess, cp_guess, primary=-1, span=.16, count=17, elements=12, degree=6):
    """`theory_shg_lamb.burst_table` for this pair: both branches and the coupling of every pair of components over ω₀(1 ± span)."""
    r = synchronous(thickness, rho, lam, mu, l, m, n, f_guess, cp_guess, primary, elements, degree); plate = Plate(thickness, rho, lam, mu, elements, degree); w0 = 2 * math.pi * r['f_hz']
    eps = np.linspace(-span, span, count); eps2 = np.linspace(-span, span, 2 * count - 1)
    first = plate.follow(r['k_rad_m'], r['U1'], w0, w0 * (1 + eps)); first = [first[float(w)] for w in np.sort(w0 * (1 + eps))]
    second = plate.follow(2 * r['k_rad_m'], r['U2'], 2 * w0, 2 * w0 * (1 + eps2)); second = [second[float(w)] for w in np.sort(2 * w0 * (1 + eps2))]
    kappa = {part: np.zeros((count, count), dtype=complex) for part in ('geometric', 'total')}
    for i in range(count):
        for j in range(i, count):
            (ka, Ua, _), (kb, Ub, _), (k2, U2, _) = first[i], first[j], second[i + j]; _, dK = plate.stiffness(k2)
            for part, constants in (('geometric', (0., 0., 0.)), ('total', (l, m, n))):
                kappa[part][i, j] = kappa[part][j, i] = 1j * (U2.conj() @ plate.mixing(Ua, ka, Ub, kb, *constants)) / (U2.conj() @ dK @ U2)
    kappa['material'] = kappa['total'] - kappa['geometric']
    return {'omega': w0, 'eps': eps, 'eps2': eps2, 'k1': np.array([b[0] for b in first]), 'group_1': np.array([b[2] for b in first]), 'k2': np.array([b[0] for b in second]), 'group_2': np.array([b[2] for b in second]), 'kappa': kappa}


MATERIAL = dict(rho=2700., lam=58.1e9, mu=26.1e9, l=-143e9, m=-297e9, n=-345e9)   # the software's aluminium
GUESS = dict(f_guess=5.044e6 / 8., cp_guess=8211.)                              # 8 mm plate: fd = 5.04 MHz·mm

if __name__ == '__main__':
    from theory_shg_lamb import pair
    r = synchronous(8e-3, **MATERIAL, **GUESS); k = r['k_rad_m']
    print('pair found: f', r['f_hz'] * 1e-6, 'MHz, fd', r['f_hz'] * 8e-3 * 1e-3, 'MHz·mm, cp', r['phase_velocity'], 'm/s, parities', r['parity_1'], r['parity_2'], 'ω₂/2ω₁ − 1', r['omega2_error'])
    print('group velocities', r['group_1'], r['group_2'], '| surface uz/ux', r['surface_uz_over_ux_1'], r['surface_uz_over_ux_2'])
    print('κ₂/k²', {part: abs(v) / k ** 2 for part, v in r['growth'].items()}, 'phase', {part: math.degrees(np.angle(v)) for part, v in r['growth'].items()})
    finer = synchronous(8e-3, **MATERIAL, **GUESS, elements=32, degree=8); print('finer discretisation: f', finer['f_hz'] / r['f_hz'] - 1, 'κ₂', abs(finer['growth']['total']) / abs(r['growth']['total']) - 1)
    # the same search finds the cp = cL pair of theory_shg_lamb when started near it
    known = pair(8e-3, **MATERIAL); again = synchronous(8e-3, **MATERIAL, f_guess=known['f_hz'] * 1.002, cp_guess=known['cl'] * .998, primary=1)
    print('S1–S2 recovered: f', again['f_hz'] / known['f_hz'] - 1, 'κ₂', abs(again['growth']['total']) / abs(known['growth']['total']) - 1)
