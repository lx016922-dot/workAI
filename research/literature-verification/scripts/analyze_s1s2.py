"""The synchronous pair S1(f) → S2(2f): the software against Rayleigh–Lamb and the perturbation solution of theory_shg_lamb.py.

Linear part (the linear control, u_x on the top face at 32 receivers):
  * phase velocity from the phase of the exact-frequency coefficient along x, against cL (S1 at this fd has cp = cL);
  * group velocity from the arrival time of the envelope's centroid, against the semi-analytical value.

Second harmonic (the increments total = complete − linear, geometric = StVK − linear, material = complete − StVK):
for a synchronous pair the theory gives, on the top face,

    u²ₓ = Re[κ₂ · x · (A e^{i(kx−ωt)})²]      with u¹ₓ = Re[A e^{i(kx−ωt)}],

so the analytic signals satisfy u²ₐ(x, t) = conj(κ₂) · (x − x₀) · u¹ₐ(x, t)² whatever the envelope of the burst. At
every receiver r(x) is the least-squares coefficient of the 2f band of the increment on u¹ₐ²; its slope along x is
conj(κ₂), compared in magnitude and phase. The other modes at 2f are not synchronous: they add a bounded,
oscillating part to r(x), which the straight-line fit leaves in its residual.

    python analyze_s1s2.py            # every finished case → results/results-s1s2.json
"""
import json, math, sys
import numpy as np
from harness import JOBS, RESULTS, series
from cases_s1s2 import CASES, CL, CT, HALF, RECEIVERS, frequency_mhz, pitch_mm
from theory_shg_lamb import pair

MATERIAL = dict(rho=2700., lam=58.1e9, mu=26.1e9, l=-143e9, m=-297e9, n=-345e9)   # the software's aluminium (docs/aluminum-material.md)
PARTS = ('total', 'geometric', 'material')


def analytic(values, dt, band=None):
    """Analytic signal along the first axis (positive frequencies only), optionally limited to a band (Hz)."""
    n = len(values); spectrum = np.fft.fft(values, axis=0); f = np.fft.fftfreq(n, dt)
    keep = (f > 0) if band is None else (f >= band[0]) & (f <= band[1])
    return np.fft.ifft(spectrum * (2. * keep).reshape((-1,) + (1,) * (values.ndim - 1)), axis=0)


def line(x, y):
    """Complex least-squares line y = slope·x + intercept; relative residual."""
    M = np.stack([x, np.ones_like(x)], 1).astype(complex); coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    return coef[0], coef[1], float(np.linalg.norm(M @ coef - y) / np.linalg.norm(y - y.mean()))


def analyse(name, half=HALF, job=None, data=None):
    """`data`: (times_us, linear, stvk, full) of u_x at the receivers when the record was not written by the worker."""
    f = frequency_mhz(half) * 1e6; omega = 2 * math.pi * f; k = omega / CL; xs = np.array(RECEIVERS) * half * 1e-3
    if data is None:
        scene = CASES[name](); points = [r['point_mm'] for r in scene['receivers']]; job = job or JOBS / name
        t, lin = series(job, 'linear', points); _, stv = series(job, 'stvk', points); _, ful = series(job, 'full', points)
        lin, stv, ful = lin[..., 0], stv[..., 0], ful[..., 0]
    else: t, lin, stv, ful = data
    dt = (t[1] - t[0]) * 1e-6; theory = pair(2 * half * 1e-3, **MATERIAL)
    u1 = analytic(lin, dt, (.5 * f, 1.5 * f)); power = np.abs(u1) ** 2
    # Linear: phase velocity from the phase of Σ u e^{−iωt} along x, group velocity from the envelope's centroid.
    C1 = lin.T @ np.exp(-2j * math.pi * f * np.arange(len(t)) * dt); phase = np.unwrap(np.angle(C1))
    k_fit = -np.polyfit(xs, phase, 1)[0]; arrival = (power * (t[:, None] * 1e-6)).sum(0) / power.sum(0); group = 1 / np.polyfit(xs, arrival, 1)[0]
    amplitude = np.abs(u1).max(0)
    out = {'case': name, 'half_thickness_mm': half, 'f_mhz': f / 1e6, 'fd_mhz_mm': f * 2 * half * 1e-9, 'x_over_wavelength': (xs * k / (2 * math.pi)).tolist(), 'x_mm': (xs * 1e3).tolist(),
           'phase_velocity': {'software': omega / k_fit, 'theory': CL, 'difference': omega / k_fit / CL - 1},
           'group_velocity': {'software': group, 'theory': theory['group_1'], 'difference': group / theory['group_1'] - 1},
           'primary_peak_m': amplitude.tolist(), 'primary_peak_variation': float(amplitude.std() / amplitude.mean()),
           'peak_strain': float(amplitude.max() * k), 'k_squared': k * k, 'parts': {}}
    square = u1 ** 2; norm = (np.abs(square) ** 2).sum(0)
    C2w = np.exp(-2j * math.pi * 2 * f * np.arange(len(t)) * dt)
    for part, increment in (('total', ful - lin), ('geometric', stv - lin), ('material', ful - stv)):
        u2 = analytic(increment, dt, (1.5 * f, 2.5 * f)); r = (u2 * square.conj()).sum(0) / norm
        slope, intercept, residual = line(xs, r); expected = np.conj(theory['growth'][part])
        explained = float(1 - ((np.abs(u2 - r * square) ** 2).sum(0) / (np.abs(u2) ** 2).sum(0)).mean())   # how much of the 2f band follows u1²
        out['parts'][part] = {'slope_per_m2': [slope.real, slope.imag], 'theory_per_m2': [expected.real, expected.imag], 'magnitude_difference': abs(slope) / abs(expected) - 1,
                              'phase_difference_deg': math.degrees(np.angle(slope / expected)), 'fit_residual': residual, 'slope_over_k_squared': abs(slope) / (k * k),
                              'theory_over_k_squared': abs(expected) / (k * k), 'r_abs_per_m': np.abs(r).tolist(), 'follows_square_of_primary': explained,
                              'A2_m': np.abs(increment.T @ C2w * 2 / len(t)).tolist(), 'peak_ratio_to_primary': float(np.abs(u2).max() / amplitude.max())}
    out['A1_m'] = np.abs(C1 * 2 / len(t)).tolist()
    return out, {'t': t, 'lin': lin, 'stv': stv, 'ful': ful}


def scaling(a, b):
    """Run b at half the traction of run a: the linear response halves, the increments quarter."""
    out = {'linear': float(np.linalg.norm(a['lin']) / np.linalg.norm(b['lin']))}
    for part, (p, q) in {'total': ('ful', 'lin'), 'geometric': ('stv', 'lin'), 'material': ('ful', 'stv')}.items():
        out[part] = float(np.linalg.norm(a[p] - a[q]) / np.linalg.norm(b[p] - b[q]))
    return out


def summary(r):
    lines = [f"{r['case']}: fd {r['fd_mhz_mm']:.4f} MHz·mm, cp {r['phase_velocity']['software']:.1f} m/s ({r['phase_velocity']['difference']:+.3%}), "
             f"cg {r['group_velocity']['software']:.1f} m/s ({r['group_velocity']['difference']:+.2%}), primary peak varies {r['primary_peak_variation']:.1%} along x, peak strain {r['peak_strain']:.1e}"]
    for part in PARTS:
        p = r['parts'][part]
        lines.append(f"  {part:9s} κ₂/k² software {p['slope_over_k_squared']:.4f}, theory {p['theory_over_k_squared']:.4f}: magnitude {p['magnitude_difference']:+.2%}, phase {p['phase_difference_deg']:+.2f}°, "
                     f"fit residual {p['fit_residual']:.3f}, {p['follows_square_of_primary']:.1%} of the 2f band follows u1², 2f peak / f peak {p['peak_ratio_to_primary']:.2e}")
    return '\n'.join(lines)


if __name__ == '__main__':
    results, raw = {}, {}
    for name in CASES:
        if (JOBS / name / 'key').exists():
            results[name], raw[name] = analyse(name); print(summary(results[name]), flush=True)
    if 'N3-e3' in raw and 'N3-half' in raw:
        results['amplitude_scaling'] = scaling(raw['N3-e3'], raw['N3-half']); print('2 MPa / 1 MPa:', {k: round(v, 4) for k, v in results['amplitude_scaling'].items()})
    (RESULTS / 'results-s1s2.json').write_text(json.dumps(results, default=float))
