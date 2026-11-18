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

The receivers are 0.6 wavelengths apart or more, too far to separate modes by wavenumber, and the other symmetric
modes at f (S0, S2) and at 2f beat against the pair there: the straight-line fit has a standard error of a few
per cent. The compact record also keeps the complete field at four instants, with nodes a thirtieth of a wavelength
apart along x. There the linear field is limited to wavenumbers around k and the increment to those around 2k,
which removes the other modes, and gives

  * the wavenumbers of both components and their profiles through the thickness, against the mode shapes of the
    theory (every case);
  * for the end-face excitation N4, where every part of the wave train started at x = 0: the fit of
    u²ₐ(x) = κ₂ · (x − x₀) · u¹ₐ(x)² along the train, with x₀ expected at the loaded end. (With the comb of N3 the
    train is built up over ten wavelengths of source, so the distance travelled differs along the train and this
    fit does not apply; N3 is compared at the receivers only.)

    python analyze_s1s2.py            # every finished case → results/results-s1s2.json
"""
import json, math, sys
import numpy as np
from pathlib import Path
from harness import JOBS, RESULTS, series
from cases_s1s2 import ABSORB, CASES, CL, HALF, LENGTH, frequency_mhz
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
    f = frequency_mhz(half) * 1e6; omega = 2 * math.pi * f; k = omega / CL
    scene = CASES[name](); points = [r['point_mm'] for r in scene['receivers']]; xs = np.array([p[0] for p in points]) * (half / HALF) * 1e-3
    if data is None:
        job = job or JOBS / name
        t, lin = series(job, 'linear', points); _, stv = series(job, 'stvk', points); _, ful = series(job, 'full', points)
        lin, stv, ful = lin[..., 0], stv[..., 0], ful[..., 0]
    else: t, lin, stv, ful = data
    dt = (t[1] - t[0]) * 1e-6; theory = pair(2 * half * 1e-3, **MATERIAL)
    u1 = analytic(lin, dt, (.5 * f, 1.5 * f)); power = np.abs(u1) ** 2
    # Linear: phase velocity from the phase of Σ u e^{−iωt} along x, group velocity from the envelope's centroid.
    C1 = lin.T @ np.exp(-2j * math.pi * f * np.arange(len(t)) * dt); phase = np.unwrap(np.angle(C1 * np.exp(1j * k * xs)))   # the receivers are 0.6 wavelengths apart
    k_fit = k - np.polyfit(xs, phase, 1)[0]; arrival = (power * (t[:, None] * 1e-6)).sum(0) / power.sum(0); group = 1 / np.polyfit(xs, arrival, 1)[0]
    amplitude = np.abs(u1).max(0)
    out = {'case': name, 'half_thickness_mm': half, 'f_mhz': f / 1e6, 'fd_mhz_mm': f * 2 * half * 1e-6, 'x_over_wavelength': (xs * k / (2 * math.pi)).tolist(), 'x_mm': (xs * 1e3).tolist(),
           'phase_velocity': {'software': omega / k_fit, 'theory': CL, 'difference': omega / k_fit / CL - 1},
           'group_velocity': {'software': group, 'theory': theory['group_1'], 'difference': group / theory['group_1'] - 1},
           'primary_peak_m': amplitude.tolist(), 'primary_peak_variation': float(amplitude.std() / amplitude.mean()),
           'peak_strain': float(amplitude.max() * k), 'k_squared': k * k, 'parts': {}}
    square = u1 ** 2; norm = (np.abs(square) ** 2).sum(0)
    C2w = np.exp(-2j * math.pi * 2 * f * np.arange(len(t)) * dt)
    for part, increment in (('total', ful - lin), ('geometric', stv - lin), ('material', ful - stv)):
        u2 = analytic(increment, dt, (1.5 * f, 2.5 * f)); r = (u2 * square.conj()).sum(0) / norm
        slope, intercept, residual = line(xs, r); expected = np.conj(theory['growth'][part])
        uncertainty = residual / math.sqrt(max(1 - residual ** 2, 1e-12) * (len(xs) - 2))   # relative standard error of the slope of a straight-line fit
        explained = float(1 - ((np.abs(u2 - r * square) ** 2).sum(0) / (np.abs(u2) ** 2).sum(0)).mean())   # how much of the 2f band follows u1²
        out['parts'][part] = {'slope_per_m2': [slope.real, slope.imag], 'theory_per_m2': [expected.real, expected.imag], 'magnitude_difference': abs(slope) / abs(expected) - 1,
                              'phase_difference_deg': math.degrees(np.angle(slope / expected)), 'fit_residual': residual, 'slope_standard_error': uncertainty, 'slope_over_k_squared': abs(slope) / (k * k),
                              'theory_over_k_squared': abs(expected) / (k * k), 'r_abs_per_m': np.abs(r).tolist(), 'follows_square_of_primary': explained,
                              'A2_m': np.abs(increment.T @ C2w * 2 / len(t)).tolist(), 'peak_ratio_to_primary': float(np.abs(u2).max() / amplitude.max())}
    out['A1_m'] = np.abs(C1 * 2 / len(t)).tolist()
    return out, {'t': t, 'lin': lin, 'stv': stv, 'ful': ful}


def lines_of(points):
    """The mesh lines along x of the plane y = min: {z: node ids sorted by x}."""
    plane = np.flatnonzero(np.abs(points[:, 1] - points[:, 1].min()) < 1e-9); levels = np.unique(np.round(points[plane, 2], 9))
    return {float(z): plane[np.abs(points[plane, 2] - z) < 1e-8][np.argsort(points[plane[np.abs(points[plane, 2] - z) < 1e-8], 0])] for z in levels}


def resample(x, values, grid, degree=4):
    """The element interpolant (degree-4 Lagrange on each element's five nodes) of nodal values on a line, on a uniform grid."""
    out = np.empty(len(grid)); element = np.clip((np.searchsorted(x[::degree], grid, side='right') - 1), 0, (len(x) - 1) // degree - 1)
    for e in np.unique(element):
        here = element == e; xn = x[e * degree:e * degree + degree + 1]; vn = values[e * degree:e * degree + degree + 1]; total = np.zeros(here.sum())
        for i in range(degree + 1):
            basis = np.ones(here.sum())
            for j in range(degree + 1):
                if j != i: basis *= (grid[here] - xn[j]) / (xn[i] - xn[j])
            total += vn[i] * basis
        out[here] = total
    return out


def band(values, dx, centre, width=.3):
    """The part of a field along x with wavenumbers within ±width·centre of `centre` (positive wavenumbers; raised-cosine edges)."""
    n = len(values); kk = 2 * math.pi * np.fft.fftfreq(n, dx); d = np.abs(kk - centre) / (width * centre)
    mask = np.where(d <= .7, 1., np.where(d < 1., .5 * (1 + np.cos(math.pi * (d - .7) / .3)), 0.)) * (kk > 0)
    return np.fft.ifft(np.fft.fft(values) * 2 * mask)


def stored_fields(record, frame):
    manifest = json.loads((record / 'recording.json').read_text(encoding='utf-8'))
    for row in manifest['chunks']:
        with np.load(record / row['file']) as z:
            if frame in z['snapshot_frames']: return dict(zip(('linear', 'stvk', 'full'), z['snapshots'][:, list(z['snapshot_frames']).index(frame)]))


def snapshot(name, half=HALF, job=None, fields=None, points=None, time_us=None, index=1, clear=None):
    """Wavenumbers, mode shapes and (end-face cases) the growth of the second harmonic from the complete field at one instant.

    `index`: which of the four stored instants. `fields` {'linear', 'stvk', 'full'}: nodal displacements (nodes × 3) with `points`
    when the record was not written by the worker. `clear` (m): the stretch of the plate free of sources and absorbing layers."""
    f = frequency_mhz(half) * 1e6; omega = 2 * math.pi * f; k = omega / CL; theory = pair(2 * half * 1e-3, **MATERIAL); end = name.startswith('N4')
    if fields is None:
        record = (job or JOBS / name) / 'record'; manifest = json.loads((record / 'recording.json').read_text(encoding='utf-8')); points = np.load(record / 'mesh.npz')['points']
        frame = manifest['snapshot_frames'][index]; time_us = frame * manifest['sample_step_us']; fields = stored_fields(record, frame)
    lines = lines_of(points); top = max(lines); x = points[lines[top], 0]; grid = np.linspace(x[0], x[-1], 16384, endpoint=False); dx = grid[1] - grid[0]
    cache = {}
    def along(part, z, axis, centre):
        key = (part, z, axis)
        if key not in cache: cache[key] = resample(points[lines[z], 0], fields[part][lines[z], axis], grid)
        return band(cache[key], dx, centre)
    u1 = along('linear', top, 0, k); envelope = np.abs(u1)
    # the stretch used: clear of the source and of the absorbing layer by two half thicknesses, where the train is at least 30 % of its peak
    lo, hi = clear or (((5. if end else 31. + 5 * 3.5922 + 2.) * half * 1e-3), (LENGTH - ABSORB - 2.) * half * 1e-3)
    inside = (grid >= lo) & (grid <= hi); use = inside & (envelope >= .3 * envelope[inside].max()); xs = grid[use]
    out = {'case': name, 'time_us': time_us, 'x_range_mm': [float(xs.min() * 1e3), float(xs.max() * 1e3)], 'wavelengths': float((xs.max() - xs.min()) * k / (2 * math.pi)),
           'primary_peak_m': float(envelope[use].max()), 'peak_strain': float(envelope[use].max() * k), 'parts': {}, 'mode_shapes': {}}
    second = {part: along(a, top, 0, 2 * k) - along(b, top, 0, 2 * k) for part, (a, b) in {'total': ('full', 'linear'), 'geometric': ('stvk', 'linear'), 'material': ('full', 'stvk')}.items()}
    # wavenumbers of the two components: the |amplitude|²-weighted mean of the local wavenumber over the train
    for label, signal, centre in (('k_primary', u1, k), ('k_second', second['total'], 2 * k)):
        local = np.gradient(np.unwrap(np.angle(signal[use] * np.exp(-1j * centre * xs))), xs); weight = np.abs(signal[use]) ** 2
        out[label + '_difference'] = float((local * weight).sum() / weight.sum() / centre)
    if end:
        square = u1[use] ** 2
        for part, u2 in second.items():
            # least squares of u2 = (slope·x + intercept)·u1² over the train
            M = np.stack([xs * square, square], 1); coef, *_ = np.linalg.lstsq(M, u2[use], rcond=None); expected = theory['growth'][part]
            ratio = (u2[use] * square.conj()).sum() / (np.abs(square) ** 2).sum(); centre_x = float((xs * np.abs(square) ** 2).sum() / (np.abs(square) ** 2).sum())
            out['parts'][part] = {'slope_per_m2': [coef[0].real, coef[0].imag], 'theory_per_m2': [expected.real, expected.imag], 'magnitude_difference': abs(coef[0]) / abs(expected) - 1,
                                  'phase_difference_deg': math.degrees(np.angle(coef[0] / expected)), 'slope_over_k_squared': abs(coef[0]) / (k * k), 'theory_over_k_squared': abs(expected) / (k * k),
                                  'fit_residual': float(np.linalg.norm(M @ coef - u2[use]) / np.linalg.norm(u2[use])), 'origin_mm': float(np.real(-coef[1] / coef[0]) * 1e3),
                                  # the whole train as one number: u2 = ratio·u1² at its centre; from the loaded end the theory gives κ₂·centre
                                  'ratio_per_m': [ratio.real, ratio.imag], 'train_centre_mm': centre_x * 1e3, 'ratio_difference_from_end': abs(ratio) / (abs(expected) * centre_x) - 1,
                                  'peak_ratio_to_primary': float(np.abs(u2[use]).max() / envelope[use].max())}
    # mode shapes through the thickness, relative to u_x on the top face: S1 in the linear field at k, S2 in the increment at 2k
    zt = theory['z']
    for label, parts, centre, U in (('S1', ('linear', None), k, theory['U1']), ('S2', ('full', 'linear'), 2 * k, theory['U2'])):
        field = lambda z, axis: along(parts[0], z, axis, centre) - (along(parts[1], z, axis, centre) if parts[1] else 0.)
        reference = field(top, 0); weight = reference[use].conj() / (np.abs(reference[use]) ** 2).sum(); zs = sorted(lines)
        measured = np.array([[(field(z, axis)[use] * weight).sum() for axis in (0, 2)] for z in zs])
        expected = np.array([[np.interp(z, zt, U[a::2].real) + 1j * np.interp(z, zt, U[a::2].imag) for a in (0, 1)] for z in zs]) / U[-2]
        out['mode_shapes'][label] = {'z_mm': [z * 1e3 for z in zs], 'ux': [[v.real, v.imag] for v in measured[:, 0]], 'uz': [[v.real, v.imag] for v in measured[:, 1]],
                                     'ux_theory': [[v.real, v.imag] for v in expected[:, 0]], 'uz_theory': [[v.real, v.imag] for v in expected[:, 1]],
                                     'difference': float(np.abs(measured - expected).max() / np.abs(expected).max())}
    return out


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
        lines.append(f"  receivers {part:9s} κ₂/k² software {p['slope_over_k_squared']:.4f}, theory {p['theory_over_k_squared']:.4f}: magnitude {p['magnitude_difference']:+.2%} ± {p['slope_standard_error']:.1%}, phase {p['phase_difference_deg']:+.2f}°, "
                     f"{p['follows_square_of_primary']:.1%} of the 2f band follows u1², 2f peak / f peak {p['peak_ratio_to_primary']:.2e}")
    for s in r['snapshots']:
        lines.append(f"  field at {s['time_us']:.0f} μs over {s['wavelengths']:.1f} wavelengths ({s['x_range_mm'][0]:.0f}–{s['x_range_mm'][1]:.0f} mm): k {s['k_primary_difference']:+.1e}, 2k {s['k_second_difference']:+.1e}; "
                     f"mode shapes S1 {s['mode_shapes']['S1']['difference']:.2%}, S2 {s['mode_shapes']['S2']['difference']:.2%}")
        for part in PARTS if s['parts'] else ():
            p = s['parts'][part]
            lines.append(f"    {part:9s} κ₂/k² software {p['slope_over_k_squared']:.4f}, theory {p['theory_over_k_squared']:.4f}: magnitude {p['magnitude_difference']:+.2%}, phase {p['phase_difference_deg']:+.2f}°, "
                         f"fit residual {p['fit_residual']:.3f}, origin {p['origin_mm']:+.1f} mm; whole train at {p['train_centre_mm']:.0f} mm against κ₂·x from the end: {p['ratio_difference_from_end']:+.2%}")
    return '\n'.join(lines)


if __name__ == '__main__':
    results, raw = {}, {}
    for name in CASES:
        if (JOBS / name / 'key').exists():
            results[name], raw[name] = analyse(name); results[name]['snapshots'] = [snapshot(name, index=i) for i in ((1, 2) if name.startswith('N4') else (1,))]; print(summary(results[name]), flush=True)
    for full, halved in (('N3-e3', 'N3-half'), ('N4-e3', 'N4-half')):
        if full in raw and halved in raw:
            results[f'amplitude_scaling_{full[:2]}'] = scaling(raw[full], raw[halved]); print(full[:2], '2 MPa / 1 MPa:', {k: round(v, 4) for k, v in results[f'amplitude_scaling_{full[:2]}'].items()})
    (RESULTS / 'results-s1s2.json').write_text(json.dumps(results, default=float))
