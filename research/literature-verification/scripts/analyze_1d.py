"""1D plane longitudinal wave: speed, amplitude and second-order (β) checks against the classical solution.

Classical result (Breazeale & Thompson 1963; Hamilton & Blackstock 1998; Matlack et al. 2015): for
ρ u_tt = ∂P/∂x with P = C11 u_x + ½(3C11 + C111) u_x², i.e. ρ u_tt = C11 u_xx (1 − β u_x), β = −(3 + C111/C11),
the second-order field of u1 = f(t − x/c) is u2 = β x (∂t u1)² / (4c²) + (a wave of fixed shape from the boundary),
whose 2f amplitude is the textbook A2 = β k² x A1² / 8. Here the geometric part has β = −3 (C111 = 0) and the material
part β = −C111/C11, so each response the software exports has its own closed-form prediction.
"""
import json, math, sys
import numpy as np
from scipy.interpolate import CubicSpline
from harness import JOBS, RESULTS, series
from physics.materials import tensors, rotation_z, presets

def dt_spectral(u, dt):
    n = len(u); N = 1 << (int(math.ceil(math.log2(4 * n)))); U = np.fft.rfft(u, N); w = 2j * math.pi * np.fft.rfftfreq(N, dt)
    return np.fft.irfft(U * w, N)[:n]

def hann_burst(t, f, cycles):
    tau = t * f / cycles; return np.where((tau >= 0) & (tau <= 1), .5 * (1 - np.cos(2 * math.pi * tau)) * np.sin(2 * math.pi * f * t), 0.)

def analyse(name, xs, material, sigma_mpa, f_mhz, cycles=5., window_us=None):
    job = JOBS / name; pts = [[x, .4, .4] for x in xs]
    t, lin = series(job, 'linear', pts); _, stv = series(job, 'stvk', pts); _, ful = series(job, 'full', pts)
    lin, stv, ful = lin[..., 0], stv[..., 0], ful[..., 0]; dt = (t[1] - t[0]) * 1e-6
    if window_us: keep = t <= window_us; t, lin, stv, ful = t[keep], lin[keep], stv[keep], ful[keep]
    rho = material['rho_kg_m3']; c2, c3, _ = tensors(material)  # global frame (the fibre angle is already applied)
    C11, C111 = c2[0, 0], c3[0, 0, 0]; c = math.sqrt(C11 / rho)
    beta = {'geometric': -3., 'material': -C111 / C11, 'total': -(3 + C111 / C11)}
    # wave speed: arrival delay of the linear burst between the first and last receiver (cross-correlation, spline-refined)
    a, b = lin[:, 0], lin[:, -1]; fine = np.arange(t[0], t[-1], .001); sa, sb = CubicSpline(t, a)(fine), CubicSpline(t, b)(fine)
    N = 1 << int(math.ceil(math.log2(2 * len(fine)))); xc = np.fft.irfft(np.fft.rfft(sb, N) * np.conj(np.fft.rfft(sa, N)), N)
    lag = (int(np.argmax(xc[:len(fine)]))) * .001e-6  # b lags a
    c_meas = (xs[-1] - xs[0]) * 1e-3 / lag
    # amplitude: particle velocity of a traction-driven half-space, v = σ s(t − x/c)/(ρc)
    v = dt_spectral(lin[:, 0], dt); v_ref = sigma_mpa * 1e6 * hann_burst(t * 1e-6 - xs[0] * 1e-3 / c, f_mhz * 1e6, cycles) / (rho * c)
    amp_err = float(np.linalg.norm(v - v_ref) / np.linalg.norm(v_ref))
    # second order: difference of increments between receivers (removes the boundary-generated wave), vs β Δx (∂t u1)² / (4c²)
    incs = {'geometric': stv - lin, 'material': ful - stv, 'total': ful - lin}; rows = []
    for j in range(1, len(xs)):
        dx = (xs[j] - xs[0]) * 1e-3; shift = dx / c; q = dx * dt_spectral(lin[:, j], dt) ** 2 / (4 * c * c)
        row = {'x_mm': xs[j], 'dx_mm': xs[j] - xs[0]}
        for part, inc in incs.items():
            back = CubicSpline(t, inc[:, 0], extrapolate=False)(t - shift * 1e6); back = np.nan_to_num(back)
            D = inc[:, j] - back; fit = float(D @ q / (q @ q)); res = float(np.linalg.norm(D - fit * q) / max(np.linalg.norm(fit * q), 1e-300))  # 0 when the increment is identically zero
            row[part] = {'beta_fit': fit, 'beta_theory': beta[part], 'waveform_residual': res,
                         'rel_diff': fit / beta[part] - 1 if abs(beta[part]) > 1e-6 else None, 'abs_diff': fit - beta[part]}
        rows.append(row)
    # 2f amplitude vs distance (textbook form A2 = β k² x A1²/8 with A1 the measured fundamental envelope peak)
    def band(u, f0):
        N = 1 << 16; U = np.fft.rfft(u, N); fr = np.fft.rfftfreq(N, dt); keep = (fr > f0 * .6e6) & (fr < f0 * 1.4e6)
        return np.abs(np.fft.irfft(np.where(keep, U, 0) * 2, N)[:len(u)] + 0j)  # crude: peak of band-limited signal
    def envelope_peak(u, f0, lo, hi):  # analytic signal of the band [lo, hi]·f0 (positive frequencies doubled)
        N = 1 << 16; U = np.fft.fft(u, N); fr = np.fft.fftfreq(N, dt); A = np.where((fr > f0 * lo * 1e6) & (fr < f0 * hi * 1e6), 2 * U, 0)
        return float(np.max(np.abs(np.fft.ifft(A, N)[:len(u)])))
    k = 2 * math.pi * f_mhz * 1e6 / c; growth = []
    for j, x in enumerate(xs):
        # A1 from the particle velocity: a traction-driven displacement burst also carries a one-sided low-frequency bump that a band edge would pick up
        A1 = envelope_peak(dt_spectral(lin[:, j], dt), f_mhz, .3, 1.7) / (2 * math.pi * f_mhz * 1e6); A2 = envelope_peak(incs['total'][:, j], f_mhz, 1.3, 2.7)
        growth.append({'x_mm': x, 'A1_m': A1, 'A2_m': A2, 'A2_textbook_m': abs(beta['total']) * k * k * x * 1e-3 * A1 * A1 / 8})
    xs_m = np.array(xs) * 1e-3; A2 = np.array([g['A2_m'] for g in growth]); A1 = np.array([g['A1_m'] for g in growth])
    slope, icpt = np.polyfit(xs_m, A2, 1)
    return {'job': name, 'c_theory_m_s': c, 'c_measured_m_s': c_meas, 'c_rel_diff': c_meas / c - 1, 'C11_GPa': C11 / 1e9, 'C111_GPa': C111 / 1e9,
            'particle_velocity_rel_L2_vs_sigma_over_rho_c': amp_err, 'beta_theory': beta, 'pairs': rows, 'growth': growth,
            'A2_slope_fit_per_m': slope, 'A2_slope_textbook_per_m': abs(beta['total']) * k * k * float(np.mean(A1)) ** 2 / 8, 'A2_intercept_m': icpt,
            'traces': {'t_us': t.tolist(), 'x_mm': xs, 'linear': lin.T.tolist(), 'geometric': incs['geometric'].T.tolist(), 'material': incs['material'].T.tolist()}}

if __name__ == '__main__':
    from cases_1d import CASES, X, cfrp
    al = presets()['murnaghan']['material']; results = {}
    plan = {'L1-al-10MPa': (X, al, 10., .5, 5., None), 'L1-al-5MPa': (X, al, 5., .5, 5., None),
            'L1-cfrp-0deg-a1': ([20., 40., 60., 80.], cfrp(0., -200.), 10., .4, 5., 22.5), 'L1-cfrp-90deg': ([20., 40., 60., 80.], cfrp(90.), 2., .4, 5., None)}
    for name in sys.argv[1:] or plan:
        if not (JOBS / name / 'key').exists(): continue
        results[name] = analyse(name, *plan[name])
    (RESULTS / 'results-1d.json').write_text(json.dumps(results, indent=1, default=float))
    for n, r in results.items():
        print(n, 'c', round(r['c_measured_m_s'], 2), 'vs', round(r['c_theory_m_s'], 2), f"({r['c_rel_diff']:+.2e})", 'v-amp L2', f"{r['particle_velocity_rel_L2_vs_sigma_over_rho_c']:.3e}")
        for row in r['pairs']: print('  dx', row['dx_mm'], {p: (round(row[p]['beta_fit'], 3), round(row[p]['beta_theory'], 3), round(row[p]['waveform_residual'], 3)) for p in ('geometric', 'material', 'total')})
        print('  A2 slope', r['A2_slope_fit_per_m'], 'textbook', r['A2_slope_textbook_per_m'], 'intercept', r['A2_intercept_m'])
