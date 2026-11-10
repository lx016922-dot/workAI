"""S0 second harmonic in a plate: the ratio R(x) = C2(x)/C1(x)² of complex 2f and f coefficients traces
R = a (e^{−iΔk x} − 1) + b for a non-synchronous mode pair (de Lima & Hamilton 2003, Müller et al. 2010), so its rotation rate
and the |sin(Δk x/2)| modulation of |A2| give the phase mismatch Δk = k_S0(2ω) − 2 k_S0(ω), compared here with Rayleigh–Lamb."""
import json, math, sys
import numpy as np
from harness import JOBS, RESULTS, series
from theory import fundamental

CL, CT = 6391.54534145318, 3109.126351029605


def analyse(name='N2-S0-0.4', f=.4e6, thickness=2e-3, width=1.25, xs=tuple(range(20, 161, 5))):
    job = JOBS / name; pts = [[float(x), width / 2, thickness * 500] for x in xs]
    t, lin = series(job, 'linear', pts); _, ful = series(job, 'full', pts); _, stv = series(job, 'stvk', pts)
    dt = (t[1] - t[0]) * 1e-6; u1 = lin[..., 0]; parts = {'total': ful[..., 0] - u1, 'geometric': stv[..., 0] - u1, 'material': ful[..., 0] - stv[..., 0]}
    n = len(t); w = np.exp(-2j * math.pi * np.outer(np.arange(n) * dt, [f, 2 * f]))  # exact-frequency DFT
    C1 = u1.T @ w[:, 0]; xs_m = np.array(xs) * 1e-3
    a1 = fundamental(f, CL, CT, thickness, 'S0'); a2 = fundamental(2 * f, CL, CT, thickness, 'S0'); dk_theory = a2['k'] - 2 * a1['k']
    out = {'job': name, 'f_mhz': f / 1e6, 'dk_theory_rad_m': dk_theory, 'coherence_length_mm': math.pi / abs(dk_theory) * 1e3, 'x_mm': list(xs), 'parts': {}}
    for part, inc in parts.items():
        C2 = inc.T @ w[:, 1]; R = C2 / C1 ** 2; best = None
        for dk in np.linspace(0, 150, 3001):
            M = np.stack([np.exp(-1j * dk * xs_m) - 1, np.ones_like(xs_m)], 1); coef, *_ = np.linalg.lstsq(M, R, rcond=None)
            res = float(np.linalg.norm(M @ coef - R) / np.linalg.norm(R))
            if best is None or res < best[0]: best = (res, dk, coef)
        out['parts'][part] = {'dk_fit_rad_m': best[1], 'fit_residual': best[0], 'dk_rel_diff': best[1] / dk_theory - 1,
                              'A2_over_A1sq': np.abs(R).tolist(), 'A2_m': np.abs(C2).tolist()}
    out['A1_m'] = np.abs(C1).tolist(); return out


if __name__ == '__main__':
    r = analyse(); (RESULTS / 'results-shg.json').write_text(json.dumps(r, default=float))
    print('dk theory', round(r['dk_theory_rad_m'], 3), 'coherence mm', round(r['coherence_length_mm'], 1))
    for p, v in r['parts'].items(): print(p, 'dk fit', round(v['dk_fit_rad_m'], 3), f"({v['dk_rel_diff']:+.3f})", 'residual', round(v['fit_residual'], 4))
    print('|A2|/|A1|^2 along x:', [f'{x}:{v:.3g}' for x, v in zip(r['x_mm'], r['parts']['total']['A2_over_A1sq'])])
