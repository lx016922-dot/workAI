"""Error of 1 ms bar records against the exact linear solution and the independent 1D reference (three controls)."""
import json, math, os, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import series  # noqa: E402
from cases import CASES, X  # noqa: E402

HERE = Path(__file__).resolve().parents[1]; REF = HERE / 'reference'
JOBS = Path(os.environ.get('WORKAI_VERIFY_JOBS') or Path(tempfile.gettempdir()) / 'workai-literature-verification')
WINDOW = 50.  # μs


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def lag_us(a, b, step):
    """Delay of a relative to b (μs), cross-correlation with parabolic refinement."""
    n = len(a); N = 1 << int(math.ceil(math.log2(2 * n))); xc = np.fft.irfft(np.fft.rfft(a, N) * np.conj(np.fft.rfft(b, N)), N)
    xc = np.concatenate([xc[-(n - 1):], xc[:n]]); k = int(np.argmax(xc)); y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
    return (k - (n - 1) + .5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * step


def analyse(name):
    h, f, step = CASES[name]; tag = f'f{int(f * 1e3):03d}k'; job = JOBS / name
    pts = [[x, h / 2, h / 2] for x in X]
    sim = {part: series(job, part, pts)[1][..., 0] for part in ('linear', 'stvk', 'full')}
    ref = {'linear': np.load(REF / f'{tag}-linear-e0.5.npy'), 'stvk': np.load(REF / f'{tag}-stvk-e0.5.npy'), 'full': np.load(REF / f'{tag}-murnaghan-e0.5.npy')}
    coarse = {'linear': np.load(REF / f'{tag}-linear-e1.npy'), 'stvk': np.load(REF / f'{tag}-stvk-e1.npy'), 'full': np.load(REF / f'{tag}-murnaghan-e1.npy')}
    exact = np.load(REF / f'{tag}-exact-linear.npy'); t = np.arange(len(exact)) * step
    responses = {'linear': (sim['linear'], ref['linear']), 'geometric': (sim['stvk'] - sim['linear'], ref['stvk'] - ref['linear']),
                 'material': (sim['full'] - sim['stvk'], ref['full'] - ref['stvk']), 'full': (sim['full'], ref['full'])}
    edges = np.arange(0, t[-1] + 1e-9, WINDOW); out = {'case': name, 'h_mm': h, 'f_mhz': f, 'degree': 4,
        'nodes_per_wavelength_f': 6391.545 / (f * 1e6) / (h / 4 * 1e-3), 'nodes_per_wavelength_2f': 6391.545 / (2 * f * 1e6) / (h / 4 * 1e-3),
        'reference_self_check': {k: rel(ref[k], coarse[k]) for k in ref}, 'reference_linear_vs_exact': rel(ref['linear'], exact), 'windows_us': edges[1:].tolist()}
    for part, (a, b) in responses.items():
        win = [rel(a[(t >= lo) & (t < lo + WINDOW)], b[(t >= lo) & (t < lo + WINDOW)]) for lo in edges[:-1]]
        cum = {f'{T:g}': rel(a[t <= T], b[t <= T]) for T in (100., 250., 500., 1000.)}
        out[part] = {'window_rel_l2': win, 'cumulative_rel_l2': cum}
    out['linear_vs_exact'] = {f'{T:g}': rel(sim['linear'][t <= T], exact[t <= T]) for T in (100., 250., 500., 1000.)}
    late = t >= t[-1] - WINDOW; out['late_lag_us_vs_exact'] = [lag_us(sim['linear'][late, j], exact[late, j], step) for j in range(len(X))]
    out['phase_velocity_rel_error_estimate'] = float(np.mean(out['late_lag_us_vs_exact']) / (t[-1] - WINDOW / 2))  # Δt / t ≈ −δc/c
    out['amplitude_ratio_late'] = float(np.linalg.norm(sim['linear'][late]) / np.linalg.norm(exact[late]))
    return out


if __name__ == '__main__':
    results = {}
    for name in sys.argv[1:] or CASES:
        if not (JOBS / name / 'key').exists(): print('skip', name); continue
        results[name] = r = analyse(name)
        print(name, f"n/λ(2f)={r['nodes_per_wavelength_2f']:.1f}", 'ref self', {k: f'{v:.1e}' for k, v in r['reference_self_check'].items()}, f"ref-exact {r['reference_linear_vs_exact']:.1e}")
        print('   linear vs exact', {k: f'{v:.2e}' for k, v in r['linear_vs_exact'].items()}, 'late lag μs', [round(v, 5) for v in r['late_lag_us_vs_exact']], f"δc/c≈{-r['phase_velocity_rel_error_estimate']:.2e}", f"amp {r['amplitude_ratio_late']:.5f}")
        for part in ('geometric', 'material', 'full'): print('  ', part, {k: f'{v:.2e}' for k, v in r[part]['cumulative_rel_l2'].items()})
    out = HERE / 'results' / 'bar-1ms.json'; old = json.loads(out.read_text()) if out.exists() else {}
    old.update(results); out.write_text(json.dumps(old, indent=1))
