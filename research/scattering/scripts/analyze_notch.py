"""Notch scattering: network vs the 2D spectral reference, S0 reflection/transmission and S0→A0 conversion.

At each x the top and bottom surface u_x split into a symmetric part (S0) and an antisymmetric part (A0). The scattered
field is the notch run minus the plate run. Amplitudes are envelope peaks in fixed time windows.
"""
import json, math, os, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import series  # noqa: E402
from notch import CASES, XS, points, F  # noqa: E402

HERE = Path(__file__).resolve().parents[1]; REF = HERE / 'reference'
JOBS = Path(os.environ.get('WORKAI_VERIFY_JOBS') or Path(tempfile.gettempdir()) / 'workai-literature-verification')
STEP = .05; CS0 = 5401.   # S0 phase velocity at 0.3 MHz·2 mm (Rayleigh–Lamb, theory.py)


def split(u):
    """u[t, 4] (x30 top, x30 bottom, x75 top, x75 bottom) → S0 and A0 parts at both stations."""
    return {'S30': .5 * (u[:, 0] + u[:, 1]), 'A30': .5 * (u[:, 0] - u[:, 1]), 'S75': .5 * (u[:, 2] + u[:, 3]), 'A75': .5 * (u[:, 2] - u[:, 3])}


def network(name, h):
    job = JOBS / f'N-{name}-h{h:g}'
    return None if not (job / 'key').exists() else series(job, 'linear', points(h))[1][..., 0]


def reference(name, h):
    p = REF / f'notch-{name}-h{h:g}.npy'
    return np.load(p)[:, :, 0] if p.exists() else None


def envelope(u):
    N = 1 << 14; U = np.fft.fft(u, N); fr = np.fft.fftfreq(N, STEP * 1e-6); A = np.where(fr > 0, 2 * U, 0); return np.abs(np.fft.ifft(A)[:len(u)])


def coefficients(total, plate, t):
    """S0 reflection (x 30), S0 transmission (x 75), A0 converted in reflection and transmission, relative to the incident S0 peak."""
    s, p = split(total), split(plate); scat = {k: s[k] - p[k] for k in s}; inc = envelope(p['S30'])[t < 16].max()
    win_r = (t > 14) & (t < 27.5); win_t = (t > 12) & (t < 30)
    return {'R_S0': float(envelope(scat['S30'])[win_r].max() / inc), 'T_S0': float(envelope(s['S75'])[win_t].max() / inc),
            'R_A0': float(envelope(scat['A30'])[win_r].max() / inc), 'T_A0': float(envelope(s['A75'])[win_t].max() / inc)}


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


if __name__ == '__main__':
    t = np.arange(int(round(45 / STEP)) + 1) * STEP; out = {}
    plate_ref, plate_ref_coarse = reference('plate', .25), reference('plate', .5)
    # propagation alone (no notch): network plate vs reference plate at every station
    pr = split(plate_ref); prc = split(plate_ref_coarse); out['plate'] = {'reference_self_check': {k: rel(prc[k], pr[k]) for k in ('S30', 'S75')}}
    for h in (1., 1.5):
        pn = network('plate', h)
        if pn is not None: pn = split(pn); out['plate'][f'network_h{h:g}_rel_l2'] = {k: rel(pn[k], pr[k]) for k in ('S30', 'S75')}
    print('plate (no notch)', out['plate'])
    for name, (kind, w, d) in CASES.items():
        if kind == 'plate': continue
        ref, coarse = reference(name, .25), reference(name, .5)
        if ref is None or coarse is None: print('skip (reference)', name); continue
        row = {'width_mm': w, 'depth_mm': d, 'reference': coefficients(ref, plate_ref, t), 'reference_coarse': coefficients(coarse, plate_ref_coarse, t)}
        sr, sc = split(ref - plate_ref), split(coarse - plate_ref_coarse)
        row['reference_self_check'] = {k: rel(sc[k], sr[k]) for k in sr}
        for h in (1., 1.5):
            sim, plate = network(name, h), network('plate', h)
            if sim is None or plate is None: continue
            row[f'network_h{h:g}'] = coefficients(sim, plate, t); ss = split(sim - plate)
            row[f'network_h{h:g}_scattered_rel_l2'] = {k: rel(ss[k], sr[k]) for k in sr}
            row[f'network_h{h:g}_total_rel_l2'] = rel(sim, ref)
            if h == 1.: row['trace'] = {'t_us': t[::4].tolist(), **{f'{src}_{k}': v[::4].tolist() for src, d in (('ref', sr), ('net', ss)) for k, v in d.items()}}
        out[name] = row
        print(name, 'ref', {k: round(v, 4) for k, v in row['reference'].items()}, 'ref self', {k: f'{v:.1e}' for k, v in row['reference_self_check'].items()})
        for h in (1., 1.5):
            if f'network_h{h:g}' in row:
                print(f'   net h{h:g}', {k: round(v, 4) for k, v in row[f'network_h{h:g}'].items()}, 'scattered rel', {k: f'{v:.1e}' for k, v in row[f'network_h{h:g}_scattered_rel_l2'].items()}, f"total {row[f'network_h{h:g}_total_rel_l2']:.1e}")
    # interference model for the width dependence (two edges of opposite reflection sign): R(w) ∝ |sin(k w)|
    k = 2 * math.pi * F * 1e6 / CS0
    widths = {n: r for n, r in out.items() if n != 'plate' and r['depth_mm'] == .8}
    for n, r in widths.items(): r['interference_sin_kw'] = abs(math.sin(k * r['width_mm'] * 1e-3))
    (HERE / 'results' / 'notch.json').write_text(json.dumps(out, indent=1))
