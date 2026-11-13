"""Through hole: network waveforms vs the plane-stress analytical solution driven by the network's own incident wave.

The plate run (no hole, same mesh rules) gives the incident u_x at the hole centre; the analytical transfer functions
(hole_theory.Hole, per unit incident u_x at the centre) turn it into the total and scattered fields at each receiver.
They are compared inside the window that ends before any wave scattered by the hole can return from the y symmetry
faces (images) or the incident can return from the far end.
"""
import json, math, os, sys, tempfile
from pathlib import Path
import numpy as np
from hole_theory import Hole
from hole import receivers, JOBS, X0, Y0, A, W, L, T

HERE = Path(__file__).resolve().parents[1]
from offline.recording import traces  # noqa: E402

LAM, MU, RHO = 58.1e9, 26.1e9, 2700.


def read(kind, pts):
    out = []; times = None
    for i in range(0, len(pts), 8):
        r = traces(JOBS / f'H-{kind}' / 'record', 'linear', pts[i:i + 8]); times = np.asarray(r['times_us']); out += [np.asarray(p['xyz_m']) for p in r['points']]
    return times, np.stack(out, 1)


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


if __name__ == '__main__':
    rec = receivers(); pts = [p for _, _, p in rec]
    kind = sys.argv[1] if len(sys.argv) > 1 else 'hole'   # 'ogrid': the defect O-grid mesh (hole.py)
    t, hole = read(kind, pts); _, plate = read('plate', pts + [[X0, Y0, T / 2]])
    inc = plate[:, -1, 0]; plate = plate[:, :-1]; dt = (t[1] - t[0]) * 1e-6
    theory = Hole(LAM, MU, RHO, A * 1e-3); c = theory.cp
    front = (X0 - A) / (c * 1e-3); window_end = front + (W - 2 * A - 1) / (c * 1e-3)   # μs: first image return
    window_end = min(window_end, (2 * L - X0 - A) / (c * 1e-3))
    # The prediction inside the window needs the incident only up to window_end + r_max/c (receivers upstream of the centre
    # see the wave earlier). Taper it to zero before the far-end reflection reaches the centre, so the spectrum is band
    # limited (a record cut at 20 μs would put content up to 10 MHz, where the Bessel series with nmax 40 does not converge).
    need = window_end + max(r for r, _, _ in rec) / (c * 1e-3); back = (2 * L - X0) / (c * 1e-3)
    taper = np.where(t <= need, 1., np.where(t >= back, 0., .5 * (1 + np.cos(np.pi * (t - need) / (back - need)))))
    inc = inc * taper
    N = 1 << 13; U = np.fft.rfft(inc, N); w = 2 * math.pi * np.fft.rfftfreq(N, dt); FMAX = 3e6
    keep = (w > 0) & (w <= 2 * math.pi * FMAX)
    out = {'ka_at_carrier': 2 * math.pi * .25e6 / c * A * 1e-3, 'window_us': [0., window_end], 'incident_taper_us': [need, back],
           'band_mhz': FMAX / 1e6, 'spectrum_above_band': float(np.abs(U[~keep & (w > 0)]).max() / np.abs(U).max()), 'receivers': []}
    win = t <= window_end
    for j, (r, th, p) in enumerate(rec):
        Ht = np.zeros((len(w), 2), complex); Hs = np.zeros((len(w), 2), complex)
        Ht[0] = [1., 0.]   # ω → 0: the incident displacement is a rigid translation, which the hole does not scatter
        for i in np.flatnonzero(keep):
            al = w[i] / c; ux, uy = theory.displacement(w[i], r * 1e-3, math.radians(th)); sx, sy = theory.displacement(w[i], r * 1e-3, math.radians(th), scattered_only=True)
            norm = 1j * al   # incident u_x at the hole centre for φ_i = e^{iαx}
            Ht[i] = np.conj([ux / norm, uy / norm]); Hs[i] = np.conj([sx / norm, sy / norm])   # numpy's e^{+iωt} vs the theory's e^{−iωt}
        pred_total = np.stack([np.fft.irfft(U * Ht[:, k], N)[:len(t)] for k in range(2)], 1)
        pred_scat = np.stack([np.fft.irfft(U * Hs[:, k], N)[:len(t)] for k in range(2)], 1)
        sim_total = hole[:, j, :2]; sim_scat = hole[:, j, :2] - plate[:, j, :2]
        row = {'r_mm': r, 'theta_deg': th, 'total_rel_l2': rel(sim_total[win], pred_total[win]), 'scattered_rel_l2': rel(sim_scat[win], pred_scat[win]),
               'scattered_peak_ratio': float(np.abs(sim_scat[win]).max() / np.abs(pred_scat[win]).max()),
               'scattered_over_incident_peak': float(np.abs(pred_scat[win]).max() / np.abs(inc).max()),
               'trace': {'t_us': t[win].tolist(), 'sim_scattered_x': sim_scat[win, 0].tolist(), 'pred_scattered_x': pred_scat[win, 0].tolist(),
                         'sim_scattered_y': sim_scat[win, 1].tolist(), 'pred_scattered_y': pred_scat[win, 1].tolist()}}
        out['receivers'].append(row)
        print(f"r={r:5.2f} θ={th:5.0f}: total {row['total_rel_l2']:.3f}  scattered {row['scattered_rel_l2']:.3f}  peak ratio {row['scattered_peak_ratio']:.3f}  |scat|/|inc| {row['scattered_over_incident_peak']:.3f}")
    (HERE / 'results' / ('hole.json' if kind == 'hole' else f'hole-{kind}.json')).write_text(json.dumps(out, indent=1))
