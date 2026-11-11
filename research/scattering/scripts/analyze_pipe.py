"""Pipe: network vs the 2.5D reference, axisymmetry of the 3D result, and mode content against the SAFE dispersion curves.

Waveforms: all three displacement components at 24 receivers (3 stations × 2 radii × 4 angles), 0–40 μs, which contain
the incident modes, their reflection and mode conversion at the free far end, and the re-reflections from the loaded end.
Mode content: a 2D Fourier transform of the axial and radial displacement along an outer-surface line (x 2–58 mm), with
the SAFE curves of the same pipe drawn over it; positive k travels towards +x, negative k back from the far end.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from pipe import CASES, JOBS, receivers, line_points, LINE_X, STEP, RO, RI
from pipe_reference import cartesian, REF, LAM, MU, RHO
from pipe_safe import Pipe

HERE = Path(__file__).resolve().parents[1]
from offline.recording import traces  # noqa: E402  (repository root is on the path via pipe → harness)

REFERENCE, CHECK = 'p8-h1-nr1-dt1', 'p6-h1-nr1-dt1'


def network(case, h, pts):
    record = JOBS / f'P-{case}-h{h:g}' / 'record'
    if not (record / 'recording.json').exists() or json.loads((record / 'recording.json').read_text())['phase'] != 'completed': return None
    out = []
    for i in range(0, len(pts), 8):
        r = traces(record, 'linear', [list(map(float, p)) for p in pts[i:i + 8]]); out += [np.asarray(p['xyz_m']) for p in r['points']]
    return np.stack(out, 1)   # [T, points, (x, y, z)]


def reference(case, tag):
    path = REF / f'pipe-{case}-{tag}.npy'
    if not path.exists(): return None
    U = np.load(path); n = CASES[case][1]; rec = receivers(); line = line_points()
    at_receivers = np.stack([cartesian(U[:, j], n, p[2]) for j, (p, _) in enumerate(rec)], 1)
    at_line = np.stack([cartesian(U[:, len(rec) + j], n, p[2]) for j, (p, _) in enumerate(line)], 1)
    return at_receivers, at_line


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def cylindrical(u, theta_deg):
    th = math.radians(theta_deg); return np.stack([u[..., 0], u[..., 1] * math.cos(th) + u[..., 2] * math.sin(th), -u[..., 1] * math.sin(th) + u[..., 2] * math.cos(th)], -1)


def fk(u_line, dt_s, dx_m):
    """|2D spectrum| (frequency ≥ 0 rows, wavenumber columns, fftshifted) of u_line[t, x], Hann-windowed in x."""
    T, X = u_line.shape; win = np.hanning(X)[None, :]; NT, NX = 4 * T, 8 * X
    S = np.fft.fftshift(np.fft.fft(np.fft.rfft(u_line * win, NT, axis=0), NX, axis=1), axes=1)
    f = np.fft.rfftfreq(NT, dt_s); k = np.fft.fftshift(np.fft.fftfreq(NX, dx_m)) * 2 * np.pi
    return f, -k, np.abs(S)   # e^{i(kx − ωt)}: with numpy's e^{−iωt}·e^{−ikx} convention a +x wave appears at −k, hence the sign flip


def safe_curves(n, fmax=1e6):
    P = Pipe(LAM, MU, RHO, RI * 1e-3, RO * 1e-3, n, 16); ks = np.linspace(5, 3500, 300); curves = []
    for k in ks:
        fr, vec = P.frequencies(k, 6); curves.append([(fr[i] / 2 / math.pi, P.polarization(vec[:, i])) for i in range(6)])
    return ks, curves


def ridge(f, k, S, f0, k_guess, half_width):
    """Wavenumber of the spectral peak at frequency f0 within ±half_width of k_guess (parabolic refinement)."""
    i = int(np.argmin(abs(f - f0))); sel = np.flatnonzero(abs(k - k_guess) <= half_width); j = sel[np.argmax(S[i, sel])]
    if 0 < j < len(k) - 1:
        a, b, c = np.log(S[i, j - 1:j + 2] + 1e-300); d = .5 * (a - c) / (a - 2 * b + c); return float(k[j] + d * (k[1] - k[0]))
    return float(k[j])


if __name__ == '__main__':
    out = {}; rec = receivers(); pts = [q for _, q in rec]; line = [q for _, q in line_points()]
    for case, (direction, n, f0, cycles) in CASES.items():
        ref, chk = reference(case, REFERENCE), reference(case, CHECK)
        if ref is None: print('skip (reference)', case); continue
        row = {'order_n': n, 'frequency_mhz': f0, 'cycles': cycles, 'direction': direction}
        if chk is not None: row['reference_self_check'] = max(rel(chk[0][:, j], ref[0][:, j]) for j in range(len(rec)))
        for h in (1.25, 1.):
            net = network(case, h, pts)
            if net is None: continue
            errs = [rel(net[:, j], ref[0][:, j]) for j in range(len(rec))]
            r = {'receiver_rel_l2': {f'x{p[0]:g}_r{p[1]:g}_t{p[2]:g}': e for (p, _), e in zip(rec, errs)}, 'max': max(errs), 'median': float(np.median(errs))}
            # axisymmetry / angular pattern: the network at four angles, in cylindrical components, against its own mean (n = 0) or the reference
            if n == 0:
                spread = []
                for x in sorted({p[0] for p, _ in rec}):
                    for rr in sorted({p[1] for p, _ in rec}):
                        js = [j for j, (p, _) in enumerate(rec) if p[0] == x and p[1] == rr]; cyl = np.stack([cylindrical(net[:, j], rec[j][0][2]) for j in js])
                        mean = cyl.mean(0); spread.append(max(rel(c, mean) for c in cyl))
                r['angular_spread_max'] = max(spread)
            # stations: dominant component traces for the figures
            j0 = [j for j, (p, _) in enumerate(rec) if p[1] == RO - .02 and p[2] == 0.]
            comp = 0 if n == 0 else 1
            r['traces'] = {'t_us': (np.arange(net.shape[0]) * STEP).tolist(), 'component': 'xyz'[comp],
                           **{f'x{rec[j][0][0]:g}': {'network': net[:, j, comp].tolist(), 'reference': ref[0][:, j, comp].tolist()} for j in j0}}
            row[f'network_h{h:g}'] = r
            print(case, f'h {h:g}: receivers max {r["max"]:.2e}, median {r["median"]:.2e}', f"angular spread {r.get('angular_spread_max', float('nan')):.1e}")
            if h == 1.25:
                netline = network(case, h, line)
                if netline is not None:
                    row['line_rel_l2'] = rel(netline, ref[1])
                    dx = (LINE_X[1] - LINE_X[0]) * 1e-3; spec = {}
                    for name, u in (('network', netline), ('reference', ref[1])):
                        fr, kk, S = fk(u[:, :, comp], STEP * 1e-6, dx); spec[name] = (fr, kk, S)
                    ks, curves = safe_curves(n)
                    # ridge wavenumbers at the carrier for each SAFE branch carrying axial/radial motion, forward and backward
                    rid = []
                    for b in range(6):
                        fb = np.array([c[b][0] for c in curves]); pol = curves[0][b][1]
                        cross = np.flatnonzero((fb[:-1] - f0 * 1e6) * (fb[1:] - f0 * 1e6) <= 0)
                        for i in cross:
                            kb = ks[i] + (f0 * 1e6 - fb[i]) * (ks[i + 1] - ks[i]) / (fb[i + 1] - fb[i]); p = curves[i][b][1]
                            if n == 0 and p[1] > .5: continue   # torsional: not excited by an axial load
                            item = {'branch': b, 'k_safe': float(kb), 'polarization_r_theta_x': [float(v) for v in p]}
                            for name, (fr, kk, S) in spec.items():
                                for sign, label in ((1, 'forward'), (-1, 'backward')):
                                    item[f'{name}_{label}_k'] = ridge(fr, kk, S, f0 * 1e6, sign * kb, 60.)
                                    item[f'{name}_{label}_peak'] = float(S[int(np.argmin(abs(fr - f0 * 1e6))), int(np.argmin(abs(kk - item[f'{name}_{label}_k'])))])
                            rid.append(item)
                    row['fk_ridges_at_carrier'] = rid
                    fr, kk, S = spec['network']; keep_f = fr <= 1.2e6; keep_k = abs(kk) <= 3000
                    row['fk_network'] = {'f_mhz': (fr[keep_f][::4] / 1e6).tolist(), 'k_rad_m': kk[keep_k][::2].tolist(), 'S': (S[keep_f][::4][:, keep_k][:, ::2] / S.max()).tolist()}
                    row['safe_curves'] = {'k_rad_m': ks.tolist(), 'f_mhz': [[c[b][0] / 1e6 for c in curves] for b in range(6)],
                                          'theta_share': [[float(c[b][1][1]) for c in curves] for b in range(6)]}
                    for it in rid:
                        print('   branch', it['branch'], f"k SAFE {it['k_safe']:.0f}", ' net fwd/back', round(it['network_forward_k']), round(it['network_backward_k']), ' ref fwd/back', round(it['reference_forward_k']), round(it['reference_backward_k']),
                              ' peaks net/ref fwd', f"{it['network_forward_peak']:.3e}/{it['reference_forward_peak']:.3e}", 'back', f"{it['network_backward_peak']:.3e}/{it['reference_backward_peak']:.3e}")
        out[case] = row
    (HERE / 'results' / 'pipe.json').write_text(json.dumps(out, indent=1))
