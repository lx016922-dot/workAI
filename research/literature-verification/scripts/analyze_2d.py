"""Lamb-wave dispersion from the software's waveforms vs the Rayleigh–Lamb equation; absorbing-layer echo level."""
import json, math, sys
import numpy as np
from harness import JOBS, RESULTS, series
from theory import fundamental

CL, CT = 6391.54534145318, 3109.126351029605   # from the aluminium card: λ 58.1, μ 26.1 GPa, ρ 2700 kg/m³
DENSE = [40. + i for i in range(8)]; COARSE = [50. + 10 * i for i in range(7)]


def dispersion(name, mode, comp):
    job = JOBS / name; xs = DENSE + COARSE; t, u = series(job, 'linear', [[x, .5, 1.] for x in xs]); u = u[..., comp]
    dt = (t[1] - t[0]) * 1e-6; N = 1 << 17; U = np.fft.rfft(u, N, axis=0); fr = np.fft.rfftfreq(N, dt)
    power = np.abs(U).mean(1); band = np.flatnonzero(power > .25 * power.max())  # within 12 dB of the spectral peak
    xs_m = np.array(xs) * 1e-3; rows = []
    for i in band[::max(1, len(band) // 40)]:
        f = fr[i]; ph = -np.angle(U[i])  # +k x for exp(i(ωt − kx))
        unwrapped = np.unwrap(ph[:8]); k = np.polyfit(xs_m[:8], unwrapped, 1)[0]
        for j in range(8, len(xs)):  # extend over the coarse points with the current k
            predicted = unwrapped[-1] + k * (xs_m[j] - xs_m[j - 1]); value = ph[j] + 2 * math.pi * round((predicted - ph[j]) / (2 * math.pi))
            unwrapped = np.append(unwrapped, value); k = np.polyfit(xs_m[:j + 1], unwrapped, 1)[0]
        theory = fundamental(f, CL, CT, 2e-3, mode)
        rows.append({'f_mhz': f / 1e6, 'cp_measured': 2 * math.pi * f / k, 'cp_theory': theory['cp'], 'cg_theory': theory['cg'], 'k_measured': k, 'k_theory': theory['k']})
    for r in rows: r['cp_rel_diff'] = r['cp_measured'] / r['cp_theory'] - 1
    ks = np.array([r['k_measured'] for r in rows]); ws = 2 * math.pi * np.array([r['f_mhz'] for r in rows]) * 1e6
    cg = 1 / np.polyval(np.polyder(np.polyfit(ws, ks, 3)), ws)  # group velocity dω/dk from a smooth cubic k(ω) over the band
    for r, g in zip(rows, cg): r['cg_measured'] = float(g); r['cg_rel_diff'] = float(g / r['cg_theory'] - 1)
    inner = rows[2:-2]
    return {'job': name, 'mode': mode, 'rows': rows, 'max_abs_cp_rel_diff': max(abs(r['cp_rel_diff']) for r in rows),
            'max_abs_cg_rel_diff_inner': max(abs(r['cg_rel_diff']) for r in inner), 'band_mhz': [rows[0]['f_mhz'], rows[-1]['f_mhz']],
            'traces': {'t_us': t.tolist(), 'x_mm': xs, 'u': u.T.tolist()}}


def echoes():
    ref_t, ref = series(JOBS / 'B1-ref', 'linear', [[40., .4, .4]]); ref = ref[:, 0, 0]; out = {}
    for name in ('B1-free', 'B1-abs6', 'B1-abs13', 'B1-abs26', 'B1-abs39'):
        if not (JOBS / name / 'key').exists(): continue
        t, u = series(JOBS / name, 'linear', [[40., .4, .4]]); u = u[:, 0, 0]; late = t > 20.  # the direct burst has passed x = 40 by 16.3 μs
        R = float(np.max(np.abs(u - ref)[late]) / np.max(np.abs(ref)))
        request = json.loads((JOBS / name / 'request.json').read_text())  # the layer length actually run
        layer = sum(b['length_mm'] for b in request['scene'].get('boundaries', []) if b['type'] == 'absorbing')
        out[name] = {'layer_mm': layer, 'reflection_ratio': R, 'reflection_db': 20 * math.log10(R),
                     'wavelength_mm': CL / .5e6 * 1e3, 'trace': u.tolist()}
    out['t_us'] = ref_t.tolist(); out['reference'] = ref.tolist(); return out


if __name__ == '__main__':
    res = {}
    for name, mode, comp in (('L2-S0', 'S0', 0), ('L2-A0', 'A0', 2)):
        if (JOBS / name / 'key').exists(): res[name] = dispersion(name, mode, comp)
    if (JOBS / 'B1-ref' / 'key').exists(): res['echo'] = echoes()
    (RESULTS / 'results-2d.json').write_text(json.dumps(res, default=float))
    for n, r in res.items():
        if n == 'echo':
            for k, v in r.items():
                if isinstance(v, dict): print(k, round(v['reflection_db'], 1), 'dB', 'layer', v['layer_mm'], 'mm / λ', round(v['wavelength_mm'], 1))
            continue
        print(n, 'band', [round(b, 3) for b in r['band_mhz']], 'max |cp err|', f"{r['max_abs_cp_rel_diff']:.2e}", 'max |cg err| inner', f"{r['max_abs_cg_rel_diff_inner']:.2e}")
        for row in r['rows'][::6]: print('   f', round(row['f_mhz'], 3), 'cp', round(row['cp_measured'], 1), round(row['cp_theory'], 1), 'cg', round(row['cg_measured'], 1), round(row['cg_theory'], 1))
