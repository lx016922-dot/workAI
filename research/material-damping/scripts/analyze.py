"""Compare the software's damped bar with the reference solutions (damped_bar.py); write results/bar.json.

    WORKAI_VERIFY_JOBS=<dir> python research/material-damping/scripts/analyze.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from cases import CASES, DAMPING, X, JOBS, load
from damped_bar import Bar, second_harmonic, wavenumber
from physics.signals import evaluate

RESULTS = HERE.parent / 'results'
l2 = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))


def reference(request, element_mm=.5):
    """The three controls of the one-dimensional reference at the receivers: {law: frames × receivers}."""
    s = request['scene']; d = s.get('damping') or {}; src = s['source']; acquisition = s['acquisition']
    signal = lambda t_us: float(evaluate(src, np.array([t_us]))[0]); out = {}
    for law in ('linear', 'stvk', 'murnaghan'):
        bar = Bar(s['geometry']['length_mm'] * 1e-3, element_mm * 1e-3, s['material'], law, d.get('mass_per_s', 0.), d.get('stiffness_s', 0.))
        out[law] = bar.run(signal, src['amplitude_mpa'] * 1e6, acquisition['duration_us'], acquisition['sample_step_us'], [r['point_mm'][0] * 1e-3 for r in s['receivers']],
                           substeps=int(round(25 * .5 / element_mm)))
    return out


def second(t, v, f):
    """Amplitudes at f and 2f: the Fourier sums over the whole record (the burst lies inside it at every receiver, so no window is needed
    and receivers at different distances are measured alike)."""
    tone = lambda g: 2 * abs(np.sum(v[:-1] * np.exp(-2j * np.pi * g * t[:-1]))) / (len(t) - 1)
    return tone(f), tone(2 * f)


def band_error(t, measured, start, distance_m, speed, alpha, beta, f):
    """The spectrum of `measured` in the band of the burst (f/2 … 3f/2) against `start` carried over `distance_m` with the exact
    complex wavenumber: relative L2 over the band. (Below the band the displacement has a slowly varying part that a record of this
    length does not resolve.)"""
    n = len(t) - 1; freq = np.fft.rfftfreq(n, (t[1] - t[0]) * 1e-6); keep = (freq >= .5 * f * 1e6) & (freq <= 1.5 * f * 1e6)
    a, b = np.fft.rfft(start[:-1])[keep], np.fft.rfft(measured[:-1])[keep]
    return float(np.linalg.norm(b - a * np.exp(-1j * wavenumber(2 * np.pi * freq[keep], speed, alpha, beta) * distance_m)) / np.linalg.norm(b))


def analyse(name, fine=False):
    request = json.loads((JOBS / name / 'request.json').read_text()); s = request['scene']; m = s['material']; d = s.get('damping') or {}
    alpha, beta = d.get('mass_per_s', 0.), d.get('stiffness_s', 0.); speed = float(np.sqrt((m['lambda_pa'] + 2 * m['mu_pa']) / m['rho_kg_m3'])); f = s['source']['frequency_mhz']
    t, soft, manifest = load(name); ref = reference(request); x = np.array([r['point_mm'][0] for r in s['receivers']])
    increments = lambda v: {'nonlinear': v['murnaghan'] - v['linear'], 'geometric': v['stvk'] - v['linear'], 'material': v['murnaghan'] - v['stvk']}
    si, ri = increments(soft), increments(ref)
    row = {'damping': d or None, 'frequency_mhz': f, 'cycles': s['source']['cycles'], 'amplitude_mpa': s['source']['amplitude_mpa'], 'h_mm': s['numerics']['h_mm'], 'x_mm': x.tolist(),
           'time_plan': {k: manifest['time_plan'].get(k) for k in ('scheme', 'layer_step_us', 'stability_number', 'stability_limit', 'evaluations_per_control')},
           'elapsed_s': float((JOBS / name / 'elapsed_s').read_text()), 'dissipated_energy_j': manifest.get('dissipated_energy_j'),
           # the linear waveform at each receiver, predicted from the software's own waveform at the first receiver with the exact complex wavenumber
           'linear_against_exact_propagation_l2': [band_error(t, soft['linear'][:, i], soft['linear'][:, 0], (x[i] - x[0]) * 1e-3, speed, alpha, beta, f) for i in range(1, len(x))],
           'against_reference_l2': {law: [l2(soft[law][:, i], ref[law][:, i]) for i in range(len(x))] for law in soft},
           'increment_against_reference_l2': {k: [l2(si[k][:, i], ri[k][:, i]) for i in range(len(x))] for k in si}}
    a1 = np.array([second(t, soft['murnaghan'][:, i], f)[0] for i in range(len(x))]); r1 = np.array([second(t, ref['murnaghan'][:, i], f)[0] for i in range(len(x))])
    row['a1_m'] = a1.tolist(); row['a1_against_reference'] = (a1 / r1 - 1).tolist()
    for k in si:
        a2 = np.array([second(t, si[k][:, i], f)[1] for i in range(len(x))]); r2 = np.array([second(t, ri[k][:, i], f)[1] for i in range(len(x))])
        row[f'a2_{k}_m'] = a2.tolist(); row[f'a2_{k}_against_reference'] = (a2 / r2 - 1).tolist()
    # attenuation of the fundamental between the receivers, against γ(ω)/c with the exact wavenumber
    w = 2 * np.pi * f * 1e6; slope = float(np.polyfit(x * 1e-3, np.log(a1), 1)[0])
    row['attenuation_np_m'] = {'software': -slope, 'exact_f': float(-wavenumber(w, speed, alpha, beta).imag) if d else 0., 'exact_2f': float(-wavenumber(2 * w, speed, alpha, beta).imag) if d else 0.}
    # the quasi-monochromatic formula for the cumulative second harmonic with attenuation, from the software's A₁ at the first receiver
    a1f, a2f = row['attenuation_np_m']['exact_f'], row['attenuation_np_m']['exact_2f']; c11 = m['lambda_pa'] + 2 * m['mu_pa']; beta_nl = -(3 + 2 * (m['l_pa'] + 2 * m['m_pa']) / c11)
    row['beta_nl'] = beta_nl; row['a2_formula_m'] = second_harmonic(x * 1e-3, a1[0] * np.exp(a1f * x[0] * 1e-3), w / speed, beta_nl, a1f, a2f).tolist()
    if fine:   # the reference on elements half the size: its own convergence
        finer = reference(request, .25); fi = increments(finer)
        row['reference_convergence_l2'] = {'murnaghan': max(l2(ref['murnaghan'][:, i], finer['murnaghan'][:, i]) for i in range(len(x))), 'nonlinear_increment': max(l2(ri['nonlinear'][:, i], fi['nonlinear'][:, i]) for i in range(len(x)))}
    return row


if __name__ == '__main__':
    names = sys.argv[1:] or [n for n in CASES if n.startswith('B-') and (JOBS / n / 'key').exists()]
    path = RESULTS / 'bar.json'; out = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        out[name] = row = analyse(name, fine=name == 'B-both'); worst = lambda v: max(abs(np.asarray(v)))
        print(name, '| linear vs exact propagation', f"{worst(row['linear_against_exact_propagation_l2']):.2e}", '| vs reference: linear', f"{worst(row['against_reference_l2']['linear']):.2e}",
              'complete', f"{worst(row['against_reference_l2']['murnaghan']):.2e}", '| increments', ' '.join(f"{k} {worst(v):.2e}" for k, v in row['increment_against_reference_l2'].items()),
              '| A2 nonlinear vs reference', f"{worst(row['a2_nonlinear_against_reference']):.2e}", '| attenuation', row['attenuation_np_m'], row.get('reference_convergence_l2', ''), flush=True)
    RESULTS.mkdir(exist_ok=True); path.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
