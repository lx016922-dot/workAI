"""The distance of the largest A2/A1² in the software's models of Ding et al. (2018), against the paper's experiment,
its finite-element simulation and the continuous-wave theory.

For every case the out-of-plane displacement of the top surface is read from the record every 2.5 mm, and A2/A1² is
formed in three ways:

* `spectrum`: A1 and A2 are the magnitudes at f and 2f of the spectrum of the whole received signal of the complete
  response — what a measurement has (the paper does not say how it windows its signals);
* `increment`: the same, with A2 taken from the nonlinear increment (complete − linear), which a computation can
  separate and a measurement cannot;
* `window`: the software's own accumulation analysis (`offline.nonlinear.accumulation`): a Hann window that
  follows the wave packet, A2 from the nonlinear increment.

The distance of the maximum is read in two ways: directly (the largest sample, refined with a parabola through its
neighbours) and, as the paper does, from a cubic polynomial fitted over the range of the paper's own figure.
The continuous-wave theory gives π/|Δk| with Δk = k(2f) − 2k(f) of the S0 mode (`offline.guided`).
"""
import json, math, sys
from pathlib import Path
import numpy as np
from cases import CASES, EXTRA, JOBS, MATERIAL, PAPER, RESULTS, REPO, scene
from offline import guided, nonlinear

STEP_MM = 2.5


def amplitude(times, values, frequency_mhz):
    """Magnitude at one frequency of the spectrum of whole signals (columns), as an amplitude."""
    return np.abs(values.T @ np.exp(-2j * math.pi * frequency_mhz * times)) * 2 / len(times)


def peak(x, y, lo=None, hi=None):
    """The first interior local maximum of y(x) in [lo, hi], refined with a parabola: (x, y), or None if y still rises at the end."""
    keep = (x >= (x[0] if lo is None else lo) - 1e-9) & (x <= (x[-1] if hi is None else hi) + 1e-9); x, y = x[keep], y[keep]
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            a, b, c = y[i - 1], y[i], y[i + 1]; shift = .5 * (a - c) / (a - 2 * b + c)
            return float(x[i] + shift * (x[1] - x[0])), float(b - .25 * (a - c) * shift)
    return None


def cubic_peak(x, y, lo, hi):
    """Maximum of the cubic polynomial fitted to y(x) on [lo, hi], as the paper reads its curves: x, or None when the cubic has no maximum inside."""
    keep = (x >= lo - 1e-9) & (x <= hi + 1e-9); c = np.polyfit(x[keep], y[keep] / y[keep].max(), 3); roots = np.roots(np.polyder(c))
    inside = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and lo < r.real < hi and np.polyval(np.polyder(c, 2), r.real) < 0]
    return inside[0] if inside else None


def minimum(x, y, lo, hi):
    keep = (x >= lo - 1e-9) & (x <= hi + 1e-9); x, y = x[keep], y[keep]
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] < y[i + 1]: return float(x[i])
    return None


def analyse(name, case):
    job = JOBS / name; s = scene(case); f = case['frequency_mhz']; d = case['thickness_mm']; width = s['geometry']['width_mm']
    x = np.arange(10., case['reach_mm'] + 1e-9, STEP_MM); points = np.stack([x, np.full_like(x, width / 2), np.full_like(x, d)], axis=1); normal = np.array([0., 0., 1.])
    times, values, m = nonlinear.series(job / 'record', points, normal); full = values['murnaghan']; increment = full - values['linear']
    a1 = amplitude(times, full, f); beta = {'spectrum': amplitude(times, full, 2 * f) / a1 ** 2, 'increment': amplitude(times, increment, 2 * f) / a1 ** 2}
    acc = nonlinear.accumulation(job, {'points': {'mode': 'line', 'start_mm': points[0].tolist(), 'end_mm': points[-1].tolist(), 'count': len(x)}, 'direction': [0., 0., 1.]})
    usable = np.array([r['usable'] for r in acc['rows']]); beta['window'] = np.array([r['ratio_per_m'] if r['usable'] else np.nan for r in acc['rows']])
    plate = guided.plate_of(s); pair = next(r for r in guided._pairs(plate, f * 1e6) if (r['primary'], r['secondary']) == ('S0', 'S0'))
    lo, hi = case['simulated_mm']; out = {'frequency_mhz': f, 'thickness_mm': d, 'cycles': case.get('cycles', 10.), 'nodes': m['nodes'], 'elapsed_s': float((job / 'elapsed_s').read_text()),
                                         'h_mm': json.loads((job / 'request.json').read_text(encoding='utf-8'))['mesh_report']['h_mm'], 'layer_step_us': (m.get('time_plan') or {}).get('layer_step_us'),
                                         'experiment_mm': case['experiment_mm'], 'paper_simulation_mm': case['paper_simulation_mm'], 'measured_mm': case['measured_mm'], 'experiment_note': case['note'],
                                         'theory': {'half_beat_mm': pair['coherence_length_m'] * 1e3, 'delta_k_rad_mm': pair['delta_k_rad_m'] * 1e-3, 'phase_velocity_f_m_s': pair['phase_velocity_primary_m_s'],
                                                    'phase_velocity_2f_m_s': pair['phase_velocity_secondary_m_s'], 'group_velocity_f_m_s': pair['group_velocity_primary_m_s'], 'group_velocity_2f_m_s': pair['group_velocity_secondary_m_s']},
                                         'measured_in_model': {'wavenumber_f_difference': acc['fit']['f']['theory']['difference'], 'group_velocity_m_s': acc['group']['velocity_m_s']},
                                         'x_mm': x.tolist(), 'a1_m': a1.tolist(), 'maximum': {}}
    for key, y in beta.items():
        good = np.isfinite(y); xs, ys = x[good], y[good]; direct = peak(xs, ys); second = minimum(xs, ys, direct[0], xs[-1]) if direct else None
        out['maximum'][key] = {'direct_mm': direct[0] if direct else None, 'value_per_m': direct[1] if direct else None, 'cubic_mm': cubic_peak(xs, ys, lo, min(hi, xs[-1])),
                               'first_minimum_mm': second, 'last_usable_mm': float(xs[-1])}
        out['beta_' + key + '_per_m'] = [None if not math.isfinite(v) else float(v) for v in y]
    # what the experiment's own sampling would give: points every 25 mm over the measured range, a cubic through them
    sampled = np.isin(np.round(x, 6), np.arange(case['measured_mm'][0], case['measured_mm'][1] + 1e-9, 25.)); y = beta['spectrum']
    out['as_measured'] = {'x_mm': x[sampled].tolist(), 'beta_per_m': y[sampled].tolist(), 'largest_point_mm': float(x[sampled][np.argmax(y[sampled])]), 'smallest_point_mm': float(x[sampled][np.argmin(y[sampled])]),
                          'cubic_mm': cubic_peak(x[sampled], y[sampled], *case['measured_mm']), 'rising_at_the_end': bool(y[sampled][-1] >= y[sampled].max() - 1e-30)}
    return out


def main():
    results = {'paper': PAPER, 'material': MATERIAL, 'step_mm': STEP_MM, 'cases': {}}
    for name, case in {**CASES, **EXTRA}.items():
        if not (JOBS / name / 'key').exists(): print(name, 'not computed', flush=True); continue
        r = analyse(name, case); results['cases'][name] = r; t = r['theory']; mx = r['maximum']
        print(f"{name}: experiment {r['experiment_mm']:g}  paper FE {r['paper_simulation_mm']:g}  theory π/|Δk| {t['half_beat_mm']:.1f}  | software: spectrum {mx['spectrum']['direct_mm']} (cubic {mx['spectrum']['cubic_mm']})  "
              f"increment {mx['increment']['direct_mm']} (cubic {mx['increment']['cubic_mm']})  window {mx['window']['direct_mm']} | as measured: largest point {r['as_measured']['largest_point_mm']}, smallest {r['as_measured']['smallest_point_mm']}, "
              f"cubic {r['as_measured']['cubic_mm']}, rising at the end {r['as_measured']['rising_at_the_end']} | k(f) {r['measured_in_model']['wavenumber_f_difference']:+.2e}", flush=True)
    RESULTS.mkdir(exist_ok=True); (RESULTS / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
