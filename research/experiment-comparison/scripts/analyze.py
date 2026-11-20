"""The distance of the largest A2/A1² in the software's models of Ding et al. (2018), against the paper's experiment,
its finite-element simulation and the continuous-wave theory.

For every case the out-of-plane displacement of the top surface is read from the record every 2.5 mm, and A2/A1² is
formed in three ways:

* `spectrum`: A1 and A2 are the magnitudes at f and 2f of the spectrum of the whole received signal of the complete
  response — what a measurement has (the paper does not say how it windows its signals);
* `increment`: the same, with A2 taken from the nonlinear increment (complete − linear), which a computation can
  separate and a measurement cannot;
* `window`: the software's own accumulation analysis (`offline.nonlinear.accumulation`): a Hann window that
  follows the wave packet, A2 from the nonlinear increment (on at most 200 points of the same line).

The distance of the maximum is read in two ways: directly (the vertex of a parabola through the points within 10 % of
the top of the first lobe) and, as the paper does, from a cubic polynomial fitted over the range of the paper's own
figure. Where the curve comes down to a minimum and rises again (case b), the minimum and the second maximum are read too.
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


def lobes(x, y):
    """The extrema of a curve made of lobes (|sin|-like): [(kind, x)] in order, 'max' and 'min' alternating. A maximum is
    the vertex of the parabola through the points within 10 % of the top of its lobe, so the small ripple on the curve
    does not decide it; a lobe that is still rising at the end of the data has no maximum. A minimum is the lowest
    point between two lobes."""
    top = float(np.max(y)); above = y >= .5 * top; out = []; i = 0; n = len(y)
    while i < n:
        if not above[i]: i += 1; continue
        j = i
        while j + 1 < n and above[j + 1]: j += 1
        xs, ys = x[i:j + 1], y[i:j + 1]; near = ys >= .9 * ys.max()
        if near.sum() >= 5:
            a, b, c = np.polyfit(xs[near], ys[near], 2); vertex = -b / (2 * a) if a < 0 else None
            if vertex is not None and xs[near][0] < vertex < xs[near][-1] and (j + 1 < n or vertex < xs[-1] - .02 * (x[-1] - x[0])): out.append(('max', float(vertex)))
        if j + 1 < n:   # the dip that follows, up to the next lobe
            k = j + 1
            while k < n and not above[k]: k += 1
            if k < n: out.append(('min', float(x[j + 1 + int(np.argmin(y[j + 1:k]))])))
        i = j + 1
    return out


def cubic_peak(x, y, lo, hi):
    """Maximum of the cubic polynomial fitted to y(x) on [lo, hi], as the paper reads its curves: x, or None when the cubic has no maximum inside."""
    keep = (x >= lo - 1e-9) & (x <= hi + 1e-9); c = np.polyfit(x[keep], y[keep] / y[keep].max(), 3); roots = np.roots(np.polyder(c))
    inside = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and lo < r.real < hi and np.polyval(np.polyder(c, 2), r.real) < 0]
    return inside[0] if inside else None


def analyse(name, case):
    job = JOBS / name; s = scene(case); f = case['frequency_mhz']; d = case['thickness_mm']; width = s['geometry']['width_mm']
    x = np.arange(10., case['reach_mm'] + 1e-9, STEP_MM); points = np.stack([x, np.full_like(x, width / 2), np.full_like(x, d)], axis=1); normal = np.array([0., 0., 1.])
    times, values, m = nonlinear.series(job / 'record', points, normal); full = values['murnaghan']; increment = full - values['linear']
    a1 = amplitude(times, full, f); beta = {'spectrum': amplitude(times, full, 2 * f) / a1 ** 2, 'increment': amplitude(times, increment, 2 * f) / a1 ** 2}
    # the software's own analysis takes at most 200 points on a line
    acc = nonlinear.accumulation(job, {'points': {'mode': 'line', 'start_mm': points[0].tolist(), 'end_mm': points[-1].tolist(), 'count': min(len(x), nonlinear.MAX_LINE)}, 'direction': [0., 0., 1.]})
    grid = {'spectrum': x, 'increment': x, 'window': x[0] + np.array([r['distance_mm'] for r in acc['rows']])}; beta['window'] = np.array([r['ratio_per_m'] if r['usable'] else np.nan for r in acc['rows']])
    plate = guided.plate_of(s); pair = next(r for r in guided._pairs(plate, f * 1e6) if (r['primary'], r['secondary']) == ('S0', 'S0'))
    lo, hi = case['simulated_mm']; out = {'frequency_mhz': f, 'thickness_mm': d, 'cycles': case.get('cycles', 10.), 'nodes': m['nodes'], 'elapsed_s': float((job / 'elapsed_s').read_text()),
                                         'h_mm': json.loads((job / 'request.json').read_text(encoding='utf-8'))['mesh_report']['h_mm'], 'layer_step_us': (m.get('time_plan') or {}).get('layer_step_us'),
                                         'experiment_mm': case['experiment_mm'], 'paper_simulation_mm': case['paper_simulation_mm'], 'measured_mm': case['measured_mm'], 'experiment_note': case['note'],
                                         **{k: case[k] for k in ('experiment_minimum_mm', 'experiment_maximum_mm') if k in case},
                                         'theory': {'half_beat_mm': pair['coherence_length_m'] * 1e3, 'delta_k_rad_mm': pair['delta_k_rad_m'] * 1e-3, 'phase_velocity_f_m_s': pair['phase_velocity_primary_m_s'],
                                                    'phase_velocity_2f_m_s': pair['phase_velocity_secondary_m_s'], 'group_velocity_f_m_s': pair['group_velocity_primary_m_s'], 'group_velocity_2f_m_s': pair['group_velocity_secondary_m_s']},
                                         'measured_in_model': {'wavenumber_f_difference': acc['fit']['f']['theory']['difference'], 'group_velocity_m_s': acc['group']['velocity_m_s']},
                                         'x_mm': x.tolist(), 'a1_m': a1.tolist(), 'maximum': {}}
    for key, y in beta.items():
        good = np.isfinite(y); xs, ys = grid[key][good], y[good]; found = lobes(xs, ys); maxima = [v for kind, v in found if kind == 'max']; minima = [v for kind, v in found if kind == 'min']
        out['maximum'][key] = {'first_mm': maxima[0] if found and found[0][0] == 'max' else None, 'first_minimum_mm': minima[0] if minima else None, 'second_mm': maxima[1] if len(maxima) > 1 else None,
                               'cubic_mm': cubic_peak(xs, ys, lo, min(hi, xs[-1])), 'last_usable_mm': float(xs[-1]), 'largest_per_m': float(ys.max())}
        out['beta_' + key + '_per_m'] = [None if not math.isfinite(v) else float(v) for v in y]
    out['x_window_mm'] = grid['window'].tolist()
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
        show = lambda v: '—' if v is None else f'{v:.1f}'
        print(f"{name}: experiment {r['experiment_mm']:g}  paper FE {r['paper_simulation_mm']:g}  theory π/|Δk| {t['half_beat_mm']:.1f}  | software first maximum: " +
              '  '.join(f"{key} {show(mx[key]['first_mm'])} (cubic {show(mx[key]['cubic_mm'])}; min {show(mx[key]['first_minimum_mm'])}, 2nd max {show(mx[key]['second_mm'])})" for key in mx) +
              f" | as measured: largest point {r['as_measured']['largest_point_mm']}, smallest {r['as_measured']['smallest_point_mm']}, cubic {show(r['as_measured']['cubic_mm'])}, rising at the end {r['as_measured']['rising_at_the_end']}"
              f" | k(f) {r['measured_in_model']['wavenumber_f_difference']:+.2e}, group {r['measured_in_model']['group_velocity_m_s']:.1f} vs {t['group_velocity_f_m_s']:.1f}", flush=True)
    RESULTS.mkdir(exist_ok=True); (RESULTS / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
