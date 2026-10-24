"""Per-case harmonics, accumulation fits, velocities and amplitude-scaling pairs across a campaign."""
import csv
import json
from pathlib import Path
import numpy as np
from solid.schema import require
from dispersion.lamb import dispersion_curves, mode_at
from dispersion.analysis import harmonic_coefficients, line_harmonics, accumulation_fit, line_velocities, amplitude_scaling
from physics.signals import support_us


def _curves(cache, thickness_mm, material):
    key = round(thickness_mm, 6)
    if key not in cache: cache[key] = dispersion_curves(thickness_mm, material, fd_max_mhz_mm=max(1., 6.), points=240)
    return cache[key]


def summarize_case(case_dir, analysis, cache):
    out = Path(case_dir) / 'out'; record = json.loads((out / 'result.json').read_text(encoding='utf-8')); scene = record['scene']; times = np.array(record['times_us'])
    frequency = analysis.get('frequency_mhz') or scene['source']['frequency_mhz']; thickness = scene['geometry']['thickness_mm']
    curves = _curves(cache, thickness, scene['material']); mode = analysis.get('mode_for_window', 'S0'); at = mode_at(curves, mode, frequency * thickness)
    row = {'case_id': Path(case_dir).name, 'frequency_mhz': frequency, 'thickness_mm': thickness, 'fd_mhz_mm': frequency * thickness, 'theory_mode': mode,
           'theory_phase_velocity_m_s': at['phase_velocity_m_s'] if at else None, 'theory_group_velocity_m_s': at['group_velocity_m_s'] if at else None,
           'energy_balance_relative': record['diagnostics']['external_work_balance_relative'], 'max_strain': record['diagnostics']['max_strain'], 'elapsed_s': record['diagnostics']['elapsed_s'],
           'nodes': record['diagnostics']['nodes'], 'internal_steps': record['diagnostics']['internal_steps'], 'receivers': [], 'line': None}
    start_us, end_us = support_us(scene['source']); window = analysis.get('point_window_us') or [times[0], times[-1]]
    for rx in record['receivers']:
        direction = np.asarray(rx['direction'])[[0, 2]]; projected = np.array(rx['u_x_m']) * direction[0] + np.array(rx['u_z_m']) * direction[1]
        h = harmonic_coefficients(times, projected, frequency, window[0], window[1])['harmonics']
        row['receivers'].append({'name': rx['name'], 'point_mm': rx['point_mm'], 'a1_m': h[0]['amplitude'], 'phase1_deg': h[0]['phase_deg'], 'a2_m': h[1]['amplitude'] if h[1]['available'] else None, 'phase2_deg': h[1].get('phase_deg')})
    lines = record['receiver_lines']
    if lines and (out / 'lines.npz').is_file():
        name = analysis.get('line') or lines[0]['name']; z = np.load(out / 'lines.npz')
        if f'line_{name}_projected_m' in z:
            distance = z[f'line_{name}_distance_mm']; traces = z[f'line_{name}_projected_m']; settings = analysis.get('window', {'mode': 'moving'})
            if settings.get('mode', 'moving') == 'moving':
                cg = settings.get('group_velocity_m_s') or (at['group_velocity_m_s'] if at else None); require(cg, f'移动窗需要群速度：{mode} 在 fd={frequency * thickness:.3g} 处无理论值，请改用固定窗或指定 group_velocity_m_s。')
                rows = line_harmonics(distance, times, traces, frequency, group_velocity_m_s=cg, burst_us=settings.get('burst_us') or (end_us - start_us), delay_us=scene['source']['delay_us'])
                velocity_window = None
            else:
                velocity_window = (settings.get('start_us', times[0]), settings.get('end_us', times[-1])); rows = line_harmonics(distance, times, traces, frequency, window_us=velocity_window)
            fit = accumulation_fit(rows); velocity = line_velocities(distance, times, traces, frequency, velocity_window)
            row['line'] = {'name': name, 'points': len(rows), 'distance_mm': fit['distance_mm'], 'a1_m': fit['a1'], 'a2_m': fit['a2'], 'ratio_per_m': fit['a2_over_a1_squared_per_m'],
                           'fit_ratio_vs_x': fit['fit_ratio_vs_x'], 'fit_a2_vs_x': fit['fit_a2_vs_x'], 'a1_variation': fit['a1_variation'],
                           'phase_velocity_m_s': velocity['phase_velocity_m_s'], 'phase_fit_r_squared': velocity['phase_fit_r_squared'], 'group_velocity_m_s': velocity['group_velocity_m_s'], 'rows': rows}
            row['_line'] = {'distance': distance, 'traces': traces, 'times': times, 'frequency': frequency, 'windows': [tuple(r['window_us']) for r in rows]}
    return row


def control_pairs(manifest, rows):
    """Cases differing only in `constitutive`: subtract the linear control on the same mesh and time step.

    The difference removes the linear spectral content of the burst at 2f (which scales like A, not A^2)
    and leaves the material plus geometric nonlinear increment; it is the sweep version of the
    five-control comparison used by the 3-D workbench."""
    by_id = {r['case_id']: r for r in rows}; entries = [c for c in manifest['cases'] if c['case_id'] in by_id]; pairs = []
    for a in entries:
        for b in entries:
            pa = dict(a['parameters']); pb = dict(b['parameters'])
            if pa.pop('constitutive', None) != 'linear' or pb.pop('constitutive', None) not in ('murnaghan', 'stvk') or pa != pb: continue
            linear = by_id[a['case_id']]; nonlinear = by_id[b['case_id']]
            if not (linear.get('_line') and nonlinear.get('_line')): continue
            L = linear['_line']; N = nonlinear['_line']
            if L['traces'].shape != N['traces'].shape or not np.allclose(L['distance'], N['distance']): continue
            increment = N['traces'] - L['traces']; harmonics = []
            for i, (start, end) in enumerate(N['windows']):
                h = harmonic_coefficients(N['times'], increment[:, i], N['frequency'], start, end)['harmonics']
                harmonics.append({'index': i, 'distance_mm': float(N['distance'][i]), 'window_us': [start, end], 'harmonics': h})
            fit = accumulation_fit(harmonics)
            nonlinear['line']['nonlinear_increment'] = {'control_case': a['case_id'], 'a1_m': fit['a1'], 'a2_m': fit['a2'], 'fit_a2_vs_x': fit['fit_a2_vs_x'], 'rows': harmonics,
                                                        'note': 'nonlinear-law minus linear control on one mesh; A1 of the increment should be small, A2 is the nonlinear second harmonic'}
            pairs.append({'nonlinear_case': b['case_id'], 'linear_control': a['case_id'], 'constitutive': b['parameters']['constitutive'], 'increment_a2_first_m': fit['a2'][0], 'increment_a2_last_m': fit['a2'][-1], 'increment_fit_r_squared': fit['fit_a2_vs_x']['r_squared']})
    return pairs


def scaling_pairs(manifest, rows):
    """Cases whose parameters differ only in source.amplitude_mpa by a factor of two.

    Raw second harmonics include the burst's own linear content at 2f (ratio 2); when a linear
    control exists the check is repeated on the nonlinear increment, which must scale by 4."""
    by_id = {r['case_id']: r for r in rows}; pairs = []
    entries = [c for c in manifest['cases'] if c['case_id'] in by_id]
    for a in entries:
        for b in entries:
            pa = dict(a['parameters']); pb = dict(b['parameters']); amp_a = pa.pop('source.amplitude_mpa', None); amp_b = pb.pop('source.amplitude_mpa', None)
            if pa != pb or amp_a is None or amp_b is None or abs(amp_b - 2 * amp_a) > 1e-9: continue
            ra = by_id[a['case_id']]; rb = by_id[b['case_id']]
            if ra['line'] and rb['line']: check = amplitude_scaling(ra['line']['rows'], rb['line']['rows'])
            else:
                checks = [{'name': x['name'], 'a1_ratio': y['a1_m'] / x['a1_m'], 'a2_ratio': (y['a2_m'] / x['a2_m']) if x['a2_m'] and y['a2_m'] else None} for x, y in zip(ra['receivers'], rb['receivers'])]
                for c in checks: c['a1_pass'] = abs(c['a1_ratio'] - 2) <= .2; c['a2_pass'] = c['a2_ratio'] is not None and abs(c['a2_ratio'] - 4) <= .4
                check = {'tolerance': .1, 'checks': checks, 'all_pass': all(c['a1_pass'] and c['a2_pass'] for c in checks)}
            pair = {'case_a': a['case_id'], 'case_2a': b['case_id'], 'amplitude_mpa': [amp_a, amp_b], 'basis': 'raw', 'all_pass': check['all_pass'], 'checks': check['checks'][:: max(1, len(check['checks']) // 8)]}
            inc_a = (ra.get('line') or {}).get('nonlinear_increment'); inc_b = (rb.get('line') or {}).get('nonlinear_increment')
            if inc_a and inc_b:
                increment = amplitude_scaling(inc_a['rows'], inc_b['rows'])   # only the 2f check applies: the increment's 1f part is itself nonlinear
                pair['increment'] = {'basis': 'nonlinear_increment', 'all_pass': all(c['a2_pass'] for c in increment['checks']), 'a2_ratios': [c['a2_ratio'] for c in increment['checks']]}
            pairs.append(pair)
    return pairs


def summarize(campaign_dir, analysis=None):
    campaign_dir = Path(campaign_dir); manifest = json.loads((campaign_dir / 'manifest.json').read_text(encoding='utf-8'))
    spec = json.loads((campaign_dir / 'spec.json').read_text(encoding='utf-8')); analysis = analysis or spec.get('analysis', {}); cache = {}
    rows = []
    for entry in manifest['cases']:
        if entry.get('phase') != 'completed': continue
        row = summarize_case(campaign_dir / 'cases' / entry['case_id'], analysis, cache); row['parameters'] = entry['parameters']; row['split'] = entry['split']; rows.append(row)
    controls = control_pairs(manifest, rows); pairs = scaling_pairs(manifest, rows)
    def public_line(line):
        out = {k: v for k, v in line.items() if k != 'rows'}
        if 'nonlinear_increment' in out: out['nonlinear_increment'] = {k: v for k, v in out['nonlinear_increment'].items() if k != 'rows'}
        return out
    summary = {'schema': 'workai.sweep.summary.v1', 'campaign': manifest['name'], 'analysis': analysis, 'completed_cases': len(rows), 'failed_cases': sum(1 for c in manifest['cases'] if c.get('phase') == 'failed'),
               'rows': [{k: v for k, v in r.items() if k not in ('line', '_line')} | {'line': public_line(r['line']) if r['line'] else None} for r in rows], 'control_pairs': controls, 'amplitude_scaling_pairs': pairs,
               'note': 'Harmonic amplitudes depend on the analysis window; A2/A1^2 slopes compare cases under one window setting and are not material beta. Raw 2f content of short bursts scales like A; use linear controls (constitutive grid) to isolate the nonlinear increment.'}
    (campaign_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    parameters = sorted({k for r in rows for k in r['parameters']})
    with (campaign_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'split'] + parameters + ['frequency_mhz', 'thickness_mm', 'fd_mhz_mm', 'theory_cp_m_s', 'theory_cg_m_s', 'line_cp_m_s', 'line_phase_r2', 'line_cg_m_s', 'line_a1_first_m', 'line_a1_last_m', 'line_a2_first_m', 'line_a2_last_m', 'ratio_slope_per_m', 'ratio_r2', 'a2_fit_r2', 'increment_a2_first_m', 'increment_a2_last_m', 'increment_a2_fit_r2', 'energy_balance', 'max_strain', 'nodes', 'internal_steps', 'elapsed_s'])
        for r in rows:
            line = r['line'] or {}; inc = line.get('nonlinear_increment') or {}
            writer.writerow([r['case_id'], r['split']] + [r['parameters'].get(p, '') for p in parameters] + [r['frequency_mhz'], r['thickness_mm'], r['fd_mhz_mm'], r['theory_phase_velocity_m_s'], r['theory_group_velocity_m_s'],
                             line.get('phase_velocity_m_s'), line.get('phase_fit_r_squared'), line.get('group_velocity_m_s'), (line.get('a1_m') or [None])[0], (line.get('a1_m') or [None])[-1], (line.get('a2_m') or [None])[0], (line.get('a2_m') or [None])[-1],
                             (line.get('fit_ratio_vs_x') or {}).get('slope_per_m'), (line.get('fit_ratio_vs_x') or {}).get('r_squared'), (line.get('fit_a2_vs_x') or {}).get('r_squared'),
                             (inc.get('a2_m') or [None])[0], (inc.get('a2_m') or [None])[-1], (inc.get('fit_a2_vs_x') or {}).get('r_squared'), r['energy_balance_relative'], r['max_strain'], r['nodes'], r['internal_steps'], r['elapsed_s']])
    return summary
