"""Automatic refinement checks: rerun a completed case with a finer mesh, a higher degree and a
smaller time step, then compare receiver waveforms and f/2f amplitudes against the base run.

Each level is an independent solver run of the same scene (space: h × factor; degree: +1 for
spectral engines; time: cfl × factor, or doubled Newmark substeps for the implicit hole solver).
Metrics follow spectral3d.compare: relative L2 of the projected waveform against the refined run,
with an absolute reporting floor for tiny signals, plus relative differences of the exact-frequency
f and 2f amplitudes in the analysis window. A level passes when every receiver and line point is
within tolerance; the case passes when all requested levels pass. Passing is evidence for this
case only, not a proof of convergence or a bound for other cases.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
from solid.schema import require
from spectral3d.compare import metric, FLOOR_M
from dispersion.analysis import harmonic_coefficients

DEFAULTS = {'levels': ['space', 'degree', 'time'], 'space_factor': 1 / math.sqrt(2), 'time_factor': .5, 'tolerance': {'waveform_relative_l2': .05, 'harmonic_relative': .05}, 'window_us': None}
LEVELS = ('space', 'degree', 'time')


def settings(spec, overrides=None):
    s = copy.deepcopy(DEFAULTS); given = copy.deepcopy(spec.get('convergence') or {}); given.pop('cases', None)
    tolerance = dict(s['tolerance'], **(given.pop('tolerance', None) or {})); s.update(given); s.update({k: v for k, v in (overrides or {}).items() if v is not None}); s['tolerance'] = tolerance
    require(isinstance(s['levels'], list) and s['levels'] and len(set(s['levels'])) == len(s['levels']) and all(level in LEVELS for level in s['levels']), '收敛层级需为 space、degree、time 的非空不重复列表。')
    require(all(type(s[k]) in (int, float) and 0 < s[k] < 1 for k in ('space_factor', 'time_factor')), '细化系数需在 (0, 1) 内。')
    require(set(tolerance) == {'waveform_relative_l2', 'harmonic_relative'} and all(type(v) in (int, float) and 0 < v <= 1 for v in tolerance.values()), '容差只接受 waveform_relative_l2、harmonic_relative，取值 (0, 1]。')
    w = s['window_us']; require(w is None or (isinstance(w, list) and len(w) == 2 and all(type(v) in (int, float) for v in w) and w[0] < w[1]), 'window_us 需为 [起点, 终点]（μs）。')
    return s


def selected_cases(spec, entries, requested=None, include_controls=True):
    """Case ids to check, in manifest order.

    `requested` (or the spec's convergence.cases): None for all completed cases, a count N (the first N),
    a list / comma-separated string of case ids, or a parameter filter {path: value or [values]}.
    Linear controls of selected nonlinear cases are added so the nonlinear increment can be checked."""
    completed = [e['case_id'] for e in entries]; requested = requested if requested is not None else (spec.get('convergence') or {}).get('cases')
    if isinstance(requested, str) and requested.strip().startswith('{'): requested = json.loads(requested)
    if requested is None: ids = list(completed)
    elif isinstance(requested, dict):
        require(requested and all(isinstance(k, str) for k in requested), '参数筛选需为 {参数路径: 取值或取值列表}。')
        match = lambda value, wanted: value in wanted if isinstance(wanted, list) else value == wanted
        ids = [e['case_id'] for e in entries if all(k in e['parameters'] and match(e['parameters'][k], v) for k, v in requested.items())]; require(ids, f'没有已完成用例满足参数筛选 {requested}。')
    elif type(requested) is int or (isinstance(requested, str) and requested.isdigit()): require(int(requested) >= 1, '收敛用例数需 ≥ 1。'); ids = completed[:int(requested)]
    else:
        ids = requested.split(',') if isinstance(requested, str) else requested; require(isinstance(ids, list) and all(i in completed for i in ids), f'收敛用例需为已完成用例：{[i for i in ids if i not in completed] if isinstance(ids, list) else ids}')
    if include_controls:
        for linear, nonlinear in control_pairs(entries):
            if nonlinear in ids and linear not in ids: ids.append(linear)
    chosen = set(ids); return [c for c in completed if c in chosen]


def variants(spec, scene, degree, substeps, conv):
    """Refined scene/settings per level; levels that do not apply to the engine are skipped with a reason."""
    engine = spec['engine']; rows = []
    for level in conv['levels']:
        variant = copy.deepcopy(scene); setting = {'degree': degree, 'substeps': substeps}; note = None
        if level == 'space': variant['numerics']['h_mm'] = scene['numerics']['h_mm'] * conv['space_factor']
        elif level == 'degree':
            if engine == '3d_tetra': rows.append({'level': level, 'skipped': 'linear tetrahedra have no degree'}); continue
            if degree >= 6: rows.append({'level': level, 'skipped': 'degree 6 is the maximum'}); continue
            setting['degree'] = degree + 1
        else:
            if engine == '3d_spectral' and scene['geometry']['kind'] == 'hole':
                if substeps * 2 > 32: rows.append({'level': level, 'skipped': 'Newmark substeps would exceed 32'}); continue
                setting['substeps'] = substeps * 2
            else:
                if scene['numerics']['cfl'] <= .05: rows.append({'level': level, 'skipped': 'CFL is already at the 0.05 minimum'}); continue
                variant['numerics']['cfl'] = max(.05, scene['numerics']['cfl'] * conv['time_factor'])
        rows.append({'level': level, 'scene': variant, 'settings': setting, 'skipped': None})
    return rows


def _projected(record, lines_npz):
    """Named projected waveforms of every receiver point and line point."""
    out = {}
    for rx in record['receivers']: out[rx['name']] = np.asarray(rx['projected_m'])
    for line in record['receiver_lines']:
        name = line['name']; projected = lines_npz[f'line_{name}_projected_m']
        for i in range(projected.shape[1]): out[f'{name}[{i}]'] = projected[:, i]
    return out


def _load(out):
    record = json.loads((Path(out) / 'result.json').read_text(encoding='utf-8'))
    return record, np.array(record['times_us']), _projected(record, np.load(Path(out) / 'lines.npz'))


def _compare(times, a, b, frequency_mhz, tolerance, window_us, orders_for_pass):
    """Waveform relative L2 and exact-frequency f/2f amplitudes, receiver by receiver; `orders_for_pass` picks the verdict terms."""
    require(set(a) == set(b), '细化运行的接收点集合不同。')
    start, end = (times[0], times[-1]) if window_us is None else window_us; rows = []; worst = {'waveform_relative_l2': 0., 'a1_relative': 0., 'a2_relative': 0.}
    for name in a:
        waveform = metric(a[name], b[name], tolerance['waveform_relative_l2'])
        ha = harmonic_coefficients(times, a[name], frequency_mhz, start, end)['harmonics']; hb = harmonic_coefficients(times, b[name], frequency_mhz, start, end)['harmonics']
        harmonic = {}
        for order, x, y in zip((1, 2), ha, hb):
            if not (x['available'] and y['available']): harmonic[f'a{order}'] = {'available': False, 'pass': True}; continue
            above = y['amplitude'] > FLOOR_M; relative = abs(x['amplitude'] - y['amplitude']) / y['amplitude'] if above else None
            harmonic[f'a{order}'] = {'available': True, 'base_m': x['amplitude'], 'refined_m': y['amplitude'], 'relative': relative, 'above_reporting_floor': above,
                                     'pass': bool(relative <= tolerance['harmonic_relative'] if above else abs(x['amplitude'] - y['amplitude']) <= FLOOR_M)}
            if above: worst[f'a{order}_relative'] = max(worst[f'a{order}_relative'], relative)
        if waveform['relative_l2'] is not None: worst['waveform_relative_l2'] = max(worst['waveform_relative_l2'], waveform['relative_l2'])
        verdict = ([waveform['pass']] if 'waveform' in orders_for_pass else []) + [harmonic[k]['pass'] for k in ('a1', 'a2') if k in orders_for_pass]
        rows.append({'receiver': name, 'waveform': waveform, 'harmonics': harmonic, 'pass': bool(all(verdict))})
    return {'receivers': rows, 'worst': worst, 'pass': all(r['pass'] for r in rows), 'window_us': [float(start), float(end)], 'frequency_mhz': frequency_mhz, 'tolerance': tolerance, 'verdict_terms': list(orders_for_pass)}


DIAGNOSTIC_KEYS = ('nodes', 'elements', 'internal_steps', 'internal_step_us', 'external_work_balance_relative', 'elapsed_s')


def compare_runs(base_out, refined_out, frequency_mhz, tolerance, window_us=None):
    base, times, a = _load(base_out); fine, fine_times, b = _load(refined_out)
    require(np.allclose(times, fine_times), '细化运行的输出时间网格不同，无法比较。')
    result = _compare(times, a, b, frequency_mhz, tolerance, window_us, ('waveform', 'a1', 'a2'))
    result.update(base_diagnostics={k: base['diagnostics'].get(k) for k in DIAGNOSTIC_KEYS}, refined_diagnostics={k: fine['diagnostics'].get(k) for k in DIAGNOSTIC_KEYS})
    return result


def control_pairs(entries):
    """(linear control, nonlinear case) ids for cases whose parameters differ only in `constitutive`."""
    pairs = []
    for a in entries:
        pa = dict(a['parameters'])
        if pa.pop('constitutive', None) != 'linear': continue
        for b in entries:
            pb = dict(b['parameters'])
            if pb.pop('constitutive', None) in ('murnaghan', 'stvk') and pa == pb: pairs.append((a['case_id'], b['case_id']))
    return pairs


def increment_comparison(linear_base, linear_refined, nonlinear_base, nonlinear_refined, frequency_mhz, tolerance, window_us=None):
    """Convergence of the nonlinear increment (nonlinear law minus linear control on the same discretisation).

    Short bursts carry linear 2f content far above the nonlinear second harmonic, so raw A2 agreement does
    not show that the nonlinear part is resolved; this compares the increment itself. Verdict on A2 only:
    the increment's waveform is reported but it also holds the f-band correction and static offsets."""
    loaded = [_load(p) for p in (linear_base, linear_refined, nonlinear_base, nonlinear_refined)]; times = loaded[0][1]
    require(all(np.allclose(times, t) for _, t, _ in loaded), '对照与非线性运行的输出时间网格不同。')
    require(all(set(x[2]) == set(loaded[0][2]) for x in loaded), '对照与非线性运行的接收点集合不同。')
    (_, _, lb), (_, _, lf), (_, _, nb), (_, _, nf) = loaded
    base = {k: nb[k] - lb[k] for k in lb}; fine = {k: nf[k] - lf[k] for k in lf}
    return _compare(times, base, fine, frequency_mhz, tolerance, window_us, ('a2',))


def refined(level, comparison):
    """The refined run must actually be finer in the quantity the level targets."""
    a = comparison['base_diagnostics']; b = comparison['refined_diagnostics']
    if level == 'time': return b['internal_step_us'] < a['internal_step_us'] * (1 - 1e-9)
    return (b['nodes'] or 0) > (a['nodes'] or 0)


def increment_summary(rows):
    checked = [r for r in rows if r.get('comparison')]
    return {'pass': bool(checked) and all(r['comparison']['pass'] for r in checked) and not any(r.get('error') for r in rows),
            'levels': {r['level']: ({'skipped': r['skipped']} if r.get('skipped') else {'error': r['error']} if r.get('error') else {'pass': r['comparison']['pass'], 'worst_a2_relative': r['comparison']['worst']['a2_relative']}) for r in rows},
            'worst_a2_relative': max([r['comparison']['worst']['a2_relative'] for r in checked], default=None)}


def report_summary(report):
    levels = {r['level']: ({'skipped': r['skipped']} if r.get('skipped') else {'error': r['error']} if r.get('error') else {'pass': r['comparison']['pass'], 'worst': r['comparison']['worst']}) for r in report['levels']}
    checked = [r for r in report['levels'] if r.get('comparison') and not r.get('error')]
    return {'pass': bool(checked) and all(r['comparison']['pass'] for r in checked) and not any(r.get('error') for r in report['levels']), 'levels': levels,
            'worst_waveform_relative_l2': max([r['comparison']['worst']['waveform_relative_l2'] for r in checked], default=None),
            'worst_a2_relative': max([r['comparison']['worst']['a2_relative'] for r in checked], default=None)}
