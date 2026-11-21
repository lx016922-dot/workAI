"""Post-processing for nonlinear guided waves, on the record of a finished computation (nothing is recomputed).

* `accumulation`: the fundamental and the second harmonic along a line of points — A1, A2 of the complete response
  and of the nonlinear increments, the relative nonlinearity parameter A2/A1² with a straight-line fit against the
  propagation distance, the wavenumbers at f and 2f from the slope of the phases, and the group velocity from the
  arrival of the envelope. With a plate, the measured wavenumbers are set against the theoretical modes
  (`offline.guided`).
* `fk`: the frequency–wavenumber spectrum along the line, with the theoretical dispersion curves to draw over it.
* `dispersion`: the theoretical curves, the modes at f and 2f and the phase-matching table of the model's plate.

Amplitudes are those of the result page (`offline.scan.harmonics`): a Hann window over the analysis window, the
weighted mean removed, the coefficient at the exact frequency, the phase referred to the start of the record. A2 is
taken from the nonlinear increment (complete − linear) whenever the linear control was computed: the complete
response also holds, at 2f, the linear spectrum of the tone burst itself, which is not a nonlinear effect.

A2/A1² depends on the length of the analysis window when the burst is shorter than the window (both amplitudes are
averages over the window), so values are comparable between computations only for the same window length; its
growth with distance does not have this dependence. It is a relative parameter of this model, source and receiver
direction — not the material's β.
"""
import base64
import json
import math
import threading
from pathlib import Path
import numpy as np
from .scan import harmonics, number

MAX_LINE, MAX_FK = 200, 256
SAMPLES = 6_000_000          # frames × points held in memory for one analysis (each of the controls)
_cache = {'key': None, 'value': None}
_lock = threading.Lock()


def _vector(value, what, unit=False):
    if not (isinstance(value, list) and len(value) == 3 and all(number(v) for v in value)): raise ValueError(f'{what}需要三个有限数值。')
    v = np.array(value, dtype=float)
    if unit:
        if np.linalg.norm(v) < 1e-12: raise ValueError(f'{what}不能为零向量。')
        v = v / np.linalg.norm(v)
    return v


def line_of(body, scene, compact, most):
    """The points of an analysis: (points mm (n × 3), distances mm, unit vector of the line, names or None, notes).

    `body['points']`: {'mode': 'line', 'start_mm', 'end_mm', 'count'} — equally spaced points, or {'mode': 'receivers'} —
    the receivers of the computation, ordered along the line from the first to the last one."""
    spec = body.get('points') or {'mode': 'receivers'}; notes = []
    if not isinstance(spec, dict) or spec.get('mode') not in ('line', 'receivers'): raise ValueError('点列需为线段等距采样（line）或计算的接收点（receivers）。')
    if spec['mode'] == 'line':
        if compact:
            raise ValueError('这次计算保存的是精简记录，只有接收点有完整波形：请选择“使用接收点”，或在“求解器配置”里把记录方式改为“完整记录”后重新计算。')
        start, end = _vector(spec.get('start_mm'), '起点'), _vector(spec.get('end_mm'), '终点'); count = spec.get('count', 'auto')
        length = float(np.linalg.norm(end - start))
        if length < 1e-6: raise ValueError('起点和终点重合。')
        if count == 'auto': count = automatic_count(scene, (end - start) / length, length, most)
        if type(count) is not int or not 2 <= count <= most: raise ValueError(f'采样点数需为 2–{most} 的整数，或 auto（按 2f 处最慢模态的波长自动选取）。')
        fraction = np.arange(count) / (count - 1)
        return start + fraction[:, None] * (end - start), fraction * length, (end - start) / length, None, notes
    receivers = scene['receivers']
    if len(receivers) < 2: raise ValueError('沿程分析至少需要两个接收点：请在模型里加一排接收点，或改用线段采样（需要完整记录）。')
    points = np.array([r['point_mm'] for r in receivers], dtype=float); span = points[-1] - points[0]; length = float(np.linalg.norm(span))
    if length < 1e-6: raise ValueError('首末两个接收点重合，不能确定沿程方向。')
    e = span / length; distance = (points - points[0]) @ e; off = np.linalg.norm(points - points[0] - distance[:, None] * e, axis=1); order = np.argsort(distance, kind='stable')
    if np.min(np.diff(distance[order])) < 1e-6: raise ValueError('有两个接收点在沿程方向上重合。')
    if off.max() > 1e-6 * max(length, 1.): notes.append(f'接收点不在一条直线上（最远偏离 {off.max():.3g} mm）：沿程距离取它们在首末接收点连线上的投影。')
    return points[order], distance[order] - distance[order][0], e, [receivers[i]['name'] for i in order], notes


def automatic_count(scene, e, length_mm, most):
    """Points on a line so that neighbours are at most a fifth of the shortest wavelength at 2f apart (41 without a
    plate theory), within 21…`most`."""
    plate, _ = _plate(scene, e, 2.6 * float(scene['source']['frequency_mhz']))
    if plate is None: return min(most, 41)
    from . import guided
    slowest = max((mode['k_rad_m'] for mode in guided._at(plate, 2e6 * float(scene['source']['frequency_mhz']))), default=0.) * 1e-3   # rad/mm
    return int(min(most, max(21, math.ceil(length_mm * slowest / (.4 * math.pi)) + 1)))


def series(record, points, direction, first=0, last=None):
    """(times μs, {control: frames × points}, manifest): the displacement along `direction` at the points, frames
    `first`…`last` of the record, read in one pass over its chunks."""
    from .recording import chunks, controls_of, is_compact, manifest
    record = Path(record); m = manifest(record); controls = controls_of(m); last = m['frame_count'] if last is None else last; frames = last - first
    if frames * len(points) > SAMPLES:
        raise ValueError(f'这次分析要读入 {frames} 个采样点 × {len(points)} 个点，超过内存预算（{SAMPLES:,}）：请减少点数，或缩短分析的时间范围。')
    key = (m['fingerprint'], np.asarray(points).tobytes(), np.asarray(direction).tobytes(), first, last)
    with _lock:
        if _cache['key'] == key: return _cache['value']
    values = {name: np.empty((frames, len(points))) for name in controls}
    if is_compact(m):
        rows = []
        for p in points:
            hit = [i for i, r in enumerate(m['receivers']) if max(abs(a - b) for a, b in zip(r['point_mm'], p)) <= 1e-6]
            if not hit: raise ValueError('精简记录里没有这个点的波形（只有接收点有）。')
            rows.append(hit[0])
        take = lambda data, name, lo, hi: data['probes'][lo:hi, rows, controls.index(name)] @ direction
    else:
        from spectral3d.mesh import interpolation
        with np.load(record / 'mesh.npz', allow_pickle=False) as data: mesh = dict(data)
        locations = []
        for i, p in enumerate(points):
            try: locations.append(interpolation(mesh, p))
            except ValueError: raise ValueError(f'第 {i + 1} 个点（{p[0]:.4g}, {p[1]:.4g}, {p[2]:.4g}）mm 不在材料内：它落在板外、孔里或缺口里。') from None
        from scipy.sparse import csr_matrix
        ids = np.unique(np.concatenate([l[0] for l in locations]))
        weights = csr_matrix((np.concatenate([l[1] for l in locations]), (np.repeat(np.arange(len(points)), [len(l[0]) for l in locations]), np.searchsorted(ids, np.concatenate([l[0] for l in locations])))),
                             shape=(len(points), len(ids)))   # points × nodes used; duplicate entries add
        take = lambda data, name, lo, hi: (weights @ (data[name][lo:hi][:, ids] @ direction).T).T
    for row, data in chunks(record, m):
        start = row['first_frame']; end = start + row['frame_count']; lo, hi = max(start, first), min(end, last)
        if lo >= hi: continue
        for name in controls: values[name][lo - first:hi - first] = take(data, name, lo - start, hi - start)
    out = ((first + np.arange(frames)) * m['sample_step_us'], values, m)
    with _lock: _cache.update(key=key, value=out)
    return out


def envelope(values):
    """Magnitude of the analytic signal of the columns of `values`."""
    n = len(values); spectrum = np.fft.fft(values, axis=0); h = np.zeros(n); h[0] = 1; h[1:(n + 1) // 2] = 2
    if n % 2 == 0: h[n // 2] = 1
    return np.abs(np.fft.ifft(spectrum * h[:, None], axis=0))


def arrivals(times, env, period_us):
    """When the first wave packet passes each point: the first maximum of the envelope (smoothed over one period of
    the carrier) that reaches half of its largest value. The largest value itself may belong to a later overlap of the
    incident wave and a reflection."""
    step = times[1] - times[0]; n = max(1, int(round(period_us / step))); kernel = np.ones(n) / n; out = np.empty(env.shape[1])
    for j in range(env.shape[1]):
        e = np.convolve(env[:, j], kernel, mode='same'); top = float(e.max())
        peaks = np.flatnonzero((e[1:-1] >= .5 * top) & (e[1:-1] >= e[:-2]) & (e[1:-1] > e[2:])) + 1
        out[j] = times[peaks[0] if len(peaks) else int(np.argmax(e))]
    return out


def wavenumber(x, phase, frequency_hz, plate, label, notes, secondary=False, forced=None):
    """Wavenumber (rad/m) from the slope of the unwrapped phase along the line, set against the theoretical modes.

    `secondary`: the field is a second harmonic, which lives only in the modes a second harmonic can have (symmetric
    Lamb modes); `forced` (rad/m): twice the wavenumber of the fundamental, the wavenumber of the second harmonic
    while it is still tied to the fundamental. With a phase mismatch the 2f field is this forced wave plus a free
    mode, and the slope of its phase lies between the two — nearer to the larger one.

    Unwrapping assumes less than half a turn between neighbouring points. When the points are further apart than half
    the shortest wavelength that could be present, the slope is only known up to multiples of 2π/Δx; for equally
    spaced points on a plate the multiple that lands on a theoretical mode (or on `forced`) within 2 %, if exactly one
    does, is taken and reported as `alias_turns`. Without a theory nothing can tell: `unambiguous_below_rad_mm` (π/Δx)
    is the largest wavenumber the points determine."""
    from . import guided
    fit = linear_fit(x, phase); k = -fit['slope']; spacing = np.diff(x); dx = float(spacing.max()); uniform = bool(np.ptp(spacing) <= 1e-6 * dx)
    out = {'wavenumber_rad_mm': k * 1e-3, 'r_squared': fit['r_squared'], 'theory': None, 'alias_turns': 0, 'reliable': True, 'unambiguous_below_rad_mm': math.pi / dx * 1e-3}
    modes = [m for m in guided._at(plate, frequency_hz) if not m['backward'] and (m['secondary'] or not secondary)] if plate else []
    targets = [(m['name'], m['k_rad_m']) for m in modes] + ([('受迫波 2k(f)', forced)] if forced else [])
    largest = max((v for _, v in targets), default=abs(k))
    if modes and largest * dx > math.pi:   # the phase may have wrapped between neighbours
        turns = set(); names = []
        if uniform:
            turn = 2 * math.pi / dx
            for n in range(-int(largest / turn) - 2, int(largest / turn) + 3):
                hit = [name for name, value in targets if abs((k + n * turn) / value - 1) < .02]
                if hit: turns.add(n); names += hit
        if len(turns) == 1:
            n = turns.pop(); k += n * turn; out.update(wavenumber_rad_mm=k * 1e-3, alias_turns=n)
            if n: notes.append(f'{label} 处点距 {dx * 1e3:.3g} mm 大于最短波长的一半，相位斜率只确定到 2π/Δx 的整数倍；按唯一落在理论值（{"、".join(names)}）上的那一个取了波数。加密点列可以不依赖理论。')
        else:
            out['reliable'] = False
            notes.append(f'{label} 处点距 {dx * 1e3:.3g} mm 大于最短波长的一半（{math.pi / largest * 1e3:.3g} mm），相位可能绕卷，波数不可靠。请加密点列。')
    elif not modes and abs(k) * dx > .8 * math.pi:
        out['reliable'] = False; notes.append(f'{label} 处相邻点的相位差接近 180°：点距 {dx * 1e3:.3g} mm 对这个波数太大，波数不可靠。请加密点列。')
    if modes and out['reliable'] and k > 0:
        best = min(modes, key=lambda m: abs(m['k_rad_m'] - k)); difference = k / best['k_rad_m'] - 1
        out['theory'] = {'mode': best['name'], 'wavenumber_rad_mm': best['k_rad_m'] * 1e-3, 'phase_velocity_m_s': best['phase_velocity_m_s'], 'group_velocity_m_s': best['group_velocity_m_s'], 'difference': difference,
                         'matched': bool(abs(difference) < .02)}
    if forced and out['reliable']: out['forced'] = {'wavenumber_rad_mm': forced * 1e-3, 'difference': k / forced - 1}
    out['phase_velocity_m_s'] = 2 * math.pi * frequency_hz / k if k > 0 and out['reliable'] else None
    return out


def linear_fit(x, y):
    """Least-squares line: {'slope', 'intercept', 'r_squared'} (r² is None for fewer than three points or a constant y)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2: return None
    slope, intercept = np.polyfit(x, y, 1); residual = float(((y - slope * x - intercept) ** 2).sum()); total = float(((y - y.mean()) ** 2).sum())
    return {'slope': float(slope), 'intercept': float(intercept), 'r_squared': 1 - residual / total if len(x) > 2 and total > 0 else None}


def _job(job):
    job = Path(job); request = json.loads((job / 'request.json').read_text(encoding='utf-8'))
    return job, request['scene'], float(request['scene']['source']['frequency_mhz'])


def _plate(scene, e, top_mhz):
    """(plate, None) for a line in the plane of a plate, else (None, why there is no theory)."""
    from . import guided
    if abs(e[2]) > 1e-6: return None, '分析线不在板面内（有厚度方向分量），不与平板导波理论对照。'
    try: return guided.plate_of(scene, math.degrees(math.atan2(e[1], e[0])), top_mhz * 1e6), None
    except ValueError as error: return None, str(error)


def accumulation(job, body):
    """A1, A2 and A2/A1² along a line, with the fits. `body`: {'points', 'direction', 'window', 'fit_mm'}.

    `window`: {'mode': 'fixed', 'start_us', 'end_us'} — the same window at every point, or {'mode': 'follow',
    'speed_m_s' (optional), 'length_us' (optional)} — a window that travels with the wave packet: centred on the
    envelope peak at the first point and moved by distance / speed. Without a speed the measured group velocity is
    used; without a length, 1.5 times the duration of the excitation signal."""
    from .domain import parts_for, response
    from .recording import controls_of, is_compact, manifest
    from physics.signals import support_us
    if not isinstance(body, dict) or set(body) - {'points', 'direction', 'window', 'fit_mm', 'receiver'}: raise ValueError('沿程分析只接受 points、direction、window、fit_mm、receiver。')
    job, scene, f = _job(job); m = manifest(job / 'record'); compact = is_compact(m)
    points, distance, e, names, notes = line_of(body, scene, compact, MAX_LINE)
    direction = _vector(body.get('direction', scene['receivers'][0]['direction']), '接收方向', unit=True)
    times, values, m = series(job / 'record', points, direction); controls = controls_of(m); step = m['sample_step_us']
    values, kept, receiver = aperture(times, values, distance, body.get('receiver'), step)   # a finite receiver reads the line through its footprint
    if receiver['type'] != 'point':
        points = [points[i] for i in kept]; distance = distance[kept]; names = [names[i] for i in kept] if names else names
    have = lambda part: part in parts_for(controls) or (part == 'nonlinear' and {'linear', 'murnaghan'} <= set(controls))
    part = lambda name: response(values, name)
    full = part('full'); basis = 'nonlinear' if have('nonlinear') else 'full'
    # ---- group velocity: arrival of the envelope of the fundamental ----
    arrival = arrivals(times, envelope(part('linear') if have('linear') else full), 1 / f)
    group = linear_fit(distance * 1e-3, arrival * 1e-6); speed_measured = 1 / group['slope'] if group and group['slope'] > 0 else None
    # ---- analysis windows ----
    window = body.get('window') or {'mode': 'follow'}
    if not isinstance(window, dict) or window.get('mode') not in ('fixed', 'follow'): raise ValueError('时间窗需为同一时间窗（fixed）或跟随波包（follow）。')
    if window['mode'] == 'fixed':
        if not (number(window.get('start_us')) and number(window.get('end_us')) and window['start_us'] < window['end_us']): raise ValueError('时间窗需要起点早于终点的两个时刻（μs）。')
        windows = [(float(window['start_us']), float(window['end_us']))] * len(points); used = {'mode': 'fixed', 'start_us': float(window['start_us']), 'end_us': float(window['end_us'])}
    else:
        speed = window.get('speed_m_s'); length = window.get('length_us'); lo, hi = support_us(scene['source'])
        if speed is not None and not (number(speed) and 10 <= speed <= 2e4): raise ValueError('跟随速度需在 10–20000 m/s。')
        if length is not None and not (number(length) and length > 0): raise ValueError('时间窗长度需为正数（μs）。')
        if speed is None and speed_measured is None: raise ValueError('没有量出沿这条线向前传播的波包（包络到达时刻不随距离增加）：请手动填写跟随速度，或改用同一时间窗。')
        speed_used = float(speed) if speed is not None else speed_measured; length = float(length) if length is not None else 1.5 * (hi - lo)
        windows = [(arrival[0] + d * 1e-3 / speed_used * 1e6 - length / 2, arrival[0] + d * 1e-3 / speed_used * 1e6 + length / 2) for d in distance]
        used = {'mode': 'follow', 'speed_m_s': speed_used, 'speed_source': 'given' if speed is not None else 'measured', 'length_us': length, 'centre_first_us': float(arrival[0])}
    # ---- harmonics ----
    wanted = [p for p in ('full', 'nonlinear', 'geometric', 'material') if have(p)]; data = {p: part(p) for p in wanted}; rows = []
    # The plate is at rest before the record starts: a window that begins earlier sees zeros there.
    lead = max(0, int(math.ceil((times[0] - min(w[0] for w in windows)) / step - 1e-9)))
    if lead:
        times = np.concatenate([times[0] - step * np.arange(lead, 0, -1), times]); data = {p: np.concatenate([np.zeros((lead, v.shape[1])), v]) for p, v in data.items()}
    for i in range(len(points)):
        start, end = windows[i]; row = {'index': i + 1, 'point_mm': [float(v) for v in points[i]], 'distance_mm': float(distance[i]), 'window_us': [float(start), float(end)], 'arrival_us': float(arrival[i])}
        if names: row['name'] = names[i]
        if end > times[-1] + 1e-9 or (end - start) / step < 8:
            row.update(usable=False, reason='时间窗超出记录的末尾' if (end - start) / step >= 8 else '时间窗不足 8 个采样点'); rows.append(row); continue
        found = {p: harmonics(times, data[p][:, i], f, (start, end)) for p in wanted}; second = found[basis][1]
        if second is None: raise ValueError('2f 超过了记录的 Nyquist 频率：请用更小的采样间隔重新计算。')
        a1 = found['full'][0]['amplitude_m']; a2 = second['amplitude_m']
        row.update(usable=True, a1_m=a1, phase1_deg=found['full'][0]['phase_deg'], a2_m=found['full'][1]['amplitude_m'], a2_basis_m=a2, phase2_deg=second['phase_deg'],
                   ratio_per_m=a2 / a1 ** 2 if a1 > 0 else None, **{f'a2_{p}_m': found[p][1]['amplitude_m'] for p in wanted if p != 'full'})
        rows.append(row)
    good = [r for r in rows if r['usable'] and r['ratio_per_m'] is not None and r['phase1_deg'] is not None and r['phase2_deg'] is not None]
    if len(good) < 2: raise ValueError('可用的点不足两个：' + ('时间窗超出了记录的末尾，请缩短线段、调整时间窗或跟随速度。' if any(not r['usable'] for r in rows) else '信号幅值为零。'))
    # ---- unwrapped phases along the line ----
    for key in ('phase1', 'phase2'):
        unwrapped = np.unwrap(np.radians([r[key + '_deg'] for r in good]))
        for r, v in zip(good, unwrapped): r[key + '_unwrapped_rad'] = float(v)
    # ---- fits ----
    fit_range = body.get('fit_mm')
    if fit_range is not None and not (isinstance(fit_range, list) and len(fit_range) == 2 and all(number(v) for v in fit_range) and fit_range[0] < fit_range[1]): raise ValueError('拟合范围需要起点小于终点的两个沿程距离（mm）。')
    chosen = [r for r in good if fit_range is None or fit_range[0] - 1e-9 <= r['distance_mm'] <= fit_range[1] + 1e-9]
    if len(chosen) < 2: raise ValueError('拟合范围内的可用点不足两个。')
    x = np.array([r['distance_mm'] for r in chosen]) * 1e-3; a1 = np.array([r['a1_m'] for r in chosen])
    ratio = linear_fit(x, [r['ratio_per_m'] for r in chosen]); a2 = linear_fit(x, [r['a2_basis_m'] for r in chosen])
    plate, why = _plate(scene, e, 2.6 * f)
    if why: notes.append(why)
    wave = {1: wavenumber(x, [r['phase1_unwrapped_rad'] for r in chosen], f * 1e6, plate, 'f', notes)}
    twice = 2e3 * wave[1]['wavenumber_rad_mm'] if wave[1]['reliable'] and wave[1]['wavenumber_rad_mm'] > 0 else None
    wave[2] = wavenumber(x, [r['phase2_unwrapped_rad'] for r in chosen], 2 * f * 1e6, plate, '2f', notes, secondary=True, forced=twice)
    # the phase matching the theory gives the primary mode that was found (the measured 2f slope cannot give it: see `wavenumber`)
    pair = None
    if plate and wave[1]['theory'] and wave[1]['theory']['matched']:
        from . import guided
        best = next((r for r in guided._pairs(plate, f * 1e6) if r['primary'] == wave[1]['theory']['mode']), None)
        if best: pair = {'primary': best['primary'], 'secondary': best['secondary'], 'mismatch': best['mismatch'], 'delta_k_rad_mm': best['delta_k_rad_m'] * 1e-3,
                         'coherence_length_mm': None if best['coherence_length_m'] is None else best['coherence_length_m'] * 1e3,
                         'group_velocity_primary_m_s': best['group_velocity_primary_m_s'], 'group_velocity_secondary_m_s': best['group_velocity_secondary_m_s']}
    variation = float((a1.max() - a1.min()) / a1.mean())
    attenuation = _attenuation(scene, f, e, points, distance, rows, chosen, x, a1, pair, wave[1], speed_measured, notes)
    if basis == 'full': notes.append('这次计算没有线性控制：A₂ 取自完整响应，2f 处还含有猝发音自身的线性频谱，A₂/A₁² 不能当作非线性的度量。请在“求解器配置”里选“完整 + 线性”或“全部”后重新计算。')
    if variation > .1: notes.append(f'拟合范围内基频幅值 A₁ 变化了 {variation:.0%}（波束扩散、多模态叠加或反射）：A₂/A₁² 只部分抵消这种变化。')
    if ratio['r_squared'] is not None and ratio['r_squared'] < .98: notes.append(f"A₂/A₁² 随距离不是直线（R² = {ratio['r_squared']:.3f}）：相位失配时二次谐波会先增后减，多模态时会出现拍。")
    if any(not r['usable'] for r in rows): notes.append(f"{sum(1 for r in rows if not r['usable'])} 个点的时间窗超出记录的末尾，未参与分析。")
    if scene['source'].get('signal', {}).get('type', 'tone_burst') not in ('tone_burst', 'gaussian_pulse'): notes.append('激励不是单一载频的猝发音：f 取主激励的 frequency_mhz，f 与 2f 的系数只在信号有这个载频时才有意义。')
    return {'schema': 'workai.nonlinear.accumulation.v1', 'job_id': job.name, 'frequency_mhz': f, 'direction': [float(v) for v in direction], 'line_direction': [float(v) for v in e],
            'points_mode': (body.get('points') or {'mode': 'receivers'})['mode'], 'record_mode': 'compact' if compact else 'full', 'controls': list(controls), 'a2_basis': basis, 'window': used, 'receiver': receiver,
            'group': {'velocity_m_s': speed_measured, 'r_squared': group['r_squared'] if group else None}, 'rows': rows,
            'fit': {'range_mm': [float(x[0] * 1e3), float(x[-1] * 1e3)], 'points': len(chosen),
                    'ratio': {'slope_per_m2': ratio['slope'], 'intercept_per_m': ratio['intercept'], 'r_squared': ratio['r_squared']},
                    'a2': {'slope_m_per_m': a2['slope'], 'intercept_m': a2['intercept'], 'r_squared': a2['r_squared']},
                    'f': wave[1], '2f': wave[2], 'pair': pair, 'a1_variation': variation, 'attenuation': attenuation},
            'notes': notes,
            'note': 'A₂/A₁² 是这个模型、激励和接收方向下的相对非线性参数，不是材料的 β；它的数值与分析时间窗的长度有关，不同算例之间要用同一窗长比较，随距离的增长不受此影响。'
                    '2f 的波数取自 2f 系数的相位斜率：相位匹配时它等于二次模态的波数；失配时 2f 场是随基波走的受迫波（波数 2k(f)）与自由的二次模态之和，'
                            '相位斜率落在两者之间，偏向幅值大的一个，两个成分要在 f–k 谱里分开看。'}


RECEIVERS = ('point', 'wedge', 'patch')


def aperture(times, values, distance_mm, receiver, step_us):
    """What a receiver of finite size reads at every position of an equally spaced line: ({control: frames × positions}, the
    indices of the points that are centres of a whole footprint, the receiver as used).

    * `point` — the displacement at the point (a laser spot): nothing changes.
    * `wedge` — an angle-beam receiver of footprint `length_mm` and trace velocity `trace_velocity_m_s`: the mean over the
      footprint of the displacement, each point advanced by its distance from the centre over the trace velocity. A wave
      whose phase velocity along the line is the trace velocity adds up in phase; others cancel in part. `apodization`
      'hann' weights the footprint with a Hann window.
    * `patch` — a bonded wafer of length `length_mm` in the pin-force picture: the displacement at its far edge minus that
      at its near edge, which is its mean strain times its length (use the in-plane direction along the line).

    Only the extent of the receiver along the line is modelled; across the line it is a point."""
    if receiver is None: receiver = {'type': 'point'}
    if not isinstance(receiver, dict) or receiver.get('type') not in RECEIVERS: raise ValueError('接收方式需为 point（点）、wedge（斜楔）或 patch（压电片）。')
    kind = receiver['type']
    if kind == 'point':
        if set(receiver) != {'type'}: raise ValueError('点接收不带其他参数。')
        return values, list(range(len(distance_mm))), {'type': 'point'}
    allowed = {'type', 'length_mm'} | ({'trace_velocity_m_s', 'apodization'} if kind == 'wedge' else set())
    if set(receiver) - allowed or not number(receiver.get('length_mm')) or receiver['length_mm'] <= 0: raise ValueError('接收换能器需要正的长度 length_mm（mm）' + ('、迹线速度 trace_velocity_m_s（m/s）。' if kind == 'wedge' else '。'))
    spacing = np.diff(distance_mm)
    if len(spacing) < 2 or np.ptp(spacing) > 1e-6 * spacing.max(): raise ValueError('有尺寸的接收换能器需要等距的点列：请用“线段等距采样”。')
    dx = float(spacing[0]); half = int(round(receiver['length_mm'] / 2 / dx))
    if half < 1: raise ValueError(f'接收换能器的长度（{receiver["length_mm"]:g} mm）小于两个点距（{2 * dx:.3g} mm）：请加密点列，或改用点接收。')
    count = len(distance_mm) - 2 * half
    if count < 2: raise ValueError(f'点列比接收换能器长不了多少：长 {receiver["length_mm"]:g} mm 的换能器在这条线上放不下两个位置。请加长线段。')
    kept = list(range(half, half + count)); used = {'type': kind, 'length_mm': 2 * half * dx, 'points': 2 * half + 1}
    if kind == 'patch':
        return {law: v[:, 2 * half:] - v[:, :count] for law, v in values.items()}, kept, used
    speed = receiver.get('trace_velocity_m_s'); shape = receiver.get('apodization', 'uniform')
    if not (number(speed) and 100 <= speed <= 1e5) or shape not in ('uniform', 'hann'): raise ValueError('斜楔接收需要 100–100000 m/s 的迹线速度；幅值分布为 uniform 或 hann。')
    offsets = np.arange(-half, half + 1); weights = .5 * (1 + np.cos(np.pi * offsets / (half + 1))) if shape == 'hann' else np.ones(len(offsets)); weights = weights / weights.sum()
    shift = offsets * dx * 1e-3 / speed * 1e6                      # μs: how much later the wave reaches that part of the footprint
    pad = int(math.ceil(np.abs(shift).max() / step_us)) + 2; frames = len(times); n = frames + 2 * pad; f = np.fft.rfftfreq(n, step_us)
    out = {}
    for law, v in values.items():
        spectrum = np.fft.rfft(np.pad(v, ((pad, pad), (0, 0))), axis=0); total = np.zeros((len(f), count), dtype=complex)
        for o, w, s in zip(offsets, weights, shift): total += w * spectrum[:, half + o:half + o + count] * np.exp(2j * np.pi * f * s)[:, None]   # advanced by its delay
        out[law] = np.fft.irfft(total, n, axis=0)[pad:pad + frames]
    return out, kept, dict(used, trace_velocity_m_s=float(speed), apodization=shape)


def _attenuation(scene, f, e, points, distance, rows, chosen, x, a1, pair, wave, speed_measured, notes):
    """The attenuation along the line: what A1 shows, what the model's damping gives for the modes, and A2/A1² with the
    attenuation taken out.

    With attenuation a₁ at f and a₂ at 2f a phase-matched second harmonic grows as
    A₂ = K A₁(0)² (e^{−2a₁X} − e^{−a₂X})/(a₂ − 2a₁), X the distance from the source, so A₂/A₁² = K (1 − e^{−mX})/m with
    m = a₂ − 2a₁. Multiplying by mX/(1 − e^{−mX}) gives K·X again: a straight line whose slope is the one the material
    would show without attenuation. X is measured from the centre of the primary source along the line."""
    measured = linear_fit(x, np.log(a1)) if np.all(a1 > 0) else None
    out = {'a1_measured_np_m': None if measured is None else -measured['slope'], 'a1_measured_r_squared': None if measured is None else measured['r_squared']}
    damping = scene.get('damping')
    if not damping: return out
    from physics.damping import decay_per_s
    theory = wave['theory'] if wave['theory'] and wave['theory']['matched'] else None
    c1 = pair['group_velocity_primary_m_s'] if pair else theory['group_velocity_m_s'] if theory else speed_measured
    c2 = pair['group_velocity_secondary_m_s'] if pair else c1
    out.update(decay_per_s={'f': decay_per_s(damping, f * 1e6), '2f': decay_per_s(damping, 2 * f * 1e6)}, group_velocity_source='theory' if pair or theory else 'measured')
    if not (c1 and c1 > 0 and c2 and c2 > 0):
        notes.append('没有群速度（理论和实测都没有），不能把模型的阻尼换算成沿程衰减。'); return out
    a1t = out['decay_per_s']['f'] / c1; a2t = out['decay_per_s']['2f'] / c2; m = a2t - 2 * a1t
    out.update(a1_np_m=a1t, a2_np_m=a2t, m_np_m=m, mode_f=pair['primary'] if pair else theory['mode'] if theory else None, mode_2f=pair['secondary'] if pair else None)
    if not pair: notes.append('没有认出二次模态：2f 处的沿程衰减按一次模态的群速度估计。')
    if pair and abs(pair['mismatch']) >= .01:
        notes.append('这一对模态相位失配：衰减修正的公式只适用于相位匹配的累积，没有做修正。'); return out
    offset = float((np.asarray(points[0], dtype=float) - np.asarray(scene['source']['center_mm'], dtype=float)) @ np.asarray(e, dtype=float)) - float(distance[0])
    factor = lambda X: 1. if abs(m * X) < 1e-9 else m * X / -math.expm1(-m * X)
    if min(r['distance_mm'] for r in chosen) + offset <= 0:
        notes.append('拟合范围内有点不在主激励的下游：衰减修正需要从激励算起的传播距离，没有做修正。'); return out
    for r in rows:
        X = (r['distance_mm'] + offset) * 1e-3
        if r['usable'] and r.get('ratio_per_m') is not None and X > 0: r['ratio_corrected_per_m'] = r['ratio_per_m'] * factor(X)
    fit = linear_fit(x + offset * 1e-3, [r['ratio_corrected_per_m'] for r in chosen])
    out.update(source_offset_mm=offset, corrected={'slope_per_m2': fit['slope'], 'intercept_per_m': fit['intercept'], 'r_squared': fit['r_squared']})
    return out


def fk(job, body):
    """Frequency–wavenumber spectrum along a line of equally spaced points. `body`: {'points', 'direction', 'part',
    'range_us', 'frequency_max_mhz'}. Positive wavenumbers travel from the first point to the last.

    Returns the magnitude in dB below its maximum (floor −60 dB) as base64 bytes, rows by frequency and columns by
    wavenumber, and — for a plate — the theoretical curves over the same range."""
    from .domain import RECIPES, parts_for, response
    from .recording import controls_of, is_compact, manifest
    if not isinstance(body, dict) or set(body) - {'points', 'direction', 'part', 'range_us', 'frequency_max_mhz'}: raise ValueError('f–k 分析只接受 points、direction、part、range_us、frequency_max_mhz。')
    job, scene, f = _job(job); m = manifest(job / 'record'); compact = is_compact(m); step = m['sample_step_us']
    points, distance, e, _, notes = line_of(body, scene, compact, MAX_FK)
    if len(points) < 8: raise ValueError('f–k 谱至少需要 8 个等间距的点' + ('：精简记录只有接收点有波形，请在模型里加一排等间距的接收点，或改用完整记录。' if compact else '。'))
    dx = float(np.mean(np.diff(distance)))
    if np.max(np.abs(np.diff(distance) - dx)) > 1e-6 * max(dx, 1e-9) + 1e-9: raise ValueError('f–k 谱需要等间距的点：这些接收点的间距不相等。')
    direction = _vector(body.get('direction', scene['receivers'][0]['direction']), '接收方向', unit=True); part = body.get('part', 'full')
    controls = controls_of(m)
    if part not in RECIPES or not (part in parts_for(controls) or (part == 'nonlinear' and {'linear', 'murnaghan'} <= set(controls))): raise ValueError('这次计算没有这种响应。')
    span = body.get('range_us'); first, last = 0, m['frame_count']
    if span is not None:
        if not (isinstance(span, list) and len(span) == 2 and all(number(v) for v in span) and span[0] < span[1]): raise ValueError('时间范围需要起点早于终点的两个时刻（μs）。')
        first = max(0, int(math.ceil(span[0] / step - 1e-9))); last = min(m['frame_count'], int(math.floor(span[1] / step + 1e-9)) + 1)
    if last - first < 32: raise ValueError('f–k 谱的时间范围至少需要 32 个采样点。')
    nyquist = .5 / step; top = body.get('frequency_max_mhz', min(nyquist, 3 * f))
    if not (number(top) and 0 < top <= nyquist + 1e-12): raise ValueError(f'频率上限需在 0–{nyquist:g} MHz（Nyquist）以内。')
    times, values, m = series(job / 'record', points, direction, first, last); data = response(values, part); nt, nx = data.shape
    data = (data - data.mean(axis=0)) * np.hanning(nt)[:, None] * np.hanning(nx)[None, :]
    n_t = 1 << max(11, int(math.ceil(math.log2(nt)))); n_x = 1 << max(8, int(math.ceil(math.log2(4 * nx))))
    rows = int(math.floor(top * step * n_t + 1e-9)) + 1; spectrum = np.empty((rows, nx), dtype=complex)
    for start in range(0, nx, 16): spectrum[:, start:start + 16] = np.fft.rfft(data[:, start:start + 16], n=n_t, axis=0)[:rows]   # e^{−iωt}; a few traces at a time
    magnitude = np.abs(np.fft.fftshift(np.fft.ifft(spectrum, n=n_x, axis=1), axes=1))   # e^{+ikx}: a wave e^{i(ωt − kx)} towards +x sits at positive k
    k_nyquist = math.pi / dx; dk = 2 * math.pi / (n_x * dx); df = 1 / (n_t * step)   # rad/mm, MHz
    from . import guided
    plate, why = _plate(scene, e, top); curves = None; k_show = k_nyquist
    if plate:
        k_show = min(k_nyquist, 1.15 * plate.reach(top * 1e6) * 1e-3)
        curves = [{'name': c['name'], 'family': c['family'], 'wavenumber_rad_mm': (c['k'] * 1e-3).tolist(), 'frequency_mhz': (c['omega'] / (2e6 * math.pi)).tolist()} for c in guided._curves(plate, top * 1e6, 240)]
        slowest = max((mode['k_rad_m'] for mode in guided._at(plate, min(top, 2 * f) * 1e6)), default=0.) * 1e-3
        if slowest > k_nyquist: notes.append(f'点距 {dx:.3g} mm 的波数上限是 {k_nyquist:.3g} rad/mm，低于 {min(top, 2 * f):.3g} MHz 处最慢模态的 {slowest:.3g} rad/mm：这个模态会混叠到别的波数上。请加密点列。')
    elif why: notes.append(why)
    centre = n_x // 2; half = min(centre - 1, int(math.ceil(k_show / dk))); magnitude = magnitude[:, centre - half:centre + half + 1]
    # reduce to the display size by keeping the largest value of each block
    fr = max(1, int(math.ceil(rows / 360))); kr = max(1, int(math.ceil(magnitude.shape[1] / 513)))
    rows_kept = rows // fr * fr; cols_kept = magnitude.shape[1] // kr * kr; offset = (magnitude.shape[1] - cols_kept) // 2
    shown = magnitude[:rows_kept, offset:offset + cols_kept].reshape(rows_kept // fr, fr, cols_kept // kr, kr).max(axis=(1, 3)); peak = float(shown.max())
    if not peak > 0: raise ValueError('这段记录里信号为零。')
    db = 20 * np.log10(np.maximum(shown / peak, 1e-3)); image = np.round((db + 60) / 60 * 255).astype(np.uint8)
    k0 = (offset - half) * dk + (kr - 1) * dk / 2; f0 = (fr - 1) * df / 2
    return {'schema': 'workai.nonlinear.fk.v1', 'job_id': job.name, 'part': part, 'frequency_mhz': f, 'direction': [float(v) for v in direction], 'line_direction': [float(v) for v in e],
            'points': len(points), 'spacing_mm': dx, 'range_us': [float(times[0]), float(times[-1])], 'samples': nt,
            'frequency': {'first_mhz': f0, 'step_mhz': fr * df, 'count': int(shown.shape[0])}, 'wavenumber': {'first_rad_mm': k0, 'step_rad_mm': kr * dk, 'count': int(shown.shape[1]), 'nyquist_rad_mm': k_nyquist},
            'floor_db': -60, 'image': base64.b64encode(image.tobytes()).decode('ascii'), 'curves': curves, 'validity': None if curves is None else guided.validity(scene), 'notes': notes,
            'note': '两个方向都加 Hann 窗并补零。正波数从第一个点传向最后一个点，负波数是反向传播（反射）。图上的峰落在哪条理论曲线上，就是哪个模态；谱的宽度来自有限的线长和记录时长，不是模态的展宽。'}


def dispersion(job, angle_deg=0., frequency_max_mhz=None):
    """Theoretical curves, modes and phase-matching table of the plate of a job."""
    from . import guided
    if not (number(angle_deg) and -360 <= angle_deg <= 360): raise ValueError('传播方向与 x 轴的夹角需在 −360° 到 360° 之间。')
    if frequency_max_mhz is not None and not number(frequency_max_mhz): raise ValueError('频率上限需为数值（MHz）。')
    _, scene, _ = _job(job)
    return guided.describe(scene, angle_deg, frequency_max_mhz)
