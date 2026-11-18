"""Parameter scans in the workbench: one model, up to three parameters varied, one computation per combination.

A scan is defined by the request of the model as it would be computed alone (scene, automatic mesh, engine and
solver settings) and by its axes: a numeric field of the scene (`source.frequency_mhz`, `geometry.thickness_mm`,
`defects.0.depth_mm`, …) with the values it takes. The combinations are the full grid of the axes, at most 200.

* Every combination is an ordinary computation of the workbench: the same validation, automatic mesh and time
  layer, record and result page as a single run, so a case of a scan and the same model run alone are the same job.
  They run one after another (each uses all the cores), in the order of the grid.
* A combination that fails does not stop the scan; `retry` puts the failed ones back in the queue. A scan can be
  paused (the running combination stops at its checkpoint) and continued, also after the application was closed.
* `summary` collects, for every finished combination and every receiver, the amplitudes at f and 2f of the complete
  response and of the nonlinear increments (complete − linear, geometric, material), computed exactly as the result
  page does (Hann window over the analysis window, weighted mean removed, coefficient at the exact frequency), and
  A2/A1² with A2 taken from the nonlinear increment when the linear control was computed. `summary_csv` is the same
  table as a file.

When a dimension of a plate changes, what sat on the moving face or on the mid-plane moves with it (`follow`):
a source centred on the top face stays on the top face when the thickness is scanned.

The research command line `python -m sweep` (other solvers, sampling plans, data sets) is a different tool; this
module needs only what the packaged application ships.
"""
import copy
import csv
import io
import itertools
import json
import math
import re
import secrets
import threading
from datetime import datetime, timezone
import numpy as np
from .domain import V2, validate_request

SCHEMA = 'workai.scan.v1'
MAX_AXES, MAX_VALUES, MAX_CASES = 3, 100, 200
ID = r'scan-[a-f0-9]{24}'
ROOTS = ('source', 'sources', 'geometry', 'defects', 'material', 'boundaries', 'receivers', 'acquisition')
PATH = re.compile(r'[a-z_]+(\.[a-z_0-9]+){1,4}')
DIMENSIONS = {'geometry.length_mm': 0, 'geometry.width_mm': 1, 'geometry.thickness_mm': 2}
TERMINAL = ('completed', 'failed', 'paused')


class Busy(ValueError):
    """Something is being computed: the request can be repeated when it has finished."""


def now(): return datetime.now(timezone.utc).isoformat()


def get_path(obj, path):
    for key in path.split('.'):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def set_path(obj, path, value):
    *keys, last = path.split('.')
    for key in keys: obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    if isinstance(obj, list): obj[int(last)] = value
    else: obj[last] = value


def number(value): return type(value) in (int, float) and math.isfinite(value)


def follow(scene, axis, old, new):
    """A plate dimension changed from `old` to `new`: source centres, receivers and defect ranges that sat on the
    moving face move with it, and those on the mid-plane stay on the mid-plane."""
    def moved(v): return new if abs(v - old) < 1e-9 else new / 2 if abs(v - old / 2) < 1e-9 else v
    for source in [scene['source'], *(scene.get('sources') or [])]:
        if isinstance(source.get('center_mm'), list) and len(source['center_mm']) == 3 and number(source['center_mm'][axis]): source['center_mm'][axis] = moved(source['center_mm'][axis])
    for receiver in scene.get('receivers') or []:
        if isinstance(receiver.get('point_mm'), list) and len(receiver['point_mm']) == 3 and number(receiver['point_mm'][axis]): receiver['point_mm'][axis] = moved(receiver['point_mm'][axis])
    if axis == 1:
        for defect in scene.get('defects') or []:
            if isinstance(defect.get('y_range_mm'), list) and len(defect['y_range_mm']) == 2 and number(defect['y_range_mm'][1]) and abs(defect['y_range_mm'][1] - old) < 1e-9: defect['y_range_mm'][1] = new
            if number(defect.get('y_mm')): defect['y_mm'] = moved(defect['y_mm'])


def validate_scan(body):
    """{'request', 'axes': [{'path', 'values'}], 'name'} from a request body; the scene itself is validated case by case."""
    if not isinstance(body, dict) or set(body) - {'request', 'axes', 'name'} or not isinstance(body.get('request'), dict):
        raise ValueError('参数扫描需要 request（当前模型的计算请求）和 axes（扫描的参数）。')
    request = copy.deepcopy(body['request']); scene = request.get('scene')
    if not isinstance(scene, dict) or scene.get('schema') != V2:
        raise ValueError('参数扫描需要 v2 场景。')
    axes = body.get('axes')
    if not isinstance(axes, list) or not 1 <= len(axes) <= MAX_AXES:
        raise ValueError(f'参数扫描需要 1–{MAX_AXES} 个参数。')
    out = []; count = 1
    for axis in axes:
        if not isinstance(axis, dict) or set(axis) != {'path', 'values'} or not isinstance(axis['path'], str) or not PATH.fullmatch(axis['path']) or axis['path'].split('.')[0] not in ROOTS:
            raise ValueError('扫描参数需要 path（场景里的数值字段，例如 source.frequency_mhz）和 values（取值列表）。')
        path = axis['path']
        try: current = get_path(scene, path)
        except (KeyError, IndexError, ValueError, TypeError): raise ValueError(f'当前模型里没有参数 {path}。') from None
        if not number(current): raise ValueError(f'参数 {path} 不是数值，不能扫描。')
        if path in (a['path'] for a in out): raise ValueError(f'参数 {path} 重复。')
        values = axis['values']
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_VALUES or not all(number(v) for v in values):
            raise ValueError(f'参数 {path} 需要 1–{MAX_VALUES} 个有限数值。')
        values = [float(v) for v in values]
        if len(set(values)) != len(values): raise ValueError(f'参数 {path} 的取值有重复。')
        out.append({'path': path, 'values': values}); count *= len(values)
    if count > MAX_CASES: raise ValueError(f'组合数 {count} 超过上限 {MAX_CASES}：请减少取值，或分几次扫描。')
    name = body.get('name', scene.get('name') or '参数扫描')
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100: raise ValueError('扫描名称需为 1–100 个字符。')
    return {'request': request, 'axes': out, 'name': name.strip()}


def label(axes, values):
    """`frequency_mhz=0.3, amplitude_mpa=2` for one combination."""
    return ', '.join(f"{axis['path'].split('.')[-1] if not axis['path'].split('.')[-1].isdigit() else axis['path']}={value:g}" for axis, value in zip(axes, values))


def combinations(scan):
    """Every combination in grid order (the first axis varies slowest): lists of values, one per axis."""
    return [list(row) for row in itertools.product(*(axis['values'] for axis in scan['axes']))]


def case_request(scan, values):
    """The request of one combination: the scan's request with its values written into the scene."""
    request = copy.deepcopy(scan['request']); scene = request['scene']
    for axis, value in zip(scan['axes'], values):
        old = get_path(scene, axis['path']); set_path(scene, axis['path'], value)
        if axis['path'] in DIMENSIONS and scene.get('geometry', {}).get('kind') != 'pipe': follow(scene, DIMENSIONS[axis['path']], old, value)
    tail = ' · ' + label(scan['axes'], values); scene['name'] = (str(scene.get('name') or '模型')[:max(1, 100 - len(tail))] + tail)[:100]
    return request


def plan(body, data_directory=None):
    """What a scan would compute, without starting it: every combination checked like the model itself, with its mesh
    and estimated cost, and the totals."""
    from .preview import preview
    scan = validate_scan(body); rows = []
    for index, values in enumerate(combinations(scan)):
        checked = preview(case_request(scan, values), data_directory); cost = checked.get('estimate') or {}; mesh = checked.get('mesh') or {}
        rows.append({'index': index, 'values': values, 'label': label(scan['axes'], values), 'ok': bool(checked['ok']), 'errors': checked['errors'], 'warnings': checked['warnings'],
                     'nodes': mesh.get('nodes'), 'h_mm': (mesh.get('auto') or {}).get('h_mm'), 'seconds': cost.get('seconds'), 'memory_bytes': cost.get('memory_bytes'), 'disk_bytes': cost.get('disk_bytes')})
    valid = [r for r in rows if r['ok']]; total = lambda key, combine: combine([r[key] for r in valid if r[key] is not None] or [0])
    return {'axes': scan['axes'], 'name': scan['name'], 'count': len(rows), 'valid': len(valid), 'cases': rows,
            'seconds_total': total('seconds', sum), 'memory_bytes_max': total('memory_bytes', max), 'disk_bytes_total': total('disk_bytes', sum)}


def harmonics(times_us, values, frequency_mhz, window_us=None):
    """Amplitude (m) and phase (degrees) at f and 2f of a waveform, as the result page computes them: a Hann window over
    the analysis window, the weighted mean removed, the Fourier coefficient at the exact frequency. None above Nyquist."""
    times = np.asarray(times_us, dtype=float); values = np.asarray(values, dtype=float)
    start, end = window_us if window_us else (times[0], times[-1]); keep = (times >= start - 1e-9) & (times <= end + 1e-9)
    if keep.sum() < 8: raise ValueError('分析时间窗至少需要 8 个采样点。')
    t = times[keep]; v = values[keep]; n = len(t); w = .5 * (1 - np.cos(2 * math.pi * np.arange(n) / (n - 1))); v = (v - (v * w).sum() / w.sum()) * w
    nyquist = .5 / (times[1] - times[0]); out = []
    for order in (1, 2):
        f = order * frequency_mhz
        if f > nyquist: out.append(None); continue
        c = (v * np.exp(-2j * math.pi * f * t)).sum() * (1 if abs(f - nyquist) < 1e-12 else 2) / w.sum()
        out.append({'amplitude_m': float(abs(c)), 'phase_deg': float(np.degrees(np.angle(c))) if abs(c) > 1e-30 else None})
    return out


def case_summary(job, window_us=None):
    """Harmonics at the receivers of one finished job: {'frequency_mhz', 'nodes', 'elapsed_s', 'max_strain', 'receivers': [...]}."""
    from .recording import manifest, traces, controls_of, available
    request = json.loads((job / 'request.json').read_text(encoding='utf-8')); status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
    scene = request['scene']; frequency = float(scene['source']['frequency_mhz']); record = job / 'record'; m = manifest(record)
    receivers = scene['receivers']; points = [[float(v) for v in r['point_mm']] for r in receivers]
    directions = [np.asarray(r['direction'], dtype=float) / np.linalg.norm(r['direction']) for r in receivers]
    found = {}
    for part in ('full', 'linear', 'nonlinear', 'geometric', 'material'):
        try: available(m, part)
        except ValueError: continue
        rows = []
        for i in range(0, len(points), 8):
            t = traces(record, part, points[i:i + 8]); times = t['times_us']
            rows += [harmonics(times, np.asarray(p['xyz_m']) @ directions[i + j], frequency, window_us) for j, p in enumerate(t['points'])]
        found[part] = rows
    out = []
    for i, receiver in enumerate(receivers):
        amplitude = lambda part, order: (found[part][i][order - 1] or {}).get('amplitude_m') if part in found else None
        phase = lambda part, order: (found[part][i][order - 1] or {}).get('phase_deg') if part in found else None
        a1 = amplitude('full', 1); increment = 'nonlinear' if 'nonlinear' in found else None; a2 = amplitude(increment, 2) if increment else amplitude('full', 2)
        out.append({'name': receiver['name'], 'point_mm': points[i], 'direction': [float(v) for v in directions[i]], 'a1_m': a1, 'phase1_deg': phase('full', 1), 'a2_m': amplitude('full', 2), 'phase2_deg': phase('full', 2),
                    'a1_linear_m': amplitude('linear', 1), 'a2_linear_m': amplitude('linear', 2), 'a2_nonlinear_m': amplitude('nonlinear', 2), 'phase2_nonlinear_deg': phase('nonlinear', 2),
                    'a2_geometric_m': amplitude('geometric', 2), 'a2_material_m': amplitude('material', 2),
                    # the relative nonlinearity parameter: A2 of the nonlinear increment when the linear control was computed, else of the complete response
                    'a2_over_a1_squared_per_m': a2 / a1 ** 2 if a1 and a2 is not None else None, 'a2_basis': 'nonlinear' if increment else 'full'})
    return {'frequency_mhz': frequency, 'nodes': status.get('nodes'), 'elapsed_s': status.get('elapsed_s'), 'max_strain': max(m['max_strain']) if m.get('max_strain') else None,
            'controls': list(controls_of(m)), 'window_us': list(window_us) if window_us else None, 'receivers': out}


class Scans:
    """The scans of an application: stored under data/scans, run through the application's own job queue."""

    def __init__(self, application):
        self.app = application; self.root = application.data / 'scans'; self.root.mkdir(parents=True, exist_ok=True); self.lock = threading.RLock()
        for path in self.root.glob('*/scan.json'):
            scan = json.loads(path.read_text(encoding='utf-8'))
            if scan['state'] == 'running':   # the application was closed during the scan
                self.sync(scan); scan.update(state='paused', message='上次退出时扫描未完成，可继续。'); self.save(scan)

    # ---------- storage ----------
    def path(self, scan_id):
        if not isinstance(scan_id, str) or not re.fullmatch(ID, scan_id): raise ValueError('扫描编号无效。')
        directory = self.root / scan_id
        if not directory.is_dir() or directory.is_symlink(): raise FileNotFoundError('扫描不存在。')
        return directory

    def load(self, scan_id): return json.loads((self.path(scan_id) / 'scan.json').read_text(encoding='utf-8'))

    def save(self, scan):
        from .app import save
        scan['updated_at_utc'] = now(); save(self.root / scan['scan_id'] / 'scan.json', scan)

    def all(self): return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.root.glob('*/scan.json'))]

    def running(self): return next((s for s in self.all() if s['state'] == 'running'), None)

    def active(self):
        """True while a scan holds the queue (between two of its combinations nothing is being computed)."""
        with self.lock: return self.running() is not None

    # ---------- state ----------
    def sync(self, scan):
        """Bring the phases of the combinations up to date with their jobs."""
        for case in scan['cases']:
            if not case.get('job_id') or case['phase'] in ('completed', 'failed', 'queued'): continue
            try: status = self.app.status(case['job_id'])
            except (OSError, ValueError): case.update(phase='failed', message='计算记录已不存在。'); continue
            phase = status['phase'] if status['phase'] in TERMINAL else 'running'
            case.update(phase=phase, message=status.get('message', '') if phase != 'completed' else '', **({'elapsed_s': status.get('elapsed_s'), 'nodes': status.get('nodes')} if phase == 'completed' else {}))
        return scan

    def public(self, scan, cases=True):
        """What the interface gets: the definition and the phases, without the stored request."""
        counts = {phase: sum(1 for c in scan['cases'] if c['phase'] == phase) for phase in ('queued', 'running', 'completed', 'failed', 'paused')}
        out = {k: scan[k] for k in ('scan_id', 'name', 'state', 'message', 'created_at_utc', 'updated_at_utc', 'axes')}; out.update(count=len(scan['cases']), counts=counts)
        if cases: out['cases'] = [dict(c, label=label(scan['axes'], c['values'])) for c in scan['cases']]
        return out

    def list(self):
        with self.lock: return sorted((self.public(self.sync(s), cases=False) for s in self.all()), key=lambda s: s['created_at_utc'], reverse=True)

    def detail(self, scan_id):
        with self.lock:
            scan = self.sync(self.load(scan_id))
            if scan['state'] != 'completed': self.save(scan)
            return self.public(scan)

    # ---------- commands ----------
    def create(self, body):
        """Check every combination, store the scan and start its first combination."""
        scan = validate_scan(body); rows = combinations(scan)
        for index, values in enumerate(rows):
            try: validate_request(case_request(scan, values))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'第 {index + 1} 个组合（{label(scan["axes"], values)}）不能计算：{error}') from None
        with self.lock:
            if self.running() is not None: raise Busy('已有参数扫描在进行，请先暂停它或等它完成。')
            self.app.settle()
            if self.app.busy(): raise Busy('已有任务正在运行，请等待完成。')
            scan.update(schema=SCHEMA, scan_id='scan-' + secrets.token_hex(12), state='running', message='', created_at_utc=now(),
                        cases=[{'index': i, 'values': values, 'phase': 'queued', 'job_id': None, 'message': '', 'attempts': 0, 'previous_jobs': []} for i, values in enumerate(rows)])
            (self.root / scan['scan_id']).mkdir(); self.save(scan); self.advance()
            return {'scan_id': scan['scan_id'], 'count': len(rows)}

    def advance(self):
        """Start the next queued combination of the running scan when nothing is being computed. Called whenever a
        computation process has ended."""
        with self.lock:
            if self.app.closed: return
            scan = self.running()
            if scan is None: return
            self.sync(scan)
            while True:
                if any(c['phase'] == 'paused' for c in scan['cases']):
                    scan.update(state='paused', message='当前组合已暂停；继续扫描时从它的检查点接着算。'); break
                if any(c['phase'] == 'running' for c in scan['cases']): break
                case = next((c for c in scan['cases'] if c['phase'] == 'queued'), None)
                if case is None:
                    failed = sum(1 for c in scan['cases'] if c['phase'] == 'failed')
                    scan.update(state='completed', message=f'{failed} 个组合未完成，可重算。' if failed else '全部组合已完成。'); break
                try:
                    job = self.app.submit(case_request(scan, case['values']), scan={'id': scan['scan_id'], 'index': case['index'], 'count': len(scan['cases'])})['job_id']
                except Busy: break   # another computation holds the machine: its end calls advance again
                except (ValueError, KeyError, TypeError, OSError) as error:
                    case.update(phase='failed', message=str(error) or '无法启动计算。', attempts=case['attempts'] + 1); continue
                case.update(phase='running', job_id=job, message='', attempts=case['attempts'] + 1); break
            self.save(scan)

    def control(self, scan_id, action):
        with self.lock:
            scan = self.sync(self.load(scan_id)); current = next((c for c in scan['cases'] if c['phase'] == 'running'), None)
            if action == 'pause':
                if scan['state'] != 'running': raise ValueError('这个扫描没有在运行。')
                scan.update(state='paused', message='已暂停：当前组合保存检查点后停止。' if current else '已暂停。'); self.save(scan)
                if current:
                    try: self.app.control(current['job_id'], 'pause')
                    except ValueError: pass   # it finished at this moment
                return self.public(scan)
            other = self.running()
            if other is not None and other['scan_id'] != scan_id: raise Busy('已有参数扫描在进行，请先暂停它或等它完成。')
            if action == 'retry':
                failed = [c for c in scan['cases'] if c['phase'] == 'failed']
                if not failed: raise ValueError('没有未完成的组合需要重算。')
                for case in failed:
                    if case.get('job_id'): case['previous_jobs'].append(case['job_id'])
                    case.update(phase='queued', job_id=None, message='')
            elif action == 'resume':
                if scan['state'] != 'paused': raise ValueError('这个扫描没有暂停。')
            else: raise ValueError('扫描操作需为 pause、resume 或 retry。')
            self.app.settle(); paused = next((c for c in scan['cases'] if c['phase'] == 'paused'), None)
            if action == 'resume' and self.app.busy() and current is None: raise Busy('已有任务正在运行，请等待完成。')
            scan.update(state='running', message=''); self.save(scan)
            if paused is not None and not self.app.busy():
                self.app.control(paused['job_id'], 'resume'); paused.update(phase='running', message=''); self.save(scan)
            else: self.advance()
            return self.public(self.sync(self.load(scan_id)))

    def job_resumed(self, link):
        """A paused combination was continued from its own record: its scan runs again."""
        with self.lock:
            try: scan = self.load(link['id'])
            except (ValueError, FileNotFoundError, KeyError, TypeError): return
            if scan['state'] == 'paused' and self.running() is None: scan.update(state='running', message=''); self.save(self.sync(scan))

    # ---------- results ----------
    def summary(self, scan_id, window_us=None):
        """The table of a scan: one row per combination, with the harmonics at every receiver of the finished ones."""
        if window_us is not None and not (isinstance(window_us, (list, tuple)) and len(window_us) == 2 and all(number(v) for v in window_us) and window_us[0] < window_us[1]):
            raise ValueError('分析时间窗需要起点早于终点的两个时刻（μs）。')
        with self.lock: scan = self.sync(self.load(scan_id))
        directory = self.path(scan_id) / 'summary'; directory.mkdir(exist_ok=True); rows = []
        for case in scan['cases']:
            row = {k: case[k] for k in ('index', 'values', 'phase', 'job_id', 'message')}; row['label'] = label(scan['axes'], case['values'])
            if case['phase'] == 'completed':
                cache = directory / f"{case['job_id']}.json"; key = list(window_us) if window_us else None
                stored = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else None
                if stored is None or stored.get('window_us') != key:
                    try: stored = case_summary(self.app.path(case['job_id']), window_us)
                    except (ValueError, OSError, KeyError) as error: stored = {'window_us': key, 'error': str(error)}
                    cache.write_text(json.dumps(stored, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                row.update(stored)
            rows.append(row)
        return dict(self.public(scan, cases=False), window_us=list(window_us) if window_us else None, rows=rows,
                    note='幅值按结果页的方法计算（分析时间窗上的 Hann 窗，去加权均值，取 f 与 2f 处的系数）。A₂/A₁² 的 A₂ 取非线性增量（完整 − 线性）；没有算线性控制时取完整响应，'
                         '那时 2f 处还含有猝发音自身的线性频谱，不能当作非线性的度量。')

    def summary_csv(self, scan_id, window_us=None):
        table = self.summary(scan_id, window_us); out = io.StringIO(); writer = csv.writer(out)
        writer.writerow(['组合', *[axis['path'] for axis in table['axes']], '状态', '计算记录', '接收点', 'x_mm', 'y_mm', 'z_mm', '方向 x', '方向 y', '方向 z', 'f (MHz)', 'A1 完整 (m)', 'A2 完整 (m)', 'A1 线性 (m)', 'A2 线性 (m)',
                         'A2 非线性增量 (m)', 'A2 几何增量 (m)', 'A2 材料增量 (m)', 'A2/A1^2 (1/m)', 'A2/A1^2 所用的 A2', '节点', '用时 (s)', '最大应变', '说明'])
        blank = lambda v: '' if v is None else v
        for row in table['rows']:
            head = [row['index'] + 1, *row['values'], row['phase'], blank(row['job_id'])]
            if not row.get('receivers'): writer.writerow(head + [''] * 17 + [blank(row.get('nodes')), blank(row.get('elapsed_s')), blank(row.get('max_strain')), row.get('error') or row.get('message') or '']); continue
            for r in row['receivers']:
                writer.writerow(head + [r['name'], *r['point_mm'], *r['direction'], row['frequency_mhz']] + [blank(r[k]) for k in ('a1_m', 'a2_m', 'a1_linear_m', 'a2_linear_m', 'a2_nonlinear_m', 'a2_geometric_m', 'a2_material_m', 'a2_over_a1_squared_per_m')]
                                + [{'nonlinear': '非线性增量', 'full': '完整响应'}[r['a2_basis']], blank(row.get('nodes')), blank(row.get('elapsed_s')), blank(row.get('max_strain')), ''])
        return ('﻿' + out.getvalue()).encode('utf-8')   # with a byte-order mark, so spreadsheet programs read the Chinese headers
