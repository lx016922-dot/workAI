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

A convergence check (`Scans.converge`) is a scan of one finished computation over its mesh: the computation itself
is the first combination, and the same model on one or two finer meshes the others. A level raises the degree of the
elements by one while it is below 5 — that refines every direction, also through a plate that is one element thick —
and multiplies the element size by 0.75 once the degree is 5. The automatic time layer follows the mesh.
`Scans.convergence` compares the receivers of every mesh with those of the finest one.

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
PHASE_WORDS = {'queued': '排队', 'running': '计算中', 'completed': '已完成', 'failed': '未完成', 'paused': '已暂停'}


CONVERGENCE_RATIO = .75                             # element size of one level relative to the level before
CONVERGENCE_TOLERANCE = {'a1': .01, 'a2': .05}      # change of A1 and of A2 from a mesh to the finest one that counts as converged


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
        if (not isinstance(axis, dict) or not {'path', 'values'} <= set(axis) <= {'path', 'values', 'label', 'unit'} or not isinstance(axis['path'], str) or not PATH.fullmatch(axis['path'])
                or axis['path'].split('.')[0] not in ROOTS):
            raise ValueError('扫描参数需要 path（场景里的数值字段，例如 source.frequency_mhz）和 values（取值列表）。')
        path = axis['path']; words = {key: axis[key].strip() for key in ('label', 'unit') if key in axis and isinstance(axis[key], str) and axis[key].strip()}
        if set(words) != {'label', 'unit'} & set(axis) or len(words.get('label', '')) > 60 or len(words.get('unit', '')) > 12 or any(c in v for v in words.values() for c in '\r\n\t'):
            raise ValueError(f'参数 {path} 的名称（label，最多 60 字）或单位（unit，最多 12 字）无效。')
        try: current = get_path(scene, path)
        except (KeyError, IndexError, ValueError, TypeError): raise ValueError(f'当前模型里没有参数 {path}。') from None
        if not number(current): raise ValueError(f'参数 {path} 不是数值，不能扫描。')
        if path in (a['path'] for a in out): raise ValueError(f'参数 {path} 重复。')
        values = axis['values']
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_VALUES or not all(number(v) for v in values):
            raise ValueError(f'参数 {path} 需要 1–{MAX_VALUES} 个有限数值。')
        values = [float(v) for v in values]
        if len(set(values)) != len(values): raise ValueError(f'参数 {path} 的取值有重复。')
        out.append({'path': path, 'values': values, **words}); count *= len(values)
    if count > MAX_CASES: raise ValueError(f'组合数 {count} 超过上限 {MAX_CASES}：请减少取值，或分几次扫描。')
    name = body.get('name', scene.get('name') or '参数扫描')
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100: raise ValueError('扫描名称需为 1–100 个字符。')
    return {'request': request, 'axes': out, 'name': name.strip()}


def label(axes, values):
    """One combination in words: `频率 0.3 MHz，幅值 2 MPa` with the names the interface gave the axes, else `frequency_mhz=0.3, amplitude_mpa=2`."""
    def one(axis, value):
        if axis.get('label'): return f"{axis['label']} {value:g}{' ' + axis['unit'] if axis.get('unit') else ''}"
        return f"{axis['path'].split('.')[-1] if not axis['path'].split('.')[-1].isdigit() else axis['path']}={value:g}"
    return ('，' if all(axis.get('label') for axis in axes) else ', ').join(one(axis, value) for axis, value in zip(axes, values))


def heading(axis):
    """Column heading of an axis: its name and unit, else its path."""
    return f"{axis['label']}{' (' + axis['unit'] + ')' if axis.get('unit') else ''}" if axis.get('label') else axis['path']


def combinations(scan):
    """Every combination in grid order (the first axis varies slowest): lists of values, one per axis."""
    return [list(row) for row in itertools.product(*(axis['values'] for axis in scan['axes']))]


def case_request(scan, values):
    """The request of one combination: the scan's request with its values written into the scene."""
    request = copy.deepcopy(scan['request']); scene = request['scene']
    if scan.get('kind') == 'convergence':   # the mesh of one level
        level = next(level for level in scan['levels'] if level['node_spacing_mm'] == values[0]); request['degree'] = level['degree']; scene['numerics']['h_mm'] = level['h_mm']
        tail = ' · ' + level['label']; scene['name'] = (str(scene.get('name') or '模型')[:max(1, 100 - len(tail))] + tail)[:100]
        return request
    for axis, value in zip(scan['axes'], values):
        old = get_path(scene, axis['path']); set_path(scene, axis['path'], value)
        if axis['path'] in DIMENSIONS and scene.get('geometry', {}).get('kind') != 'pipe': follow(scene, DIMENSIONS[axis['path']], old, value)
    tail = ' · ' + label(scan['axes'], values); scene['name'] = (str(scene.get('name') or '模型')[:max(1, 100 - len(tail))] + tail)[:100]
    return request


def convergence_definition(request, name, job_id, levels=1):
    """The scan that checks a finished computation on finer meshes: its request with the mesh written out (degree and
    element size instead of 'auto') and one axis, the mean node spacing h / degree of every level."""
    if type(levels) is not int or not 1 <= levels <= 2: raise ValueError('加密级数需为 1 或 2。')
    request = copy.deepcopy(request); scene = request.get('scene')
    if not isinstance(scene, dict) or scene.get('schema') != V2: raise ValueError('收敛检查需要 v2 场景的计算。')
    if request.get('engine', 'network') != 'network':
        raise ValueError('快速引擎在模型自带的粗网格上运行，网格不能加密：收敛检查只用于冻结网络引擎的计算。')
    report = request.pop('mesh_report', None)
    if request.pop('mesh', None) == 'auto':
        if not isinstance(report, dict): raise ValueError('这次计算没有保存自动网格的报告，不能确定它的单元尺寸。')
        request['degree'] = int(report['degree']); scene['numerics']['h_mm'] = float(report['h_mm'])
    meshes = [(int(request['degree']), float(scene['numerics']['h_mm']))]
    for _ in range(levels):
        degree, h = meshes[-1]; meshes.append((degree + 1, h) if degree < 5 else (degree, round(h * CONVERGENCE_RATIO, 4)))
    rows = [{'degree': degree, 'h_mm': h, 'node_spacing_mm': round(h / degree, 6), 'label': f'{degree} 阶 · 单元 {h:g} mm'} for degree, h in meshes]; tail = ' · 收敛检查'
    return {'request': request, 'axes': [{'path': 'mesh.node_spacing_mm', 'values': [r['node_spacing_mm'] for r in rows], 'label': '平均节点间距', 'unit': 'mm'}], 'levels': rows,
            'name': (str(name or '模型')[:100 - len(tail)] + tail), 'kind': 'convergence', 'base_job': job_id}


def case_label(scan, values):
    """One combination in words: the mesh of a convergence check, else the values of the axes."""
    if scan.get('kind') == 'convergence': return next(level['label'] for level in scan['levels'] if level['node_spacing_mm'] == values[0])
    return label(scan['axes'], values)


def wrapped(angle_deg):
    """An angle difference in (−180°, 180°]."""
    return (angle_deg + 180.) % 360. - 180. if angle_deg % 360. != 180. else 180.


def result_bytes(cost, nodes):
    """Size of the result files a job writes beside its record (the animation of every response), before compression."""
    from spectral3d.display import FRAMES
    from .domain import parts_for
    from .resources import BROWSER_NODES
    parts = len(parts_for(cost['controls']))
    if cost['record'] == 'full' and nodes <= BROWSER_NODES:   # the computation mesh itself, float64, within the 192 MiB display budget of one response
        return parts * max(2, min(FRAMES, cost['frames'], (192 * 1024 ** 2) // (nodes * 24))) * nodes * 24
    return parts * min(FRAMES, cost['frames']) * min(BROWSER_NODES, nodes) * 12   # the display mesh, float32


def plan(body, data_directory=None):
    """What a scan would compute, without starting it: every combination checked like the model itself, with its mesh
    and estimated cost, and the totals. `disk_bytes` is the record and the result files of a combination; every
    combination of a scan keeps both."""
    import shutil
    from .preview import preview
    scan = validate_scan(body); rows = []
    for index, values in enumerate(combinations(scan)):
        checked = preview(case_request(scan, values), data_directory); cost = checked.get('estimate') or {}; mesh = checked.get('mesh') or {}
        disk = cost['disk_bytes'] + result_bytes(cost, mesh['nodes']) if cost and mesh.get('nodes') else None
        rows.append({'index': index, 'values': values, 'label': label(scan['axes'], values), 'ok': bool(checked['ok']), 'errors': checked['errors'], 'warnings': checked['warnings'],
                     'nodes': mesh.get('nodes'), 'h_mm': (mesh.get('auto') or {}).get('h_mm'), 'seconds': cost.get('seconds'), 'memory_bytes': cost.get('memory_bytes'), 'disk_bytes': disk, 'record': cost.get('record')})
    valid = [r for r in rows if r['ok']]; total = lambda key, combine: combine([r[key] for r in valid if r[key] is not None] or [0])
    free = shutil.disk_usage(data_directory).free if data_directory else None; disk = total('disk_bytes', sum)
    return {'axes': scan['axes'], 'name': scan['name'], 'count': len(rows), 'valid': len(valid), 'cases': rows, 'seconds_total': total('seconds', sum), 'memory_bytes_max': total('memory_bytes', max),
            'disk_bytes_total': disk, 'free_bytes': free, 'disk_short': bool(free is not None and disk > free), 'full_records': sum(1 for r in valid if r['record'] == 'full')}


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


def receiver_series(record, points):
    """(times_us, {control: samples × points × 3}) at points of a finished record, read in one pass over its chunks. A compact
    record holds the waveforms of its receivers; a complete record is interpolated at the points."""
    from .recording import chunks, controls_of, is_compact, manifest
    m = manifest(record); controls = controls_of(m); values = {name: np.empty((m['frame_count'], len(points), 3)) for name in controls}
    if is_compact(m):
        rows = []
        for p in points:
            hit = [i for i, r in enumerate(m['receivers']) if max(abs(a - b) for a, b in zip(r['point_mm'], p)) <= 1e-6]
            if not hit: raise ValueError('精简记录里没有这个接收点的波形。')
            rows.append(hit[0])
        for row, data in chunks(record, m):
            probes = data['probes'][:, rows]   # samples × receivers × controls × xyz
            for i, name in enumerate(controls): values[name][row['first_frame']:row['first_frame'] + row['frame_count']] = probes[:, :, i]
    else:
        from spectral3d.mesh import interpolation
        with np.load(record / 'mesh.npz', allow_pickle=False) as data: mesh = dict(data)
        locations = [interpolation(mesh, p) for p in points]
        for row, data in chunks(record, m):
            start = row['first_frame']; end = start + row['frame_count']
            for name in controls:
                field = data[name]
                for j, (ids, w) in enumerate(locations): values[name][start:end, j] = np.einsum('n,tnc->tc', w, field[:, ids])
    return np.arange(m['frame_count']) * m['sample_step_us'], values, m


def case_summary(job, window_us=None):
    """Harmonics at the receivers of one finished job: {'frequency_mhz', 'nodes', 'elapsed_s', 'max_strain', 'receivers': [...]}."""
    from .domain import parts_for, response
    from .recording import controls_of
    request = json.loads((job / 'request.json').read_text(encoding='utf-8')); status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
    scene = request['scene']; frequency = float(scene['source']['frequency_mhz']); receivers = scene['receivers']; points = [[float(v) for v in r['point_mm']] for r in receivers]
    directions = [np.asarray(r['direction'], dtype=float) / np.linalg.norm(r['direction']) for r in receivers]
    times, values, m = receiver_series(job / 'record', points); controls = controls_of(m)
    wanted = [part for part in ('full', 'linear', 'nonlinear', 'geometric', 'material') if part in parts_for(controls) or (part == 'nonlinear' and {'linear', 'murnaghan'} <= set(controls))]
    found = {part: [harmonics(times, response(values, part)[:, j] @ directions[j], frequency, window_us) for j in range(len(points))] for part in wanted}
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
        out = {k: scan[k] for k in ('scan_id', 'name', 'state', 'message', 'created_at_utc', 'updated_at_utc', 'axes', 'kind', 'base_job') if k in scan}; out.update(count=len(scan['cases']), counts=counts)
        if cases: out['cases'] = [dict(c, label=case_label(scan, c['values'])) for c in scan['cases']]
        if 'levels' in scan: out['levels'] = scan['levels']
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

    # ---------- convergence checks ----------
    def _definition(self, job_id, levels):
        status = self.app.status(job_id)
        if status['phase'] != 'completed': raise ValueError('计算完成后才能做收敛检查。')
        request = json.loads((self.app.path(job_id) / 'request.json').read_text(encoding='utf-8'))
        return convergence_definition(request, status.get('name'), job_id, levels)

    def checks_of(self, job_id):
        """The convergence checks of a job, newest first (public form, without cases)."""
        with self.lock: return sorted((self.public(self.sync(s), cases=False) for s in self.all() if s.get('base_job') == job_id), key=lambda s: s['created_at_utc'], reverse=True)

    def converge_plan(self, job_id, levels=1):
        """What a convergence check of a job would compute: every finer mesh checked like a model, with its size and estimated cost."""
        import shutil
        from .preview import preview
        scan = self._definition(job_id, levels); rows = []; base = self.app.status(job_id)
        for level, values in list(zip(scan['levels'], combinations(scan)))[1:]:
            checked = preview(case_request(scan, values), self.app.data); cost = checked.get('estimate') or {}; mesh = checked.get('mesh') or {}
            rows.append({**level, 'ok': bool(checked['ok']), 'errors': checked['errors'], 'nodes': mesh.get('nodes'), 'seconds': cost.get('seconds'), 'memory_bytes': cost.get('memory_bytes'),
                         'disk_bytes': cost['disk_bytes'] + result_bytes(cost, mesh['nodes']) if cost and mesh.get('nodes') else None, 'record': cost.get('record')})
        valid = [r for r in rows if r['ok']]; free = shutil.disk_usage(self.app.data).free; disk = sum(r['disk_bytes'] or 0 for r in valid)
        return {'job_id': job_id, 'levels': levels, 'ratio': CONVERGENCE_RATIO, 'tolerance': CONVERGENCE_TOLERANCE, 'base': {**scan['levels'][0], 'nodes': base.get('nodes'), 'seconds': base.get('elapsed_s')},
                'cases': rows, 'ok': len(valid) == len(rows), 'seconds_total': sum(r['seconds'] or 0 for r in valid), 'memory_bytes_max': max([r['memory_bytes'] or 0 for r in valid] or [0]),
                'disk_bytes_total': disk, 'free_bytes': free, 'disk_short': disk > free, 'existing': self.checks_of(job_id)}

    def converge(self, job_id, levels=1):
        """Start the convergence check of a finished job: the job is the first combination, the finer meshes are queued."""
        scan = self._definition(job_id, levels); rows = combinations(scan)
        for values in rows[1:]:
            try: validate_request(case_request(scan, values))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'加密网格（{case_label(scan, values)}）不能计算：{error}') from None
        with self.lock:
            if self.running() is not None: raise Busy('已有参数扫描或收敛检查在进行，请先暂停它或等它完成。')
            self.app.settle()
            if self.app.busy(): raise Busy('已有任务正在运行，请等待完成。')
            base = self.app.status(job_id)
            scan.update(schema=SCHEMA, scan_id='scan-' + secrets.token_hex(12), state='running', message='', created_at_utc=now(),
                        cases=[{'index': 0, 'values': rows[0], 'phase': 'completed', 'job_id': job_id, 'message': '', 'attempts': 0, 'previous_jobs': [], 'elapsed_s': base.get('elapsed_s'), 'nodes': base.get('nodes')}] +
                              [{'index': i, 'values': values, 'phase': 'queued', 'job_id': None, 'message': '', 'attempts': 0, 'previous_jobs': []} for i, values in enumerate(rows) if i])
            (self.root / scan['scan_id']).mkdir(); self.save(scan); self.advance()
            return {'scan_id': scan['scan_id'], 'count': len(rows)}

    def convergence(self, scan_id, window_us=None):
        """The receivers of every finished mesh of a convergence check against those of the finest finished mesh:
        the change of A1 (complete response) and of A2 (nonlinear increment when the linear control was computed), of the
        phase at f, and the relative difference of the waveforms inside the analysis window."""
        from .domain import parts_for, response
        from .recording import controls_of
        if window_us is not None and not (isinstance(window_us, (list, tuple)) and len(window_us) == 2 and all(number(v) for v in window_us) and window_us[0] < window_us[1]):
            raise ValueError('分析时间窗需要起点早于终点的两个时刻（μs）。')
        with self.lock: scan = self.sync(self.load(scan_id))
        if scan.get('kind') != 'convergence': raise ValueError('这个扫描不是收敛检查。')
        done = [c for c in scan['cases'] if c['phase'] == 'completed']; key = {'window_us': list(window_us) if window_us else None, 'jobs': [c['job_id'] for c in done]}
        cache = self.path(scan_id) / 'convergence.json'
        if cache.exists():
            stored = json.loads(cache.read_text(encoding='utf-8'))
            if stored.get('key') == key: return dict(self.public(scan, cases=False), **stored['value'])
        scene = scan['request']['scene']; f = float(scene['source']['frequency_mhz']); receivers = scene['receivers']; points = [[float(v) for v in r['point_mm']] for r in receivers]
        directions = [np.asarray(r['direction'], dtype=float) / np.linalg.norm(r['direction']) for r in receivers]; loaded = []
        for case in done:
            job = self.app.path(case['job_id']); times, values, m = receiver_series(job / 'record', points); controls = controls_of(m)
            basis = 'nonlinear' if {'linear', 'murnaghan'} <= set(controls) else 'full'; first = 'linear' if 'linear' in controls else 'full'
            wave = {part: np.stack([response(values, part)[:, j] @ directions[j] for j in range(len(points))], axis=1) for part in {'full', first, basis}}
            plan = m.get('time_plan') or {}
            loaded.append({'case': case, 'times': times, 'wave': wave, 'basis': basis, 'first': first, 'nodes': m.get('nodes'), 'layer_step_us': plan.get('layer_step_us', m.get('layer_step_us')), 'scheme': plan.get('scheme'),
                           'harmonics': {part: [harmonics(times, wave[part][:, j], f, window_us) for j in range(len(points))] for part in ('full', basis)}})
        rows = []; reference = loaded[-1] if loaded else None
        for item in loaded:
            keep = slice(None) if window_us is None else (item['times'] >= window_us[0] - 1e-9) & (item['times'] <= window_us[1] + 1e-9); out = []
            for j, receiver in enumerate(receivers):
                h1 = item['harmonics']['full'][j][0]; h2 = item['harmonics'][item['basis']][j][1]; r1 = reference['harmonics']['full'][j][0]; r2 = reference['harmonics'][reference['basis']][j][1]
                relative = lambda a, b: None if a is None or b is None or not b['amplitude_m'] > 0 else a['amplitude_m'] / b['amplitude_m'] - 1
                def difference(part):
                    a, b = item['wave'][part][keep, j], reference['wave'][part][keep, j]; size = float(np.linalg.norm(b))
                    return float(np.linalg.norm(a - b)) / size if size > 0 else None
                out.append({'name': receiver['name'], 'a1_m': h1['amplitude_m'] if h1 else None, 'a2_m': h2['amplitude_m'] if h2 else None, 'a1_change': relative(h1, r1), 'a2_change': relative(h2, r2),
                            'phase1_change_deg': wrapped(h1['phase_deg'] - r1['phase_deg']) if h1 and r1 and h1['phase_deg'] is not None and r1['phase_deg'] is not None else None,
                            'waveform_change': difference(item['first']), 'increment_waveform_change': difference(item['basis']) if item['basis'] == 'nonlinear' else None})
            worst = lambda name: max((abs(r[name]) for r in out if r[name] is not None), default=None)
            rows.append({'index': item['case']['index'], **scan['levels'][item['case']['index']], 'job_id': item['case']['job_id'], 'nodes': item['nodes'], 'elapsed_s': item['case'].get('elapsed_s'), 'layer_step_us': item['layer_step_us'],
                         'scheme': item['scheme'], 'is_reference': item is reference, 'receivers': out,
                         'worst': {name: worst(name) for name in ('a1_change', 'a2_change', 'phase1_change_deg', 'waveform_change', 'increment_waveform_change')}})
        verdict = None
        if len(rows) >= 2 and rows[0]['index'] == 0:
            w = rows[0]['worst']; ok = w['a1_change'] is not None and w['a1_change'] <= CONVERGENCE_TOLERANCE['a1'] and (w['a2_change'] is None or w['a2_change'] <= CONVERGENCE_TOLERANCE['a2'])
            share = lambda v: '不到 0.01%' if v < 5e-5 else f'{v:.2%}'
            verdict = {'converged': bool(ok), 'reference': rows[-1]['label'], 'a1_change': w['a1_change'], 'a2_change': w['a2_change'],
                       'text': (f"原网格（{rows[0]['label']}）与最细网格（{rows[-1]['label']}）相比，各接收点 A₁ 最多变化 {share(w['a1_change'])}"
                                + (f"，A₂ 最多变化 {share(w['a2_change'])}" if w['a2_change'] is not None else '') + '。'
                                + (f"按 A₁ ≤ {CONVERGENCE_TOLERANCE['a1']:.0%}、A₂ ≤ {CONVERGENCE_TOLERANCE['a2']:.0%} 的判据，原网格的结果已收敛。" if ok else
                                   f"超过了 A₁ ≤ {CONVERGENCE_TOLERANCE['a1']:.0%}、A₂ ≤ {CONVERGENCE_TOLERANCE['a2']:.0%} 的判据：请采用加密网格的结果，或再加密一级确认。"))}
        value = {'window_us': key['window_us'], 'frequency_mhz': f, 'a2_basis': loaded[-1]['basis'] if loaded else None, 'tolerance': CONVERGENCE_TOLERANCE, 'ratio': CONVERGENCE_RATIO, 'rows': rows, 'verdict': verdict,
                 'pending': [{'index': c['index'], **scan['levels'][c['index']], 'phase': c['phase'], 'message': c.get('message', '')} for c in scan['cases'] if c['phase'] != 'completed'],
                 'note': '每加密一级：单元阶次低于 5 时阶次加一（各个方向的节点都加密，板厚方向只有一层单元时也一样），阶次已是 5 时单元尺寸乘 0.75。自动时间层按每套网格重新选步长'
                         '（表中列出），网格变细时步长通常随之缩短，但不保证。差值相对最细的已完成网格计算；'
                         '判据只说明这个模型在这些接收点上的变化，不是误差的上界。波形差是分析时间窗内的相对 L2 差，对相位很敏感：相位差 1° 约对应 1.7 %。'}
        cache.write_text(json.dumps({'key': key, 'value': value}, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return dict(self.public(scan, cases=False), **value)

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
            row = {k: case[k] for k in ('index', 'values', 'phase', 'job_id', 'message')}; row['label'] = case_label(scan, case['values'])
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
        writer.writerow(['组合', *[heading(axis) for axis in table['axes']], '状态', '计算记录', '接收点', 'x_mm', 'y_mm', 'z_mm', '方向 x', '方向 y', '方向 z', 'f (MHz)', 'A1 完整 (m)', 'A2 完整 (m)', 'A1 线性 (m)', 'A2 线性 (m)',
                         'A2 非线性增量 (m)', 'A2 几何增量 (m)', 'A2 材料增量 (m)', 'A2/A1^2 (1/m)', 'A2/A1^2 所用的 A2', '节点', '用时 (s)', '最大应变', '说明'])
        blank = lambda v: '' if v is None else v
        for row in table['rows']:
            head = [row['index'] + 1, *row['values'], PHASE_WORDS.get(row['phase'], row['phase']), blank(row['job_id'])]
            if not row.get('receivers'): writer.writerow(head + [''] * 17 + [blank(row.get('nodes')), blank(row.get('elapsed_s')), blank(row.get('max_strain')), row.get('error') or row.get('message') or '']); continue
            for r in row['receivers']:
                writer.writerow(head + [r['name'], *r['point_mm'], *r['direction'], row['frequency_mhz']] + [blank(r[k]) for k in ('a1_m', 'a2_m', 'a1_linear_m', 'a2_linear_m', 'a2_nonlinear_m', 'a2_geometric_m', 'a2_material_m', 'a2_over_a1_squared_per_m')]
                                + [{'nonlinear': '非线性增量', 'full': '完整响应'}[r['a2_basis']], blank(row.get('nodes')), blank(row.get('elapsed_s')), blank(row.get('max_strain')), ''])
        return ('﻿' + out.getvalue()).encode('utf-8')   # with a byte-order mark, so spreadsheet programs read the Chinese headers
