"""Editor preview: validation, source signals, mesh size and resolution for a scene. Never starts a computation."""
import copy
import math
import shutil

import numpy as np

from .domain import V2, validate_request

# Ranges of the original eight-case acceptance (version-1 scenes); outside them results are research results.
ORIGINAL_SCOPE = [('geometry.length_mm', '长度', 8, 12, 'mm'), ('geometry.width_mm', '板宽', 3, 4, 'mm'), ('geometry.thickness_mm', '板厚', 1.5, 2.1, 'mm'),
                  ('source.frequency_mhz', '主激励频率', .18, .28, 'MHz'), ('source.amplitude_mpa', '主激励幅值', .6, 1.5, 'MPa')]
CONTROLS = 3


def _get(scene, path):
    value = scene
    for key in path.split('.'):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def scope(scene):
    """Items outside the original acceptance range (information only)."""
    out = []
    kind = scene.get('geometry', {}).get('kind')
    for path, label, lo, hi, unit in ORIGINAL_SCOPE:
        if kind == 'pipe' and path in ('geometry.width_mm', 'geometry.thickness_mm'): continue
        value = _get(scene, path)
        if isinstance(value, (int, float)) and not lo <= value <= hi: out.append(f'{label} {value:g} {unit}（原验收 {lo:g}–{hi:g}）')
    from physics.materials import is_custom
    if is_custom(scene.get('material')):
        m = scene['material']; out.append('材料：' + ('自定义应变能 ' if m.get('model') == 'energy' else 'Murnaghan 自定义常数 ') + str(m.get('name') or ''))
    if scene.get('schema') == V2:
        extra = len(scene.get('sources') or []); faces = {s.get('face') for s in [scene.get('source', {}), *(scene.get('sources') or [])]}
        if extra: out.append(f'{extra + 1} 个激励源')
        if faces - {'xmin', 'xmax'}: out.append('非 X 端面加载')
        if scene.get('boundaries'): out.append('对称面 / 吸收层')
        signals = {(s.get('signal') or {}).get('type', 'tone_burst') for s in [scene.get('source', {}), *(scene.get('sources') or [])]}
        if signals - {'tone_burst'}: out.append('非猝发音信号')
    return out


def _material(material):
    """Constants, wave speeds, checks and reference stresses of the scene material, or the reason it is invalid."""
    from physics.materials import describe, is_custom, validate_material
    try:
        normalized = validate_material(material) if is_custom(material) else material
        out = describe(normalized); out['ok'] = True
        out['engine'] = ('物理 AI：求导滤波器与时间层沿用训练结果，能量层使用该材料的二阶、三阶弹性常数。精度需用谱元参考或 COMSOL 逐例核对。'
                         if is_custom(material) else '物理 AI：使用训练好的能量层（该多晶铝即训练材料）。')
        return out
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError, np.linalg.LinAlgError) as error:
        return {'ok': False, 'error': str(error)}


def _signals(scene):
    """Waveform and spectrum of every source whose own signal is valid (the rest of the scene may still have errors)."""
    from physics.signals import evaluate, support_us, bandwidth_mhz, highest_frequency_mhz, validate_signal, default_signal
    rows = []
    for index, raw in enumerate([scene.get('source') or {}, *(scene.get('sources') or [])]):
        try:
            src = dict(raw); src['signal'] = validate_signal(raw.get('signal') or default_signal(), raw)
            if not all(isinstance(src.get(k), (int, float)) for k in ('frequency_mhz', 'amplitude_mpa', 'cycles', 'delay_us', 'phase_deg')): continue
        except (ValueError, KeyError, TypeError):
            continue
        start, end = support_us(src); top = highest_frequency_mhz(src)
        step = min((end - start) / 600, 1 / (40 * top)); times = np.arange(start - 0.05 * (end - start), end + 0.05 * (end - start) + step, step)
        values = evaluate(src, times) * src['amplitude_mpa']
        count = int(2 ** math.ceil(math.log2(max(16 * len(times), 4096))))
        spectrum = np.abs(np.fft.rfft(evaluate(src, start + np.arange(count) * step))); freqs = np.fft.rfftfreq(count, step)
        keep = freqs <= max(3 * src['frequency_mhz'], 1.3 * top); spectrum = spectrum[keep]; freqs = freqs[keep]
        if len(freqs) > 400: pick = np.linspace(0, len(freqs) - 1, 400).round().astype(int); freqs, spectrum = freqs[pick], spectrum[pick]
        if len(times) > 800: pick = np.linspace(0, len(times) - 1, 800).round().astype(int); times, values = times[pick], values[pick]
        peak = float(spectrum.max()) or 1.
        rows.append({'index': index, 'name': src.get('name') or ('主激励' if index == 0 else f'激励 {index + 1}'), 'type': (src.get('signal') or {}).get('type', 'tone_burst'),
                     'frequency_mhz': src['frequency_mhz'], 'support_us': [start, end], 'band_30db_mhz': bandwidth_mhz(src, 30.),
                     'times_us': times.round(6).tolist(), 'values_mpa': values.round(9).tolist(),
                     'spectrum_mhz': freqs.round(6).tolist(), 'spectrum_db': (20 * np.log10(np.maximum(spectrum / peak, 1e-6))).round(3).tolist()})
    return rows


def _mesh(scene, degree):
    from spectral3d.mesh import make_mesh
    mesh = make_mesh(scene, degree, scene['numerics']['h_mm'])
    out = {'nodes': int(len(mesh['points'])), 'elements': int(len(mesh['elements'])), 'degree': degree, 'h_mm': scene['numerics']['h_mm'],
           'node_spacing_mm': scene['numerics']['h_mm'] / degree}
    if 'origin' in mesh and len(mesh['origin']) <= 6000:
        edges = []
        for axis in range(3):
            values = np.unique(np.round(np.concatenate([mesh['origin'][:, axis], mesh['origin'][:, axis] + mesh['size'][:, axis]]) * 1000, 6))
            edges.append(values.tolist())
        out['grid_mm'] = edges
    return out


def preview(body, data_directory=None):
    """{'ok', 'errors', 'scope', 'signals', 'mesh', 'resolution', 'record'} for an editor scene; errors never raise."""
    out = {'ok': False, 'errors': [], 'warnings': [], 'scope': [], 'signals': [], 'mesh': None, 'resolution': [], 'record': None, 'material': None}
    if not isinstance(body, dict) or not isinstance(body.get('scene'), dict):
        out['errors'].append('需要场景对象。'); return out
    try:
        request = validate_request(copy.deepcopy(body)); scene = request['scene']; out['ok'] = True
    except (ValueError, KeyError, TypeError) as error:
        out['errors'].append(str(error)); scene = None
    raw = scene or body['scene']; out['scope'] = scope(raw); out['material'] = _material(raw.get('material'))
    if raw.get('schema') == V2:
        try:
            out['signals'] = _signals(scene or raw)
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as error:
            if not out['errors']: out['errors'].append(str(error))
    degree = body.get('degree') if body.get('degree') in (3, 4, 5) else 4
    if body.get('engine') == 'coarse':
        # the fast engine runs on the learned model's own coarse grid
        from .coarse_engine import model_summary, out_of_range
        summary = model_summary(); degree = summary['degree']
        out['engine'] = dict(summary, out_of_training_range=out_of_range(scene or body['scene']) if (scene or body['scene']).get('schema') == V2 else [])
        if scene is not None: scene = dict(scene, numerics=dict(scene['numerics'], h_mm=summary['h_mm']))
    if scene is not None:
        try:
            out['mesh'] = _mesh(scene, degree)
        except (ValueError, KeyError) as error:
            out['ok'] = False; out['errors'].append(f'网格：{error}')
        from physics.materials import slowest_shear_speed
        shear = slowest_shear_speed(scene['material']); spacing = scene['numerics']['h_mm'] / degree
        for row in out['signals']:
            nodes = shear / (2 * row['band_30db_mhz'] * 1e6) / (spacing * 1e-3)
            out['resolution'].append({'name': row['name'], 'band_30db_mhz': row['band_30db_mhz'], 'nodes_per_2f_shear_wavelength': nodes})
        acq = scene['acquisition']; samples = int(round(acq['duration_us'] / acq['sample_step_us'])) + 1
        if out['mesh']:
            size = samples * out['mesh']['nodes'] * 3 * 8 * CONTROLS
            free = shutil.disk_usage(data_directory).free if data_directory else None
            out['record'] = {'samples': samples, 'uncompressed_bytes': size, 'free_bytes': free}
            if free is not None and size > free: out['warnings'].append('完整记录超过数据目录的可用磁盘空间。')
    return out
