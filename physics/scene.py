"""Scene version 2: version-1 core plus optional signals, extra sources, boundaries.

A version-1 scene is accepted unchanged through `solid.schema.validate_scene`,
so every existing job, test and the offline network keep their exact inputs.
A version-2 scene is a version-1 scene (`schema` set to the v2 identifier)
whose `source` may carry `signal` and `profile`, plus optional top-level
`sources` (further tractions of the same shape) and `boundaries` (symmetry,
antisymmetry and absorbing faces). Validation reuses the version-1 rules for
every source and adds the signal support, bandwidth and boundary checks.
"""
import copy
import math
from solid.schema import validate_scene as validate_v1, require, number, exact
from .signals import validate_signal, support_us, bandwidth_mhz, signal_of
from .loads import validate_profile
from .boundaries import FACES, RAMP_POWER

V1 = 'workai.hyperelastic.solid.v1'
V2 = 'workai.hyperelastic.solid.v2'
MAX_SOURCES = 16
MAX_BOUNDARIES = 8
MAX_LINES = 8
MAX_LINE_POINTS = 1024
BOUNDARY_TYPES = ('symmetry', 'antisymmetry', 'absorbing')
DIMENSIONS = ('3d', '2d_plane_strain')
PLANE_FACES = ('xmin', 'xmax', 'zmin', 'zmax')

def is_v2(scene):
    return isinstance(scene, dict) and scene.get('schema') == V2

def dimension(scene):
    return scene.get('dimension', '3d')

def is_plane_strain(scene):
    return dimension(scene) == '2d_plane_strain'

def receiver_lines(scene):
    return list(scene.get('receiver_lines') or [])

def sources(scene):
    """Primary source followed by the extra sources (empty list for version 1)."""
    return [scene['source']] + list(scene.get('sources') or [])

def boundaries(scene):
    return list(scene.get('boundaries') or [])

def extent_mm(geometry, axis):
    """Bounding extent of a parametric geometry along x, y, z; None for CAD."""
    kind = geometry['kind']
    if kind == 'cad': return None
    if axis == 0: return geometry['length_mm']
    if kind == 'pipe': return 2 * geometry['outer_radius_mm']
    return geometry['width_mm'] if axis == 1 else geometry['thickness_mm']

def _shear_speed(material):
    return math.sqrt(material['mu_pa'] / material['rho_kg_m3'])

def _validate_source(raw, core):
    """Validate one source (primary or extra) with the version-1 rules plus signal/profile."""
    require(isinstance(raw, dict), '激励需要是对象。')
    stripped = dict(raw); signal = stripped.pop('signal', None); profile = stripped.pop('profile', None); name = stripped.pop('name', None)
    require(name is None or (isinstance(name, str) and 1 <= len(name) <= 60), '激励 name 需为 1–60 个字符。')
    tone = signal is None or (isinstance(signal, dict) and signal.get('type') == 'tone_burst')
    probe = copy.deepcopy(core); probe['schema'] = V1; probe.pop('sources', None); probe.pop('boundaries', None)
    probe['source'] = dict(stripped, cycles=stripped.get('cycles') if tone else 1)
    validated = validate_v1(probe)['source']
    if not tone: validated['cycles'] = number(stripped.get('cycles'), 1, 20, 'cycles')
    if signal is not None: validated['signal'] = validate_signal(signal, validated)
    if profile is not None: validated['profile'] = validate_profile(profile, validated)
    if name is not None: validated['name'] = name
    start, end = support_us(validated)
    require(end <= core['acquisition']['duration_us'] + 1e-9, '计算时长需要覆盖完整激励信号。')
    # Mesh screen on the real signal content, not only the carrier.
    checked = max(2 * validated['frequency_mhz'], bandwidth_mhz(validated, 30.))
    require(core['numerics']['h_mm'] <= _shear_speed(core['material']) / (checked * 1e6 * 8) * 1000 + 1e-12,
            f'网格需满足信号 −30 dB 带宽（{checked:.4g} MHz）对应剪切波长至少 8 个单元；请减小网格或信号带宽。')
    return validated

def _validate_boundaries(items, core):
    require(isinstance(items, list) and len(items) <= MAX_BOUNDARIES, f'边界条件需为最多 {MAX_BOUNDARIES} 项的列表。')
    kind = core['geometry']['kind']; used = set(); loaded = {s['face'] for s in sources(core)}; rows = []
    for raw in items:
        require(isinstance(raw, dict) and raw.get('type') in BOUNDARY_TYPES, '边界类型需为 symmetry、antisymmetry 或 absorbing。')
        face = raw.get('face'); require(face in FACES, '边界面需为 xmin/xmax/ymin/ymax/zmin/zmax。')
        require(kind != 'pipe' or face[0] == 'x', '管只能在 X 端面施加对称或吸收边界。')
        require(face != core['boundary']['fixed_face'], '边界面不能与固定面相同。')
        require(face not in used, '同一面只能有一种边界条件。'); used.add(face)
        b = dict(raw)
        if b['type'] == 'absorbing':
            require(face not in loaded, '吸收层不能放在加载面上。')
            keys = ['type', 'face', 'length_mm', 'ramp'] + (['alpha_max_per_s'] if 'alpha_max_per_s' in b else ['attenuation_db'])
            exact(b, keys, '吸收边界'); require(b['ramp'] in RAMP_POWER, '吸收层渐变需为 linear、quadratic 或 cubic。')
            b['length_mm'] = number(b['length_mm'], .01, 1000, '吸收层长度')
            if 'alpha_max_per_s' in b: b['alpha_max_per_s'] = number(b['alpha_max_per_s'], 1e2, 1e9, 'alpha_max_per_s')
            else: b['attenuation_db'] = number(b['attenuation_db'], 10, 120, '设计衰减')
            extent = extent_mm(core['geometry'], 'xyz'.index(face[0]))
            require(extent is None or b['length_mm'] <= extent / 2 + 1e-9, '吸收层长度不能超过该方向尺寸的一半。')
            require(b['length_mm'] >= core['numerics']['h_mm'], '吸收层长度不应小于网格尺寸。')
        else: exact(b, ['type', 'face'], '对称边界')
        rows.append(b)
    return rows

def _validate_lines(items, core):
    from solid.schema import vector
    require(isinstance(items, list) and len(items) <= MAX_LINES, f'接收线最多 {MAX_LINES} 条。')
    rows = []; names = set()
    for raw in items:
        exact(raw, ['name', 'from_mm', 'to_mm', 'count', 'direction'], '接收线')
        require(isinstance(raw['name'], str) and 1 <= len(raw['name']) <= 60 and raw['name'] not in names, '接收线需要唯一的 1–60 字符名称。'); names.add(raw['name'])
        require(type(raw['count']) is int and 2 <= raw['count'] <= MAX_LINE_POINTS, f'接收线点数需为 2–{MAX_LINE_POINTS} 的整数。')
        start = vector(raw['from_mm'], '接收线起点'); end = vector(raw['to_mm'], '接收线终点'); direction = vector(raw['direction'], '接收线方向', True)
        require(math.dist(start, end) > 1e-9, '接收线起点与终点重合。')
        rows.append({'name': raw['name'], 'from_mm': start, 'to_mm': end, 'count': raw['count'], 'direction': direction})
    return rows

def _plane_strain_checks(core):
    """The x–z section: no y faces, no y motion, no y-dependence."""
    require(core['geometry']['kind'] in ('plate', 'notch'), '二维平面应变目前支持板与贯穿宽度的缺口。')
    for src in sources(core):
        require(src['face'] in PLANE_FACES, '二维平面应变的激励面需为 xmin/xmax/zmin/zmax。')
        require(abs(src['direction'][1]) < 1e-12, '二维平面应变的激励方向不能有 y 分量。')
        require((src.get('profile') or {}).get('along') != 'y', '二维平面应变的梳状方向不能是 y。')
    require(core['boundary']['fixed_face'] in ('none',) + PLANE_FACES, '二维平面应变的固定面需为 xmin/xmax/zmin/zmax。')
    for b in boundaries(core): require(b['face'] in PLANE_FACES, '二维平面应变的边界面需为 xmin/xmax/zmin/zmax。')
    for rx in core['receivers'] + receiver_lines(core): require(abs(rx['direction'][1]) < 1e-12, '二维平面应变的接收方向不能有 y 分量。')

def validate_scene(value):
    """Version-1 scenes pass through unchanged; version-2 scenes return a normalized copy."""
    require(isinstance(value, dict), '场景需要是对象。')
    if value.get('schema') != V2: return validate_v1(value)
    s = copy.deepcopy(value); extra_sources = s.pop('sources', []); extra_boundaries = s.pop('boundaries', [])
    dim = s.pop('dimension', '3d'); lines = s.pop('receiver_lines', []); require(dim in DIMENSIONS, 'dimension 需为 3d 或 2d_plane_strain。')
    require(isinstance(extra_sources, list) and len(extra_sources) <= MAX_SOURCES - 1, f'额外激励最多 {MAX_SOURCES - 1} 个。')
    primary = s['source'] if isinstance(s.get('source'), dict) else {}
    s['source'] = {k: v for k, v in primary.items() if k not in ('signal', 'profile', 'name')}; s['schema'] = V1
    if isinstance(primary.get('signal'), dict) and primary['signal'].get('type') != 'tone_burst': s['source']['cycles'] = 1
    core = validate_v1(s); core['schema'] = V2
    core['source'] = _validate_source(primary, core)
    core['sources'] = [_validate_source(src, core) for src in extra_sources]
    names = [src.get('name') for src in core['sources']]
    require(all(isinstance(name, str) for name in names), '额外激励需要 1–60 字符的 name。')
    require(len(set(names)) == len(names) and core['source'].get('name', 'primary') not in names, '激励名称不能重复。')
    core['boundaries'] = _validate_boundaries(extra_boundaries, core)
    core['dimension'] = dim; core['receiver_lines'] = _validate_lines(lines, core)
    if is_plane_strain(core): _plane_strain_checks(core)
    return core

def spectral_admissible(scene):
    """What the GLL solvers can load: parametric shapes and complete planar faces (pipes: x ends)."""
    kind = scene['geometry']['kind']; require(kind in ('plate', 'pipe', 'hole', 'notch'), '高阶计算支持参数化板、管、通孔和缺口。')
    require(not is_plane_strain(scene), '该场景声明为二维平面应变，请使用 spectral2d 求解器。')
    for src in sources(scene):
        require(src['face'] in FACES, '高阶计算的激励需放在坐标平面端面上，不支持曲面局部区域。')
        require(kind != 'pipe' or src['face'][0] == 'x', '管的高阶计算只能在 X 端面加载。')
    fixed = scene['boundary']['fixed_face']; require(kind != 'pipe' or fixed in ('none', 'xmin', 'xmax'), '管的高阶计算固定面只能是 X 端面。')
    return scene

def summary(scene):
    """Short, JSON-ready description for status files and diagnostics."""
    from .signals import describe
    return {'schema': scene['schema'], 'dimension': dimension(scene), 'sources': [dict(describe(src), name=src.get('name', 'primary'), face=src['face'], profile=(src.get('profile') or {'type': 'uniform'})['type']) for src in sources(scene)],
            'boundaries': boundaries(scene), 'fixed_face': scene['boundary']['fixed_face']}
