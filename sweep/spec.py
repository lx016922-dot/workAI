"""Sweep specification: validation, case expansion, sampling, links and splits."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from solid.schema import require, exact
from physics.scene import validate_scene, is_plane_strain, V2

SCHEMA = 'workai.sweep.v1'
MAX_CASES = 5000
SAMPLING = ('latin_hypercube', 'uniform')


def get_path(obj, path):
    for key in path.split('.'):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def set_path(obj, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    last = keys[-1]
    if isinstance(obj, list): obj[int(last)] = value
    else: obj[last] = value


def _scalar(value, label):
    require(type(value) in (int, float, bool, str) or value is None or isinstance(value, list), f'{label} 的取值需为 JSON 标量或列表。')
    return value


def load_spec(path):
    path = Path(path); spec = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(spec.get('base_scene'), str):
        scene_path = (path.parent / spec['base_scene']).resolve(); spec['base_scene'] = json.loads(scene_path.read_text(encoding='utf-8')); spec['base_scene_path'] = str(scene_path)
    return validate_spec(spec)


def validate_spec(spec):
    require(isinstance(spec, dict) and spec.get('schema') == SCHEMA, f'扫描规格 schema 需为 {SCHEMA}。')
    s = copy.deepcopy(spec); s.pop('base_scene_path', None)
    allowed = ['schema', 'name', 'base_scene', 'degree', 'grid', 'cases', 'sampling', 'links', 'outputs', 'analysis', 'split', 'budget']
    require(all(k in allowed for k in s), f'扫描规格只接受字段 {allowed}。')
    require(isinstance(s.get('name'), str) and 1 <= len(s['name']) <= 120, '扫描需要 1–120 字符的名称。')
    require(isinstance(s.get('base_scene'), dict) and s['base_scene'].get('schema') == V2 and s['base_scene'].get('dimension') == '2d_plane_strain', '目前只支持 dimension 为 2d_plane_strain 的 v2 基准场景。')
    s['degree'] = s.get('degree', 4); require(type(s['degree']) is int and 2 <= s['degree'] <= 6, '阶次需为 2–6 的整数。')
    grid = s.get('grid', {}); require(isinstance(grid, dict) and all(isinstance(k, str) and isinstance(v, list) and 1 <= len(v) <= 200 for k, v in grid.items()), '网格轴需为 路径 → 取值列表。')
    for k, v in grid.items():
        for value in v: _scalar(value, k)
    cases = s.get('cases', []); require(isinstance(cases, list) and all(isinstance(c, dict) and all(isinstance(k, str) for k in c) for c in cases), '显式用例需为 路径 → 取值 的字典列表。')
    sampling = s.get('sampling')
    if sampling is not None:
        exact(sampling, ['method', 'count', 'seed', 'ranges'], '采样设置'); require(sampling['method'] in SAMPLING, '采样方法需为 latin_hypercube 或 uniform。')
        require(type(sampling['count']) is int and 1 <= sampling['count'] <= MAX_CASES and type(sampling['seed']) is int, '采样数量与随机种子需为整数。')
        require(isinstance(sampling['ranges'], dict) and sampling['ranges'] and all(isinstance(r, list) and len(r) == 2 and all(type(x) in (int, float) for x in r) and r[0] < r[1] for r in sampling['ranges'].values()), '采样范围需为 路径 → [下限, 上限]。')
    links = s.get('links', {}); require(isinstance(links, dict), '联动需为字典。')
    for target, link in links.items():
        exact(link, ['path', 'scale', 'offset'], f'联动 {target}'); require(type(link['scale']) in (int, float) and type(link['offset']) in (int, float), '联动的 scale 与 offset 需为数值。')
    outputs = s.get('outputs', {}); s['outputs'] = {'fields': bool(outputs.get('fields', False)), 'csv': bool(outputs.get('csv', False)), 'browser_record': bool(outputs.get('browser_record', False))}
    analysis = s.get('analysis', {}); require(isinstance(analysis, dict), '分析设置需为字典。')
    s['analysis'] = {'frequency_mhz': analysis.get('frequency_mhz'), 'line': analysis.get('line'), 'window': analysis.get('window', {'mode': 'moving'}), 'point_window_us': analysis.get('point_window_us'), 'mode_for_window': analysis.get('mode_for_window', 'S0')}
    require(s['analysis']['window'].get('mode') in ('moving', 'fixed'), '分析时间窗 mode 需为 moving 或 fixed。')
    split = s.get('split')
    if split is not None:
        exact(split, ['train', 'validation', 'test', 'group_by', 'seed'], '划分设置')
        fractions = [split['train'], split['validation'], split['test']]; require(all(type(f) in (int, float) and 0 <= f <= 1 for f in fractions) and abs(sum(fractions) - 1) < 1e-9, '划分比例需为和为 1 的三个数。')
        require(isinstance(split['group_by'], list) and all(isinstance(k, str) for k in split['group_by']) and type(split['seed']) is int, 'group_by 需为路径列表，seed 为整数。')
    budget = s.get('budget', {}); s['budget'] = {'seconds_per_case': int(budget.get('seconds_per_case', 1800)), 'workers': int(budget.get('workers', 1))}
    require(1 <= s['budget']['seconds_per_case'] <= 86400 and 1 <= s['budget']['workers'] <= 64, '预算：每例 1–86400 秒，并行 1–64。')
    return s


def _samples(sampling):
    rng = np.random.default_rng(sampling['seed']); paths = list(sampling['ranges']); count = sampling['count']
    if sampling['method'] == 'latin_hypercube':
        unit = np.stack([(rng.permutation(count) + rng.uniform(size=count)) / count for _ in paths], axis=1)
    else: unit = rng.uniform(size=(count, len(paths)))
    rows = []
    for i in range(count):
        rows.append({p: float(lo + (hi - lo) * unit[i, j]) for j, (p, (lo, hi)) in enumerate(sampling['ranges'].items())})
    return rows


def expand_cases(spec):
    """All case parameter dictionaries: factorial grid × explicit cases, plus sampled cases."""
    grid = spec.get('grid', {}); keys = list(grid); combos = [{}]
    for key in keys: combos = [dict(c, **{key: v}) for c in combos for v in grid[key]]
    explicit = spec.get('cases') or [{}]
    rows = [dict(c, **e) for c in combos for e in explicit] if (grid or spec.get('cases') or not spec.get('sampling')) else []
    if spec.get('sampling'): rows += _samples(spec['sampling'])
    require(1 <= len(rows) <= MAX_CASES, f'用例数需在 1–{MAX_CASES} 之间，当前 {len(rows)}。')
    return rows


def build_scene(spec, parameters):
    """Apply overrides and links to the base scene, then validate it like the solver does."""
    scene = copy.deepcopy(spec['base_scene'])
    for path, value in parameters.items(): set_path(scene, path, value)
    for target, link in (spec.get('links') or {}).items():
        set_path(scene, target, get_path(scene, link['path']) * link['scale'] + link['offset'])
    h = scene['numerics']['h_mm']; probe = copy.deepcopy(scene); probe['numerics']['h_mm'] = max(.02, h / spec['degree'])
    validated = validate_scene(probe); validated['numerics']['h_mm'] = h
    require(is_plane_strain(validated), '用例必须保持二维平面应变。')
    return validated


def case_id(scene, degree):
    return 'case-' + hashlib.sha256(json.dumps({'scene': scene, 'degree': degree}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]


def assign_splits(spec, cases):
    """Group-aware split: every case sharing the group key lands in the same subset."""
    split = spec.get('split')
    if split is None: return ['train'] * len(cases)
    keys = []
    for case in cases:
        group = tuple(json.dumps(get_path(case['scene'], p), sort_keys=True) for p in split['group_by']) if split['group_by'] else (case['case_id'],)
        keys.append(group)
    unique = sorted(set(keys), key=lambda g: hashlib.sha256((json.dumps(g) + str(split['seed'])).encode()).hexdigest())
    n = len(unique); n_train = round(split['train'] * n); n_val = round(split['validation'] * n)
    if n_train + n_val > n: n_val = n - n_train
    label = {}
    for i, g in enumerate(unique): label[g] = 'train' if i < n_train else 'validation' if i < n_train + n_val else 'test'
    return [label[k] for k in keys]


def plan(spec):
    """Expand, build, validate and identify every case; duplicates (same validated scene) collapse."""
    rows = []; seen = {}
    for parameters in expand_cases(spec):
        scene = build_scene(spec, parameters); cid = case_id(scene, spec['degree'])
        if cid in seen: continue
        seen[cid] = True; rows.append({'case_id': cid, 'parameters': parameters, 'scene': scene})
    splits = assign_splits(spec, rows)
    for row, split in zip(rows, splits): row['split'] = split
    return rows
