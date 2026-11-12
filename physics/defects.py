"""Defects in a plate: cracks, delaminations, contact interfaces, notches and holes (scene version 2).

A version-2 plate scene may list `defects`. Every defect is written in plate coordinates (millimetres):
x along the length from the xmin end, y across the width from the ymin side, z through the thickness from the
bottom face (zmax is the top face). Each type reduces to the three primitives the GLL mesh can represent exactly:

* split surface: an axis-aligned rectangle on which the mesh nodes are duplicated, so the two faces separate
  (crack, delamination). With ``closed`` (crack, delamination) or type ``contact`` the faces touch at rest and
  carry unilateral normal contact: compression passes, tension opens the interface (a "breathing" defect);
* void box: elements removed (notch);
* hole: a circular cylinder through the thickness (axis z) or through the width (axis y, side-drilled), meshed
  with a curved O-grid patch that conforms to the surrounding grid.

Crack fronts that lie inside the material are closed (the nodes there stay shared); crack edges on the plate
surface (or on a symmetry plane, where the defect continues in the mirrored half) are open.
"""
import copy
from solid.schema import require, number

TYPES = ('crack', 'delamination', 'contact', 'notch', 'hole')
LABELS = {'crack': '裂纹', 'delamination': '分层', 'contact': '接触界面', 'notch': '缺口', 'hole': '孔'}
MAX_DEFECTS = 12
GAP_MM = .2           # smallest clear distance between two defects
MIN_LIGAMENT_MM = .1  # smallest ligament between a defect and an outer face it does not open onto
FIELDS = {
    'crack': {'required': ('x_mm', 'depth_mm'), 'optional': ('name', 'from', 'z_mm', 'y_range_mm', 'closed')},
    'delamination': {'required': ('x_start_mm', 'x_end_mm', 'depth_mm'), 'optional': ('name', 'y_range_mm', 'closed')},
    'contact': {'required': ('orientation',), 'optional': ('name', 'x_mm', 'depth_mm', 'x_start_mm', 'x_end_mm', 'y_range_mm')},
    'notch': {'required': ('x_mm', 'width_mm', 'depth_mm'), 'optional': ('name', 'from', 'y_range_mm')},
    'hole': {'required': ('x_mm', 'radius_mm'), 'optional': ('name', 'axis', 'y_mm', 'z_mm')},
}


def plate_size(scene):
    g = scene['geometry']
    return g['length_mm'], g['width_mm'], g['thickness_mm']


def _range(value, lo, hi, name):
    require(isinstance(value, list) and len(value) == 2, f'{name} 需要两个数值 [起点, 终点]。')
    a = number(value[0], lo, hi, name); b = number(value[1], lo, hi, name)
    require(b - a >= .1, f'{name} 的终点需比起点大 0.1 mm 以上。')
    return [a, b]


def _bool(value, name):
    require(type(value) is bool, f'{name} 需为 true 或 false。')
    return value


def validate_defect(raw, scene, index):
    """Normalized copy of one defect (all optional fields filled)."""
    L, W, T = plate_size(scene)
    require(isinstance(raw, dict) and raw.get('type') in TYPES, f'缺陷 {index + 1}：类型需为 裂纹 crack、分层 delamination、接触界面 contact、缺口 notch 或 孔 hole。')
    kind = raw['type']; spec = FIELDS[kind]; label = f'{LABELS[kind]} {index + 1}'
    extra = set(raw) - {'type'} - set(spec['required']) - set(spec['optional'])
    require(not extra, f'{label}：不认识的字段 {", ".join(sorted(extra))}。')
    for key in spec['required']: require(key in raw, f'{label}：缺少 {key}。')
    d = {'type': kind}
    name = raw.get('name')
    require(name is None or (isinstance(name, str) and 1 <= len(name) <= 40), f'{label}：名称需为 1–40 个字符。')
    d['name'] = name or label
    if 'y_range_mm' in spec['optional']:
        d['y_range_mm'] = _range(raw['y_range_mm'], 0, W, f'{label} 的宽度范围') if raw.get('y_range_mm') is not None else [0., W]
    if kind == 'crack':
        d['x_mm'] = number(raw['x_mm'], MIN_LIGAMENT_MM, L - MIN_LIGAMENT_MM, f'{label} 的位置 x')
        d['from'] = raw.get('from', 'top'); require(d['from'] in ('top', 'bottom', 'inside'), f'{label}：起始面需为 top（上表面开口）、bottom（下表面开口）或 inside（内部埋藏）。')
        if d['from'] == 'inside':
            d['depth_mm'] = number(raw['depth_mm'], .1, T - 2 * MIN_LIGAMENT_MM, f'{label} 的高度')
            d['z_mm'] = number(raw.get('z_mm', T / 2), 0, T, f'{label} 的中心高度 z')
            require(d['z_mm'] - d['depth_mm'] / 2 >= MIN_LIGAMENT_MM - 1e-9 and d['z_mm'] + d['depth_mm'] / 2 <= T - MIN_LIGAMENT_MM + 1e-9,
                    f'{label}：内部裂纹需完整位于板内，距上下表面至少 {MIN_LIGAMENT_MM} mm。')
        else:
            require('z_mm' not in raw, f'{label}：只有内部裂纹（from=inside）需要 z_mm。')
            d['depth_mm'] = number(raw['depth_mm'], .1, T - MIN_LIGAMENT_MM, f'{label} 的深度')
        d['closed'] = _bool(raw.get('closed', False), f'{label} 的 closed')
    elif kind == 'delamination':
        d['x_start_mm'] = number(raw['x_start_mm'], MIN_LIGAMENT_MM, L - MIN_LIGAMENT_MM, f'{label} 的起点 x')
        d['x_end_mm'] = number(raw['x_end_mm'], MIN_LIGAMENT_MM, L - MIN_LIGAMENT_MM, f'{label} 的终点 x')
        require(d['x_end_mm'] - d['x_start_mm'] >= .2, f'{label}：分层长度至少 0.2 mm（终点需大于起点）。')
        d['depth_mm'] = number(raw['depth_mm'], MIN_LIGAMENT_MM, T - MIN_LIGAMENT_MM, f'{label} 距上表面的深度')
        d['closed'] = _bool(raw.get('closed', False), f'{label} 的 closed')
    elif kind == 'contact':
        d['orientation'] = raw['orientation']; require(d['orientation'] in ('vertical', 'horizontal'), f'{label}：方向需为 vertical（竖直截面，整个横截面接触）或 horizontal（水平层间界面）。')
        if d['orientation'] == 'vertical':
            require(all(k not in raw for k in ('depth_mm', 'x_start_mm', 'x_end_mm')), f'{label}：竖直接触界面只需要 x_mm。')
            d['x_mm'] = number(raw.get('x_mm', L / 2), MIN_LIGAMENT_MM, L - MIN_LIGAMENT_MM, f'{label} 的位置 x')
            require(d['y_range_mm'][0] <= 1e-9 and d['y_range_mm'][1] >= W - 1e-9, f'{label}：竖直接触界面贯穿整个横截面。')
        else:
            require('x_mm' not in raw, f'{label}：水平接触界面用 x_start_mm、x_end_mm 与 depth_mm。')
            d['x_start_mm'] = number(raw.get('x_start_mm', L / 4), MIN_LIGAMENT_MM, L - MIN_LIGAMENT_MM, f'{label} 的起点 x')
            d['x_end_mm'] = number(raw.get('x_end_mm', 3 * L / 4), MIN_LIGAMENT_MM, L - MIN_LIGAMENT_MM, f'{label} 的终点 x')
            require(d['x_end_mm'] - d['x_start_mm'] >= .2, f'{label}：接触长度至少 0.2 mm（终点需大于起点）。')
            d['depth_mm'] = number(raw.get('depth_mm', T / 2), MIN_LIGAMENT_MM, T - MIN_LIGAMENT_MM, f'{label} 距上表面的深度')
    elif kind == 'notch':
        d['width_mm'] = number(raw['width_mm'], .1, L / 2, f'{label} 的宽度')
        d['x_mm'] = number(raw['x_mm'], 0, L, f'{label} 的位置 x')
        require(d['width_mm'] / 2 + MIN_LIGAMENT_MM <= d['x_mm'] <= L - d['width_mm'] / 2 - MIN_LIGAMENT_MM, f'{label}：缺口需完整位于板长范围内。')
        d['depth_mm'] = number(raw['depth_mm'], .1, T - MIN_LIGAMENT_MM, f'{label} 的深度')
        d['from'] = raw.get('from', 'top'); require(d['from'] in ('top', 'bottom'), f'{label}：缺口开口面需为 top 或 bottom。')
    else:
        d['axis'] = raw.get('axis', 'z'); require(d['axis'] in ('z', 'y'), f'{label}：孔的轴向需为 z（贯穿板厚的通孔）或 y（贯穿板宽的横孔）。')
        d['radius_mm'] = number(raw['radius_mm'], .1, 1000, f'{label} 的半径')
        d['x_mm'] = number(raw['x_mm'], 0, L, f'{label} 的位置 x'); r = d['radius_mm']
        require(r + MIN_LIGAMENT_MM <= d['x_mm'] <= L - r - MIN_LIGAMENT_MM, f'{label}：孔需完整位于板长范围内。')
        if d['axis'] == 'z':
            require('z_mm' not in raw, f'{label}：通孔（axis z）用 y_mm 定位。')
            d['y_mm'] = number(raw.get('y_mm', W / 2), 0, W, f'{label} 的位置 y')
            require(r + MIN_LIGAMENT_MM <= d['y_mm'] <= W - r - MIN_LIGAMENT_MM, f'{label}：通孔需完整位于板宽内（距侧面至少 {MIN_LIGAMENT_MM} mm）。')
        else:
            require('y_mm' not in raw, f'{label}：横孔（axis y）用 z_mm 定位。')
            d['z_mm'] = number(raw.get('z_mm', T / 2), 0, T, f'{label} 的位置 z')
            require(r + MIN_LIGAMENT_MM <= d['z_mm'] <= T - r - MIN_LIGAMENT_MM, f'{label}：横孔需完整位于板厚内（距上下表面至少 {MIN_LIGAMENT_MM} mm）。')
    return d


def footprint(d, scene):
    """Axis-aligned box (x, y, z ranges in mm) the defect occupies; holes include their mesh patch."""
    L, W, T = plate_size(scene)
    if d['type'] == 'crack':
        z = {'top': [T - d['depth_mm'], T], 'bottom': [0., d['depth_mm']]}.get(d['from']) or [d['z_mm'] - d['depth_mm'] / 2, d['z_mm'] + d['depth_mm'] / 2]
        return [[d['x_mm'], d['x_mm']], list(d['y_range_mm']), z]
    if d['type'] == 'delamination' or (d['type'] == 'contact' and d['orientation'] == 'horizontal'):
        z = T - d['depth_mm']; return [[d['x_start_mm'], d['x_end_mm']], list(d.get('y_range_mm') or [0., W]), [z, z]]
    if d['type'] == 'contact':
        return [[d['x_mm'], d['x_mm']], [0., W], [0., T]]
    if d['type'] == 'notch':
        z = [T - d['depth_mm'], T] if d['from'] == 'top' else [0., d['depth_mm']]
        return [[d['x_mm'] - d['width_mm'] / 2, d['x_mm'] + d['width_mm'] / 2], list(d['y_range_mm']), z]
    return hole_block(d, scene)


def hole_block(d, scene):
    """The square patch around a hole that the O-grid replaces: x, y, z ranges (mm). Edges near an outer face snap to it."""
    L, W, T = plate_size(scene); r = d['radius_mm']; half = r + max(r, .5)
    in_plane = ('x', 'y') if d['axis'] == 'z' else ('x', 'z')
    centre = {'x': d['x_mm'], 'y': d.get('y_mm'), 'z': d.get('z_mm')}; size = {'x': L, 'y': W, 'z': T}
    box = {'x': None, 'y': [0., W], 'z': [0., T]}
    for axis in in_plane:
        a, b = centre[axis] - half, centre[axis] + half
        if a < .3 * half: a = 0.
        if b > size[axis] - .3 * half: b = size[axis]
        box[axis] = [a, b]
    return [box['x'], box['y'], box['z']]


def _overlap(a, b, gap):
    return all(lo1 < hi2 + gap and lo2 < hi1 + gap for (lo1, hi1), (lo2, hi2) in zip(a, b))


def validate_defects(items, scene):
    """Normalized defect list; checks types, positions and that defects keep apart."""
    if items is None: return []
    require(isinstance(items, list) and len(items) <= MAX_DEFECTS, f'缺陷需为最多 {MAX_DEFECTS} 项的列表。')
    if not items: return []
    require(scene['geometry']['kind'] == 'plate', '缺陷目前加在平板上：请先把几何设为平板（通孔与缺口也可作为缺陷添加）。')
    rows = [validate_defect(raw, scene, i) for i, raw in enumerate(items)]
    names = [d['name'] for d in rows]; require(len(set(names)) == len(names), '缺陷名称不能重复。')
    boxes = [footprint(d, scene) for d in rows]
    for i in range(len(rows)):
        for j in range(i):
            require(not _overlap(boxes[i], boxes[j], GAP_MM),
                    f'{rows[j]["name"]} 与 {rows[i]["name"]} 重叠或距离小于 {GAP_MM} mm；目前不支持相交的缺陷，请拉开位置。')
    return rows


def summary(defects):
    return [{'type': d['type'], 'name': d['name'], 'contact': is_contact(d)} for d in defects]


def is_contact(d):
    return d['type'] == 'contact' or (d['type'] in ('crack', 'delamination') and d.get('closed', False))


def primitives(defects, scene):
    """Mesh primitives: split surfaces, void boxes and holes (mm)."""
    L, W, T = plate_size(scene); splits, voids, holes = [], [], []
    for index, d in enumerate(defects):
        box = footprint(d, scene)
        if d['type'] in ('crack', 'delamination', 'contact'):
            axis = 0 if box[0][0] == box[0][1] else 2
            splits.append({'defect': index, 'axis': axis, 'position_mm': box[axis][0], 'ranges_mm': box, 'law': 'contact' if is_contact(d) else 'open'})
        elif d['type'] == 'notch':
            voids.append({'defect': index, 'box_mm': box})
        else:
            holes.append({'defect': index, 'axis': 2 if d['axis'] == 'z' else 1, 'centre_mm': [d['x_mm'], d.get('y_mm', W / 2), d.get('z_mm', T / 2)],
                          'radius_mm': d['radius_mm'], 'block_mm': box})
    return {'splits': splits, 'voids': voids, 'holes': holes}


def default_defect(kind, scene, existing=()):
    """A one-click defect for the plate in `scene`, placed clear of the existing ones (used by the interface and the text parser)."""
    L, W, T = plate_size(scene)
    base = {'crack': {'type': 'crack', 'x_mm': L / 2, 'depth_mm': max(.1, round(min(.5, .3 * T), 1)), 'from': 'top', 'closed': False},
            'delamination': {'type': 'delamination', 'x_start_mm': L / 2 - min(5., L / 8), 'x_end_mm': L / 2 + min(5., L / 8), 'depth_mm': round(T / 2, 3), 'closed': False},
            'contact': {'type': 'contact', 'orientation': 'vertical', 'x_mm': L / 2},
            'notch': {'type': 'notch', 'x_mm': L / 2, 'width_mm': round(min(1., L / 20), 3), 'depth_mm': max(.1, round(.3 * T, 1)), 'from': 'top'},
            'hole': {'type': 'hole', 'axis': 'z', 'x_mm': L / 2, 'y_mm': W / 2, 'radius_mm': round(min(.5, .3 * W), 3)}}[kind]
    d = copy.deepcopy(base)
    taken = [footprint(validate_defect(e, scene, i), scene) for i, e in enumerate(existing)]
    for shift in [0] + [s * k for k in range(1, 40) for s in (1, -1)]:
        trial = copy.deepcopy(d); dx = shift * max(2., L / 20)
        for key in ('x_mm', 'x_start_mm', 'x_end_mm'):
            if key in trial: trial[key] = round(trial[key] + dx, 3)
        try:
            box = footprint(validate_defect(trial, scene, len(existing)), scene)
        except ValueError:
            continue
        if not any(_overlap(box, other, GAP_MM) for other in taken): return trial
    raise ValueError(f'板上没有空位再放一个{LABELS[kind]}；请先调整已有缺陷或加长板。')
