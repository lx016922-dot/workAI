"""Conforming Cartesian GLL quadrilaterals for the x–z section: plate and through-width notch."""
import math
import numpy as np
from spectral3d.mesh import gll


def make_mesh(scene, degree=4, h_mm=1.):
    g = scene['geometry']; kind = g['kind']
    if kind not in ('plate', 'notch'): raise ValueError('二维平面应变目前支持板与贯穿宽度的缺口。')
    if not np.isfinite(h_mm) or h_mm <= 0: raise ValueError('Positive element size required.')
    x, w, d, bary = gll(degree)
    limits = [[0, g['length_mm']], [0, g['thickness_mm']]]
    if kind == 'notch':
        a = g['notch_x_mm'] - g['notch_width_mm'] / 2; b = g['notch_x_mm'] + g['notch_width_mm'] / 2; top = g['thickness_mm'] - g['notch_depth_mm']
        if not 0 < a < b < g['length_mm'] or not 0 < top < g['thickness_mm']: raise ValueError('Invalid notch.')
        limits = [[0, a, b, g['length_mm']], [0, top, g['thickness_mm']]]
    axes = []; edges = []
    for cuts in limits:
        edge = np.concatenate([np.linspace(p, q, math.ceil((q - p) / h_mm) + 1)[:-1] for p, q in zip(cuts[:-1], cuts[1:])] + [[cuts[-1]]]) * .001
        axis = np.r_[np.concatenate([p + (x[:-1] + 1) * (q - p) / 2 for p, q in zip(edge[:-1], edge[1:])]), edge[-1]]
        edges.append(edge); axes.append(axis)
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    index = np.arange(len(points)).reshape(len(axes[0]), len(axes[1])); elements = []; origins = []; sizes = []
    for i in range(len(edges[0]) - 1):
        for k in range(len(edges[1]) - 1):
            origin = np.array([edges[0][i], edges[1][k]]); size = np.array([edges[0][i + 1], edges[1][k + 1]]) - origin; center = (origin + size / 2) * 1000
            if kind == 'notch' and a < center[0] < b and center[1] > top: continue
            elements.append(index[i * degree:(i + 1) * degree + 1, k * degree:(k + 1) * degree + 1]); origins.append(origin); sizes.append(size)
    elements = np.asarray(elements, dtype=np.int32); used = np.unique(elements); remap = np.full(len(points), -1, dtype=np.int32); remap[used] = np.arange(len(used))
    points = points[used]; elements = remap[elements]
    if len(points) > 400000 or elements.size > 4000000: raise ValueError('Plane-strain mesh budget exceeded.')
    sizes = np.array(sizes); area_weight = np.prod(sizes, axis=1)[:, None, None] / 4 * np.outer(w, w)
    mass = np.zeros(len(points)); np.add.at(mass, elements.ravel(), (area_weight * scene['material']['rho_kg_m3']).ravel())
    if np.any(mass <= 0): raise ValueError('Nonpositive mass.')
    return {'points': points, 'elements': elements, 'origin': np.array(origins), 'size': sizes, 'degree': degree, 'gll': x, 'weights_1d': w,
            'derivative': d, 'barycentric': bary, 'area_weight': area_weight, 'mass': mass,
            'min_node_spacing': float(min(np.min(np.diff(a)) for a in axes)), 'h_mm': h_mm}


def points3d(mesh):
    """Section nodes lifted to (x, 0, z) so the shared boundary helpers apply."""
    p = mesh['points']; return np.stack([p[:, 0], np.zeros(len(p)), p[:, 1]], axis=1)


def interpolation(mesh, point_mm):
    """Tensor Lagrange weights of a point (x, y, z) in millimetres; y is ignored."""
    point = np.asarray(point_mm, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all(): raise ValueError('Invalid receiver.')
    section = point[[0, 2]] * .001
    inside = np.flatnonzero(np.all(section >= mesh['origin'] - 1e-13, axis=1) & np.all(section <= mesh['origin'] + mesh['size'] + 1e-13, axis=1))
    if not len(inside): raise ValueError('Receiver is outside the section, including the notch void.')
    e = int(inside[0]); local = 2 * (section - mesh['origin'][e]) / mesh['size'][e] - 1; rows = []
    for value in local:
        delta = value - mesh['gll']; hit = np.flatnonzero(abs(delta) < 1e-12)
        if len(hit): row = np.eye(mesh['degree'] + 1)[hit[0]]
        else: row = mesh['barycentric'] / delta; row /= row.sum()
        rows.append(row)
    return mesh['elements'][e].ravel(), np.outer(rows[0], rows[1]).ravel()


def boundary_segments(mesh):
    """Element edges on the outer boundary: (element, local axis, end, node ids along the edge)."""
    edges = {}
    for e, element in enumerate(mesh['elements']):
        for axis in range(2):
            for end in (0, -1):
                ids = element[end, :] if axis == 0 else element[:, end]
                key = tuple(sorted((int(ids[0]), int(ids[-1]))))
                edges[key] = None if key in edges else (e, axis, end, ids)
    return [v for v in edges.values() if v is not None]


def surface_lines(mesh):
    """Boundary edge segments between consecutive GLL nodes (for drawing)."""
    lines = []
    for _, _, _, ids in boundary_segments(mesh):
        lines.extend([[int(ids[i]), int(ids[i + 1])] for i in range(len(ids) - 1)])
    return lines
