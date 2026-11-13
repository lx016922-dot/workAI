"""GLL meshes for plates with defects: cracks, delaminations, contact interfaces, notches and holes.

The plate is a block-structured tensor grid whose grid lines pass through every defect edge (the same
construction as the frozen `spectral3d.mesh` notch plate, with more cut lines). On that grid

* a split surface (crack, delamination, contact interface) duplicates the nodes of the grid faces it covers;
  the elements on the + side of the plane get the copies, so the two faces move independently. Nodes on a
  front that lies inside the material stay shared (the crack closes at its tip); nodes on an edge that lies
  on an outer face or a symmetry plane are duplicated too (the crack opens there);
* a void box (notch) removes the elements inside it;
* a hole replaces the elements of a square patch around it with a curved O-grid: radial lines from every
  grid node on the patch boundary to the circle, graded towards the hole. The boundary nodes are the grid
  nodes themselves, so the patch conforms exactly; with a hole present the whole mesh is isoparametric.

For a contact interface the mesh also lists the node pairs (− side, + side), the plane normal (− to +) and the
GLL surface weight of each pair. Without defects `make_mesh` is the frozen mesh unchanged.
"""
import math
import numpy as np
from solid.schema import require
from .mesh import gll, make_mesh as frozen_make_mesh

AXES = 'xyz'
CUT_GAP_MM = .08


def has_defects(scene):
    return bool(scene.get('defects'))


# The frozen mesher stops at 100 000 nodes (its reference budget). Larger defect-free plates use `large_plate_mesh`,
# the same construction written with arrays; what a machine can actually run is decided by `offline.resources`.
FROZEN_NODES, FROZEN_ENTRIES = 100000, 2000000
LARGE_NODES = 8000000
DEFECT_NODES = 1000000


def plate_node_count(scene, degree, h_mm):
    """Nodes of the defect-free plate / notch grid before the notch void is removed (exact for a plate)."""
    g = scene['geometry']; limits = [[0, g['length_mm']], [0, g['width_mm']], [0, g['thickness_mm']]]
    if g['kind'] == 'notch':
        limits[0] = [0, g['notch_x_mm'] - g['notch_width_mm'] / 2, g['notch_x_mm'] + g['notch_width_mm'] / 2, g['length_mm']]
        limits[2] = [0, g['thickness_mm'] - g['notch_depth_mm'], g['thickness_mm']]
    counts = [sum(math.ceil((b - a) / h_mm) for a, b in zip(cuts[:-1], cuts[1:])) for cuts in limits]
    return int(np.prod([c * degree + 1 for c in counts])), int(np.prod(counts)) * (degree + 1) ** 3


def make_mesh(scene, degree=3, h_mm=1.6):
    """The frozen mesh for scenes without defects (its array form beyond the frozen size); the featured mesh otherwise."""
    if has_defects(scene): return featured_mesh(scene, degree, h_mm)
    if scene['geometry']['kind'] in ('plate', 'notch') and np.isfinite(h_mm) and h_mm > 0:
        nodes, entries = plate_node_count(scene, degree, h_mm)
        if nodes > FROZEN_NODES or entries > FROZEN_ENTRIES: return large_plate_mesh(scene, degree, h_mm)
    return frozen_make_mesh(scene, degree, h_mm)


def large_plate_mesh(scene, degree, h_mm):
    """`spectral3d.mesh.make_mesh` for plates and through-width notches, without its size limit and element loop.

    Same grid lines, node numbering, element order and arithmetic, so the result is identical to the frozen mesh
    wherever both exist (tested)."""
    g = scene['geometry']; kind = g['kind']
    require(kind in ('plate', 'notch'), '大规模网格目前只支持平板和贯穿缺口。')
    x, w, d, bary = gll(degree); p = degree
    limits = [[0, g['length_mm']], [0, g['width_mm']], [0, g['thickness_mm']]]
    if kind == 'notch':
        a = g['notch_x_mm'] - g['notch_width_mm'] / 2; b = g['notch_x_mm'] + g['notch_width_mm'] / 2; z = g['thickness_mm'] - g['notch_depth_mm']
        require(0 < a < b < g['length_mm'] and 0 < z < g['thickness_mm'], 'Invalid notch.')
        limits[0] = [0, a, b, g['length_mm']]; limits[2] = [0, z, g['thickness_mm']]
    nodes, _ = plate_node_count(scene, degree, h_mm)
    require(nodes <= LARGE_NODES, f'网格有 {nodes} 个节点，超过单次计算的上限 {LARGE_NODES}；请增大单元尺寸或缩小模型。')
    edges, axes = zip(*[_edges(cuts, h_mm, x) for cuts in limits])
    shape = tuple(len(axis) for axis in axes); counts = [len(edge) - 1 for edge in edges]
    I, J, K = (v.ravel() for v in np.meshgrid(*(np.arange(c) for c in counts), indexing='ij'))
    origin = np.stack([edges[0][I], edges[1][J], edges[2][K]], axis=1)
    size = np.stack([edges[0][I + 1], edges[1][J + 1], edges[2][K + 1]], axis=1) - origin
    if kind == 'notch':
        centre = (origin + size / 2) * 1000; inside = (a < centre[:, 0]) & (centre[:, 0] < b) & (centre[:, 2] > z)
        I, J, K, origin, size = I[~inside], J[~inside], K[~inside], origin[~inside], size[~inside]
    local = np.arange(p + 1)
    elements = (((I[:, None, None, None] * p + local[None, :, None, None]) * shape[1] + (J[:, None, None, None] * p + local[None, None, :, None])) * shape[2]
                + (K[:, None, None, None] * p + local[None, None, None, :]))
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    if kind == 'notch':
        used = np.unique(elements); remap = np.full(len(points), -1, dtype=np.int32); remap[used] = np.arange(len(used)); points = points[used]; elements = remap[elements]
    elements = np.ascontiguousarray(elements, dtype=np.int32)
    weights = np.einsum('i,j,k->ijk', w, w, w); volume_weight = np.prod(size, axis=1)[:, None, None, None] / 8 * weights
    mass = np.bincount(elements.ravel(), weights=(volume_weight * scene['material']['rho_kg_m3']).ravel(), minlength=len(points))
    require(not np.any(mass <= 0), 'Nonpositive mass.')
    return {'points': points, 'elements': elements, 'origin': origin, 'size': size, 'degree': degree,
            'gll': x, 'weights_1d': w, 'derivative': d, 'barycentric': bary, 'volume_weight': volume_weight,
            'mass': mass, 'min_node_spacing': float(min(np.min(np.diff(axis)) for axis in axes)), 'h_mm': h_mm}


def _edges(cuts_mm, h_mm, x):
    """Element edges (m) and GLL node coordinates (m) along one axis, as in the frozen mesh."""
    edge = np.concatenate([np.linspace(a, b, math.ceil((b - a) / h_mm) + 1)[:-1] for a, b in zip(cuts_mm[:-1], cuts_mm[1:])] + [[cuts_mm[-1]]]) * .001
    axis = np.r_[np.concatenate([a + (x[:-1] + 1) * (b - a) / 2 for a, b in zip(edge[:-1], edge[1:])]), edge[-1]]
    return edge, axis


def _edge_index(edge, value_mm):
    i = int(np.argmin(np.abs(edge * 1000 - value_mm)))
    require(abs(edge[i] * 1000 - value_mm) < 1e-6, f'网格内部错误：{value_mm} mm 不在网格线上。')
    return i


def featured_mesh(scene, degree, h_mm):
    from physics.defects import validate_defects, primitives, plate_size
    require(scene['geometry']['kind'] == 'plate', '缺陷目前只能加在平板上。')
    require(np.isfinite(h_mm) and h_mm > 0, 'Positive element size required.')
    defects = validate_defects(scene['defects'], scene); prims = primitives(defects, scene)
    L, W, T = plate_size(scene); size_mm = (L, W, T)
    x, w, d, bary = gll(degree); p = degree
    # Cut lines: plate faces, every defect edge, hole patches (at least two elements per patch side).
    cuts = [{0.: '板面', L: '板面'}, {0.: '板面', W: '板面'}, {0.: '板面', T: '板面'}]
    def add(axis, value, label):
        cuts[axis].setdefault(round(float(value), 9), label)
    for s in prims['splits'] + prims['voids']:
        label = defects[s['defect']]['name']
        for axis, (a, b) in enumerate(s.get('ranges_mm') or s['box_mm']): add(axis, a, label); add(axis, b, label)
    for hole in prims['holes']:
        label = defects[hole['defect']]['name']
        for axis, (a, b) in enumerate(hole['block_mm']):
            add(axis, a, label); add(axis, b, label)
            if axis != hole['axis'] and math.ceil((b - a) / h_mm) < 2: add(axis, (a + b) / 2, label)
    values = []
    for axis in range(3):
        row = sorted(cuts[axis])
        for a, b in zip(row[:-1], row[1:]):
            require(b - a >= CUT_GAP_MM, f'{cuts[axis][a]} 与 {cuts[axis][b]} 的边界在 {AXES[axis]} 方向只相距 {b - a:.3g} mm，'
                                         f'会产生极薄的单元层；请让它们对齐或至少相距 {CUT_GAP_MM} mm。')
        values.append(row)
    edges, axes = zip(*[_edges(row, h_mm, x) for row in values])
    index = np.arange(np.prod([len(a) for a in axes])).reshape(*(len(a) for a in axes))
    points = list(np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3))
    removed = [v['box_mm'] for v in prims['voids']] + [hole['block_mm'] for hole in prims['holes']]
    elements, cells, origins, sizes = [], [], [], []
    for i in range(len(edges[0]) - 1):
        for j in range(len(edges[1]) - 1):
            for k in range(len(edges[2]) - 1):
                origin = np.array([edges[0][i], edges[1][j], edges[2][k]]); size = np.array([edges[0][i + 1], edges[1][j + 1], edges[2][k + 1]]) - origin
                centre = (origin + size / 2) * 1000
                if any(all(a < c < b for c, (a, b) in zip(centre, box)) for box in removed): continue
                elements.append(index[i * p:(i + 1) * p + 1, j * p:(j + 1) * p + 1, k * p:(k + 1) * p + 1]); cells.append((i, j, k)); origins.append(origin); sizes.append(size)
    # Curved O-grid patches around holes.
    for hole in prims['holes']:
        _ogrid(hole, edges, axes, index, points, elements, cells, origins, sizes, x, p, h_mm)
    elements = [np.array(e, dtype=np.int64) for e in elements]
    # Split surfaces: duplicate the nodes, + side elements take the copies.
    pairs_a, pairs_b, normals, areas, split_rows = [], [], [], [], []
    cell_lookup = {c: n for n, c in enumerate(cells) if c is not None}
    for s in prims['splits']:
        a = s['axis']; others = [q for q in range(3) if q != a]; ia = _edge_index(edges[a], s['position_mm']); na = ia * p
        ranges = {q: (_edge_index(edges[q], s['ranges_mm'][q][0]), _edge_index(edges[q], s['ranges_mm'][q][1])) for q in others}
        opened = {q: (s['ranges_mm'][q][0] <= 1e-9, s['ranges_mm'][q][1] >= size_mm[q] - 1e-9) for q in others}
        def duplicable(q, n):
            lo, hi = ranges[q][0] * p, ranges[q][1] * p
            return lo < n < hi or (n == lo and opened[q][0]) or (n == hi and opened[q][1])
        area = {}
        b, c = others
        for jb in range(*ranges[b]):
            for jc in range(*ranges[c]):
                face_weight = np.outer(w, w) * (edges[b][jb + 1] - edges[b][jb]) * (edges[c][jc + 1] - edges[c][jc]) / 4
                for u in range(p + 1):
                    for v in range(p + 1):
                        loc = [0, 0, 0]; loc[a] = na; loc[b] = jb * p + u; loc[c] = jc * p + v
                        node = int(index[tuple(loc)]); area[node] = area.get(node, 0.) + face_weight[u, v]
        copies = {}
        for node in sorted(area):
            loc = np.unravel_index(node, index.shape)
            if duplicable(b, loc[b]) and duplicable(c, loc[c]):
                copies[node] = len(points); points.append(points[node].copy())
        require(copies, f'{defects[s["defect"]]["name"]} 没有覆盖任何网格面。')
        for jb in range(*ranges[b]):
            for jc in range(*ranges[c]):
                cell = [0, 0, 0]; cell[a] = ia; cell[b] = jb; cell[c] = jc
                n = cell_lookup.get(tuple(cell)); require(n is not None, f'{defects[s["defect"]]["name"]} 与其他缺陷的单元冲突。')
                e = elements[n]
                for old, new in copies.items(): e[e == old] = new
        if s['law'] == 'contact':
            normal = np.zeros(3); normal[a] = 1.
            for old, new in copies.items():
                pairs_a.append(old); pairs_b.append(new); normals.append(normal); areas.append(area[old])
        split_rows.append({'defect': s['defect'], 'axis': AXES[a], 'nodes_split': len(copies), 'law': s['law']})
    points = np.asarray(points); elements = np.asarray(elements, dtype=np.int64)
    used = np.unique(elements); remap = np.full(len(points), -1, dtype=np.int64); remap[used] = np.arange(len(used))
    points = points[used]; elements = remap[elements].astype(np.int32)
    contact = {'contact_a': remap[np.asarray(pairs_a, dtype=np.int64)].astype(np.int64) if pairs_a else np.zeros(0, dtype=np.int64),
               'contact_b': remap[np.asarray(pairs_b, dtype=np.int64)].astype(np.int64) if pairs_b else np.zeros(0, dtype=np.int64),
               'contact_normal': np.asarray(normals, dtype=np.float64).reshape(-1, 3), 'contact_area': np.asarray(areas, dtype=np.float64)}
    require(len(points) <= DEFECT_NODES, f'含缺陷网格有 {len(points)} 个节点，超过上限 {DEFECT_NODES}；请增大单元尺寸或缩小模型。')
    rho = scene['material']['rho_kg_m3']
    if prims['holes']:
        mesh = prepare_curved(points, elements, degree, rho, h_mm)
    else:
        sizes = np.array(sizes); weights = np.einsum('i,j,k->ijk', w, w, w)
        volume_weight = np.prod(sizes, axis=1)[:, None, None, None] / 8 * weights
        mass = np.zeros(len(points)); np.add.at(mass, elements.ravel(), (volume_weight * rho).ravel())
        require(not np.any(mass <= 0), 'Nonpositive mass.')
        mesh = {'points': points, 'elements': elements, 'origin': np.array(origins), 'size': sizes, 'degree': degree,
                'gll': x, 'weights_1d': w, 'derivative': d, 'barycentric': bary, 'volume_weight': volume_weight,
                'mass': mass, 'min_node_spacing': float(min(np.min(np.diff(a)) for a in axes)), 'h_mm': h_mm}
    mesh.update(contact)
    mesh['split_nodes'] = np.array([r['nodes_split'] for r in split_rows], dtype=np.int64)
    return mesh


def prepare_curved(points, elements, degree, rho, h_mm):
    """`spectral3d.curved_mesh.prepare` (isoparametric geometry of a mesh with curved elements) without its 100 000-node limit.

    The same operations in the same order, so the result is identical to the frozen function wherever both exist (tested)."""
    x, w, d, bary = gll(degree); points = np.asarray(points); elements = np.asarray(elements, dtype=np.int32); local = points[elements]
    jac = np.stack([np.einsum('ia,eajkc->eijkc', d, local), np.einsum('ja,eiakc->eijkc', d, local), np.einsum('ka,eijac->eijkc', d, local)], axis=-1)
    determinant = np.linalg.det(jac)
    require(not np.any(determinant <= 1e-20), 'Nonpositive or degenerate curved Jacobian.')
    inverse = np.linalg.inv(jac); volume = determinant * np.einsum('i,j,k->ijk', w, w, w); mass = np.zeros(len(points)); np.add.at(mass, elements.ravel(), (rho * volume).ravel())
    spacing = float(np.linalg.svd(jac, compute_uv=False)[..., -1].min() * np.diff(x).min())
    require(spacing >= 1e-7, 'Curved mesh quality exceeds the reference budget.')
    return {'points': points, 'elements': elements, 'degree': degree, 'gll': x, 'weights_1d': w, 'derivative': d, 'barycentric': bary, 'volume_weight': volume, 'mass': mass,
            'jacobian': jac, 'inverse_jacobian': inverse, 'min_node_spacing': spacing, 'h_mm': h_mm, 'mapping_kind': 'isoparametric',
            'bounds_min': local.min(axis=(1, 2, 3)), 'bounds_max': local.max(axis=(1, 2, 3))}


def _ogrid(hole, edges, axes, index, points, elements, cells, origins, sizes, x, p, h_mm):
    """Append the O-grid elements of one hole; nodes on the patch boundary are the grid's own nodes."""
    a = hole['axis']; u = 0; v = 2 if a == 1 else 1
    (u0, u1), (v0, v1) = hole['block_mm'][u], hole['block_mm'][v]
    iu0, iu1 = _edge_index(edges[u], u0) * p, _edge_index(edges[u], u1) * p
    iv0, iv1 = _edge_index(edges[v], v0) * p, _edge_index(edges[v], v1) * p
    walk = ([(n, iv0) for n in range(iu0, iu1)] + [(iu1, n) for n in range(iv0, iv1)] +
            [(n, iv1) for n in range(iu1, iu0, -1)] + [(iu0, n) for n in range(iv1, iv0, -1)])
    Q = len(walk); require(Q % p == 0 and Q >= 4 * p, '孔周围网格分段无效。')
    centre = np.array([hole['centre_mm'][u], hole['centre_mm'][v]]) * .001; radius = hole['radius_mm'] * .001
    outer = np.array([[axes[u][nu], axes[v][nv]] for nu, nv in walk])
    # Circle nodes: segment end points at the angles of the grid nodes, GLL-spaced in angle in between.
    theta = np.unwrap(np.arctan2(outer[:, 1] - centre[1], outer[:, 0] - centre[0]))
    vertices = np.r_[theta[::p], theta[0] + math.copysign(2 * np.pi, theta[-1] - theta[0])]
    theta = np.concatenate([a + (x[:-1] + 1) * (b - a) / 2 for a, b in zip(vertices[:-1], vertices[1:])])
    inner = centre + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    reach = float(np.max(np.linalg.norm(outer - centre, axis=1)) - radius)
    layers = max(1, math.ceil(reach / h_mm * 1000)); s_edges = np.linspace(0, 1, layers + 1)
    s = np.r_[np.concatenate([lo + (x[:-1] + 1) * (hi - lo) / 2 for lo, hi in zip(s_edges[:-1], s_edges[1:])]), 1.]
    R = len(s) - 1; ids = {}
    def node(qi, ri, na):
        qi %= Q
        if ri == R:
            loc = [0, 0, 0]; loc[u] = walk[qi][0]; loc[v] = walk[qi][1]; loc[a] = na; return int(index[tuple(loc)])
        key = (qi, ri, na)
        if key not in ids:
            pos = inner[qi] + s[ri] * (outer[qi] - inner[qi]); xyz = np.zeros(3); xyz[u] = pos[0]; xyz[v] = pos[1]; xyz[a] = axes[a][na]
            ids[key] = len(points); points.append(xyz)
        return ids[key]
    flip = None
    for layer in range(layers):
        for segment in range(Q // p):
            for ka in range(len(edges[a]) - 1):
                element = np.empty((p + 1,) * 3, dtype=np.int64)
                for i in range(p + 1):
                    for j in range(p + 1):
                        for k in range(p + 1): element[i, j, k] = node(segment * p + j, layer * p + i, ka * p + k)
                if flip is None:
                    corner = np.asarray(points)[[element[0, 0, 0], element[-1, 0, 0], element[0, -1, 0], element[0, 0, -1]]]
                    flip = np.linalg.det(np.stack([corner[1] - corner[0], corner[2] - corner[0], corner[3] - corner[0]])) < 0
                elements.append(element[:, ::-1, :] if flip else element); cells.append(None); origins.append(None); sizes.append(None)
