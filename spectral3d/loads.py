"""Multi-source traction assembly, constraints and damping for GLL meshes.

Version-1 scenes keep using the frozen `mesh.boundary_data`; this module is
only entered for version-2 scenes. It integrates the traction with the GLL
surface quadrature (Cartesian: face size; curved: surface Jacobian) and
multiplies each face node by the region/profile weight.
"""

import math
import numpy as np
from solid.schema import require
from physics import scene as scene_v2
from physics.loads import region_weights
from physics.boundaries import build_constraints, build_damping, face_bound, tolerance
from .mesh import boundary_data


GAUSS = np.polynomial.legendre.leggauss(8)


def basis_integrals(mesh, lo, hi, function, breaks, centre):
    """∫ N_i(x) f(x − centre) dx and ∫ |f| dx over the element edge [lo, hi] for the Lagrange basis on the GLL nodes of the mesh,
    integrated piece by piece between the break points of f (8 Gauss points a piece)."""
    nodes = np.asarray(mesh['gll'])
    bary = np.asarray(mesh['barycentric'])
    cuts = sorted({lo, hi, *(centre + b for b in breaks if lo < centre + b < hi)})
    out = np.zeros(len(nodes))
    measure = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        x = (a + b) / 2 + (b - a) / 2 * GAUSS[0]
        w = (b - a) / 2 * GAUSS[1]
        values = function(x - centre)
        if not np.any(values):
            continue
        xi = (2 * x - lo - hi) / (hi - lo)
        gap = xi[:, None] - nodes[None, :]
        on = np.abs(gap) < 1e-13  # a Gauss point on a GLL node: the basis is 1 there
        terms = bary[None, :] / np.where(on, 1.0, gap)
        shape = terms / terms.sum(1, keepdims=True)
        hit = on.any(1)
        shape[hit] = on[hit].astype(float)
        out += (w * values) @ shape
        measure += float(np.sum(w * np.abs(values)))
    return out, measure


def face_load(mesh, source):
    """Nodal force (N) for one planar-face source; returns force, |profile|-weighted area and signed profile-weighted area.

    A stored profile is integrated with the GLL face quadrature (the weight at the face nodes). The plain sources of a
    transducer profile whose weight is a product of two functions of the in-plane coordinates (`physics.loads.separable`:
    the strips of a wedge, the edges of a wafer) are integrated exactly against the element basis instead, so that the
    load does not depend on where the element boundaries fall inside a strip or an edge zone."""
    points = mesh['points']
    force = np.zeros_like(points)
    axis, bound = face_bound(points, source['face'])
    tol = tolerance(points)
    direction = np.asarray(source['direction']) * source['amplitude_mpa'] * 1e6
    area = 0.0
    weighted = 0.0
    n = mesh['degree'] + 1
    from physics.loads import separable

    factors = separable(source, axis) if str(mesh.get('mapping_kind', '')) != 'isoparametric' else None
    if factors is not None:
        centre = np.asarray(source['center_mm']) * 1e-3
        t0, t1 = [q for q in range(3) if q != axis]
        cache = {}

        def edge(q, lo, hi):  # elements in a row or a column share their edge
            key = (q, round(lo, 12), round(hi, 12))
            if key not in cache:
                cache[key] = basis_integrals(mesh, lo, hi, factors[q][0], factors[q][1], centre[q])
            return cache[key]

        for e, nodes in enumerate(mesh['elements']):
            loc = [slice(None)] * 3
            loc[axis] = 0 if source['face'].endswith('min') else -1
            ids = nodes[tuple(loc)]
            if not np.all(np.abs(points[ids, axis] - bound) <= tol):
                continue
            origin, size = mesh['origin'][e], mesh['size'][e]
            (a, ma), (b, mb) = (
                edge(t0, float(origin[t0]), float(origin[t0] + size[t0])),
                edge(t1, float(origin[t1]), float(origin[t1] + size[t1])),
            )
            if ma == 0 or mb == 0:
                continue
            nodal = np.outer(a, b)
            area += ma * mb
            weighted += float(nodal.sum())
            np.add.at(force, ids.ravel(), nodal.reshape(-1, 1) * direction)
        return force, area, weighted
    if str(mesh.get('mapping_kind', '')) == 'isoparametric':
        from .curved_mesh import boundary_faces

        for e, local_axis, end, ids in boundary_faces(mesh):
            if not np.all(np.abs(points[ids, axis] - bound) <= tol):
                continue
            selection = [slice(None)] * 3
            selection[local_axis] = end
            jac = mesh['jacobian'][e][tuple(selection)]
            other = [i for i in range(3) if i != local_axis]
            quadrature = np.linalg.norm(np.cross(jac[..., other[0]], jac[..., other[1]]), axis=-1) * np.outer(
                mesh['weights_1d'], mesh['weights_1d']
            )
            weights = region_weights(points[ids.ravel()], source, axis).reshape(quadrature.shape)
            area += float((quadrature * np.minimum(np.abs(weights), 1)).sum())
            weighted += float((quadrature * weights).sum())
            np.add.at(force, ids.ravel(), ((quadrature * weights)[..., None] * direction).reshape(-1, 3))
    else:
        for e, nodes in enumerate(mesh['elements']):
            loc = [slice(None)] * 3
            loc[axis] = 0 if source['face'].endswith('min') else -1
            ids = nodes[tuple(loc)]
            if not np.all(np.abs(points[ids, axis] - bound) <= tol):
                continue
            tangential = [q for q in range(3) if q != axis]
            quadrature = np.outer(mesh['weights_1d'], mesh['weights_1d']) * np.prod(mesh['size'][e, tangential]) / 4
            weights = region_weights(points[ids.ravel()], source, axis).reshape(n, n)
            area += float((quadrature * np.minimum(np.abs(weights), 1)).sum())
            weighted += float((quadrature * weights).sum())
            np.add.at(force, ids.ravel(), ((quadrature * weights)[..., None] * direction).reshape(n * n, 3))
    return force, area, weighted


MAX_PARTS = 128  # plain sources after the transducer profiles are expanded (a wedge footprint is a row of strips)


def source_loads(mesh, source, load=None):
    """[(nodal force, plain source)] of one source and its loaded and signed-weighted measure. A transducer profile is
    several plain sources (`physics.loads.expand`); a strip of a wedge footprint that holds no face node is left out
    (its neighbours carry those nodes), but the source as a whole must load something."""
    from physics.loads import expand

    parts = []
    area = 0.0
    weighted = 0.0
    for part in expand(source):
        force, a, w = (load or face_load)(mesh, part)
        if a > 0:
            parts.append((force, part))
            area += a
            weighted += w
    require(
        area > 0,
        f'{source.get("name", "primary")} 的加载区域内没有面节点；请检查加载面、中心、半径、梳状或换能器的尺寸（边缘传力区至少要盖住一排节点）。',
    )
    return parts, area, weighted


def prepare_excitation(mesh, scene):
    """Loads, constraints and damping for a version-2 scene; None for version 1."""
    if not scene_v2.is_v2(scene):
        return None
    scene_v2.spectral_admissible(scene)
    points = mesh['points']
    material = scene['material']
    loads = []
    rows = []
    for src in scene_v2.sources(scene):
        parts, area, weighted = source_loads(mesh, src)
        loads += parts
        rows.append(
            {
                'name': src.get('name', 'primary'),
                'face': src['face'],
                'loaded_area_mm2': area * 1e6,
                'profile_weighted_area_mm2': weighted * 1e6,
                **(
                    {'parts': len(parts), 'profile': src['profile']['type']}
                    if len(parts) > 1 or (src.get('profile') or {}).get('type') in ('wedge', 'patch')
                    else {}
                ),
            }
        )
    require(len(loads) <= MAX_PARTS, f'激励展开后有 {len(loads)} 个分量，超过上限 {MAX_PARTS}；请减少斜楔激励的个数或缩短底面。')
    mask, constraint_rows = build_constraints(points, scene['boundary']['fixed_face'], scene_v2.boundaries(scene))
    total = sum(force for force, _ in loads)
    require(not np.any(total[mask] != 0), '激励作用在固定或对称约束的位移分量上；请改变加载方向或边界类型。')
    from physics.materials import is_custom, fastest_speed

    speed = (
        fastest_speed(material)
        if is_custom(material)
        else math.sqrt((material['lambda_pa'] + 2 * material['mu_pa']) / material['rho_kg_m3'])
    )
    alpha, damping_rows = build_damping(points, scene_v2.boundaries(scene), speed)
    from physics.damping import coefficients

    mass_damping, stiffness_damping = coefficients(
        scene
    )  # Rayleigh damping of the material: α joins the nodal coefficient, β goes to the layer
    if mass_damping > 0:
        alpha = np.full(len(points), mass_damping) if alpha is None else alpha + mass_damping
    summary = {'excitation_model': 'multi_source_v2', 'sources': rows, 'constraints': constraint_rows, 'damping': damping_rows}
    if scene.get('damping'):
        summary['material_damping'] = dict(scene['damping'], model='rayleigh')
    return {
        'loads': loads,
        'constrained': mask,
        'alpha': alpha,
        'beta': stiffness_damping,
        'area_m2': sum(r['loaded_area_mm2'] for r in rows) * 1e-6,
        'summary': summary,
    }


def loaded_surface_faces(surface, loads):
    """Surface triangles whose three nodes all carry traction (for the viewer)."""
    total = np.linalg.norm(sum(force for force, _ in loads), axis=1) > 0
    return np.flatnonzero(np.all(total[np.asarray(surface)], axis=1)).tolist() if len(surface) else []
