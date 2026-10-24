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


def face_load(mesh, source):
    """Nodal force (N) for one planar-face source; returns force, area and weighted area."""
    points = mesh['points']; force = np.zeros_like(points); axis, bound = face_bound(points, source['face']); tol = tolerance(points)
    direction = np.asarray(source['direction']) * source['amplitude_mpa'] * 1e6; area = 0.; weighted = 0.; n = mesh['degree'] + 1
    if str(mesh.get('mapping_kind', '')) == 'isoparametric':
        from .curved_mesh import boundary_faces
        for e, local_axis, end, ids in boundary_faces(mesh):
            if not np.all(np.abs(points[ids, axis] - bound) <= tol): continue
            selection = [slice(None)] * 3; selection[local_axis] = end; jac = mesh['jacobian'][e][tuple(selection)]; other = [i for i in range(3) if i != local_axis]
            quadrature = np.linalg.norm(np.cross(jac[..., other[0]], jac[..., other[1]]), axis=-1) * np.outer(mesh['weights_1d'], mesh['weights_1d'])
            weights = region_weights(points[ids.ravel()], source, axis).reshape(quadrature.shape)
            area += float(quadrature[weights != 0].sum()); weighted += float((quadrature * weights).sum())
            np.add.at(force, ids.ravel(), ((quadrature * weights)[..., None] * direction).reshape(-1, 3))
    else:
        for e, nodes in enumerate(mesh['elements']):
            loc = [slice(None)] * 3; loc[axis] = 0 if source['face'].endswith('min') else -1; ids = nodes[tuple(loc)]
            if not np.all(np.abs(points[ids, axis] - bound) <= tol): continue
            tangential = [q for q in range(3) if q != axis]
            quadrature = np.outer(mesh['weights_1d'], mesh['weights_1d']) * np.prod(mesh['size'][e, tangential]) / 4
            weights = region_weights(points[ids.ravel()], source, axis).reshape(n, n)
            area += float(quadrature[weights != 0].sum()); weighted += float((quadrature * weights).sum())
            np.add.at(force, ids.ravel(), ((quadrature * weights)[..., None] * direction).reshape(n * n, 3))
    require(area > 0, f'{source.get("name", "primary")} 的加载区域内没有面节点；请检查加载面、中心、半径或梳状参数。')
    return force, area, weighted


def prepare_excitation(mesh, scene):
    """Loads, constraints and damping for a version-2 scene; None for version 1."""
    if not scene_v2.is_v2(scene): return None
    scene_v2.spectral_admissible(scene); points = mesh['points']; material = scene['material']
    loads = []; rows = []
    for src in scene_v2.sources(scene):
        force, area, weighted = face_load(mesh, src)
        loads.append((force, src)); rows.append({'name': src.get('name', 'primary'), 'face': src['face'], 'loaded_area_mm2': area * 1e6, 'profile_weighted_area_mm2': weighted * 1e6})
    mask, constraint_rows = build_constraints(points, scene['boundary']['fixed_face'], scene_v2.boundaries(scene))
    total = sum(force for force, _ in loads)
    require(not np.any(total[mask] != 0), '激励作用在固定或对称约束的位移分量上；请改变加载方向或边界类型。')
    speed = math.sqrt((material['lambda_pa'] + 2 * material['mu_pa']) / material['rho_kg_m3'])
    alpha, damping_rows = build_damping(points, scene_v2.boundaries(scene), speed)
    return {'loads': loads, 'constrained': mask, 'alpha': alpha, 'area_m2': sum(r['loaded_area_mm2'] for r in rows) * 1e-6,
            'summary': {'excitation_model': 'multi_source_v2', 'sources': rows, 'constraints': constraint_rows, 'damping': damping_rows}}


def loaded_surface_faces(surface, loads):
    """Surface triangles whose three nodes all carry traction (for the viewer)."""
    total = np.linalg.norm(sum(force for force, _ in loads), axis=1) > 0
    return np.flatnonzero(np.all(total[np.asarray(surface)], axis=1)).tolist() if len(surface) else []
