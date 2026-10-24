"""Traction, constraint and damping assembly on the plane-strain GLL mesh (per unit width)."""
import math
import numpy as np
from solid.schema import require
from physics import scene as scene_v2
from physics.loads import region_weights
from physics.boundaries import build_constraints, build_damping, face_bound, tolerance
from .mesh import points3d


def face_load(mesh, source):
    """Nodal force per unit width (N/m) for a source on xmin/xmax/zmin/zmax; returns force, length, weighted length."""
    lifted = points3d(mesh); axis3 = 'xyz'.index(source['face'][0]); axis2 = 0 if axis3 == 0 else 1
    _, bound = face_bound(lifted, source['face']); tol = tolerance(lifted)
    direction = np.asarray(source['direction'])[[0, 2]] * source['amplitude_mpa'] * 1e6
    force = np.zeros_like(mesh['points']); length = 0.; weighted = 0.; end = 0 if source['face'].endswith('min') else -1
    probe = lifted.copy(); probe[:, 1] = source['center_mm'][1] * 1e-3   # the section has no y extent
    for e, element in enumerate(mesh['elements']):
        ids = element[end, :] if axis2 == 0 else element[:, end]
        if not np.all(np.abs(lifted[ids, axis3] - bound) <= tol): continue
        quadrature = mesh['weights_1d'] * mesh['size'][e, 1 - axis2] / 2
        weights = region_weights(probe[ids], source, axis3)
        length += float((quadrature * np.minimum(np.abs(weights), 1)).sum()); weighted += float((quadrature * weights).sum())
        np.add.at(force, ids, (quadrature * weights)[:, None] * direction)
    require(length > 0, f'{source.get("name", "primary")} 的加载边上没有节点；请检查加载面、中心、半径或梳状参数。')
    return force, length, weighted


def prepare_excitation(mesh, scene):
    require(scene_v2.is_v2(scene) and scene_v2.is_plane_strain(scene), '二维求解器需要 dimension 为 2d_plane_strain 的 v2 场景。')
    lifted = points3d(mesh); material = scene['material']; loads = []; rows = []
    for src in scene_v2.sources(scene):
        force, length, weighted = face_load(mesh, src); loads.append((force, src))
        rows.append({'name': src.get('name', 'primary'), 'face': src['face'], 'loaded_length_mm': length * 1e3, 'profile_weighted_length_mm': weighted * 1e3})
    mask3, constraint_rows = build_constraints(lifted, scene['boundary']['fixed_face'], scene_v2.boundaries(scene)); mask = mask3[:, [0, 2]]
    total = sum(force for force, _ in loads)
    require(not np.any(total[mask] != 0), '激励作用在固定或对称约束的位移分量上；请改变加载方向或边界类型。')
    speed = math.sqrt((material['lambda_pa'] + 2 * material['mu_pa']) / material['rho_kg_m3'])
    alpha, damping_rows = build_damping(lifted, scene_v2.boundaries(scene), speed)
    return {'loads': loads, 'constrained': mask, 'alpha': alpha, 'length_m': sum(r['loaded_length_mm'] for r in rows) * 1e-3,
            'summary': {'excitation_model': 'multi_source_v2', 'dimension': '2d_plane_strain', 'sources': rows, 'constraints': constraint_rows, 'damping': damping_rows}}


def receiver_points(scene):
    """Named receiver points: scene receivers followed by the expanded receiver lines."""
    rows = [dict(rx, group=None, index=None, distance_mm=None) for rx in scene['receivers']]
    for line in scene_v2.receiver_lines(scene):
        start = np.asarray(line['from_mm']); end = np.asarray(line['to_mm'])
        for i in range(line['count']):
            point = start + (end - start) * i / (line['count'] - 1)
            rows.append({'name': f"{line['name']}[{i}]", 'point_mm': point.tolist(), 'direction': line['direction'], 'group': line['name'], 'index': i,
                         'distance_mm': float(np.linalg.norm(point - start))})
    return rows
