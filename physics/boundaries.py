"""Node-based boundary conditions shared by the tetrahedral and spectral meshes.

* fixed face (version 1): all three components zero;
* symmetry plane: normal component zero, tangential free;
* antisymmetry plane: tangential components zero, normal free;
* absorbing layer (ALID): mass-proportional damping alpha(x) rising from 0 at
  the layer start to alpha_max at the face with a linear/quadratic/cubic ramp.

alpha_max is chosen from a design attenuation: a wave crossing the layer and
back sees exp(-alpha_max L / ((p + 1) c)) with ramp power p, so the default
60 dB gives alpha_max = (p + 1) c ln(10) dB / (20 L). The actual reflection
depends on frequency and layer length; solvers report the dissipated energy
and tests measure the residual echo. No PML, no split fields.
"""
import math
import numpy as np
from solid.schema import require

FACES = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
RAMP_POWER = {'linear': 1, 'quadratic': 2, 'cubic': 3}

def tolerance(points):
    return max(float(np.ptp(points, axis=0).max()) * 1e-7, 1e-10)

def face_bound(points, face):
    axis = 'xyz'.index(face[0])
    return axis, float(points[:, axis].min() if face.endswith('min') else points[:, axis].max())

def plane_nodes(points, face):
    axis, bound = face_bound(points, face)
    return np.flatnonzero(np.abs(points[:, axis] - bound) <= tolerance(points))

def build_constraints(points, fixed_face, boundaries):
    """Boolean (nodes, 3) mask of zero displacement components, plus a summary."""
    mask = np.zeros(points.shape, dtype=bool); rows = []
    if fixed_face != 'none':
        ids = plane_nodes(points, fixed_face); require(len(ids) >= 3, f'{fixed_face} 上没有可固定的平面节点。')
        mask[ids] = True; rows.append({'type': 'fixed', 'face': fixed_face, 'nodes': int(len(ids))})
    for b in boundaries:
        if b['type'] not in ('symmetry', 'antisymmetry'): continue
        ids = plane_nodes(points, b['face']); require(len(ids) >= 3, f'{b["face"]} 上没有节点，无法施加{b["type"]}条件。')
        axis = 'xyz'.index(b['face'][0])
        if b['type'] == 'symmetry': mask[ids, axis] = True
        else:
            for other in range(3):
                if other != axis: mask[ids, other] = True
        rows.append({'type': b['type'], 'face': b['face'], 'nodes': int(len(ids))})
    return mask, rows

def alpha_max_per_s(boundary, speed_m_s):
    if boundary.get('alpha_max_per_s') is not None: return float(boundary['alpha_max_per_s'])
    power = RAMP_POWER[boundary['ramp']]
    return (power + 1) * speed_m_s * boundary['attenuation_db'] * math.log(10) / (20 * boundary['length_mm'] * 1e-3)

def build_damping(points, boundaries, speed_m_s):
    """Per-node damping coefficient alpha (1/s) or None, plus a summary."""
    alpha = None; rows = []
    for b in boundaries:
        if b['type'] != 'absorbing': continue
        axis, bound = face_bound(points, b['face']); extent = float(np.ptp(points[:, axis])); length = b['length_mm'] * 1e-3
        require(length <= extent / 2 + tolerance(points), f'{b["face"]} 吸收层长度不能超过该方向尺寸的一半。')
        depth = np.clip((length - np.abs(points[:, axis] - bound)) / length, 0., 1.)
        inside = int(np.count_nonzero(depth > 0)); require(inside >= 2, f'{b["face"]} 吸收层内没有足够节点；请加长吸收层或细化网格。')
        peak = alpha_max_per_s(b, speed_m_s); layer = peak * depth ** RAMP_POWER[b['ramp']]
        alpha = layer if alpha is None else np.maximum(alpha, layer)
        rows.append({'type': 'absorbing', 'face': b['face'], 'length_mm': b['length_mm'], 'ramp': b['ramp'], 'alpha_max_per_s': peak,
                     'design_attenuation_db': b.get('attenuation_db'), 'nodes_in_layer': inside, 'reference_speed_m_s': speed_m_s})
    return alpha, rows
