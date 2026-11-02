"""Spatial distribution of a surface traction over its loading region.

The region is the version-1 selection (whole planar face, or a disc of
`radius_mm` around `center_mm` on that face, or a `surface_patch`). The
optional `profile` weights that region: uniform, Gaussian, or a comb of
parallel teeth. Both discretizations evaluate the weights at their own
quadrature locations (GLL face nodes, triangle centroids).
"""
import numpy as np
from solid.schema import require, number, exact

PROFILES = ('uniform', 'gaussian', 'comb')

def validate_profile(profile, source):
    if profile is None: return None
    require(isinstance(profile, dict) and profile.get('type') in PROFILES, '载荷分布需为 uniform、gaussian 或 comb。')
    p = dict(profile)
    if p['type'] == 'uniform': exact(p, ['type'], '载荷分布')
    elif p['type'] == 'gaussian':
        exact(p, ['type', 'sigma_mm'], '载荷分布'); p['sigma_mm'] = number(p['sigma_mm'], .01, 1000, '高斯宽度')
    else:
        exact(p, ['type', 'pitch_mm', 'teeth', 'tooth_width_mm', 'along', 'alternating_sign'], '载荷分布')
        p['pitch_mm'] = number(p['pitch_mm'], .01, 1000, '梳状间距'); require(type(p['teeth']) is int and 1 <= p['teeth'] <= 64, '梳状齿数需为 1–64 的整数。')
        p['tooth_width_mm'] = number(p['tooth_width_mm'], .001, p['pitch_mm'], '齿宽'); require(p['along'] in ('x', 'y', 'z'), '梳状方向需为 x、y 或 z。')
        require(type(p['alternating_sign']) is bool, 'alternating_sign 需为布尔值。')
        require(source['face'] == 'surface_patch' or p['along'] != source['face'][0], '梳状方向不能与加载面法向相同。')
    return p

def region_weights(points_m, source, face_axis):
    """Weight of each evaluation point: region indicator times profile value."""
    points = np.asarray(points_m, dtype=np.float64); center = np.asarray(source['center_mm']) * 1e-3
    delta = points - center
    if face_axis is not None: delta[:, face_axis] = 0.
    distance = np.linalg.norm(delta, axis=1); edge = 1e-9
    # Indicator integrated at nodes: points exactly on a region edge count half,
    # so a region whose edges fall on nodes integrates to its exact area.
    inside = _indicator(source['radius_mm'] * 1e-3 - distance, edge) if source['radius_mm'] > 0 else np.ones(len(points))
    profile = source.get('profile') or {'type': 'uniform'}
    if profile['type'] == 'uniform': weight = np.ones(len(points))
    elif profile['type'] == 'gaussian': weight = np.exp(-.5 * (distance / (profile['sigma_mm'] * 1e-3)) ** 2)
    else:
        along = 'xyz'.index(profile['along']); pitch = profile['pitch_mm'] * 1e-3; teeth = profile['teeth']
        offset = points[:, along] - center[along] + (teeth - 1) * pitch / 2
        index = np.rint(offset / pitch); local = offset - index * pitch
        tooth = ((index >= 0) & (index < teeth)) * _indicator(profile['tooth_width_mm'] * 1e-3 / 2 - np.abs(local), edge)
        weight = tooth * (np.where(index % 2 == 0, 1., -1.) if profile['alternating_sign'] else 1.)
    return inside * weight

def _indicator(margin, edge):
    """1 inside (margin > edge), 0.5 on the edge, 0 outside."""
    return np.where(margin > edge, 1., np.where(margin >= -edge, .5, 0.))
