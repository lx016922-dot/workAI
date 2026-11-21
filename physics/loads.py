"""Spatial distribution of a surface traction over its loading region.

The region is the version-1 selection (whole planar face, or a disc of
`radius_mm` around `center_mm` on that face, or a `surface_patch`). The
optional `profile` weights that region: uniform, Gaussian, or a comb of
parallel teeth. Both discretizations evaluate the weights at their own
quadrature locations (GLL face nodes, triangle centroids).

Two profiles stand for transducers by the traction they put on the surface (no piezoelectric or wedge body is
modelled; what the transducer does to the plate is prescribed):

* `wedge` — an angle-beam (wedge) transducer: the traction of the source travels along the surface under a
  rectangular footprint, s(t − x/v) with the trace velocity v = wedge speed / sin(angle), which selects the guided
  mode whose phase velocity is v. A load whose delay varies from point to point is not a product of a shape and a
  signal; `expand` writes it as a row of overlapping strips (hat functions along the footprint), each with the signal
  delayed to its own position: linear interpolation of the delay, with at least `STRIPS_PER_WAVELENGTH` strips per
  trace wavelength.
* `patch` — a bonded piezoelectric wafer in the pin-force model: the wafer drives the plate by in-plane shear at its
  edges, outwards when it expands. A strip has two edges, a rectangle four, a disc a ring of radial traction.
  `amplitude_mpa` is the shear traction in the edge zone of width `edge_mm`; the line force is their product.

`expand(source)` turns either into sources with a plain signed weight and a fixed direction, which every solver can
integrate; for any other profile it returns the source itself.
"""
import numpy as np
from solid.schema import require, number, exact

PROFILES = ('uniform', 'gaussian', 'comb', 'wedge', 'patch')
INTERNAL = ('wedge_strip', 'patch_edges', 'patch_ring')   # what `expand` produces; never stored in a scene
STRIPS_PER_WAVELENGTH = 10     # of the trace wavelength v/f; fewer only when the footprint needs more than MAX_STRIPS
STRIPS_PER_WAVELENGTH_MIN = 6
MAX_STRIPS = 64
APODIZATIONS = ('uniform', 'hann')
PATCH_SHAPES = ('strip', 'rectangle', 'disc')

def validate_profile(profile, source):
    if profile is None: return None
    require(isinstance(profile, dict) and profile.get('type') in PROFILES, '载荷分布需为 uniform、gaussian、comb、wedge（斜楔）或 patch（压电片）。')
    p = dict(profile)
    if p['type'] in ('wedge', 'patch'): return _validate_transducer(p, source)
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


def _in_plane(source, what):
    """The two axes of the loaded face; transducer profiles need a coordinate face."""
    require(source['face'] != 'surface_patch' and source['face'][0] in 'xyz', f'{what}需要加在坐标平面的面上。')
    normal = 'xyz'.index(source['face'][0]); return [a for a in range(3) if a != normal], normal


def _validate_transducer(p, source):
    if p['type'] == 'wedge':
        exact(p, ['type', 'along', 'length_mm', 'width_mm', 'trace_velocity_m_s', 'reverse', 'apodization'], '斜楔激励')
        axes, normal = _in_plane(source, '斜楔激励'); require(p['along'] in ('x', 'y', 'z') and 'xyz'.index(p['along']) in axes, '斜楔的行进方向需在加载面内。')
        p['length_mm'] = number(p['length_mm'], .01, 1000, '斜楔底面长度'); p['width_mm'] = number(p['width_mm'], 0, 1000, '斜楔底面宽度（0 表示整个面宽）')
        p['trace_velocity_m_s'] = number(p['trace_velocity_m_s'], 100, 1e5, '表面迹线速度'); require(type(p['reverse']) is bool, 'reverse 需为布尔值。')
        require(p['apodization'] in APODIZATIONS, '斜楔的幅值分布需为 uniform 或 hann。')
        require(source['radius_mm'] == 0, '斜楔激励的范围由底面长度和宽度给出，radius_mm 需为 0。')
        strips(p, source['frequency_mhz'])   # refuses a footprint that cannot be resolved
    else:
        exact(p, ['type', 'shape', 'length_mm', 'width_mm', 'edge_mm'], '压电片激励'); require(p['shape'] in PATCH_SHAPES, '压电片形状需为 strip、rectangle 或 disc。')
        axes, normal = _in_plane(source, '压电片激励'); d = source['direction']
        require(abs(d[normal]) < 1e-9, '压电片的 direction 需在加载面内：它是片的长度方向（圆片不用，但也需在面内）。')
        require(p['shape'] == 'disc' or max(abs(d[a]) for a in axes) > 1 - 1e-9, '压电片的长度方向需沿加载面内的坐标轴。')
        p['length_mm'] = number(p['length_mm'], .01, 1000, '压电片长度（圆片为直径）'); p['width_mm'] = number(p['width_mm'], 0, 1000, '压电片宽度（条形片 0 表示整个面宽）')
        p['edge_mm'] = number(p['edge_mm'], .001, p['length_mm'] / 2, '边缘传力区宽度')
        require(p['shape'] != 'rectangle' or p['width_mm'] >= 2 * p['edge_mm'], '矩形压电片的宽度至少是两个边缘传力区。')
        require(source['radius_mm'] == 0, '压电片的范围由它的尺寸给出，radius_mm 需为 0。')
    return p


def strips(profile, frequency_mhz):
    """How many strips carry a wedge footprint: at least two, STRIPS_PER_WAVELENGTH per trace wavelength, at most MAX_STRIPS."""
    wavelength = profile['trace_velocity_m_s'] / (frequency_mhz * 1e6) * 1e3; n = int(np.ceil(profile['length_mm'] / (wavelength / STRIPS_PER_WAVELENGTH))) + 1
    if n > MAX_STRIPS:
        require(profile['length_mm'] / (MAX_STRIPS - 1) <= wavelength / STRIPS_PER_WAVELENGTH_MIN + 1e-12,
                f'斜楔底面有 {profile["length_mm"] / wavelength:.1f} 个迹线波长（{wavelength:.3g} mm），超过了能表示的 {(MAX_STRIPS - 1) / STRIPS_PER_WAVELENGTH_MIN:.1f} 个；请缩短底面。')
        n = MAX_STRIPS
    return max(n, 2)


def travel_us(source):
    """How much later than its `delay_us` the last part of a source starts (the far end of a wedge footprint); 0 for any other source."""
    p = source.get('profile') or {}
    return p['length_mm'] * 1e-3 / p['trace_velocity_m_s'] * 1e6 if p.get('type') == 'wedge' else 0.


def expand(source):
    """The plain sources a transducer profile stands for (see the module text); [source] for any other profile."""
    p = source.get('profile') or {}
    if p.get('type') == 'wedge':
        n = strips(p, source['frequency_mhz']); spacing = p['length_mm'] / (n - 1); out = []
        for i in range(n):
            position = i * spacing   # from the end where the wave enters the footprint
            out.append(dict(source, delay_us=source['delay_us'] + position * 1e-3 / p['trace_velocity_m_s'] * 1e6,
                            profile={'type': 'wedge_strip', 'along': p['along'], 'length_mm': p['length_mm'], 'width_mm': p['width_mm'], 'reverse': p['reverse'], 'apodization': p['apodization'],
                                     'index': i, 'count': n}))
        return out
    if p.get('type') == 'patch':
        axes, normal = _in_plane(source, '压电片激励'); d = np.asarray(source['direction'], dtype=float)
        if p['shape'] == 'disc':   # radial traction on a ring: its two in-plane components
            return [dict(source, direction=[float(k == a) for k in range(3)], profile=dict(p, type='patch_ring', component=a)) for a in axes]
        along = int(np.argmax(np.abs(d))); across = next(a for a in axes if a != along); unit = lambda a: [float(k == a) for k in range(3)]
        out = [dict(source, direction=unit(along), profile=dict(p, type='patch_edges', axis=along, half_mm=p['length_mm'] / 2, span_axis=across, span_mm=p['width_mm']))]
        if p['shape'] == 'rectangle':
            out.append(dict(source, direction=unit(across), profile=dict(p, type='patch_edges', axis=across, half_mm=p['width_mm'] / 2, span_axis=along, span_mm=p['length_mm'])))
        return out
    return [source]


def expand_all(sources):
    """Every source of a scene as plain sources, in order."""
    return [part for source in sources for part in expand(source)]

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
    require(profile['type'] not in ('wedge', 'patch'), '斜楔和压电片激励需要先由 physics.loads.expand 展开。')
    if profile['type'] == 'uniform': weight = np.ones(len(points))
    elif profile['type'] == 'gaussian': weight = np.exp(-.5 * (distance / (profile['sigma_mm'] * 1e-3)) ** 2)
    elif profile['type'] == 'wedge_strip':
        along = 'xyz'.index(profile['along']); length = profile['length_mm'] * 1e-3; n = profile['count']; spacing = length / (n - 1)
        s = delta[:, along] * (-1. if profile['reverse'] else 1.) + length / 2          # 0 where the wave enters the footprint, `length` where it leaves
        within = _indicator(np.minimum(s, length - s), edge)
        hat = np.clip(1 - np.abs(s - profile['index'] * spacing) / spacing, 0., 1.)        # linear interpolation of the delay between the strips
        taper = .5 * (1 - np.cos(2 * np.pi * np.clip(s / length, 0., 1.))) if profile['apodization'] == 'hann' else 1.
        weight = within * hat * taper * _span(delta, along, face_axis, profile['width_mm'], edge)
    elif profile['type'] == 'patch_edges':
        axis = profile['axis']; half = profile['half_mm'] * 1e-3; zone = profile['edge_mm'] * 1e-3
        # the edge zones lie inside the wafer: half − zone ≤ |s| ≤ half; the traction points outwards
        band = _indicator(np.minimum(np.abs(delta[:, axis]) - (half - zone), half - np.abs(delta[:, axis])), edge)
        across = _indicator(profile['span_mm'] * 1e-3 / 2 - np.abs(delta[:, profile['span_axis']]), edge) if profile['span_mm'] > 0 else 1.
        weight = band * np.sign(delta[:, axis]) * across
    elif profile['type'] == 'patch_ring':
        radius = profile['length_mm'] * 1e-3 / 2; zone = profile['edge_mm'] * 1e-3
        band = _indicator(np.minimum(distance - (radius - zone), radius - distance), edge)
        weight = band * delta[:, profile['component']] / np.maximum(distance, 1e-300)      # the component of the outward radial unit vector
    else:
        along = 'xyz'.index(profile['along']); pitch = profile['pitch_mm'] * 1e-3; teeth = profile['teeth']
        offset = points[:, along] - center[along] + (teeth - 1) * pitch / 2
        index = np.rint(offset / pitch); local = offset - index * pitch
        tooth = ((index >= 0) & (index < teeth)) * _indicator(profile['tooth_width_mm'] * 1e-3 / 2 - np.abs(local), edge)
        weight = tooth * (np.where(index % 2 == 0, 1., -1.) if profile['alternating_sign'] else 1.)
    return inside * weight

def separable(source, face_axis):
    """A plain source of `expand` as a product of two functions of the in-plane coordinates, for exact integration:
    {axis: (function of the coordinate relative to the centre in m, its break points in m)} for both in-plane axes, or
    None when the weight is not such a product (the ring of a disc wafer, every stored profile).

    The break points are where a function jumps or has a corner; between them it is smooth, so an element load can be
    integrated piece by piece (`spectral3d.loads`) wherever the element boundaries lie."""
    p = source.get('profile') or {}; axes = [a for a in range(3) if a != face_axis]; one = (lambda v: np.ones_like(v), [])
    box = lambda half: ((lambda v: (np.abs(v) <= half).astype(float)), [-half, half])
    if p.get('type') == 'wedge_strip':
        along = 'xyz'.index(p['along']); across = next(a for a in axes if a != along); length = p['length_mm'] * 1e-3; n = p['count']; spacing = length / (n - 1); sign = -1. if p['reverse'] else 1.
        def travelling(v):
            t = v * sign + length / 2; hat = np.clip(1 - np.abs(t - p['index'] * spacing) / spacing, 0., 1.)
            taper = .5 * (1 - np.cos(2 * np.pi * np.clip(t / length, 0., 1.))) if p['apodization'] == 'hann' else 1.
            return ((t >= 0) & (t <= length)) * hat * taper
        corners = [sign * ((p['index'] + j) * spacing - length / 2) for j in (-1, 0, 1)] + [-length / 2, length / 2]
        return {along: (travelling, corners), across: box(p['width_mm'] * 1e-3 / 2) if p['width_mm'] > 0 else one}
    if p.get('type') == 'patch_edges':
        half = p['half_mm'] * 1e-3; zone = p['edge_mm'] * 1e-3
        edges = lambda v: ((np.abs(v) >= half - zone) & (np.abs(v) <= half)) * np.sign(v)
        return {p['axis']: (edges, [-half, -(half - zone), half - zone, half]), p['span_axis']: box(p['span_mm'] * 1e-3 / 2) if p['span_mm'] > 0 else one}
    return None


def _span(delta, along, face_axis, width_mm, edge):
    """Indicator of the footprint across its travel direction (1 everywhere for width 0)."""
    if not width_mm > 0: return 1.
    across = next(a for a in range(3) if a not in (along, face_axis))
    return _indicator(width_mm * 1e-3 / 2 - np.abs(delta[:, across]), edge)


def _indicator(margin, edge):
    """1 inside (margin > edge), 0.5 on the edge, 0 outside."""
    return np.where(margin > edge, 1., np.where(margin >= -edge, .5, 0.))
