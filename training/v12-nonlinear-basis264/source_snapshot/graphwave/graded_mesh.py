"""Boundary-graded Cartesian meshes for fixed-end plate research candidates.

Only the spatial discretization changes. Geometry, mass density, traction and
the conservative graph operator retain their physical definitions.
"""
import numpy as np

from spectral3d.mesh import gll, make_mesh


def _edges(length, count, ratio, symmetric, reverse):
    if count == 1 or ratio == 1:
        return np.linspace(0., length, count + 1)
    if symmetric:
        distance = abs(np.arange(count) - (count - 1) / 2)
        spread = distance.max() - distance.min()
        exponent = (distance.max() - distance) / spread if spread else distance * 0
    else:
        exponent = np.linspace(1., 0., count) if reverse else np.linspace(0., 1., count)
    widths = ratio ** exponent
    widths *= length / widths.sum()
    result = np.r_[0., np.cumsum(widths)]
    result[-1] = length
    if not np.all(np.diff(result) > 0):
        raise ValueError('Graded mesh contains a nonpositive element width')
    return result


def make_graded_mesh(scene, degree, h_mm, ratios=(1., 1., 1.), counts=None):
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios < 1) or np.any(ratios > 100):
        raise ValueError('Three finite grading ratios in [1, 100] required')
    if counts is None:
        mesh = make_mesh(scene, degree, h_mm)
        if np.all(ratios == 1):
            return mesh
        counts = [(len(np.unique(mesh['points'][:, axis])) - 1) // degree for axis in range(3)]
    else:
        if len(counts) != 3 or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError('Three positive integer element counts required')
        nodes, quadrature, derivative, barycentric = gll(degree)
        if np.prod([count * degree + 1 for count in counts]) > 100000 or np.prod(counts) * (degree + 1)**3 > 2000000:
            raise ValueError('Graded mesh exceeds the existing memory budget')
        index = np.arange(np.prod([count * degree + 1 for count in counts])).reshape(
            *[count * degree + 1 for count in counts])
        cells = np.array([index[i*degree:(i+1)*degree+1,
                                j*degree:(j+1)*degree+1,
                                k*degree:(k+1)*degree+1]
                          for i in range(counts[0]) for j in range(counts[1])
                          for k in range(counts[2])], dtype=np.int32)
        mesh = {'degree': degree, 'h_mm': h_mm, 'gll': nodes, 'weights_1d': quadrature,
                'derivative': derivative, 'barycentric': barycentric, 'elements': cells}
    if scene['geometry']['kind'] != 'plate' or scene['boundary']['fixed_face'] not in ['xmin', 'xmax']:
        raise ValueError('Boundary grading currently requires a plate fixed on an x face')
    geometry = scene['geometry']
    lengths = np.array([geometry[k] for k in ['length_mm', 'width_mm', 'thickness_mm']]) * 1e-3
    edges = [_edges(length, count, ratio, axis != 0, scene['boundary']['fixed_face'] == 'xmax')
             for axis, (length, count, ratio) in enumerate(zip(lengths, counts, ratios))]
    axes = [np.r_[np.concatenate([left + (mesh['gll'][:-1] + 1) * (right - left) / 2
                                 for left, right in zip(edge[:-1], edge[1:])]), edge[-1]]
            for edge in edges]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    if 'points' in mesh and points.shape != mesh['points'].shape:
        raise ValueError('Grading changed the declared mesh topology')
    cells = mesh['elements']
    origin = points[cells[:, 0, 0, 0]]
    size = points[cells[:, -1, -1, -1]] - origin
    quadrature = np.einsum('i,j,k->ijk', *([mesh['weights_1d']] * 3))
    volume = np.prod(size, axis=1)[:, None, None, None] / 8 * quadrature
    mass = np.zeros(len(points))
    np.add.at(mass, cells.ravel(), (volume * scene['material']['rho_kg_m3']).ravel())
    if not np.all(mass > 0) or not np.all(np.isfinite(mass)):
        raise ValueError('Graded mesh has a nonpositive or nonfinite nodal mass')
    mesh.update(points=points, origin=origin, size=size, volume_weight=volume, mass=mass,
                min_node_spacing=float(min(np.diff(axis).min() for axis in axes)),
                grading_ratios=ratios.tolist(), axis_element_counts=list(counts),
                axis_edges_m=[edge.tolist() for edge in edges])
    return mesh
