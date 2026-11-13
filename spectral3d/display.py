"""Display mesh: the surface of a GLL mesh, reduced to what the result animation can draw.

The computation mesh may be far larger than a browser can animate. The animation only needs the surface, so
the display mesh keeps surface nodes only and thins them in two steps until it fits the display budget:

1. inside every surface element face, keep every `stride`-th GLL node (stride 1 keeps them all, stride = degree
   keeps the four corners);
2. if the corner mesh is still too large, merge the corner nodes that fall in the same cell of a uniform grid
   (vertex clustering) and drop the triangles that collapse.

The computation is never changed by this: receivers are recorded at full precision on the computation mesh.
"""
import numpy as np

TRIANGLE_BUDGET = 50000
NODE_BUDGET = 60000
FIELD_BUDGET_BYTES = 192 * 1024 ** 2
FRAMES = 281


def surface_faces(elements):
    """(F, n, n) node ids of the element faces that belong to one element only, in a deterministic order."""
    elements = np.asarray(elements)
    faces = np.concatenate([np.take(elements, end, axis=axis + 1) for axis in range(3) for end in (0, -1)])
    corners = np.sort(np.stack([faces[:, 0, 0], faces[:, -1, 0], faces[:, -1, -1], faces[:, 0, -1]], axis=1), axis=1)
    _, inverse, counts = np.unique(corners, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


def _triangles(faces, stride):
    picked = faces[:, ::stride, ::stride]
    a, b, c, d = picked[:, :-1, :-1], picked[:, 1:, :-1], picked[:, 1:, 1:], picked[:, :-1, 1:]
    return np.concatenate([np.stack([a, b, c], axis=-1).reshape(-1, 3), np.stack([a, c, d], axis=-1).reshape(-1, 3)])


def _cluster(points, triangles, cell):
    """Merge the nodes of each grid cell into its first node; drop collapsed and repeated triangles."""
    used = np.unique(triangles); keys = np.floor((points[used] - points[used].min(axis=0)) / cell + 1e-9).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    representative = np.zeros(int(triangles.max()) + 1, dtype=np.int64); representative[used] = used[first][inverse.ravel()]
    merged = representative[triangles]
    merged = merged[(merged[:, 0] != merged[:, 1]) & (merged[:, 1] != merged[:, 2]) & (merged[:, 0] != merged[:, 2])]
    _, keep = np.unique(np.sort(merged, axis=1), axis=0, return_index=True)
    return merged[np.sort(keep)]


def display_mesh(mesh, triangle_budget=TRIANGLE_BUDGET, node_budget=NODE_BUDGET):
    """{'nodes': computation node ids shown, 'triangles': (T, 3) indices into them, 'stride', 'cell_mm', 'surface_faces', 'full_surface_triangles'}."""
    degree = int(mesh['degree']); faces = surface_faces(mesh['elements']); points = np.asarray(mesh['points'])
    fits = lambda t: len(t) <= triangle_budget and len(np.unique(t)) <= node_budget
    stride = cell = None
    for stride in [s for s in range(1, degree + 1) if degree % s == 0]:
        triangles = _triangles(faces, stride)
        if fits(triangles): break
    else:
        corners = triangles; edge = points[corners[:, 1]] - points[corners[:, 0]]; cell = 1.05 * float(np.median(np.linalg.norm(edge, axis=1)))
        while True:
            triangles = _cluster(points, corners, cell)
            if fits(triangles): break
            cell *= 1.1
    nodes = np.unique(triangles)
    return {'nodes': nodes.astype(np.int64), 'triangles': np.searchsorted(nodes, triangles).astype(np.int32), 'stride': int(stride),
            'cell_mm': None if cell is None else cell * 1000, 'surface_faces': int(len(faces)), 'full_surface_triangles': int(len(faces)) * 2 * degree * degree}


def frames_shown(frame_count, nodes, bytes_per_value):
    """Animation frames: up to 281 evenly spaced samples, fewer when one response would exceed the display field budget."""
    shown = max(2, min(FRAMES, frame_count, FIELD_BUDGET_BYTES // (nodes * 3 * bytes_per_value)))
    return np.unique(np.linspace(0, frame_count - 1, shown, dtype=int))


def loaded_triangles(display, loads, level=1e-3):
    """Display triangles inside the loaded region: all three nodes carry at least `level` of the largest nodal traction.

    The threshold keeps the far tail of a Gaussian profile, which is nonzero over the whole face, from being drawn as loaded."""
    total = np.linalg.norm(sum(force for force, _ in loads), axis=1); total = total > level * total.max()
    return np.flatnonzero(np.all(total[display['nodes'][display['triangles']]], axis=1)).tolist()
