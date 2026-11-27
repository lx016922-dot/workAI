"""Wave-field record: the displacement of a whole surface (or of every node) at every kept sample.

A compact record keeps the complete waveforms of the receivers only. A study of a wave field — a continuous wave in a
plate with a defect, read as a scanning laser vibrometer reads it — needs the waveform at every point of a surface;
the complete record of every node of such a model is far too large (a 200 × 200 × 2 mm plate for 1 ms: over 100 GiB a
control). The wave-field record is the middle: the nodes of one face (or all nodes), the chosen components, every
n-th sample, float32.

    request['wavefield'] = {'region': 'top' | 'bottom' | 'all', 'components': 'z' | 'xyz', 'every': n}

`graphnet.rollout` stores it in the chunks of the compact record (`wavefield`, `wavefield_frames`) with the node list
in `wavefield.npz`; this module chooses the nodes, estimates the size, reads the record back and exports it.
"""

from pathlib import Path

import numpy as np

REGIONS = {'top': '上表面（z 最大）', 'bottom': '下表面（z = 0）', 'all': '全部节点'}
COMPONENTS = {'z': [2], 'xyz': [0, 1, 2]}
COMPONENT_LABELS = {'z': '只存 Z 位移（离面）', 'xyz': '三个分量'}
AXES = 'xyz'
EVERY_MAX = 1000
FACE_TOLERANCE_MM = 1e-6
MAT_LIMIT = 2 * 1024**3 - 1024  # scipy writes MATLAB 5 files: one variable stays under 2 GiB


def validate(value):
    """The normalized setting, or None for no wave-field record."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {'region', 'components', 'every'}:
        raise ValueError('波场记录需为 {region, components, every}。')
    region = value.get('region', 'top')
    components = value.get('components', 'z')
    every = value.get('every', 1)
    if region not in REGIONS:
        raise ValueError('波场记录的范围需为 top（上表面）、bottom（下表面）或 all（全部节点）。')
    if components not in COMPONENTS:
        raise ValueError('波场记录的分量需为 z（只存离面位移）或 xyz（三个分量）。')
    if type(every) is not int or not 1 <= every <= EVERY_MAX:
        raise ValueError(f'波场记录的时间抽稀需为 1–{EVERY_MAX} 的整数（每几个采样点存一次）。')
    return {'region': region, 'components': components, 'every': every}


def check_scene(spec, scene):
    """A face of a plate is a face of the model; a pipe has none."""
    if spec and spec['region'] != 'all' and scene['geometry']['kind'] == 'pipe':
        raise ValueError('圆管没有上、下表面：波场记录的范围请选“全部节点”。')


def node_ids(points_m, region):
    """Sorted ids of the nodes of a region. `points_m`: the mesh points (m). Nodes split by a defect are all kept."""
    points = np.asarray(points_m)
    if region == 'all':
        return np.arange(len(points), dtype=np.int64)
    z = points[:, 2] * 1e3
    level = z.max() if region == 'top' else z.min()
    return np.flatnonzero(np.abs(z - level) <= FACE_TOLERANCE_MM).astype(np.int64)


def for_rollout(spec, mesh):
    """The setting as `graphnet.rollout.rollout(..., wavefield=...)` takes it."""
    return {
        'nodes': node_ids(mesh['points'], spec['region']),
        'components': COMPONENTS[spec['components']],
        'every': spec['every'],
        'region': spec['region'],
    }


def frames_kept(frames, every):
    return len(range(0, frames, every))


def size_bytes(frames, nodes, spec, parts):
    """Bytes of the record on disk: kept samples × nodes × components × 4, for every stored part."""
    return frames_kept(frames, spec['every']) * nodes * len(COMPONENTS[spec['components']]) * 4 * parts


def face_nodes_estimate(nodes, points_m=None, region='top'):
    """Nodes of a region; from the mesh when it is there."""
    return len(node_ids(points_m, region)) if points_m is not None else nodes


def describe(spec, nodes, frames, sample_step_us, parts, size):
    """One line for the messages before a computation."""
    kept = frames_kept(frames, spec['every'])
    return (
        f'波场记录：{REGIONS[spec["region"]]}的 {nodes:,} 个节点，{COMPONENT_LABELS[spec["components"]]}，每 {spec["every"]} 个采样点存一次'
        f'（{kept:,} 个时刻，间隔 {sample_step_us * spec["every"]:g} μs），float32，{parts} 种响应共约 '
        + (f'{size / 1024**3:.2f} GiB' if size >= 0.1 * 1024**3 else f'{size / 1024**2:.1f} MiB')
    )


# ---------- reading a finished record ----------


def info(record):
    """The wave-field entry of a record's manifest with its nodes: {'points_mm', 'nodes', ...}, or None."""
    from .recording import manifest

    m = manifest(record)
    entry = m.get('wavefield')
    if not entry:
        return None
    with np.load(Path(record) / entry['file'], allow_pickle=False) as data:
        return dict(
            entry,
            node_ids=data['nodes'],
            points_mm=data['points_mm'],
            frame_count=m['frame_count'],
            record_sample_step_us=m['sample_step_us'],
        )


def grid(points_mm):
    """A face whose nodes form a full tensor grid: (x mm, y mm, index[ny, nx] into the nodes); None when they do not (a hole,
    nodes split by a surface-breaking defect, all nodes of the volume)."""
    points = np.asarray(points_mm)
    if np.ptp(points[:, 2]) > FACE_TOLERANCE_MM:
        return None
    x = np.unique(np.round(points[:, 0], 9))
    y = np.unique(np.round(points[:, 1], 9))
    if len(x) * len(y) != len(points):
        return None
    ix = np.searchsorted(x, np.round(points[:, 0], 9))
    iy = np.searchsorted(y, np.round(points[:, 1], 9))
    index = np.full((len(y), len(x)), -1, dtype=np.int64)
    index[iy, ix] = np.arange(len(points))
    return (x, y, index) if (index >= 0).all() else None


# A record stores the linear response and the increments (formed in float64, so the small increments keep their digits); the
# other responses are their sums.
SUMS = {
    'full': (('linear', 'geometric', 'material'), ('linear', 'nonlinear')),
    'stvk': (('linear', 'geometric'),),
    'nonlinear': (('geometric', 'material'),),
}


def stored_for(parts, part):
    """The stored parts whose sum is a response, or None when the record does not hold it."""
    if part in parts:
        return (part,)
    return next((names for names in SUMS.get(part, ()) if set(names) <= set(parts)), None)


def responses(parts):
    """The responses a record with these stored parts can give, in the order the result page lists them."""
    return [name for name in ('full', 'linear', 'stvk', 'nonlinear', 'geometric', 'material') if stored_for(parts, name)]


def read(record, part, nodes=None):
    """(times μs, values[kept samples, nodes, components] float32) of one response; `nodes`: indices into the record's nodes."""
    from .recording import chunks, manifest

    m = manifest(record)
    entry = m.get('wavefield')
    if not entry:
        raise ValueError('这次计算没有保存波场记录。请在“求解器配置”里打开“波场记录”后重新计算。')
    names = stored_for(entry['parts'], part)
    if not names:
        raise ValueError(f'波场记录里没有“{part}”这种响应；可以给出：' + '、'.join(responses(entry['parts'])) + '。')
    slots = [entry['parts'].index(name) for name in names]
    count = entry['nodes'] if nodes is None else len(nodes)
    values = np.empty((entry['frames'], count, len(entry['components'])), dtype=np.float32)
    done = 0
    for _, data in chunks(record, m):
        block = data['wavefield'][slots[0]] if len(slots) == 1 else data['wavefield'][slots].sum(0, dtype=np.float32)
        values[done : done + len(block)] = block if nodes is None else block[:, nodes]
        done += len(block)
    if done != entry['frames']:
        raise ValueError('波场记录不完整。')
    return np.arange(entry['frames']) * entry['sample_step_us'], values


def nearest(entry, point_mm, tolerance_mm=None):
    """Index of the recorded node nearest to a point and its distance (mm); None when the point is farther than the tolerance
    (default: the largest spacing between neighbouring nodes of the record along x or y)."""
    points = entry['points_mm']
    distance = np.linalg.norm(points - np.asarray(point_mm, dtype=float), axis=1)
    i = int(np.argmin(distance))
    if tolerance_mm is None:
        tolerance_mm = (
            max((float(np.diff(np.unique(np.round(points[:, a], 9))).max()) if np.ptp(points[:, a]) > 0 else 0.0) for a in (0, 1, 2))
            or 1e-6
        )
    return (i, float(distance[i])) if distance[i] <= tolerance_mm + 1e-9 else None


def traces(record, part, points):
    """Waveforms at points of the recorded region, taken at the nearest recorded node (float32; components that were not stored are
    zero and listed). Returns None when a point is not on the recorded region, so that the caller can say what the record holds."""
    entry = info(record)
    if not entry or not stored_for(entry['parts'], part):
        return None
    hits = [nearest(entry, p) for p in points]
    if any(h is None for h in hits):
        return None
    times, values = read(record, part, np.array([h[0] for h in hits]))
    full = np.zeros((len(times), len(points), 3))
    full[:, :, entry['components']] = values
    return {
        'times_us': times.tolist(),
        'points': [
            {
                'location': {
                    'point_mm': entry['points_mm'][h[0]].tolist(),
                    'ids': [],
                    'weights': [],
                    'wavefield_node': int(h[0]),
                    'distance_mm': h[1],
                    'components': [AXES[a] for a in entry['components']],
                    'requested_mm': list(p),
                },
                'xyz_m': full[:, i].tolist(),
            }
            for i, (p, h) in enumerate(zip(points, hits))
        ],
    }


def export(record, part, kind, target):
    """Write the wave field of one part to `target` as a NumPy archive ('npz') or a MATLAB file ('mat').

    Arrays: time_us [frames], points_mm [nodes, 3], u_m [frames, nodes, components] (float32, metres), components (e.g. 'z'),
    and for a full rectangular face x_mm, y_mm and grid_index [ny, nx] (u_m[:, grid_index] is the field on the grid;
    1-based in the MATLAB file)."""
    entry = info(record)
    if not entry:
        raise ValueError('这次计算没有保存波场记录。')
    times, values = read(record, part)
    arrays = {'time_us': times, 'points_mm': entry['points_mm'], 'u_m': values, 'components': ''.join(AXES[a] for a in entry['components'])}
    structured = grid(entry['points_mm'])
    if structured:
        arrays.update(x_mm=structured[0], y_mm=structured[1], grid_index=structured[2])
    target = Path(target)
    tmp = target.with_suffix(target.suffix + '.tmp')
    if kind == 'npz':
        with tmp.open('wb') as stream:
            np.savez(stream, **arrays)
    elif kind == 'mat':
        if values.nbytes > MAT_LIMIT:
            raise ValueError(
                f'这份波场有 {values.nbytes / 1024**3:.1f} GiB，超过 MATLAB 5 文件单个变量 2 GiB 的上限：请下载 NumPy 格式（.npz），'
                '或重新计算时加大时间抽稀、只存 Z 分量。'
            )
        from scipy.io import savemat

        if structured:
            arrays['grid_index'] = structured[2] + 1
        with tmp.open('wb') as stream:
            savemat(stream, arrays, do_compression=False)
    else:
        raise ValueError('波场导出格式需为 npz 或 mat。')
    tmp.replace(target)
    return target
