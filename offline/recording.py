"""Read records with bounded memory: complete float64 records and compact records (`graphnet.rollout`).

A compact record holds every sample only at the receivers, so waveforms can be extracted at the receivers; the
animation comes from the display mesh stored at the animation frames."""
import json
from pathlib import Path
import numpy as np
from graphwave.long_rollout import digest
from spectral3d.mesh import interpolation


def manifest(directory):
    directory = Path(directory)
    m = json.loads((directory / 'recording.json').read_text(encoding='utf-8'))
    if m['phase'] != 'completed' or m['completed_frames'] != m['frame_count']:
        raise ValueError('完整记录尚未完成。')
    if digest(directory / 'mesh.npz') != m['mesh_sha256']:
        raise ValueError('记录网格校验失败。')
    return m


def chunks(directory, m):
    offset = 0
    for row in m['chunks']:
        path = (Path(directory) / row['file']).resolve()
        if not path.is_relative_to(Path(directory).resolve()) or row['first_frame'] != offset or digest(path) != row['sha256']:
            raise ValueError('完整记录分块校验失败。')
        with np.load(path, allow_pickle=False) as data:
            yield row, data
        offset += row['frame_count']
    if offset != m['frame_count']:
        raise ValueError('记录缺少采样点。')


def response(data, part):
    if part == 'full': return data['murnaghan']
    if part in ('linear', 'stvk'): return data[part]
    if part == 'material': return data['murnaghan'] - data['stvk']
    if part == 'geometric': return data['stvk'] - data['linear']
    raise ValueError('响应类型无效。')


LAW = {'linear': 0, 'stvk': 1, 'murnaghan': 2}


def is_compact(m):
    return m.get('record_mode') == 'compact'


def _receiver_traces(directory, m, part, points):
    """Waveforms of a compact record: the requested points must be receivers of the computation."""
    rows = []
    for p in points:
        if not (isinstance(p, list) and len(p) == 3 and all(type(v) in (int, float) for v in p)):
            raise ValueError('点位需要三个坐标（mm）。')
        hit = [i for i, r in enumerate(m['receivers']) if max(abs(a - b) for a, b in zip(r['point_mm'], p)) <= 1e-6]
        if not hit:
            raise ValueError('这次计算保存的是精简记录：只有接收点保存了每个采样点的完整波形。请把该点加为接收点后重新计算，'
                             '或在“求解器配置”里把记录方式改为“完整记录”。已保存的接收点：' + '、'.join(r['name'] for r in m['receivers']))
        rows.append(hit[0])
    values = np.empty((m['frame_count'], len(points), 3))
    for row, data in chunks(directory, m):
        probes = data['probes'][:, rows]   # samples × receivers × controls × xyz
        field = {'full': lambda: probes[:, :, 2], 'linear': lambda: probes[:, :, 0], 'stvk': lambda: probes[:, :, 1],
                 'material': lambda: probes[:, :, 2] - probes[:, :, 1], 'geometric': lambda: probes[:, :, 1] - probes[:, :, 0]}[part]()
        values[row['first_frame']:row['first_frame'] + row['frame_count']] = field
    return {'times_us': (np.arange(m['frame_count']) * m['sample_step_us']).tolist(), 'recording_fingerprint': m['fingerprint'],
            'points': [{'location': {'point_mm': p, 'ids': [], 'weights': [], 'receiver': m['receivers'][r]['name']}, 'xyz_m': values[:, i].tolist()}
                       for i, (p, r) in enumerate(zip(points, rows))]}


def traces(directory, part, points):
    if not isinstance(points, list) or not 1 <= len(points) <= 8:
        raise ValueError('每次提取需要 1–8 个实体内点位。')
    m = manifest(directory)
    if part not in ('full', 'linear', 'stvk', 'material', 'geometric'): raise ValueError('响应类型无效。')
    if is_compact(m): return _receiver_traces(directory, m, part, points)
    with np.load(Path(directory) / 'mesh.npz', allow_pickle=False) as data:
        mesh = dict(data)
    locations = [interpolation(mesh, p) for p in points]
    values = np.empty((m['frame_count'], len(points), 3))
    for row, data in chunks(directory, m):
        field = response(data, part)
        start = row['first_frame']; end = start + row['frame_count']
        for j,(ids,w) in enumerate(locations):
            values[start:end,j] = np.einsum('n,tnc->tc', w, field[:,ids])
    return {'times_us': (np.arange(m['frame_count'])*m['sample_step_us']).tolist(),
            'recording_fingerprint': m['fingerprint'],
            'points': [{'location': {'point_mm': p, 'ids': ids.tolist(), 'weights': w.tolist()},
                        'xyz_m': values[:,i].tolist()} for i,(p,(ids,w)) in enumerate(zip(points,locations))]}


class _Energy:
    """Largest energy-balance error of each control over the chunks of a record."""
    def __init__(self): self.scale = np.zeros(3); self.error = np.zeros(3)

    def add(self, energies):
        for law in range(3):
            energy = energies[law]
            # v2 records carry a fifth column with the energy taken out by absorbing layers.
            total = energy[:,1]+energy[:,2]+(energy[:,4] if energy.shape[1] > 4 else 0.)
            self.scale[law] = max(self.scale[law],float(np.max(abs(total))),float(np.max(abs(energy[:,3]))))
            self.error[law] = max(self.error[law],float(np.max(abs(total-energy[:,3]))))

    def diagnostics(self, m, output_dtype='float64'):
        return [dict(law=name, layer_step_us=m['layer_step_us'], internal_step_us=m['layer_step_us'],
            time_order=m['time_order'], calculation_dtype='float64', output_dtype=output_dtype,
            external_work_balance_relative=float(self.error[i]/max(self.scale[i],1e-30)),
            energy_balance_within_1_percent=bool(self.error[i]<=.01*max(self.scale[i],1e-30)),
            max_strain=m['max_strain'][i],min_J=m['min_J'][i],nodes=m['nodes']) for i,name in enumerate(('linear','stvk','murnaghan'))]


def preview(directory):
    """Animation of a complete record on the computation mesh: (manifest, times, fields per control, diagnostics)."""
    m = manifest(directory)
    # Animation frames: up to 281, fewer for large meshes so one response stays within the 192 MiB display budget.
    shown = max(2, min(281, m['frame_count'], (192 * 1024**2) // (m['nodes'] * 3 * 8)))
    indices = np.unique(np.linspace(0,m['frame_count']-1,shown,dtype=int))
    fields = np.empty((3,len(indices),m['nodes'],3)); energy = _Energy()
    for row, data in chunks(directory,m):
        selected = np.flatnonzero((indices>=row['first_frame']) & (indices<row['first_frame']+row['frame_count']))
        local = indices[selected]-row['first_frame']
        for law,name in enumerate(('linear','stvk','murnaghan')): fields[law,selected] = data[name][local]
        energy.add(data['energies'])
    return m,indices*m['sample_step_us'],fields,energy.diagnostics(m)


def display_preview(directory):
    """Animation on the display mesh, for any record: (manifest, times, {response part: float32 frames × display nodes × 3}, diagnostics, display mesh).

    The parts are formed in float64 before they are stored as float32, so the nonlinear increments keep their own
    precision (a difference of two float32 fields would not)."""
    from spectral3d.display import display_mesh, frames_shown
    directory = Path(directory); m = manifest(directory); energy = _Energy()
    if is_compact(m):
        if digest(directory / m['display']['file']) != m['display']['sha256']: raise ValueError('显示网格校验失败。')
        with np.load(directory / m['display']['file'], allow_pickle=False) as data: display = {'nodes': data['nodes'], 'triangles': data['triangles']}
        display.update({k: m['display'][k] for k in ('stride', 'cell_mm', 'surface_faces')})
        indices = np.asarray(m['display']['frames']); pieces = []
        for row, data in chunks(directory, m):
            pieces.append(data['display']); energy.add(data['energies'])
        stored = np.concatenate(pieces, axis=1)
        if stored.shape[1] != len(indices): raise ValueError('精简记录缺少动画帧。')
        parts = {'linear': stored[0], 'geometric': stored[1], 'material': stored[2],
                 'stvk': (stored[0].astype(np.float64) + stored[1]).astype(np.float32), 'full': (stored[0].astype(np.float64) + stored[1] + stored[2]).astype(np.float32)}
    else:
        with np.load(directory / 'mesh.npz', allow_pickle=False) as data: mesh = dict(data)
        display = display_mesh(mesh); nodes = display['nodes']; indices = frames_shown(m['frame_count'], len(nodes), 4)
        parts = {name: np.empty((len(indices), len(nodes), 3), dtype=np.float32) for name in ('linear', 'stvk', 'full', 'geometric', 'material')}
        for row, data in chunks(directory, m):
            selected = np.flatnonzero((indices >= row['first_frame']) & (indices < row['first_frame'] + row['frame_count']))
            if len(selected):
                local = indices[selected] - row['first_frame']
                linear, stvk, full = (data[name][local][:, nodes] for name in ('linear', 'stvk', 'murnaghan'))
                for name, value in (('linear', linear), ('stvk', stvk), ('full', full), ('geometric', stvk - linear), ('material', full - stvk)): parts[name][selected] = value
            energy.add(data['energies'])
    return m, indices * m['sample_step_us'], parts, energy.diagnostics(m, 'float32'), display
