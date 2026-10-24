"""Read complete float64 records with bounded memory."""
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


def traces(directory, part, points):
    if not isinstance(points, list) or not 1 <= len(points) <= 8:
        raise ValueError('每次提取需要 1–8 个实体内点位。')
    m = manifest(directory)
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


def preview(directory):
    m = manifest(directory)
    indices = np.unique(np.linspace(0,m['frame_count']-1,min(281,m['frame_count']),dtype=int))
    fields = np.empty((3,len(indices),m['nodes'],3))
    energy_scale = np.zeros(3); energy_error = np.zeros(3)
    for row, data in chunks(directory,m):
        selected = np.flatnonzero((indices>=row['first_frame']) & (indices<row['first_frame']+row['frame_count']))
        local = indices[selected]-row['first_frame']
        for law,name in enumerate(('linear','stvk','murnaghan')):
            fields[law,selected] = data[name][local]
            energy = data['energies'][law]
            # v2 records carry a fifth column with the energy taken out by absorbing layers.
            total = energy[:,1]+energy[:,2]+(energy[:,4] if energy.shape[1] > 4 else 0.)
            energy_scale[law] = max(energy_scale[law],float(np.max(abs(total))),float(np.max(abs(energy[:,3]))))
            energy_error[law] = max(energy_error[law],float(np.max(abs(total-energy[:,3]))))
    diagnostics = [dict(law=name, layer_step_us=m['layer_step_us'], internal_step_us=m['layer_step_us'],
        time_order=m['time_order'], calculation_dtype='float64', output_dtype='float64',
        external_work_balance_relative=float(energy_error[i]/max(energy_scale[i],1e-30)),
        energy_balance_within_1_percent=bool(energy_error[i]<=.01*max(energy_scale[i],1e-30)),
        max_strain=m['max_strain'][i],min_J=m['min_J'][i],nodes=m['nodes']) for i,name in enumerate(('linear','stvk','murnaghan'))]
    return m,indices*m['sample_step_us'],fields,diagnostics
