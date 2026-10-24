"""Long records through the version-2 input layer: chunked, checkpointed, resumable.

Same durability rules as the frozen `graphwave.long_rollout` (atomic manifest,
SHA-256 for every chunk and checkpoint, fingerprint over model, sources, scene
and mesh), with the multi-source / masked / damped recurrence. Version-1 scenes
are routed to the frozen rollout unchanged.
"""
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
from numba import get_num_threads, set_num_threads
from solid.schema import require
from physics.scene import is_v2
from spectral3d.mesh import interpolation
from graphwave.inference import inverse_mapping
from .inputs import build, signal_table
from .recurrence import three_segments

STATE = ('positions', 'velocities', 'accelerations', 'externals', 'works', 'dissipations', 'potentials', 'strains', 'jacobians')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); tmp.replace(path)


def save_arrays(path, **arrays):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def source_hashes():
    here = Path(__file__).resolve().parent; frozen = here.parent / 'graphwave'
    rows = {f'graphnet/{p.name}': digest(p) for p in sorted(here.glob('*.py'))}
    rows.update({f'graphwave/{name}': digest(frozen / name) for name in ('inference.py', 'buffered_inference.py')})
    return rows


def rollout(scene, mesh, model, directory, identity, callback=None, stop_after_chunks=None):
    """`model` from graphnet.model.load(..., long_record=True)."""
    if not is_v2(scene):
        from graphwave.long_rollout import rollout as frozen
        return frozen(scene, mesh, model['weights'], model['temporal'], directory, identity, callback=callback, stop_after_chunks=stop_after_chunks)
    directory = Path(directory); (directory / 'fields').mkdir(parents=True, exist_ok=True); (directory / 'states').mkdir(parents=True, exist_ok=True)
    set_num_threads(min(3, get_num_threads())); start = time.monotonic()
    weights = model['weights']; duration, sample = scene['acquisition']['duration_us'], scene['acquisition']['sample_step_us']
    frames = round(duration / sample) + 1
    require(2 <= frames <= 200001 and abs((frames - 1) * sample - duration) <= 1e-8, '长记录需要整数个采样间隔，最多 200001 个采样点。')
    composition = np.asarray(model['temporal']['weights'], dtype=np.float64)
    require(composition.shape == (3,) and np.isfinite(composition).all() and abs(composition.sum() - 1) <= 1e-13 and abs(np.sum(composition ** 3)) <= 1e-13, '时间组合权重无效。')
    cfl = scene['numerics']['cfl']; require(.05 <= cfl <= .3, '网络层间距的验证范围为 CFL 0.05–0.3。')
    sample_s = sample * 1e-6; subdivisions = int(math.ceil(sample_s / (cfl * float(mesh['min_node_spacing']) / 10000.))); dt = sample_s / subdivisions
    total_steps = (frames - 1) * subdivisions; require(total_steps <= 20000000, '长记录超过 2000 万层，请增大单元尺寸。')
    chunk_frames = max(2, min(256, 20000 // subdivisions))
    inputs = build(scene, mesh); damped = inputs['alpha'] is not None; alpha = inputs['alpha'] if damped else np.zeros(len(mesh['mass']))
    inverse = inverse_mapping(mesh); receivers = [interpolation(mesh, r['point_mm']) for r in scene['receivers']]
    sources = source_hashes()
    fingerprint = hashlib.sha256(json.dumps({'identity': identity, 'sources': sources, 'scene': scene, 'degree': int(mesh['degree']), 'h_mm': float(mesh['h_mm'])}, sort_keys=True).encode()).hexdigest()
    manifest_path = directory / 'recording.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        require(manifest['fingerprint'] == fingerprint, '保存的计算属于不同的模型、配置或运行代码。')
        checkpoint_path = directory / manifest['checkpoint_file']; require(digest(checkpoint_path) == manifest['checkpoint_sha256'], '检查点校验失败。')
        for row in manifest['chunks']: require(digest(directory / row['file']) == row['sha256'], f'记录块校验失败：{row["file"]}')
        if manifest['phase'] == 'completed': return manifest
        with np.load(checkpoint_path, allow_pickle=False) as saved: state = {name: saved[name] for name in STATE}
        next_frame = manifest['completed_frames']; elapsed_before = manifest['elapsed_s']
    else:
        estimate = frames * len(mesh['points']) * 3 * 8 * 3
        require(shutil.disk_usage(directory).free >= estimate + 512 * 1024 ** 2, f'预计完整长记录需要约 {estimate / 1024 ** 3:.2f} GiB 空间，请选择空间充足的数据目录。')
        save_arrays(directory / 'mesh.npz', **mesh); shape = (3, len(mesh['mass']), 3)
        state = {name: np.zeros(shape) for name in STATE[:4]}; state.update(works=np.zeros(3), dissipations=np.zeros(3), potentials=np.zeros(3), strains=np.zeros(3), jacobians=np.ones(3))
        next_frame, elapsed_before = 0, 0.
        manifest = {'schema': 'workai.graphnet.long_record.v1', 'phase': 'running', 'duration_us': duration, 'sample_step_us': sample, 'frame_count': frames, 'nodes': len(mesh['mass']),
                    'degree': int(mesh['degree']), 'layer_step_us': dt * 1e6, 'time_order': 4, 'composition': composition.tolist(), 'fingerprint': fingerprint, 'identity': identity,
                    'execution_source_sha256': sources, 'mesh_sha256': digest(directory / 'mesh.npz'), 'input_layer': 'v2_multi_source', 'excitation': inputs['summary'],
                    'energy_columns': ['time_us', 'kinetic_j', 'strain_j', 'external_work_j', 'dissipated_j'], 'chunks': [], 'completed_frames': 0, 'elapsed_s': 0.,
                    'raw_storage_upper_bound_bytes': estimate, 'numerical_correction': False, 'neural_forward_only': True}
    chunks_this_run = 0
    while next_frame < frames:
        last_frame = min(frames - 1, next_frame + chunk_frames - 1)
        start_step = max(0, next_frame - 1) * subdivisions; end_step = last_frame * subdivisions
        stage_times = (np.arange(start_step, end_step)[:, None] + np.cumsum(composition)[None]) * dt * 1e6
        signals = signal_table(inputs, stage_times)
        fields, energies = three_segments(*(state[name] for name in STATE), mesh['elements'], weights['derivative_' + str(int(mesh['degree']))], inverse, mesh['volume_weight'], mesh['mass'],
                                          inputs['loads'], inputs['free'], alpha, damped, weights['quadratic_tensor'], weights['cubic_tensor'], composition, dt, subdivisions,
                                          start_step, end_step, next_frame == 0, signals)
        probes = np.stack([np.stack([np.einsum('n,tnc->tc', w, f[:, ids]) for ids, w in receivers], 1) for f in fields], 2) if receivers else np.zeros((fields.shape[1], 0, 3, 3))
        path = directory / 'fields' / f'{next_frame:06d}.npz'
        save_arrays(path, linear=fields[0], stvk=fields[1], murnaghan=fields[2], probes=probes, energies=energies)
        row = {'file': str(path.relative_to(directory)).replace(os.sep, '/'), 'first_frame': next_frame, 'frame_count': last_frame - next_frame + 1, 'sha256': digest(path), 'bytes': path.stat().st_size}
        checkpoint_path = directory / 'states' / f'{last_frame:06d}.npz'; save_arrays(checkpoint_path, **state)
        manifest['chunks'].append(row); next_frame = last_frame + 1
        manifest.update(completed_frames=next_frame, completed_time_us=last_frame * sample, checkpoint_file=str(checkpoint_path.relative_to(directory)).replace(os.sep, '/'),
                        checkpoint_sha256=digest(checkpoint_path), elapsed_s=elapsed_before + time.monotonic() - start, max_strain=state['strains'].tolist(), min_J=state['jacobians'].tolist(),
                        dissipated_energy_j=state['dissipations'].tolist())
        if next_frame == frames: manifest['phase'] = 'completed'
        atomic(manifest_path, manifest)
        for old in sorted((directory / 'states').glob('*.npz'))[:-2]: old.unlink()
        if callback: callback(manifest)
        chunks_this_run += 1
        if stop_after_chunks and chunks_this_run >= stop_after_chunks: return manifest
    return manifest


def read_record(directory):
    """Concatenate a finished record: times, fields per law (frames × nodes × 3), probes (frames × receivers × 3 × laws), energies (laws × frames × 5)."""
    directory = Path(directory); manifest = json.loads((directory / 'recording.json').read_text(encoding='utf-8'))
    require(manifest['phase'] == 'completed', '长记录尚未完成。')
    parts = {'linear': [], 'stvk': [], 'murnaghan': [], 'probes': [], 'energies': []}
    for row in manifest['chunks']:
        require(digest(directory / row['file']) == row['sha256'], f'记录块校验失败：{row["file"]}')
        with np.load(directory / row['file'], allow_pickle=False) as z:
            for key in parts: parts[key].append(z[key] if key != 'energies' else z[key])
    out = {k: np.concatenate(v, axis=1 if k == 'energies' else 0) for k, v in parts.items()}
    out['times_us'] = np.arange(manifest['frame_count']) * manifest['sample_step_us']; out['manifest'] = manifest
    return out
