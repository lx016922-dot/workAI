"""Durable long records, bounded memory, and exact checkpoint continuation."""
import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from numba import get_num_threads, set_num_threads
from spectral3d.mesh import make_mesh, boundary_data, interpolation
from .inference import inverse_mapping
from .signals import pulse_values
from .long_inference import three_segments


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def save_arrays(path, **arrays):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def rollout(scene, mesh, weights, temporal, directory, identity, callback=None, stop_after_chunks=None):
    directory = Path(directory)
    (directory / 'fields').mkdir(parents=True, exist_ok=True)
    (directory / 'states').mkdir(parents=True, exist_ok=True)
    set_num_threads(min(3, get_num_threads()))
    start = time.monotonic()
    duration, sample = scene['acquisition']['duration_us'], scene['acquisition']['sample_step_us']
    frames = round(duration / sample) + 1
    if not 2 <= frames <= 200001 or abs((frames - 1) * sample - duration) > 1e-8:
        raise ValueError('Long record requires an integral sampling grid with at most 200001 samples.')
    composition = np.asarray(temporal['weights'], dtype=np.float64)
    if composition.shape != (3,) or not np.isfinite(composition).all() or abs(composition.sum() - 1) > 1e-13 or abs(np.sum(composition**3)) > 1e-13:
        raise ValueError('Invalid learned time-composition weights.')
    sample_s = sample * 1e-6
    subdivisions = int(math.ceil(sample_s / (.3 * float(mesh['min_node_spacing']) / 10000.)))
    dt = sample_s / subdivisions
    total_steps = (frames - 1) * subdivisions
    if total_steps > 20000000:
        raise ValueError('Long record exceeds 20 million time layers; use a larger element size.')
    chunk_frames = max(2, min(256, 20000 // subdivisions))
    load, fixed, _ = boundary_data(mesh, scene)
    inverse = inverse_mapping(mesh)
    receivers = [interpolation(mesh, r['point_mm']) for r in scene['receivers']]
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in ['long_rollout.py', 'long_inference.py', 'buffered_inference.py', 'inference.py', 'signals.py']}
    fingerprint = hashlib.sha256(json.dumps({'identity': identity, 'sources': source_hashes, 'scene': scene,
        'degree': int(mesh['degree']), 'h_mm': float(mesh['h_mm'])}, sort_keys=True).encode()).hexdigest()
    manifest_path = directory / 'recording.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('Saved computation belongs to a different model, configuration, or runtime.')
        checkpoint_path = directory / manifest['checkpoint_file']
        assert digest(checkpoint_path) == manifest['checkpoint_sha256']
        for row in manifest['chunks']:
            assert digest(directory / row['file']) == row['sha256']
        if manifest['phase'] == 'completed':
            return manifest
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            state = {name: saved[name] for name in ['positions','velocities','accelerations','externals','works','potentials','strains','jacobians']}
        next_frame = manifest['completed_frames']
        elapsed_before = manifest['elapsed_s']
    else:
        estimate = frames * len(mesh['points']) * 3 * 8 * 3
        if shutil.disk_usage(directory).free < estimate + 512 * 1024**2:
            raise ValueError(f'预计完整长记录需要约 {estimate/1024**3:.2f} GiB 空间，请选择空间充足的数据目录。')
        save_arrays(directory / 'mesh.npz', **mesh)
        shape = (3, len(load), 3)
        state = {name: np.zeros(shape) for name in ['positions','velocities','accelerations','externals']}
        state.update(works=np.zeros(3), potentials=np.zeros(3), strains=np.zeros(3), jacobians=np.ones(3))
        next_frame, elapsed_before = 0, 0.
        manifest = {'schema':'workai.long_record.v1','phase':'running','duration_us':duration,'sample_step_us':sample,
                    'frame_count':frames,'nodes':len(load),'degree':int(mesh['degree']),'layer_step_us':dt*1e6,
                    'time_order':4,'composition':composition.tolist(),'fingerprint':fingerprint,'identity':identity,
                    'execution_source_sha256':source_hashes,'mesh_sha256':digest(directory/'mesh.npz'),
                    'chunks':[],'completed_frames':0,'elapsed_s':0.,'raw_storage_upper_bound_bytes':estimate,
                    'numerical_correction':False,'neural_forward_only':True}
    chunks_this_run = 0
    while next_frame < frames:
        last_frame = min(frames - 1, next_frame + chunk_frames - 1)
        start_step = max(0, next_frame - 1) * subdivisions
        end_step = last_frame * subdivisions
        stage_times = (np.arange(start_step, end_step)[:, None] + np.cumsum(composition)[None]) * dt * 1e6
        pulse = pulse_values(scene, stage_times)
        fields, energies = three_segments(*(state[name] for name in ['positions','velocities','accelerations','externals','works','potentials','strains','jacobians']),
            mesh['elements'], weights['derivative_' + str(int(mesh['degree']))], inverse, mesh['volume_weight'], mesh['mass'], load, fixed,
            weights['quadratic_tensor'], weights['cubic_tensor'], composition, dt, subdivisions, start_step, end_step, next_frame == 0, pulse)
        probes = np.stack([np.stack([np.einsum('n,tnc->tc', w, f[:, ids]) for ids, w in receivers], 1) for f in fields], 2)
        path = directory / 'fields' / f'{next_frame:06d}.npz'
        save_arrays(path, linear=fields[0], stvk=fields[1], murnaghan=fields[2], probes=probes, energies=energies)
        row = {'file':str(path.relative_to(directory)).replace(os.sep,'/'),'first_frame':next_frame,
               'frame_count':last_frame-next_frame+1,'sha256':digest(path),'bytes':path.stat().st_size}
        checkpoint_path = directory / 'states' / f'{last_frame:06d}.npz'
        save_arrays(checkpoint_path, **state)
        manifest['chunks'].append(row)
        next_frame = last_frame + 1
        manifest.update(completed_frames=next_frame,completed_time_us=last_frame*sample,
                        checkpoint_file=str(checkpoint_path.relative_to(directory)).replace(os.sep,'/'),
                        checkpoint_sha256=digest(checkpoint_path),elapsed_s=elapsed_before+time.monotonic()-start,
                        max_strain=state['strains'].tolist(),min_J=state['jacobians'].tolist())
        if next_frame == frames:
            manifest['phase'] = 'completed'
        atomic(manifest_path, manifest)
        # The manifest now points to the new durable state. Keep one previous
        # state as well; an interruption before the atomic update remains safe.
        for old in sorted((directory / 'states').glob('*.npz'))[:-2]:
            old.unlink()
        if callback:
            callback(manifest)
        chunks_this_run += 1
        if stop_after_chunks and chunks_this_run >= stop_after_chunks:
            return manifest
    return manifest


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.input.read_text(encoding='utf-8'))
    temporal = json.loads(args.temporal.read_text(encoding='utf-8'))
    with np.load(args.weights, allow_pickle=False) as saved:
        weights = dict(saved)
    scene = request['scene']
    mesh = make_mesh(scene, request['degree'], request.get('h_mm', scene['numerics']['h_mm']))
    manifest = rollout(scene, mesh, weights, temporal, args.output,
        {'weights_sha256':digest(args.weights),'time_weights_sha256':digest(args.temporal),'input_sha256':digest(args.input)},
        callback=lambda m: print(json.dumps({k:m[k] for k in ['completed_time_us','duration_us','elapsed_s','phase']}),flush=True),
        stop_after_chunks=args.stop_after_chunks)
    print(json.dumps({'phase':manifest['phase'],'chunks':len(manifest['chunks']),'frame_count':manifest['frame_count']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--temporal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-after-chunks',type=int)
    run(parser.parse_args())
