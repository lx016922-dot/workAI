"""Long records through the version-2 input layer: chunked, checkpointed, resumable.

Same durability rules as the frozen `graphwave.long_rollout` (atomic manifest,
SHA-256 for every chunk and checkpoint, fingerprint over model, sources, scene
and mesh), with the multi-source / masked / damped recurrence. Version-1 scenes
are routed to the frozen rollout unchanged.

Two things can be chosen per record (the defaults reproduce the frozen behaviour bit for bit):

* `time_plan`: 'frozen' is the trained three-stage layer at CFL × spacing / 10000 m/s; 'auto' lets
  `graphnet.timestep` pick the cheapest layer that meets its stability and phase-error conditions;
* `spread`: how the layer uses the cores. False gives each control one thread (`recurrence.three_segments`); True
  spreads every control over all threads (`graphnet.parallel`); None (the default) takes the faster of the two for
  the machine. The two give the same bits, so this is a matter of speed only;
* `controls`: the constitutive controls to compute, any of linear, stvk and murnaghan (the complete response).
  All three give the five responses (complete, linear, StVK, geometric increment, material increment); linear and
  murnaghan give the complete and linear responses and their difference; murnaghan alone gives the complete response;
* `record`: 'full' stores every node at every sample (float64); 'compact' stores what a large model can
  afford: the receivers at every sample (float64, bit-identical to the full record), the display mesh
  (`spectral3d.display`) at the animation frames as float32 (`display_parts`: the linear response and the
  increments when the controls allow them), and a few full float64 snapshots.
"""
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
from numba import config as numba_config, set_num_threads
from solid.schema import require
from physics.scene import is_v2
from spectral3d.mesh import interpolation
from graphwave.inference import inverse_mapping
from .inputs import build, signal_table
from .recurrence import three_segments
from .contact import arrays as contact_arrays, stiffness as contact_stiffness

# Readable form of the recurrence guard (strain ≤ 1e-3, J > 0, finite energy); the worker shows it as is.
GUARD_MESSAGE = ('计算超出网络的有效范围：应变超过 1e-3（材料三阶展开的适用上限）、体积比 J ≤ 0 或能量非有限。'
                 '请降低激励幅值。自由端和角点附近入射波与反射波叠加，应变可达入射波的两倍')
from .model import energy_layer, speed_bound

STATE = ('positions', 'velocities', 'accelerations', 'externals', 'works', 'dissipations', 'potentials', 'strains', 'jacobians')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'); tmp.replace(path)


def save_arrays(path, **arrays):
    # Uncompressed: float64 wave fields compress by about 6 % and the compression took longer than the write.
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream: np.savez(stream, **arrays)
    tmp.replace(path)


def source_hashes():
    here = Path(__file__).resolve().parent; frozen = here.parent / 'graphwave'
    rows = {f'graphnet/{p.name}': digest(p) for p in sorted(here.glob('*.py'))}
    rows.update({f'graphwave/{name}': digest(frozen / name) for name in ('inference.py', 'buffered_inference.py')})
    return rows


RECORDS = ('full', 'compact')
LAWS = ('linear', 'stvk', 'murnaghan')
LAW_LABELS = {'linear': '线性', 'stvk': 'StVK', 'murnaghan': '完整'}
SNAPSHOTS = 4
WORK_PER_CHUNK = 6e8     # node × element-message evaluations between checkpoints (a few minutes on a desktop CPU)


def snapshot_frames(frames, count=SNAPSHOTS):
    """Samples whose complete float64 field a compact record keeps: `count` evenly spaced, ending with the last sample."""
    return np.unique(np.linspace(0, frames - 1, count + 1, dtype=int)[1:]) if count else np.zeros(0, dtype=int)


def display_parts(controls):
    """What a compact record stores on the display mesh for these controls: (name, minuend, subtrahend or None).

    Increments are formed in float64 before they are stored as float32."""
    has = set(controls)
    if has == set(LAWS): return (('linear', 'linear', None), ('geometric', 'stvk', 'linear'), ('material', 'murnaghan', 'stvk'))
    if has == {'linear', 'murnaghan'}: return (('linear', 'linear', None), ('nonlinear', 'murnaghan', 'linear'))
    return tuple(('full' if name == 'murnaghan' else name, name, None) for name in controls)


def storage_bytes(frames, nodes, record, display_nodes=0, shown=0, receivers=0, snapshots=SNAPSHOTS, controls=3):
    """Upper bound of the record on disk, checkpoints included."""
    checkpoints = 2 * nodes * controls * 3 * 8 * 4
    if record == 'full': return frames * nodes * 3 * 8 * controls + checkpoints
    return frames * (receivers * 24 * controls + 40 * controls) + shown * display_nodes * 3 * 4 * controls + snapshots * nodes * 24 * controls + checkpoints


def rollout(scene, mesh, model, directory, identity, callback=None, stop_after_chunks=None, time_plan='frozen', record='full', display=None, controls=LAWS,
            spread=None, threads=None):
    """`model` from graphnet.model.load(..., long_record=True). See the module text for `time_plan`, `record`, `controls` and `spread`;
    `threads` limits the threads used (default: all that numba was started with)."""
    if not is_v2(scene):
        from graphwave.long_rollout import rollout as frozen
        return frozen(scene, mesh, model['weights'], model['temporal'], directory, identity, callback=callback, stop_after_chunks=stop_after_chunks)
    from .timestep import plan as layer_plan, largest_frequency, MODES
    require(time_plan in MODES and record in RECORDS, '记录设置无效：time_plan 为 auto 或 frozen，record 为 full 或 compact。')
    controls = tuple(controls)
    require(controls and set(controls) <= set(LAWS) and controls == tuple(name for name in LAWS if name in controls), '控制需为 linear、stvk、murnaghan 中的若干个，按此顺序且不重复。')
    laws = np.array([LAWS.index(name) for name in controls], dtype=np.int64); count = len(controls)
    directory = Path(directory); (directory / 'fields').mkdir(parents=True, exist_ok=True); (directory / 'states').mkdir(parents=True, exist_ok=True)
    from .parallel import ARRAYS, choose, parallel_segments, prepare
    available = max(1, min(int(threads or numba_config.NUMBA_NUM_THREADS), numba_config.NUMBA_NUM_THREADS))
    if spread is None: spread, used = choose(controls, available, len(mesh['points']))
    else: spread = bool(spread); used = available if spread else max(1, min(count, available))
    set_num_threads(used); start = time.monotonic()
    weights = model['weights']; duration, sample = scene['acquisition']['duration_us'], scene['acquisition']['sample_step_us']
    frames = round(duration / sample) + 1; nodes = len(mesh['points']); compact = record == 'compact'
    require(2 <= frames <= 200001 and abs((frames - 1) * sample - duration) <= 1e-8, '长记录需要整数个采样间隔，最多 200001 个采样点。')
    trained = np.asarray(model['temporal']['weights'], dtype=np.float64)
    require(trained.shape == (3,) and np.isfinite(trained).all() and abs(trained.sum() - 1) <= 1e-13 and abs(np.sum(trained ** 3)) <= 1e-13, '时间组合权重无效。')
    inputs = build(scene, mesh); damped = inputs['alpha'] is not None; alpha = inputs['alpha'] if damped else np.zeros(len(mesh['mass']))
    quadratic, cubic, layer = energy_layer(scene, model); derivative = weights['derivative_' + str(int(mesh['degree']))]
    inverse = inverse_mapping(mesh); receivers = [interpolation(mesh, r['point_mm']) for r in scene['receivers']]
    sources = source_hashes()
    fingerprint = hashlib.sha256(json.dumps({'identity': identity, 'sources': sources, 'scene': scene, 'degree': int(mesh['degree']), 'h_mm': float(mesh['h_mm']),
                                             **({} if (time_plan, record) == ('frozen', 'full') else {'time_plan': time_plan, 'record': record}),
                                             **({} if controls == LAWS else {'controls': list(controls)})}, sort_keys=True).encode()).hexdigest()
    # Interface stiffness: that of the trained layer spacing whatever the plan, so the interface is the same physical model
    # (its reflection and its clapping) and only the time layer differs.
    from .timestep import frozen_plan
    trained_step = frozen_plan(scene, mesh, trained, speed_bound(scene))['layer_step_us'] * 1e-6
    ca, cb, cn, _ = contact_arrays(mesh); ck, contact = contact_stiffness(mesh, scene, trained_step)
    manifest_path = directory / 'recording.json'; manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        require(manifest['fingerprint'] == fingerprint, '保存的计算属于不同的模型、配置或运行代码。')
        checkpoint_path = directory / manifest['checkpoint_file']; require(digest(checkpoint_path) == manifest['checkpoint_sha256'], '检查点校验失败。')
        for row in manifest['chunks']: require(digest(directory / row['file']) == row['sha256'], f'记录块校验失败：{row["file"]}')
        if manifest['phase'] == 'completed': return manifest
        layers = manifest['time_plan']   # a resumed record keeps the layer it started with
    else:
        omega = None
        if time_plan == 'auto':
            omega = largest_frequency(mesh, derivative, inverse, quadratic, cubic, inputs['free'], contact=(ca, cb, cn, ck))
        layers = layer_plan(scene, mesh, trained, speed_bound(scene), omega, time_plan, interface_omega=contact['omega_dt_max'] / trained_step if contact else None)
    composition = np.asarray(layers['composition'], dtype=np.float64); stages = len(composition); subdivisions = int(layers['subdivisions'])
    sample_s = sample * 1e-6; dt = sample_s / subdivisions
    total_steps = (frames - 1) * subdivisions; require(total_steps <= 20000000, '长记录超过 2000 万层，请增大单元尺寸。')
    work_frames = int(WORK_PER_CHUNK // (nodes * subdivisions * stages)); shown_parts = display_parts(controls)
    if compact:
        if display is None:
            from spectral3d.display import display_mesh
            display = display_mesh(mesh)
        from spectral3d.display import frames_shown
        shown = frames_shown(frames, len(display['nodes']), 4); kept = snapshot_frames(frames); shown_set = set(shown.tolist()); kept_set = set(kept.tolist())
        keep = np.unique(np.concatenate([ids for ids, _ in receivers])).astype(np.int64) if receivers else np.zeros(0, dtype=np.int64)
        local = [(np.searchsorted(keep, ids), w) for ids, w in receivers]
        chunk_frames = max(2, min(256, 20000 // subdivisions, work_frames, (256 * 1024 ** 2) // (len(display['nodes']) * 12 * len(shown_parts))))
    else:
        keep = np.zeros(0, dtype=np.int64); local = receivers
        chunk_frames = max(2, min(256, 20000 // subdivisions, work_frames, (512 * 1024 ** 2) // (nodes * 24 * count)))   # one chunk (all controls) ≤ 512 MiB
    if manifest is not None:
        with np.load(checkpoint_path, allow_pickle=False) as saved: state = {name: saved[name] for name in STATE}
        next_frame = manifest['completed_frames']; elapsed_before = manifest['elapsed_s']
    else:
        estimate = storage_bytes(frames, nodes, record, len(display['nodes']) if compact else 0, len(shown) if compact else 0, len(receivers), controls=count)
        require(shutil.disk_usage(directory).free >= estimate + 512 * 1024 ** 2, f'预计{"精简" if compact else "完整长"}记录需要约 {estimate / 1024 ** 3:.2f} GiB 空间，请选择空间充足的数据目录。')
        save_arrays(directory / 'mesh.npz', **mesh); shape = (count, len(mesh['mass']), 3)
        state = {name: np.zeros(shape) for name in STATE[:4]}; state.update(works=np.zeros(count), dissipations=np.zeros(count), potentials=np.zeros(count), strains=np.zeros(count), jacobians=np.ones(count))
        next_frame, elapsed_before = 0, 0.
        manifest = {'schema': 'workai.graphnet.long_record.v1', 'phase': 'running', 'duration_us': duration, 'sample_step_us': sample, 'frame_count': frames, 'nodes': len(mesh['mass']),
                    'degree': int(mesh['degree']), 'layer_step_us': dt * 1e6, 'time_order': layers['time_order'], 'composition': composition.tolist(), 'fingerprint': fingerprint, 'identity': identity,
                    'execution_source_sha256': sources, 'mesh_sha256': digest(directory / 'mesh.npz'), 'input_layer': 'v2_multi_source', 'excitation': inputs['summary'],
                    'energy_columns': ['time_us', 'kinetic_j', 'strain_j', 'external_work_j', 'dissipated_j'], 'chunks': [], 'completed_frames': 0, 'elapsed_s': 0.,
                    'raw_storage_upper_bound_bytes': estimate, 'numerical_correction': False, 'neural_forward_only': True, **layer,
                    'time_plan': layers, 'record_mode': record, 'controls': list(controls),
                    **({'contact': dict(contact, stiffness_layer_step_us=trained_step * 1e6, layer_omega_dt_max=contact['omega_dt_max'] * dt / trained_step)} if contact else {})}
        if compact:
            save_arrays(directory / 'display.npz', nodes=display['nodes'], triangles=display['triangles'])
            manifest['display'] = {'file': 'display.npz', 'sha256': digest(directory / 'display.npz'), 'nodes': int(len(display['nodes'])), 'triangles': int(len(display['triangles'])),
                                   'stride': display['stride'], 'cell_mm': display['cell_mm'], 'surface_faces': display['surface_faces'], 'frames': shown.tolist(), 'dtype': 'float32',
                                   'parts': [name for name, _, _ in shown_parts], 'layout': 'part,frame,display node,xyz'}
            manifest['snapshot_frames'] = kept.tolist()
            manifest['receivers'] = [{'name': r['name'], 'point_mm': list(r['point_mm'])} for r in scene['receivers']]
    manifest.update(layer_threads=used, layer_spread=spread)   # how this run uses the cores (a resumed run may differ: the bits do not)
    workspace = prepare(mesh, inputs['loads'], alpha, laws, used) if spread else None
    def advance(first, last, include_initial):
        """Samples first..last (first − 1 is the saved state): fields on the kept nodes or the whole mesh, and energies."""
        start_step = max(0, first - 1) * subdivisions; end_step = last * subdivisions
        stage_times = (np.arange(start_step, end_step)[:, None] + np.cumsum(composition)[None]) * dt * 1e6
        signals = signal_table(inputs, stage_times)
        arguments = (*(state[name] for name in STATE), mesh['elements'], derivative, inverse, mesh['volume_weight'], mesh['mass'],
                     inputs['loads'], inputs['free'], alpha, damped, quadratic, cubic, composition, dt, subdivisions,
                     start_step, end_step, include_initial, signals, ca, cb, cn, ck, keep, compact, laws)
        if spread: fields, energies, failed = parallel_segments(*arguments, *(workspace[name] for name in ARRAYS))
        else: fields, energies, failed = three_segments(*arguments)
        if (failed >= 0).any():
            tripped = '、'.join(LAW_LABELS[name] for name, step in zip(controls, failed) if step >= 0)
            raise ValueError(f'{GUARD_MESSAGE}（{tripped}控制，约 {int(failed.max()) * dt * 1e6:.2f} μs，最大应变 {float(np.max(state["strains"])):.2e}；'
                             f'已保存到 {next_frame * sample:g} μs 的完整记录）')
        return fields, energies
    chunks_this_run = 0
    while next_frame < frames:
        last_frame = min(frames - 1, next_frame + chunk_frames - 1)
        if compact:
            # Stop at every animation and snapshot sample inside the chunk and read it from the state; only the receiver nodes are returned per sample.
            marks = sorted({int(f) for f in np.concatenate([shown, kept]) if next_frame <= f <= last_frame} | {last_frame})
            shown_here = [f for f in marks if f in shown_set]; kept_here = [f for f in marks if f in kept_set]
            shots = np.empty((len(shown_parts), len(shown_here), len(display['nodes']), 3), dtype=np.float32); snaps = np.empty((count, len(kept_here), nodes, 3))
            def take(frame):
                if frame in shown_here:
                    values = dict(zip(controls, state['positions'][:, display['nodes']])); at = shown_here.index(frame)   # increments are formed in float64, then stored as float32
                    for slot, (_, a, b) in enumerate(shown_parts): shots[slot, at] = values[a] if b is None else values[a] - values[b]
                if frame in kept_here: snaps[:, kept_here.index(frame)] = state['positions']
            pieces = []; cursor = next_frame
            if cursor == 0 and marks[0] == 0: take(0); marks = marks[1:]
            for target in marks:
                pieces.append(advance(cursor, target, cursor == 0)); take(target); cursor = target + 1
            fields = np.concatenate([f for f, _ in pieces], axis=1); energies = np.concatenate([e for _, e in pieces], axis=1)
        else:
            fields, energies = advance(next_frame, last_frame, next_frame == 0)
        probes = np.stack([np.stack([np.einsum('n,tnc->tc', w, f[:, ids]) for ids, w in local], 1) for f in fields], 2) if receivers else np.zeros((fields.shape[1], 0, count, 3))
        path = directory / 'fields' / f'{next_frame:06d}.npz'
        if compact: save_arrays(path, probes=probes, energies=energies, display=shots, display_frames=np.asarray(shown_here, dtype=np.int64), snapshots=snaps, snapshot_frames=np.asarray(kept_here, dtype=np.int64))
        else: save_arrays(path, probes=probes, energies=energies, **{name: fields[i] for i, name in enumerate(controls)})
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
    """Concatenate a finished record: times, fields per computed control (frames × nodes × 3), probes (frames × receivers × controls × 3), energies (controls × frames × 5)."""
    directory = Path(directory); manifest = json.loads((directory / 'recording.json').read_text(encoding='utf-8'))
    require(manifest['phase'] == 'completed', '长记录尚未完成。')
    compact = manifest.get('record_mode') == 'compact'
    parts = {'probes': [], 'energies': [], 'display': [], 'snapshots': []} if compact else {**{name: [] for name in manifest.get('controls', LAWS)}, 'probes': [], 'energies': []}
    for row in manifest['chunks']:
        require(digest(directory / row['file']) == row['sha256'], f'记录块校验失败：{row["file"]}')
        with np.load(directory / row['file'], allow_pickle=False) as z:
            for key in parts: parts[key].append(z[key] if key != 'energies' else z[key])
    out = {k: np.concatenate(v, axis=0 if k in ('linear', 'stvk', 'murnaghan', 'probes') else 1) for k, v in parts.items()}
    out['times_us'] = np.arange(manifest['frame_count']) * manifest['sample_step_us']; out['manifest'] = manifest
    return out
