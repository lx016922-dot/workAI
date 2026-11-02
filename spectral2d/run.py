"""Run one plane-strain job from a version-2 scene file; writes JSON, NPZ and optional CSV.

    python -m spectral2d.run --scene scene.json --output out-dir [--degree 4] [--fields] [--csv]

No remote service, no neural network, no numerical correction.
"""
import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from solid.schema import require
from physics.scene import validate_scene, is_plane_strain, summary
from .mesh import make_mesh, interpolation, surface_lines
from .loads import prepare_excitation, receiver_points
from .solver import solve

SCHEMA = 'workai.plane_strain.result.v1'
ENGINE = 'workai_spectral_plane_strain_2d'


def prepared_scene(raw, degree):
    """Validate like the 3-D worker: the resolution screen sees h / degree."""
    require(isinstance(raw, dict) and isinstance(raw.get('numerics'), dict), '场景需要 numerics。')
    h = raw['numerics']['h_mm']; require(type(h) in (int, float) and np.isfinite(h) and .02 <= h <= 100, '单元尺寸必须在 0.02–100 mm 范围内。')
    require(type(degree) is int and 2 <= degree <= 8, '阶次需为 2–8 的整数。')
    s = copy.deepcopy(raw); s['numerics']['h_mm'] = max(.02, h / degree); scene = validate_scene(s); scene['numerics']['h_mm'] = h
    require(is_plane_strain(scene), '二维求解器需要 dimension 为 2d_plane_strain 的 v2 场景。')
    return scene


def simulate(raw, degree=4, keep_fields=False, callback=None, before_solve=None):
    """Python API: returns scene, mesh, receivers and the computed record. `before_solve` may inspect or reject the prepared mesh."""
    scene = prepared_scene(raw, degree); mesh = make_mesh(scene, degree, scene['numerics']['h_mm'])
    excitation = prepare_excitation(mesh, scene); receivers = receiver_points(scene)
    for rx in receivers: interpolation(mesh, rx['point_mm'])
    if before_solve: before_solve({'scene': scene, 'mesh': mesh, 'receivers': receivers})
    computed = solve(scene, mesh, excitation, receivers, cfl=scene['numerics']['cfl'], keep_fields=keep_fields, callback=callback)
    return {'scene': scene, 'mesh': mesh, 'receivers': receivers, 'computed': computed, 'degree': degree}


def browser_record(run, job_id, field_budget_bytes=48 * 1024**2):
    """Result contract for the web viewer: decimated float32 field, full-rate receiver lines, boundary polylines."""
    import gzip
    from .mesh import boundary_segments
    scene = run['scene']; mesh = run['mesh']; computed = run['computed']; receivers = run['receivers']; probes = computed['probes_m']; times = computed['times_us']
    fields = computed['fields_m']; frames = len(times); nodes = len(mesh['points'])
    sha = lambda b: hashlib.sha256(b).hexdigest()
    every = max(1, int(np.ceil(frames * nodes * 2 * 4 / field_budget_bytes))) if fields is not None else None
    indices = list(range(0, frames, every)) if fields is not None else []
    if indices and indices[-1] != frames - 1: indices.append(frames - 1)
    field_raw = np.ascontiguousarray(fields[indices], dtype='<f4').tobytes() if fields is not None else b''
    field_gz = gzip.compress(field_raw, compresslevel=5, mtime=0) if fields is not None else b''
    line_indices = [i for i, rx in enumerate(receivers) if rx['group'] is not None]
    lines_raw = np.ascontiguousarray(probes[:, line_indices], dtype='<f8').tobytes(); lines_gz = gzip.compress(lines_raw, compresslevel=5, mtime=0)
    groups = {}
    for position, i in enumerate(line_indices): groups.setdefault(receivers[i]['group'], []).append((position, i))
    record = {'schema': SCHEMA, 'engine': ENGINE, 'engine_scope': 'numerical_plane_strain_solution_not_neural_prediction', 'job_id': job_id, 'dimension': '2d_plane_strain', 'degree': run['degree'],
              'scene': scene, 'excitation': summary(scene), 'accepted_for_solver_routing': False, 'response_part': 'full',
              'mesh': {'nodes_mm': (mesh['points'] * 1e3).tolist(), 'elements': mesh['elements'].reshape(len(mesh['elements']), -1).tolist(), 'degree': mesh['degree'], 'h_mm': mesh['h_mm'],
                       'boundary_lines': [[int(ids[i]), int(ids[i + 1])] for _, _, _, ids in boundary_segments(mesh) for i in range(len(ids) - 1)]},
              'times_us': times.tolist(), 'diagnostics': computed['diagnostics'],
              'receivers': [{'name': receivers[i]['name'], 'point_mm': receivers[i]['point_mm'], 'direction': receivers[i]['direction'], 'u_x_m': probes[:, i, 0].tolist(), 'u_z_m': probes[:, i, 1].tolist()}
                            for i, rx in enumerate(receivers) if rx['group'] is None],
              'receiver_lines': [{'name': name, 'direction': receivers[members[0][1]]['direction'], 'count': len(members), 'columns': [p for p, _ in members],
                                  'points_mm': [receivers[i]['point_mm'] for _, i in members], 'distance_mm': [receivers[i]['distance_mm'] for _, i in members]} for name, members in groups.items()],
              'lines': {'name': 'lines.bin.gz', 'layout': 'time,point,xz', 'dtype': 'little_endian_float64', 'unit': 'm', 'shape': [frames, len(line_indices), 2],
                        'compressed_bytes': len(lines_gz), 'uncompressed_bytes': len(lines_raw), 'sha256': sha(lines_gz), 'uncompressed_sha256': sha(lines_raw)},
              'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    if fields is not None:
        record['field'] = {'name': 'field.bin.gz', 'layout': 'frame,node,xz', 'dtype': 'little_endian_float32', 'unit': 'm', 'shape': [len(indices), nodes, 2], 'frame_indices': indices,
                           'every_n_frames': every, 'compressed_bytes': len(field_gz), 'uncompressed_bytes': len(field_raw), 'sha256': sha(field_gz), 'uncompressed_sha256': sha(field_raw)}
    return record, field_gz, lines_gz


def write_outputs(run, output, write_csv=False):
    from physics.runs import write_run_outputs
    mesh = run['mesh']; run = dict(run, mesh_summary={'elements': len(mesh['elements']), 'degree': mesh['degree'], 'h_mm': mesh['h_mm']})
    return write_run_outputs(run, output, SCHEMA, ENGINE, '2d_plane_strain', write_csv=write_csv, engine_scope='numerical_plane_strain_solution_not_neural_prediction',
                             lines_extra={'nodes_mm': mesh['points'] * 1e3, 'elements': mesh['elements']}, fields_extra={'elements': mesh['elements'], 'surface_lines': np.array(surface_lines(mesh))})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scene', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--degree', type=int, default=4); parser.add_argument('--substeps', type=int, help='ignored: explicit solver'); parser.add_argument('--fields', action='store_true'); parser.add_argument('--csv', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    raw = json.loads(args.scene.read_text(encoding='utf-8'))
    def progress(value): print(json.dumps(value), flush=True)
    run = simulate(raw, args.degree, keep_fields=args.fields, callback=progress)
    result = write_outputs(run, args.output, write_csv=args.csv)
    d = result['diagnostics']
    print(json.dumps({'completed': True, 'output': str(args.output), 'nodes': d['nodes'], 'elements': d['elements'], 'internal_steps': d['internal_steps'],
                      'energy_balance_error': d['external_work_balance_relative'], 'seconds': time.monotonic() - started}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
