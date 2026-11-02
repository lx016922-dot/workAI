"""Run one 3-D high-order job from a scene file (version 1 or 2) with the sweep/run output layout.

    python -m spectral3d.run --scene scene.json --output out-dir [--degree 3] [--substeps 4] [--fields] [--csv]

Plates, notches and pipes use the explicit GLL solver; through-holes use the implicit Newmark
solver with `--substeps` internal steps per output sample, exactly as the web worker does.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from solid.schema import require
from physics.scene import validate_scene, is_v2, is_plane_strain, spectral_admissible
from physics.receivers import receiver_points
from physics.runs import write_run_outputs
from .mesh import make_mesh, boundary_data, interpolation
from .loads import prepare_excitation

SCHEMA = 'workai.spectral3d.run.result.v1'
ENGINE = 'workai_spectral_reference_3d'
FACES = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')


def admissible(scene):
    """What the GLL solvers can load; version-1 scenes keep the web worker's rule (complete x end faces)."""
    require(not is_plane_strain(scene), '该场景声明为二维平面应变，请使用 spectral2d.run。')
    if is_v2(scene): return spectral_admissible(scene)
    require(scene['geometry']['kind'] in ('plate', 'pipe', 'hole', 'notch'), '高阶计算支持参数化板、管、通孔和缺口。')
    require(scene['source']['face'] in ('xmin', 'xmax') and scene['source']['radius_mm'] == 0 and scene['boundary']['fixed_face'] in ('none', 'xmin', 'xmax'), '版本 1 场景的高阶计算使用完整 X 端面加载，固定面可选另一 X 端面。')
    return scene


def prepared_scene(raw, degree):
    require(isinstance(raw, dict) and isinstance(raw.get('numerics'), dict), '场景需要 numerics。')
    h = raw['numerics']['h_mm']; require(type(h) in (int, float) and np.isfinite(h) and .02 <= h <= 100, '单元尺寸必须在 0.02–100 mm 范围内。')
    require(type(degree) is int and 2 <= degree <= 6, '阶次需为 2–6 的整数。')
    s = copy.deepcopy(raw); s['numerics']['h_mm'] = max(.02, h / degree); scene = validate_scene(s); scene['numerics']['h_mm'] = h
    return admissible(scene)


def simulate(raw, degree=3, keep_fields=False, callback=None, substeps=4, before_solve=None):
    scene = prepared_scene(raw, degree); mesh = make_mesh(scene, degree, scene['numerics']['h_mm'])
    excitation = prepare_excitation(mesh, scene)
    if excitation is None: boundary_data(mesh, scene)
    receivers = receiver_points(scene)
    for rx in receivers: interpolation(mesh, rx['point_mm'])
    if before_solve: before_solve({'scene': scene, 'mesh': mesh, 'receivers': receivers})
    solver_scene = dict(scene, receivers=[{'name': rx['name'], 'point_mm': rx['point_mm'], 'direction': rx['direction']} for rx in receivers])
    if scene['geometry']['kind'] == 'hole':
        from .implicit import solve
        computed = solve(solver_scene, mesh, substeps=substeps, keep_fields=keep_fields, callback=callback, excitation=excitation)
    else:
        from .solver import solve
        computed = solve(solver_scene, mesh, cfl=scene['numerics']['cfl'], keep_fields=keep_fields, callback=callback, excitation=excitation)
    return {'scene': scene, 'mesh': mesh, 'receivers': receivers, 'computed': computed, 'degree': degree, 'substeps': substeps if scene['geometry']['kind'] == 'hole' else None}


def write_outputs(run, output, write_csv=False):
    mesh = run['mesh']; run = dict(run, mesh_summary={'elements': len(mesh['elements']), 'degree': mesh['degree'], 'h_mm': mesh['h_mm'], 'curved': str(mesh.get('mapping_kind', '')) == 'isoparametric'})
    return write_run_outputs(run, output, SCHEMA, ENGINE, '3d', write_csv=write_csv, extra={'substeps': run['substeps']})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scene', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--substeps', type=int, default=4); parser.add_argument('--fields', action='store_true'); parser.add_argument('--csv', action='store_true')
    args = parser.parse_args(); started = time.monotonic(); raw = json.loads(args.scene.read_text(encoding='utf-8'))
    run = simulate(raw, args.degree, keep_fields=args.fields, callback=lambda v: print(json.dumps(v), flush=True), substeps=args.substeps)
    result = write_outputs(run, args.output, write_csv=args.csv); d = result['diagnostics']
    print(json.dumps({'completed': True, 'output': str(args.output), 'nodes': d['nodes'], 'elements': d['elements'], 'internal_steps': d['internal_steps'], 'energy_balance_error': d['external_work_balance_relative'], 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__': main()
