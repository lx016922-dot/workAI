"""Run one tetrahedral job from a scene file (version 1 or 2) with the sweep/run output layout.

    python -m solid.run --scene scene.json --output out-dir [--fields] [--csv]

Parametric shapes only (CAD assets belong to the web workbench); float64 fields; receivers and
receiver lines are interpolated from the field at the output frames.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .schema import require
from .mesh import make_mesh, nodal_load, face_selection, locate, prepare_excitation
from .fem import solve
from physics.scene import validate_scene, is_v2, is_plane_strain
from physics.receivers import receiver_points
from physics.runs import write_run_outputs

SCHEMA = 'workai.solid.run.result.v1'
ENGINE = 'workai_explicit_hyperelastic_3d'


def simulate(raw, keep_fields=False, callback=None, before_solve=None):
    scene = validate_scene(raw); require(not is_plane_strain(scene), '该场景声明为二维平面应变，请使用 spectral2d.run。')
    require(scene['geometry']['kind'] != 'cad', '命令行四面体运行只支持参数化结构；CAD 导入请使用网页任务。')
    mesh = make_mesh(scene); receivers = receiver_points(scene); taps = [locate(mesh, rx['point_mm']) for rx in receivers]
    excitation = prepare_excitation(mesh, scene)
    if before_solve: before_solve({'scene': scene, 'mesh': mesh, 'receivers': receivers})
    started = time.monotonic()
    if excitation is None:
        load, selected = nodal_load(mesh, scene['source']); fixed = np.array([], dtype=int)
        if scene['boundary']['fixed_face'] != 'none': fixed = np.unique(mesh['surface'][face_selection(mesh, scene['boundary']['fixed_face'])])
        if np.any(load[fixed] != 0): raise ValueError('激励与固定节点重叠；请调整端面/孔径。')
        fields, diagnostics = solve(scene, mesh, load, fixed, callback, 'float64')
    else: fields, diagnostics = solve(scene, mesh, None, None, callback, 'float64', excitation=excitation)
    diagnostics = dict(diagnostics, nodes=len(mesh['points']), elements=len(mesh['tetra']), elapsed_s=time.monotonic() - started)
    probes = np.zeros((len(fields), len(taps), 3))
    for i, (index, weights) in enumerate(taps): probes[:, i] = np.einsum('a,tac->tc', weights, fields[:, mesh['tetra'][index]])
    computed = {'probes_m': probes, 'fields_m': fields if keep_fields else None, 'times_us': np.arange(len(fields)) * scene['acquisition']['sample_step_us'], 'diagnostics': diagnostics}
    return {'scene': scene, 'mesh': mesh, 'receivers': receivers, 'computed': computed, 'degree': None}


def write_outputs(run, output, write_csv=False):
    mesh = run['mesh']; run = dict(run, mesh_summary={'elements': len(mesh['tetra']), 'min_altitude_mm': mesh['min_altitude'] * 1e3})
    return write_run_outputs(run, output, SCHEMA, ENGINE, '3d', write_csv=write_csv)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scene', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--degree', type=int, help='ignored: linear tetrahedra'); parser.add_argument('--substeps', type=int, help='ignored'); parser.add_argument('--fields', action='store_true'); parser.add_argument('--csv', action='store_true')
    args = parser.parse_args(); started = time.monotonic(); raw = json.loads(args.scene.read_text(encoding='utf-8'))
    run = simulate(raw, keep_fields=args.fields, callback=lambda v: print(json.dumps(v), flush=True))
    result = write_outputs(run, args.output, write_csv=args.csv); d = result['diagnostics']
    print(json.dumps({'completed': True, 'output': str(args.output), 'elements': result['mesh']['elements'], 'internal_steps': d['internal_steps'], 'energy_balance_error': d['external_work_balance_relative'], 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__': main()
