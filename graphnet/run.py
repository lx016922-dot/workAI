"""Run the frozen graph network on one scene (version 1 or 2) with the sweep/run output layout.

    python -m graphnet.run --scene scene.json --output out-dir [--degree 3] [--laws all|scene] [--fields] [--csv]

The main result is the scene's constitutive law; with `--laws all` (default) the
linear and StVK controls are also computed and saved to controls.npz, as the
offline software does. No numerical solver module is imported; the process
checks this before writing results.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from solid.schema import require
from physics.receivers import receiver_points
from physics.runs import write_run_outputs
from spectral3d.mesh import make_mesh, interpolation
from .model import load, MODEL
from .inputs import network_scene
from .predict import predict, LAWS, FORBIDDEN

SCHEMA = 'workai.graphnet.run.result.v1'
ENGINE = 'workai_graphwave_v2_input'


def simulate(raw, degree=3, laws='all', callback=None, model_dir=MODEL):
    model = load(degree, model_dir); require(model['material_check']['pass'], '冻结权重与训练材料不一致。')
    scene = network_scene(raw, degree, model['material']); mesh = make_mesh(scene, degree, scene['numerics']['h_mm'])
    receivers = receiver_points(scene); taps = [interpolation(mesh, rx['point_mm']) for rx in receivers]
    main = LAWS.index(scene['constitutive']); selected = (0, 1, 2) if laws == 'all' else (main,)
    started = time.monotonic(); predicted = predict(scene, mesh, model, selected, callback)
    probes = {name: np.stack([np.einsum('n,tnc->tc', w, field[:, ids]) for ids, w in taps], 1) if taps else np.zeros((len(field), 0, 3)) for name, field in predicted['fields'].items()}
    d = dict(predicted['diagnostics'][selected.index(main)], elapsed_s=time.monotonic() - started)
    computed = {'probes_m': probes[LAWS[main]], 'fields_m': predicted['fields'][LAWS[main]], 'times_us': np.arange(len(probes[LAWS[main]])) * scene['acquisition']['sample_step_us'], 'diagnostics': d}
    return {'scene': scene, 'mesh': mesh, 'receivers': receivers, 'computed': computed, 'degree': degree, 'model': model, 'route': predicted['route'], 'probes': probes, 'controls': predicted['diagnostics']}


def write_outputs(run, output, write_csv=False, keep_fields=False):
    mesh = run['mesh']; model = run['model']; output = Path(output)
    computed = dict(run['computed'], fields_m=run['computed']['fields_m'] if keep_fields else None)
    controls = [{k: v for k, v in d.items() if k not in ('energy_samples',)} for d in run['controls']]
    extra = {'route': run['route'], 'neural_model': {'weights_file': model['weights_file'], 'weights_sha256': model['weights_sha256'], 'material_check': model['material_check'],
             'neural_forward_only': True, 'numerical_correction': False, 'solver_modules_absent': list(FORBIDDEN)}, 'controls': controls}
    result = write_run_outputs(dict(run, computed=computed, mesh_summary={'elements': len(mesh['elements']), 'degree': mesh['degree'], 'h_mm': mesh['h_mm'], 'curved': str(mesh.get('mapping_kind', '')) == 'isoparametric'}),
                               output, SCHEMA, ENGINE, '3d', write_csv=write_csv, extra=extra, engine_scope='frozen_graph_network_forward_only')
    np.savez_compressed(output / 'controls.npz', times_us=computed['times_us'], receiver_names=np.array([rx['name'] for rx in run['receivers']]), **{f'{law}_probes_m': p for law, p in run['probes'].items()})
    import hashlib
    result['files']['controls.npz'] = hashlib.sha256((output / 'controls.npz').read_bytes()).hexdigest()
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scene', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--laws', choices=['all', 'scene'], default='all'); parser.add_argument('--substeps', type=int, help='ignored: explicit recurrence')
    parser.add_argument('--fields', action='store_true'); parser.add_argument('--csv', action='store_true')
    args = parser.parse_args(); started = time.monotonic(); raw = json.loads(args.scene.read_text(encoding='utf-8'))
    run = simulate(raw, args.degree, args.laws, callback=lambda d: print(json.dumps({k: d[k] for k in ('law', 'recurrent_layers', 'elapsed_s', 'external_work_balance_relative')}), flush=True))
    loaded = [name for name in FORBIDDEN if name in sys.modules]
    if loaded: raise RuntimeError(f'数值求解模块进入了网络推理进程：{loaded}')
    result = write_outputs(run, args.output, write_csv=args.csv, keep_fields=args.fields); d = result['diagnostics']
    print(json.dumps({'completed': True, 'output': str(args.output), 'route': run['route'], 'nodes': result['mesh']['nodes'], 'internal_steps': d['internal_steps'],
                      'energy_balance_error': d['external_work_balance_relative'], 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__': main()
