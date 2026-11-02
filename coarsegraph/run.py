"""Run a trained coarse-graph model on one version-2 scene with the sweep/run output layout.

    python -m coarsegraph.run --model coarsegraph/models/p3-h3/model.json --scene scene.json --output out [--laws all|scene] [--csv]

The scene is validated as written (its own mesh size must pass the usual
resolution screen); the model then runs it on its own coarse graph (degree,
element size and CFL stored in the model). The energy network and state update
are the frozen graph network's; the element operators are the learned ones.
No numerical solver module is imported.
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
from graphnet.model import load
from graphnet.inputs import network_scene
from graphnet.predict import FORBIDDEN
from .operator import parameterization
from .simulate import Coarse

SCHEMA = 'workai.coarsegraph.run.result.v1'
ENGINE = 'workai_coarse_graph_learned_operator'
LAWS = ('linear', 'stvk', 'murnaghan')


def load_model(path):
    model = json.loads(Path(path).read_text(encoding='utf-8')); require(model.get('schema') == 'workai.coarsegraph.model.v1', '不是粗网格算子模型文件。')
    P = parameterization(model['degree'], model.get('axes', 'shared')); D, q, m = P.unpack(np.asarray(model['theta']))
    require(np.allclose(D, model['operators']['derivative'], atol=1e-12) and np.allclose(q, model['operators']['quadrature_weights'], atol=1e-14) and np.allclose(m, model['operators']['mass_weights'], atol=1e-14),
            '模型文件中的算子与参数不一致。')
    check = P.check(model['theta']); require(check['positive'] and all(v < 1e-10 for k, v in check.items() if k != 'positive'), '模型参数违反结构约束。')
    return model, (D, q, m)


def simulate(model_path, raw, laws='all', callback=None):
    model, (D, q, m) = load_model(model_path); network = load(3); require(network['material_check']['pass'], '冻结能量网络与训练材料不一致。')
    scene = network_scene(raw, model['degree'], network['material'])
    c = Coarse(scene, model['degree'], model['h_mm'], model['cfl']); D, volume, mass = c.operators(D, q, m)
    stability = c.stability(D, volume, mass, network['weights']); require(stability <= 1.5, f'该场景在粗网格上的时间步不稳定（{stability:.2f}）。')
    main = LAWS.index(scene['constitutive']); selected = (0, 1, 2) if laws == 'all' else (main,); probes = {}; rows = []
    for law in selected:
        started = time.monotonic(); _, energy, p = c.run(D, volume, mass, law, network['weights'], probes_out=True); probes[LAWS[law]] = p
        stored = energy[:, 1] + energy[:, 2] + energy[:, 4]; balance = float(np.max(np.abs(stored - energy[:, 3])) / max(np.max(np.abs(energy[:, 3])), 1e-30))
        rows.append({'law': LAWS[law], 'elapsed_s': time.monotonic() - started, 'internal_steps': c.steps, 'internal_step_us': c.dt * 1e6, 'nodes': len(mass), 'elements': len(c.mesh['elements']),
                     'degree': model['degree'], 'h_mm': model['h_mm'], 'stability_number': stability, 'external_work_balance_relative': balance, 'energy_balance_within_1_percent': balance <= .01,
                     'max_strain': None, 'neural_forward_only': True, 'numerical_correction': False})
        if callback: callback(rows[-1])
    d = rows[selected.index(main)]
    computed = {'probes_m': probes[LAWS[main]], 'fields_m': None, 'times_us': c.times_us, 'diagnostics': d}
    return {'scene': scene, 'mesh': c.mesh, 'receivers': receiver_points(scene), 'computed': computed, 'degree': model['degree'], 'model': model, 'model_path': str(model_path), 'probes': probes, 'controls': rows}


def write_outputs(run, output, write_csv=False):
    model = run['model']; output = Path(output)
    extra = {'coarse_model': {'path': run['model_path'], 'degree': model['degree'], 'h_mm': model['h_mm'], 'cfl': model['cfl'], 'theta': model['theta'], 'summary': model['summary'],
                              'accuracy_note': 'learned coarse operator; accuracy is the held-out test result recorded in the model, not a bound for other scenes'},
             'controls': run['controls']}
    result = write_run_outputs(dict(run, mesh_summary={'elements': len(run['mesh']['elements']), 'degree': model['degree'], 'h_mm': model['h_mm']}), output, SCHEMA, ENGINE, '3d',
                               write_csv=write_csv, extra=extra, engine_scope='learned_coarse_graph_forward_only')
    np.savez_compressed(output / 'controls.npz', times_us=run['computed']['times_us'], **{f'{law}_probes_m': p for law, p in run['probes'].items()})
    import hashlib
    result['files']['controls.npz'] = hashlib.sha256((output / 'controls.npz').read_bytes()).hexdigest()
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', type=Path, required=True); parser.add_argument('--scene', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--laws', choices=['all', 'scene'], default='all'); parser.add_argument('--degree', type=int, help='ignored: the model fixes the degree')
    parser.add_argument('--substeps', type=int, help='ignored'); parser.add_argument('--fields', action='store_true', help='not supported; ignored'); parser.add_argument('--csv', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    run = simulate(args.model, json.loads(args.scene.read_text(encoding='utf-8')), args.laws, callback=lambda d: print(json.dumps({k: d[k] for k in ('law', 'internal_steps', 'elapsed_s')}), flush=True))
    loaded = [name for name in FORBIDDEN if name in sys.modules]
    if loaded: raise RuntimeError(f'数值求解模块进入了推理进程：{loaded}')
    result = write_outputs(run, args.output, args.csv); d = result['diagnostics']
    print(json.dumps({'completed': True, 'output': str(args.output), 'nodes': d['nodes'], 'internal_steps': d['internal_steps'], 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__': main()
