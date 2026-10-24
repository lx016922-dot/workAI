"""Predeclared, geometry-disjoint continuation benchmark and isolated jobs."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from spectral3d.campaign import atomic, now
from spectral3d.compare import digest
from spectral3d.dataset import plan as generate_scenes
from spectral3d.mesh import make_mesh, interpolation
from solid.schema import validate_scene


def identity(scene):
    return hashlib.sha256(json.dumps(scene['geometry'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_plan(args):
    args.output.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((args.model / 'checkpoint-freeze.json').read_text())
    assert digest(args.model / 'weights.npz') == freeze['weights_sha256']
    assert digest(args.model / 'best.pt') == freeze['checkpoint_sha256']
    repo = Path(__file__).resolve().parents[1]
    old = set()
    for path in [repo / 'offline/model/metrics.json',
                 Path('/workspace/workai-remote/spectral3d/datasets/aluminum16-20261023/planned-inputs.json')]:
        if path.exists():
            old.update(identity(c['scene']) for c in json.loads(path.read_text())['cases'])
    rng = np.random.default_rng(args.seed + 1)
    cases = []
    for case in generate_scenes(args.seed)['cases']:
        variant = int(case['case_id'].rsplit('-', 1)[-1])
        if variant not in (0, 3):
            continue
        scene = case['scene']
        kind = scene['geometry']['kind']
        cid = 'continued-' + kind + ('-free' if variant == 0 else '-fixed')
        scene['name'] = cid
        g = scene['geometry']
        mesh = make_mesh(scene, 5, 2. if kind == 'hole' else 1.4)
        # These locations depend only on geometry, before any response is read.
        while len(scene['receivers']) < 8:
            x = float(rng.uniform(.12, .88) * g['length_mm'])
            if kind == 'pipe':
                r = float(rng.uniform(g['inner_radius_mm'] + .1, g['outer_radius_mm'] - .1))
                theta = float(rng.uniform(0, 2 * np.pi))
                point = [x, float(r * np.cos(theta)), float(r * np.sin(theta))]
            else:
                point = [x, float(rng.uniform(.12, .88) * g['width_mm']),
                         float(rng.uniform(.12, .88) * g['thickness_mm'])]
            try:
                interpolation(mesh, point)
            except ValueError:
                continue
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            scene['receivers'].append({'name': 'P' + str(len(scene['receivers']) + 1),
                                       'point_mm': point, 'direction': direction.tolist()})
        scene = validate_scene(scene)
        group = identity(scene)
        assert group not in old
        h = 2. if kind == 'hole' else 1.4
        fine_h = 1.4 if kind == 'hole' else 1.
        cases.append({'case_id': cid, 'scene': scene, 'split': 'test', 'geometry_group_sha256': group,
                      'h_mm': h, 'fine_h_mm': fine_h, 'prediction_degrees': {'parent': 4, 'continued': 5},
                      'phase': 'planned'})
    assert len(cases) == 8 and len({c['geometry_group_sha256'] for c in cases}) == 8
    hashes = {}
    for module in ['graphwave', 'spectral3d', 'solid', 'operator3d', 'spectralop']:
        dest = args.output / 'source_snapshot' / module
        dest.mkdir(parents=True)
        for source in (repo / module).glob('*.py'):
            shutil.copyfile(source, dest / source.name)
            hashes[module + '/' + source.name] = digest(source)
    for name in ['inference.py', 'extended_predict.py', 'model.py']:
        assert digest(args.output / 'source_snapshot/graphwave' / name) == digest(args.model / 'source_snapshot/graphwave' / name)
    value = {'schema': 'workai.graphwave_continuation_plan.v1', 'created_at_utc': now(),
             'frozen_at_utc': freeze['frozen_at_utc'], 'seed': args.seed,
             'weights_sha256': freeze['weights_sha256'], 'checkpoint_sha256': freeze['checkpoint_sha256'],
             'parent_weights_sha256': digest(args.parent / 'weights.npz'),
             'test_used_for_weight_selection': False, 'prior_geometry_count': len(old),
             'threshold': .05, 'reporting_floor_m': 1e-13, 'receivers_per_case': 8,
             'comparison': 'Original p4 vs continued p5 at the same element size; higher p5 compute cost is reported.',
             'reference_protocol': 'p4 base, independently finer p4 field, p5 order check, and halved internal time step; no tolerance relaxation.',
             'reference_fallback': 'At most one predeclared h refinement (1.4->1.0 or 2.0->1.4) if reference checks fail; candidate weights never change.',
             'cases': cases, 'source_sha256': hashes,
             'scope': 'Additional bounded synthetic aluminum cases, not a universal 5% guarantee.'}
    assert value['created_at_utc'] > freeze['frozen_at_utc']
    atomic(args.output / 'planned-inputs.json', value)
    print(json.dumps({'cases': len(cases), 'receivers': sum(len(c['scene']['receivers']) for c in cases),
                      'weights_sha256': value['weights_sha256'], 'created_after_freeze': True}), flush=True)


def run_jobs(args):
    plan_path = args.output / 'planned-inputs.json'
    plan = json.loads(plan_path.read_text())
    assert digest(args.model / 'weights.npz') == plan['weights_sha256']
    assert digest(args.parent / 'weights.npz') == plan['parent_weights_sha256']
    for name, expected in plan['source_sha256'].items():
        assert digest(args.output / 'source_snapshot' / name) == expected
    buffered = args.stage == 'predictions' and args.engine == 'buffered'
    prefix = 'predictions-buffered' if buffered else args.stage
    execution = args.output / 'source_snapshot'
    if buffered:
        execution = args.output / 'buffered_source_snapshot'
        manifest_path = args.output / 'buffered-execution-freeze.json'
        if not execution.exists():
            evidence = args.model / 'verification/buffered-transient-equivalence.json'
            assert json.loads(evidence.read_text())['passed']
            hashes = {}
            repo = Path(__file__).resolve().parents[1]
            for module in ['graphwave', 'spectral3d', 'solid']:
                dest = execution / module
                dest.mkdir(parents=True)
                for file in (repo / module).glob('*.py'):
                    shutil.copyfile(file, dest / file.name)
                    hashes[module + '/' + file.name] = digest(file)
            atomic(manifest_path, {'created_at_utc': now(), 'source_sha256': hashes,
                   'transient_equivalence_sha256': digest(evidence), 'weights_changed': False,
                   'reason': 'Reuse pointwise buffers; all learned coefficients and recurrence retained.'})
        frozen = json.loads(manifest_path.read_text())
        for name, expected in frozen['source_sha256'].items():
            assert digest(execution / name) == expected
    status_path = args.output / (prefix + '-status.json')
    status = {'stage': args.stage, 'phase': 'running', 'planned_inputs_sha256': digest(plan_path),
              'started_at_utc': now(), 'cases': []}
    lock, hole_slot = threading.RLock(), threading.Semaphore(1)
    for case in plan['cases']:
        status['cases'].append({'case_id': case['case_id'], 'phase': 'waiting', 'completed_jobs': []})

    def publish():
        with lock:
            status['updated_at_utc'] = now()
            atomic(status_path, status)

    def work(pair):
        case, row = pair
        cid = case['case_id']
        kind = case['scene']['geometry']['kind']
        token = hole_slot if kind == 'hole' else threading.Semaphore(1)
        with token:
            try:
                row['phase'] = 'running'
                publish()
                h = case['h_mm']
                if args.stage == 'references':
                    jobs = [('base', 4, h, .3, 4, True), ('space', 4, case['fine_h_mm'], .3, 4, True),
                            ('degree', 5, h, .3, 4, False), ('time', 4, h, .15, 8, False)]
                    for label, p, size, cfl, sub, fields in jobs:
                        directory = args.output / 'references' / cid / label
                        row['current_job'] = label
                        publish()
                        if directory.exists():
                            assert json.loads((directory / 'manifest.json').read_text())['phase'] == 'completed', str(directory)
                        else:
                            directory.parent.mkdir(parents=True, exist_ok=True)
                            command = [sys.executable, '-m', 'spectral3d.campaign', '--campaign', str(plan_path),
                                       '--output', str(directory), '--cases', cid, '--degree', str(p), '--h', str(size),
                                       '--cfl', str(cfl), '--substeps', str(sub), '--integrator', 'newmark' if kind == 'hole' else 'verlet']
                            if fields:
                                command.append('--fields')
                            with directory.with_suffix('.log').open('x') as stream:
                                subprocess.run(command, cwd=args.output / 'source_snapshot',
                                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                                    stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=10800)
                        row['completed_jobs'].append(label)
                        publish()
                else:
                    for label, p, model in [('parent', 4, args.parent), ('continued', 5, args.model)]:
                        directory = args.output / prefix / cid / label
                        row['current_job'] = label
                        publish()
                        if directory.exists():
                            receipt = json.loads((directory / 'prediction.json').read_text())
                            assert digest(directory / 'fields.npz') == receipt['field_sha256']
                        else:
                            directory.parent.mkdir(parents=True, exist_ok=True)
                            request = {'scene': case['scene'], 'degree': p, 'h_mm': h, 'cfl': .3,
                                       'weights_sha256': digest(model / 'weights.npz')}
                            input_path = directory.with_suffix('.input.json')
                            atomic(input_path, request)
                            module = 'graphwave.buffered_predict' if buffered else ('graphwave.predict' if label == 'parent' else 'graphwave.extended_predict')
                            command = [sys.executable, '-m', module,
                                       '--input', str(input_path), '--weights', str(model / 'weights.npz'), '--output', str(directory)]
                            with directory.with_suffix('.log').open('x') as stream:
                                subprocess.run(command, cwd=execution,
                                    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                                    stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=10800)
                        row['completed_jobs'].append(label)
                        publish()
                row['phase'] = 'completed'
            except Exception as error:
                row.update(phase='failed', error=str(error))
            publish()
            print(json.dumps(row), flush=True)

    publish()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, zip(plan['cases'], status['cases'])))
    status['phase'] = 'completed' if all(c['phase'] == 'completed' for c in status['cases']) else 'completed_with_failures'
    publish()
    if status['phase'] != 'completed':
        raise SystemExit('Some campaign jobs failed; inspect retained logs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['plan', 'references', 'predictions'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026102417)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--engine', choices=['original', 'buffered'], default='original')
    args = parser.parse_args()
    make_plan(args) if args.stage == 'plan' else run_jobs(args)
