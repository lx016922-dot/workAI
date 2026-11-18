"""Train the coarse-graph element operators against converged fine references.

    python -m coarsegraph.train --labels DIR [DIR ...] --splits coarsegraph/specs/splits.json \
        --degree 3 --h-mm 3 --output coarsegraph/models/p3-h3 [--iterations 30] [--workers 3]

Residuals per training pair (nonlinear case + its linear control, same scene):

    r_full = (N_coarse − N_fine) / ‖N_fine‖
    r_inc  = √w · ((N_coarse − L_coarse) − (N_fine − L_fine)) / ‖N_fine − L_fine‖

over every receiver and frame, so ‖r_full‖² is the relative L2² of the whole
nonlinear record and ‖r_inc‖² that of the nonlinear increment. Levenberg–Marquardt
with a forward-difference Jacobian (the operator has 6–16 parameters). After each
accepted step the validation pairs are scored with the same residual; the
parameter vector with the lowest validation loss is kept. Test pairs are only
evaluated once, after the parameters are frozen.

Every accepted step is saved to OUTPUT/checkpoint.json. Use the same command
with --resume (without --init) after an interruption; --iterations is the TOTAL
iteration target, not a number of additional steps. Once final testing has
completed, start a separate run with --init OUTPUT/model.json and a new output
directory instead of resuming model selection on the tested run.
"""
import argparse
import copy
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from solid.schema import require
from .operator import parameterization
from .simulate import Coarse, fine_projected, label_pairs
from .checkpoint import (atomic_json, load_checkpoint, operator_snapshot, restore_operators,
                         save_checkpoint, training_identity)

REPO = Path(__file__).resolve().parents[1]
FAILED = 10.

_contexts = {}
_weights = None


def _context(key, scene, degree, h_mm, cfl):
    global _weights
    if _weights is None:
        from graphnet.model import load
        _weights = load(3)['weights']
    if key not in _contexts: _contexts[key] = Coarse(scene, degree, h_mm, cfl)
    return _contexts[key]


def _task(job):
    """One coarse run in a worker: (projected waveforms or None, stability number)."""
    key, scene, degree, h_mm, cfl, theta, law, max_stability, axes = job
    c = _context((key, degree, h_mm, cfl), scene, degree, h_mm, cfl); P = parameterization(degree, axes)
    D, q, m = P.unpack(theta)
    if q.min() <= 0 or m.min() <= 0: return None, None
    D, volume, mass = c.operators(D, q, m)
    stability = c.stability(D, volume, mass, _weights)
    if not np.isfinite(stability) or stability > max_stability: return None, stability
    try: projected, _ = c.run(D, volume, mass, law, _weights)
    except ValueError: return None, stability
    if not np.all(np.isfinite(projected)): return None, stability
    return projected, stability


def harmonic_targets(pair, floor=.1, spread=(.9, 1., 1.1)):
    """Hann-weighted complex projections at f·spread (record) and 2f·spread (record and increment), with per-receiver scales.

    Each receiver is normalized by its own reference amplitude, but never by less than `floor` of the case's
    largest one, mirroring the gate metric's relative reporting floor."""
    t = np.asarray(pair['times']); w = np.hanning(len(t)); f = pair['frequency_mhz']; targets = {}
    for name, order, of in (('f_record', 1, 'record'), ('2f_record', 2, 'record'), ('2f_increment', 2, 'increment')):
        freqs = np.array(spread) * order * f; basis = (w[None, :] * np.exp(-2j * np.pi * freqs[:, None] * t[None, :])) * 2 / w.sum()
        fine = basis @ (pair['fine_N'] if of == 'record' else pair['fine_I']); amplitude = np.abs(fine[list(spread).index(1.)])
        scale = np.maximum(amplitude, floor * amplitude.max())[None, :]
        targets[name] = {'basis': basis, 'fine': fine, 'scale': scale, 'of': of, 'size': 2 * fine.size}
    return targets


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_source_hashes():
    """Bind cached residuals to the complete local forward/metric implementation."""
    modules = ('coarsegraph', 'graphnet', 'graphwave', 'physics', 'spectral3d', 'solid', 'dispersion')
    return {f'{module}/{path.name}': digest(path) for module in modules
            for path in sorted((REPO / module).glob('*.py')) if not path.name.startswith('test_')}


class Problem:
    def __init__(self, pairs, degree, h_mm, cfl, w_increment, pool, max_stability=1.5, axes='shared', w_harmonic=0., floor=.1):
        self.pairs = pairs; self.degree = degree; self.h_mm = h_mm; self.cfl = cfl; self.w = w_increment; self.pool = pool; self.max_stability = max_stability; self.axes = axes; self.wh = w_harmonic
        if w_harmonic:
            for p in pairs: p['harmonics'] = harmonic_targets(p, floor)
        self.parameterization = parameterization(degree, axes); self.evaluations = 0

    def simulate(self, thetas, pairs):
        jobs = [(p['key'], p['scene'], self.degree, self.h_mm, self.cfl, np.asarray(t, dtype=float), law, self.max_stability, self.axes) for t in thetas for p in pairs for law in ('linear', 'murnaghan')]
        out = list(self.pool.map(_task, jobs)); self.evaluations += len(thetas); results = []; at = 0
        for _ in thetas:
            row = []
            for _ in pairs: row.append({'linear': out[at][0], 'murnaghan': out[at + 1][0], 'stability': out[at + 1][1]}); at += 2
            results.append(row)
        return results

    def residual(self, sims, pairs):
        """Time-domain record and increment residuals, plus per-receiver harmonic residuals around f and 2f (gate-aligned)."""
        parts = []
        for sim, p in zip(sims, pairs):
            size = 2 * p['fine_N'].size + sum(h['size'] for h in p['harmonics'].values()) if self.wh else 2 * p['fine_N'].size
            if sim['linear'] is None or sim['murnaghan'] is None: parts.append(np.full(size, FAILED / np.sqrt(size))); continue
            full = (sim['murnaghan'] - p['fine_N']) / np.linalg.norm(p['fine_N'])
            inc = np.sqrt(self.w) * ((sim['murnaghan'] - sim['linear']) - p['fine_I']) / np.linalg.norm(p['fine_I'])
            parts += [full.ravel(), inc.ravel()]
            if self.wh:
                for name, h in p['harmonics'].items():
                    signal = sim['murnaghan'] if h['of'] == 'record' else sim['murnaghan'] - sim['linear']
                    d = (h['basis'] @ signal - h['fine']) / h['scale']
                    parts.append(np.sqrt(self.wh) * np.concatenate([d.real.ravel(), d.imag.ravel()]) / np.sqrt(d.size))
        return np.concatenate(parts)

    def residuals(self, thetas, pairs):
        return [self.residual(s, pairs) for s in self.simulate(thetas, pairs)]


def metrics(sims, pairs, floor=.1):
    """Gate metrics (relative reporting floor, 5 %) per pair: nonlinear waveform, f and 2f, and the increment's 2f."""
    from .evaluate import case_metrics
    rows = []
    for sim, p in zip(sims, pairs):
        row = {'key': p['key'], 'split': p['split'], 'parameters': p['parameters'], 'stability': sim['stability']}
        if sim['linear'] is None or sim['murnaghan'] is None: row.update(valid=False, passed=False)
        else: row.update(valid=True, **case_metrics(p['times'], p['frequency_mhz'], sim['murnaghan'], sim['linear'], p['fine_N'], p['fine_I'], floor))
        rows.append(row)
    return rows


def evaluate(problem, pairs, thetas, output):
    """Gate metrics for named parameter vectors on every pair; saves predictions.npz in `output`."""
    evaluation = {}; predictions = {}
    for name, theta in thetas.items():
        t = time.monotonic(); sims = problem.simulate([theta], pairs)[0]; elapsed = time.monotonic() - t
        evaluation[name] = {'rows': metrics(sims, pairs), 'wall_s_all_pairs': elapsed}
        for sim, p in zip(sims, pairs):
            for law in ('linear', 'murnaghan'):
                if sim[law] is not None: predictions[f'{name}|{p["key"]}|{law}'] = sim[law]
    np.savez_compressed(Path(output) / 'predictions.npz', **predictions)
    return evaluation


def summarize(evaluation):
    summary = {}
    for name, block in evaluation.items():
        def worst(key, s): return max((r[key] for r in block['rows'] if r['split'] == s and r['valid'] and key in r), default=None)
        summary[name] = {s: {'pairs': len([r for r in block['rows'] if r['split'] == s]), 'passed': sum(1 for r in block['rows'] if r['split'] == s and r['passed']),
                             'invalid': sum(1 for r in block['rows'] if r['split'] == s and not r['valid']),
                             'worst_record_l2': worst('record_l2', s), 'worst_waveform_l2': worst('worst_waveform_l2', s), 'worst_a1': worst('worst_a1', s), 'worst_a2': worst('worst_a2', s),
                             'worst_increment_a2': worst('worst_increment_a2', s), 'worst_a2_vector': worst('a2_vector_relative', s),
                             'worst_increment_a2_vector': worst('increment_a2_vector_relative', s)} for s in ('train', 'validation', 'test')}
    return summary


def reevaluate():
    """python -m coarsegraph.train reevaluate --model M --labels ... --splits S --output DIR: same metrics for a frozen model, without retraining."""
    parser = argparse.ArgumentParser(); parser.add_argument('command'); parser.add_argument('--model', type=Path, required=True); parser.add_argument('--labels', type=Path, nargs='+', required=True)
    parser.add_argument('--splits', type=Path, required=True); parser.add_argument('--workers', type=int, default=4); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); model = json.loads(args.model.read_text(encoding='utf-8'))
    pairs = load_pairs(args.labels, args.splits, model['degree'], model['h_mm'], model['cfl'])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        problem = Problem(pairs, model['degree'], model['h_mm'], model['cfl'], 1., pool, axes=model.get('axes', 'shared'))
        from .run import local_theta  # the stored operators in this machine's parameter basis (θ in the file is the training machine's)
        evaluation = evaluate(problem, pairs, {'baseline_gll': np.zeros(problem.parameterization.size), 'learned': local_theta(model)}, args.output)
    result = {'schema': 'workai.coarsegraph.evaluation.v1', 'model': str(args.model), 'model_sha256': digest(args.model), 'summary': summarize(evaluation), 'evaluation': evaluation,
              'source_sha256': {f'coarsegraph/{p.name}': digest(p) for p in sorted((REPO / 'coarsegraph').glob('*.py'))}, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (args.output / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['summary']), flush=True)


def load_pairs(label_dirs, splits, degree, h_mm, cfl):
    rules = json.loads(Path(splits).read_text(encoding='utf-8')); pairs = []
    for directory in label_dirs:
        for pair in label_pairs(directory):
            rule = rules.get(pair['campaign']); require(rule is not None, f'划分规则缺少 {pair["campaign"]}。')
            split = rule.get('all')
            for name in ('train', 'validation', 'test'):
                if split is None and any(all(abs(pair['parameters'][k] - v) < 1e-12 for k, v in item.items()) for item in rule.get(name, [])): split = name
            if split is None: continue
            scene = json.loads(Path(pair['murnaghan'], 'scene.json').read_text(encoding='utf-8'))
            from physics.receivers import receiver_points
            order = [rx['name'] for rx in receiver_points(scene)]
            N, times = fine_projected(Path(pair['murnaghan'], 'out'), order); L, linear_times = fine_projected(Path(pair['linear'], 'out'), order)
            require(N.shape == L.shape and np.array_equal(times, linear_times), '线性与非线性标签的接收点或时间轴不一致。')
            manifest = {d: json.loads((Path(pair[d]).parents[1] / 'manifest.json').read_text(encoding='utf-8')) for d in ('murnaghan',)}
            fine_cost = [c for c in manifest['murnaghan']['cases'] if c['case_id'] == Path(pair['murnaghan']).name][0]
            pairs.append(dict(pair, split=split, scene=scene, names=order, times=times, fine_N=N, fine_I=N - L, frequency_mhz=scene['source']['frequency_mhz'],
                              fine_nodes=fine_cost['nodes'], fine_layers=fine_cost['internal_steps'], fine_elapsed_s=fine_cost['elapsed_s'],
                              label_sha256={'linear': digest(Path(pair['linear'], 'out', 'result.json')), 'murnaghan': digest(Path(pair['murnaghan'], 'out', 'result.json'))},
                              input_sha256={law: {name: digest(Path(pair[law]) / name) for name in ('scene.json', 'out/result.json', 'out/lines.npz')}
                                            for law in ('linear', 'murnaghan')}))
    return pairs


def train(problem, train_pairs, validation_pairs, iterations, log, mu=1e-2, step=1e-6, theta0=None, stall=1e-2,
          resume=None, checkpoint=None):
    """Optimize up to a total iteration target; checkpoints never depend on test pairs.

    `resume` is the restored optimizer state from load_checkpoint. `checkpoint`
    receives a complete state at initialization and after each accepted step.
    """
    require(iterations >= 0, '总迭代上限不能为负。')
    require(resume is None or theta0 is None, '恢复检查点与初始参数不能同时提供。')
    n = problem.parameterization.size
    if resume is None:
        theta = np.zeros(n) if theta0 is None else np.asarray(theta0, dtype=float).copy()
        r = problem.residuals([theta], train_pairs)[0]; loss = float(r @ r)
        val = problem.residuals([theta], validation_pairs)[0]; vl = float(val @ val)
        best = {'theta': theta.copy(), 'validation_loss': vl, 'iteration': 0}
        history = [{'iteration': 0, 'train_loss': loss, 'validation_loss': vl, 'mu': mu, 'theta': theta.tolist(), 'evaluations': problem.evaluations}]
        log(history[-1]); iteration = 0; stalls = 0; stopped = None
    else:
        state = copy.deepcopy(resume); iteration = state['iteration']
        require(iterations >= iteration, '总迭代上限小于检查点迭代数。')
        theta = state['current']['theta']; r = state['current']['residual']; loss = state['current']['train_loss']
        vl = state['current']['validation_loss']; best = state['best']; history = state['history']
        mu = state['mu']; stalls = state['stalls']; stopped = state.get('stopped'); problem.evaluations = state['evaluations']
        if state.get('recompute_residual'):
            r = problem.residuals([theta], train_pairs)[0]; loss = float(r @ r)
        log({'resumed_iteration': iteration, 'evaluations': problem.evaluations,
             'recomputed_residual': bool(state.get('recompute_residual'))})

    def save():
        if checkpoint:
            checkpoint({'iteration': iteration, 'mu': mu, 'stalls': stalls, 'evaluations': problem.evaluations,
                        'stopped': stopped, 'current': {'theta': theta.copy(), 'train_loss': loss,
                                                       'validation_loss': vl, 'residual': r.copy()},
                        'best': copy.deepcopy(best), 'history': copy.deepcopy(history)})

    save()
    if stopped is not None: return best, history
    for it in range(iteration + 1, iterations + 1):
        h = step * np.maximum(1, np.abs(theta)); trial = [theta + np.eye(n)[k] * h[k] for k in range(n)]
        J = np.stack([(rk - r) / h[k] for k, rk in enumerate(problem.residuals(trial, train_pairs))], 1)
        A = J.T @ J; g = J.T @ r; accepted = False
        for _ in range(8):
            delta = np.linalg.solve(A + mu * np.diag(np.diag(A)) + 1e-12 * np.trace(A) / n * np.eye(n), -g)
            candidate = theta + delta; rc = problem.residuals([candidate], train_pairs)[0]; lc = float(rc @ rc)
            if lc < loss:
                improvement = (loss - lc) / loss; theta, r, loss = candidate, rc, lc; mu = max(mu / 3, 1e-8); accepted = True; break
            mu *= 4
        iteration = it
        if not accepted:
            stopped = 'no decrease after 8 damping increases'
            log({'iteration': it, 'stopped': stopped, 'mu': mu}); save(); break
        val = problem.residuals([theta], validation_pairs)[0]; vl = float(val @ val)
        if vl < best['validation_loss']: best = {'theta': theta.copy(), 'validation_loss': vl, 'iteration': it}
        history.append({'iteration': it, 'train_loss': loss, 'validation_loss': vl, 'mu': mu, 'theta': theta.tolist(), 'evaluations': problem.evaluations}); log(history[-1])
        stalls = stalls + 1 if improvement < stall else 0
        if stalls >= 2: stopped = f'relative train improvement below {stall:g} twice'
        save()
        if stopped is not None: log({'iteration': it, 'stopped': stopped}); break
    return best, history


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--labels', type=Path, nargs='+', required=True); parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--degree', type=int, default=3); parser.add_argument('--h-mm', type=float, default=3.); parser.add_argument('--cfl', type=float, default=.3)
    parser.add_argument('--increment-weight', type=float, default=1.)
    parser.add_argument('--iterations', type=int, default=30, help='total iteration target, including iterations before --resume (default 30)')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--harmonic-weight', type=float, default=0., help='weight of the per-receiver f/2f residuals (0 = time-domain loss only)')
    parser.add_argument('--stall', type=float, default=1e-2, help='stop after two accepted steps that each improve the training loss by less than this fraction')
    parser.add_argument('--axes', choices=['shared', 'per_axis'], default='shared')
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--init', type=Path, help='start an independent run from model.json (shared parameters are copied to every axis)')
    start.add_argument('--resume', action='store_true', help='resume OUTPUT/checkpoint.json after interruption; completed test runs require --init and a new output directory')
    parser.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv); started = time.monotonic()
    require(args.iterations >= 0 and args.workers > 0, '迭代数必须非负，进程数必须为正。')
    require(all(np.isfinite(v) and v >= 0 for v in (args.increment_weight, args.harmonic_weight, args.stall)), '损失权重和停止阈值必须是非负有限值。')
    checkpoint_path = args.output / 'checkpoint.json'
    if args.resume: require(checkpoint_path.is_file(), '--resume 需要输出目录中的 checkpoint.json。')
    else:
        require(not any((args.output / name).exists() for name in ('checkpoint.json', 'model.json', 'evaluation.json', 'training-log.jsonl')),
                '输出目录已有训练结果；中断任务使用 --resume，独立训练请换新目录。')
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.labels, args.splits, args.degree, args.h_mm, args.cfl)
    split = {s: [p for p in pairs if p['split'] == s] for s in ('train', 'validation', 'test')}
    require(split['train'] and split['validation'] and split['test'], '训练、验证、测试集都不能为空。')
    config = {'degree': args.degree, 'h_mm': args.h_mm, 'cfl': args.cfl, 'axes': args.axes,
              'increment_weight': args.increment_weight, 'harmonic_weight': args.harmonic_weight, 'stall': args.stall,
              'difference_step': 1e-6, 'initial_mu': 1e-2, 'max_stability': 1.5, 'harmonic_floor': .1,
              'energy_weights_sha256': digest(REPO / 'offline/model/weights.npz'),
              'network_metadata_sha256': {name: digest(REPO / 'offline/model' / name) for name in ('model.json', 'config.json')},
              'source_sha256': training_source_hashes()}
    identity = training_identity(config, pairs, json.loads(args.splits.read_text(encoding='utf-8')))
    P = parameterization(args.degree, args.axes)
    resumed = load_checkpoint(checkpoint_path, identity, P, args.iterations) if args.resume else None
    metadata = copy.deepcopy(resumed['metadata']) if resumed else {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'resume_count': 0}
    prior_elapsed = metadata.get('elapsed_s', 0.)
    if resumed: metadata['resume_count'] = metadata.get('resume_count', 0) + 1
    initial_model = resumed['initial_model'] if resumed else None
    latest = {'state': resumed['state']} if resumed else {}
    log_path = args.output / 'training-log.jsonl'
    def log(row):
        print(json.dumps(row), flush=True)
        with log_path.open('a', encoding='utf-8') as handle: handle.write(json.dumps(row) + '\n')
    log({'pairs': {s: [p['key'] for p in v] for s, v in split.items()}, 'degree': args.degree, 'h_mm': args.h_mm, 'cfl': args.cfl,
         'training_identity_sha256': identity['sha256'], 'iterations_total_target': args.iterations})

    def persist(state, phase='training'):
        latest['state'] = state
        metadata['elapsed_s'] = prior_elapsed + time.monotonic() - started
        save_checkpoint(checkpoint_path, identity, state, P, phase=phase, iterations_target=args.iterations,
                        initial_model=initial_model, metadata=metadata)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        problem = Problem(pairs, args.degree, args.h_mm, args.cfl, args.increment_weight, pool, axes=args.axes, w_harmonic=args.harmonic_weight); theta0 = None
        if args.init:
            from .run import load_model, local_theta
            start, _ = load_model(args.init); require(start['degree'] == args.degree, '初始模型的阶次不同。')
            theta0 = local_theta(start); theta0 = problem.parameterization.from_shared(theta0) if args.axes == 'per_axis' and start.get('axes', 'shared') == 'shared' else theta0
            require(len(theta0) == problem.parameterization.size, '初始模型参数个数不符。')
            initial_model = {'path': str(args.init), 'sha256': digest(args.init), 'snapshot': operator_snapshot(P, theta0, args.axes),
                             'evaluation_scope': 'initial operators on this run\'s grid, time step and labels'}
            log({'initialized_from': initial_model['path'], 'initial_model_sha256': initial_model['sha256'], 'theta0': theta0.tolist()})
        if resumed and resumed['phase'] == 'frozen':
            best = resumed['state']['best']; history = resumed['state']['history']; problem.evaluations = resumed['state']['evaluations']
            log({'resuming_frozen_evaluation': True, 'selected_iteration': best['iteration']})
        else:
            best, history = train(problem, split['train'], split['validation'], args.iterations, log, theta0=theta0, stall=args.stall,
                                  resume=resumed['state'] if resumed else None, checkpoint=persist)
        frozen = best['theta']; persist(latest['state'], phase='frozen')
        candidates = {'baseline_gll': np.zeros(P.size), 'learned': frozen}
        if initial_model is not None: candidates['initial_model'] = restore_operators(initial_model['snapshot'], P, args.axes)[0]
        evaluation = evaluate(problem, pairs, candidates, args.output)
    D, q, m = P.unpack(frozen); probe = Coarse(pairs[0]['scene'], args.degree, args.h_mm, args.cfl)
    summary = summarize(evaluation)
    model = {'schema': 'workai.coarsegraph.model.v1', 'degree': args.degree, 'h_mm': args.h_mm, 'cfl': args.cfl, 'axes': args.axes, 'parameter_count': P.size, 'theta': frozen.tolist(),
             'initialized_from': initial_model['path'] if initial_model else None,
             'initial_model_sha256': initial_model['sha256'] if initial_model else None,
             'operators': {'derivative': D.tolist(), 'quadrature_weights': q.tolist(), 'mass_weights': m.tolist(), 'gll_weights': P.w.tolist(), 'gll_nodes': P.x.tolist()},
             'constraints': P.check(frozen), 'selected_iteration': best['iteration'], 'validation_loss': best['validation_loss'],
             'energy_network': 'frozen graph network quadratic/cubic tensors and state weights (offline/model/weights.npz)',
             'training': {'increment_weight': args.increment_weight, 'harmonic_weight': args.harmonic_weight, 'stall': args.stall,
                          'iterations_requested': args.iterations, 'iterations_run': latest['state']['iteration'], 'iterations_semantics': 'total target',
                          'function_evaluations': problem.evaluations, 'elapsed_s': prior_elapsed + time.monotonic() - started,
                          'resume_count': metadata['resume_count'], 'training_identity_sha256': identity['sha256'], 'checkpoint': 'checkpoint.json',
                          'initial_model_evaluation': initial_model['evaluation_scope'] if initial_model else None,
                          'labels': [{'key': p['key'], 'split': p['split'], 'label_sha256': p['label_sha256'], 'input_sha256': p['input_sha256']} for p in pairs], 'test_used_for_selection': False},
             'cost': {'coarse_nodes': len(probe.mesh['points']), 'coarse_layers': probe.steps, 'fine_nodes_layers_examples': [{'key': p['key'], 'nodes': p['fine_nodes'], 'layers': p['fine_layers'], 'elapsed_s': p['fine_elapsed_s']} for p in pairs[:3]]},
             'summary': summary, 'source_sha256': {f'coarsegraph/{p.name}': digest(p) for p in sorted((REPO / 'coarsegraph').glob('*.py'))},
             'created_at_utc': datetime.now(timezone.utc).isoformat(), 'accepted_for_solver_routing': False}
    atomic_json(args.output / 'model.json', model)
    atomic_json(args.output / 'evaluation.json', evaluation)
    atomic_json(args.output / 'history.json', history)
    persist(latest['state'], phase='completed')
    log({'done': True, 'summary': summary})


if __name__ == '__main__':
    import sys
    reevaluate() if len(sys.argv) > 1 and sys.argv[1] == 'reevaluate' else main()
