"""Portable, atomic checkpoints for the coarse-operator optimizer.

The physical operators are authoritative: null-space coordinates can change
between BLAS implementations. On the same basis we also retain the exact saved
coordinates and residual, so interruption does not perturb the next LM step.
"""
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from solid.schema import require

SCHEMA = 'workai.coarsegraph.training_checkpoint.v1'


def atomic_json(path, value):
    """Replace a complete JSON file; serialization or write failure keeps the old file."""
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def array_identity(value):
    array = np.ascontiguousarray(value)
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def training_identity(config, pairs, splits):
    """Bind the run to the actual inputs, including receiver arrays outside result.json.

Paths are excluded so an intact campaign can move to another machine. Pair
order is retained because it determines the optimizer's residual ordering.
"""
    labels = []
    for p in pairs:
        labels.append({'key': p['key'], 'split': p['split'], 'parameters': p['parameters'],
                       'scene': p['scene'], 'receiver_names': p['names'],
                       'frequency_mhz': p['frequency_mhz'],
                       'times': array_identity(p['times']), 'nonlinear': array_identity(p['fine_N']),
                       'increment': array_identity(p['fine_I']),
                       'label_sha256': p['label_sha256'], 'input_sha256': p.get('input_sha256', {})})
    value = {'config': config, 'splits': splits, 'labels': labels}
    return {'sha256': fingerprint(value), **value}


def operator_snapshot(P, theta, axes='shared'):
    D, q, m = P.unpack(theta)
    return {'degree': P.degree, 'axes': axes, 'theta': np.asarray(theta, dtype=float).tolist(),
            'operators': {'derivative': D.tolist(), 'quadrature_weights': q.tolist(), 'mass_weights': m.tolist()}}


def restore_operators(snapshot, P, axes='shared'):
    """Return local coordinates and whether their saved value is exactly reusable."""
    require(snapshot['degree'] == P.degree and snapshot.get('axes', 'shared') == axes,
            '检查点的算子阶次或方向模式不同。')
    from .run import local_theta
    ops = snapshot['operators']
    arrays = [np.asarray(ops[k], dtype=float) for k in ('derivative', 'quadrature_weights', 'mass_weights')]
    require(all(np.isfinite(a).all() for a in arrays), '检查点算子含非有限值。')
    check = P.check_operators(*arrays)
    require(check['positive'] and all(v < 1e-10 for k, v in check.items() if k != 'positive'),
            '检查点算子违反结构约束。')
    theta = local_theta(snapshot)
    saved = np.asarray(snapshot['theta'], dtype=float)
    same = saved.shape == (P.size,) and np.isfinite(saved).all()
    if same:
        same = all(np.array_equal(a, b) for a, b in zip(P.unpack(saved), arrays))
    # The inverse mapping introduces roundoff even on an unchanged machine.
    # Reuse exact coordinates only after verifying the actual saved operators.
    return (saved.copy() if same else theta), bool(same)


def save_checkpoint(path, identity, state, P, *, phase='training', iterations_target,
                    initial_model=None, metadata=None):
    require(phase in ('training', 'frozen', 'completed'), '未知检查点阶段。')
    axes = identity['config']['axes']; current = state['current']; best = state['best']
    value = {'schema': SCHEMA, 'identity': identity, 'phase': phase, 'iterations_target': iterations_target,
             'iteration': state['iteration'], 'mu': state['mu'], 'stalls': state['stalls'],
             'evaluations': state['evaluations'], 'stopped': state.get('stopped'),
             'current': {**operator_snapshot(P, current['theta'], axes),
                         'train_loss': current['train_loss'], 'validation_loss': current['validation_loss'],
                         'residual': np.asarray(current['residual'], dtype=float).tolist()},
             'best': {**operator_snapshot(P, best['theta'], axes), 'iteration': best['iteration'],
                      'validation_loss': best['validation_loss']},
             'history': state['history'], 'initial_model': initial_model, 'metadata': metadata or {}}
    atomic_json(path, value)


def load_checkpoint(path, identity, P, iterations_target):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require(value.get('schema') == SCHEMA, '不是粗网格训练检查点。')
    stored = value['identity']
    require(stored.get('sha256') == fingerprint({k: v for k, v in stored.items() if k != 'sha256'}),
            '检查点身份记录已损坏。')
    require(stored == identity, '检查点与当前训练配置、数据或划分不兼容；请使用新输出目录开始训练。')
    require(value['phase'] in ('training', 'frozen', 'completed'), '未知检查点阶段。')
    require(value['phase'] != 'completed',
            '此训练已冻结并完成测试，不能用 --resume 继续选参；请用 --init model.json 和新 --output 开始独立训练。')
    require(iterations_target >= value['iteration'], '--iterations 是总迭代上限，不能小于检查点迭代数。')
    if value['phase'] == 'frozen':
        require(iterations_target == value['iterations_target'],
                '检查点已冻结并开始最终评估，不能改变迭代上限；--resume 只会完成同一候选的评估。')
    axes = identity['config']['axes']
    current_theta, same = restore_operators(value['current'], P, axes)
    best_theta, _ = restore_operators(value['best'], P, axes)
    residual = np.asarray(value['current']['residual'], dtype=float)
    require(residual.ndim == 1 and residual.size > 0 and np.isfinite(residual).all(), '检查点训练残差无效。')
    require(np.isfinite(value['mu']) and value['mu'] > 0 and value['stalls'] >= 0,
            '检查点阻尼或停滞计数无效。')
    state = {'iteration': value['iteration'], 'mu': value['mu'], 'stalls': value['stalls'],
             'evaluations': value['evaluations'], 'stopped': value.get('stopped'),
             'current': {'theta': current_theta, 'train_loss': value['current']['train_loss'],
                         'validation_loss': value['current']['validation_loss'], 'residual': residual},
             'best': {'theta': best_theta, 'validation_loss': value['best']['validation_loss'],
                      'iteration': value['best']['iteration']},
             'history': copy.deepcopy(value['history']), 'recompute_residual': not same}
    initial = value.get('initial_model')
    if initial is not None: restore_operators(initial['snapshot'], P, axes)
    return {'state': state, 'phase': value['phase'], 'initial_model': initial, 'metadata': value.get('metadata', {})}
