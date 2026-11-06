"""Frozen weights for the version-2 input layer, verified against the offline package."""
import hashlib
import json
from pathlib import Path
import numpy as np
from solid.schema import require

MODEL = Path(__file__).resolve().parents[1] / 'offline' / 'model'
MATERIAL_KEYS = ('rho_kg_m3', 'lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa')
WEIGHT_KEYS = ('quadratic_tensor', 'cubic_tensor', 'state_weights')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _murnaghan_stress(material, z):
    """Second-order (quadratic in strain) stress of the training labels, Mandel form, for z = strain / 1e-4."""
    s = np.sqrt(2.); e = np.array([[z[0], z[5] / s, z[4] / s], [z[5] / s, z[1], z[3] / s], [z[4] / s, z[3] / s, z[2]]]) * 1e-4
    a = material['n_pa']; b = material['m_pa'] - a / 2; c = material['l_pa'] - material['m_pa'] + a / 2
    tr = np.trace(e); e2 = e @ e; m = (b * np.trace(e2) + c * tr ** 2) * np.eye(3) + 2 * b * tr * e + a * e2
    return np.array([m[0, 0], m[1, 1], m[2, 2], s * m[1, 2], s * m[0, 2], s * m[0, 1]])


def material_check(weights, material, tolerance=1e-9):
    """Largest relative mismatch between the energy network and a material's elastic and Murnaghan stresses."""
    lam, mu = material['lambda_pa'], material['mu_pa']
    elastic = lam * np.outer([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]) + 2 * mu * np.eye(6)
    linear = float(np.max(np.abs(2e11 * weights['quadratic_tensor'] - elastic)) / np.max(np.abs(elastic)))
    rng = np.random.default_rng(20261024); worst = 0.
    for _ in range(64):
        z = rng.normal(size=6); exact = _murnaghan_stress(material, z); learned = 3e3 * np.einsum('abc,b,c->a', weights['cubic_tensor'], z, z)
        worst = max(worst, float(np.max(np.abs(learned - exact)) / np.max(np.abs(exact))))
    return {'linear_relative': linear, 'murnaghan_relative': worst, 'pass': linear <= tolerance and worst <= tolerance}


def load(degree, model_dir=MODEL, long_record=False):
    """Weights able to run `degree`: the original 3/4 weights, or the continued weights for degree 5 and long records."""
    model_dir = Path(model_dir); info = json.loads((model_dir / 'model.json').read_text(encoding='utf-8'))
    extension = json.loads((model_dir / 'extension.json').read_text(encoding='utf-8')) if (model_dir / 'extension.json').exists() else None
    use_continued = long_record or degree == 5
    if use_continued:
        require(extension is not None, '软件包缺少续训权重，无法使用五阶或长记录。')
        path = model_dir / extension['weights_file']; require(digest(path) == extension['files_sha256'][extension['weights_file']], '续训权重校验失败。')
    else:
        path = model_dir / 'weights.npz'; require(digest(path) == info['weights_sha256'], '原始权重校验失败。')
    with np.load(path, allow_pickle=False) as saved: weights = dict(saved)
    require(f'derivative_{degree}' in weights, f'冻结网络没有 {degree} 阶图层；可用阶次：{sorted(int(k.split("_")[1]) for k in weights if k.startswith("derivative_"))}。')
    require(all(np.isfinite(v).all() for v in weights.values()) and all(k in weights for k in WEIGHT_KEYS), '网络权重无效。')
    material = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))['material']
    temporal = None
    if long_record:
        tpath = model_dir / extension['time_weights_file']; require(digest(tpath) == extension['files_sha256'][extension['time_weights_file']], '时间层权重校验失败。')
        temporal = json.loads(tpath.read_text(encoding='utf-8'))
    return {'weights': weights, 'weights_file': path.name, 'weights_sha256': digest(path), 'material': material, 'temporal': temporal,
            'time_weights_sha256': digest(model_dir / extension['time_weights_file']) if long_record else None, 'material_check': material_check(weights, material)}


TRAINED_SPEED_BOUND_M_S = 10000.


def energy_layer(scene, model):
    """(quadratic, cubic, info) for the energy layer.

    The trained material (legacy scenes without `material.model`) uses the trained tensors unchanged. A scene
    material (`physics.materials`) supplies its own second- and third-order constants in the same Mandel layout:
    quadratic = c2 / 2e11, cubic = c3 / 6e11 (see graphwave.inference.energy_activation). The learned derivative
    filters and time composition are material independent and stay as trained."""
    from physics.materials import is_custom, tensors
    weights = model['weights']; material = scene['material']
    if not is_custom(material):
        return weights['quadratic_tensor'], weights['cubic_tensor'], {'energy_layer': 'trained', 'material_model': 'reference_aluminium'}
    c2, c3, _ = tensors(material)
    return (np.ascontiguousarray(c2 / 2e11), np.ascontiguousarray(c3 / 6e11),
            {'energy_layer': 'scene_material_constants', 'material_model': material['model'], 'material_name': material.get('name'),
             'note': '能量层使用场景材料的二阶、三阶弹性常数；求导滤波器与时间层沿用训练结果。'})


def speed_bound(scene):
    """Wave speed used for the layer spacing: the trained 10000 m/s, or a faster material's fastest phase speed,
    so the Courant number stays inside the validated range."""
    from physics.materials import is_custom, speed_bounds
    if not is_custom(scene['material']): return TRAINED_SPEED_BOUND_M_S
    return max(TRAINED_SPEED_BOUND_M_S, speed_bounds(scene['material'])[0])
