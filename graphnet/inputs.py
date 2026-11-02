"""Scene → network input tensors (version-2 excitation channel)."""
import copy
import numpy as np
from solid.schema import require
from physics.scene import validate_scene, is_v2, is_plane_strain, spectral_admissible
from physics.signals import evaluate, bandwidth_mhz
from spectral3d.loads import prepare_excitation

KINDS = ('plate', 'pipe', 'hole', 'notch')


def network_scene(raw, degree, material):
    """Validated scene the frozen network can run: parametric GLL geometry, the network's own material, CFL ≤ 0.3."""
    require(isinstance(raw, dict) and isinstance(raw.get('numerics'), dict), '场景需要 numerics。')
    h = raw['numerics']['h_mm']; require(type(h) in (int, float) and np.isfinite(h) and .02 <= h <= 100, '单元尺寸必须在 0.02–100 mm 范围内。')
    require(type(degree) is int and 3 <= degree <= 5, '冻结网络的图阶次为 3、4 或 5。')
    s = copy.deepcopy(raw); s['numerics']['h_mm'] = max(.02, h / degree); scene = validate_scene(s); scene['numerics']['h_mm'] = h
    require(not is_plane_strain(scene), '冻结网络是三维图网络；二维平面应变请用 spectral2d。')
    require(scene['geometry']['kind'] in KINDS, '图网络的几何来自参数化 GLL 网格：板、管、通孔、缺口。')
    require(scene['numerics']['cfl'] <= .3, '网络层间距的验证上限为 CFL 0.3。')
    for key, value in material.items():
        require(abs(scene['material'][key] - value) <= 1e-9 * abs(value), f'场景材料 {key} 与网络训练材料不一致（网络只含该多晶铝的能量系数）。')
    if is_v2(scene): spectral_admissible(scene)
    else:
        require(scene['source']['face'] in ('xmin', 'xmax') and scene['source']['radius_mm'] == 0 and scene['boundary']['fixed_face'] in ('none', 'xmin', 'xmax'),
                '版本 1 场景沿用冻结网络规则：完整 X 端面加载，固定面可选另一 X 端面；其他加载请写成 v2 场景。')
    return scene


def build(scene, mesh):
    """Input tensors for a version-2 scene; None for version 1 (routed to the frozen predictor)."""
    excitation = prepare_excitation(mesh, scene)
    if excitation is None: return None
    loads = np.ascontiguousarray(np.stack([force for force, _ in excitation['loads']]))
    sources = [src for _, src in excitation['loads']]
    constrained = np.asarray(excitation['constrained'], dtype=bool)
    alpha = excitation['alpha']
    return {'loads': loads, 'sources': sources, 'free': np.ascontiguousarray((~constrained).astype(np.float64)), 'constrained': constrained,
            'alpha': None if alpha is None else np.ascontiguousarray(np.asarray(alpha, dtype=np.float64)), 'area_m2': excitation['area_m2'], 'summary': excitation['summary']}


def signal_table(inputs, times_us):
    """K × len(times) signal values; times may be any shape, the result keeps it after the source axis."""
    times = np.asarray(times_us, dtype=np.float64)
    return np.ascontiguousarray(np.stack([evaluate(src, times.ravel()).reshape(times.shape) for src in inputs['sources']]))


def resolution(scene, mesh):
    """Nodes per second-harmonic shear wavelength at each source's 30 dB band edge (information, not a gate)."""
    m = scene['material']; shear = np.sqrt(m['mu_pa'] / m['rho_kg_m3']); spacing = float(mesh['h_mm']) / int(mesh['degree'])
    from physics.scene import sources
    rows = []
    for src in sources(scene):
        top = bandwidth_mhz(src); rows.append({'name': src.get('name', 'primary'), 'band_upper_mhz': top, 'nodes_per_2f_shear_wavelength': shear / (2 * top * 1e6) / (spacing * 1e-3)})
    return rows
