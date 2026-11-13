"""Time layer plan: which time integration and layer spacing a version-2 record uses.

The frozen long-record layer is the trained symmetric three-stage composition (fourth order in time) at
Δt = CFL × smallest node spacing / 10000 m/s. That spacing is far inside the stability range, and a short
record does not need fourth order: the plan below chooses, for each record, the cheapest of

* `verlet2`      one velocity-Verlet stage per layer (second order, one element message per layer);
* `composition4` the trained three-stage composition (fourth order, three element messages per layer),

with the largest layer spacing that keeps both

* stability: Δt·ω_max ≤ 0.75 × the scheme's linear stability limit (2 for one stage, about 1.57 for the
  composition). ω_max² is the largest eigenvalue of M⁻¹K of this mesh and material, measured with a Lanczos
  iteration on the linear element message, so it follows the real element sizes, curved elements and the
  material's own speeds instead of a wave-speed bound;
* accuracy: the phase error of a linear wave at the highest second-harmonic frequency of the sources (twice the
  −30 dB band edge, the frequency the automatic mesh resolves) that travels during the whole record stays below
  `PHASE_TOLERANCE_RAD`. The phase error of each scheme is evaluated exactly from its amplification matrix,
  not from a leading-order estimate.

Both conditions are worst cases: a wave that leaves through an absorbing layer travels for less than the record.
Records with contact interfaces keep the frozen layer, because the penalty stiffness of the interface is tied to
the layer spacing (`graphnet.contact`). The plan is never dearer than the frozen layer: when the conditions would need
more element messages than it (very long records), the frozen layer is kept and its own phase-error bound is
reported. `mode='frozen'` always returns the frozen layer.
"""
import math
import numpy as np
from solid.schema import require

VERLET = (1.,)
STABILITY_FRACTION = .75
PHASE_TOLERANCE_RAD = .01
MARGIN = 1.03          # on the Lanczos estimate of ω_max² (a Ritz value is a lower bound)
MODES = ('auto', 'frozen')


def _stage(y):
    """One velocity-Verlet layer of the oscillator ẍ = −ω²x on (x, v/ω), y = ω·h."""
    return np.array([[1 - y * y / 2, y], [-y * (1 - y * y / 4), 1 - y * y / 2]])


def _half_trace(composition, y):
    product = np.eye(2)
    for c in composition: product = _stage(c * y) @ product
    return float(product[0, 0] + product[1, 1]) / 2


def phase_error(composition, y):
    """Relative phase error |θ/y − 1| of one layer for y = ω·Δt (inf when the layer is unstable)."""
    if y <= 0: return 0.
    half = _half_trace(composition, y)
    if abs(half) > 1: return math.inf
    return abs(math.acos(half) / y - 1)


_LIMITS = {}


def stability_limit(composition):
    """Largest x with the layer stable for every ω·Δt in (0, x]."""
    key = tuple(float(c) for c in composition)
    if key not in _LIMITS:
        step = 1e-3; y = step
        while y < 4 and abs(_half_trace(key, y)) <= 1 + 1e-12: y += step
        lo, hi = y - step, y
        for _ in range(60):
            mid = (lo + hi) / 2
            if abs(_half_trace(key, mid)) <= 1 + 1e-12: lo = mid
            else: hi = mid
        _LIMITS[key] = lo
    return _LIMITS[key]


def band_top_mhz(scene):
    """Highest second-harmonic frequency a record must resolve: twice the −30 dB band edge of every source (at least 2f)."""
    from physics.scene import sources
    from physics.signals import bandwidth_mhz
    top = 0.
    for src in sources(scene):
        f = float(src.get('frequency_mhz') or .2)
        try: top = max(top, 2 * f, 2 * bandwidth_mhz(src, 30.))
        except Exception: top = max(top, 2 * f)
    return top


def largest_frequency(mesh, derivative, inverse, quadratic, cubic, free, iterations=80, tolerance=1e-4):
    """ω_max (rad/s) of the linear network on this mesh: √λ_max(M⁻¹K) on the unconstrained components.

    Lanczos on M^(−1/2) K M^(−1/2) with the linear element message (no reorthogonalisation: three vectors of memory)."""
    from graphwave.buffered_inference import graph_message
    scale = 1 / np.sqrt(np.asarray(mesh['mass']))[:, None] * free; cells = mesh['elements']; volume = mesh['volume_weight']
    def apply(v):
        return graph_message(np.ascontiguousarray(v * scale), cells, derivative, inverse, volume, quadratic, cubic, 0)[0] * scale
    v = np.random.default_rng(20261113).standard_normal(scale.shape) * free; v /= np.linalg.norm(v)
    previous = np.zeros_like(v); beta = 0.; alphas, betas = [], []; estimate = 0.
    for k in range(iterations):
        w = apply(v) - beta * previous; alpha = float(np.sum(w * v)); w -= alpha * v
        alphas.append(alpha); beta = float(np.linalg.norm(w))
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1); top = float(np.linalg.eigvalsh(T)[-1])
        if k >= 8 and abs(top - estimate) <= tolerance * top: estimate = top; break
        estimate = top
        if beta <= 1e-14 * abs(top): break
        betas.append(beta); previous = v; v = w / beta
    require(np.isfinite(estimate) and estimate > 0, '无法估计网格的最高固有频率。')
    return math.sqrt(estimate * MARGIN)


def frozen_plan(scene, mesh, composition, speed):
    """The trained layer: three-stage composition, Δt from CFL × smallest node spacing / the trained speed bound."""
    cfl = scene['numerics']['cfl']; require(.05 <= cfl <= .3, '网络层间距的验证范围为 CFL 0.05–0.3。')
    sample = scene['acquisition']['sample_step_us'] * 1e-6
    subdivisions = int(math.ceil(sample / (cfl * float(mesh['min_node_spacing']) / speed)))
    return {'mode': 'frozen', 'scheme': 'composition4', 'composition': [float(c) for c in composition], 'subdivisions': subdivisions, 'layer_step_us': sample / subdivisions * 1e6,
            'time_order': 4, 'reason': '训练时的三级四阶时间层（CFL × 最小节点间距 / 10000 m/s）'}


def _subdivisions(composition, sample, duration, omega_max, omega_top):
    """Fewest layers per sample that satisfy the stability and phase conditions."""
    limit = STABILITY_FRACTION * stability_limit(composition); n = max(1, int(math.ceil(sample * omega_max / limit - 1e-12)))
    while omega_top * duration * phase_error(composition, omega_top * sample / n) > PHASE_TOLERANCE_RAD: n = max(n + 1, int(math.ceil(n * 1.02)))
    return n


def plan(scene, mesh, composition, speed, omega_max=None, mode='auto'):
    """Layer plan for a record. `omega_max` (rad/s) from `largest_frequency`; it is needed for mode 'auto'."""
    require(mode in MODES, '时间层设置需为 auto 或 frozen。')
    frozen = frozen_plan(scene, mesh, composition, speed)
    acquisition = scene['acquisition']; sample = acquisition['sample_step_us'] * 1e-6; duration = acquisition['duration_us'] * 1e-6
    frames = round(acquisition['duration_us'] / acquisition['sample_step_us'])
    frozen['evaluations_per_control'] = frames * frozen['subdivisions'] * 3
    if mode == 'frozen': return frozen
    if 'contact_a' in mesh and len(mesh['contact_a']):
        return dict(frozen, mode='auto', reason='含接触界面：界面罚刚度与层间距绑定，沿用训练时的时间层')
    require(omega_max is not None and omega_max > 0, '自动时间层需要网格的最高固有频率。')
    omega_top = 2 * math.pi * band_top_mhz(scene) * 1e6; best = None
    measured = lambda weights, n: {'stability_number': omega_max * sample / n, 'stability_limit': stability_limit(weights),
                                   'phase_error_bound_rad': omega_top * duration * phase_error(weights, omega_top * sample / n),
                                   'omega_max_rad_s': omega_max, 'band_top_mhz': band_top_mhz(scene), 'phase_tolerance_rad': PHASE_TOLERANCE_RAD}
    for scheme, weights, order in (('verlet2', VERLET, 2), ('composition4', tuple(float(c) for c in composition), 4)):
        n = _subdivisions(weights, sample, duration, omega_max, omega_top); cost = frames * n * len(weights)
        row = {'mode': 'auto', 'scheme': scheme, 'composition': list(weights), 'subdivisions': n, 'layer_step_us': sample / n * 1e6, 'time_order': order,
               'evaluations_per_control': cost, **measured(weights, n)}
        if best is None or cost < best['evaluations_per_control']: best = row
    if best['evaluations_per_control'] >= frozen['evaluations_per_control']:
        # Never dearer than the trained layer (the behaviour before this plan existed); its own numbers are reported with it.
        return dict(frozen, mode='auto', reason='训练时的时间层已是最省的选择', **measured(tuple(frozen['composition']), frozen['subdivisions']))
    best.update(frozen_evaluations_per_control=frozen['evaluations_per_control'], frozen_layer_step_us=frozen['layer_step_us'],
                reason=('二阶单级时间层' if best['scheme'] == 'verlet2' else '三级四阶时间层') +
                       f'，层间距按实测稳定极限的 {STABILITY_FRACTION:g} 倍与相位误差上界 {PHASE_TOLERANCE_RAD:g} rad 选取')
    return best
