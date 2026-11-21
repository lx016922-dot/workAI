"""Material damping: Rayleigh damping of the whole body, m ü + (α m + β K) u̇ + f(u) = F.

A scene (version 2) may carry

    "damping": {"mass_per_s": α, "stiffness_s": β}

Every wave of angular frequency ω then decays in time as exp(−γ t) with

    γ(ω) = α/2 + β ω²/2          (amplitude; valid while γ ≪ ω)

whatever its mode, because the damping matrix is a combination of the mass and stiffness matrices. Along its path a
guided mode loses γ/c_g nepers per metre, c_g its group velocity. The loss factor (1/Q) is η(ω) = 2γ/ω = α/ω + βω.

Two numbers describe the material, so the decay can be set at two frequencies — for second-harmonic work at f and 2f,
which is what decides how far the second harmonic accumulates (`from_decay`). Between and beyond them the model
follows α/2 + βω²/2, which is not a law of the material: mass-proportional damping alone decays every frequency at the
same rate, stiffness-proportional damping alone (a Kelvin–Voigt solid) as ω².

In the nonlinear controls the stiffness-proportional term acts through the tangent stiffness (`graphnet.recurrence`).
"""

import math
from solid.schema import require, number, exact

ALPHA_MAX = 1e7  # 1/s
BETA_MAX = 1e-6  # s
NEPER_DB = 20 / math.log(10)


def validate_damping(raw):
    """Normalized damping or None when it is absent or zero."""
    if raw is None:
        return None
    require(isinstance(raw, dict), '材料阻尼需要是对象：{"mass_per_s": α, "stiffness_s": β}。')
    exact(raw, ['mass_per_s', 'stiffness_s'], '材料阻尼')
    alpha = number(raw['mass_per_s'], 0, ALPHA_MAX, '质量比例阻尼系数 α（1/s）')
    beta = number(raw['stiffness_s'], 0, BETA_MAX, '刚度比例阻尼系数 β（s）')
    return {'mass_per_s': alpha, 'stiffness_s': beta} if alpha > 0 or beta > 0 else None


def coefficients(scene):
    """(α in 1/s, β in s) of a scene; zeros without damping."""
    d = scene.get('damping') or {}
    return float(d.get('mass_per_s', 0.0)), float(d.get('stiffness_s', 0.0))


def decay_per_s(damping, frequency_hz):
    """Amplitude decay rate γ (1/s) at a frequency."""
    if not damping:
        return 0.0
    omega = 2 * math.pi * frequency_hz
    return damping['mass_per_s'] / 2 + damping['stiffness_s'] * omega * omega / 2


def attenuation_np_m(damping, frequency_hz, group_velocity_m_s):
    """Nepers per metre of a wave that travels with this group velocity."""
    return decay_per_s(damping, frequency_hz) / group_velocity_m_s if group_velocity_m_s and group_velocity_m_s > 0 else None


def from_decay(first, second=None):
    """Damping from the decay rate at one or two frequencies: (frequency in Hz, γ in 1/s).

    With one point the damping is stiffness-proportional when `second` is None. With two, α and β are the unique pair that
    gives both rates; they must come out non-negative, which needs 1 ≤ γ₂/γ₁ ≤ (f₂/f₁)² for f₂ > f₁."""
    f1, g1 = first
    require(f1 > 0 and g1 >= 0, '衰减点需要正的频率和非负的衰减率。')
    if second is None:
        return validate_damping({'mass_per_s': 0.0, 'stiffness_s': 2 * g1 / (2 * math.pi * f1) ** 2})
    f2, g2 = second
    require(f2 > 0 and g2 >= 0 and abs(f2 - f1) > 1e-9 * f1, '两个衰减点需要不同的正频率。')
    w1, w2 = (2 * math.pi * f1) ** 2, (2 * math.pi * f2) ** 2
    beta = 2 * (g2 - g1) / (w2 - w1)
    alpha = 2 * g1 - beta * w1
    tolerance = 1e-9 * max(g1, g2, 1e-300)
    require(
        beta >= -tolerance / max(w1, w2) and alpha >= -tolerance,
        '这两个衰减率不能由质量比例加刚度比例阻尼得到：频率高的一个的衰减率需要在低的一个的 1 倍到（频率比）² 倍之间。',
    )
    return validate_damping({'mass_per_s': max(alpha, 0.0), 'stiffness_s': max(beta, 0.0)})


def only_mass(excitation, solver):
    """For the solvers that take the nodal coefficient α only: refuse a stiffness-proportional term instead of ignoring it."""
    require(not (excitation or {}).get('beta'), f'{solver}只支持质量比例的材料阻尼；刚度比例阻尼（stiffness_s）请用桌面软件的长记录计算。')


def describe(damping, frequencies_hz=()):
    """JSON-ready summary: the coefficients and, at the given frequencies, γ, the loss factor and Q."""
    if not damping:
        return None
    rows = []
    for f in frequencies_hz:
        g = decay_per_s(damping, f)
        eta = 2 * g / (2 * math.pi * f)
        rows.append({'frequency_mhz': f * 1e-6, 'decay_per_s': g, 'loss_factor': eta, 'q': 1 / eta if eta > 0 else None})
    return {'model': 'rayleigh', 'mass_per_s': damping['mass_per_s'], 'stiffness_s': damping['stiffness_s'], 'at': rows}
