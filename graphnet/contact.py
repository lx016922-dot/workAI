"""Unilateral normal contact on interface node pairs: closed cracks, closed delaminations, contact interfaces.

The mesh (`spectral3d.feature_mesh`) duplicates the nodes of an interface and lists the pairs (a on the − side,
b on the + side), the normal n (− to +) and the GLL surface weight A of each pair. With the normal gap
g = (u_b − u_a)·n the interface energy is

    U_c = Σ ½ k (min(g, 0))²,        k = K·A,

so compression (g < 0, the faces pressed into each other) is carried by a stiff penalty spring and tension
(g > 0) opens the interface with zero traction: the "breathing" / clapping nonlinearity of a closed crack.
Tangential motion is free (frictionless). The force −∂U_c/∂u joins the frozen network's element messages;
U_c joins the strain energy, so the energy balance still closes.

The stiffness per area K is as large as the explicit layer allows: the target 10·M/s (M the P-wave
modulus, s the smallest node spacing) is capped pair by pair so that the interface spring alone keeps
ω·Δt ≤ 0.6, well inside the stability range of the network's time layer. The finite K transmits a
compressional wave with a small reflection R = 1/√(1 + (2K/ωZ)²) (Z = ρ c_L), reported for the source
frequency and its second harmonic.
"""
import math
import numpy as np

TARGET = 10.
STABILITY = .6


def arrays(mesh):
    """(a, b, normal, area) as contiguous arrays; empty when the mesh has no contact interface."""
    if 'contact_a' not in mesh or len(mesh['contact_a']) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros(0)
    return (np.ascontiguousarray(mesh['contact_a'], dtype=np.int64), np.ascontiguousarray(mesh['contact_b'], dtype=np.int64),
            np.ascontiguousarray(mesh['contact_normal'], dtype=np.float64), np.ascontiguousarray(mesh['contact_area'], dtype=np.float64))


def modulus(material):
    """P-wave modulus (Pa) and density: λ + 2μ for an isotropic card, ρ c_max² for a custom energy material."""
    from physics.materials import is_custom, fastest_speed
    rho = float(material['rho_kg_m3'])
    if is_custom(material): return rho * fastest_speed(material) ** 2, rho
    return float(material['lambda_pa'] + 2 * material['mu_pa']), rho


def stiffness(mesh, scene, dt):
    """Pair stiffnesses k (N/m) for layer step dt, and a JSON summary (None without contact)."""
    a, b, normal, area = arrays(mesh)
    if not len(a): return np.zeros(0), None
    M, rho = modulus(scene['material']); spacing = float(mesh['min_node_spacing']); mass = np.asarray(mesh['mass'])
    target = TARGET * M / spacing * area
    effective = mass[a] * mass[b] / (mass[a] + mass[b])
    limit = (STABILITY / dt) ** 2 * effective
    k = np.minimum(target, limit); per_area = k / area; median = float(np.median(per_area))
    Z = math.sqrt(rho * M)
    from physics.scene import sources
    f0 = min(src['frequency_mhz'] for src in sources(scene)) * 1e6
    reflection = lambda f: 1 / math.sqrt(1 + (2 * median / (2 * math.pi * f * Z)) ** 2)
    return k, {'model': 'unilateral_penalty_frictionless', 'pairs': int(len(a)), 'stiffness_per_area_n_m3': {'min': float(per_area.min()), 'median': median, 'max': float(per_area.max())},
               'stability_limited_fraction': float(np.mean(limit < target)), 'omega_dt_max': float(np.max(np.sqrt(k / effective)) * dt),
               'compression_reflection': {'source_frequency': reflection(f0), 'second_harmonic': reflection(2 * f0)},
               'note': '受压时界面靠罚函数弹簧传力（有限刚度带来很小的反射，见 compression_reflection），受拉时张开、无牵引；切向无摩擦。'}
