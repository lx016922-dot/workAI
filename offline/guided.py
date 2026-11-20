"""Guided waves of a free plate: dispersion curves, the modes at a frequency and the (f, 2f) pairs of second-harmonic
generation, for the plate and the material of a model.

Semi-analytical finite elements through the thickness: u(x, z, t) = U(z) exp(i(kx − ωt)) with x the direction of
propagation in the plane of the plate, GLL elements in z, all three displacement components and the full elastic
tensor of the material. For a wavenumber k this is the Hermitian eigenvalue problem

    (K0 + i k (Aᵀ − A) + k² K2) U = ω² M U,

so every branch ω(k) comes out at once, for isotropic and anisotropic materials alike, and no characteristic
function has to be searched for roots. For an isotropic plate the Lamb branches agree with the Rayleigh–Lamb
equations (`dispersion.lamb`) and the SH branches with ω² = cT²(k² + (nπ/d)²) (offline/test_guided.py).

Branches are separated by the symmetries the plate really has, found numerically from the matrices:

* the mid-plane (z → −z): symmetric and antisymmetric modes;
* the sagittal plane (the plane of x and z): Lamb modes (uₓ, u_z) and SH modes (u_y). An isotropic plate has both,
  an anisotropic one only for propagation along its symmetry directions.

Inside one class branches do not cross, so the n-th frequency at every k is one branch and its number is the
order of its cut-off: S0, S1, …, A0, A1, …, SH0, SH1, … (SH even numbers are symmetric). Where a branch runs
backwards (group velocity opposite to the phase velocity, next to a zero-group-velocity point) it keeps its name
and is marked `backward`.

A plate here is infinitely wide. The model's plate has a width: with symmetry faces on both sides in y it carries
exactly these Lamb modes; with free sides it is a strip and its modes differ, the more the narrower it is
(`validity` says which case a scene is).
"""
import math
from functools import lru_cache
import numpy as np

FAMILY_WORDS = {'S': '对称 Lamb', 'A': '反对称 Lamb', 'SH': '水平剪切 SH', 'Sq': '对称', 'Aq': '反对称', 'L': 'Lamb（矢面内）', 'M': '导波'}
SYMMETRY_TOLERANCE = 1e-9


def _gll(degree):
    from spectral3d.mesh import gll
    x, w, d, _ = gll(degree); return x, w, d


class Plate:
    """A free plate of thickness d (m), density ρ and elastic tensor C_ijkl (Pa, model axes), for waves travelling in
    the plane at `angle_deg` from the model's x axis."""

    def __init__(self, thickness_m, rho, C, angle_deg=0., elements=8, degree=6):
        if not (thickness_m > 0 and rho > 0): raise ValueError('板厚和密度必须为正。')
        a = math.radians(angle_deg); R = np.array([[math.cos(a), math.sin(a), 0.], [-math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
        C = np.einsum('ia,jb,kc,ld,abcd->ijkl', R, R, R, R, np.asarray(C, dtype=float))   # axis 0: propagation, 1: in-plane transverse, 2: thickness
        self.thickness, self.rho, self.C, self.angle_deg = float(thickness_m), float(rho), C, float(angle_deg)
        xi, w, D = _gll(degree); edges = np.linspace(-thickness_m / 2, thickness_m / 2, elements + 1); half = thickness_m / elements / 2
        self.nodes = n = elements * degree + 1; self.z = np.zeros(n); size = 3 * n
        K0 = np.zeros((size, size)); A = np.zeros((size, size)); K2 = np.zeros((size, size)); mass = np.zeros(size)
        for e in range(elements):
            ids = e * degree + np.arange(degree + 1); self.z[ids] = edges[e] + (xi + 1) * half; dN = D / half   # dN[q, a] = N_a'(z_q)
            for q in range(degree + 1):
                weight = w[q] * half; node = ids[q]
                for i in range(3):
                    mass[3 * node + i] += rho * weight
                    for j in range(3):
                        K2[3 * node + i, 3 * node + j] += weight * C[i, 0, j, 0]
                        A[3 * node + i, 3 * ids + j] += weight * C[i, 0, j, 2] * dN[q]
                        K0[np.ix_(3 * ids + i, 3 * ids + j)] += weight * C[i, 2, j, 2] * np.outer(dN[q], dN[q])
        self.K0, self.K1, self.K2, self.mass = K0, A.T - A, K2, mass
        self.classes = self._classes()

    # ---------- symmetry ----------
    def _commutes(self, T):
        scale = max(float(np.max(np.abs(M))) for M in (self.K0, self.K1, self.K2)) or 1.
        return all(float(np.max(np.abs(T @ M @ T.T - M))) <= SYMMETRY_TOLERANCE * scale for M in (self.K0, self.K1, self.K2))

    def _classes(self):
        """[{'family', 'parity', 'Q'}]: orthonormal bases of the symmetry classes the plate has."""
        n = self.nodes; size = 3 * n; mirror = np.zeros((size, size)); sagittal = np.zeros((size, size))
        for node in range(n):
            for c, (sm, ss) in enumerate(((1., 1.), (1., -1.), (-1., 1.))):
                mirror[3 * (n - 1 - node) + c, 3 * node + c] = sm; sagittal[3 * node + c, 3 * node + c] = ss
        self.mirror_symmetric, self.sagittal_symmetric = self._commutes(mirror), self._commutes(sagittal)
        groups = [((0, 2), 'lamb'), ((1,), 'sh')] if self.sagittal_symmetric else [((0, 1, 2), 'all')]
        out = []
        for components, kind in groups:
            columns = {'sym': [], 'anti': []} if self.mirror_symmetric else {'none': []}
            for c in components:
                sign = -1. if c == 2 else 1.
                for node in range(n):
                    partner = n - 1 - node
                    if not self.mirror_symmetric:
                        v = np.zeros(size); v[3 * node + c] = 1.; columns['none'].append(v)
                    elif node == partner:
                        v = np.zeros(size); v[3 * node + c] = 1.; columns['sym' if sign > 0 else 'anti'].append(v)
                    elif node < partner:
                        for parity, s in (('sym', sign), ('anti', -sign)):
                            v = np.zeros(size); v[3 * node + c] = math.sqrt(.5); v[3 * partner + c] = s * math.sqrt(.5); columns[parity].append(v)
            for parity, vectors in columns.items():
                family = {('lamb', 'sym'): 'S', ('lamb', 'anti'): 'A', ('sh', 'sym'): 'SH', ('sh', 'anti'): 'SH', ('all', 'sym'): 'Sq', ('all', 'anti'): 'Aq',
                          ('lamb', 'none'): 'L', ('sh', 'none'): 'SH', ('all', 'none'): 'M'}[(kind, parity)]
                Q = np.array(vectors).T; out.append({'family': family, 'kind': kind, 'parity': parity, 'Q': Q, 'root': 1 / np.sqrt((Q * Q).T @ self.mass)})
        return out

    def name(self, cls, rank):
        """S0, A1, SH2, …: the family and the order of the cut-off (even SH numbers are symmetric)."""
        if cls['kind'] == 'sh' and cls['parity'] != 'none': return f"SH{2 * rank + (cls['parity'] == 'anti')}"
        return f"{cls['family'].rstrip('q')}{rank}"

    def secondary(self, cls):
        """True for the class a second harmonic belongs to: the quadratic forcing of any mode is even under every
        symmetry of the plate, so it drives only symmetric Lamb modes (or what is left of them)."""
        return cls['parity'] in ('sym', 'none') and cls['kind'] in ('lamb', 'all')

    # ---------- eigenproblem ----------
    def solve(self, cls, k, vectors=False):
        """Angular frequencies (ascending) of one class at wavenumber k (rad/m); with `vectors`, also the group
        velocities (m/s) and the mode shapes (columns, three components per node, unit mass norm)."""
        Q, root = cls['Q'], cls['root']
        K = (Q.T @ (self.K0 + k * k * self.K2) @ Q + 1j * k * (Q.T @ self.K1 @ Q)) * root[:, None] * root[None, :]
        if not vectors: return np.sqrt(np.maximum(np.linalg.eigvalsh(K), 0.))
        values, v = np.linalg.eigh(K); omega = np.sqrt(np.maximum(values, 0.))
        dK = (2 * k * (Q.T @ self.K2 @ Q) + 1j * (Q.T @ self.K1 @ Q)) * root[:, None] * root[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            group = np.real(np.einsum('in,ij,jn->n', v.conj(), dK, v)) / (2 * omega)
        return omega, group, Q @ (v * root[:, None])

    def reach(self, frequency_hz):
        """A wavenumber (rad/m) beyond every mode at this frequency: the slowest branch is above it there."""
        omega = 2 * math.pi * frequency_hz; k = omega / math.sqrt(float(np.min(np.diag(self.C.reshape(9, 9))[[2, 5, 8]])) / self.rho)   # a shear speed as the first guess
        for _ in range(40):
            if min(self.solve(cls, k)[0] for cls in self.classes) > omega: return k
            k *= 1.4
        raise ValueError('无法确定这个频率下的最大波数。')

    def curves(self, frequency_max_hz, points=240):
        """Every branch up to a frequency: [{'name', 'family', 'k' (rad/m), 'omega' (rad/s), 'group' (m/s)}], sampled
        on a wavenumber grid that is denser near k = 0 (where the phase velocity of a branch above its cut-off changes fastest)."""
        top = 2 * math.pi * frequency_max_hz; k = self.reach(frequency_max_hz) * 1.05 * (np.arange(points + 1) / points) ** 1.5; out = []
        for cls in self.classes:
            solved = [self.solve(cls, float(v), vectors=True)[:2] for v in k]; omega = np.array([s[0] for s in solved]); group = np.array([s[1] for s in solved])
            for rank in range(omega.shape[1]):
                if omega[:, rank].min() > top: break
                out.append({'name': self.name(cls, rank), 'family': cls['family'], 'secondary': self.secondary(cls), 'k': k.copy(), 'omega': omega[:, rank], 'group': group[:, rank]})
        return out

    def at(self, frequency_hz, points=240):
        """The propagating modes at a frequency, by increasing phase velocity: [{'name', 'family', 'k_rad_m',
        'phase_velocity_m_s', 'group_velocity_m_s', 'backward', 'secondary', 'surface'}]. `surface` is the size of the
        displacement at the top face along the propagation direction, across it and through the thickness (largest = 1)."""
        from scipy.optimize import brentq
        omega = 2 * math.pi * frequency_hz; k = self.reach(frequency_hz) * (np.arange(points + 1) / points) ** 1.5; out = []
        for cls in self.classes:
            table = np.array([self.solve(cls, float(v)) for v in k]) - omega
            for rank in range(table.shape[1]):
                if table[:, rank].min() > 0: break
                for i in np.flatnonzero(table[:-1, rank] * table[1:, rank] < 0):
                    root = brentq(lambda v: self.solve(cls, v)[rank] - omega, k[i], k[i + 1], xtol=1e-12 * k[-1], rtol=1e-13)
                    _, group, shapes = self.solve(cls, root, vectors=True); top = np.abs(shapes[3 * (self.nodes - 1):3 * self.nodes, rank]); scale = float(np.max(np.abs(shapes[:, rank]))) or 1.
                    out.append({'name': self.name(cls, rank), 'family': cls['family'], 'k_rad_m': float(root), 'phase_velocity_m_s': omega / root, 'group_velocity_m_s': float(group[rank]),
                                'backward': bool(group[rank] < 0), 'secondary': self.secondary(cls), 'surface': [float(v) for v in top / scale]})
        return sorted(out, key=lambda m: m['phase_velocity_m_s'])

    def pairs(self, frequency_hz):
        """Every (mode at f, admissible mode at 2f): the mismatch of the phase velocities, Δk = k(2f) − 2k(f) and the
        distance π/|Δk| over which the second harmonic grows before it falls again (None when matched to rounding).
        By increasing mismatch. Phase matching is necessary, not sufficient: the power flux from the primary to the
        secondary mode must not vanish, and the group velocities decide how long the two pulses stay together."""
        first = [m for m in self.at(frequency_hz) if not m['backward']]; second = [m for m in self.at(2 * frequency_hz) if m['secondary'] and not m['backward']]; rows = []
        for a in first:
            for b in second:
                dk = b['k_rad_m'] - 2 * a['k_rad_m']
                rows.append({'primary': a['name'], 'secondary': b['name'], 'phase_velocity_primary_m_s': a['phase_velocity_m_s'], 'phase_velocity_secondary_m_s': b['phase_velocity_m_s'],
                             'mismatch': b['phase_velocity_m_s'] / a['phase_velocity_m_s'] - 1, 'delta_k_rad_m': dk, 'coherence_length_m': math.pi / abs(dk) if abs(dk) > 1e-9 * b['k_rad_m'] else None,
                             'group_velocity_primary_m_s': a['group_velocity_m_s'], 'group_velocity_secondary_m_s': b['group_velocity_m_s']})
        return sorted(rows, key=lambda r: abs(r['mismatch']))


def elements_for(thickness_m, shear_speed, frequency_max_hz):
    """Elements of degree 6 through the thickness: about 15 nodes per shear wavelength at the highest frequency, at least 4."""
    return int(min(40, max(4, math.ceil(2.5 * thickness_m * frequency_max_hz / shear_speed))))


def plate_of(scene, angle_deg=0., frequency_max_hz=None):
    """The plate of a model. Raises ValueError for a pipe (its modes are those of a cylinder, which this module does not have)."""
    from physics.materials import is_custom, slowest_shear_speed, tensors
    from solid.schema import MATERIAL
    geometry = scene['geometry']
    if geometry['kind'] == 'pipe': raise ValueError('圆管的导波模态与平板不同，目前没有它的理论频散曲线；f–k 谱仍可计算。')
    material = scene['material']; rho = float((material if is_custom(material) else MATERIAL)['rho_kg_m3']); thickness = geometry['thickness_mm'] * 1e-3
    top = 2.6 * float(scene['source']['frequency_mhz']) * 1e6 if frequency_max_hz is None else frequency_max_hz
    return _plate(thickness, rho, tensors(material)[0].tobytes(), float(angle_deg), elements_for(thickness, slowest_shear_speed(material), top))


@lru_cache(maxsize=8)
def _plate(thickness, rho, c2_bytes, angle_deg, elements):
    from physics.materials import stiffness4
    return Plate(thickness, rho, stiffness4(np.frombuffer(c2_bytes).reshape(6, 6)), angle_deg, elements)


def validity(scene):
    """How far the infinite plate describes the model: {'plane_strain', 'note'}."""
    geometry = scene['geometry']; faces = {b['face']: b['type'] for b in scene.get('boundaries') or []}
    both = faces.get('ymin') == 'symmetry' and faces.get('ymax') == 'symmetry'
    if both: return {'plane_strain': True, 'note': '两个 y 面都是对称面：模型就是无限宽板里沿 x 传播的 Lamb 波，理论曲线直接适用（对称面不允许 SH 波）。'}
    free = [f for f in ('ymin', 'ymax') if faces.get(f) not in ('symmetry', 'absorbing')]
    return {'plane_strain': False, 'note': f"理论曲线是无限宽板的。这个模型板宽 {geometry['width_mm']:g} mm" + ('、侧面自由' if free else '') +
            '：宽度与波长相当时它是一根条带，模态会偏离这些曲线；要对照 Lamb 理论，请把两个 y 面设为对称面。'}


def describe(scene, angle_deg=0., frequency_max_mhz=None, points=240):
    """What the result page shows: the curves up to `frequency_max_mhz` (default 2.6 f), the modes at f and 2f and the pairs."""
    f = float(scene['source']['frequency_mhz']); top = 2.6 * f if frequency_max_mhz is None else frequency_max_mhz
    if not (0 < top <= 100): raise ValueError('频散曲线的频率上限需在 0–100 MHz。')
    plate = plate_of(scene, angle_deg, top * 1e6); curves = _curves(plate, top * 1e6, points)
    public = lambda m: {'name': m['name'], 'family': m['family'], 'wavenumber_rad_mm': m['k_rad_m'] * 1e-3, 'phase_velocity_m_s': m['phase_velocity_m_s'], 'group_velocity_m_s': m['group_velocity_m_s'],
                        'wavelength_mm': 2e3 * math.pi / m['k_rad_m'], 'backward': m['backward'], 'secondary': m['secondary'], 'surface': m['surface']}
    pairs = [{**{k: r[k] for k in ('primary', 'secondary', 'mismatch', 'phase_velocity_primary_m_s', 'phase_velocity_secondary_m_s', 'group_velocity_primary_m_s', 'group_velocity_secondary_m_s')},
              'delta_k_rad_mm': r['delta_k_rad_m'] * 1e-3, 'coherence_length_mm': None if r['coherence_length_m'] is None else r['coherence_length_m'] * 1e3} for r in _pairs(plate, f * 1e6)]
    return {'schema': 'workai.plate_dispersion.v1', 'thickness_mm': scene['geometry']['thickness_mm'], 'angle_deg': float(angle_deg), 'frequency_mhz': f, 'frequency_max_mhz': top,
            'symmetry': {'mid_plane': plate.mirror_symmetric, 'sagittal_plane': plate.sagittal_symmetric}, 'families': {key: FAMILY_WORDS[key] for key in sorted({c['family'] for c in curves})},
            'validity': validity(scene),
            'curves': [{'name': c['name'], 'family': c['family'], 'secondary': c['secondary'], 'wavenumber_rad_mm': (c['k'] * 1e-3).tolist(), 'frequency_mhz': (c['omega'] / (2e6 * math.pi)).tolist(),
                        'group_velocity_m_s': [None if not math.isfinite(v) else float(v) for v in c['group']]} for c in curves],
            'modes': {'f': [public(m) for m in _at(plate, f * 1e6)], '2f': [public(m) for m in _at(plate, 2 * f * 1e6)]}, 'pairs': pairs,
            'note': '相位匹配（Δk = 0）是二次谐波累积的必要条件，不是充分条件：还要求一次模态到二次模态的功率流不为零，群速度决定两个波包能一起走多远。'
                    '二次谐波只出现在对称 Lamb 模态里，所以配对的二次模态只列对称族。'}


@lru_cache(maxsize=8)
def _curves(plate, top, points): return plate.curves(top, points)


@lru_cache(maxsize=32)
def _at(plate, frequency): return plate.at(frequency)


@lru_cache(maxsize=16)
def _pairs(plate, frequency): return plate.pairs(frequency)
