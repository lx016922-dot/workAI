"""Hyperelastic materials for version-2 scenes: isotropic Murnaghan, or any strain-energy density W(E).

A scene's `material` is either the legacy reference aluminium (the six Murnaghan keys, no `model`), or

    {"model": "murnaghan", "rho_kg_m3", "lambda_pa", "mu_pa", "l_pa", "m_pa", "n_pa"}      isotropic, any constants
    {"model": "energy", "name", "rho_kg_m3", "energy", "parameters", "fiber_angle_deg"}    user-defined W(E), e.g. CFRP

`energy` is the strain-energy density per unit reference volume, in GPa (1 GPa = 1e9 J/m³), written in the
material frame (axis 1 = fibre) with the Green–Lagrange strain E = (FᵀF − I)/2:

    E11 E22 E33 E12 E13 E23        tensor components (E21 = E12 …)
    i1 i2 i3                       principal invariants: tr E, ½[(tr E)² − tr E²], det E
    tr2 tr3                        tr E², tr E³
    i4 i5                          fibre invariants a·E·a and a·E²·a, a = material axis 1
    exp log sqrt sin cos tanh sinh cosh, numbers, + − * / **, parentheses, and the named parameters

`fiber_angle_deg` turns the material frame about the global z axis (fibre in the x–y plane, measured from x).

Every solver here (the GLL spectral reference and the network's energy layer) uses the cubic Taylor expansion of
W about the stress-free state, W ≈ ½ c2:E:E + ⅙ c3:E:E:E, i.e. the second- and third-order elastic constants.
At ultrasonic strains (≤ 1e-3, enforced by the strain guard) the quartic remainder is reported by `describe`.
The constants are obtained by complex-step differentiation of W (first derivative exact to round-off) and central
differences of that stress (second and third derivatives, step 1e-4).
"""
import ast
import math

import numpy as np

from solid.schema import require, number, MATERIAL as REFERENCE

MODELS = ('murnaghan', 'energy')
MURNAGHAN_KEYS = ('rho_kg_m3', 'lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa')
VARIABLES = ('E11', 'E22', 'E33', 'E12', 'E13', 'E23', 'E21', 'E31', 'E32', 'i1', 'i2', 'i3', 'tr2', 'tr3', 'i4', 'i5')
FUNCTIONS = {'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'tanh': np.tanh, 'sinh': np.sinh, 'cosh': np.cosh}
MAX_EXPRESSION = 4000
MAX_PARAMETERS = 40
SQRT2 = math.sqrt(2.)
# Mandel order: 11, 22, 33, 23, 13, 12 (orthonormal: the stress of an energy is its gradient in these coordinates).
PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
STEP = 1e-4
CHECK_STRAINS = (np.diag([1e-3, 0, 0]), np.array([[0, 1e-3, 0], [1e-3, 0, 0], [0, 0, 0]]),
                 np.array([[1, .4, -.2], [.4, -.5, .3], [-.2, .3, .7]]) * 1e-3)


def is_custom(material):
    return isinstance(material, dict) and 'model' in material


def require_isotropic(material, solver):
    """Solvers written for the isotropic Murnaghan law refuse strain-energy materials with a clear message."""
    require(not (is_custom(material) and material['model'] == 'energy'),
            f'{solver}只支持各向同性 Murnaghan 材料；自定义应变能材料请用三维显式谱元（spectral3d）或物理 AI 网络。')


# ---------- expressions ----------
class _Checker(ast.NodeVisitor):
    ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load, ast.Call)

    def __init__(self, names):
        self.names = names; self.used = set()

    def generic_visit(self, node):
        require(isinstance(node, self.ALLOWED), f'应变能表达式不支持 {type(node).__name__}；只能用数字、变量、参数、+ − * / ** 和 {"、".join(FUNCTIONS)}。')
        if isinstance(node, ast.Constant): require(type(node.value) in (int, float), '表达式中只能有数字常量。')
        if isinstance(node, ast.Name):
            require(node.id in self.names, f'未知名称 {node.id}：可用变量 {", ".join(VARIABLES[:6])}、i1–i5、tr2、tr3 和已定义的参数。'); self.used.add(node.id)
        if isinstance(node, ast.Call):
            require(isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords, f'只能调用单参数函数 {"、".join(FUNCTIONS)}。')
            self.visit(node.args[0]); return
        super().generic_visit(node)


def compile_energy(expression, parameters):
    """Validated, compiled W(E-tensor array, ...×3×3) → GPa; works on complex arrays."""
    require(isinstance(expression, str) and 0 < len(expression.strip()) <= MAX_EXPRESSION, f'应变能表达式需要 1–{MAX_EXPRESSION} 个字符。')
    require(isinstance(parameters, dict) and len(parameters) <= MAX_PARAMETERS, f'参数最多 {MAX_PARAMETERS} 个。')
    for key, value in parameters.items():
        require(isinstance(key, str) and key.isidentifier() and key not in VARIABLES and key not in FUNCTIONS and not key.startswith('_'), f'参数名 {key!r} 无效或与变量、函数重名。')
        number(value, -1e9, 1e9, f'参数 {key}')
    try: tree = ast.parse(expression.replace('^', '**'), mode='eval')
    except SyntaxError as error: raise ValueError(f'应变能表达式语法错误：{error.msg}（第 {error.offset} 个字符附近）。') from None
    checker = _Checker(set(VARIABLES) | set(parameters)); checker.visit(tree)
    require(checker.used & set(VARIABLES), '应变能表达式没有用到应变（E11…E23 或不变量）。')
    code = compile(tree, '<strain-energy>', 'eval')
    constants = {k: float(v) for k, v in parameters.items()}

    def energy(E):
        e2 = np.einsum('...ij,...jk->...ik', E, E); e3 = np.einsum('...ij,...jk->...ik', e2, E)
        tr1 = np.einsum('...ii->...', E); tr2 = np.einsum('...ii->...', e2); tr3 = np.einsum('...ii->...', e3)
        det = (E[..., 0, 0] * (E[..., 1, 1] * E[..., 2, 2] - E[..., 1, 2] * E[..., 2, 1]) - E[..., 0, 1] * (E[..., 1, 0] * E[..., 2, 2] - E[..., 1, 2] * E[..., 2, 0])
               + E[..., 0, 2] * (E[..., 1, 0] * E[..., 2, 1] - E[..., 1, 1] * E[..., 2, 0]))
        scope = {'E11': E[..., 0, 0], 'E22': E[..., 1, 1], 'E33': E[..., 2, 2], 'E12': E[..., 0, 1], 'E13': E[..., 0, 2], 'E23': E[..., 1, 2],
                 'E21': E[..., 1, 0], 'E31': E[..., 2, 0], 'E32': E[..., 2, 1], 'i1': tr1, 'i2': .5 * (tr1 * tr1 - tr2), 'i3': det, 'tr2': tr2, 'tr3': tr3,
                 'i4': E[..., 0, 0], 'i5': e2[..., 0, 0], **FUNCTIONS, **constants}
        with np.errstate(all='ignore'):
            value = eval(code, {'__builtins__': {}}, scope)  # noqa: S307 — AST restricted to arithmetic, whitelisted names and functions
        return np.broadcast_to(value, E.shape[:-2]) + 0 * tr1
    return energy


# ---------- Mandel helpers ----------
def tensor(z):
    """Mandel 6-vector(s) → symmetric 3×3 tensor(s)."""
    z = np.asarray(z); out = np.zeros(z.shape[:-1] + (3, 3), dtype=z.dtype)
    for k, (i, j) in enumerate(PAIRS):
        value = z[..., k] if i == j else z[..., k] / SQRT2
        out[..., i, j] = value; out[..., j, i] = value
    return out


def mandel(a):
    a = np.asarray(a); return np.stack([a[..., i, j] if i == j else SQRT2 * a[..., i, j] for i, j in PAIRS], -1)


def rotation_z(angle_deg):
    t = math.radians(angle_deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def mandel_rotation(R):
    """Q with mandel(R A Rᵀ) = Q mandel(A) for symmetric A."""
    return np.stack([mandel(R @ tensor(np.eye(6)[k]) @ R.T) for k in range(6)], 1)


def _stress(energy, z):
    """Mandel second Piola stress (Pa) at Mandel strains z (n×6), complex step on W (GPa)."""
    h = 1e-30; out = np.empty(z.shape)
    for k in range(6):
        shifted = z.astype(complex); shifted[:, k] += 1j * h
        out[:, k] = np.imag(energy(tensor(shifted))) / h * 1e9
    return out


def taylor_constants(energy):
    """c2 (6×6) and c3 (6×6×6), Pa, Mandel, material frame; plus the stress at zero strain."""
    eye = np.eye(6); d = STEP
    s0 = _stress(energy, np.zeros((1, 6)))[0]
    plus = _stress(energy, d * eye); minus = _stress(energy, -d * eye)
    c2 = ((plus - minus) / (2 * d)).T  # c2[I, J] = ∂S_I/∂z_J
    c3 = np.empty((6, 6, 6))
    for j in range(6):
        for k in range(j, 6):
            points = np.array([d * (eye[j] + eye[k]), d * (eye[j] - eye[k]), d * (-eye[j] + eye[k]), -d * (eye[j] + eye[k])])
            s = _stress(energy, points); value = (s[0] - s[1] - s[2] + s[3]) / (4 * d * d)
            c3[:, j, k] = value; c3[:, k, j] = value
    c2 = .5 * (c2 + c2.T)
    c3 = sum(np.transpose(c3, p) for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))) / 6
    return c2, c3, s0


def murnaghan_tensors(m):
    """Exact Mandel c2, c3 (Pa) of the Murnaghan energy ½(λ+2μ)i1² − 2μ i2 + ⅓(l+2m)i1³ − 2m i1 i2 + n i3."""
    lam, mu, l, mm, n = (m[k] for k in MURNAGHAN_KEYS[1:])
    c2 = 2 * mu * np.eye(6); c2[:3, :3] += lam
    a = n; b = mm - n / 2; c = l - mm + n / 2

    def quadratic_stress(z):  # the part of S quadratic in E; equals ½ c3 z z
        e = tensor(z); tr = np.trace(e); e2 = e @ e
        return mandel((b * np.trace(e2) + c * tr ** 2) * np.eye(3) + 2 * b * tr * e + a * e2)
    eye = np.eye(6); c3 = np.zeros((6, 6, 6))
    for j in range(6):
        for k in range(6): c3[:, j, k] = (quadratic_stress(eye[j] + eye[k]) - quadratic_stress(eye[j] - eye[k])) / 2  # polarisation: ½c3(u+v)² − ½c3(u−v)² = 2 c3 u v
    return c2, c3


# ---------- validation ----------
def validate_material(raw):
    """Normalized copy of a `model` material (legacy aluminium is validated by the version-1 core)."""
    require(isinstance(raw, dict) and raw.get('model') in MODELS, '材料 model 需为 murnaghan 或 energy。')
    m = dict(raw)
    if m['model'] == 'murnaghan':
        require(set(m) == {'model', 'name', *MURNAGHAN_KEYS} or set(m) == {'model', *MURNAGHAN_KEYS}, 'Murnaghan 材料需要 rho_kg_m3、lambda_pa、mu_pa、l_pa、m_pa、n_pa。')
        m['rho_kg_m3'] = number(m['rho_kg_m3'], 1, 1e5, '密度'); m['lambda_pa'] = number(m['lambda_pa'], -1e13, 1e13, 'λ'); m['mu_pa'] = number(m['mu_pa'], 1e6, 1e13, 'μ')
        for k in ('l_pa', 'm_pa', 'n_pa'): m[k] = number(m[k], -1e14, 1e14, k)
        require(3 * m['lambda_pa'] + 2 * m['mu_pa'] > 0, '体积模量 λ + 2μ/3 必须为正。')
    else:
        allowed = {'model', 'name', 'rho_kg_m3', 'energy', 'parameters', 'fiber_angle_deg'}
        require(set(m) <= allowed and {'rho_kg_m3', 'energy'} <= set(m), '自定义材料需要 rho_kg_m3、energy，可选 name、parameters、fiber_angle_deg。')
        m['rho_kg_m3'] = number(m['rho_kg_m3'], 1, 1e5, '密度'); m['parameters'] = dict(m.get('parameters') or {}); m['fiber_angle_deg'] = number(m.get('fiber_angle_deg', 0.), -180, 180, '纤维角')
        compile_energy(m['energy'], m['parameters'])
    if 'name' in m: require(isinstance(m['name'], str) and 1 <= len(m['name']) <= 60, '材料名称需为 1–60 个字符。')
    c2, c3, _ = tensors(m, checked=True)
    return m


def tensors(material, checked=False):
    """Global-frame Mandel (c2, c3, info) in Pa. Legacy aluminium is the 'murnaghan' reference."""
    m = material if is_custom(material) else dict(REFERENCE, model='murnaghan')
    if m['model'] == 'murnaghan':
        c2, c3 = murnaghan_tensors(m); info = {'stress_free_residual_pa': 0.}
    else:
        energy = compile_energy(m['energy'], m.get('parameters') or {}); c2, c3, s0 = taylor_constants(energy)
        require(np.all(np.isfinite(c2)) and np.all(np.isfinite(c3)), '应变能在零应变附近不可微或出现非有限值。')
        scale = float(np.max(np.abs(c2))) or 1.
        require(float(np.max(np.abs(s0))) <= 1e-9 * scale, f'参考构形必须无应力：零应变处应力为 {np.max(np.abs(s0)) / 1e6:.3g} MPa；请去掉应变的一次项。')
        Q = mandel_rotation(rotation_z(m.get('fiber_angle_deg', 0.))); c2 = Q @ c2 @ Q.T; c3 = np.einsum('ia,jb,kc,abc->ijk', Q, Q, Q, c3)
        info = {'stress_free_residual_pa': float(np.max(np.abs(s0)))}
    eig = np.linalg.eigvalsh(c2)
    if checked:
        require(eig[0] > 1e-6 * eig[-1] and eig[0] > 0, f'二阶刚度不正定（最小特征值 {eig[0] / 1e9:.4g} GPa）：材料在小变形下不稳定。')
    info['eigenvalues_gpa'] = (eig / 1e9).tolist()
    return c2, c3, info


# ---------- derived quantities ----------
def stiffness4(c2):
    """Full C_ijkl from Mandel c2."""
    C = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(PAIRS):
        for J, (k, l) in enumerate(PAIRS):
            w = (1 if i == j else SQRT2) * (1 if k == l else SQRT2); v = c2[I, J] / w
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)): C[a, b, c, d] = v
    return C


def christoffel(c2, rho, direction):
    n = np.asarray(direction, float); n = n / np.linalg.norm(n)
    G = np.einsum('ijkl,j,l->ik', stiffness4(c2), n, n) / rho
    values, vectors = np.linalg.eigh(G)
    order = np.argsort(values)[::-1]
    return np.sqrt(np.maximum(values[order], 0)), vectors[:, order].T


def _sphere(count=2000):
    k = np.arange(count) + .5; phi = np.arccos(1 - 2 * k / count); theta = math.pi * (1 + 5 ** .5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def speed_bounds(material):
    """(fastest quasi-longitudinal, slowest quasi-shear) phase speeds over all directions, m/s.

    Sampled on 4000 directions plus the global and material axes, then refined locally around the extremes."""
    c2, _, _ = tensors(material); rho = material['rho_kg_m3']; C = stiffness4(c2)
    angle = material.get('fiber_angle_deg', 0.) if is_custom(material) else 0.
    def eig(dirs):
        dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
        return np.linalg.eigvalsh(np.einsum('ijkl,nj,nl->nik', C, dirs, dirs) / rho)
    dirs = np.concatenate([_sphere(4000), np.eye(3), rotation_z(angle).T, (np.eye(3)[[0, 0, 1]] + np.eye(3)[[1, 2, 2]]) / SQRT2])
    values = eig(dirs); out = []
    for column, pick in ((-1, np.argmax), (0, np.argmin)):
        best = dirs[pick(values[:, column])]; value = values[pick(values[:, column]), column]; spread = .05
        rng = np.random.default_rng(7)
        for _ in range(6):
            trial = best + spread * rng.normal(size=(200, 3)); v = eig(trial)[:, column]; k = pick(v)
            if (v[k] > value) if column == -1 else (v[k] < value): best, value = trial[k] / np.linalg.norm(trial[k]), v[k]
            spread /= 3
        out.append(float(np.sqrt(max(value, 0.))))
    return out[0], out[1]


_BOUNDS = {}


def _key(material):
    import json
    return json.dumps(material, sort_keys=True)


def fastest_speed(material):
    key = _key(material)
    if key not in _BOUNDS: _BOUNDS[key] = speed_bounds(material)
    return _BOUNDS[key][0]


def slowest_shear_speed(material):
    """Slowest quasi-shear phase speed (the legacy isotropic value √(μ/ρ) for Murnaghan materials)."""
    if not is_custom(material) or material['model'] == 'murnaghan': return math.sqrt(material['mu_pa'] / material['rho_kg_m3'])
    key = _key(material)
    if key not in _BOUNDS: _BOUNDS[key] = speed_bounds(material)
    return _BOUNDS[key][1]


def voigt(c2, c3):
    """Voigt (engineering shear) constants: C_IJ and Brugger C_IJK, Pa."""
    w = np.array([1, 1, 1, SQRT2, SQRT2, SQRT2])
    return c2 / np.outer(w, w), c3 / np.einsum('i,j,k->ijk', w, w, w)


def describe(material):
    """Everything the material panel shows: constants, checks, wave speeds and reference stresses."""
    c2, c3, info = tensors(material, checked=True); rho = material['rho_kg_m3']
    C2, C3 = voigt(c2, c3)
    angle = material.get('fiber_angle_deg', 0.) if is_custom(material) and material['model'] == 'energy' else 0.
    fiber = rotation_z(angle)[:, 0]; transverse = rotation_z(angle)[:, 1]
    directions = {'fiber': fiber, 'in_plane_transverse': transverse, 'thickness': np.array([0., 0., 1.]), 'x': np.array([1., 0., 0.])}
    speeds = {k: christoffel(c2, rho, d)[0].tolist() for k, d in directions.items()}
    fastest, slowest = speed_bounds(material)
    out = {'model': material.get('model', 'murnaghan') if is_custom(material) else 'reference_aluminium', 'rho_kg_m3': rho,
           'voigt_c2_gpa': (C2 / 1e9).round(6).tolist(), 'eigenvalues_gpa': info['eigenvalues_gpa'],
           # third-order constants above round-off of the differentiation (1e-7 of the stiffness scale)
           'c3_nonzero_gpa': [{'index': ''.join(str(i + 1) for i in idx), 'value': float(C3[idx] / 1e9)} for idx in sorted({tuple(sorted(t)) for t in zip(*np.nonzero(np.abs(C3) > 1e-7 * np.max(np.abs(C2))))})],
           'speeds_m_s': speeds, 'fastest_m_s': fastest, 'slowest_shear_m_s': slowest, 'stress_free_residual_pa': info['stress_free_residual_pa'],
           'check_strains': [e.tolist() for e in CHECK_STRAINS], 'check_stress_mpa': [], 'truncation': None}
    if is_custom(material) and material['model'] == 'energy':
        energy = compile_energy(material['energy'], material.get('parameters') or {}); R = rotation_z(angle)
        for e in CHECK_STRAINS:  # global strain → material frame; exact S from W, rotated back
            local = R.T @ e @ R; s = tensor(_stress(energy, mandel(local)[None])[0]); out['check_stress_mpa'].append((R @ s @ R.T / 1e6).round(6).tolist())
        # Remainder of the cubic expansion at |E| = 1e-3 (material frame), relative to the cubic term, or to W2 when c3 = 0.
        c2m, c3m, _ = tensors(dict(material, fiber_angle_deg=0.))
        dirs = np.random.default_rng(1).normal(size=(96, 6)); dirs *= 1e-3 / np.linalg.norm(dirs, axis=1)[:, None]
        exact = np.real(energy(tensor(dirs.astype(complex)))) * 1e9
        quadratic = .5 * np.einsum('ni,ij,nj->n', dirs, c2m, dirs); cubic = np.einsum('nj,ijk,nk,ni->n', dirs, c3m, dirs, dirs) / 6
        zero_c3 = bool(np.max(np.abs(c3m)) <= 1e-9 * np.max(np.abs(c2m)))
        base = np.max(np.abs(quadratic)) if zero_c3 else np.max(np.abs(cubic))
        out['truncation'] = {'strain': 1e-3, 'remainder_relative': float(np.max(np.abs(exact - quadratic - cubic)) / base), 'relative_to': 'quadratic' if zero_c3 else 'cubic', 'third_order_zero': zero_c3}
    else:
        c2m, c3m = c2, c3
        for e in CHECK_STRAINS:
            z = mandel(e); s = c2m @ z + .5 * np.einsum('ijk,j,k->i', c3m, z, z); out['check_stress_mpa'].append((tensor(s) / 1e6).round(6).tolist())
    return out


# ---------- presets ----------
def ti_stiffness_from_engineering(E1, E2, G12, nu12, nu23):
    """Transversely isotropic (axis 1) C11, C12, C22, C23, C55 in the units of the moduli."""
    S = np.array([[1 / E1, -nu12 / E1, -nu12 / E1], [-nu12 / E1, 1 / E2, -nu23 / E2], [-nu12 / E1, -nu23 / E2, 1 / E2]])
    C = np.linalg.inv(S)
    return {'C11': float(C[0, 0]), 'C12': float(C[0, 1]), 'C22': float(C[1, 1]), 'C23': float(C[1, 2]), 'C55': float(G12)}


TI_ENERGY = ('0.5*C11*E11**2 + C12*E11*(E22 + E33) + 0.5*C22*(E22**2 + E33**2) + C23*E22*E33 + (C22 - C23)*E23**2 + 2*C55*(E12**2 + E13**2)'
             ' + a1*i1**3 + a2*i1*tr2 + a3*tr3 + a4*i1**2*i4 + a5*i1*i4**2 + a6*i4**3 + a7*i1*i5 + a8*tr2*i4 + a9*i4*i5')
ORTHO_ENERGY = ('0.5*(C11*E11**2 + C22*E22**2 + C33*E33**2) + C12*E11*E22 + C13*E11*E33 + C23*E22*E33'
                ' + 2*(C44*E23**2 + C55*E13**2 + C66*E12**2)')


def presets():
    """Built-in materials. CFRP values are illustrative magnitudes of a unidirectional carbon/epoxy ply, not a certified grade;
    third-order coefficients are zero until measured values are entered."""
    ti = {k: round(v, 4) for k, v in ti_stiffness_from_engineering(135., 9., 5., .3, .45).items()}
    return {
        'aluminium': {'label': '多晶铝（Murnaghan，物理 AI 训练材料）', 'material': dict(REFERENCE)},
        'murnaghan': {'label': '各向同性 Murnaghan（自定义常数）', 'material': dict(REFERENCE, model='murnaghan', name='各向同性 Murnaghan')},
        'cfrp_ud': {'label': 'CFRP 单向板（横观各向同性，自定义应变能）', 'material': {
            'model': 'energy', 'name': 'CFRP 单向（示例常数）', 'rho_kg_m3': 1560., 'fiber_angle_deg': 0., 'energy': TI_ENERGY,
            'parameters': {**ti, **{f'a{k}': 0. for k in range(1, 10)}}},
            'engineering': {'E1': 135., 'E2': 9., 'G12': 5., 'nu12': .3, 'nu23': .45}},
        'orthotropic': {'label': '正交各向异性（9 个二阶常数，可追加三阶项）', 'material': {
            'model': 'energy', 'name': '正交各向异性', 'rho_kg_m3': 1560., 'fiber_angle_deg': 0., 'energy': ORTHO_ENERGY,
            'parameters': {'C11': 140., 'C22': 12., 'C33': 11., 'C12': 6., 'C13': 6., 'C23': 5.5, 'C44': 3.5, 'C55': 5., 'C66': 5.2}}},
    }
