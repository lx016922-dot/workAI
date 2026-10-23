// Material-point reference implementation, SI units, 3-D reference configuration.
// This is not a guided-wave solver and is not called by the legacy COMSOL bridge.
const eye = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const matrix = f => eye.map((row, i) => row.map((_, j) => f(i, j)));
const transpose = a => matrix((i, j) => a[j][i]);
const mul = (a, b) => matrix((i, j) => a[i].reduce((v, x, k) => v + x * b[k][j], 0));
const trace = a => a[0][0] + a[1][1] + a[2][2];
const determinant = a => a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

function validateMatrix(a) {
  if (!Array.isArray(a) || a.length !== 3 || a.some(row => !Array.isArray(row) || row.length !== 3 || row.some(x => typeof x !== 'number' || !Number.isFinite(x)))) throw new Error('需要有限数值组成的 3×3 张量。');
}
function constants(c) {
  for (const key of ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']) if (typeof c?.[key] !== 'number' || !Number.isFinite(c[key])) throw new Error('本构常数必须明确以 Pa 给出。');
  if (c.mu_pa <= 0 || c.lambda_pa + 2 * c.mu_pa / 3 <= 0) throw new Error('参考态剪切和体积模量必须为正。');
  return c;
}
export function toLandau(c) {
  const { lambda_pa, mu_pa, l_pa, m_pa, n_pa } = constants(c);
  return { lambda_pa, mu_pa, A_pa: n_pa, B_pa: m_pa - n_pa / 2, C_pa: l_pa - m_pa + n_pa / 2 };
}
export function murnaghanResponse(F, c, { materialNonlinearity = true } = {}) {
  validateMatrix(F); constants(c);
  if (typeof materialNonlinearity !== 'boolean') throw new Error('材料非线性开关必须为布尔值。');
  const J = determinant(F);
  if (!Number.isFinite(J) || J <= 0) throw new Error('变形梯度必须保持正体积。');
  const FtF = mul(transpose(F), F), E = matrix((i, j) => (FtF[i][j] - eye[i][j]) / 2);
  const norm = Math.hypot(...E.flat());
  // A conservative computational limit, NOT a measured elastic limit for aluminum.
  if (!Number.isFinite(norm) || norm > .01) throw new Error('材料点参考实现限定 ||E||F ≤ 0.01；此限制不是试样弹性极限。');
  const E2 = mul(E, E), t = trace(E), t2 = trace(E2), t3 = trace(mul(E2, E));
  const { lambda_pa: lam, mu_pa: mu, A_pa: a, B_pa: b, C_pa: cc } = toLandau(c);
  const A = materialNonlinearity ? a : 0, B = materialNonlinearity ? b : 0, C = materialNonlinearity ? cc : 0;
  const energy_j_m3 = lam / 2 * t ** 2 + mu * t2 + A / 3 * t3 + B * t * t2 + C / 3 * t ** 3;
  const S_pa = matrix((i, j) => (lam * t + B * t2 + C * t ** 2) * eye[i][j] + (2 * mu + 2 * B * t) * E[i][j] + A * E2[i][j]);
  const P_pa = mul(F, S_pa), PFt = mul(P_pa, transpose(F)), cauchy_pa = matrix((i, j) => PFt[i][j] / J);
  if (![energy_j_m3, ...S_pa.flat(), ...P_pa.flat(), ...cauchy_pa.flat()].every(Number.isFinite)) throw new Error('本构计算超出有限数值范围。');
  return { E, J, energy_j_m3, S_pa, P_pa, cauchy_pa };
}
export function linearElasticResponse(H, c) {
  validateMatrix(H); constants(c);
  const epsilon = matrix((i, j) => (H[i][j] + H[j][i]) / 2), t = trace(epsilon);
  const stress_pa = matrix((i, j) => c.lambda_pa * t * eye[i][j] + 2 * c.mu_pa * epsilon[i][j]);
  return { epsilon, stress_pa, energy_j_m3: c.lambda_pa / 2 * t ** 2 + c.mu_pa * trace(mul(epsilon, epsilon)) };
}
