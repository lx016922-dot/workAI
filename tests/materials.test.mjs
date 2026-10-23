import test from 'node:test';
import assert from 'node:assert/strict';
import { aluminumMaterial } from '../server/materials.mjs';
import { defaultResearchCase, compileResearchCase } from '../server/research.mjs';
import { murnaghanResponse, linearElasticResponse, toLandau } from '../server/physics/murnaghan.mjs';

const r = aluminumMaterial().record;
const c = { lambda_pa: r.lambda_pa, mu_pa: r.mu_pa, l_pa: r.third_order.l_pa, m_pa: r.third_order.m_pa, n_pa: r.third_order.n_pa };
const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const mat = f => I.map((row, i) => row.map((_, j) => f(i, j)));
const product = (a, b) => mat((i, j) => a[i].reduce((s, v, k) => s + v * b[k][j], 0));
const close = (actual, expected, rel = 1e-10, abs = 1e-6) => assert.ok(Math.abs(actual - expected) <= Math.max(abs, rel * Math.abs(expected)), `${actual} != ${expected}`);

test('the cited polycrystalline Al table is stored in Pa, with reported uncertainty and explicit provenance', () => {
  assert.deepEqual(c, { lambda_pa: 58.1e9, mu_pa: 26.1e9, l_pa: -143e9, m_pa: -297e9, n_pa: -345e9 });
  assert.deepEqual(['l', 'm', 'n'].map(k => r.third_order.reported_uncertainty_pa[k]), [13e9, 6e9, 4e9]);
  assert.equal(r.grade, null); assert.equal(r.temperature_k, null);
  assert.equal(r.sources[0].doi, '10.1063/1.4993443');
  assert.equal(r.sources[1].parameters.join(), 'rho_kg_m3');
  const { linear } = aluminumMaterial();
  close(linear.rho_kg_m3 * linear.ct_m_s ** 2, c.mu_pa);
  close(linear.rho_kg_m3 * (linear.cl_m_s ** 2 - 2 * linear.ct_m_s ** 2), c.lambda_pa);
  assert.deepEqual(toLandau(c), { lambda_pa: 58.1e9, mu_pa: 26.1e9, A_pa: -345e9, B_pa: -124.5e9, C_pa: -18.5e9 });
});

test('Landau energy equals the independently evaluated Murnaghan principal-invariant energy in 3-D', () => {
  for (const e of [[.001, -.0004, .0003], [.002, .002, .002], [.001, 0, 0], [-.0001, .0002, .0007]]) {
    const F = mat((i, j) => i === j ? Math.sqrt(1 + 2 * e[i]) : 0);
    const i1 = e[0] + e[1] + e[2], i2 = e[0] * e[1] + e[0] * e[2] + e[1] * e[2], i3 = e[0] * e[1] * e[2];
    const expected = (c.lambda_pa + 2 * c.mu_pa) / 2 * i1 ** 2 - 2 * c.mu_pa * i2 + (c.l_pa + 2 * c.m_pa) / 3 * i1 ** 3 - 2 * c.m_pa * i1 * i2 + c.n_pa * i3;
    close(murnaghanResponse(F, c).energy_j_m3, expected);
  }
});

test('all nine first Piola components equal numerical derivatives of stored energy, including shear', () => {
  const F = [[1.001, .0008, -.0003], [-.0002, .9997, .0004], [.0005, -.0001, 1.0002]], h = 1e-7;
  const result = murnaghanResponse(F, c);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const plus = structuredClone(F), minus = structuredClone(F); plus[i][j] += h; minus[i][j] -= h;
    const derivative = (murnaghanResponse(plus, c).energy_j_m3 - murnaghanResponse(minus, c).energy_j_m3) / (2 * h);
    close(derivative, result.P_pa[i][j], 2e-6, .2);
  }
});

test('material response is objective under superposed rigid rotation and has a stress-free reference state', () => {
  const angle = .73, R = [[Math.cos(angle), -Math.sin(angle), 0], [Math.sin(angle), Math.cos(angle), 0], [0, 0, 1]];
  const F = [[1.001, .0002, .0001], [0, .9997, .0003], [-.0001, 0, 1.0005]];
  const a = murnaghanResponse(F, c), b = murnaghanResponse(product(R, F), c);
  close(b.energy_j_m3, a.energy_j_m3);
  const rotatedP = product(R, a.P_pa);
  b.P_pa.forEach((row, i) => row.forEach((v, j) => close(v, rotatedP[i][j], 1e-9, 1e-3)));
  for (const rotation of [I, R]) {
    const zero = murnaghanResponse(rotation, c);
    close(zero.energy_j_m3, 0, 0, 1e-15);
    zero.P_pa.flat().forEach(value => close(value, 0, 0, 1e-3));
  }
});

test('infinitesimal limit agrees with Hooke elasticity; disabling third-order energy keeps geometric nonlinearity', () => {
  const H = [[.3, .4, -.2], [.1, -.1, .25], [-.1, .2, .2]], eps = 1e-7;
  const reference = linearElasticResponse(H, c).stress_pa, response = murnaghanResponse(mat((i, j) => I[i][j] + eps * H[i][j]), c);
  response.P_pa.forEach((row, i) => row.forEach((v, j) => close(v / eps, reference[i][j], 2e-6)));
  const h = .001, F = [[1 + h, 0, 0], [0, 1, 0], [0, 0, 1]];
  const geometric = murnaghanResponse(F, c, { materialNonlinearity: false }).P_pa[0][0];
  const modulus = c.lambda_pa + 2 * c.mu_pa;
  close(geometric, modulus * (h + 1.5 * h ** 2 + .5 * h ** 3));
  assert.ok(Math.abs(geometric - modulus * h) > 1e5);
});

test('material contribution has the expected quadratic harmonic scaling at a point, without claiming propagation', () => {
  const samples = 256;
  function second(amplitude) {
    let real = 0, imag = 0;
    for (let k = 0; k < samples; k++) {
      const phase = k * 2 * Math.PI / samples, F = [[1 + amplitude * Math.sin(phase), 0, 0], [0, 1, 0], [0, 0, 1]];
      const materialP = murnaghanResponse(F, c).P_pa[0][0] - murnaghanResponse(F, c, { materialNonlinearity: false }).P_pa[0][0];
      real += materialP * Math.cos(2 * phase); imag -= materialP * Math.sin(2 * phase);
    }
    return Math.hypot(real, imag) * 2 / samples;
  }
  const a = second(1e-5), b = second(2e-5);
  close(a, Math.abs(c.l_pa + 2 * c.m_pa) * 1e-10 / 2, 1e-7);
  close(b / a, 4, 1e-6);
});

test('invalid constitutive inputs and strains outside the declared reference implementation are rejected', () => {
  for (const F of [[[1, 0], [0, 1]], [[NaN, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1.1, 0, 0], [0, 1, 0], [0, 0, 1]]]) assert.throws(() => murnaghanResponse(F, c));
  assert.throws(() => murnaghanResponse(I, { ...c, l_pa: '-143e9' }));
  assert.throws(() => murnaghanResponse(I, { ...c, mu_pa: 0 }));
});

test('literature provenance must match saved material; no third-order constants leak into the old linear protocol', () => {
  const scene = defaultResearchCase(), parameters = compileResearchCase(scene);
  assert.equal(scene.material_reference.usage, 'linear_constants_only');
  for (const name of ['l_pa', 'm_pa', 'n_pa', 'material_reference', 'materialNonlinearity']) assert.equal(Object.hasOwn(parameters, name), false);
  for (const change of [c => { c.material.rho_kg_m3 = 2701; }, c => { c.material_reference.record_sha256 = '0'.repeat(64); }, c => { c.material_reference.usage = 'nonlinear'; }, c => { c.material_reference.extra = true; }]) {
    const altered = structuredClone(scene); change(altered); assert.throws(() => compileResearchCase(altered));
  }
  delete scene.material_reference; scene.material.rho_kg_m3 = 2701;
  assert.equal(compileResearchCase(scene).rho, 2701);
  // The historical reference case remains interpretable and retains its old speeds.
  scene.material = { rho_kg_m3: 2700, cl_m_s: 6320, ct_m_s: 3130 };
  assert.equal(compileResearchCase(scene).cl, 6320);
});
