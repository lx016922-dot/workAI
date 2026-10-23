import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { defaultNonlinearCase, NONLINEAR_SCENE, NONLINEAR_METHOD, CONTROLS } from '../server/nonlinear.mjs';
import { compileResearchCase, verifyResearchRequest, RESEARCH_CURRENT_SOURCE, RESEARCH_SOURCE, researchReady } from '../server/research.mjs';
import { analyzeHarmonics, s0Dispersion, windowedCoefficient } from '../server/physics/harmonics.mjs';
import { RemoteClient } from '../server/remote.mjs';
import { attachSpatialReference } from '../server/physics/convergence.mjs';
const hash = s => createHash('sha256').update(s).digest('hex');

test('artifact downloads are serialized and duplicate concurrent reads share one verified operation', async () => {
  const remote = new RemoteClient('/tmp/workai-test-unused'), id = 'comsol-' + '7'.repeat(32);
  let active = 0, maximum = 0, calls = 0;
  remote.readArtifact = async (_, name) => { calls++; active++; maximum = Math.max(maximum, active); await new Promise(resolve => setTimeout(resolve, 5)); active--; if (name === 'coarse.csv') throw new Error('Synthetic read failure'); return { name }; };
  const results = await Promise.allSettled([remote.artifact(id, 'coarse.csv'), remote.artifact(id, 'fine.csv'), remote.artifact(id, 'fine.csv'), remote.artifact(id, 'timecheck.csv')]);
  assert.equal(results[0].status, 'rejected'); assert.equal(calls, 3); assert.equal(maximum, 1);
  assert.strictEqual(results[1].value, results[2].value); assert.equal(results[3].value.name, 'timecheck.csv');
  assert.equal(remote.artifactReads.size, 0);
});

test('nonlinear transport binds five independent single-carrier controls without repurposing thermal fields', () => {
  const c = defaultNonlinearCase(), p = compileResearchCase(c);
  assert.equal(p.shot_count, 5); assert.equal(p.thermal, 0); assert.equal(p.conductivity, 0);
  for (let i = 0; i < 5; i++) { assert.equal(p[`tx${i}_frequency_count`], 1); assert.equal(p[`tx${i}_f0`], .4); assert.equal(p[`tx${i}_amplitude_pa`], i >= 3 ? 1e7 : 5e6); }
  assert.equal(c.nonlinear.n_pa, -345e9);
  const health = { schema: 'workai.comsol.research.health.v1', status: 'ready_for_job', source_sha256: RESEARCH_SOURCE };
  assert.equal(researchReady(health, NONLINEAR_SCENE), false);
  assert.equal(researchReady({ ...health, source_sha256: RESEARCH_CURRENT_SOURCE }, NONLINEAR_SCENE), true);
});
test('nonlinear input rejects identity, definition, sampling, reflection and aggregate export mismatches', () => {
  for (const alter of [c => c.nonlinear.l_pa *= 1e-9, c => c.nonlinear.definition = 'Landau', c => c.nonlinear.controls.reverse(),
    c => c.material_reference.usage = 'linear_constants_only', c => c.source.profile = 'top_gaussian', c => c.source.frequency_mhz = 3,
    c => c.source.amplitude_pa = 6e6, c => c.geometry.length_mm = 60, c => c.numerics.max_step_us = .1,
    c => c.numerics.field_nx = 101, c => c.source.cycles = 2, c => c.nonlinear.expression = 'execute()', c => c.material.ct_m_s = 3130]) {
    const c = defaultNonlinearCase(); alter(c); assert.throws(() => compileResearchCase(c));
  }
});
function fixture() {
  const scene = defaultNonlinearCase(), p = compileResearchCase(scene), id = 'comsol-' + '3'.repeat(32);
  const req = { job_id: id, kind: 'research_reference', source_sha256: RESEARCH_CURRENT_SOURCE, scene_json: JSON.stringify(scene), parameters_json: JSON.stringify(p) };
  req.scene_sha256 = hash(req.scene_json); req.parameters_sha256 = hash(req.parameters_json);
  const rec = { job_id: id, source_sha256: req.source_sha256, scene_sha256: req.scene_sha256, parameters_sha256: req.parameters_sha256,
    schema: 'workai.comsol.research.receipt.v1', status: 'completed', method: NONLINEAR_METHOD, controls: CONTROLS.join(','), constitutive: scene.constitutive,
    material_record_sha256: scene.material_reference.record_sha256, quadrature_order: '20', source_profile: 'left_edge_uniform_nominal_traction', energy_definition: 'murnaghan_invariants_reference_volume', spatial_assumption: 'plane_strain_z_invariant', mph_solution_state: 'cleared', mph_reopen_verified: 'true' };
  for (const k of ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']) rec[k] = String(scene.nonlinear[k]);
  for (const level of ['coarse', 'fine', 'timecheck']) for (let i = 0; i < 5; i++) { rec[`${level}_shot${i}_max_green_strain`] = '.0002'; rec[`${level}_shot${i}_min_J`] = '.9998'; }
  return { scene, p, id, req, rec };
}
test('nonlinear receipt verifies constitutive parameters, control order and all 15 strain checks', () => {
  const { id, req, rec } = fixture(), text = r => Object.entries(r).map(([k, v]) => `${k}=${v}`).join('\n');
  const check = (r = req, receipt = rec) => verifyResearchRequest(JSON.stringify(r), text(receipt), id);
  assert.equal(check().constitutive, 'murnaghan_green_lagrange');
  assert.equal(check(req, { ...rec, completed_at_utc: '2026-10-22T07\\:11\\:31Z' }).receipt.completed_at_utc, '2026-10-22T07:11:31Z');
  for (const [key, value] of [['method', 'comsol_research_2d_axisymmetric_weakform_v1'], ['l_pa', '-1.43e2'], ['controls', CONTROLS.toReversed().join(',')], ['fine_shot2_min_J', '-1'], ['coarse_shot4_max_green_strain', '.01']]) assert.throws(() => check(req, { ...rec, [key]: value }));
  const missing = { ...rec }; delete missing.timecheck_shot4_max_green_strain; assert.throws(() => check(req, missing));
  assert.throws(() => check({ ...req, source_sha256: RESEARCH_SOURCE }, { ...rec, source_sha256: RESEARCH_SOURCE }));
  const altered = { ...req, parameters_json: req.parameters_json.replace('"tx4_f0":0.4', '"tx4_f0":0.8') };
  altered.parameters_sha256 = hash(altered.parameters_json); assert.throws(() => check(altered, { ...rec, parameters_sha256: altered.parameters_sha256 }));
});
test('Rayleigh-Lamb S0 agrees with thin-plate limit and independent SciPy roots; reports dispersion at 2f', () => {
  const c = defaultNonlinearCase(), { cl_m_s: cl, ct_m_s: ct } = c.material;
  const low = s0Dispersion(1000, .001, cl, ct), thin = 2 * ct * Math.sqrt(1 - ct * ct / (cl * cl));
  assert.ok(Math.abs(low.phase_velocity_m_s / thin - 1) < 1e-5);
  const fundamental = s0Dispersion(.4e6, .001, cl, ct), second = s0Dispersion(.8e6, .001, cl, ct);
  assert.ok(Math.abs(fundamental.wavenumber_rad_m - 463.7687684350077) < 1e-7);
  assert.ok(Math.abs(second.wavenumber_rad_m - 935.2803370934298) < 1e-7);
  assert.ok(second.group_velocity_m_s < fundamental.group_velocity_m_s);
  assert.equal(s0Dispersion(4e6, .001, cl, ct), null);
});
test('complex harmonic projection preserves amplitude and phase while rejecting incomplete windows', () => {
  const points = Array.from({ length: 261 }, (_, i) => ({ time_us: i * .1 }));
  const phase = .37, values = points.map(p => 90 + 3 * Math.cos(2 * Math.PI * .8 * p.time_us + phase));
  const out = windowedCoefficient(points, values, .8, 5, 20);
  assert.ok(Math.abs(out.amplitude_nm - 3) < 1e-12);
  assert.ok(Math.abs(out.re_nm - 3 * Math.cos(phase)) < 1e-12);
  assert.ok(Math.abs(out.im_nm - 3 * Math.sin(phase)) < 1e-12);
  assert.throws(() => windowedCoefficient(points, values, .8, 20, 30));
});
test('material increment is subtracted before projection; quadratic ratio and complex discretization errors stay separate', () => {
  const c = defaultNonlinearCase(), levels = {};
  const k = s0Dispersion(.4e6, .001, c.material.cl_m_s, c.material.ct_m_s).wavenumber_rad_m;
  for (const level of ['coarse', 'fine', 'timecheck']) {
    levels[level] = [];
    for (let shot = 0; shot < 5; shot++) for (let rx = 0; rx < 3; rx++) {
      const amplitude = shot >= 3 ? 2 : 1, geom = shot === 0 ? 0 : .01, mat = shot === 2 || shot === 4 ? .02 : 0;
      const points = Array.from({ length: 261 }, (_, i) => {
        const t = i * .1, phase = k * c.receivers[rx].x_mm * .001, harmonic = Math.cos(2 * Math.PI * .8 * t + .3 - 2 * phase);
        return { time_us: t, u_nm: amplitude * (Math.sin(2 * Math.PI * .4 * t - phase) + .4 * harmonic) + amplitude ** 2 * (rx + 1) * (geom + mat) * harmonic, v_nm: 0, w_nm: 0 };
      }); levels[level].push({ shot, rx, points });
    }
  }
  const receipt = { coarse_elements: '100', fine_elements: '400' };
  const a = analyzeHarmonics(c, levels, receipt);
  assert.equal(a.all_checks_within_5_percent, true);
  assert.equal(analyzeHarmonics(c, levels, { coarse_elements: '234', fine_elements: '234' }).all_checks_within_5_percent, false);
  for (const r of a.rows) { assert.ok(Math.abs(r.levels.fine.amplitude_scaling - 4) < 1e-10); assert.ok(Math.abs(r.levels.fine.material_second.amplitude_nm - .02 * (r.rx + 1)) < 1e-8); }
  assert.equal(a.linear_phase_checks.length, 2);
  for (const r of a.linear_phase_checks) assert.ok(Math.abs(r.relative_wavenumber_difference) < 1e-10);
  // A pure phase error must fail even when the scalar amplitude is unchanged.
  for (const tr of levels.coarse.filter(t => t.shot === 2)) for (const p of tr.points) { const phase = 2 * k * c.receivers[tr.rx].x_mm * .001; p.u_nm += .02 * (tr.rx + 1) * (Math.cos(2 * Math.PI * .8 * p.time_us + 1.3 - phase) - Math.cos(2 * Math.PI * .8 * p.time_us + .3 - phase)); }
  assert.equal(analyzeHarmonics(c, levels, receipt).all_checks_within_5_percent, false);
});

function spatialFixture() {
  const scene = defaultNonlinearCase();
  const target = { verified: true, job_id: 'target', source_sha256: RESEARCH_CURRENT_SOURCE, scene,
    receipt: { scene_sha256: 'a', fine_internal_step_s: '2.5e-8', fine_elements: '234', fine_mesh_h_m: '.002' },
    harmonics: { available: true, rows: [{ rx: 0, x_mm: 20, available: true, window_us: [1, 2], mesh_relative_complex_difference: 0,
      time_relative_complex_difference: .03, levels: { fine: { material_second: { re_nm: 2, im_nm: 0, amplitude_nm: 2 } }, timecheck: { amplitude_scaling: 4 } } }] } };
  const reference = { verified: true, job_id: 'reference', source_sha256: RESEARCH_CURRENT_SOURCE, scene: structuredClone(scene),
    receipt: { scene_sha256: 'b', timecheck_internal_step_s: '2.5e-8', timecheck_elements: '640', timecheck_mesh_h_m: '.001' },
    harmonics: { available: true, rows: [{ rx: 0, x_mm: 20, available: true, window_us: [1, 2], levels: { timecheck: { material_second: { re_nm: 2, im_nm: .02, amplitude_nm: Math.hypot(2, .02) } } } }] }, artifacts: [] };
  const record = { target_job_id: 'target', reference_job_id: 'reference', target_scene_sha256: 'a', reference_scene_sha256: 'b', target_level: 'fine', reference_level: 'timecheck' };
  return { target, reference, record };
}
test('cross-job spatial validation uses actual refinement at the same time step and records tighter tolerances', () => {
  const { target, reference, record } = spatialFixture(), result = attachSpatialReference(target, reference, record);
  assert.equal(result.all_checks_within_5_percent, true); assert.equal(result.mesh_refinement_demonstrated, true);
  assert.ok(Math.abs(result.rows[0].mesh_relative_complex_difference - .02 / Math.hypot(2, .02)) < 1e-15);
  assert.equal(result.spatial_reference.reference_relative_tolerance, 2.5e-9);
  assert.equal(target.harmonics.rows[0].mesh_relative_complex_difference, 0);
});
test('spatial evidence cannot be borrowed from a different material, source, receiver/window, time step or unrefined mesh', () => {
  for (const alter of [r => r.verified = false, r => r.source_sha256 = RESEARCH_SOURCE, r => r.scene.source.frequency_mhz = .8,
    r => r.scene.geometry.thickness_mm = 2, r => r.scene.material_reference.record_sha256 = 'changed', r => r.scene.receivers[0].x_mm = 21,
    r => r.receipt.timecheck_internal_step_s = '1e-8', r => r.receipt.timecheck_elements = '234', r => r.harmonics.rows[0].window_us = [2, 3]]) {
    const f = spatialFixture(); alter(f.reference); assert.throws(() => attachSpatialReference(f.target, f.reference, f.record));
  }
});
