import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { defaultResearchCase, compileResearchCase, verifyResearchRequest, parseResearchCsv, RESEARCH_SOURCE, researchReady } from '../server/research.mjs';
import { ModelStore } from '../server/models.mjs';
import { JobStore } from '../server/store.mjs';
import { createApp } from '../server/app.mjs';

const hash = value => createHash('sha256').update(value).digest('hex');
const id = 'comsol-' + 'f'.repeat(32);
const health = { schema: 'workai.comsol.research.health.v1', status: 'ready_for_job', source_sha256: RESEARCH_SOURCE };
function inputs() {
  const scene = defaultResearchCase();
  scene.acquisition.duration_us = .1; scene.acquisition.sample_step_us = .05;
  scene.numerics.field_frames = 2; scene.source.frequency_mhz = 20; scene.source.cycles = 1;
  scene.schema = 'workai.original.scene.fixture';
  // Use complete parameter metadata here without pretending this tiny fixture is
  // a physically useful sample. Production compilation enforces carrier sampling.
  const parameters = { ...compileResearchCase(defaultResearchCase()), duration_us: .1, sample_step_us: .05, receiver_count: 1 };
  const request = { job_id: id, kind: 'research_reference', source_sha256: RESEARCH_SOURCE, scene_json: JSON.stringify(scene), parameters_json: JSON.stringify(parameters) };
  request.scene_sha256 = hash(request.scene_json); request.parameters_sha256 = hash(request.parameters_json);
  const receipt = Object.entries({ schema: 'workai.comsol.research.receipt.v1', status: 'completed', method: 'comsol_research_2d_axisymmetric_weakform_v1', job_id: id, source_sha256: request.source_sha256, scene_sha256: request.scene_sha256, parameters_sha256: request.parameters_sha256 }).map(([k, v]) => `${k}=${v}`).join('\n');
  const csv = 'shot,rx,time_s,u_m,v_m,w_m,temperature_rise_k\n0,0,0,0,0,0,0\n0,0,5e-8,1e-9,-2e-9,0,0\n0,0,1e-7,2e-9,-3e-9,0,0\n';
  return { parameters, request, receipt, csv };
}
test('research compilation preserves units, receiver ordering and the original linear model semantics', () => {
  const scene = defaultResearchCase(), p = compileResearchCase(scene);
  assert.equal(p.height_mm, scene.geometry.thickness_mm);
  assert.equal(p.tx0_f0, .5); assert.equal(p.tx0_amplitude_pa, 100000);
  assert.equal(p.tx0_direction, 1); assert.equal(p.shot_count, 1); assert.equal(p.tx0_frequency_count, 1);
  assert.deepEqual([p.rx0_x_mm, p.rx1_x_mm, p.rx2_x_mm], [8, 12, 16]);
  assert.equal(p.axisymmetric, 0); assert.equal(p.thermal, 0); assert.equal(p.notch_enabled, 0);
  assert.equal(p.element_order, 4); assert.equal(p.field_frames, 7);
  assert.equal(researchReady(health), true);
  assert.equal(researchReady({ ...health, source_sha256: '0'.repeat(64) }), false);
});
test('unsupported nonlinear inputs and invalid scientific discretization cannot be submitted as linear research', () => {
  for (const alter of [
    c => { c.constitutive = 'murnaghan'; }, c => { c.material.l = -300e9; },
    c => { c.source.frequency_mhz = '0.5'; }, c => { c.source.direction = 1; },
    c => { c.receivers[0].x_mm = 21; }, c => { c.receivers[1].id = 'rx0'; },
    c => { c.numerics.fine_h_mm = c.numerics.coarse_h_mm; },
    c => { c.acquisition.duration_us = 12.01; }, c => { c.numerics.field_frames = 8; },
    c => { c.numerics.field_nx = 101; c.numerics.field_ny = 41; },
    c => { c.source.delay_us = 10; }, c => { c.material.ct_m_s = 10000; },
  ]) { const c = defaultResearchCase(); alter(c); assert.throws(() => compileResearchCase(c)); }
});
test('research result validation binds the receipt and both serialized inputs to the requested job and source', () => {
  const { request, receipt } = inputs();
  assert.equal(verifyResearchRequest(JSON.stringify(request), receipt, id).source_sha256, RESEARCH_SOURCE);
  for (const bad of [{ ...request, source_sha256: '0'.repeat(64) }, { ...request, scene_json: '{}' }, { ...request, parameters_json: '{}' }, { ...request, job_id: 'comsol-' + 'e'.repeat(32) }]) assert.throws(() => verifyResearchRequest(JSON.stringify(bad), receipt, id));
  assert.throws(() => verifyResearchRequest(JSON.stringify(request), receipt.replace('status=completed', 'status=failed'), id));
  assert.throws(() => verifyResearchRequest(JSON.stringify(request), receipt.replace(`parameters_sha256=${request.parameters_sha256}`, `parameters_sha256=${'0'.repeat(64)}`), id));
});
test('native scene and solver parameters must agree even when both hashes are internally valid', () => {
  const scene = defaultResearchCase(), parameters = compileResearchCase(scene);
  parameters.tx0_amplitude_pa *= 2;
  const request = { job_id: id, kind: 'research_reference', source_sha256: RESEARCH_SOURCE, scene_json: JSON.stringify(scene), parameters_json: JSON.stringify(parameters) };
  request.scene_sha256 = hash(request.scene_json); request.parameters_sha256 = hash(request.parameters_json);
  const receipt = Object.entries({ schema: 'workai.comsol.research.receipt.v1', status: 'completed', method: 'comsol_research_2d_axisymmetric_weakform_v1', job_id: id, source_sha256: request.source_sha256, scene_sha256: request.scene_sha256, parameters_sha256: request.parameters_sha256 }).map(([k, v]) => `${k}=${v}`).join('\n');
  assert.throws(() => verifyResearchRequest(JSON.stringify(request), receipt, id), /实际研究参数不一致/);
});
test('A-scan parsing preserves signs and physical units; rejects partial, duplicated, out-of-order and nonfinite samples', () => {
  const { csv, parameters } = inputs();
  const traces = parseResearchCsv(csv, parameters);
  assert.equal(traces.length, 1); assert.equal(traces[0].x_mm, 8);
  assert.ok(Math.abs(traces[0].points[1].time_us - .05) < 1e-12);
  assert.deepEqual([traces[0].points[1].u_nm, traces[0].points[1].v_nm, traces[0].points[1].w_nm], [1, -2, 0]);
  for (const bad of [csv.replace('5e-8', '0'), csv.replace('1e-7', '3e-8'), csv.replace('1e-9', 'NaN'), csv.replace('1e-9', ''), csv.trim().split('\n').slice(0, -1).join('\n'), csv.replace('0,0,1e-7', '0,1,1e-7')]) assert.throws(() => parseResearchCsv(bad, parameters));
});
async function appFixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'workai-research-test-'));
  const remote = { health: async () => health, validateCase: async file => ({ valid: true, case_sha256: hash(await readFile(file)) }) };
  const models = new ModelStore(join(directory, 'models'), remote); await models.init();
  const saved = await models.save({ name: '线性参考', case: defaultResearchCase(), expected_revision: null });
  const store = new JobStore(join(directory, 'jobs'), join(directory, 'requests')); await store.init();
  const server = createApp({ remote, models, store }); await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); await rm(directory, { force: true, recursive: true }); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const { csrf_token } = await (await fetch(origin + '/api/bootstrap')).json();
  const submit = kind => fetch(origin + '/api/studies', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', 'X-WorkAI-CSRF': csrf_token }, body: JSON.stringify({ job_id: id, model_revision: saved.revision, kind }) });
  return { remote, models, store, saved, submit, directory, origin };
}
test('a changed research source or mismatched job kind is blocked before snapshot/submission', async t => {
  const f = await appFixture(t);
  f.remote.submitStudy = () => { assert.fail('Unexpected submission'); };
  assert.equal((await f.submit('plate_smoke')).status, 422);
  f.remote.health = async () => ({ ...health, source_sha256: '0'.repeat(64) });
  assert.equal((await f.submit('research_reference')).status, 409);
  assert.equal((await f.store.list()).length, 0);
  assert.deepEqual(await readdir(join(f.directory, 'models')), ['current.json']);
});
test('research dispatch retains immutable input and does not retry an uncertain submission', async t => {
  const f = await appFixture(t); let calls = 0;
  f.remote.submitStudy = async (jobId, file, kind) => {
    calls++; assert.equal(kind, 'research_reference'); assert.equal(jobId, id);
    assert.equal(hash(await readFile(file)), f.saved.case_sha256);
    assert.equal((await f.store.get(id)).submission_state, 'unknown');
    throw new Error('Synthetic lost response');
  };
  const response = await f.submit('research_reference');
  assert.equal(response.status, 202); assert.equal((await response.json()).accepted, null);
  assert.equal((await f.submit('research_reference')).status, 409); assert.equal(calls, 1);
});
test('A-scan endpoint requires a completed research job and verified, matching receipt/input files', async t => {
  const f = await appFixture(t), fixture = inputs();
  const files = { 'request.json': JSON.stringify(fixture.request), 'completed.properties': fixture.receipt, 'coarse.csv': fixture.csv, 'fine.csv': fixture.csv, 'timecheck.csv': fixture.csv };
  f.remote.job = async () => ({ kind: 'research_reference', phase: 'running' });
  f.remote.artifact = async (_, name) => ({ name, bytes: Buffer.from(files[name]), sha256: hash(files[name]) });
  assert.equal((await fetch(f.origin + `/api/jobs/${id}/waveforms`)).status, 422);
  f.remote.job = async () => ({ kind: 'research_reference', phase: 'completed' });
  const response = await fetch(f.origin + `/api/jobs/${id}/waveforms`);
  assert.equal(response.status, 200); assert.equal((await response.json()).levels.timecheck[0].points[1].v_nm, -2);
  files['completed.properties'] = fixture.receipt.replace(`job_id=${id}`, 'job_id=comsol-' + '0'.repeat(32));
  assert.equal((await fetch(f.origin + `/api/jobs/${id}/waveforms`)).status, 422);
});
