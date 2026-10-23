import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { defaultCase, ModelStore } from '../server/models.mjs';
import { JobStore } from '../server/store.mjs';
import { createApp } from '../server/app.mjs';

async function setup(t) {
  const directory = await mkdtemp(join(tmpdir(), 'workai-model-test-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const remote = {
    validateCase: async file => ({ valid: true, case_sha256: createHash('sha256').update(await readFile(file)).digest('hex') }),
    health: async () => ({ model_ready: false }),
    submitStudy: async () => { throw new Error('Remote submit must not be reached'); },
  };
  const models = new ModelStore(join(directory, 'models'), remote); await models.init();
  return { directory, models, remote };
}
test('model save preserves revisions and scientific input after stale or invalid edits', async t => {
  const { models, remote } = await setup(t);
  const saved = await models.save({ name: '测试模型', case: defaultCase(), expected_revision: null });
  assert.equal(saved.validated, true);
  await assert.rejects(models.save({ name: '过时修改', case: defaultCase('volume'), expected_revision: null }), /更新/);
  remote.validateCase = async () => ({ valid: false, validation_error: 'outside approved range' });
  await assert.rejects(models.save({ name: '无效修改', case: {}, expected_revision: saved.revision }), /未通过/);
  assert.deepEqual(await models.get(), saved);
});
test('study snapshots retain the exact validated case and cannot overwrite an existing job case', async t => {
  const { models } = await setup(t);
  const saved = await models.save({ name: '平板', case: defaultCase(), expected_revision: null });
  const id = 'comsol-' + 'c'.repeat(32);
  const snapshot = await models.snapshot(id, saved.revision);
  assert.equal(createHash('sha256').update(await readFile(snapshot.file)).digest('hex'), saved.case_sha256);
  await models.save({ name: '大体积', case: defaultCase('volume'), expected_revision: saved.revision });
  assert.equal(JSON.parse(await readFile(snapshot.file)).schema, 'workai.comsol.plate.case.v1');
  await assert.rejects(models.snapshot(id, saved.revision), /变化/);
});
test('unready volume model is blocked before creating a submission or invoking remote compute', async t => {
  const { directory, models, remote } = await setup(t);
  const saved = await models.save({ name: '大体积', case: defaultCase('volume'), expected_revision: null });
  const store = new JobStore(join(directory, 'jobs'), join(directory, 'requests')); await store.init();
  const server = createApp({ remote, store, models });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const { csrf_token } = await (await fetch(origin + '/api/bootstrap')).json();
  const response = await fetch(origin + '/api/studies', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', 'X-WorkAI-CSRF': csrf_token }, body: JSON.stringify({ job_id: 'comsol-' + 'd'.repeat(32), model_revision: saved.revision, kind: 'plate_smoke' }) });
  assert.equal(response.status, 409); assert.match((await response.json()).error, /大体积模型尚未就绪/);
  assert.equal((await store.list()).length, 0);
  assert.deepEqual(await readdir(join(directory, 'models')), ['current.json']);
});
test('plate study dispatch uses the saved immutable model and retains the job before calling the client', async t => {
  const { directory, models, remote } = await setup(t);
  const saved = await models.save({ name: '已保存平板', case: defaultCase(), expected_revision: null });
  const store = new JobStore(join(directory, 'jobs'), join(directory, 'requests')); await store.init();
  let submissions = 0;
  remote.submitStudy = async (id, file, kind) => {
    submissions++;
    assert.equal((await store.get(id)).submission_state, 'unknown');
    assert.equal((await store.get(id)).model_revision, saved.revision);
    assert.equal(createHash('sha256').update(await readFile(file)).digest('hex'), saved.case_sha256);
    assert.equal(kind, 'plate_reference');
    return { schema: 'workai.comsol.http.job_status.v1', job_id: id, kind, phase: 'starting', artifacts: [] };
  };
  const server = createApp({ remote, store, models });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const { csrf_token } = await (await fetch(origin + '/api/bootstrap')).json();
  const send = () => fetch(origin + '/api/studies', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', 'X-WorkAI-CSRF': csrf_token }, body: JSON.stringify({ job_id: 'comsol-' + 'e'.repeat(32), model_revision: saved.revision, kind: 'plate_reference' }) });
  const response = await send(); assert.equal(response.status, 202); assert.equal((await response.json()).accepted, true);
  assert.equal((await send()).status, 409); assert.equal(submissions, 1);
});
