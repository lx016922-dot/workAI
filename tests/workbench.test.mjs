import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { request as httpRequest } from 'node:http';
import { createApp } from '../server/app.mjs';
import { JobStore } from '../server/store.mjs';
import { AppError, RemoteClient, parseSmokeCsv, decodeText, validateJob } from '../server/remote.mjs';

const id = 'comsol-' + 'a'.repeat(32), second = 'comsol-' + 'b'.repeat(32);
const csv = Buffer.from('frequency_hz,R_real,R_imag,T_stress_real,T_stress_imag\r\n3000000,0.3,0.4,1,0\r\n6000000,0.6,0.8,1.2,0.5\r\n');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const completed = jobId => ({ schema: 'workai.comsol.http.job_status.v1', job_id: jobId, kind: 'smoke', phase: 'completed', artifacts: [{ name: 'coarse.csv', bytes: csv.length, sha256: hash(csv) }], state: { started_at_utc: '2026-10-22T03:00:00Z', updated_at_utc: '2026-10-22T03:00:20Z' } });
async function fixture(t, overrides = {}, options = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'workai-web-test-'));
  const store = new JobStore(join(directory, 'jobs'), join(directory, 'requests'));
  await store.init();
  const calls = { submit: 0, health: 0 };
  const remote = {
    health: async type => { calls.health++; return { schema: `test.${type}`, model_ready: false }; },
    job: async jobId => completed(jobId),
    submit: async jobId => { calls.submit++; return { ...completed(jobId), phase: 'starting', artifacts: [] }; },
    artifact: async (jobId, name) => ({ name, sha256: hash(csv), bytes: csv }),
    ...overrides,
  };
  const server = createApp({ remote, store, ...options });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); await rm(directory, { recursive: true, force: true }); });
  const get = path => fetch(origin + path);
  const bootstrap = await get('/api/bootstrap');
  const csrf = bootstrap.ok ? (await bootstrap.json()).csrf_token : '';
  const post = (path, body, headers = {}) => fetch(origin + path, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', 'X-WorkAI-CSRF': csrf, ...headers }, body: JSON.stringify(body) });
  return { directory, store, remote, calls, origin, server, get, post };
}

test('serves the workbench without exposing private paths or the checkout', async t => {
  const f = await fixture(t);
  const page = await f.get('/'); assert.equal(page.status, 200); assert.match(await page.text(), /超声研究工作台/);
  assert.match(page.headers.get('content-security-policy'), /frame-ancestors 'none'/);
  for (const path of ['/private/client-private.xml', '/.git/config', '/server/index.mjs', '/package.json']) assert.equal((await f.get(path)).status, 404);
});
test('rejects cross-origin submissions, missing CSRF, and DNS rebinding hosts', async t => {
  const f = await fixture(t);
  assert.equal((await f.post('/api/jobs', { job_id: id, kind: 'smoke' }, { Origin: 'https://untrusted.example' })).status, 403);
  assert.equal((await f.post('/api/jobs', { job_id: id, kind: 'smoke' }, { 'X-WorkAI-CSRF': '' })).status, 403);
  // Fetch normalizes Host; send the actual conflicting wire header here.
  const rebound = await new Promise((resolve, reject) => {
    const request = httpRequest(f.origin + '/api/bootstrap', { headers: { Host: 'untrusted.example' } }, response => { response.resume(); resolve(response.statusCode); });
    request.on('error', reject); request.end();
  });
  assert.equal(rebound, 403);
  assert.equal(f.calls.submit, 0);
});
test('password protects HTML, API responses, and files', async t => {
  const password = 'only-a-generated-test-password';
  const f = await fixture(t, {}, { password });
  for (const path of ['/', '/api/health', `/api/jobs/${id}/files/coarse.csv`]) {
    const response = await f.get(path); assert.equal(response.status, 401); assert.match(response.headers.get('www-authenticate'), /Basic/);
  }
  const response = await fetch(f.origin + '/api/bootstrap', { headers: { Authorization: 'Basic ' + Buffer.from('workai:' + password).toString('base64') } });
  assert.equal(response.status, 200);
});
test('health failure remains visible while other services are usable; no solve is triggered', async t => {
  const f = await fixture(t, { health: async type => { if (type === 'volume') throw new AppError(502, '离线'); return { status: 'ready_for_job' }; } });
  const result = await (await f.get('/api/health')).json();
  assert.equal(result.services.base.verified, true); assert.equal(result.services.volume.ok, false); assert.equal(result.services.volume.verified, false);
  assert.equal(f.calls.submit, 0);
});
test('imports an existing remote job and stores its verified final status', async t => {
  const f = await fixture(t);
  const response = await f.post('/api/jobs/import', { job_id: id }); assert.equal(response.status, 200);
  const records = await (await f.get('/api/jobs')).json();
  assert.equal(records.jobs.length, 1); assert.equal(records.jobs[0].last_status.phase, 'completed');
  assert.equal(f.calls.submit, 0);
});
test('retains a submission before sending and prevents duplicate submissions', async t => {
  const f = await fixture(t);
  f.remote.submit = async jobId => {
    assert.equal((await f.store.get(jobId)).submission_state, 'unknown');
    f.calls.submit++;
    return { ...completed(jobId), phase: 'starting', artifacts: [] };
  };
  assert.equal((await f.post('/api/jobs', { job_id: id, kind: 'smoke' })).status, 202);
  assert.equal((await f.post('/api/jobs', { job_id: id, kind: 'smoke' })).status, 409);
  assert.equal((await f.post('/api/jobs', { job_id: second, kind: 'smoke' })).status, 409);
  assert.equal(f.calls.submit, 1);
  await f.get(`/api/jobs/${id}`);
  assert.equal((await f.post('/api/jobs', { job_id: second, kind: 'smoke' })).status, 202);
});
test('a lost POST response stays unresolved and cannot be retried, including after reloading the store', async t => {
  const f = await fixture(t, { submit: async () => { throw new AppError(502, '连接中断'); } });
  const result = await (await f.post('/api/jobs', { job_id: id, kind: 'smoke' })).json();
  assert.equal(result.accepted, null); assert.equal(result.job_id, id);
  const reopened = new JobStore(join(f.directory, 'jobs'), join(f.directory, 'requests')); await reopened.init();
  assert.equal((await reopened.get(id)).submission_state, 'unknown');
  assert.equal((await f.post('/api/jobs', { job_id: id, kind: 'smoke' })).status, 409);
  assert.equal((await f.post('/api/jobs', { job_id: second, kind: 'smoke' })).status, 409);
});
test('rejects invalid IDs, extra command fields, and unsupported job types before a remote POST', async t => {
  const f = await fixture(t);
  for (const body of [{ job_id: '../../private', kind: 'smoke' }, { job_id: id, kind: 'smoke', command: 'anything' }, { job_id: id, kind: 'plate_reference' }]) {
    assert.equal((await f.post('/api/jobs', body)).status, 400);
  }
  assert.equal(f.calls.submit, 0);
});
test('returns verified result bytes and previews without executing content', async t => {
  const f = await fixture(t);
  const response = await f.get(`/api/jobs/${id}/files/coarse.csv`);
  assert.equal(response.status, 200); assert.equal(response.headers.get('x-workai-artifact-sha256'), hash(csv));
  assert.deepEqual(Buffer.from(await response.arrayBuffer()), csv);
  const preview = await (await f.get(`/api/jobs/${id}/preview/coarse.csv`)).json();
  assert.equal(preview.verified, true); assert.match(preview.text, /frequency_hz/);
  assert.equal((await f.get(`/api/jobs/${id}/files/private.xml`)).status, 400);
});
test('frequency preview computes complex magnitudes from real outputs', async t => {
  const f = await fixture(t);
  const result = await (await f.get(`/api/jobs/${id}/series`)).json();
  assert.equal(result.coarse[0].frequency_mhz, 3); assert.equal(result.coarse[0].reflection, 0.5);
  assert.ok(Math.abs(result.fine[1].transmission - 1.3) < 1e-12); assert.equal(result.verified, true);
});
test('CSV parser rejects missing, empty, or non-finite scientific values', () => {
  for (const body of ['', 'wrong\n1,2,3', 'frequency_hz,R_real,R_imag,T_stress_real,T_stress_imag\n3000000,,0,1,0', 'frequency_hz,R_real,R_imag,T_stress_real,T_stress_imag\n3000000,NaN,0,1,0']) {
    assert.throws(() => parseSmokeCsv(Buffer.from(body)), AppError);
  }
  assert.equal(decodeText(Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from('已完成', 'utf16le')])), '已完成');
});
test('bridge refuses unsigned responses and mismatched job identities', async () => {
  const remote = new RemoteClient('/tmp/unused-workai-root');
  remote.command = async () => ({ http_status: 200, server_signature_verified: false, body: completed(id) });
  await assert.rejects(remote.job(id), /签名/);
  remote.command = async () => ({ http_status: 200, server_signature_verified: true, body: completed(second) });
  await assert.rejects(remote.job(id), /不匹配/);
  assert.throws(() => validateJob({ ...completed(id), artifacts: null }, id));
});
test('bridge refuses modified cached artifacts even if the helper reports success', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'workai-artifact-test-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const destination = join(directory, 'requests', id, 'artifacts'); await mkdir(destination, { recursive: true });
  const path = join(destination, 'coarse.csv'); await writeFile(path, csv);
  const remote = new RemoteClient(directory);
  remote.command = async () => ({ job_id: id, phase: 'completed', server_signature_verified: true, artifacts: [{ name: 'coarse.csv', bytes: csv.length, sha256: hash(csv), path }] });
  assert.deepEqual((await remote.artifact(id, 'coarse.csv')).bytes, csv);
  const tampered = Buffer.from(csv); tampered[tampered.length - 3] ^= 1; await writeFile(path, tampered);
  await assert.rejects(remote.artifact(id, 'coarse.csv'), /校验失败/);
});
test('discovers retained request IDs without overwriting existing records or importing arbitrary files', async t => {
  const f = await fixture(t);
  const requests = join(f.directory, 'requests'); await mkdir(join(requests, id), { recursive: true });
  await writeFile(join(requests, id, 'submission.json'), JSON.stringify({ job_id: id, kind: 'smoke', prepared_at_utc: '2026-10-22T03:00:00Z' }));
  await mkdir(join(requests, 'not-a-job'));
  await f.store.init(); assert.equal((await f.store.list()).length, 1);
  await f.store.update(id, { last_status: completed(id) }); await f.store.init();
  assert.equal((await f.store.get(id)).last_status.phase, 'completed');
  const onDisk = await readFile(join(f.directory, 'jobs', `${id}.json`), 'utf8'); assert.doesNotMatch(onDisk, /private_key/);
});
