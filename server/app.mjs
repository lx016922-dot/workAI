import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { AppError, assertJobId, assertArtifact, MAX_ARTIFACT_BYTES, decodeText, parseSmokeCsv } from './remote.mjs';
import { defaultCase } from './models.mjs';
import { aluminumMaterial } from './materials.mjs';
import { readResearchResults } from './research-results.mjs';
import { RESEARCH_CURRENT_SOURCE, isResearchScene, researchReady } from './research.mjs';

const publicRoot = fileURLToPath(new URL('../public/', import.meta.url));
const staticFiles = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/model-ui.js', ['model-ui.js', 'text/javascript; charset=utf-8']],
  ['/research-ui.js', ['research-ui.js', 'text/javascript; charset=utf-8']],
  ['/nonlinear-ui.js', ['nonlinear-ui.js', 'text/javascript; charset=utf-8']],
  ['/material-ui.js', ['material-ui.js', 'text/javascript; charset=utf-8']],
  ['/waveform-ui.js', ['waveform-ui.js', 'text/javascript; charset=utf-8']],
  ['/offline-report.js', ['offline-report.js', 'text/javascript; charset=utf-8']],
  ['/training-ui.js', ['training-ui.js', 'text/javascript; charset=utf-8']],
  ['/solid-ui.js', ['solid-ui.js', 'text/javascript; charset=utf-8']],
  ['/solid-viewer.js', ['solid-viewer.js', 'text/javascript; charset=utf-8']],
  ['/solid-data.js', ['solid-data.js', 'text/javascript; charset=utf-8']],
  ['/spectral-data.js', ['spectral-data.js', 'text/javascript; charset=utf-8']],
  ['/solid-harmonics.js', ['solid-harmonics.js', 'text/javascript; charset=utf-8']],
  ['/solid-study-ui.js', ['solid-study-ui.js', 'text/javascript; charset=utf-8']],
  ['/solid-harmonic-viewer.js', ['solid-harmonic-viewer.js', 'text/javascript; charset=utf-8']],
  ['/solid-harmonic-report.js', ['solid-harmonic-report.js', 'text/javascript; charset=utf-8']],
  ['/plate-dispersion.js', ['plate-dispersion.js', 'text/javascript; charset=utf-8']],
  ['/solid-propagation.js', ['solid-propagation.js', 'text/javascript; charset=utf-8']],
  ['/solid-propagation-viewer.js', ['solid-propagation-viewer.js', 'text/javascript; charset=utf-8']],
  ['/solid-propagation-report.js', ['solid-propagation-report.js', 'text/javascript; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/favicon.svg', ['favicon.svg', 'image/svg+xml']],
]);
const digest = value => createHash('sha256').update(value).digest();
const equal = (a, b) => timingSafeEqual(digest(a), digest(b));
function json(response, status, value) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(value));
}
async function readJson(request,limit=65536) {
  if (!/^application\/json(?:;|$)/i.test(request.headers['content-type'] || '')) throw new AppError(415, '需要 JSON 请求。');
  const chunks = []; let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > limit) throw new AppError(413, '请求内容过大。');
    chunks.push(chunk);
  }
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error();
    return value;
  } catch { throw new AppError(400, '请求内容不是有效 JSON。'); }
}
function exactKeys(value, keys) {
  if (Object.keys(value).length !== keys.length || keys.some(k => !Object.hasOwn(value, k))) throw new AppError(400, '请求字段不正确。');
}

export function createApp({ remote, store, models, solid, spectral, trainingReader = async () => ({ available: false }), password = '', origin = '', connection = {} }) {
  const csrf = randomBytes(32).toString('base64url');
  let submitting = false;
  let healthPromise = null;
  let healthCache = null;
  let healthAt = 0;
  const server = createServer(async (request, response) => {
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('X-Content-Type-Options', 'nosniff');
    response.setHeader('Referrer-Policy', 'no-referrer');
    response.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
    try {
      const localOrigin = `http://127.0.0.1:${server.address().port}`;
      const expectedOrigin = origin || localOrigin;
      const allowedHosts = origin ? [new URL(origin).host] : [new URL(localOrigin).host, `localhost:${server.address().port}`];
      if (!allowedHosts.includes(request.headers.host)) throw new AppError(403, '网站地址不在允许范围。');
      const requestOrigin = request.headers.origin;
      const sameOrigins = origin ? [origin] : [localOrigin, `http://localhost:${server.address().port}`];
      if ((requestOrigin && !sameOrigins.includes(requestOrigin)) || request.headers['sec-fetch-site'] === 'cross-site') throw new AppError(403, '不接受来自其他网站的请求。');
      if (password) {
        const authorization = request.headers.authorization || '';
        const valid = authorization.startsWith('Basic ') && equal(Buffer.from(authorization.slice(6), 'base64').toString('utf8'), `workai:${password}`);
        if (!valid) {
          response.setHeader('WWW-Authenticate', 'Basic realm="WorkAI", charset="UTF-8"');
          throw new AppError(401, '请登录研究工作台。');
        }
      }
      const url = new URL(request.url, expectedOrigin);
      if (request.method === 'GET' && staticFiles.has(url.pathname)) {
        const [name, contentType] = staticFiles.get(url.pathname);
        const body = await readFile(join(publicRoot, name));
        response.writeHead(200, { 'Content-Type': contentType }); response.end(body); return;
      }
      if (!['GET', 'POST'].includes(request.method)) throw new AppError(405, '不支持该请求方法。');
      if (request.method === 'POST' && (!requestOrigin || !sameOrigins.includes(requestOrigin) || !equal(request.headers['x-workai-csrf'] || '', csrf))) {
        throw new AppError(403, '页面会话已失效，请刷新后重试。');
      }
      if (request.method === 'GET' && url.pathname === '/api/bootstrap') {
        json(response, 200, { csrf_token: csrf, connection, max_artifact_bytes: MAX_ARTIFACT_BYTES, migration: { original_source_available: false, research_source_available: true, research_source_sha256: RESEARCH_CURRENT_SOURCE, research_constitutive: 'murnaghan_green_lagrange', material_nonlinearity: true } }); return;
      }
      if (models && request.method === 'GET' && url.pathname === '/api/model') {
        json(response, 200, { model: await models.get(), templates: { plate: defaultCase('plate'), volume: defaultCase('volume'), research: defaultCase('research'), nonlinear: defaultCase('nonlinear') }, materials: { aluminum: aluminumMaterial() } }); return;
      }
      if (request.method === 'GET' && url.pathname === '/api/training') {
        json(response, 200, await trainingReader()); return;
      }
      if(spectral && url.pathname.startsWith('/api/spectral/')) {
        if(url.pathname==='/api/spectral/jobs') {
          if(request.method==='GET')json(response,200,{jobs:await spectral.list()});
          else if(request.method==='POST')json(response,202,await spectral.submit(await readJson(request)));
          else throw new AppError(405,'该操作方法不支持。');
          return;
        }
        const match=url.pathname.match(/^\/api\/spectral\/jobs\/(spectral-[a-f0-9]{32})(?:\/(result|field))?$/);
        if(request.method==='GET'&&match) {
          const [,id,action]=match;
          if(action==='field'){const bytes=await spectral.field(id);response.writeHead(200,{'Content-Type':'application/gzip','Content-Length':bytes.length});response.end(bytes);}
          else json(response,200,action==='result'?await spectral.result(id):await spectral.status(id));
          return;
        }
        throw new AppError(404,'没有找到该高阶任务接口。');
      }
      if(solid && url.pathname.startsWith('/api/solid/')) {
        if(url.pathname==='/api/solid/studies'){
          if(request.method==='GET')json(response,200,{studies:await solid.studies.list()});
          else json(response,202,await solid.studies.submit(await readJson(request)));
          return;
        }
        const studyMatch=url.pathname.match(/^\/api\/solid\/studies\/(study-[a-f0-9]{32})(?:\/(result|probe))?$/);
        if(studyMatch){const [,id,action]=studyMatch;
          if(request.method==='POST'&&action==='probe')json(response,200,await solid.studies.probe(id,await readJson(request)));
          else if(request.method==='GET'&&action!=='probe')json(response,200,action==='result'?(await solid.studies.result(id)).study:await solid.studies.status(id));
          else throw new AppError(405,'该操作方法不支持。');
          return;
        }
        if(request.method==='POST'&&url.pathname==='/api/solid/assets'){json(response,201,await solid.upload(await readJson(request,12*1024*1024)));return;}
        if(request.method==='POST'&&url.pathname==='/api/solid/assets/inspect'){json(response,200,await solid.command('inspect',await readJson(request)));return;}
        if(request.method==='GET' && url.pathname==='/api/solid/templates'){json(response,200,{templates:await solid.command('templates')});return;}
        if(url.pathname==='/api/solid/jobs') {
          if(request.method==='GET')json(response,200,{jobs:await solid.list()});
          else json(response,202,await solid.submit(await readJson(request)));
          return;
        }
        const match=url.pathname.match(/^\/api\/solid\/jobs\/(solid-[a-f0-9]{32})(?:\/(result|field))?$/);
        if(request.method==='GET' && match) {
          const [,id,action]=match;
          if(action==='field'){const bytes=await solid.field(id);response.writeHead(200,{'Content-Type':'application/gzip','Content-Length':bytes.length});response.end(bytes);}
          else json(response,200,action==='result'?await solid.result(id):await solid.status(id));
          return;
        }
        throw new AppError(404,'没有找到该三维任务接口。');
      }
      if (models && request.method === 'POST' && url.pathname === '/api/model') {
        const body = await readJson(request); exactKeys(body, ['name', 'case', 'expected_revision']);
        json(response, 200, await models.save(body)); return;
      }
      if (models && request.method === 'POST' && url.pathname === '/api/studies') {
        const body = await readJson(request); exactKeys(body, ['job_id', 'model_revision', 'kind']); assertJobId(body.job_id);
        if (!['plate_smoke', 'plate_reference', 'research_reference'].includes(body.kind)) throw new AppError(400, '不支持该研究类型。');
        if (submitting) throw new AppError(409, '正在提交任务，请稍后重试。');
        submitting = true;
        try {
          if (await store.get(body.job_id)) throw new AppError(409, '此任务已有记录，请查询状态。');
          const unresolved = (await store.list()).find(record => record.origin === 'web' && !['completed', 'failed'].includes(record.last_status?.phase));
          if (unresolved) throw new AppError(409, '已有正在执行或尚未确认的任务，请先查询并处理。');
          const current = await models.get();
          const research = isResearchScene(current.case.schema);
          if (research !== (body.kind === 'research_reference')) throw new AppError(422, '任务类型与已保存模型不一致。');
          if (research && !researchReady(await remote.health('research'), current.case.schema)) throw new AppError(409, '研究模型版本与已核对源码不一致，已停止提交。');
          if (current.case.schema === 'workai.physics_ai.volume.case.v1') {
            const health = await remote.health('volume');
            if (health.model_ready !== true) throw new AppError(409, '大体积模型尚未就绪，可以保存参数，暂时不能开始计算。');
          }
          const { model, file } = await models.snapshot(body.job_id, body.model_revision);
          await store.put({ job_id: body.job_id, kind: body.kind, model_name: model.name, model_revision: model.revision, case_sha256: model.case_sha256, origin: 'web', attempted: true, created_at: new Date().toISOString(), submission_state: 'unknown' });
          try {
            const job = await remote.submitStudy(body.job_id, file, body.kind);
            const record = await store.update(body.job_id, { last_status: job, submission_state: 'accepted', checked_at: new Date().toISOString() });
            json(response, 202, { job_id: body.job_id, accepted: true, record });
          } catch { json(response, 202, { job_id: body.job_id, accepted: null, message: '研究任务提交结果未确认，请按当前编号查询，不要重复提交。' }); }
        } finally { submitting = false; }
        return;
      }
      if (request.method === 'GET' && url.pathname === '/api/health') {
        if (!healthPromise && (!healthCache || Date.now() - healthAt > 10_000)) {
          healthPromise = (async () => {
            const services = {};
            await Promise.all(['base', 'plate', 'volume', 'research'].map(async type => {
              try { services[type] = { ok: true, verified: true, data: await remote.health(type) }; }
              catch (error) { services[type] = { ok: false, verified: false, error: error instanceof AppError ? error.message : '状态检查失败。' }; }
            }));
            healthCache = { checked_at: new Date().toISOString(), services }; healthAt = Date.now();
          })().finally(() => { healthPromise = null; });
        }
        if (healthPromise) await healthPromise;
        json(response, 200, healthCache); return;
      }
      if (request.method === 'GET' && url.pathname === '/api/jobs') { json(response, 200, { jobs: await store.list() }); return; }
      if (request.method === 'POST' && url.pathname === '/api/jobs/import') {
        const body = await readJson(request); exactKeys(body, ['job_id']); assertJobId(body.job_id);
        const job = await remote.job(body.job_id);
        const previous = await store.get(body.job_id);
        const record = await store.update(body.job_id, {
          kind: job.kind, origin: previous?.origin || 'imported', attempted: true,
          created_at: previous?.created_at || job.state?.started_at_utc || new Date().toISOString(),
          last_status: job, checked_at: new Date().toISOString(),
        });
        json(response, 200, record); return;
      }
      if (request.method === 'POST' && url.pathname === '/api/jobs') {
        const body = await readJson(request); exactKeys(body, ['job_id', 'kind']); assertJobId(body.job_id);
        if (body.kind !== 'smoke') throw new AppError(400, '目前仅支持创建连接验证任务。');
        if (submitting) throw new AppError(409, '正在提交任务，请先等待当前请求返回。');
        submitting = true;
        try {
          if (await store.get(body.job_id)) throw new AppError(409, '此任务编号已有记录，请查询状态；不会重复提交。');
          const unresolved = (await store.list()).find(record => record.origin === 'web' &&
            !['completed', 'failed'].includes(record.last_status?.phase));
          if (unresolved) throw new AppError(409, '已有正在执行或尚未确认的任务，请先查询并处理该任务。');
          await store.put({ job_id: body.job_id, kind: 'smoke', origin: 'web', attempted: true, created_at: new Date().toISOString(), submission_state: 'unknown' });
          try {
            const job = await remote.submit(body.job_id);
            const record = await store.update(body.job_id, { last_status: job, submission_state: 'accepted', checked_at: new Date().toISOString() });
            json(response, 202, { job_id: body.job_id, accepted: true, record });
          } catch {
            // A timeout does not establish that the remote POST failed. Keep
            // the ID durable and never automatically send it a second time.
            json(response, 202, { job_id: body.job_id, accepted: null, message: '提交结果暂未确认。请查询此任务编号，不要重复创建。' });
          }
        } finally { submitting = false; }
        return;
      }
      const match = url.pathname.match(/^\/api\/jobs\/(comsol-[a-f0-9]{32})(?:\/(files|preview|series|waveforms)(?:\/([a-z.]+))?)?$/);
      if (request.method === 'GET' && match) {
        const [, id, action, name] = match;
        if (!action) {
          const job = await remote.job(id);
          const checked_at = new Date().toISOString();
          if (await store.get(id)) await store.update(id, { last_status: job, checked_at });
          json(response, 200, { verified: true, checked_at, job }); return;
        }
        if (action === 'series' && !name) {
          const job = await remote.job(id);
          if (job.kind !== 'smoke' || job.phase !== 'completed') throw new AppError(422, '完成的连接验证任务才支持此频率预览。');
          const coarse = await remote.artifact(id, 'coarse.csv');
          const fine = await remote.artifact(id, 'fine.csv');
          json(response, 200, { verified: true, job_id: id, kind: 'smoke', coarse: parseSmokeCsv(coarse.bytes), fine: parseSmokeCsv(fine.bytes) }); return;
        }
        if (action === 'waveforms' && !name) {
          const job = await remote.job(id);
          if (job.kind !== 'research_reference' || job.phase !== 'completed') throw new AppError(422, '完成的二维研究任务才支持 A 扫预览。');
          try {
            json(response, 200, await readResearchResults(remote, id));
          } catch (error) { throw new AppError(422, error.message); }
          return;
        }
        if (['files', 'preview'].includes(action) && name) {
          assertArtifact(name);
          if (action === 'preview' && name.endsWith('.mph')) throw new AppError(422, 'MPH 模型请下载后使用 COMSOL 打开。');
          const artifact = await remote.artifact(id, name);
          if (action === 'preview') json(response, 200, { name, sha256: artifact.sha256, verified: true, text: decodeText(artifact.bytes) });
          else {
            response.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': artifact.bytes.length, 'Content-Disposition': `attachment; filename="${name}"`, 'X-WorkAI-Artifact-SHA256': artifact.sha256 });
            response.end(artifact.bytes);
          }
          return;
        }
      }
      throw new AppError(404, '没有找到该页面或接口。');
    } catch (error) {
      if (!response.headersSent) json(response, error instanceof AppError ? error.status : 500, { error: error instanceof AppError ? error.message : '网站服务暂时出错，请稍后重试。' });
      else response.destroy();
    }
  });
  server.requestTimeout = 90_000;
  server.headersTimeout = 15_000;
  return server;
}
