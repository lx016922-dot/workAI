import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { readFile, lstat, writeFile, unlink } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { isResearchScene, compileResearchCase } from './research.mjs';

const execute = promisify(execFile);
export const JOB_ID = /^comsol-[a-f0-9]{32}$/;
export const MAX_ARTIFACT_BYTES = 1024 * 1024;
export const ARTIFACTS = new Set([
  'request.json', 'case.properties', 'completed.properties', 'coarse.csv', 'fine.csv',
  'coarse.mph', 'fine.mph', 'compile.log', 'solver.log', 'worker.log', 'status.json',
  'timecheck.csv', 'timecheck.mph', 'coarse.field.csv', 'fine.field.csv', 'timecheck.field.csv',
]);
export class AppError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
export function assertJobId(id) {
  if (typeof id !== 'string' || !JOB_ID.test(id)) throw new AppError(400, '任务编号格式不正确。');
}
export function assertArtifact(name) {
  if (!ARTIFACTS.has(name)) throw new AppError(400, '不支持该结果文件。');
}
export function validateJob(body, id) {
  if (body?.schema !== 'workai.comsol.http.job_status.v1' || body.job_id !== id ||
      typeof body.phase !== 'string' || !Array.isArray(body.artifacts)) {
    throw new AppError(502, '远程任务响应与请求不匹配。');
  }
  return body;
}

// Delegate signing and verification to the already tested recovery client.
// Arguments are passed without a shell. No private key enters this web process.
export class RemoteClient {
  constructor(root) { this.root = resolve(root); this.active = 0; this.waiting = []; this.artifactQueue = Promise.resolve(); this.artifactReads = new Map(); }
  async command(program, args) {
    if (this.waiting.length >= 24) throw new AppError(429, '请求较多，请稍后重试。');
    if (this.active >= 4) await new Promise(resolve => this.waiting.push(resolve));
    else this.active++;
    try {
      let result;
      try {
        result = await execute(program, args, {
          cwd: this.root, timeout: 65_000, maxBuffer: 2 * 1024 * 1024,
          encoding: 'utf8', windowsHide: true,
        });
      } catch (error) {
        // A verified non-2xx reply is emitted as JSON even when the CLI exits 2.
        if (error.code === 2 && error.stdout) result = error;
        else throw new AppError(502, '未取得可验证的远程响应，请检查 COMSOL 服务和网络连接。');
      }
      try { return JSON.parse(result.stdout.replace(/^\uFEFF/, '')); }
      catch { throw new AppError(502, '远程客户端未返回有效结果。'); }
    } finally {
      const next = this.waiting.shift();
      if (next) next(); else this.active--;
    }
  }
  async action(action, id) {
    const args = ['-Action', action];
    if (id) { assertJobId(id); args.push('-JobId', id); }
    const result = await this.command(join(this.root, 'bin/workai'), args);
    if (result.server_signature_verified !== true) throw new AppError(502, '远程响应未通过签名验证。');
    if (result.http_status < 200 || result.http_status >= 300) {
      const status = result.http_status === 404 ? 404 : result.http_status === 409 ? 409 : 502;
      throw new AppError(status, status === 404 ? '远程服务器没有这个任务。' : '远程服务器拒绝了请求，请查询任务状态后再处理。');
    }
    return result.body;
  }
  async health(type) {
    if (type === 'research') {
      const result = await this.researchAction('health');
      if (result.server_signature_verified !== true || result.http_status !== 200 || result.body?.schema !== 'workai.comsol.research.health.v1') throw new AppError(502, '研究服务状态未通过校验。');
      return result.body;
    }
    const action = type === 'base' ? 'health' : `${type}-health`;
    const body = await this.action(action);
    const schema = type === 'base' ? 'workai.comsol.http.health.v1' : `workai.comsol.${type}.health.v1`;
    if (body?.schema !== schema) throw new AppError(502, '服务状态响应格式不正确。');
    return body;
  }
  async job(id) { return validateJob(await this.action('job', id), id); }
  async submit(id) { return validateJob(await this.action('smoke', id), id); }
  async plateAction(action, caseFile, id, kind) {
    const args = ['-NoLogo', '-NoProfile', '-NonInteractive', '-File', fileURLToPath(new URL('./Invoke-Plate.ps1', import.meta.url)), '-Root', this.root, '-Action', action, '-CaseFile', caseFile];
    if (id) { assertJobId(id); args.push('-JobId', id, '-Kind', kind); }
    return this.command(join(this.root, 'bin/workai-pwsh'), args);
  }
  async researchAction(action, caseFile, id) {
    const args = ['-NoLogo', '-NoProfile', '-NonInteractive', '-File', fileURLToPath(new URL('./Invoke-Research.ps1', import.meta.url)), '-Root', this.root, '-Action', action];
    let parameterFile;
    if (caseFile) {
      let parameters;
      try { parameters = compileResearchCase(JSON.parse(await readFile(caseFile, 'utf8'))); }
      catch (error) {
        if (action === 'validate') return { valid: false, validation_error: error.message };
        throw new AppError(422, error.message);
      }
      parameterFile = caseFile + '.parameters.json';
      await writeFile(parameterFile, JSON.stringify(parameters), { mode: 0o600, flag: 'wx' });
      args.push('-SceneFile', caseFile, '-ParametersFile', parameterFile);
    }
    if (id) { assertJobId(id); args.push('-JobId', id); }
    try { return await this.command(join(this.root, 'bin/workai-pwsh'), args); }
    finally { if (parameterFile && action === 'validate') await unlink(parameterFile); }
  }
  async validateCase(caseFile) {
    const scene = JSON.parse(await readFile(caseFile, 'utf8'));
    return isResearchScene(scene.schema) ? this.researchAction('validate', caseFile) : this.plateAction('validate', caseFile);
  }
  async submitStudy(id, caseFile, kind) {
    const result = kind === 'research_reference' ? await this.researchAction('submit', caseFile, id) : await this.plateAction('submit', caseFile, id, kind);
    if (result.server_signature_verified !== true || result.http_status !== 202) throw new AppError(502, '研究任务未取得可验证的提交响应，请查询任务编号。');
    return validateJob(result.body, id);
  }
  async artifact(id, name) {
    assertJobId(id); assertArtifact(name);
    const key = `${id}/${name}`;
    if (this.artifactReads.has(key)) return this.artifactReads.get(key);
    if (this.artifactReads.size >= 24) throw new AppError(429, '结果读取请求较多，请稍后重试。');
    // The recovered HTTP listener serves one response at a time. Serialize
    // downloads and share concurrent reads of the same artifact to avoid proxy
    // timeouts and two readers racing to create the same verified cache file.
    const pending = this.artifactQueue.then(() => this.readArtifact(id, name));
    this.artifactQueue = pending.catch(() => {});
    this.artifactReads.set(key, pending);
    try { return await pending; } finally { this.artifactReads.delete(key); }
  }
  async readArtifact(id, name) {
    const result = await this.command(join(this.root, 'bin/workai-pwsh'), [
      '-NoLogo', '-NoProfile', '-NonInteractive', '-File', join(this.root, 'Get-WorkAIArtifacts.ps1'),
      '-JobId', id, '-Names', name,
    ]);
    const entries = result.artifacts;
    const expectedPath = join(this.root, 'requests', id, 'artifacts', name);
    if (result.job_id !== id || result.server_signature_verified !== true ||
        !['completed', 'failed'].includes(result.phase) || !Array.isArray(entries) || entries.length !== 1) {
      throw new AppError(502, '结果文件验证未完成。');
    }
    const entry = entries[0];
    if (entry.name !== name || resolve(entry.path || '') !== expectedPath ||
        !Number.isSafeInteger(entry.bytes) || entry.bytes < 0 || entry.bytes > MAX_ARTIFACT_BYTES ||
        !/^[a-f0-9]{64}$/.test(entry.sha256)) throw new AppError(502, '结果文件元数据不正确或超过 1 MiB 限制。');
    const info = await lstat(expectedPath);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== entry.bytes) throw new AppError(502, '本地结果文件不匹配。');
    const bytes = await readFile(expectedPath);
    if (createHash('sha256').update(bytes).digest('hex') !== entry.sha256) throw new AppError(502, '结果文件校验失败。');
    return { name, sha256: entry.sha256, bytes };
  }
}

export function decodeText(bytes) {
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return bytes.subarray(2).toString('utf16le');
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return Buffer.from(bytes.subarray(2)).swap16().toString('utf16le');
  return bytes.toString('utf8').replace(/^\uFEFF/, '');
}
export function parseSmokeCsv(bytes) {
  const lines = decodeText(bytes).trim().split(/\r?\n/);
  if (lines.shift() !== 'frequency_hz,R_real,R_imag,T_stress_real,T_stress_imag') {
    throw new AppError(422, '该文件不是连接验证任务的频率结果，可下载原文件查看。');
  }
  const points = lines.map(line => {
    const cells = line.split(',');
    const values = cells.map(Number);
    if (cells.length !== 5 || cells.some(v => !v.trim()) || values.some(v => !Number.isFinite(v)) || values[0] <= 0) {
      throw new AppError(422, '频率结果包含无效数据。');
    }
    return { frequency_mhz: values[0] / 1e6, reflection: Math.hypot(values[1], values[2]), transmission: Math.hypot(values[3], values[4]) };
  });
  if (!points.length) throw new AppError(422, '频率结果为空。');
  return points;
}
