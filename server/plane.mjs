import { spawn } from 'node:child_process';
import { mkdir, readdir, readFile, writeFile, lstat, open } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomBytes } from 'node:crypto';
import { AppError } from './remote.mjs';

const repository = fileURLToPath(new URL('../', import.meta.url));
const validId = id => /^plane-[a-f0-9]{32}$/.test(id);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function bounded(path, max) {
  const stat = await lstat(path);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > max) throw new AppError(422, '二维结果文件无效或超过大小限制。');
  return readFile(path);
}

/** Plane-strain (x–z) spectral jobs: version-2 scenes with dimension 2d_plane_strain, run by spectral2d.worker. */
export class PlaneService {
  constructor(root, solid) { this.root = join(root, 'spectral2d'); this.jobs = join(this.root, 'jobs'); this.cache = join(this.root, 'dispersion'); this.solid = solid; this.python = join(root, 'solid/venv/bin/python'); }
  async init() { await mkdir(this.jobs, { recursive: true }); await mkdir(this.cache, { recursive: true }); }
  path(id) { if (!validId(id)) throw new AppError(400, '二维任务编号格式不正确。'); return join(this.jobs, id); }
  async status(id) {
    try { const result = JSON.parse(await bounded(join(this.path(id), 'status.json'), 256 * 1024)); if (result.job_id !== id) throw new AppError(422, '二维任务身份不符。'); return result; }
    catch (error) { if (error.code === 'ENOENT') throw new AppError(404, '二维任务不存在。'); throw error; }
  }
  async list() {
    const rows = []; for (const id of (await readdir(this.jobs)).filter(validId)) rows.push(await this.status(id));
    return rows.sort((a, b) => String(b.started_at_utc).localeCompare(String(a.started_at_utc)));
  }
  async assertIdle() { if ((await this.list()).some(r => !['completed', 'failed'].includes(r.phase))) throw new AppError(409, '已有二维计算正在运行或状态待确认。'); }
  template(plate) {
    // Built from the validated 3-D plate template so material, acquisition and numerics stay the recorded defaults.
    const scene = structuredClone(plate);
    scene.schema = 'workai.hyperelastic.solid.v2'; scene.dimension = '2d_plane_strain'; scene.name = '2 mm 铝板 · 平面应变 · S0 基频 0.5 MHz';
    scene.geometry.length_mm = 200; scene.geometry.thickness_mm = 2; scene.geometry.width_mm = 1;
    scene.source = { ...scene.source, face: 'xmin', center_mm: [0, 0, 1], radius_mm: 0, direction: [1, 0, 0], amplitude_mpa: 1, frequency_mhz: .5, cycles: 5, delay_us: 0, phase_deg: 0, signal: { type: 'tone_burst', window: 'hann' } };
    scene.boundary = { fixed_face: 'none' }; scene.boundaries = [{ type: 'absorbing', face: 'xmax', length_mm: 40, ramp: 'cubic', attenuation_db: 60 }];
    scene.acquisition = { duration_us: 50, sample_step_us: .05 }; scene.numerics = { ...scene.numerics, h_mm: 1, cfl: .3 };
    scene.receivers = [{ name: 'P1', point_mm: [60, 0, 2], direction: [1, 0, 0] }, { name: 'P2', point_mm: [120, 0, 2], direction: [1, 0, 0] }];
    scene.receiver_lines = [{ name: 'top', from_mm: [40, 0, 2], to_mm: [150, 0, 2], count: 111, direction: [1, 0, 0] }];
    return scene;
  }
  async templates() { const t = await this.solid.command('templates'); return { plate: this.template(t.plate) }; }
  async submit(body) {
    if (!body || Object.keys(body).sort().join(',') !== 'degree,scene' || ![2, 3, 4, 5, 6].includes(body.degree)) throw new AppError(400, '请提供二维场景及 2–6 的阶次。');
    return this.solid.exclusive(async () => {
      await this.solid.assertIdle(); await this.assertIdle();
      const input = structuredClone(body.scene), h = input?.numerics?.h_mm;
      if (typeof h !== 'number' || !Number.isFinite(h) || h < .02 || h > 100) throw new AppError(422, '单元尺寸必须在 0.02–100 mm 范围内。');
      input.numerics.h_mm = Math.max(.02, h / body.degree);   // resolution screen sees the node spacing, as for the 3-D high-order solver
      const scene = await this.solid.command('validate', input); scene.numerics.h_mm = h;
      if (scene.schema !== 'workai.hyperelastic.solid.v2' || scene.dimension !== '2d_plane_strain') throw new AppError(422, '二维计算需要 dimension 为 2d_plane_strain 的 v2 场景。');
      const id = 'plane-' + randomBytes(16).toString('hex'), path = this.path(id); await mkdir(path);
      await writeFile(join(path, 'scene.json'), JSON.stringify(scene), { flag: 'wx' }); await writeFile(join(path, 'settings.json'), JSON.stringify({ degree: body.degree }), { flag: 'wx' });
      await writeFile(join(path, 'status.json'), JSON.stringify({ job_id: id, phase: 'starting', started_at_utc: new Date().toISOString(), name: scene.name }), { flag: 'wx' });
      const log = await open(join(path, 'worker.log'), 'wx');
      const child = spawn(this.python, ['-m', 'spectral2d.worker', '--job-dir', path], { cwd: repository, env: { ...process.env, OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1' }, detached: true, stdio: ['ignore', log.fd, log.fd] });
      const fail = async message => { try { const prior = await this.status(id); if (!['completed', 'failed'].includes(prior.phase)) await writeFile(join(path, 'status.json'), JSON.stringify({ ...prior, phase: 'failed', error: message })); } catch {} };
      child.on('error', () => void fail('二维求解进程未能启动。')); child.on('exit', code => void fail(`二维求解进程退出（${code}），未发布完整结果。`));
      child.unref(); await log.close(); return { job_id: id, scene, degree: body.degree };
    });
  }
  async result(id) {
    const status = await this.status(id); if (status.phase !== 'completed') throw new AppError(409, '二维计算尚未完成。');
    const bytes = await bounded(join(this.path(id), 'result.json'), 64 * 1024 * 1024);
    if (sha(bytes) !== status.result_sha256) throw new AppError(422, '二维结果摘要不匹配。');
    const result = JSON.parse(bytes);
    if (result.schema !== 'workai.plane_strain.result.v1' || result.engine !== 'workai_spectral_plane_strain_2d' || result.job_id !== id || result.accepted_for_solver_routing !== false) throw new AppError(422, '二维结果身份不匹配。');
    if (sha(await bounded(join(this.path(id), 'scene.json'), 1024 * 1024)) !== result.scene_sha256 || sha(await bounded(join(this.path(id), 'settings.json'), 4096)) !== result.settings_sha256) throw new AppError(422, '二维场景或阶次设置摘要不匹配。');
    return result;
  }
  async binary(id, kind) {
    const result = await this.result(id); const descriptor = result[kind];
    if (!descriptor) throw new AppError(404, '该结果没有此数据文件。');
    const bytes = await bounded(join(this.path(id), descriptor.name), 128 * 1024 * 1024);
    if (bytes.length !== descriptor.compressed_bytes || sha(bytes) !== descriptor.sha256) throw new AppError(422, '二维数据文件摘要不匹配。');
    return bytes;
  }
  async dispersion(thicknessMm) {
    const thickness = Number(thicknessMm);
    if (!Number.isFinite(thickness) || thickness < .1 || thickness > 100) throw new AppError(400, '板厚需在 0.1–100 mm 之间。');
    const key = thickness.toFixed(4).replace(/\.?0+$/, ''), file = join(this.cache, `lamb-${key}mm.json`);
    try { return JSON.parse(await bounded(file, 16 * 1024 * 1024)); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    await new Promise((resolve, reject) => {
      const child = spawn(this.python, ['-m', 'dispersion.lamb', '--thickness-mm', String(thickness), '--fd-max', '8', '--points', '320', '--output', file], { cwd: repository, stdio: ['ignore', 'pipe', 'pipe'] });
      let error = ''; const timer = setTimeout(() => { child.kill('SIGTERM'); reject(new AppError(504, '频散计算超时。')); }, 120000);
      child.stderr.on('data', c => { error = (error + c).slice(-2000); }); child.stdout.on('data', () => {});
      child.on('error', () => { clearTimeout(timer); reject(new AppError(503, '二维求解环境尚未安装。')); });
      child.on('close', code => { clearTimeout(timer); if (code === 0) resolve(); else reject(new AppError(422, '频散计算未完成：' + error.split('\n').filter(Boolean).at(-1))); });
    });
    return JSON.parse(await bounded(file, 16 * 1024 * 1024));
  }
}
