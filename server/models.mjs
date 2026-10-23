import { mkdir, readFile, writeFile, rename, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { AppError, assertJobId } from './remote.mjs';
import { defaultResearchCase } from './research.mjs';
import { defaultNonlinearCase } from './nonlinear.mjs';

export function defaultCase(mode = 'plate') {
  if (mode === 'research') return defaultResearchCase();
  if (mode === 'nonlinear') return defaultNonlinearCase();
  const volume = mode === 'volume';
  return {
    schema: volume ? 'workai.physics_ai.volume.case.v1' : 'workai.comsol.plate.case.v1',
    geometry: { length_mm: volume ? 200 : 40, width_mm: volume ? 100 : 20, thickness_mm: 2 },
    material: { rho_kg_m3: 2700, cl_m_s: 6320, ct_m_s: 3130 },
    boundary: volume ? { x_min: 'absorbing', x_max: 'absorbing', y_min: 'free', y_max: 'free', z_min: 'free', z_max: 'free' } : 'free',
    notch: { enabled: false, x_mm: volume ? 100 : 20, y_mm: volume ? 50 : 10, length_mm: 5, width_mm: .5, depth_mm: 1 },
    sources: [volume ? { id: 'tx0', kind: 'prescribed_surface_displacement', position_mm: [10, 50, 2], direction: [0, 0, 1], width_mm: 1, span_mm: 1, amplitude_nm: 1, frequencies_mhz: [.2], cycles: 5, delay_us: 0, phase_deg: 0 } : { id: 'tx0', position_mm: [6, 10, 2], direction: [0, 0, 1], radius_mm: 1, frequencies_mhz: [.2], cycles: 5, delay_us: 0, amplitude_pa: 100000, phase_deg: 0 }],
    receivers: [{ id: 'rx0', position_mm: volume ? [100, 50, 1] : [30, 10, 2] }],
    acquisition: { duration_us: volume ? 150 : 80, sample_step_us: .5 },
    ...(!volume ? { numerics: { coarse_h_mm: 2, fine_h_mm: 1.5, max_step_us: .05, relative_tolerance: .00001, field_nx: 31, field_ny: 21, field_frames: 41 } } : {}),
  };
}
export class ModelStore {
  constructor(directory, remote) { this.directory = directory; this.remote = remote; this.saving = false; }
  async init() { await mkdir(this.directory, { recursive: true, mode: 0o700 }); }
  async get() {
    try { return JSON.parse(await readFile(join(this.directory, 'current.json'), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') return { name: '铝板超声模型', case: defaultCase(), revision: null, validated: false, saved_at: null }; throw error; }
  }
  async save({ name, case: modelCase, expected_revision }) {
    if (this.saving) throw new AppError(409, '正在保存模型，请稍后重试。');
    if (typeof name !== 'string' || name.trim().length < 1 || name.length > 80) throw new AppError(400, '请填写 1–80 字的模型名称。');
    if (!modelCase || typeof modelCase !== 'object' || Array.isArray(modelCase)) throw new AppError(400, '模型内容格式不正确。');
    const text = JSON.stringify(modelCase);
    if (text.length > 30000) throw new AppError(400, '模型参数超过当前协议限制。');
    this.saving = true;
    const caseFile = join(this.directory, `${randomUUID()}.case.json`);
    try {
      const current = await this.get();
      if (expected_revision !== current.revision) throw new AppError(409, '模型已在其他页面更新，请重新载入后再编辑。');
      await writeFile(caseFile, text, { mode: 0o600, flag: 'wx' });
      const result = await this.remote.validateCase(caseFile);
      if (result.valid !== true) throw new AppError(422, `参数未通过服务器原有规则：${result.validation_error || '格式不正确'}`);
      const hash = createHash('sha256').update(text).digest('hex');
      if (result.case_sha256 !== hash) throw new AppError(502, '模型参数摘要不匹配。');
      const model = { name: name.trim(), case: modelCase, revision: randomUUID(), validated: true, case_sha256: hash, saved_at: new Date().toISOString() };
      const temporary = join(this.directory, `${randomUUID()}.tmp`);
      await writeFile(temporary, JSON.stringify(model, null, 2) + '\n', { mode: 0o600, flag: 'wx' });
      await rename(temporary, join(this.directory, 'current.json'));
      return model;
    } finally { await unlink(caseFile).catch(() => {}); this.saving = false; }
  }
  async snapshot(id, revision) {
    assertJobId(id);
    const model = await this.get();
    if (!model.validated || model.revision !== revision) throw new AppError(409, '模型尚未保存或已发生变化，请重新校验并保存。');
    const file = join(this.directory, `${id}.case.json`);
    await writeFile(file, JSON.stringify(model.case), { mode: 0o600, flag: 'wx' });
    return { model, file };
  }
}
