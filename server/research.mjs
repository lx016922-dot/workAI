import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { aluminumMaterial, verifyMaterialReference } from './materials.mjs';
import { NONLINEAR_SCENE, NONLINEAR_METHOD, CONTROLS, compileNonlinearCase } from './nonlinear.mjs';

export const RESEARCH_SCENE = 'workai.recovered.guided_plate.scene.v1';
export const RESEARCH_SOURCE = 'ac090340bd65c47240b7954d2a51125b010f18d0e608bed012a6525febacccd7';
export const RESEARCH_CURRENT_SOURCE = JSON.parse(readFileSync(new URL('./comsol/model.json', import.meta.url))).source_sha256;
if (createHash('sha256').update(readFileSync(new URL('./comsol/UltraResearch.java', import.meta.url))).digest('hex') !== RESEARCH_CURRENT_SOURCE) throw new Error('研究模型源码与版本摘要不符。');
export const isResearchScene = schema => [RESEARCH_SCENE, NONLINEAR_SCENE].includes(schema);
export const researchReady = (health, schema = RESEARCH_SCENE) => health?.schema === 'workai.comsol.research.health.v1' && health.status === 'ready_for_job' &&
  (schema === NONLINEAR_SCENE ? health.source_sha256 === RESEARCH_CURRENT_SOURCE : [RESEARCH_SOURCE, RESEARCH_CURRENT_SOURCE].includes(health.source_sha256));
export function defaultResearchCase() {
  const aluminum = aluminumMaterial();
  return {
    schema: RESEARCH_SCENE, constitutive: 'linear_isotropic',
    geometry: { length_mm: 20, thickness_mm: 1 },
    material: aluminum.linear, material_reference: aluminum.reference, boundary: 'free',
    source: { x_mm: 3, radius_mm: .5, frequency_mhz: .5, amplitude_pa: 100000, cycles: 3, delay_us: 0, phase_deg: 0, direction: 'normal' },
    receivers: [{ id: 'rx0', x_mm: 8 }, { id: 'rx1', x_mm: 12 }, { id: 'rx2', x_mm: 16 }],
    acquisition: { duration_us: 12, sample_step_us: .05 },
    numerics: { coarse_h_mm: 1, fine_h_mm: .5, max_step_us: .025, relative_tolerance: 1e-6, field_nx: 21, field_ny: 5, field_frames: 7 },
  };
}
function keys(value, expected, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== expected.length || expected.some(k => !Object.hasOwn(value, k))) throw new Error(`${label}字段不符合已恢复的研究模型。`);
}
function number(value, min, max, label, integer = false) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) throw new Error(`${label}应为 ${min}–${max} 范围内的${integer ? '整数' : '数值'}。`);
  return value;
}
// Only expose the intact Cartesian plate subset whose constitutive law was read.
// The installed model is linear: third-order constants/contact are NOT accepted.
export function compileResearchCase(c) {
  if (c?.schema === NONLINEAR_SCENE) return compileNonlinearCase(c, compileResearchCase);
  keys(c, ['schema', 'constitutive', 'geometry', 'material', 'boundary', 'source', 'receivers', 'acquisition', 'numerics', ...(Object.hasOwn(c || {}, 'material_reference') ? ['material_reference'] : [])], '二维研究');
  if (c.schema !== RESEARCH_SCENE || c.constitutive !== 'linear_isotropic') throw new Error('已恢复的研究模型仅支持线性各向同性弹性体。');
  if (!['free', 'side_impedance'].includes(c.boundary)) throw new Error('边界条件不受支持。');
  keys(c.geometry, ['length_mm', 'thickness_mm'], '几何');
  keys(c.material, ['rho_kg_m3', 'cl_m_s', 'ct_m_s'], '材料');
  if (Object.hasOwn(c, 'material_reference')) verifyMaterialReference(c.material_reference, c.material);
  keys(c.source, ['x_mm', 'radius_mm', 'frequency_mhz', 'amplitude_pa', 'cycles', 'delay_us', 'phase_deg', 'direction'], '激励');
  keys(c.acquisition, ['duration_us', 'sample_step_us'], '采样');
  keys(c.numerics, ['coarse_h_mm', 'fine_h_mm', 'max_step_us', 'relative_tolerance', 'field_nx', 'field_ny', 'field_frames'], '离散参数');
  const g = c.geometry, m = c.material, s = c.source, a = c.acquisition, n = c.numerics;
  number(g.length_mm, .1, 2000, '长度'); number(g.thickness_mm, .01, 200, '厚度');
  number(m.rho_kg_m3, 500, 20000, '密度'); number(m.cl_m_s, 500, 15000, '纵波速度'); number(m.ct_m_s, 100, 10000, '横波速度');
  if (m.cl_m_s ** 2 <= 4 * m.ct_m_s ** 2 / 3) throw new Error('材料体积模量必须为正。');
  number(a.duration_us, .01, 10000, '采集时长'); number(n.max_step_us, .000001, 10, '最大时间步');
  number(a.sample_step_us, n.max_step_us, 10, '采样间隔'); number(n.relative_tolerance, 1e-8, 1e-6, '相对容差');
  number(n.coarse_h_mm, .0001, 100, '粗网格'); number(n.fine_h_mm, .0001, n.coarse_h_mm * .8, '细网格');
  number(n.field_nx, 5, 101, '场采样 X', true); number(n.field_ny, 3, 41, '场采样厚度方向', true); number(n.field_frames, 2, 101, '场采样帧数', true);
  const intervals = Math.round(a.duration_us / a.sample_step_us);
  if (Math.abs(a.duration_us / a.sample_step_us - intervals) > 1e-6 || intervals % (n.field_frames - 1) !== 0) throw new Error('采集时长需包含整数采样间隔，场帧需均匀落在采样点上。');
  if (a.duration_us / (n.max_step_us * .5) > 100000 || 2 * g.length_mm * g.thickness_mm / n.fine_h_mm ** 2 > 40000) throw new Error('网格或时间步超过原研究模型预算。');
  number(s.x_mm, 0, g.length_mm, '激励 X'); number(s.radius_mm, .0001, g.length_mm / 2, '激励半径');
  number(s.frequency_mhz, .001, 100, '频率'); number(s.amplitude_pa, 1e-6, 1e9, '应力幅值'); number(s.cycles, 1, 100, '周期数');
  number(s.delay_us, 0, a.duration_us, '延迟'); number(s.phase_deg, -36000, 36000, '相位');
  const direction = ['tangential', 'normal', 'sh'].indexOf(s.direction);
  if (direction < 0) throw new Error('激励方向不受支持。');
  if (s.delay_us + s.cycles / s.frequency_mhz > a.duration_us + 1e-9) throw new Error('采集时间未覆盖完整激励脉冲。');
  if (s.frequency_mhz * a.sample_step_us >= .5) throw new Error('采样率不能解析载频。');
  if (!Array.isArray(c.receivers) || c.receivers.length < 1 || c.receivers.length > 8) throw new Error('请设置 1–8 个顶面接收点。');
  const ids = new Set();
  for (const rx of c.receivers) {
    keys(rx, ['id', 'x_mm'], '接收点');
    if (typeof rx.id !== 'string' || !/^[a-zA-Z][a-zA-Z0-9_-]{0,15}$/.test(rx.id) || ids.has(rx.id)) throw new Error('接收点 ID 应唯一，由字母开头。');
    ids.add(rx.id); number(rx.x_mm, 0, g.length_mm, '接收点 X');
  }
  // Keep each CSV within the retained verified downloader's 1 MiB envelope.
  if (c.receivers.length * (intervals + 1) > 4000 || n.field_nx * n.field_ny * n.field_frames > 4000) throw new Error('当前下载器限制单个文件 1 MiB，请减少采样点或场帧（每类最多 4000 行）。');
  const parameters = {
    schema: 'workai.comsol.research.case.v1', scenario: 'guided_plate', boundary: c.boundary,
    length_mm: g.length_mm, height_mm: g.thickness_mm, axisymmetric: 0, outer_radius_mm: 1,
    rho: m.rho_kg_m3, cl: m.cl_m_s, ct: m.ct_m_s,
    conductivity: 0, heat_capacity: 0, expansion: 0, thermal: 0,
    duration_us: a.duration_us, sample_step_us: a.sample_step_us, relative_tolerance: n.relative_tolerance,
    element_order: 4, element_budget: 40000, field_nx: n.field_nx, field_ny: n.field_ny, field_frames: n.field_frames,
    notch_enabled: 0, notch_x_mm: 0, notch_width_mm: 0, notch_depth_mm: 0, notch_bottom: 0,
    shot_count: 1, receiver_count: c.receivers.length, coarse_h_mm: n.coarse_h_mm, fine_h_mm: n.fine_h_mm, max_step_us: n.max_step_us,
    bandwidth_estimate_mhz: s.frequency_mhz * (1 + 2 / s.cycles),
    tx0_kind: 0, tx0_x_mm: s.x_mm, tx0_radius_mm: s.radius_mm, tx0_delay_us: s.delay_us, tx0_phase_deg: s.phase_deg,
    tx0_cycles: s.cycles, tx0_amplitude_pa: s.amplitude_pa, tx0_direction: direction, tx0_frequency_count: 1, tx0_f0: s.frequency_mhz,
  };
  c.receivers.forEach((rx, index) => { parameters[`rx${index}_x_mm`] = rx.x_mm; });
  return parameters;
}

export function verifyResearchRequest(text, receiptText, id) {
  const request = JSON.parse(text), hash = text => createHash('sha256').update(text).digest('hex');
  if (request.job_id !== id || request.kind !== 'research_reference' || ![RESEARCH_SOURCE, RESEARCH_CURRENT_SOURCE].includes(request.source_sha256) ||
      typeof request.scene_json !== 'string' || typeof request.parameters_json !== 'string' ||
      hash(request.scene_json) !== request.scene_sha256 || hash(request.parameters_json) !== request.parameters_sha256) throw new Error('研究输入身份或摘要不一致。');
  const receipt = {};
  const unescape = value => value.replace(/\\u([a-fA-F0-9]{4})|\\(.)/g, (_, code, char) => code ? String.fromCharCode(parseInt(code, 16)) : ({ n: '\n', r: '\r', t: '\t', f: '\f' }[char] ?? char));
  for (const line of receiptText.split(/\r?\n/).filter(line => !line.startsWith('#') && line.includes('='))) {
    const at = line.indexOf('='), key = unescape(line.slice(0, at));
    if (Object.hasOwn(receipt, key) || ['__proto__', 'constructor', 'prototype'].includes(key)) throw new Error('研究回执包含重复或无效字段。');
    receipt[key] = unescape(line.slice(at + 1));
  }
  for (const key of ['job_id', 'source_sha256', 'scene_sha256', 'parameters_sha256']) if (receipt[key] !== request[key]) throw new Error('完成回执与研究输入不一致。');
  const p = JSON.parse(request.parameters_json);
  const scene = JSON.parse(request.scene_json);
  const nonlinear = scene?.schema === NONLINEAR_SCENE;
  if (receipt.schema !== 'workai.comsol.research.receipt.v1' || receipt.status !== 'completed' || receipt.method !== (nonlinear ? NONLINEAR_METHOD : 'comsol_research_2d_axisymmetric_weakform_v1')) throw new Error('研究完成回执类型不匹配。');
  if (nonlinear) {
    if (request.source_sha256 !== RESEARCH_CURRENT_SOURCE || receipt.controls !== CONTROLS.join(',') || receipt.constitutive !== 'murnaghan_green_lagrange' || receipt.material_record_sha256 !== aluminumMaterial().record_sha256 || receipt.quadrature_order !== '20') throw new Error('非线性本构或对照配置回执不匹配。');
    for (const k of ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']) if (Number(receipt[k]) !== scene.nonlinear[k]) throw new Error('实际本构常数与材料卡不符。');
    if (receipt.source_profile !== 'left_edge_uniform_nominal_traction' || receipt.energy_definition !== 'murnaghan_invariants_reference_volume' || receipt.spatial_assumption !== 'plane_strain_z_invariant' || receipt.mph_solution_state !== 'cleared' || receipt.mph_reopen_verified !== 'true') throw new Error('非线性求解配置回执不完整。');
    for (const level of ['coarse', 'fine', 'timecheck']) for (let shot = 0; shot < CONTROLS.length; shot++) {
      const strain = Number(receipt[`${level}_shot${shot}_max_green_strain`]), jacobian = Number(receipt[`${level}_shot${shot}_min_J`]);
      if (!Number.isFinite(strain) || strain < 0 || strain > .001 || !Number.isFinite(jacobian) || jacobian <= 0) throw new Error('应变或正体积检查回执不完整。');
    }
  }
  if (isResearchScene(scene?.schema)) {
    const expected = compileResearchCase(scene);
    if (Object.keys(expected).length !== Object.keys(p).length || Object.entries(expected).some(([key, value]) => p[key] !== value)) throw new Error('保存的场景与实际研究参数不一致。');
  }
  if (p.schema !== 'workai.comsol.research.case.v1' || p.scenario !== 'guided_plate' || p.axisymmetric !== 0 || p.thermal !== 0) throw new Error('当前 A 扫视图只解释二维弹性平板研究。');
  number(p.shot_count, 1, 64, '激励次数', true); number(p.receiver_count, 1, 32, '接收点数', true);
  number(p.duration_us, .01, 10000, '时长'); number(p.sample_step_us, .000001, 10, '采样间隔');
  return { parameters: p, receipt, scene, constitutive: nonlinear ? 'murnaghan_green_lagrange' : 'linear_isotropic', controls: nonlinear ? CONTROLS : null, source_sha256: request.source_sha256 };
}
export function parseResearchCsv(text, parameters) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.shift() !== 'shot,rx,time_s,u_m,v_m,w_m,temperature_rise_k') throw new Error('不是已恢复研究模型的 A 扫 CSV。');
  const intervals = Math.round(parameters.duration_us / parameters.sample_step_us), nt = intervals + 1;
  if (lines.length > 16384 || lines.length !== parameters.shot_count * parameters.receiver_count * nt) throw new Error('A 扫采样数量不完整或超过读取预算。');
  const traces = new Map();
  for (const line of lines) {
    const cells = line.split(','), values = cells.map(Number);
    if (cells.length !== 7 || cells.some(c => !c.trim()) || values.some(v => !Number.isFinite(v))) throw new Error('A 扫包含空值或非有限数据。');
    const [shot, rx, time, u, v, w, temperature] = values;
    if (!Number.isInteger(shot) || !Number.isInteger(rx) || shot < 0 || shot >= parameters.shot_count || rx < 0 || rx >= parameters.receiver_count) throw new Error('A 扫接收点或激励编号不匹配。');
    const key = `${shot}:${rx}`;
    if (!traces.has(key)) traces.set(key, { shot, rx, x_mm: parameters[`rx${rx}_x_mm`], points: [] });
    const trace = traces.get(key), expectedTime = trace.points.length * parameters.sample_step_us * 1e-6;
    if (trace.points.length >= nt || Math.abs(time - expectedTime) > 1e-11 || temperature !== 0) throw new Error('A 扫时间网格或物理量与模型不一致。');
    const point = { time_us: time * 1e6, u_nm: u * 1e9, v_nm: v * 1e9, w_nm: w * 1e9 };
    if (Object.values(point).some(v => !Number.isFinite(v))) throw new Error('A 扫单位换算溢出。');
    trace.points.push(point);
  }
  if (traces.size !== parameters.shot_count * parameters.receiver_count || [...traces.values()].some(t => t.points.length !== nt)) throw new Error('A 扫存在缺失接收点。');
  return [...traces.values()].sort((a, b) => a.shot - b.shot || a.rx - b.rx);
}
