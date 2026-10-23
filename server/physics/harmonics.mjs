import {s0Dispersion} from '../../public/plate-dispersion.js';
export {s0Dispersion} from '../../public/plate-dispersion.js';
export function windowedCoefficient(points, valuesNm, frequencyMhz, startUs, endUs) {
  if (points.length !== valuesNm.length || points.length < 4 || !(endUs > startUs) || !(frequencyMhz > 0) ||
      startUs < points[0].time_us - 1e-9 || endUs > points.at(-1).time_us + 1e-9) throw new Error('谐波提取窗不完整。');
  const weights = points.map(p => p.time_us < startUs || p.time_us > endUs ? 0 : .5 * (1 - Math.cos(2 * Math.PI * (p.time_us - startUs) / (endUs - startUs))));
  const sum = weights.reduce((a, b) => a + b, 0);
  if (sum <= 2 || valuesNm.some(v => !Number.isFinite(v))) throw new Error('谐波提取点数或信号无效。');
  const mean = weights.reduce((s, w, i) => s + w * valuesNm[i], 0) / sum;
  let re = 0, im = 0;
  for (let i = 0; i < points.length; i++) {
    const phase = 2 * Math.PI * frequencyMhz * points[i].time_us, value = weights[i] * (valuesNm[i] - mean);
    re += value * Math.cos(phase); im -= value * Math.sin(phase);
  }
  re *= 2 / sum; im *= 2 / sum;
  return { re_nm: re, im_nm: im, amplitude_nm: Math.hypot(re, im) };
}
const relativeComplex = (a, b) => b.amplitude_nm === 0 ? null : Math.hypot(a.re_nm - b.re_nm, a.im_nm - b.im_nm) / b.amplitude_nm;
export function analyzeHarmonics(scene, levels, receipt = {}) {
  const meshRefined = Number(receipt.fine_elements) > Number(receipt.coarse_elements);
  const frequency = scene.source.frequency_mhz, material = scene.material, thickness = scene.geometry.thickness_mm * 1e-3;
  const fundamental = s0Dispersion(frequency * 1e6, thickness, material.cl_m_s, material.ct_m_s);
  const second = s0Dispersion(2 * frequency * 1e6, thickness, material.cl_m_s, material.ct_m_s);
  if (!fundamental || !second) return { available: false, reason: '当前频厚积超出本版本 S0 分支和移动窗分析范围；仍可查看原始 A 扫。' };
  const mismatch = second.wavenumber_rad_m - 2 * fundamental.wavenumber_rad_m;
  const rows = scene.receivers.map((rx, index) => {
    const arrival = rx.x_mm * 1000 / fundamental.group_velocity_m_s + scene.source.delay_us;
    const start = arrival - .5 / frequency, end = arrival + (scene.source.cycles + .5) / frequency;
    if (start < 0 || end > scene.acquisition.duration_us) return { rx: index, x_mm: rx.x_mm, available: false, reason: '直达波分析窗未被记录完整覆盖。', window_us: [start, end] };
    const byLevel = {};
    for (const [level, traces] of Object.entries(levels)) {
      const shots = [0, 1, 2, 3, 4].map(shot => traces.find(t => t.shot === shot && t.rx === index));
      if (shots.some(t => !t)) throw new Error('非线性对照波形不完整。');
      const points = shots[0].points, values = shots.map(t => t.points.map(p => p.u_nm));
      const coefficient = (v, harmonic) => windowedCoefficient(points, v, harmonic * frequency, start, end);
      const diff = (a, b, factor = 1) => values[a].map((v, i) => v - factor * values[b][i]);
      const A1 = coefficient(values[2], 1), linear1 = coefficient(values[0], 1), linear2 = coefficient(values[0], 2);
      const geometric2 = coefficient(diff(1, 0), 2), material2 = coefficient(diff(2, 1), 2), total2 = coefficient(diff(2, 0), 2);
      const materialDouble2 = coefficient(diff(4, 3), 2);
      byLevel[level] = { fundamental: A1, linear_fundamental: linear1, linear_second: linear2, geometric_second: geometric2, material_second: material2, total_second: total2,
        material_double_second: materialDouble2, amplitude_scaling: material2.amplitude_nm === 0 ? null : materialDouble2.amplitude_nm / material2.amplitude_nm,
        relative_indicator_per_m: A1.amplitude_nm === 0 ? null : material2.amplitude_nm / A1.amplitude_nm ** 2 * 1e9 };
    }
    const mesh = relativeComplex(byLevel.coarse.material_second, byLevel.fine.material_second);
    const time = relativeComplex(byLevel.fine.material_second, byLevel.timecheck.material_second);
    const scaling = byLevel.timecheck.amplitude_scaling;
    return { rx: index, x_mm: rx.x_mm, available: true, window_us: [start, end], levels: byLevel,
      mesh_relative_complex_difference: mesh, time_relative_complex_difference: time,
      checks_within_5_percent: meshRefined && mesh !== null && time !== null && scaling !== null && mesh <= .05 && time <= .05 && Math.abs(scaling / 4 - 1) <= .05 };
  });
  const phaseChecks = [], ordered = rows.filter(r => r.available).toSorted((a, b) => a.x_mm - b.x_mm);
  for (let i = 1; i < ordered.length; i++) {
    const a = ordered[i - 1], b = ordered[i], dx = (b.x_mm - a.x_mm) * 1e-3;
    if (dx <= 0) continue;
    const z1 = a.levels.timecheck.linear_fundamental, z2 = b.levels.timecheck.linear_fundamental;
    if (z1.amplitude_nm === 0 || z2.amplitude_nm === 0) continue;
    const measured = Math.atan2(z2.im_nm, z2.re_nm) - Math.atan2(z1.im_nm, z1.re_nm), expected = -fundamental.wavenumber_rad_m * dx;
    const residual = Math.atan2(Math.sin(measured - expected), Math.cos(measured - expected));
    const inferredK = -(expected + residual) / dx;
    phaseChecks.push({ from_mm: a.x_mm, to_mm: b.x_mm, phase_residual_rad: residual, inferred_wavenumber_rad_m: inferredK,
      relative_wavenumber_difference: (inferredK - fundamental.wavenumber_rad_m) / fundamental.wavenumber_rad_m,
      phase_unwrap_reference: 'nearest_predicted_S0_branch; not_independent_modal_identification' });
  }
  return { available: true, component: 'u', signal_unit: 'nm', window: 'arrival_aligned_hann_one_period_padding_weighted_mean_removed',
    extraction: 'complex_single_frequency_projection; material increment = full finite strain minus geometric-only, before projection',
    controls: ['linear_A', 'geometric_A', 'material_A', 'geometric_2A', 'material_2A'],
    dispersion: { model: 'isotropic_free_plate_Rayleigh_Lamb_S0_candidate', fundamental, second, mismatch_rad_m: mismatch,
      coherence_length_mm: Math.abs(mismatch) < 1e-12 ? null : Math.PI / Math.abs(mismatch) * 1000 },
    mesh_refinement_demonstrated: meshRefined,
    mesh_check_note: meshRefined ? '实际单元数量增加。' : '实际细网格单元数量未增加，不能仅凭尺寸设置或小差异宣称空间加密通过。',
    all_checks_within_5_percent: rows.every(r => r.available && r.checks_within_5_percent), rows, linear_phase_checks: phaseChecks,
    limitations: ['移动窗投影给出有定义的相对幅值，未分离纯模态。', '有限脉冲、群速度差及传播距离限制累积；5% 是当前算例的数值检查阈值。', 'A2/A1² 有量纲，不是材料绝对非线性系数；文献参数未针对特定铝合金标定。', '当前平面应变路径不能独立辨识全部三阶常数。'] };
}
