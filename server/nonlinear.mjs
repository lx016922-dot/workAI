import { aluminumMaterial } from './materials.mjs';

export const NONLINEAR_SCENE = 'workai.murnaghan.guided_plate.scene.v1';
export const NONLINEAR_METHOD = 'comsol_murnaghan_total_lagrangian_2d_v1';
export const CONTROLS = ['linear_A', 'geometric_A', 'material_A', 'geometric_2A', 'material_2A'];
export function defaultNonlinearCase() {
  const a = aluminumMaterial();
  return {
    schema: NONLINEAR_SCENE, constitutive: 'murnaghan_green_lagrange',
    geometry: { length_mm: 160, thickness_mm: 1 }, material: a.linear,
    material_reference: { ...a.reference, usage: 'murnaghan_total_lagrangian' }, boundary: 'free',
    nonlinear: { definition: 'murnaghan_green_lagrange_reference_volume', lambda_pa: a.record.lambda_pa, mu_pa: a.record.mu_pa,
      l_pa: a.record.third_order.l_pa, m_pa: a.record.third_order.m_pa, n_pa: a.record.third_order.n_pa, controls: [...CONTROLS] },
    source: { profile: 'left_edge_uniform', frequency_mhz: .4, amplitude_pa: 5e6, cycles: 5, delay_us: 0, phase_deg: 0 },
    receivers: [{ id: 'rx0', x_mm: 20 }, { id: 'rx1', x_mm: 40 }, { id: 'rx2', x_mm: 60 }],
    acquisition: { duration_us: 26, sample_step_us: .1 },
    numerics: { coarse_h_mm: 2, fine_h_mm: 1, max_step_us: .025, relative_tolerance: 1e-8, field_nx: 41, field_ny: 3, field_frames: 3 },
  };
}
const exact = (a, keys, name) => {
  if (!a || typeof a !== 'object' || Array.isArray(a) || Object.keys(a).length !== keys.length || keys.some(k => !Object.hasOwn(a, k))) throw new Error(`${name}字段不匹配。`);
};
// The five shots are independent constitutive/amplitude controls, all driven by
// the same single carrier. The typed scene contains the third-order constants;
// no thermal or unrelated numeric transport field is repurposed.
export function compileNonlinearCase(c, compileLinear) {
  const d = defaultNonlinearCase();
  exact(c, Object.keys(d), '材料非线性场景');
  if (c.schema !== NONLINEAR_SCENE || c.constitutive !== d.constitutive || c.boundary !== 'free') throw new Error('当前非线性模板仅支持自由平板。');
  for (const key of ['nonlinear', 'material', 'material_reference']) {
    exact(c[key], Object.keys(d[key]), key);
    for (const [k, v] of Object.entries(d[key])) if (JSON.stringify(c[key][k]) !== JSON.stringify(v)) throw new Error('此版本采用已核对的多晶铝材料卡；修改本构需创建新的可追溯版本。');
  }
  exact(c.source, Object.keys(d.source), '端面激励');
  if (c.source.profile !== 'left_edge_uniform') throw new Error('非线性对照需采用同一均匀端面激励。');
  if (!(c.source.amplitude_pa >= 1e5 && c.source.amplitude_pa <= 5e6)) throw new Error('本验证模板基准应力限于 0.1–5 MPa；双幅值最高 10 MPa。');
  const linear = { ...c, schema: 'workai.recovered.guided_plate.scene.v1', constitutive: 'linear_isotropic',
    material_reference: { ...c.material_reference, usage: 'linear_constants_only' },
    source: { ...c.source, x_mm: 0, radius_mm: c.geometry.thickness_mm / 2, direction: 'tangential' } };
  delete linear.nonlinear; delete linear.source.profile;
  const p = compileLinear(linear);
  const intervals = Math.round(c.acquisition.duration_us / c.acquisition.sample_step_us);
  if (5 * c.receivers.length * (intervals + 1) > 4000 || 5 * c.numerics.field_nx * c.numerics.field_ny * c.numerics.field_frames > 4000) throw new Error('五组对照合计每类 CSV 不能超过 4000 行。');
  if (c.geometry.thickness_mm > 5 || c.geometry.length_mm > 400 || c.source.frequency_mhz > 2 || c.source.cycles < 4) throw new Error('超出当前薄板非线性验证范围。');
  if (c.source.frequency_mhz * c.acquisition.sample_step_us > .1 || c.source.frequency_mhz * c.numerics.max_step_us > .025) throw new Error('二次谐波需更密的输出采样或内部时间步。');
  const maxX = Math.max(...c.receivers.map(r => r.x_mm));
  // Conservative bound using bulk longitudinal speed: the fastest right-end
  // reflection must not reach any receiver during the retained record.
  if (c.acquisition.duration_us >= (2 * c.geometry.length_mm - maxX) * 1000 / c.material.cl_m_s) throw new Error('记录可能含右端回波，请增大板长或缩短时间窗。');
  p.shot_count = CONTROLS.length;
  for (let shot = 1; shot < CONTROLS.length; shot++) for (const [key, value] of Object.entries(p).filter(([key]) => key.startsWith('tx0_'))) p[key.replace('tx0_', `tx${shot}_`)] = key === 'tx0_amplitude_pa' && shot >= 3 ? 2 * value : value;
  return p;
}
