const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
const physical = scene => { const copy = structuredClone(scene); delete copy.numerics; return JSON.stringify(canonical(copy)); };

// Both inputs must already have passed signed-artifact and receipt validation.
// Compare h=2 mm against h=1 mm at the same internal dt=0.025 us; the finer
// reference also has tighter algebraic tolerances, explicitly recorded below.
export function attachSpatialReference(target, reference, record) {
  if (target.verified !== true || reference.verified !== true || target.job_id !== record.target_job_id || reference.job_id !== record.reference_job_id ||
      target.source_sha256 !== reference.source_sha256 || target.receipt.scene_sha256 !== record.target_scene_sha256 || reference.receipt.scene_sha256 !== record.reference_scene_sha256 ||
      physical(target.scene) !== physical(reference.scene)) throw new Error('空间参考任务的物理场景或身份不匹配。');
  const a = record.target_level, b = record.reference_level;
  if (!['coarse', 'fine', 'timecheck'].includes(a) || !['coarse', 'fine', 'timecheck'].includes(b)) throw new Error('空间参考层级无效。');
  if (Number(target.receipt[`${a}_internal_step_s`]) !== Number(reference.receipt[`${b}_internal_step_s`]) ||
      !(Number(reference.receipt[`${b}_elements`]) > Number(target.receipt[`${a}_elements`])) ||
      !(Number(reference.receipt[`${b}_mesh_h_m`]) < Number(target.receipt[`${a}_mesh_h_m`]))) throw new Error('空间参考必须在同一时间步下增加实际单元数并减小网格尺寸。');
  const tolerance = (data, level) => data.scene.numerics.relative_tolerance * (level === 'timecheck' ? .25 : 1);
  if (tolerance(reference, b) > tolerance(target, a)) throw new Error('空间参考的求解容差不能更松。');
  const h = structuredClone(target.harmonics), ref = reference.harmonics;
  if (!h?.available || !ref?.available || h.rows.length !== ref.rows.length) throw new Error('空间参考分析不完整。');
  for (const row of h.rows) {
    const other = ref.rows.find(r => r.rx === row.rx && r.x_mm === row.x_mm);
    if (!row.available || !other?.available || JSON.stringify(row.window_us) !== JSON.stringify(other.window_us)) throw new Error('空间参考的分析窗或接收点不一致。');
    const coarse = row.levels[a].material_second, fine = other.levels[b].material_second;
    row.within_job_mesh_relative_difference = row.mesh_relative_complex_difference;
    row.mesh_relative_complex_difference = fine.amplitude_nm === 0 ? null : Math.hypot(coarse.re_nm - fine.re_nm, coarse.im_nm - fine.im_nm) / fine.amplitude_nm;
    row.spatial_reference_coefficient = fine;
    const mesh = row.mesh_relative_complex_difference, time = row.time_relative_complex_difference, ratio = row.levels.timecheck.amplitude_scaling;
    row.checks_within_5_percent = mesh !== null && time !== null && ratio !== null && mesh <= .05 && time <= .05 && Math.abs(ratio / 4 - 1) <= .05;
  }
  h.mesh_refinement_demonstrated = true;
  h.mesh_check_note = '空间列采用两项已校验任务在相同时间步下的 234/640 单元对照，参考任务的代数容差也更紧。当前任务两档网格均为 234 单元，单独不构成空间加密。';
  h.spatial_reference = { ...record, source_sha256: reference.source_sha256, internal_step_s: Number(target.receipt[`${a}_internal_step_s`]),
    target_elements: Number(target.receipt[`${a}_elements`]), reference_elements: Number(reference.receipt[`${b}_elements`]),
    target_relative_tolerance: tolerance(target, a), reference_relative_tolerance: tolerance(reference, b), reference_artifacts: reference.artifacts };
  h.all_checks_within_5_percent = h.rows.every(r => r.checks_within_5_percent);
  return h;
}
