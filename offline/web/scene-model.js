// Scene editing model for the workbench: pure functions on version-2 scenes (no DOM).
export const V2 = 'workai.hyperelastic.solid.v2';
export const FACES = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'];
export const FACE_LABELS = { xmin: 'X 最小端', xmax: 'X 最大端', ymin: 'Y 最小面', ymax: 'Y 最大面', zmin: '底面 Z 最小', zmax: '顶面 Z 最大' };
export const KIND_LABELS = { plate: '平板', pipe: '圆管', hole: '带通孔板', notch: '带缺口板' };
export const SIGNAL_LABELS = { tone_burst: '猝发音（加窗正弦）', gaussian_pulse: '高斯脉冲', ricker: 'Ricker 子波', chirp: '线性调频', multi_tone: '多频合成', samples: '任意采样信号' };
export const WINDOW_LABELS = { hann: 'Hann', tukey: 'Tukey', gaussian: 'Gaussian', rectangular: '矩形' };
export const PROFILE_LABELS = { uniform: '均匀', gaussian: '高斯分布', comb: '梳状（叉指）' };
export const BOUNDARY_LABELS = { symmetry: '对称面', antisymmetry: '反对称面', absorbing: '吸收层' };
export const RAMP_LABELS = { linear: '线性', quadratic: '二次', cubic: '三次' };

const clone = value => structuredClone(value);
const axisOf = face => 'xyz'.indexOf(face[0]);

/** Bounding box size [Lx, Ly, Lz] in mm and its minimum corner. Pipes are centred on the X axis. */
export function extent(geometry) {
  if (geometry.kind === 'pipe') { const r = geometry.outer_radius_mm; return { size: [geometry.length_mm, 2 * r, 2 * r], min: [0, -r, -r] }; }
  return { size: [geometry.length_mm, geometry.width_mm, geometry.thickness_mm], min: [0, 0, 0] };
}

/** Centre point of a bounding face (mm). */
export function faceCenter(geometry, face) {
  const { size, min } = extent(geometry), axis = axisOf(face), c = size.map((s, i) => min[i] + s / 2);
  c[axis] = face.endsWith('min') ? min[axis] : min[axis] + size[axis];
  return c.map(v => Math.round(v * 1e6) / 1e6);
}

/** Inward unit normal of a bounding face. */
export function inwardNormal(face) { const n = [0, 0, 0]; n[axisOf(face)] = face.endsWith('min') ? 1 : -1; return n; }

export function defaultSignal(type, source) {
  const f = source.frequency_mhz || .2;
  switch (type) {
    case 'tone_burst': return { type, window: 'hann' };
    case 'gaussian_pulse': return { type, bandwidth: .6 };
    case 'ricker': return { type };
    case 'chirp': return { type, end_frequency_mhz: +(f * 1.4).toFixed(4), duration_us: +(Math.max(5, 8 / f)).toFixed(3), window: 'hann' };
    case 'multi_tone': return { type, components: [{ frequency_mhz: f, weight: 1, phase_deg: 0 }, { frequency_mhz: +(f * 1.5).toFixed(4), weight: .5, phase_deg: 0 }], duration_us: +(Math.max(5, 10 / f)).toFixed(3), window: 'hann' };
    case 'samples': { const step = .01, n = Math.round(5 / f / step); return { type, sample_step_us: step, normalize: true, values: Array.from({ length: n }, (_, i) => +(Math.sin(2 * Math.PI * f * i * step) * Math.sin(Math.PI * i / (n - 1)) ** 2).toFixed(6)) }; }
    default: throw Error('未知信号类型 ' + type);
  }
}

export function defaultProfile(type, source) {
  if (type === 'uniform') return { type };
  if (type === 'gaussian') return { type, sigma_mm: 1 };
  const along = source.face[0] === 'x' ? 'y' : 'x';
  return { type, pitch_mm: 4, teeth: 3, tooth_width_mm: 1.5, along, alternating_sign: false };
}

/** A new source on `face`, centred on it, pushing inward, with the scene's carrier settings. */
export function defaultSource(scene, face = 'zmax', index = 1) {
  const base = scene.source || {};
  return { name: `激励 ${index + 1}`, face, center_mm: faceCenter(scene.geometry, face), radius_mm: 0, direction: inwardNormal(face),
    amplitude_mpa: 1, frequency_mhz: base.frequency_mhz || .2, cycles: 5, delay_us: 0, phase_deg: 0, signal: { type: 'tone_burst', window: 'hann' }, profile: { type: 'gaussian', sigma_mm: 1 } };
}

export function defaultBoundary(type, face = 'xmax', scene = null) {
  if (type !== 'absorbing') return { type, face };
  const length = scene ? Math.max(1, Math.min(10, +(extent(scene.geometry).size[axisOf(face)] / 5).toFixed(2))) : 8;
  return { type, face, length_mm: length, ramp: 'cubic', attenuation_db: 60 };
}

export function defaultReceiver(scene, index) {
  const { size, min } = extent(scene.geometry);
  return { name: `P${index + 1}`, point_mm: [min[0] + size[0] * .75, min[1] + size[1] / 2, min[2] + size[2]].map(v => +v.toFixed(4)), direction: [0, 0, 1] };
}

/** Version-1 offline template or scene → equivalent version-2 scene (signal moves into the source). */
export function upgrade(input) {
  const scene = clone(input);
  if (scene.schema === V2) { delete scene.signal; return scene; }
  const legacy = scene.signal || { kind: 'hann' }; delete scene.signal;
  scene.schema = V2;
  const kind = legacy.kind || 'hann';
  if (kind === 'chirp') scene.source.signal = { type: 'chirp', end_frequency_mhz: legacy.end_frequency_mhz ?? .28, duration_us: +(scene.source.cycles / ((scene.source.frequency_mhz + (legacy.end_frequency_mhz ?? .28)) / 2)).toFixed(6), window: 'hann' };
  else scene.source.signal = { type: 'tone_burst', window: kind === 'rectangular' ? 'rectangular' : kind === 'gaussian' ? 'gaussian' : 'hann', ...(kind === 'gaussian' ? { gaussian_width: 1 / 6 } : {}) };
  scene.sources ??= []; scene.boundaries ??= [];
  return scene;
}

export const sourcesOf = scene => [scene.source, ...(scene.sources || [])];
export function sourceAt(scene, index) { return index === 0 ? scene.source : scene.sources[index - 1]; }

export function addSource(scene, face = 'zmax') { scene.sources ??= []; const s = defaultSource(scene, face, scene.sources.length + 1); scene.sources.push(s); return scene.sources.length; }
export function removeSource(scene, index) { if (index === 0) throw Error('主激励不能删除；可以修改它的所有参数。'); scene.sources.splice(index - 1, 1); if (!scene.sources.length) delete scene.sources; }
export function addBoundary(scene, type, face) {
  scene.boundaries ??= [];
  if (scene.boundaries.length >= 8) throw Error('最多 8 个边界条件。');
  const used = new Set([...scene.boundaries.map(b => b.face), scene.boundary.fixed_face]);
  const pick = face || FACES.find(f => !used.has(f) && !sourcesOf(scene).some(s => s.face === f)) || 'xmax';
  scene.boundaries.push(defaultBoundary(type, pick, scene)); return scene.boundaries.length - 1;
}
export function removeBoundary(scene, index) { scene.boundaries.splice(index, 1); }
export function addReceiver(scene) { if (scene.receivers.length >= 32) throw Error('最多 32 个接收点。'); scene.receivers.push(defaultReceiver(scene, scene.receivers.length)); return scene.receivers.length - 1; }
export function removeReceiver(scene, index) { if (scene.receivers.length <= 1) throw Error('至少需要一个接收点。'); scene.receivers.splice(index, 1); }

/** Switch a source's signal family, keeping the carrier settings. */
export function setSignalType(source, type) { source.signal = defaultSignal(type, source); if (type !== 'tone_burst') source.cycles = Math.max(1, Math.min(20, source.cycles || 1)); }
export function setProfileType(source, type) { source.profile = defaultProfile(type, source); }

/** Moving a source to another face re-centres it and points it inward. */
export function setSourceFace(scene, source, face) {
  source.face = face; source.center_mm = faceCenter(scene.geometry, face); source.direction = inwardNormal(face);
  if (source.profile?.type === 'comb' && source.profile.along === face[0]) source.profile.along = face[0] === 'x' ? 'y' : 'x';
}

/** Parse a signal from CSV/TSV/whitespace text: one value per line, or `time,value` pairs (time in μs). */
export function parseSamples(text) {
  const rows = String(text).split(/\r?\n/).map(l => l.trim()).filter(l => l && !/^[#%]/.test(l));
  const parsed = rows.map(l => l.split(/[\s,;]+/).map(Number)).filter(r => r.every(Number.isFinite));
  if (parsed.length < 8) throw Error('采样信号至少需要 8 个数值行。');
  if (parsed.length > 20000) throw Error('采样信号最多 20000 个数值。');
  if (parsed[0].length >= 2) {
    const t = parsed.map(r => r[0]), step = (t.at(-1) - t[0]) / (t.length - 1);
    if (!(step > 0) || t.some((v, i) => Math.abs(v - (t[0] + i * step)) > 1e-6 * Math.max(1, Math.abs(v)) + step * 1e-3)) throw Error('时间列需要等间隔递增（单位 μs）。');
    return { sample_step_us: +step.toPrecision(12), values: parsed.map(r => r[1]) };
  }
  return { sample_step_us: null, values: parsed.map(r => r[0]) };
}

export function sourceSummary(source) {
  const sig = source.signal || { type: 'tone_burst', window: 'hann' }, prof = source.profile || { type: 'uniform' };
  const kind = sig.type === 'tone_burst' ? `${WINDOW_LABELS[sig.window] || ''} ${source.cycles} 周期` : SIGNAL_LABELS[sig.type];
  return `${FACE_LABELS[source.face] || source.face} · ${PROFILE_LABELS[prof.type]}${source.radius_mm > 0 ? ` r=${source.radius_mm}` : ''} · ${source.frequency_mhz} MHz ${kind}`;
}
export function boundarySummary(b) { return `${BOUNDARY_LABELS[b.type]} · ${FACE_LABELS[b.face]}${b.type === 'absorbing' ? ` · ${b.length_mm} mm ${b.attenuation_db} dB` : ''}`; }
export function geometrySummary(g) {
  if (g.kind === 'pipe') return `${KIND_LABELS.pipe} L ${g.length_mm} · R ${g.outer_radius_mm}/${g.inner_radius_mm} mm`;
  return `${KIND_LABELS[g.kind]} ${g.length_mm} × ${g.width_mm} × ${g.thickness_mm} mm`;
}

export const materialKind = m => (m && m.model === 'energy' ? 'energy' : m && m.model === 'murnaghan' ? 'murnaghan' : 'reference');
export function materialSummary(m) {
  const kind = materialKind(m);
  if (kind === 'reference') return '多晶铝 · Murnaghan 三阶（训练材料）';
  if (kind === 'murnaghan') return `${m.name || 'Murnaghan'} · 各向同性 · ρ ${m.rho_kg_m3}`;
  return `${m.name || '自定义应变能'} · W(E) · 纤维 ${m.fiber_angle_deg ?? 0}°`;
}

/** Model tree rows for the left panel: [{id, label, detail, group, removable}]. */
export function tree(scene) {
  const rows = [{ id: 'geometry', group: 'model', label: '几何', detail: geometrySummary(scene.geometry) },
    { id: 'material', group: 'model', label: '材料与本构', detail: materialSummary(scene.material) }];
  rows.push({ id: 'sources', group: 'header', label: '激励源', detail: String(sourcesOf(scene).length), add: 'source' });
  sourcesOf(scene).forEach((s, i) => rows.push({ id: `source:${i}`, group: 'item', label: s.name || (i ? `激励 ${i + 1}` : '主激励'), detail: sourceSummary(s), removable: i > 0 }));
  rows.push({ id: 'boundaries', group: 'header', label: '边界条件', detail: String((scene.boundaries || []).length + (scene.boundary.fixed_face !== 'none' ? 1 : 0)), add: 'boundary' });
  if (scene.boundary.fixed_face !== 'none') rows.push({ id: 'fixed', group: 'item', label: '固定面', detail: FACE_LABELS[scene.boundary.fixed_face] });
  (scene.boundaries || []).forEach((b, i) => rows.push({ id: `boundary:${i}`, group: 'item', label: BOUNDARY_LABELS[b.type], detail: boundarySummary(b), removable: true }));
  rows.push({ id: 'receivers', group: 'header', label: '接收点', detail: String(scene.receivers.length), add: 'receiver' });
  scene.receivers.forEach((r, i) => rows.push({ id: `receiver:${i}`, group: 'item', label: r.name, detail: `(${r.point_mm.join(', ')}) mm`, removable: scene.receivers.length > 1 }));
  rows.push({ id: 'study', group: 'model', label: '记录与网格', detail: `${scene.acquisition.duration_us} μs · Δt ${scene.acquisition.sample_step_us} μs · h ${scene.numerics.h_mm} mm` });
  return rows;
}

/** Read/write a nested value with a dotted path (array indices allowed). */
export function getPath(object, path) { return path.split('.').reduce((o, k) => o?.[k], object); }
export function setPath(object, path, value) { const keys = path.split('.'), last = keys.pop(); const target = keys.reduce((o, k) => o[k], object); target[last] = value; }

/** Templates for the workbench: v1 offline templates upgraded, plus v2 examples. */
export function workbenchTemplates(v1) {
  const out = {};
  for (const [kind, scene] of Object.entries(v1 || {})) out[kind] = { label: `${KIND_LABELS[kind]} · 端面激励（原验收范围）`, scene: upgrade(scene) };
  const plate = v1?.plate ? upgrade(v1.plate) : null;
  if (plate) {
    const a = clone(plate);
    Object.assign(a, { name: '铝板 · 顶面高斯激励 · 两端吸收 · 半模型' });
    Object.assign(a.geometry, { length_mm: 48, width_mm: 2, thickness_mm: 1.6 });
    a.source = { name: '顶面高斯', face: 'zmax', center_mm: [16, 1, 1.6], radius_mm: 0, direction: [0, 0, -1], amplitude_mpa: 1, frequency_mhz: .4, cycles: 5, delay_us: 0, phase_deg: 0, signal: { type: 'tone_burst', window: 'hann' }, profile: { type: 'gaussian', sigma_mm: 2 } };
    a.sources = []; a.boundary = { fixed_face: 'none' };
    a.boundaries = [{ type: 'symmetry', face: 'ymax' }, { type: 'absorbing', face: 'xmin', length_mm: 8, ramp: 'cubic', attenuation_db: 60 }, { type: 'absorbing', face: 'xmax', length_mm: 8, ramp: 'cubic', attenuation_db: 60 }];
    a.acquisition = { duration_us: 30, sample_step_us: .05 }; a.numerics = { ...a.numerics, h_mm: 1 };
    a.receivers = [22, 28, 34].map((x, i) => ({ name: `top_${i}`, point_mm: [x, 1, 1.6], direction: [0, 0, 1] }));
    out.gaussian = { label: '顶面高斯激励 · 两端吸收（v2）', scene: a };
    const c = clone(a);
    c.name = '铝板 · 双贴片反相 · 两端吸收 · 半模型'; c.geometry.thickness_mm = 2;
    c.source = { ...c.source, name: '贴片 1', center_mm: [13, 1, 2], frequency_mhz: .25, profile: { type: 'gaussian', sigma_mm: 1.5 } };
    c.sources = [{ ...clone(c.source), name: '贴片 2', center_mm: [19, 1, 2], direction: [0, 0, 1], delay_us: 1 }];
    c.receivers = c.receivers.map(r => ({ ...r, point_mm: [r.point_mm[0], 1, 2] }));
    out.twopatch = { label: '双贴片反相激励（v2）', scene: c };
  }
  return out;
}
