// Property inspector: one form per selected model-tree item, bound to the scene by dotted paths.
import {
  FACES, FACE_LABELS, KIND_LABELS, SIGNAL_LABELS, WINDOW_LABELS, PROFILE_LABELS, BOUNDARY_LABELS, RAMP_LABELS,
  getPath, setPath, sourceAt, setSignalType, setProfileType, setSourceFace, inwardNormal, parseSamples, extent,
} from './scene-model.js';

import { escapeHtml } from './inspector-util.js';
import { materialMarkup, bindMaterial } from './material-form.js';
export { escapeHtml };
const options = (entries, current) => Object.entries(entries).map(([k, v]) => `<option value="${k}" ${String(current) === String(k) ? 'selected' : ''}>${escapeHtml(v)}</option>`).join('');

function num(label, path, value, { unit = '', min, max, step = 'any', hint = '', scale = 1 } = {}) {
  const shown = value === undefined || value === null ? '' : +(value / scale).toPrecision(12);
  return `<label class="wb-field"><span>${escapeHtml(label)}${unit ? ` <em>${escapeHtml(unit)}</em>` : ''}</span>
    <input type="number" step="${step}" ${min !== undefined ? `min="${min}"` : ''} ${max !== undefined ? `max="${max}"` : ''} data-path="${path}" data-kind="number" data-scale="${scale}" value="${shown}">
    ${hint ? `<small>${escapeHtml(hint)}</small>` : ''}</label>`;
}
function sel(label, path, value, entries, { action = '', hint = '' } = {}) {
  return `<label class="wb-field"><span>${escapeHtml(label)}</span><select data-path="${path}" data-kind="select" ${action ? `data-action="${action}"` : ''}>${options(entries, value)}</select>${hint ? `<small>${escapeHtml(hint)}</small>` : ''}</label>`;
}
function text(label, path, value, max = 60) {
  return `<label class="wb-field wide"><span>${escapeHtml(label)}</span><input type="text" maxlength="${max}" data-path="${path}" data-kind="text" value="${escapeHtml(value ?? '')}"></label>`;
}
function check(label, path, value) {
  return `<label class="wb-check"><input type="checkbox" data-path="${path}" data-kind="bool" ${value ? 'checked' : ''}><span>${escapeHtml(label)}</span></label>`;
}
function vector(label, path, value, unit, hint = '') {
  return `<div class="wb-field wide"><span>${escapeHtml(label)}${unit ? ` <em>${unit}</em>` : ''}</span><div class="wb-vector">${['X', 'Y', 'Z'].map((a, i) =>
    `<label><b>${a}</b><input type="number" step="any" data-path="${path}.${i}" data-kind="number" data-scale="1" value="${+(+value[i]).toPrecision(12)}"></label>`).join('')}</div>${hint ? `<small>${escapeHtml(hint)}</small>` : ''}</div>`;
}
const section = (title, body, note = '') => `<section class="wb-section"><h3>${escapeHtml(title)}</h3><div class="wb-grid">${body}</div>${note ? `<p class="wb-note">${note}</p>` : ''}</section>`;

function geometryForm(scene) {
  const g = scene.geometry, p = 'geometry';
  let body = sel('结构类型', `${p}.kind`, g.kind, KIND_LABELS, { action: 'geometry-kind' }) + num('长度', `${p}.length_mm`, g.length_mm, { unit: 'mm', min: .001 });
  if (g.kind === 'pipe') body += num('外半径', `${p}.outer_radius_mm`, g.outer_radius_mm, { unit: 'mm' }) + num('内半径', `${p}.inner_radius_mm`, g.inner_radius_mm, { unit: 'mm' });
  else body += num('宽度 Y', `${p}.width_mm`, g.width_mm, { unit: 'mm' }) + num('厚度 Z', `${p}.thickness_mm`, g.thickness_mm, { unit: 'mm' });
  if (g.kind === 'hole') body += num('孔半径', `${p}.hole_radius_mm`, g.hole_radius_mm, { unit: 'mm' }) + num('孔中心 X', `${p}.hole_x_mm`, g.hole_x_mm, { unit: 'mm' }) + num('孔中心 Y', `${p}.hole_y_mm`, g.hole_y_mm, { unit: 'mm' });
  if (g.kind === 'notch') body += num('缺口中心 X', `${p}.notch_x_mm`, g.notch_x_mm, { unit: 'mm' }) + num('缺口宽度', `${p}.notch_width_mm`, g.notch_width_mm, { unit: 'mm' }) + num('缺口深度', `${p}.notch_depth_mm`, g.notch_depth_mm, { unit: 'mm' });
  return section('几何', body, g.kind === 'pipe' ? '管轴为 X，截面中心在 Y = Z = 0；管只能在 X 端面加载。' : '坐标原点在板的一角，X 为长度方向，Z 为厚度方向。') +
    section('名称', text('场景名称', 'name', scene.name, 100));
}

function signalForm(src, p) {
  const s = src.signal || { type: 'tone_burst', window: 'hann' }, q = `${p}.signal`;
  let body = sel('信号类型', `${q}.type`, s.type, SIGNAL_LABELS, { action: 'signal-type' });
  if (s.type === 'tone_burst') {
    body += num('周期数', `${p}.cycles`, src.cycles, { min: 1, max: 20, step: .5 }) + sel('窗函数', `${q}.window`, s.window, WINDOW_LABELS, { action: 'rerender' });
    if (s.window === 'tukey') body += num('Tukey α', `${q}.tukey_alpha`, s.tukey_alpha, { min: .05, max: 1 });
    if (s.window === 'gaussian') body += num('高斯宽度 σ/T', `${q}.gaussian_width`, s.gaussian_width, { min: .05, max: .5 });
  } else if (s.type === 'gaussian_pulse') body += num('相对带宽', `${q}.bandwidth`, s.bandwidth, { min: .1, max: 2, hint: '−6 dB 全带宽 / 中心频率' });
  else if (s.type === 'chirp') body += num('终止频率', `${q}.end_frequency_mhz`, s.end_frequency_mhz, { unit: 'MHz' }) + num('扫频时长', `${q}.duration_us`, s.duration_us, { unit: 'μs' }) + sel('窗函数', `${q}.window`, s.window, { hann: 'Hann', rectangular: '矩形' });
  else if (s.type === 'multi_tone') {
    body += num('包络时长', `${q}.duration_us`, s.duration_us, { unit: 'μs' }) + sel('窗函数', `${q}.window`, s.window, { hann: 'Hann', rectangular: '矩形' });
    body += `<div class="wb-field wide"><span>频率分量（载频须等于其中一个）</span><table class="wb-table compact"><tr><th>频率 MHz</th><th>权重</th><th>相位 °</th><th></th></tr>${s.components.map((c, i) => `<tr>
      <td><input type="number" step="any" data-path="${q}.components.${i}.frequency_mhz" data-kind="number" data-scale="1" value="${c.frequency_mhz}"></td>
      <td><input type="number" step="any" min="0" max="1" data-path="${q}.components.${i}.weight" data-kind="number" data-scale="1" value="${c.weight}"></td>
      <td><input type="number" step="any" data-path="${q}.components.${i}.phase_deg" data-kind="number" data-scale="1" value="${c.phase_deg}"></td>
      <td>${s.components.length > 1 ? `<button type="button" class="wb-icon" data-action="component-remove" data-index="${i}" title="删除分量">×</button>` : ''}</td></tr>`).join('')}</table>
      ${s.components.length < 8 ? '<button type="button" class="wb-button small" data-action="component-add">+ 添加分量</button>' : ''}</div>`;
  } else if (s.type === 'samples') {
    body += num('采样间隔', `${q}.sample_step_us`, s.sample_step_us, { unit: 'μs', min: .0001, max: 10 }) + check('按峰值归一化（幅值由下方“幅值”给出）', `${q}.normalize`, s.normalize);
    body += `<div class="wb-field wide"><span>采样值：${s.values.length} 个，时长 ${(s.values.length * s.sample_step_us).toPrecision(4)} μs</span>
      <label class="wb-button small file">从 CSV / TXT 导入<input type="file" accept=".csv,.txt,.tsv,text/plain" data-action="samples-file" hidden></label>
      <small>每行一个数值，或“时间 μs, 数值”两列（等间隔）。最多 20000 个。</small></div>`;
  }
  if (s.type !== 'tone_burst') body += `<input type="hidden" data-path="${p}.cycles" value="${src.cycles}">`;
  return body;
}

function profileForm(src, p) {
  const prof = src.profile || { type: 'uniform' }, q = `${p}.profile`;
  let body = sel('空间分布', `${q}.type`, prof.type, PROFILE_LABELS, { action: 'profile-type' }) + num('作用区半径（0 = 整个面）', `${p}.radius_mm`, src.radius_mm, { unit: 'mm', min: 0 });
  if (prof.type === 'gaussian') body += num('高斯宽度 σ', `${q}.sigma_mm`, prof.sigma_mm, { unit: 'mm', min: .01 });
  if (prof.type === 'comb') body += num('齿间距', `${q}.pitch_mm`, prof.pitch_mm, { unit: 'mm' }) + num('齿数', `${q}.teeth`, prof.teeth, { min: 1, max: 64, step: 1 }) +
    num('齿宽', `${q}.tooth_width_mm`, prof.tooth_width_mm, { unit: 'mm' }) + sel('排列方向', `${q}.along`, prof.along, { x: 'X', y: 'Y', z: 'Z' }) + check('相邻齿反相', `${q}.alternating_sign`, prof.alternating_sign);
  return body;
}

function sourceForm(scene, index) {
  const src = sourceAt(scene, index), p = index === 0 ? 'source' : `sources.${index - 1}`;
  const faces = Object.fromEntries(FACES.map(f => [f, FACE_LABELS[f]]));
  const head = text('名称', `${p}.name`, src.name ?? (index ? `激励 ${index + 1}` : '主激励'));
  const place = sel('加载面', `${p}.face`, src.face, faces, { action: 'source-face' }) + vector('作用中心', `${p}.center_mm`, src.center_mm, 'mm') +
    vector('牵引方向', `${p}.direction`, src.direction, '', '名义牵引（按参考面积，方向固定）。') + `<div class="wb-field wide"><button type="button" class="wb-button small" data-action="source-normal">设为该面的内法向</button></div>`;
  const carrier = num('幅值', `${p}.amplitude_mpa`, src.amplitude_mpa, { unit: 'MPa', min: .00001, max: 10 }) + num('载频', `${p}.frequency_mhz`, src.frequency_mhz, { unit: 'MHz', min: .001, max: 10 }) +
    num('延迟', `${p}.delay_us`, src.delay_us, { unit: 'μs', min: 0, max: 1000 }) + num('初相', `${p}.phase_deg`, src.phase_deg, { unit: '°', min: -360, max: 360 });
  return section('激励源', head + (index ? `<div class="wb-field wide"><button type="button" class="wb-button danger small" data-action="remove" data-id="source:${index}">删除此激励</button></div>` : '')) +
    section('位置与方向', place) + section('时间信号', carrier + signalForm(src, p), '信号形状与频谱在“激励信号”页预览。') + section('空间分布', profileForm(src, p));
}

function boundaryForm(scene, index) {
  const b = scene.boundaries[index], p = `boundaries.${index}`;
  let body = sel('类型', `${p}.type`, b.type, BOUNDARY_LABELS, { action: 'boundary-type' }) + sel('作用面', `${p}.face`, b.face, Object.fromEntries(FACES.map(f => [f, FACE_LABELS[f]])));
  if (b.type === 'absorbing') body += num('层厚', `${p}.length_mm`, b.length_mm, { unit: 'mm', min: 0 }) + sel('阻尼分布', `${p}.ramp`, b.ramp, RAMP_LABELS) + num('目标衰减', `${p}.attenuation_db`, b.attenuation_db, { unit: 'dB', min: 1, max: 200 });
  const notes = { symmetry: '对称面：法向位移为零，切向自由。用来只算半个模型。', antisymmetry: '反对称面：切向位移为零，法向自由。', absorbing: '质量比例阻尼层 f = −ρ₀α(d)∂u/∂t，α 按层内深度的幂次增加，峰值由层厚和目标衰减确定；三个分量都加。不会改动材料本构。' };
  return section('边界条件', body + `<div class="wb-field wide"><button type="button" class="wb-button danger small" data-action="remove" data-id="boundary:${index}">删除此边界</button></div>`, notes[b.type]);
}

function receiverForm(scene, index) {
  const r = scene.receivers[index], p = `receivers.${index}`;
  return section('接收点', text('名称', `${p}.name`, r.name) + vector('位置', `${p}.point_mm`, r.point_mm, 'mm', '在参考构形中取值；结果页可再任取点和直线批量提取。') + vector('投影方向', `${p}.direction`, r.direction, '') +
    (scene.receivers.length > 1 ? `<div class="wb-field wide"><button type="button" class="wb-button danger small" data-action="remove" data-id="receiver:${index}">删除此接收点</button></div>` : ''));
}

const ENGINES = { network: '物理 AI · 冻结网络（细网格，默认）', coarse: '物理 AI · 粗网格算子（快速，研究模型）' };

function studyForm(scene, degree, engine = 'network') {
  const a = scene.acquisition, n = scene.numerics, faces = { none: '无（全部自由）', ...Object.fromEntries(FACES.map(f => [f, FACE_LABELS[f]])) };
  return section('记录', num('记录时长', 'acquisition.duration_us', a.duration_us, { unit: 'μs', min: .1, max: 20000, hint: '最长 20 ms；5 ms 及以上的记录可在计算中暂停、之后续算。' }) +
      num('输出采样间隔', 'acquisition.sample_step_us', a.sample_step_us, { unit: 'μs', min: .01, max: .2, hint: '二次谐波每周期至少 8 个采样点。' })) +
    section('计算引擎', `<label class="wb-field wide"><span>引擎</span><select data-action="engine">${Object.entries(ENGINES).map(([k, v]) => `<option value="${k}" ${k === engine ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
      <div class="wb-field wide" id="engine-derived"></div>`,
      engine === 'coarse' ? '粗网格算子是同一个网络换上学到的单元算子（求导滤波、刚度与质量积分），在模型自带的粗网格上计算，比冻结网络快约一个数量级。它是研究模型，只在 48 × 2 mm 半宽铝板、板厚 1.6–2.4 mm、0.2–0.4 MHz、30 μs 记录上训练和测试过；超出范围要逐例核对。下面的 h 与 p 只用于检查场景本身，计算用模型的网格。' : '') +
    section('网格与物理 AI 求解', num('单元尺寸 h', 'numerics.h_mm', n.h_mm, { unit: 'mm', min: .8, max: 2.5 }) +
      `<label class="wb-field"><span>图阶次 p</span><select data-action="degree">${[3, 4, 5].map(d => `<option value="${d}" ${d === degree ? 'selected' : ''}>${d}</option>`).join('')}</select><small>节点间距 = h / p</small></label>` +
      sel('固定面', 'boundary.fixed_face', scene.boundary.fixed_face, faces),
      '求解器是冻结图网络（物理 AI）：学到的求导、能量和状态更新逐时前向传播，不调用有限元求解器，也不做预测后的数值校正。精度由节点间距决定，见下方“2f 分辨率”；新场景要用细网格参考逐例核对。');
}

/** HTML for the selected item. */
export function inspectorMarkup(scene, selection, degree, context = {}) {
  const [kind, raw] = selection.split(':'), index = Number(raw);
  if (kind === 'geometry') return geometryForm(scene);
  if (kind === 'material') return materialMarkup(scene, context.materials);
  if (kind === 'source') return sourceForm(scene, index);
  if (kind === 'boundary') return boundaryForm(scene, index);
  if (kind === 'receiver') return receiverForm(scene, index);
  if (kind === 'fixed' || kind === 'study') return studyForm(scene, degree, context.engine);
  if (kind === 'sources') return section('激励源', '<p class="wb-note">点击左侧的“+”添加激励源，或选中一个激励编辑。每个激励有独立的加载面、方向、时间信号和空间分布。</p>');
  if (kind === 'boundaries') return section('边界条件', '<p class="wb-note">点击“+”添加对称面、反对称面或吸收层。固定面在“记录与网格”中设置。</p>');
  if (kind === 'receivers') return section('接收点', '<p class="wb-note">点击“+”添加接收点。完整位移场全部保存，结果页可以任取点、直线提取波形。</p>');
  return '';
}

/**
 * Bind inputs in `root` to `scene`. `onChange(kind)` is called after each edit with
 * 'value' | 'structure' (the tree or the form must be re-rendered) | {select: id}.
 */
export function bindInspector(root, scene, state, onChange) {
  const sourceIndex = () => Number(state.selection.split(':')[1]);
  root.querySelectorAll('[data-path]').forEach(el => {
    if (el.type === 'hidden') return;
    el.addEventListener('change', () => {
      const path = el.dataset.path;
      try {
        if (el.dataset.kind === 'number') {
          if (el.value === '' || !Number.isFinite(Number(el.value))) { el.value = getPath(scene, path); return; }
          setPath(scene, path, Number(el.value) * Number(el.dataset.scale || 1));
        } else if (el.dataset.kind === 'bool') setPath(scene, path, el.checked);
        else if (el.dataset.kind === 'text') setPath(scene, path, el.value.trim() || getPath(scene, path));
        else if (el.dataset.action === 'signal-type') setSignalType(sourceAt(scene, sourceIndex()), el.value);
        else if (el.dataset.action === 'profile-type') setProfileType(sourceAt(scene, sourceIndex()), el.value);
        else if (el.dataset.action === 'source-face') setSourceFace(scene, sourceAt(scene, sourceIndex()), el.value);
        else if (el.dataset.action === 'boundary-type') { const b = scene.boundaries[sourceIndex()]; const face = b.face; Object.keys(b).forEach(k => delete b[k]); Object.assign(b, el.value === 'absorbing' ? { type: 'absorbing', face, length_mm: Math.min(8, extent(scene.geometry).size['xyz'.indexOf(face[0])] / 4), ramp: 'cubic', attenuation_db: 60 } : { type: el.value, face }); }
        else if (el.dataset.action === 'geometry-kind') { scene.geometry.kind = el.value; if (el.value === 'pipe') for (const s of [scene.source, ...(scene.sources || [])]) if (s.face[0] !== 'x') setSourceFace(scene, s, 'xmin'); }
        else setPath(scene, path, el.value);
        onChange(el.dataset.action ? 'structure' : (path.endsWith('.name') || path.startsWith('geometry') || path.includes('face') ? 'structure' : 'value'));
      } catch (error) { onChange({ error }); }
    });
  });
  if (state.selection === 'material') bindMaterial(root, scene, state.materials || {}, onChange);
  root.querySelectorAll('select[data-action="engine"]').forEach(el => el.addEventListener('change', () => { state.engine = el.value; onChange('structure'); }));
  root.querySelectorAll('select[data-action="degree"]').forEach(el => el.addEventListener('change', () => { state.degree = Number(el.value); onChange('value'); }));
  root.querySelectorAll('button[data-action]').forEach(el => el.addEventListener('click', () => {
    const action = el.dataset.action;
    try {
      if (action === 'remove') return onChange({ remove: el.dataset.id });
      const src = sourceAt(scene, sourceIndex());
      if (action === 'source-normal') src.direction = inwardNormal(src.face);
      if (action === 'component-add') src.signal.components.push({ frequency_mhz: +(src.frequency_mhz * (1 + .5 * src.signal.components.length)).toFixed(4), weight: .5, phase_deg: 0 });
      if (action === 'component-remove') src.signal.components.splice(Number(el.dataset.index), 1);
      onChange('structure');
    } catch (error) { onChange({ error }); }
  }));
  root.querySelectorAll('input[data-action="samples-file"]').forEach(el => el.addEventListener('change', async () => {
    try {
      const file = el.files[0]; if (!file) return; if (file.size > 600000) throw Error('文件太大（上限约 600 KB）。');
      const parsed = parseSamples(await file.text()), src = sourceAt(scene, sourceIndex());
      src.signal.values = parsed.values; if (parsed.sample_step_us) src.signal.sample_step_us = parsed.sample_step_us;
      const peak = Math.max(...parsed.values.map(Math.abs)); if (peak > 1) src.signal.normalize = true;
      onChange('structure');
    } catch (error) { onChange({ error }); } finally { el.value = ''; }
  }));
}
