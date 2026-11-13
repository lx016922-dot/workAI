// Property inspector: one form per selected model-tree item, bound to the scene by dotted paths.
import {
  FACES, FACE_LABELS, KIND_LABELS, SIGNAL_LABELS, WINDOW_LABELS, PROFILE_LABELS, BOUNDARY_LABELS, RAMP_LABELS,
  getPath, setPath, sourceAt, setSignalType, setProfileType, setSourceFace, inwardNormal, parseSamples, extent,
  DEFECT_LABELS, CRACK_FROM, defectName,
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
  return section('几何', body, g.kind === 'pipe' ? '管轴为 X，截面中心在 Y = Z = 0；管只能在 X 端面加载。' : '坐标原点在板的一角，X 为长度方向，Z 为厚度方向（Z 最大为上表面）。' +
      (g.kind === 'plate' ? '裂纹、孔、分层、接触界面和缺口在功能区“几何 → 一键添加缺陷”中添加，也可以在顶部文字框里描述，例如“在 100 mm 处添加裂纹”。' : '')) +
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
  const notes = { symmetry: '对称面（镜像面）：几何、材料和所有激励都关于这个面镜像对称时，只建一半模型，结果与整块板相同、计算量减半。面上法向位移为零，切向自由。例：默认模板半宽 2 mm 加 y 最大面对称，代表 4 mm 宽的整板。', antisymmetry: '反对称面：几何和材料关于这个面对称、激励镜像后反向时（例如垂直于该面的力）用它只算一半。面上切向位移为零，法向自由。', absorbing: '质量比例阻尼层 f = −ρ₀α(d)∂u/∂t，α 按层内深度的幂次增加，峰值由层厚和目标衰减确定；三个分量都加。不会改动材料本构。' };
  return section('边界条件', body + `<div class="wb-field wide"><button type="button" class="wb-button danger small" data-action="remove" data-id="boundary:${index}">删除此边界</button></div>`, notes[b.type]);
}

function receiverForm(scene, index) {
  const r = scene.receivers[index], p = `receivers.${index}`;
  return section('接收点', text('名称', `${p}.name`, r.name) + vector('位置', `${p}.point_mm`, r.point_mm, 'mm', '在参考构形中取值；结果页可再任取点和直线批量提取。') + vector('投影方向', `${p}.direction`, r.direction, '') +
    (scene.receivers.length > 1 ? `<div class="wb-field wide"><button type="button" class="wb-button danger small" data-action="remove" data-id="receiver:${index}">删除此接收点</button></div>` : ''));
}

const ENGINES = { network: '物理 AI · 冻结网络（细网格，默认）', coarse: '物理 AI · 粗网格算子（快速，研究模型）' };
/** Shipped coarse models (offline/coarse_engine.py MODELS); the first is the default. */
export const COARSE_MODELS = { 'p4-h3-harmonic': '快速 · h 3 mm（默认）', 'p4-h2-multisource': '精确 · h 2 mm（节点约 1.5 倍，二次谐波更准）' };
export const DEFAULT_COARSE_MODEL = Object.keys(COARSE_MODELS)[0];

/** Record modes and time layers of the network engine (offline/domain.py RECORDS, TIME_STEPPING). */
export const RECORD_MODES = { auto: '自动（完整记录不超过 2 GiB 时保存全部节点，否则精简）', full: '完整记录（全部节点、全部采样点）', compact: '精简记录（接收点完整波形 + 表面动画 + 快照）' };
/** Constitutive controls a computation runs (offline/domain.py CONTROL_SETS) and the responses they give. */
export const CONTROL_SETS = { all: '全部：线性、StVK、完整（五种响应，可分离几何与材料增量）', full_linear: '完整 + 线性（完整、线性、非线性增量）', full: '仅完整响应' };
export const TIME_LAYERS = { auto: '自动（按稳定性和相位误差选最省的时间层）', frozen: '训练时的时间层（三级四阶，CFL 0.3）' };

function recordForm(scene) {
  const a = scene.acquisition;
  return section('瞬态记录', num('记录时长', 'acquisition.duration_us', a.duration_us, { unit: 'μs', min: .1, max: 20000, hint: '最长 20 ms；计算中可以暂停、之后续算。' }) +
      num('输出采样间隔', 'acquisition.sample_step_us', a.sample_step_us, { unit: 'μs', min: .01, max: .2, hint: '二次谐波每周期至少 8 个采样点。' }),
    '模型不大时保存全部节点、全部输出时刻的完整位移场（所算的各组本构控制），结果窗口可任取点和直线提取波形与频谱。完整记录超过 2 GiB 时自动改用精简记录，只在接收点保存完整波形；记录方式可在“求解器配置”中指定。');
}

function engineForm(engine = 'network', coarseModel = DEFAULT_COARSE_MODEL, record = 'auto', timeStepping = 'auto', controls = 'all') {
  const pick = (action, options, value) => `<select data-action="${action}">${Object.entries(options).map(([k, v]) => `<option value="${k}" ${k === value ? 'selected' : ''}>${v}</option>`).join('')}</select>`;
  return section('计算引擎', `<label class="wb-field wide"><span>引擎</span><select data-action="engine">${Object.entries(ENGINES).map(([k, v]) => `<option value="${k}" ${k === engine ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
      ${engine === 'coarse' && Object.keys(COARSE_MODELS).length > 1 ? `<label class="wb-field wide"><span>粗网格模型</span><select data-action="coarse-model">${Object.entries(COARSE_MODELS).map(([k, v]) => `<option value="${k}" ${k === coarseModel ? 'selected' : ''}>${v}</option>`).join('')}</select></label>` : ''}
      <label class="wb-field wide"><span>计算的响应</span>${pick('control-set', CONTROL_SETS, controls)}</label>
      ${engine === 'coarse' ? '' : `<label class="wb-field wide"><span>记录方式</span>${pick('record-mode', RECORD_MODES, record)}</label>
      <label class="wb-field wide"><span>时间层</span>${pick('time-layer', TIME_LAYERS, timeStepping)}</label>`}
      <div class="wb-field wide" id="engine-derived"></div>`,
    engine === 'coarse' ? '粗网格算子是同一个网络换上学到的单元算子（求导滤波、刚度与质量积分），在模型自带的粗网格上计算，比细网格冻结网络快数倍到一个数量级。快速模型（h 3 mm）最省；精确模型（h 2 mm）节点约 1.5 倍，顶面、端面和双贴片的留出测试全部通过，需要二次谐波时选它。它们是研究模型，只在 48 × 2 mm 半宽铝板、板厚 1.6–2.4 mm、0.2–0.4 MHz、30 μs 记录上训练和测试过；超出范围要逐例核对。网格设置中的 h 与 p 只用于检查模型本身，计算用模型自带的网格。'
      : '求解器是冻结图网络（物理 AI）：学到的求导、能量和状态更新逐时前向传播，不调用有限元求解器，也不做预测后的数值校正。精度由节点间距决定，见“消息”窗口中的 2f 分辨率；新模型要用细网格参考逐例核对。模型大小只受本机内存、磁盘和时间限制：大模型自动保存精简记录，动画用抽稀的表面网格。时间层“自动”按实测的稳定极限和相位误差上界（0.01 rad）为短记录选二阶单级、为长记录选三级四阶；要逐位重现旧结果请选“训练时的时间层”。“计算的响应”决定算几组本构控制：只要完整响应或只要总的非线性增量时不必三组都算，可省 CPU、内存和磁盘；每组控制占一个线程，核数不少于控制数时耗时由完整控制决定。');
}

function meshForm(scene, auto) {
  const rule = '软件按每个激励的 −30 dB 带宽和二次谐波（2f）自动选择单元尺寸：2f 剪切波长内的节点数目标为 10（记录 ≥ 1 ms 时为 15，按 1 ms 平板实测末段误差约 0.1%），'
    + '单元尺寸限定在网络验证过的 0.8–2.5 mm、阶次 4；网格线穿过每个缺陷的边缘（裂纹尖端、分层前沿、缺口角点、孔的 O 型网格）。节点数只受本机内存限制（三组控制都算时约 2 kB / 节点，只算完整响应约 0.9 kB / 节点；不超过物理内存的 80%）：放不下时才放粗，并在“消息”中提示。';
  if (!auto) return section('网格（自动）', '<p class="wb-note">正在根据当前模型选择网格…（模型有错误时显示在“消息”窗口）</p>', rule);
  const f = (v, d = 3) => (+v).toFixed(d).replace(/\.?0+$/, '');
  const rows = [['单元尺寸 h', `${f(auto.h_mm)} mm`], ['阶次 p', auto.degree], ['节点间距 h / p', `${f(auto.node_spacing_mm)} mm（最小 ${f(auto.min_node_spacing_mm)} mm）`],
    ['2f 分辨率', `${f(auto.nodes_per_2f_shear_wavelength, 1)} 个节点 / 2f 剪切波长（目标 ${auto.target}）`], ['二次谐波上限', `${f(auto.band_top_mhz)} MHz（−30 dB 带宽上限的 2 倍）`],
    ['规模', `${auto.nodes.toLocaleString()} 个节点 · ${auto.elements.toLocaleString()} 个单元${auto.curved ? ' · 含曲边（孔）' : ''}`], ['选择依据', auto.reason]];
  if (auto.split_nodes) rows.push(['缺陷面', `分开 ${auto.split_nodes} 个节点${auto.contact_pairs ? `，${auto.contact_pairs} 对接触` : ''}`]);
  return section('网格（自动）', `<table class="wb-table">${rows.map(([k, v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(String(v))}</td></tr>`).join('')}</table>` +
    (auto.warning ? `<p class="wb-note warn">${escapeHtml(auto.warning)}</p>` : ''), rule);
}

const SPLIT_NOTE = '缺陷面上的节点一分为二，两侧独立运动；位于材料内部的前沿（裂纹尖端）保持连接，位于外表面或对称面上的边缘张开。';
const CONTACT_NOTE = '闭合：两个面静止时贴合，受压时由罚函数弹簧传递法向力，受拉时张开、不传力，切向无摩擦（“呼吸”裂纹，产生接触非线性的高次谐波）。罚刚度取时间层稳定允许的最大值，受压时的残余反射显示在计算结果的诊断中。';
function defectForm(scene, index) {
  const d = scene.defects[index], p = `defects.${index}`, T = scene.geometry.thickness_mm;
  let body = text('名称', `${p}.name`, defectName(d, index), 40) + `<div class="wb-field"><span>类型</span><b>${escapeHtml(DEFECT_LABELS[d.type])}</b></div>`, note = '';
  if (d.type === 'crack') {
    body += num('位置 x', `${p}.x_mm`, d.x_mm, { unit: 'mm' }) + sel('起始面', `${p}.from`, d.from || 'top', CRACK_FROM, { action: 'defect-shape' }) +
      num(d.from === 'inside' ? '高度' : '深度', `${p}.depth_mm`, d.depth_mm, { unit: 'mm', min: .1, max: T }) + (d.from === 'inside' ? num('中心高度 z', `${p}.z_mm`, d.z_mm, { unit: 'mm', hint: '从底面算起' }) : '') +
      check('闭合（接触，呼吸裂纹）', `${p}.closed`, d.closed);
    note = '竖直裂纹面垂直于 X，贯穿板宽。' + SPLIT_NOTE + (d.closed ? CONTACT_NOTE : '');
  } else if (d.type === 'delamination') {
    body += num('起点 x', `${p}.x_start_mm`, d.x_start_mm, { unit: 'mm' }) + num('终点 x', `${p}.x_end_mm`, d.x_end_mm, { unit: 'mm' }) +
      num('距上表面', `${p}.depth_mm`, d.depth_mm, { unit: 'mm', min: .1, max: T }) + check('闭合（接触）', `${p}.closed`, d.closed);
    note = '水平分层面平行于板面，贯穿板宽。' + SPLIT_NOTE + (d.closed ? CONTACT_NOTE : '');
  } else if (d.type === 'contact') {
    body += sel('方向', `${p}.orientation`, d.orientation, { vertical: '竖直截面（整个横截面）', horizontal: '水平层间' }, { action: 'defect-shape' });
    body += d.orientation === 'vertical' ? num('位置 x', `${p}.x_mm`, d.x_mm, { unit: 'mm' })
      : num('起点 x', `${p}.x_start_mm`, d.x_start_mm, { unit: 'mm' }) + num('终点 x', `${p}.x_end_mm`, d.x_end_mm, { unit: 'mm' }) + num('距上表面', `${p}.depth_mm`, d.depth_mm, { unit: 'mm', min: .1, max: T });
    note = '接触界面：' + CONTACT_NOTE + '适合闭合裂纹、弱粘接（kissing bond）和两件对接。';
  } else if (d.type === 'notch') {
    body += num('中心 x', `${p}.x_mm`, d.x_mm, { unit: 'mm' }) + num('宽度', `${p}.width_mm`, d.width_mm, { unit: 'mm', min: .1 }) + num('深度', `${p}.depth_mm`, d.depth_mm, { unit: 'mm', min: .1, max: T }) +
      sel('开口面', `${p}.from`, d.from || 'top', { top: '上表面', bottom: '下表面' });
    note = '矩形槽，贯穿板宽；槽内的单元被删除。';
  } else {
    body += sel('孔的方向', `${p}.axis`, d.axis || 'z', { z: '通孔（贯穿板厚）', y: '横孔（贯穿板宽，侧钻孔）' }, { action: 'defect-shape' }) +
      num('中心 x', `${p}.x_mm`, d.x_mm, { unit: 'mm' }) + num('直径', `${p}.radius_mm`, d.radius_mm, { unit: 'mm', min: .2, scale: .5 }) +
      ((d.axis || 'z') === 'z' ? num('中心 y', `${p}.y_mm`, d.y_mm, { unit: 'mm' }) : num('中心高度 z', `${p}.z_mm`, d.z_mm, { unit: 'mm', hint: '从底面算起' }));
    const mirrored = (scene.boundaries || []).some(b => b.type === 'symmetry' && b.face === 'ymax') && (d.axis || 'z') === 'z';
    note = '孔周围是贴合孔边的曲边网格（O 型网格），与周围网格逐节点连接。' + (mirrored ? '注意：模型在 Y 最大面对称，这个孔在整体模型中会被镜像为两个孔。' : '');
  }
  return section(DEFECT_LABELS[d.type], body + `<div class="wb-field wide"><button type="button" class="wb-button danger small" data-action="remove" data-id="defect:${index}">删除此缺陷</button></div>`, note);
}

/** Switching a defect's variant (crack origin, contact orientation, hole axis) adds or drops the fields that depend on it. */
export function reshapeDefect(d, scene) {
  const { length_mm: L, width_mm: W, thickness_mm: T } = scene.geometry, r = v => +(+v).toFixed(4);
  if (d.type === 'crack') { if (d.from === 'inside') { d.z_mm ??= r(T / 2); d.depth_mm = Math.min(d.depth_mm, r(T - .3)); } else delete d.z_mm; }
  if (d.type === 'contact') {
    if (d.orientation === 'vertical') { const x = d.x_mm ?? ((d.x_start_mm + d.x_end_mm) / 2 || L / 2); for (const k of ['x_start_mm', 'x_end_mm', 'depth_mm', 'y_range_mm']) delete d[k]; d.x_mm = r(x); }
    else { const x = d.x_mm ?? L / 2; delete d.x_mm; d.x_start_mm = r(Math.max(.2, x - 5)); d.x_end_mm = r(Math.min(L - .2, x + 5)); d.depth_mm = r(T / 2); }
  }
  if (d.type === 'hole') { if ((d.axis || 'z') === 'z') { delete d.z_mm; d.y_mm ??= r(W / 2); } else { delete d.y_mm; d.z_mm ??= r(T / 2); d.radius_mm = Math.min(d.radius_mm, r(T / 2 - .2)); } }
  return d;
}

function fixedForm(scene) {
  const faces = { none: '无（全部自由）', ...Object.fromEntries(FACES.map(f => [f, FACE_LABELS[f]])) };
  return section('固定面', sel('固定面', 'boundary.fixed_face', scene.boundary.fixed_face, faces) +
    '<div class="wb-field wide"><button type="button" class="wb-button danger small" data-action="remove" data-id="fixed">删除固定面</button></div>', '固定面上三个位移分量为零。管只能固定 X 端面。');
}

function overviewForm(scene, kind) {
  if (kind === 'scene') return section('模型', text('名称', 'name', scene.name, 100), '模型文件保存为 JSON（文件 → 保存），包含几何、材料、激励、边界、接收点、记录、网格和计算引擎设置。');
  if (kind === 'component') return section('组件 1', `<p class="wb-note">三维固体组件。单位：长度 mm，时间 μs，频率 MHz，应力 MPa。坐标原点在${scene.geometry.kind === 'pipe' ? '管轴一端（管轴为 X）' : '板的一角，X 为长度方向，Z 为厚度方向'}。</p>`);
  if (kind === 'physics') return section('超声导波（固体力学）', `<p class="wb-note">非线性弹性动力学：线性、StVK（几何非线性）和完整 Murnaghan / 应变能（材料非线性）三组控制同时计算，结果给出完整、线性、StVK、几何增量和材料增量五种响应。</p>
    <p class="wb-note">在下方添加激励源、边界条件（对称、反对称、吸收层、固定面）和接收点。</p>`);
  return section('结果', '<p class="wb-note">每次计算生成一条记录。选中一条记录查看位移场动画、任意点波形与频谱、多点提取，并可导出全部结果 ZIP。</p>');
}

/** HTML for the selected item. */
export function inspectorMarkup(scene, selection, degree, context = {}) {
  const [kind, raw] = selection.split(':'), index = Number(raw);
  if (kind === 'geometry') return geometryForm(scene);
  if (kind === 'material') return materialMarkup(scene, context.materials);
  if (kind === 'source') return sourceForm(scene, index);
  if (kind === 'boundary') return boundaryForm(scene, index);
  if (kind === 'receiver') return receiverForm(scene, index);
  if (kind === 'fixed') return fixedForm(scene);
  if (kind === 'study') return recordForm(scene);
  if (kind === 'engine') return engineForm(context.engine, context.coarseModel, context.record, context.timeStepping, context.controls);
  if (kind === 'mesh') return meshForm(scene, context.meshAuto);
  if (kind === 'defect') return defectForm(scene, index);
  if (kind === 'defects') return section('缺陷', '<p class="wb-note">在功能区“几何 → 一键添加缺陷”中添加裂纹、孔、分层、接触界面或缺口，或在顶部文字框里描述（例如“在 100 mm 处添加深 0.5 mm 的裂纹”）。缺陷之间需至少相隔 0.2 mm。</p>');
  if (['scene', 'component', 'physics', 'results'].includes(kind)) return overviewForm(scene, kind);
  if (kind === 'sources') return section('激励源', '<p class="wb-note">点击“+”添加激励源，或选中一个激励编辑。每个激励有独立的加载面、方向、时间信号和空间分布。</p>');
  if (kind === 'boundaries') return section('边界条件', '<p class="wb-note">点击“+”添加对称面、反对称面、吸收层或固定面。</p>');
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
        else if (el.dataset.action === 'defect-shape') { setPath(scene, path, el.value); reshapeDefect(scene.defects[sourceIndex()], scene); }
        else if (el.dataset.action === 'geometry-kind') { scene.geometry.kind = el.value; if (el.value === 'pipe') for (const s of [scene.source, ...(scene.sources || [])]) if (s.face[0] !== 'x') setSourceFace(scene, s, 'xmin'); }
        else setPath(scene, path, el.value);
        onChange(el.dataset.action ? 'structure' : (path.endsWith('.name') || path.startsWith('geometry') || path.includes('face') ? 'structure' : 'value'));
      } catch (error) { onChange({ error }); }
    });
  });
  if (state.selection === 'material') bindMaterial(root, scene, state.materials || {}, onChange);
  root.querySelectorAll('select[data-action="engine"]').forEach(el => el.addEventListener('change', () => { state.engine = el.value; onChange('structure'); }));
  root.querySelectorAll('select[data-action="coarse-model"]').forEach(el => el.addEventListener('change', () => { state.coarseModel = el.value; onChange('structure'); }));
  root.querySelectorAll('select[data-action="control-set"]').forEach(el => el.addEventListener('change', () => { state.controls = el.value; onChange('structure'); }));
  root.querySelectorAll('select[data-action="record-mode"]').forEach(el => el.addEventListener('change', () => { state.record = el.value; onChange('structure'); }));
  root.querySelectorAll('select[data-action="time-layer"]').forEach(el => el.addEventListener('change', () => { state.timeStepping = el.value; onChange('structure'); }));
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
