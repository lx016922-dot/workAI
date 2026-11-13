// WorkAI desktop workbench: ribbon, model builder, settings, graphics / signals / results, messages-progress-log dock.
import { unitVector } from './solid-data.js';
import { decodeSpectralField, spectralPointSeries } from './spectral-data.js';
import { solidResultMarkup, mountSolidResult, chart, responseLabels } from './solid-viewer.js';
import { solidPathMarkup, mountSolidPath } from './solid-path-viewer.js';
import { treeNodes, upgrade, workbenchTemplates, addSource, removeSource, addBoundary, removeBoundary, addReceiver, removeReceiver, sourcesOf, V2, BOUNDARY_LABELS, FACES, KIND_LABELS,
  DEFECT_LABELS, DEFECT_ICONS, addDefect, removeDefect, defectName } from './scene-model.js';
import { drawScene, pick } from './model-view.js';
import { inspectorMarkup, bindInspector, escapeHtml, COARSE_MODELS, DEFAULT_COARSE_MODEL, RECORD_MODES, TIME_LAYERS, CONTROL_SETS } from './inspector.js';
import { signalsMarkup } from './signal-view.js';
import { materialDerivedMarkup } from './material-form.js';
import { mountRibbon } from './ribbon.js';
import { icon } from './icons.js';
import { renderTreeView, popupMenu, createLog, dialog, initSplitters, previewMessages, renderMessages, fmtBytes, fmtDuration, estimateMarkup } from './shell.js';

const $ = s => document.querySelector(s);
const PHASES = { starting: '准备中', meshing: '生成连接图', running: '计算中', exporting: '保存结果', completed: '已完成', failed: '未完成', paused: '已暂停' };
const VIEWS = { iso: [-.62, .52], front: [0, 0], top: [0, 1.5699], side: [-Math.PI / 2, 0] };
const DRAFT = 'workai.workbench.draft.v1';
const ENGINE_LABELS = { network: '物理 AI · 冻结网络', coarse: '物理 AI · 粗网格算子' };
const host = window.chrome?.webview || null;   // present inside the WorkAI desktop window (WebView2)
const state = { scene: null, degree: 4, engine: 'network', coarseModel: DEFAULT_COARSE_MODEL, record: 'auto', timeStepping: 'auto', controls: 'all', selection: 'geometry', preview: null, jobs: [], job: '', part: 'full', tab: 'model',
  view: { yaw: VIEWS.iso[0], pitch: VIEWS.iso[1], zoom: 1, exaggerate: true, grid: true, preset: 'iso' }, history: [], templates: {}, token: '', picks: [], submitting: false,
  expanded: new Set(['scene', 'component', 'physics', 'sources', 'boundaries', 'receivers', 'study', 'results']), info: null, phases: {}, lastPreviewKey: '' };
let previewTimer = 0, previewStamp = 0, poll = 0, generation = 0, viewer = null, pathViewer = null, ribbon = null, log = null;

async function api(url, body) {
  const response = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-WorkAI-Token': state.token }, body: JSON.stringify(body) } : undefined);
  const data = await response.json(); if (!response.ok) throw Error(data.error || '请求失败'); return data;
}
function toast(message, kind = 'error') {
  const el = $('#toast'); el.textContent = message; el.className = `wb-toast ${kind}`; el.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { el.hidden = true; }, kind === 'error' ? 7000 : 3500);
  log?.add(kind === 'error' ? 'error' : 'info', message);
}
const fail = error => toast(error?.message || String(error));
/** The mesh is always chosen by the software (offline/automesh.py): requests carry mesh 'auto', never a size or degree. */
function requestBody() {
  return { scene: state.scene, mesh: 'auto', ...(state.controls !== 'all' ? { controls: state.controls } : {}), ...(state.engine === 'coarse' ? { engine: 'coarse', coarse_model: state.coarseModel }
    : { ...(state.record !== 'auto' ? { record: state.record } : {}), ...(state.timeStepping !== 'auto' ? { time_stepping: state.timeStepping } : {}) }) };
}
/** Engine, coarse model, record mode and time layer from a stored request, draft or imported file (unknown values fall back to the defaults). */
function takeEngine(d) {
  state.engine = d?.engine === 'coarse' ? 'coarse' : 'network'; state.coarseModel = COARSE_MODELS[d?.coarse_model] ? d.coarse_model : DEFAULT_COARSE_MODEL;
  state.record = RECORD_MODES[d?.record] ? d.record : 'auto'; state.timeStepping = TIME_LAYERS[d?.time_stepping] ? d.time_stepping : 'auto'; state.controls = CONTROL_SETS[d?.controls] ? d.controls : 'all';
}
function storeDraft() { try { localStorage.setItem(DRAFT, JSON.stringify(requestBody())); } catch { /* storage unavailable */ } }
function readDraft() { try { const d = JSON.parse(localStorage.getItem(DRAFT) || 'null'); return d?.scene?.schema === V2 ? d : null; } catch { return null; } }
const running = () => state.jobs.find(j => !['completed', 'failed', 'paused'].includes(j.phase));
window.workaiBusy = () => Boolean(running());   // asked by the desktop window before it closes

// ---------- editing ----------
function snapshot() { return JSON.stringify(requestBody()); }
function remember() { state.history.push(snapshot()); if (state.history.length > 60) state.history.shift(); ribbon?.refresh(); }
function undo() {
  const last = state.history.pop(); if (!last) return;
  const d = JSON.parse(last); state.scene = d.scene; takeEngine(d);
  if (!validSelection(state.selection)) state.selection = 'geometry';
  renderAll(); schedulePreview(); storeDraft(); log.add('info', '已撤销上一步编辑。');
}
function validSelection(id) {
  const [kind, raw] = id.split(':'), i = Number(raw), s = state.scene;
  if (kind === 'source') return i === 0 || i <= (s.sources || []).length;
  if (kind === 'boundary') return i < (s.boundaries || []).length;
  if (kind === 'receiver') return i < s.receivers.length;
  if (kind === 'defect') return i < (s.defects || []).length;
  if (kind === 'defects') return s.geometry.kind === 'plate';
  if (kind === 'fixed') return s.boundary.fixed_face !== 'none';
  if (kind === 'job') return state.jobs.some(j => j.job_id === raw);
  return true;
}
function loadScene(scene, degree = state.degree, keepHistory = true) {
  if (keepHistory && state.scene) remember();
  state.scene = upgrade(scene); state.degree = degree; state.selection = 'geometry';
  renderAll(); schedulePreview(); storeDraft();
}
function changed(kind) {
  if (kind?.error) { fail(kind.error); renderInspector(); return; }
  if (kind?.remove) { removeItem(kind.remove); return; }
  if (kind === 'structure') { renderTree(); renderInspector(); }
  else renderTree();
  renderTitle(); drawView(); schedulePreview(); storeDraft(); ribbon.refresh();
}
function removeItem(id) {
  const [kind, raw] = id.split(':'), i = Number(raw);
  try {
    remember();
    if (kind === 'source') removeSource(state.scene, i);
    if (kind === 'boundary') removeBoundary(state.scene, i);
    if (kind === 'receiver') removeReceiver(state.scene, i);
    if (kind === 'fixed') state.scene.boundary.fixed_face = 'none';
    if (kind === 'defect') removeDefect(state.scene, i);
    if (!validSelection(state.selection)) state.selection = kind === 'source' ? 'sources' : kind === 'receiver' ? 'receivers' : kind === 'defect' ? (state.scene.defects?.length ? `defect:${state.scene.defects.length - 1}` : 'defects') : 'boundaries';
    renderAll(); schedulePreview(); storeDraft(); log.add('info', `已删除 ${id}。`);
  } catch (error) { state.history.pop(); fail(error); }
}
function freeFace() {
  const s = state.scene, used = new Set([...(s.boundaries || []).map(b => b.face), ...sourcesOf(s).map(x => x.face)]);
  const faces = s.geometry.kind === 'pipe' ? ['xmax', 'xmin'] : ['xmax', 'xmin', ...FACES.filter(f => f[0] !== 'x')];
  return faces.find(f => !used.has(f)) || null;
}
function addItem(kind, type) {
  try {
    remember();
    if (kind === 'source') state.selection = `source:${addSource(state.scene, state.scene.geometry.kind === 'pipe' ? 'xmin' : 'zmax')}`;
    if (kind === 'boundary' && type === 'fixed') {
      const face = freeFace(); if (!face) throw Error('没有可用的端面放置固定面。');
      state.scene.boundary.fixed_face = face; state.selection = 'fixed';
    } else if (kind === 'boundary') state.selection = `boundary:${addBoundary(state.scene, type)}`;
    if (kind === 'receiver') state.selection = `receiver:${addReceiver(state.scene)}`;
    state.expanded.add(kind === 'source' ? 'sources' : kind === 'receiver' ? 'receivers' : 'boundaries');
    renderAll(); schedulePreview(); storeDraft();
    log.add('info', `已添加${kind === 'source' ? '激励源' : kind === 'receiver' ? '接收点' : (type === 'fixed' ? '固定面' : BOUNDARY_LABELS[type])}。`);
  } catch (error) { state.history.pop(); fail(error); }
}
/** One-click defect: the server places a default of this type clear of the existing ones (physics/defects.default_defect). */
async function addDefectOf(type) {
  try {
    if (state.scene.geometry.kind !== 'plate') throw Error('缺陷目前只能加在平板上：请先在功能区“几何”中选择平板。');
    const { defect, summary } = await api('/api/defect', { type, scene: state.scene });
    remember(); const i = addDefect(state.scene, defect); state.selection = `defect:${i}`; state.expanded.add('defects');
    renderAll(); schedulePreview(); storeDraft(); setTab('model'); log.add('info', `已添加${summary}`);
  } catch (error) { fail(error); }
}
function defectMenu() { return Object.entries(DEFECT_LABELS).map(([k, v]) => ({ id: k, label: v, icon: DEFECT_ICONS[k] })); }
/** Text box: the server reads the sentences (offline/nlp.py); the operations are shown and applied only after confirmation. */
async function interpretText() {
  const input = $('#command-input'), text = input.value.trim(); if (!text) { input.focus(); return; }
  try {
    $('#command-run').disabled = true;
    const r = await api('/api/interpret', { text, scene: state.scene });
    const rows = r.operations.map(op => `<li class="${op.applied ? 'ok' : 'bad'}">${icon(op.applied ? 'check' : 'error')}<div><b>${escapeHtml(op.summary)}</b>${op.defaults?.length ? `<small>默认值：${escapeHtml(op.defaults.join('；'))}</small>` : ''}<small class="muted">“${escapeHtml(op.text)}”</small></div></li>`).join('');
    const unparsed = r.unparsed.length ? `<p class="wb-note warn">没有理解：${r.unparsed.map(escapeHtml).join('；')}。可以换一种说法，例如“在 100 mm 处添加深 0.5 mm 的裂纹”。</p>` : '';
    const choice = await dialog({ title: '文字描述 → 模型修改', wide: true, body: `<ul class="nl-ops">${rows || '<li>没有可执行的修改。</li>'}</ul>${unparsed}<p class="muted">确认后修改模型，可用 Ctrl+Z 撤销。</p>`,
      buttons: r.changed ? [{ id: 'apply', label: '应用修改', primary: true }, { id: 'cancel', label: '取消' }] : [{ id: 'cancel', label: '关闭', primary: true }] });
    if (choice !== 'apply') return;
    remember(); const before = (state.scene.defects || []).length; state.scene = r.scene;
    const after = (state.scene.defects || []).length; if (after > before) { state.selection = `defect:${after - 1}`; state.expanded.add('defects'); } else if (!validSelection(state.selection)) state.selection = 'geometry';
    renderAll(); schedulePreview(); storeDraft(); setTab('model'); input.value = '';
    for (const op of r.operations.filter(o => o.applied)) log.add('info', `文字描述：${op.summary}`);
  } catch (error) { fail(error); } finally { $('#command-run').disabled = false; }
}
function setKind(kind) {
  if (state.scene.geometry.kind === kind) { select('geometry'); return; }
  remember(); state.scene.geometry.kind = kind;
  if (kind === 'pipe') for (const s of sourcesOf(state.scene)) if (s.face[0] !== 'x') { s.face = 'xmin'; s.center_mm = [0, 0, 0]; s.direction = [1, 0, 0]; }
  state.selection = 'geometry'; renderAll(); schedulePreview(); storeDraft(); log.add('info', `结构类型改为${KIND_LABELS[kind]}。`);
}

// ---------- rendering ----------
const NODE_TITLES = { scene: '模型', component: '组件 1', geometry: '几何 1', material: '材料', physics: '超声导波（固体力学）', sources: '激励源', source: '激励源', boundaries: '边界条件',
  boundary: '边界条件', fixed: '固定面', receivers: '接收点', receiver: '接收点', mesh: '网格 1（自动）', study: '研究 1 · 瞬态', engine: '求解器配置', results: '结果', job: '计算记录',
  defects: '缺陷', defect: '缺陷' };
const NODE_ICONS = { scene: 'document', component: 'component', geometry: 'geometry', material: 'material', physics: 'physics', sources: 'folder', source: 'source', boundaries: 'folder',
  boundary: 'boundary', fixed: 'fixed', receivers: 'folder', receiver: 'receiver', mesh: 'mesh', study: 'study', engine: 'engine', results: 'results', job: 'job', defects: 'folder', defect: 'crack' };
function select(id, focus = false) {
  state.selection = id; const kind = id.split(':')[0];
  if (kind === 'job') { setTab('result'); openJob(id.slice(4)); }
  renderTree(); renderInspector(); drawView(); ribbon?.refresh();
  if (focus) $('#tree').querySelector('.mb-row.sel')?.focus();
}
function renderTree() {
  const auto = state.preview?.mesh?.auto, meshDetail = state.engine === 'coarse' ? '快速引擎用模型自带网格' : auto ? `h ${+auto.h_mm.toFixed(3)} mm · p ${auto.degree} · ${auto.nodes.toLocaleString()} 节点` : '';
  renderTreeView($('#tree'), treeNodes(state.scene, { engine: ENGINE_LABELS[state.engine], jobs: state.jobs, phases: PHASES, mesh: meshDetail }), {
    selected: state.selection, expanded: state.expanded,
    onSelect: (id, focus) => select(id, focus),
    onToggle: id => { state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id); renderTree(); },
    onAdd: (kind, anchor) => {
      if (kind !== 'boundary' && kind !== 'defect') return addItem(kind);
      const r = anchor.getBoundingClientRect();
      if (kind === 'defect') return popupMenu(r.left, r.bottom + 2, defectMenu(), type => addDefectOf(type));
      popupMenu(r.left, r.bottom + 2, boundaryMenu(), type => addItem('boundary', type));
    },
    onRemove: id => removeItem(id),
    onContext: (id, x, y) => contextMenu(id, x, y),
  });
}
function boundaryMenu() { return [...Object.entries(BOUNDARY_LABELS).map(([k, v]) => ({ id: k, label: v, icon: k })), { id: 'fixed', label: '固定面', icon: 'fixed' }]; }
function contextMenu(id, x, y) {
  const kind = id.split(':')[0], rows = [];
  if (kind === 'sources') rows.push({ id: 'add:source', label: '添加激励源', icon: 'source' });
  if (kind === 'boundaries') rows.push(...boundaryMenu().map(r => ({ ...r, id: `add-boundary:${r.id}`, label: `添加${r.label}` })));
  if (kind === 'receivers') rows.push({ id: 'add:receiver', label: '添加接收点', icon: 'receiver' });
  if (kind === 'defects' || kind === 'geometry' && state.scene.geometry.kind === 'plate') rows.push(...defectMenu().map(r => ({ ...r, id: `add-defect:${r.id}`, label: `添加${r.label}` })), ...(kind === 'geometry' ? [{ sep: true }] : []));
  if (kind === 'job') rows.push({ id: 'open-job', label: '打开结果', icon: 'results' }, { id: 'reuse-job', label: '载入该计算的模型', icon: 'open' }, { id: 'export-result', label: '导出结果 ZIP', icon: 'export' });
  if (kind === 'study' || kind === 'engine') rows.push({ id: 'run', label: '计算', icon: 'compute' });
  if (kind === 'geometry') rows.push(...Object.entries(KIND_LABELS).map(([k, v]) => ({ id: `kind:${k}`, label: `改为${v}`, icon: 'geometry' })));
  const removable = ['source', 'boundary', 'receiver', 'fixed', 'defect'].includes(kind) && !(kind === 'source' && id === 'source:0') && !(kind === 'receiver' && state.scene.receivers.length < 2);
  if (removable) rows.push({ sep: true }, { id: 'remove', label: '删除', icon: 'remove' });
  if (!rows.length) return;
  popupMenu(x, y, rows, cmd => cmd === 'remove' ? removeItem(id) : cmd === 'open-job' ? select(id) : cmd === 'reuse-job' ? reuseJob(id.slice(4)) : command(cmd));
}
function nodeLabel(id) {
  const [kind, raw] = id.split(':'), i = Number(raw), s = state.scene;
  if (kind === 'source') return sourcesOf(s)[i]?.name || `激励 ${i + 1}`;
  if (kind === 'boundary') return BOUNDARY_LABELS[s.boundaries[i]?.type] || '边界条件';
  if (kind === 'receiver') return s.receivers[i]?.name || '接收点';
  if (kind === 'defect') return s.defects?.[i] ? defectName(s.defects[i], i) : '缺陷';
  if (kind === 'job') return state.jobs.find(j => j.job_id === raw)?.name || '计算记录';
  if (kind === 'scene') return s.name;
  return NODE_TITLES[kind] || '设置';
}
function renderInspector() {
  const kind = state.selection.split(':')[0];
  $('#inspector-title').textContent = NODE_TITLES[kind] || '设置';
  const index = Number(state.selection.split(':')[1]);
  const iconName = kind === 'boundary' ? state.scene.boundaries[index]?.type || 'boundary' : kind === 'defect' ? DEFECT_ICONS[state.scene.defects?.[index]?.type] : NODE_ICONS[kind];
  $('#inspector-head').innerHTML = `${icon(iconName)}<b>${escapeHtml(nodeLabel(state.selection))}</b>`;
  if (kind === 'job') { renderJobInspector(state.selection.slice(4)); return; }
  $('#inspector').innerHTML = inspectorMarkup(state.scene, state.selection, 4, { materials: state.materials, engine: state.engine, coarseModel: state.coarseModel, meshAuto: state.preview?.mesh?.auto, record: state.record, timeStepping: state.timeStepping, controls: state.controls });
  renderMaterialDerived(); renderEngineDerived();
  let before = snapshot();
  bindInspector($('#inspector'), state.scene, state, kind => {
    if (!kind?.error && !kind?.remove) { state.history.push(before); if (state.history.length > 60) state.history.shift(); before = snapshot(); }
    changed(kind);
  });
}
function renderJobInspector(id) {
  const j = state.jobs.find(x => x.job_id === id); if (!j) { $('#inspector').innerHTML = ''; return; }
  const progress = j.duration_us ? Math.round(100 * (j.completed_time_us || 0) / j.duration_us) : null;
  $('#inspector').innerHTML = `<section class="wb-section"><h3>计算记录</h3><table class="wb-table">
      <tr><th>名称</th><td>${escapeHtml(j.name || id)}</td></tr><tr><th>状态</th><td><span class="wb-phase ${j.phase}">${PHASES[j.phase] || escapeHtml(j.phase)}</span>${progress !== null && j.phase !== 'completed' ? ` ${progress}%` : ''}</td></tr>
      <tr><th>开始</th><td>${new Date(j.started_at_utc).toLocaleString()}</td></tr><tr><th>编号</th><td><code>${escapeHtml(id)}</code></td></tr></table>
      <div class="st-actions"><button type="button" class="wb-button" data-job-act="open">${icon('results')}打开结果</button><button type="button" class="wb-button" data-job-act="reuse">${icon('open')}载入模型</button>
      ${j.phase === 'completed' ? `<a class="wb-button" href="/api/jobs/${id}/export">${icon('export')}导出 ZIP</a>` : ''}</div></section>`;
  $('#inspector').querySelector('[data-job-act="open"]').onclick = () => { setTab('result'); openJob(id); };
  $('#inspector').querySelector('[data-job-act="reuse"]').onclick = () => reuseJob(id);
}
function renderEngineDerived() {
  const box = $('#engine-derived'), e = state.preview?.engine; if (!box) return;
  if (state.engine !== 'coarse') { box.innerHTML = `<small>冻结网络：求导滤波器与时间层等价于 GLL 谱元，精度由节点间距决定；支持全部几何、材料和长记录。</small>${estimateMarkup(state.preview?.estimate)}`; return; }
  if (!e) { box.innerHTML = '<small>正在读取模型信息…</small>'; return; }
  box.innerHTML = `<table class="wb-table"><tr><th>模型</th><td>${escapeHtml(e.name)}（${e.degree} 阶，h ${e.h_mm} mm，${e.parameters} 个参数）</td></tr>
    <tr><th>留出测试</th><td>全部门槛通过 ${e.test_passed}/${e.test_cases}（同网格 GLL 基线 ${e.baseline_test_passed}/${e.test_cases}）；最差波形差 ${(100 * e.test_worst_waveform_l2).toFixed(2)}%，最差增量 2f ${(100 * e.test_worst_increment_a2).toFixed(1)}%</td></tr>
    ${(e.test_by_excitation || []).length ? `<tr><th>按激励形式</th><td>${e.test_by_excitation.map(r => `<span class="wb-res ${r.passed === r.cases ? 'ok' : r.passed ? 'warn' : 'bad'}" title="最差波形差 ${(100 * r.worst_waveform_l2).toFixed(2)}%，最差 2f ${(100 * r.worst_a2).toFixed(1)}%，最差增量 2f ${(100 * r.worst_increment_a2).toFixed(1)}%；GLL 基线 ${r.baseline_passed}/${r.cases}">${escapeHtml(r.label)} ${r.passed}/${r.cases}</span>`).join(' ')}</td></tr>` : ''}
    <tr><th>训练范围</th><td>${e.out_of_training_range.length ? `<span class="wb-res warn">超出 ${e.out_of_training_range.length} 项：${escapeHtml(e.out_of_training_range.join('；'))}</span>` : '<span class="wb-res ok">在训练范围内</span>'}</td></tr></table>`;
}
function renderMaterialDerived() { const box = $('#material-derived'); if (box) box.innerHTML = materialDerivedMarkup(state.preview?.material); }
function drawView() {
  if (state.tab !== 'model' || !state.scene) return;
  try { state.picks = drawScene($('#view'), state.scene, { ...state.view, selected: state.selection, gridMm: state.preview?.mesh?.grid_mm }); }
  catch (error) { console.error(error); }
}
function renderSignals() { $('#signals').innerHTML = signalsMarkup(state.preview, $('#signals').clientWidth || 900); }
function renderTitle() {
  $('#scene-title').textContent = state.scene.name; $('#rb-doc').textContent = state.scene.name;
  document.title = `${state.scene.name} — WorkAI ${state.info?.app?.version || ''}`.trim();
}
function renderStatus() {
  const p = state.preview, rows = previewMessages(p);
  renderMessages($('#dock-messages'), rows);
  const issues = rows.filter(r => r.kind === 'error' || r.kind === 'warn').length, badge = $('#msg-badge');
  badge.hidden = !issues; badge.textContent = issues; badge.className = `dock-badge${rows.some(r => r.kind === 'error') ? ' bad' : ''}`;
  const parts = [];
  if (p?.mesh) parts.push(`${p.mesh.nodes.toLocaleString()} 节点 · ${p.mesh.elements.toLocaleString()} 单元 · p ${p.mesh.degree}`);
  if (p?.record) parts.push(`${p.record.samples.toLocaleString()} 采样点 · ${p.record.mode === 'compact' ? '精简记录 ' : ''}${fmtBytes(p.record.uncompressed_bytes)}`);
  if (p?.estimate) parts.push(`预计 ${fmtDuration(p.estimate.seconds)} · 内存 ${fmtBytes(p.estimate.memory_bytes)}`);
  if (p?.resolution?.length) parts.push(`2f 分辨率 ${Math.min(...p.resolution.map(r => r.nodes_per_2f_shear_wavelength)).toFixed(1)}`);
  $('#sb-model').textContent = parts.join('　|　');
  renderStateText();
  if (p) {
    const key = JSON.stringify([p.ok, p.errors, p.mesh?.nodes]);
    if (key !== state.lastPreviewKey) { state.lastPreviewKey = key; log?.add(p.ok ? 'ok' : 'error', p.ok ? `模型检查通过${p.mesh ? `：${p.mesh.nodes.toLocaleString()} 个节点` : ''}。` : `模型有 ${p.errors.length} 个问题：${p.errors.join('；')}`); }
  }
  ribbon?.refresh();
}
function renderStateText() {
  const r = running(), el = $('#sb-state');
  if (r) { const pct = r.duration_us ? Math.round(100 * (r.completed_time_us || 0) / r.duration_us) : 0; el.innerHTML = `${icon('jobRunning')}${PHASES[r.phase] || r.phase} ${pct}% · ${escapeHtml(r.name || '')}`; el.className = 'sb-state busy'; return; }
  if (!state.preview) { el.innerHTML = '正在检查模型…'; el.className = 'sb-state'; return; }
  el.innerHTML = state.preview.ok ? `${icon('check')}就绪` : `${icon('error')}模型有 ${state.preview.errors.length} 个问题`; el.className = `sb-state ${state.preview.ok ? 'ok' : 'bad'}`;
}
function renderAll() { renderTitle(); renderTree(); renderInspector(); drawView(); renderStatus(); ribbon?.refresh(); }
function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll('[data-tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  for (const t of ['model', 'signal', 'result']) $(`#tab-${t}`).hidden = t !== tab;
  if (tab === 'model') requestAnimationFrame(drawView);
  if (tab === 'signal') renderSignals();
  ribbon?.refresh();
}
function setDock(name) {
  document.querySelectorAll('[data-dock]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.dock === name)));
  for (const n of ['messages', 'progress', 'log']) $(`#dock-${n}`).hidden = n !== name;
}

// ---------- preview ----------
function schedulePreview() {
  clearTimeout(previewTimer); state.preview = null; renderStatus();
  previewTimer = setTimeout(async () => {
    const stamp = ++previewStamp;
    try {
      const result = await api('/api/preview', requestBody());
      if (stamp !== previewStamp) return;
      state.preview = result; renderStatus(); drawView(); renderMaterialDerived(); renderEngineDerived(); if (state.tab === 'signal') renderSignals();
      renderTree(); if (state.selection === 'mesh') renderInspector();
    } catch (error) { if (stamp === previewStamp) { state.preview = { ok: false, errors: [error.message], signals: [] }; renderStatus(); } }
  }, 320);
}

// ---------- jobs and results ----------
async function refreshJobs() {
  state.jobs = await api('/api/jobs');
  for (const j of state.jobs) {
    const before = state.phases[j.job_id];
    if (before && before !== j.phase) log.add(j.phase === 'completed' ? 'ok' : j.phase === 'failed' ? 'error' : j.phase === 'paused' ? 'warn' : 'info', `${j.name || j.job_id}：${PHASES[j.phase] || j.phase}${j.message ? `（${j.message}）` : ''}`);
    state.phases[j.job_id] = j.phase;
  }
  if (!validSelection(state.selection)) state.selection = 'results';
  renderTree(); renderProgress(); renderStateText(); ribbon?.refresh();
  if (state.selection.startsWith('job:')) renderInspector();   // keep the record's status current
}
function renderProgress() {
  const j = running() || state.jobs.find(x => x.job_id === state.job) || state.jobs[0], box = $('#dock-progress');
  if (!j) { box.innerHTML = '<p class="dock-empty">还没有计算。设置好模型后点击功能区的“计算”（F8）。</p>'; return; }
  const pct = j.phase === 'completed' ? 100 : j.duration_us ? 100 * (j.completed_time_us || 0) / j.duration_us : 0;
  box.innerHTML = `<div class="pg"><div class="pg-head">${icon(j.phase === 'completed' ? 'job' : j.phase === 'failed' ? 'jobBad' : j.phase === 'paused' ? 'jobPaused' : 'jobRunning')}<b>${escapeHtml(j.name || j.job_id)}</b>
      <span class="wb-phase ${j.phase}">${PHASES[j.phase] || escapeHtml(j.phase)}</span><span class="muted">${new Date(j.started_at_utc).toLocaleString()}</span></div>
      <div class="pg-bar"><span style="width:${pct.toFixed(1)}%"></span></div><div class="pg-foot"><span>${pct.toFixed(0)}%${j.completed_time_us !== undefined && j.duration_us ? ` · 已算到 ${(+j.completed_time_us).toFixed(2)} / ${j.duration_us} μs` : ''}${j.remaining_s > 0 && !['completed', 'failed', 'paused'].includes(j.phase) ? ` · 预计还需 ${fmtDuration(j.remaining_s)}` : ''}</span>
      <span>${j.message ? escapeHtml(j.message) : ''}</span></div></div>`;
}
function clearResults() { viewer?.(); pathViewer?.cleanup(); viewer = null; pathViewer = null; $('#results').replaceChildren(); }
async function reuseJob(id) {
  try { const d = await api('/api/jobs/' + id + '/request'); takeEngine(d); loadScene(d.scene); setTab('model'); toast('已载入该计算的模型。', 'info'); } catch (e) { fail(e); }
}
async function openJob(id, restore = false) {
  clearTimeout(poll); const stamp = ++generation, oldView = restore ? viewer?.snapshot() : null, oldPath = restore ? pathViewer?.snapshot() : null;
  clearResults(); state.job = id; renderProgress(); ribbon?.refresh();
  if (!id) { $('#result-head').innerHTML = '<p class="wb-empty">在模型开发器的“结果”下选择一条计算记录，或点击功能区的“计算”计算当前模型。</p>'; return; }
  try {
    const s = await api('/api/jobs/' + id); if (stamp !== generation) return;
    const done = ['completed', 'failed', 'paused'].includes(s.phase), progress = s.duration_us ? 100 * (s.completed_time_us || 0) / s.duration_us : 100 * (s.controls_completed || 0) / (s.controls?.length || 3);
    // A job computes the controls its request asked for: offer only the responses it has, and fall back to the complete one.
    const parts = Array.isArray(s.parts) && s.parts.length ? s.parts : ['full', 'linear', 'stvk', 'geometric', 'material'];
    if (!parts.includes(state.part)) state.part = 'full';
    $('#result-head').innerHTML = `<div class="wb-result-title"><h2>${escapeHtml(s.name || id)}</h2><span class="wb-phase ${s.phase}">${PHASES[s.phase] || escapeHtml(s.phase)}</span>${s.engine === 'coarse' ? `<span class="wb-tag">粗网格算子${s.coarse_model ? ' · ' + escapeHtml(s.coarse_model) : ''}</span>` : ''}
      ${s.record_mode === 'compact' ? '<span class="wb-tag">精简记录</span>' : ''}<span class="muted">${Math.round(s.elapsed_s || 0)} 秒${!done && s.remaining_s > 0 ? ` · 预计还需 ${fmtDuration(s.remaining_s)}` : ''}${s.message ? ' · ' + escapeHtml(s.message) : ''}</span></div>
      ${s.time_plan?.scheme ? `<p class="muted small-text">时间层：${s.time_plan.scheme === 'verlet2' ? '二阶单级' : '三级四阶'}，Δt ${(+s.time_plan.layer_step_us * 1000).toFixed(2)} ns；每组控制 ${(+s.time_plan.evaluations_per_control).toLocaleString()} 次单元消息${s.time_plan.frozen_evaluations_per_control > s.time_plan.evaluations_per_control ? `（训练时的时间层需 ${(+s.time_plan.frozen_evaluations_per_control).toLocaleString()} 次）` : ''}。</p>` : ''}
      ${done ? '' : `<progress max="100" value="${progress.toFixed(1)}"></progress>`}
      <div class="wb-result-tools">${s.phase === 'completed' ? `<label class="wb-field inline"><span>响应</span><select id="part">${parts.map(k => `<option value="${k}" ${k === state.part ? 'selected' : ''}>${responseLabels[k] || k}</option>`).join('')}</select></label>
        <a class="wb-button" href="/api/jobs/${id}/export">${icon('export')}下载全部结果 ZIP</a>` : ''}
        <button type="button" class="wb-button" id="reuse">${icon('open')}载入该计算的模型</button>
        ${(s.resumable ?? s.duration_us >= 5000) && !done ? `<button type="button" class="wb-button" id="pause-job">${icon('pause')}保存后暂停</button>` : ''}${s.phase === 'paused' ? `<button type="button" class="wb-button primary" id="resume-job">${icon('resume')}继续计算</button>` : ''}</div>`;
    $('#part')?.addEventListener('change', ev => { state.part = ev.target.value; openJob(id, true); });
    $('#reuse').onclick = () => reuseJob(id);
    $('#pause-job')?.addEventListener('click', () => command('pause'));
    $('#resume-job')?.addEventListener('click', () => command('resume'));
    if (s.phase === 'completed') {
      const base = '/api/jobs/' + id, query = '?part=' + state.part, result = await api(base + '/result' + query), response = await fetch(base + '/field' + query);
      if (!response.ok) throw Error('位移场读取失败');
      const field = await decodeSpectralField(result, await response.arrayBuffer()); if (stamp !== generation) return;
      $('#results').innerHTML = solidResultMarkup(result) + solidPathMarkup(result.recording ? 8 : 64);
      const extract = result.recording ? recordExtractor(base, result) : spectralPointSeries;
      viewer = mountSolidResult($('#results'), result, field, extract, oldView); pathViewer = mountSolidPath($('#results'), result, field, extract, chart, oldPath);
    } else if (!done) poll = setTimeout(async () => { try { await refreshJobs(); if (state.job === id) await openJob(id); } catch (e) { fail(e); } }, 2500);
    else await refreshJobs();
  } catch (error) { if (stamp === generation) fail(error); }
}
function recordExtractor(base, result) {
  const cache = new Map(), pending = new Map();
  const batch = async (points, direction) => {
    if (points.length > 8) throw Error('长记录每批最多 8 个点。');
    const missing = points.filter(p => !cache.has(JSON.stringify(p)));
    if (missing.length) {
      const key = JSON.stringify(missing); let request = pending.get(key);
      if (!request) { request = api(base + '/traces', { part: result.response_part, points: missing }); pending.set(key, request); }
      let data; try { data = await request; } finally { pending.delete(key); }
      if (data.recording_fingerprint !== result.recording.fingerprint) throw Error('完整记录与当前结果不一致');
      data.points.forEach((row, i) => cache.set(JSON.stringify(missing[i]), { ...row, times_us: data.times_us }));
    }
    const d = unitVector(direction), rows = points.map(p => { const row = cache.get(JSON.stringify(p)); return { ...row, direction: d, projected_m: row.xyz_m.map(v => v.reduce((sum, x, i) => sum + x * d[i], 0)) }; });
    while (cache.size > 8) cache.delete(cache.keys().next().value);
    return rows;
  };
  const extract = async (_r, _f, point, direction) => (await batch([point], direction))[0]; extract.batch = batch; return extract;
}
async function run() {
  state.submitting = true; ribbon.refresh();
  try {
    const d = await api('/api/jobs', requestBody());
    state.job = d.job_id; log.add('info', `已提交计算：${state.scene.name}（${ENGINE_LABELS[state.engine]}）。首次运行需要编译，请稍候。`);
    await refreshJobs(); state.expanded.add('results'); state.selection = `job:${d.job_id}`; renderTree(); renderInspector();
    setTab('result'); setDock('progress'); await openJob(d.job_id);
  } catch (error) { fail(error); } finally { state.submitting = false; await refreshJobs().catch(fail); }
}
function selectedJob() { const raw = state.selection.startsWith('job:') ? state.selection.slice(4) : state.job; return state.jobs.find(j => j.job_id === raw); }

// ---------- commands (ribbon, menus, keys) ----------
function download(name, data) {
  const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function setView(name) {
  const [yaw, pitch] = VIEWS[name]; Object.assign(state.view, { yaw, pitch, zoom: 1, panX: 0, panY: 0, preset: name });
  if (state.tab !== 'model') setTab('model'); drawView(); renderGraphicsTools(); ribbon.refresh();
}
async function command(id) {
  const [verb, arg] = id.split(':');
  try {
    switch (verb) {
      case 'template': { const t = state.templates[arg]; if (t) { loadScene(structuredClone(t.scene)); setTab('model'); log.add('info', `从模板新建：${t.label}`); } break; }
      case 'new': ribbon.showTab('home'); document.querySelector('[data-cmd="new"]')?.click(); break;
      case 'open': $('#import-scene').click(); break;
      case 'save': download(`${(state.scene.name || 'workai-model').replace(/[\\/:*?"<>|]/g, '_')}.json`, requestBody()); log.add('info', '已保存模型文件。'); break;
      case 'undo': undo(); break;
      case 'sel': select(arg); if (['geometry', 'material', 'mesh', 'study', 'engine', 'sources', 'receivers', 'results'].includes(arg)) setTab(arg === 'results' ? 'result' : 'model'); break;
      case 'add': if (arg === 'defect') await addDefectOf('crack'); else addItem(arg); break;
      case 'add-defect': await addDefectOf(arg); break;
      case 'focus-command': $('#command-input').focus(); break;
      case 'add-boundary': addItem('boundary', arg); break;
      case 'kind': setKind(arg); break;
      case 'run': if (!disabled('run')) await run(); break;
      case 'pause': { const j = running(); if (j) { await api(`/api/jobs/${j.job_id}/pause`, {}); log.add('warn', '当前分块保存后暂停。'); } break; }
      case 'resume': { const j = selectedJob(); if (j?.phase === 'paused') { await api(`/api/jobs/${j.job_id}/resume`, {}); log.add('info', `继续计算：${j.name}`); await refreshJobs(); await openJob(j.job_id); setDock('progress'); } break; }
      case 'tab': setTab(arg); if (arg === 'result' && !state.job && state.jobs[0]) openJob(state.jobs[0].job_id); break;
      case 'export-result': { const j = selectedJob(); if (j?.phase === 'completed') { const a = document.createElement('a'); a.href = `/api/jobs/${j.job_id}/export`; a.click(); log.add('info', `导出结果：${j.name}`); } break; }
      case 'refresh': await refreshJobs(); if (state.job) await openJob(state.job); break;
      case 'view': setView(arg); break;
      case 'zoom-fit': Object.assign(state.view, { zoom: 1, panX: 0, panY: 0 }); drawView(); break;
      case 'toggle':
        if (arg === 'grid') { state.view.grid = !state.view.grid; drawView(); }
        if (arg === 'thin') { state.view.exaggerate = !state.view.exaggerate; drawView(); }
        if (arg === 'dock') $('#desk').classList.toggle('no-dock');
        renderGraphicsTools(); ribbon.refresh(); window.dispatchEvent(new Event('resize')); break;
      case 'screenshot': if (state.tab !== 'model') setTab('model'); $('#view').toBlob(b => b && download(`${state.scene.name || 'workai'}-图形.png`, b)); break;
      case 'help': await showHelp(); break;
      case 'about': await showAbout(); break;
      case 'exit':
        if (host) host.postMessage({ type: 'exit' });
        else toast('浏览器模式：关闭此页面，并在启动窗口按 Ctrl+C 结束程序。', 'info');
        break;
      default: break;
    }
  } catch (error) { fail(error); }
}
function disabled(id) {
  const [verb, arg] = id.split(':');
  if (verb === 'undo') return !state.history.length;
  if (verb === 'run') return state.submitting || Boolean(running()) || !state.preview?.ok;
  if (verb === 'pause') { const j = running(); return !j || !(j.resumable ?? j.duration_us >= 5000); }
  if (verb === 'resume') return selectedJob()?.phase !== 'paused' || Boolean(running());
  if (verb === 'export-result') return selectedJob()?.phase !== 'completed';
  if (verb === 'add-boundary' && arg === 'fixed') return state.scene?.boundary.fixed_face !== 'none';
  if (verb === 'add-defect' || id === 'add:defect') return state.scene?.geometry.kind !== 'plate';
  return false;
}
function active(id) {
  const [verb, arg] = id.split(':');
  if (verb === 'view') return state.view.preset === arg;
  if (verb === 'toggle') return arg === 'grid' ? state.view.grid : arg === 'thin' ? state.view.exaggerate : arg === 'dock' ? !$('#desk').classList.contains('no-dock') : false;
  if (verb === 'kind') return state.scene?.geometry.kind === arg;
  if (verb === 'tab') return state.tab === arg;
  return false;
}
function renderGraphicsTools() {
  const b = (cmd, ic, tip) => `<button type="button" class="gw-tool${active(cmd) ? ' on' : ''}" data-cmd="${cmd}" title="${tip}">${icon(ic)}</button>`;
  $('#gw-tools').innerHTML = b('zoom-fit', 'zoomFit', '缩放到窗口（双击图形也可复位）') + '<span class="gw-sep"></span>' + b('view:iso', 'viewIso', '等轴视图') + b('view:front', 'viewFront', '正视 XZ') + b('view:top', 'viewTop', '俯视 XY') + b('view:side', 'viewSide', '侧视 YZ') +
    '<span class="gw-sep"></span>' + b('toggle:grid', 'grid', '显示单元网格') + b('toggle:thin', 'thin', '放大薄方向') + '<span class="gw-sep"></span>' + b('screenshot', 'camera', '保存图形截图（PNG）') +
    '<span class="gw-hint">拖动旋转 · 滚轮缩放 · Shift+拖动平移 · 点击选中</span>';
  $('#gw-tools').querySelectorAll('[data-cmd]').forEach(x => x.addEventListener('click', () => command(x.dataset.cmd)));
}

// ---------- dialogs ----------
async function showAbout() {
  const i = state.info || {}, m = i.model || {};
  await dialog({ title: '关于 WorkAI', body: `<div class="about">${icon('logo')}<div><h2>WorkAI 超声导波仿真</h2><p class="ver">版本 ${escapeHtml(i.app?.version || '')}${i.app?.build ? ` · 构建 ${escapeHtml(i.app.build)}` : ''}</p>
      <p>非线性超声导波物理 AI 仿真软件。冻结图网络等价于 GLL 谱元离散，逐时前向计算线性、几何非线性（StVK）与材料非线性（Murnaghan / 自定义应变能）三组响应。</p>
      <table class="wb-table"><tr><th>物理 AI 模型</th><td>${escapeHtml(m.version || '')}</td></tr><tr><th>运行方式</th><td>本机离线，不需要联网</td></tr><tr><th>数据目录</th><td><code>${escapeHtml(i.data_directory || '')}</code></td></tr></table>
      <p class="muted">核验报告（文献对照、1 ms 精度、缺口 / 通孔 / 管的散射）见源码仓库 research/ 目录。</p></div></div>` });
}
async function showHelp() {
  await dialog({ title: '使用说明', wide: true, body: `<ol class="help">
      <li><b>新建模型：</b>功能区“主屏幕 → 新建”选择模板（平板、圆管、带孔板、带缺口板、高斯激励、双贴片等），或“打开”已保存的模型文件（JSON）。</li>
      <li><b>设置模型：</b>在左侧“模型开发器”中选中节点，在“设置”窗口编辑：几何、材料、激励源、边界条件（对称 / 反对称 / 吸收层 / 固定面）、接收点和研究（记录时长、采样间隔、求解器）。右键节点可添加或删除。</li>
      <li><b>缺陷：</b>功能区“几何 → 一键添加缺陷”添加裂纹、孔、分层、接触界面或缺口；裂纹和分层可设为闭合（受压接触、受拉张开）。也可以在顶部文字框里描述，例如“在 100 mm 处添加深 0.5 mm 的裂纹”“在 60 到 80 mm 之间添加分层，距上表面 0.8 mm”，确认后修改模型。</li>
      <li><b>网格：</b>由软件按激励频率、记录时长和缺陷位置自动选择，“网格 1（自动）”节点显示所选尺寸和依据，不需要设置。模型大小只受本机内存、磁盘和时间限制，“消息”窗口给出预计耗时和内存；大模型自动保存精简记录（接收点完整波形 + 表面动画 + 快照），请先把关心的位置设为接收点。</li>
      <li><b>检查：</b>“消息”窗口实时列出模型检查结果、网格和记录大小、每个激励的 2f 分辨率，以及是否超出原验收范围。图形窗口显示几何、加载区、边界和接收点。</li>
      <li><b>计算：</b>点击“计算”（F8）。“进度”窗口显示计算进度；5 ms 及以上的长记录可“暂停”并在之后“继续”。首次计算需要几分钟编译。</li>
      <li><b>结果：</b>在“结果”下选择计算记录：位移场动画、任意点波形与频谱、多点提取、五种响应（完整、线性、StVK、几何增量、材料增量），可导出全部结果 ZIP。</li>
      <li><b>精度：</b>自动网格的 2f 分辨率目标为 10（1 ms 及以上为 15）；模型太大时会放粗并提示。新模型的结果属于研究结果，应用细网格参考逐例核对。</li></ol>
      <p class="muted">快捷键：Ctrl+O 打开 · Ctrl+S 保存 · Ctrl+Z 撤销 · F8 计算 · F1 帮助</p>` });
}

// ---------- view interaction ----------
function bindView() {
  const canvas = $('#view'); let drag = null;
  canvas.addEventListener('pointerdown', ev => { drag = { x: ev.clientX, y: ev.clientY, moved: false, pan: ev.shiftKey || ev.button === 1 }; canvas.setPointerCapture(ev.pointerId); });
  canvas.addEventListener('pointermove', ev => {
    if (!drag) return; const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.pan) { state.view.panX = (state.view.panX || 0) + dx; state.view.panY = (state.view.panY || 0) + dy; }
    else { state.view.yaw -= dx * .008; state.view.pitch = Math.max(-1.55, Math.min(1.5699, state.view.pitch + dy * .008)); if (state.view.preset) { state.view.preset = ''; renderGraphicsTools(); ribbon.refresh(); } }
    drag.x = ev.clientX; drag.y = ev.clientY; drawView();
  });
  canvas.addEventListener('pointerup', ev => {
    if (drag && !drag.moved) { const r = canvas.getBoundingClientRect(), hit = pick(state.picks, ev.clientX - r.left, ev.clientY - r.top); if (hit) select(hit); }
    drag = null;
  });
  canvas.addEventListener('wheel', ev => { ev.preventDefault(); state.view.zoom = Math.max(.3, Math.min(8, state.view.zoom * (ev.deltaY < 0 ? 1.12 : 1 / 1.12))); drawView(); }, { passive: false });
  canvas.addEventListener('dblclick', () => setView('iso'));
  new ResizeObserver(() => drawView()).observe(canvas);
  let lastWidth = 0; new ResizeObserver(() => { const w = $('#signals').clientWidth; if (state.tab === 'signal' && Math.abs(w - lastWidth) > 40) { lastWidth = w; renderSignals(); } }).observe($('#signals'));
}

// ---------- start ----------
async function start() {
  log = createLog($('#dock-log'));
  ribbon = mountRibbon($('#ribbon'), { command, disabled, active,
    menu: kind => kind === 'boundary' ? boundaryMenu() : kind === 'defect' ? defectMenu() : Object.entries(state.templates).map(([k, t]) => ({ id: k, label: t.label, icon: 'new' })) });
  $('#command-run').addEventListener('click', interpretText);
  $('#command-input').addEventListener('keydown', ev => { if (ev.key === 'Enter') { ev.preventDefault(); interpretText(); } });
  renderGraphicsTools();
  document.querySelectorAll('[data-tab]').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab)));
  document.querySelectorAll('[data-dock]').forEach(b => b.addEventListener('click', () => setDock(b.dataset.dock)));
  $('#inspector').addEventListener('click', ev => { const h = ev.target.closest('.wb-section > h3'); if (h) h.parentElement.classList.toggle('collapsed'); });
  $('#tree-expand').addEventListener('click', () => { const walk = n => { state.expanded.add(n.id); (n.children || []).forEach(walk); }; walk(treeNodes(state.scene, { jobs: state.jobs })); renderTree(); });
  $('#tree-collapse').addEventListener('click', () => { state.expanded = new Set(['scene']); renderTree(); });
  document.addEventListener('keydown', ev => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName);
    if ((ev.ctrlKey || ev.metaKey) && ev.key === 'z' && !typing) { ev.preventDefault(); undo(); }
    else if ((ev.ctrlKey || ev.metaKey) && ev.key === 's') { ev.preventDefault(); command('save'); }
    else if ((ev.ctrlKey || ev.metaKey) && ev.key === 'o') { ev.preventDefault(); command('open'); }
    else if (ev.key === 'F8') { ev.preventDefault(); command('run'); }
    else if (ev.key === 'F1') { ev.preventDefault(); command('help'); }
  });
  $('#import-scene').addEventListener('change', async ev => {
    try {
      const file = ev.target.files[0]; if (!file) return; if (file.size > 65536) throw Error('模型文件不能超过 64 KiB。');
      const d = JSON.parse(await file.text()), scene = d.scene || d; if (!scene?.geometry || !scene.source) throw Error('模型文件格式不正确：需要 {scene, degree} 或场景对象。');
      takeEngine(d); loadScene(scene); toast(`已打开模型：${file.name}`, 'info');
    } catch (error) { fail(error); } finally { ev.target.value = ''; }
  });
  initSplitters($('#desk'), [
    { handle: $('#split-left'), variable: '--mb-w', axis: 'x', min: 200, max: 560 },
    { handle: $('#split-mid'), variable: '--st-w', axis: 'x', min: 260, max: 680 },
    { handle: $('#split-dock'), variable: '--dock-h', axis: 'y', min: 80, max: 520, invert: true },
  ]);
  bindView();
  try {
    const [info, templates, materials] = await Promise.all([api('/api/info'), api('/api/templates'), api('/api/materials')]);
    state.info = info; state.token = info.token; state.templates = workbenchTemplates(templates); state.materials = materials;
    if (materials.cfrp_ud && state.templates.gaussian) {
      const c = structuredClone(state.templates.gaussian.scene); c.name = 'CFRP 单向板 · 顶面高斯激励 · 两端吸收 · 半模型';
      c.material = structuredClone(materials.cfrp_ud.material); c.source.frequency_mhz = .3;
      state.templates.cfrp = { label: 'CFRP 单向板 · 顶面高斯激励（自定义应变能）', scene: c };
    }
    $('#sb-app').textContent = `WorkAI ${info.app?.version || ''} · 物理 AI ${info.model.version || ''} · 本机离线`;
    $('#sb-app').title = `数据目录：${info.data_directory}`;
    const draft = readDraft();
    if (draft) { state.scene = draft.scene; takeEngine(draft); } else { state.scene = structuredClone(state.templates.gaussian?.scene || Object.values(state.templates)[0].scene); }
    log.add('info', `WorkAI ${info.app?.version || ''} 已启动。数据目录：${info.data_directory}`);
    renderAll(); schedulePreview(); setDock('messages');
    await refreshJobs(); await openJob(state.jobs[0]?.job_id || '');
    setInterval(() => { if (running()) refreshJobs().catch(() => {}); }, 4000);
    document.body.dataset.ready = '1';
  } catch (error) { fail(error); document.body.dataset.ready = 'error'; }
}
start();
