// WorkAI workbench: model tree, 3-D view, property inspector, signal preview, local physics-AI jobs and results.
import { unitVector } from './solid-data.js';
import { decodeSpectralField, spectralPointSeries } from './spectral-data.js';
import { solidResultMarkup, mountSolidResult, chart, responseLabels } from './solid-viewer.js';
import { solidPathMarkup, mountSolidPath } from './solid-path-viewer.js';
import { tree, upgrade, workbenchTemplates, addSource, removeSource, addBoundary, removeBoundary, addReceiver, removeReceiver, V2, BOUNDARY_LABELS } from './scene-model.js';
import { drawScene, pick } from './model-view.js';
import { inspectorMarkup, bindInspector, escapeHtml } from './inspector.js';
import { signalsMarkup } from './signal-view.js';
import { materialDerivedMarkup } from './material-form.js';

const $ = s => document.querySelector(s);
const PHASES = { starting: '准备中', meshing: '生成连接图', running: '计算中', exporting: '保存结果', completed: '已完成', failed: '未完成', paused: '已暂停' };
const VIEWS = { iso: [-.62, .52], front: [0, 0], top: [0, 1.5699], side: [-Math.PI / 2, 0] };
const DRAFT = 'workai.workbench.draft.v1';
const state = { scene: null, degree: 4, engine: 'network', selection: 'geometry', preview: null, jobs: [], job: '', part: 'full', tab: 'model',
  view: { yaw: VIEWS.iso[0], pitch: VIEWS.iso[1], zoom: 1, exaggerate: true, grid: true }, history: [], templates: {}, token: '', picks: [], submitting: false };
let previewTimer = 0, previewStamp = 0, poll = 0, generation = 0, viewer = null, pathViewer = null;

async function api(url, body) {
  const response = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-WorkAI-Token': state.token }, body: JSON.stringify(body) } : undefined);
  const data = await response.json(); if (!response.ok) throw Error(data.error || '请求失败'); return data;
}
function toast(message, kind = 'error') {
  const el = $('#toast'); el.textContent = message; el.className = `wb-toast ${kind}`; el.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { el.hidden = true; }, kind === 'error' ? 7000 : 3500);
}
const fail = error => toast(error?.message || String(error));
function requestBody() { return { scene: state.scene, degree: state.degree, ...(state.engine === 'coarse' ? { engine: 'coarse' } : {}) }; }
function storeDraft() { try { localStorage.setItem(DRAFT, JSON.stringify(requestBody())); } catch { /* storage unavailable */ } }
function readDraft() { try { const d = JSON.parse(localStorage.getItem(DRAFT) || 'null'); return d?.scene?.schema === V2 ? d : null; } catch { return null; } }

// ---------- editing ----------
function snapshot() { return JSON.stringify(requestBody()); }
function remember() { state.history.push(snapshot()); if (state.history.length > 60) state.history.shift(); $('#undo').disabled = false; }
function undo() {
  const last = state.history.pop(); if (!last) return;
  const d = JSON.parse(last); state.scene = d.scene; state.degree = d.degree; state.engine = d.engine || 'network'; $('#undo').disabled = !state.history.length;
  if (!validSelection(state.selection)) state.selection = 'geometry';
  renderAll(); schedulePreview(); storeDraft();
}
function validSelection(id) {
  const [kind, raw] = id.split(':'), i = Number(raw), s = state.scene;
  if (kind === 'source') return i === 0 || i <= (s.sources || []).length;
  if (kind === 'boundary') return i < (s.boundaries || []).length;
  if (kind === 'receiver') return i < s.receivers.length;
  if (kind === 'fixed') return s.boundary.fixed_face !== 'none';
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
  $('#scene-title').textContent = state.scene.name;
  drawView(); schedulePreview(); storeDraft();
}
function removeItem(id) {
  const [kind, raw] = id.split(':'), i = Number(raw);
  try {
    remember();
    if (kind === 'source') removeSource(state.scene, i);
    if (kind === 'boundary') removeBoundary(state.scene, i);
    if (kind === 'receiver') removeReceiver(state.scene, i);
    if (!validSelection(state.selection)) state.selection = kind === 'source' ? 'source:0' : kind === 'receiver' ? 'receiver:0' : 'boundaries';
    renderAll(); schedulePreview(); storeDraft();
  } catch (error) { state.history.pop(); fail(error); }
}
function addItem(kind, type) {
  try {
    remember();
    if (kind === 'source') state.selection = `source:${addSource(state.scene, 'zmax')}`;
    if (kind === 'boundary') state.selection = `boundary:${addBoundary(state.scene, type)}`;
    if (kind === 'receiver') state.selection = `receiver:${addReceiver(state.scene)}`;
    renderAll(); schedulePreview(); storeDraft();
  } catch (error) { state.history.pop(); fail(error); }
}

// ---------- rendering ----------
function renderTree() {
  const rows = tree(state.scene);
  $('#tree').innerHTML = rows.map(r => {
    const active = r.id === state.selection ? ' active' : '';
    if (r.group === 'header') {
      const add = r.add === 'boundary'
        ? `<span class="wb-add-menu"><button type="button" class="wb-icon" data-add="boundary" title="添加边界条件" aria-label="添加边界条件">+</button><span class="wb-menu" hidden>${Object.entries(BOUNDARY_LABELS).map(([k, v]) => `<button type="button" data-add-boundary="${k}">${v}</button>`).join('')}</span></span>`
        : `<button type="button" class="wb-icon" data-add="${r.add}" title="添加" aria-label="添加${escapeHtml(r.label)}">+</button>`;
      return `<div class="wb-tree-head${active}" data-id="${r.id}"><span>${escapeHtml(r.label)}</span><span class="count">${escapeHtml(r.detail)}</span>${add}</div>`;
    }
    return `<div class="wb-tree-row ${r.group}${active}" data-id="${r.id}" tabindex="0" role="button"><span class="label">${escapeHtml(r.label)}</span><span class="detail">${escapeHtml(r.detail)}</span>${r.removable ? `<button type="button" class="wb-icon remove" data-remove="${r.id}" title="删除" aria-label="删除">×</button>` : ''}</div>`;
  }).join('');
  $('#tree').querySelectorAll('[data-id]').forEach(el => {
    const select = () => { state.selection = el.dataset.id; renderTree(); renderInspector(); drawView(); };
    el.addEventListener('click', ev => { if (ev.target.closest('button')) return; select(); });
    el.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(); } });
  });
  $('#tree').querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', () => removeItem(b.dataset.remove)));
  $('#tree').querySelectorAll('[data-add]').forEach(b => b.addEventListener('click', ev => {
    if (b.dataset.add === 'boundary') { const menu = b.nextElementSibling; menu.hidden = !menu.hidden; ev.stopPropagation(); return; }
    addItem(b.dataset.add);
  }));
  $('#tree').querySelectorAll('[data-add-boundary]').forEach(b => b.addEventListener('click', () => addItem('boundary', b.dataset.addBoundary)));
}
function renderInspector() {
  const titles = { geometry: '几何', material: '材料与本构', source: '激励源', boundary: '边界条件', receiver: '接收点', study: '记录与网格', fixed: '记录与网格', sources: '激励源', boundaries: '边界条件', receivers: '接收点' };
  $('#inspector-title').textContent = titles[state.selection.split(':')[0]] || '属性';
  $('#inspector').innerHTML = inspectorMarkup(state.scene, state.selection, state.degree, { materials: state.materials, engine: state.engine });
  renderMaterialDerived(); renderEngineDerived();
  let before = snapshot();
  bindInspector($('#inspector'), state.scene, state, kind => {
    if (!kind?.error && !kind?.remove) { state.history.push(before); if (state.history.length > 60) state.history.shift(); $('#undo').disabled = false; before = snapshot(); }
    changed(kind);
  });
}
function renderEngineDerived() {
  const box = $('#engine-derived'), e = state.preview?.engine; if (!box) return;
  if (state.engine !== 'coarse') { box.innerHTML = '<small>冻结网络：求导滤波器与时间层等价于 GLL 谱元，精度由节点间距决定；支持全部几何、材料和长记录。</small>'; return; }
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
function bytes(n) { return n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GiB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MiB` : `${Math.round(n / 1024)} KiB`; }
function renderStatus() {
  const p = state.preview, bar = $('#status-bar'), issues = $('#issues');
  if (!p) { bar.innerHTML = '<span class="wb-check pending">正在检查场景…</span>'; return; }
  const parts = [p.ok ? '<span class="wb-check ok">✓ 场景有效，可以运行</span>' : `<button type="button" class="wb-check bad" id="show-issues">✗ ${p.errors.length} 个问题</button>`];
  if (p.engine) parts.push(`<span class="wb-res ${p.engine.out_of_training_range.length ? 'warn' : 'ok'}" title="${escapeHtml(p.engine.out_of_training_range.join('；') || '在训练范围内')}">快速引擎 ${escapeHtml(p.engine.name)} · 测试 ${p.engine.test_passed}/${p.engine.test_cases}${p.engine.out_of_training_range.length ? ` · 超出训练范围 ${p.engine.out_of_training_range.length} 项` : ''}</span>`);
  if (p.mesh) parts.push(`<span>${p.engine ? '粗网格' : '网格'} ${p.mesh.elements.toLocaleString()} 单元 · ${p.mesh.nodes.toLocaleString()} 节点 · p=${p.mesh.degree}</span>`);
  if (p.record) parts.push(`<span title="三组控制的完整未压缩记录">记录 ${p.record.samples.toLocaleString()} 点 · ${bytes(p.record.uncompressed_bytes)}${p.record.free_bytes ? ` / 可用 ${bytes(p.record.free_bytes)}` : ''}</span>`);
  for (const r of p.resolution || []) {
    const n = r.nodes_per_2f_shear_wavelength, level = n >= 8 ? 'ok' : n >= 4 ? 'warn' : 'bad';
    parts.push(`<span class="wb-res ${level}" title="2f 剪切波长内的节点数（按 −30 dB 带宽上限）。小于 4：二次谐波分辨不足；4–8：短记录可用，长记录需复核；≥ 8：分辨充分。">2f 分辨率 · ${escapeHtml(r.name)} ${n.toFixed(1)}</span>`);
  }
  if (p.scope?.length) parts.push(`<button type="button" class="wb-scope" id="show-scope" title="${escapeHtml(p.scope.join('；'))}">超出原验收范围 ${p.scope.length} 项</button>`);
  else if (p.ok) parts.push('<span class="wb-scope ok">在原验收范围内</span>');
  for (const w of p.warnings || []) parts.push(`<span class="wb-res warn">${escapeHtml(w)}</span>`);
  bar.innerHTML = parts.join('');
  const show = (title, list, note) => { issues.innerHTML = `<header><b>${title}</b><button type="button" class="wb-icon" aria-label="关闭">×</button></header><ul>${list.map(e => `<li>${escapeHtml(e)}</li>`).join('')}</ul>${note ? `<p>${note}</p>` : ''}`; issues.hidden = false; issues.querySelector('button').onclick = () => { issues.hidden = true; }; };
  $('#show-issues')?.addEventListener('click', () => show('需要修改', p.errors));
  $('#show-scope')?.addEventListener('click', () => show('超出原八例验收范围', p.scope, '可以计算，但结果属于研究结果：原验收结论不能直接套用，需要用细网格参考（谱元求解器或 COMSOL）逐例核对。'));
  updateRun();
}
function renderAll() {
  $('#scene-title').textContent = state.scene.name; renderTree(); renderInspector(); drawView(); renderStatus(); $('#undo').disabled = !state.history.length;
}
function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll('[data-tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  for (const t of ['model', 'signal', 'result']) $(`#tab-${t}`).hidden = t !== tab;
  if (tab === 'model') requestAnimationFrame(drawView);
  if (tab === 'signal') renderSignals();
}

// ---------- preview ----------
function schedulePreview() {
  clearTimeout(previewTimer); state.preview = null; renderStatus(); $('#issues').hidden = true;
  previewTimer = setTimeout(async () => {
    const stamp = ++previewStamp;
    try {
      const result = await api('/api/preview', requestBody());
      if (stamp !== previewStamp) return;
      state.preview = result; renderStatus(); drawView(); renderMaterialDerived(); renderEngineDerived(); if (state.tab === 'signal') renderSignals();
    } catch (error) { if (stamp === previewStamp) { state.preview = { ok: false, errors: [error.message], signals: [] }; renderStatus(); } }
  }, 320);
}
function updateRun() {
  const busy = state.jobs.some(j => !['completed', 'failed', 'paused'].includes(j.phase));
  $('#run').disabled = state.submitting || busy || !state.preview?.ok;
  $('#run').title = busy ? '已有任务在计算' : state.preview?.ok ? '用物理 AI 网络计算这个场景' : '先修正场景中的问题';
}

// ---------- jobs and results ----------
async function refreshJobs() {
  state.jobs = await api('/api/jobs');
  $('#jobs').innerHTML = state.jobs.length ? state.jobs.map(j => {
    const progress = j.duration_us ? Math.round(100 * (j.completed_time_us || 0) / j.duration_us) : null;
    return `<button type="button" class="wb-job${j.job_id === state.job ? ' active' : ''}" data-job="${j.job_id}"><span class="name">${escapeHtml(j.name || j.job_id)}</span>
      <span class="wb-phase ${j.phase}">${PHASES[j.phase] || escapeHtml(j.phase)}${progress !== null && !['completed', 'failed'].includes(j.phase) ? ` ${progress}%` : ''}</span>
      <span class="when">${new Date(j.started_at_utc).toLocaleString()}</span></button>`;
  }).join('') : '<p class="wb-empty small">还没有计算记录。</p>';
  $('#jobs').querySelectorAll('[data-job]').forEach(b => b.addEventListener('click', () => { setTab('result'); openJob(b.dataset.job); }));
  updateRun();
}
function clearResults() { viewer?.(); pathViewer?.cleanup(); viewer = null; pathViewer = null; $('#results').replaceChildren(); }
async function openJob(id, restore = false) {
  clearTimeout(poll); const stamp = ++generation, oldView = restore ? viewer?.snapshot() : null, oldPath = restore ? pathViewer?.snapshot() : null;
  clearResults(); state.job = id; $('#jobs').querySelectorAll('[data-job]').forEach(b => b.classList.toggle('active', b.dataset.job === id));
  if (!id) { $('#result-head').innerHTML = '<p class="wb-empty">在左侧“计算记录”中选择一个任务，或点击“运行”计算当前场景。</p>'; return; }
  try {
    const s = await api('/api/jobs/' + id); if (stamp !== generation) return;
    const done = ['completed', 'failed', 'paused'].includes(s.phase), progress = s.duration_us ? 100 * (s.completed_time_us || 0) / s.duration_us : 100 * (s.controls_completed || 0) / 3;
    $('#result-head').innerHTML = `<div class="wb-result-title"><h2>${escapeHtml(s.name || id)}</h2><span class="wb-phase ${s.phase}">${PHASES[s.phase] || escapeHtml(s.phase)}</span>${s.engine === 'coarse' ? '<span class="wb-tag">粗网格算子</span>' : ''}
      <span class="muted">${Math.round(s.elapsed_s || 0)} 秒${s.message ? ' · ' + escapeHtml(s.message) : ''}</span></div>
      ${done ? '' : `<progress max="100" value="${progress.toFixed(1)}"></progress>`}
      <div class="wb-result-tools">${s.phase === 'completed' ? `<label class="wb-field inline"><span>响应</span><select id="part">${Object.entries(responseLabels).map(([k, v]) => `<option value="${k}" ${k === state.part ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
        <a class="wb-button" href="/api/jobs/${id}/export">下载全部结果 ZIP</a>` : ''}
        <button type="button" class="wb-button" id="reuse">载入该任务的场景</button>
        ${s.duration_us >= 5000 && !done ? '<button type="button" class="wb-button" id="pause-job">保存后暂停</button>' : ''}${s.phase === 'paused' ? '<button type="button" class="wb-button primary" id="resume-job">继续计算</button>' : ''}</div>`;
    $('#part')?.addEventListener('change', ev => { state.part = ev.target.value; openJob(id, true); });
    $('#reuse').onclick = async () => { try { const d = await api('/api/jobs/' + id + '/request'); state.engine = d.engine === 'coarse' ? 'coarse' : 'network'; loadScene(d.scene, d.degree ?? 4); setTab('model'); toast('已载入该任务的场景。', 'info'); } catch (e) { fail(e); } };
    $('#pause-job')?.addEventListener('click', async () => { try { await api('/api/jobs/' + id + '/pause', {}); toast('当前分块保存后暂停。', 'info'); } catch (e) { fail(e); } });
    $('#resume-job')?.addEventListener('click', async () => { try { await api('/api/jobs/' + id + '/resume', {}); await refreshJobs(); await openJob(id); } catch (e) { fail(e); } });
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
  state.submitting = true; updateRun();
  try {
    const d = await api('/api/jobs', requestBody());
    state.job = d.job_id; await refreshJobs(); setTab('result'); await openJob(d.job_id); toast('已开始计算。首次运行需要编译，请稍候。', 'info');
  } catch (error) { fail(error); } finally { state.submitting = false; await refreshJobs().catch(fail); }
}

// ---------- view interaction ----------
function bindView() {
  const canvas = $('#view'); let drag = null;
  canvas.addEventListener('pointerdown', ev => { drag = { x: ev.clientX, y: ev.clientY, moved: false, pan: ev.shiftKey || ev.button === 1 }; canvas.setPointerCapture(ev.pointerId); });
  canvas.addEventListener('pointermove', ev => {
    if (!drag) return; const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.pan) { state.view.panX = (state.view.panX || 0) + dx; state.view.panY = (state.view.panY || 0) + dy; }
    else { state.view.yaw -= dx * .008; state.view.pitch = Math.max(-1.55, Math.min(1.5699, state.view.pitch + dy * .008)); }
    drag.x = ev.clientX; drag.y = ev.clientY; document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active')); drawView();
  });
  canvas.addEventListener('pointerup', ev => {
    if (drag && !drag.moved) { const r = canvas.getBoundingClientRect(), hit = pick(state.picks, ev.clientX - r.left, ev.clientY - r.top); if (hit) { state.selection = hit; renderTree(); renderInspector(); drawView(); } }
    drag = null;
  });
  canvas.addEventListener('wheel', ev => { ev.preventDefault(); state.view.zoom = Math.max(.3, Math.min(8, state.view.zoom * (ev.deltaY < 0 ? 1.12 : 1 / 1.12))); drawView(); }, { passive: false });
  canvas.addEventListener('dblclick', () => { Object.assign(state.view, { yaw: VIEWS.iso[0], pitch: VIEWS.iso[1], zoom: 1, panX: 0, panY: 0 }); drawView(); });
  document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => {
    const [yaw, pitch] = VIEWS[b.dataset.view]; Object.assign(state.view, { yaw, pitch, zoom: 1, panX: 0, panY: 0 });
    document.querySelectorAll('[data-view]').forEach(x => x.classList.toggle('active', x === b)); drawView();
  }));
  $('#show-grid').addEventListener('change', ev => { state.view.grid = ev.target.checked; drawView(); });
  $('#exaggerate').addEventListener('change', ev => { state.view.exaggerate = ev.target.checked; drawView(); });
  new ResizeObserver(() => drawView()).observe(canvas);
  let lastWidth = 0; new ResizeObserver(() => { const w = $('#signals').clientWidth; if (state.tab === 'signal' && Math.abs(w - lastWidth) > 40) { lastWidth = w; renderSignals(); } }).observe($('#signals'));
}

// ---------- start ----------
function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function start() {
  document.querySelectorAll('[data-tab]').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab)));
  document.addEventListener('click', ev => { if (!ev.target.closest('.wb-add-menu')) document.querySelectorAll('.wb-menu').forEach(m => { m.hidden = true; }); });
  document.addEventListener('keydown', ev => { if ((ev.ctrlKey || ev.metaKey) && ev.key === 'z' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) { ev.preventDefault(); undo(); } });
  $('#undo').addEventListener('click', undo);
  $('#run').addEventListener('click', run);
  $('#refresh').addEventListener('click', () => refreshJobs().then(() => state.job && openJob(state.job)).catch(fail));
  $('#export-scene').addEventListener('click', () => download('workai-scene.json', requestBody()));
  $('#import-scene').addEventListener('change', async ev => {
    try {
      const file = ev.target.files[0]; if (!file) return; if (file.size > 65536) throw Error('配置不能超过 64 KiB。');
      const d = JSON.parse(await file.text()), scene = d.scene || d; if (!scene?.geometry || !scene.source) throw Error('配置格式不正确：需要 {scene, degree} 或场景对象。');
      state.engine = d.engine === 'coarse' ? 'coarse' : 'network'; loadScene(scene, [3, 4, 5].includes(d.degree) ? d.degree : state.degree); toast('已导入场景。', 'info');
    } catch (error) { fail(error); } finally { ev.target.value = ''; }
  });
  bindView();
  try {
    const [info, templates, materials] = await Promise.all([api('/api/info'), api('/api/templates'), api('/api/materials')]);
    state.token = info.token; state.templates = workbenchTemplates(templates); state.materials = materials;
    if (materials.cfrp_ud && state.templates.gaussian) {
      const c = structuredClone(state.templates.gaussian.scene); c.name = 'CFRP 单向板 · 顶面高斯激励 · 两端吸收 · 半模型';
      c.material = structuredClone(materials.cfrp_ud.material); c.source.frequency_mhz = .3;
      state.templates.cfrp = { label: 'CFRP 单向板 · 顶面高斯激励（自定义应变能）', scene: c };
    }
    $('#engine-chip').textContent = `● 本机离线 · 物理 AI ${info.model.version || ''}`.trim(); $('#engine-chip').title = info.model.user_notice || '';
    $('#template').innerHTML = '<option value="">从模板新建…</option>' + Object.entries(state.templates).map(([k, t]) => `<option value="${k}">${escapeHtml(t.label)}</option>`).join('');
    $('#template').addEventListener('change', ev => { const t = state.templates[ev.target.value]; ev.target.value = ''; if (t) { loadScene(structuredClone(t.scene), state.degree); setTab('model'); } });
    const draft = readDraft();
    if (draft) { state.scene = draft.scene; state.degree = draft.degree || 4; state.engine = draft.engine === 'coarse' ? 'coarse' : 'network'; } else { state.scene = structuredClone(state.templates.gaussian?.scene || Object.values(state.templates)[0].scene); }
    renderAll(); schedulePreview();
    await refreshJobs(); await openJob(state.jobs[0]?.job_id || '');
  } catch (error) { fail(error); }
}
start();
