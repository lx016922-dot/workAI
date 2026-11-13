// Desktop shell pieces: model-builder tree, dock (messages / progress / log), status bar, dialogs, splitters.
import { icon } from './icons.js';
import { escapeHtml } from './inspector-util.js';

// ---------- model builder tree ----------
/**
 * Render a node tree {id, label, icon, detail, count, add, removable, children} into `root`.
 * opts: selected, expanded (Set of ids), onSelect(id), onToggle(id), onAdd(kind, anchor), onRemove(id), onContext(id, x, y).
 */
export function renderTreeView(root, node, opts) {
  const rows = [];
  const walk = (n, depth) => {
    const kids = n.children || [], open = opts.expanded.has(n.id), sel = n.id === opts.selected;
    rows.push(`<div class="mb-row${sel ? ' sel' : ''}" role="treeitem" aria-selected="${sel}" ${kids.length ? `aria-expanded="${open}"` : ''} tabindex="${sel ? 0 : -1}" data-node="${escapeHtml(n.id)}" style="--depth:${depth}" title="${escapeHtml(n.detail || n.label)}">
      <span class="mb-twisty">${kids.length ? `<button type="button" class="mb-toggle${open ? ' open' : ''}" data-toggle="${escapeHtml(n.id)}" tabindex="-1" aria-label="${open ? '折叠' : '展开'}">${icon('chevron')}</button>` : ''}</span>
      ${icon(n.icon)}<span class="mb-label">${escapeHtml(n.label)}</span>${n.count !== undefined ? `<span class="mb-count">${n.count}</span>` : ''}${n.detail ? `<span class="mb-detail">${escapeHtml(n.detail)}</span>` : ''}
      <span class="mb-acts">${n.add ? `<button type="button" class="mb-act" data-add="${n.add}" title="添加" tabindex="-1">${icon('add')}</button>` : ''}${n.removable ? `<button type="button" class="mb-act" data-remove="${escapeHtml(n.id)}" title="删除" tabindex="-1">${icon('remove')}</button>` : ''}</span></div>`);
    if (open) kids.forEach(k => walk(k, depth + 1));
  };
  walk(node, 0);
  root.innerHTML = rows.join('');
  root.querySelectorAll('[data-node]').forEach(el => {
    el.addEventListener('click', ev => { if (ev.target.closest('.mb-act, .mb-toggle')) return; opts.onSelect(el.dataset.node); });
    el.addEventListener('dblclick', ev => { if (!ev.target.closest('.mb-act') && el.hasAttribute('aria-expanded')) opts.onToggle(el.dataset.node); });
    el.addEventListener('contextmenu', ev => { ev.preventDefault(); opts.onSelect(el.dataset.node); opts.onContext?.(el.dataset.node, ev.clientX, ev.clientY); });
  });
  root.querySelectorAll('[data-toggle]').forEach(b => b.addEventListener('click', () => opts.onToggle(b.dataset.toggle)));
  root.querySelectorAll('[data-add]').forEach(b => b.addEventListener('click', ev => { ev.stopPropagation(); opts.onAdd(b.dataset.add, b); }));
  root.querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', ev => { ev.stopPropagation(); opts.onRemove(b.dataset.remove); }));
  root.onkeydown = ev => {
    const items = [...root.querySelectorAll('[data-node]')], i = items.findIndex(e => e.dataset.node === opts.selected);
    if (ev.key === 'ArrowDown' && i < items.length - 1) { ev.preventDefault(); opts.onSelect(items[i + 1].dataset.node, true); }
    if (ev.key === 'ArrowUp' && i > 0) { ev.preventDefault(); opts.onSelect(items[i - 1].dataset.node, true); }
    const row = items[i];
    if (ev.key === 'ArrowRight' && row?.getAttribute('aria-expanded') === 'false') opts.onToggle(row.dataset.node);
    if (ev.key === 'ArrowLeft' && row?.getAttribute('aria-expanded') === 'true') opts.onToggle(row.dataset.node);
    if (ev.key === 'Delete' && row?.querySelector('[data-remove]')) opts.onRemove(row.dataset.node);
  };
}

/** Small floating menu at (x, y); rows [{id, label, icon, sep}] → onPick(id). */
export function popupMenu(x, y, rows, onPick) {
  document.querySelectorAll('.ctx-menu').forEach(m => m.remove());
  const menu = document.createElement('div'); menu.className = 'ctx-menu rb-menu';
  menu.innerHTML = rows.map(r => r.sep ? '<hr>' : `<button type="button" data-pick="${escapeHtml(r.id)}">${r.icon ? icon(r.icon) : '<span class="ic"></span>'}<span>${escapeHtml(r.label)}</span></button>`).join('');
  document.body.append(menu);
  const w = menu.offsetWidth, h = menu.offsetHeight;
  menu.style.left = `${Math.min(x, innerWidth - w - 4)}px`; menu.style.top = `${Math.min(y, innerHeight - h - 4)}px`;
  menu.querySelectorAll('[data-pick]').forEach(b => b.addEventListener('click', () => { menu.remove(); onPick(b.dataset.pick); }));
  setTimeout(() => document.addEventListener('click', () => menu.remove(), { once: true }), 0);
}

// ---------- log ----------
export function createLog(root, onAdd) {
  const lines = [];
  const stamp = () => new Date().toLocaleTimeString('zh-CN', { hour12: false });
  return {
    add(kind, text) {
      lines.push({ kind, text, at: stamp() }); if (lines.length > 400) lines.shift();
      root.innerHTML = lines.map(l => `<div class="log-line ${l.kind}"><time>${l.at}</time>${icon(l.kind === 'error' ? 'error' : l.kind === 'warn' ? 'warning' : l.kind === 'ok' ? 'check' : 'info')}<span>${escapeHtml(l.text)}</span></div>`).join('');
      root.scrollTop = root.scrollHeight; onAdd?.(kind);
    },
    clear() { lines.length = 0; root.innerHTML = ''; },
  };
}

// ---------- dialogs ----------
/** Modal dialog; buttons [{id, label, primary}] → resolves with the clicked id (Escape → 'cancel'). */
export function dialog({ title, body, buttons = [{ id: 'ok', label: '确定', primary: true }], wide = false }) {
  return new Promise(resolve => {
    const back = document.createElement('div'); back.className = 'dlg-back';
    back.innerHTML = `<div class="dlg${wide ? ' wide' : ''}" role="dialog" aria-modal="true" aria-label="${escapeHtml(title)}"><header>${escapeHtml(title)}<button type="button" class="dlg-x" data-id="cancel" aria-label="关闭">${icon('remove')}</button></header>
      <div class="dlg-body">${body}</div><footer>${buttons.map(b => `<button type="button" class="wb-button${b.primary ? ' primary' : ''}" data-id="${b.id}">${escapeHtml(b.label)}</button>`).join('')}</footer></div>`;
    const done = id => { back.remove(); document.removeEventListener('keydown', key); resolve(id); };
    const key = ev => { if (ev.key === 'Escape') done('cancel'); };
    back.querySelectorAll('[data-id]').forEach(b => b.addEventListener('click', () => done(b.dataset.id)));
    back.addEventListener('mousedown', ev => { if (ev.target === back) done('cancel'); });
    document.addEventListener('keydown', key); document.body.append(back);
    back.querySelector('footer .primary, footer button')?.focus();
  });
}

// ---------- splitters ----------
/** Drag handles that write CSS variables on `host` (px). spec: [{handle, variable, axis:'x'|'y', min, max, invert}] */
export function initSplitters(host, spec) {
  const KEY = 'workai.desktop.layout.v1';
  let saved = {}; try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { /* storage unavailable */ }
  for (const s of spec) {
    if (saved[s.variable]) host.style.setProperty(s.variable, saved[s.variable]);
    s.handle.addEventListener('pointerdown', ev => {
      ev.preventDefault(); s.handle.setPointerCapture(ev.pointerId); s.handle.classList.add('drag');
      const start = s.axis === 'x' ? ev.clientX : ev.clientY, base = parseFloat(getComputedStyle(host).getPropertyValue(s.variable)) || 0;
      const move = e => {
        const d = (s.axis === 'x' ? e.clientX : e.clientY) - start, v = Math.max(s.min, Math.min(s.max, base + (s.invert ? -d : d)));
        host.style.setProperty(s.variable, `${v}px`);
      };
      const up = () => {
        s.handle.removeEventListener('pointermove', move); s.handle.classList.remove('drag'); window.dispatchEvent(new Event('resize'));
        try { saved[s.variable] = host.style.getPropertyValue(s.variable); localStorage.setItem(KEY, JSON.stringify(saved)); } catch { /* storage unavailable */ }
      };
      s.handle.addEventListener('pointermove', move); s.handle.addEventListener('pointerup', up, { once: true });
    });
  }
}

// ---------- messages ----------
const fmtBytes = n => n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GiB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MiB` : `${Math.round(n / 1024)} KiB`;
export { fmtBytes };
/** Seconds as a short Chinese duration (estimates: no false precision). */
export function fmtDuration(seconds) {
  if (!(seconds >= 0)) return '—';
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} 秒`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} 分钟`;
  if (seconds < 172800) return `${(seconds / 3600).toFixed(1)} 小时`;
  return `${(seconds / 86400).toFixed(1)} 天`;
}
/** What the computation will cost on this machine (offline/resources.py), for the solver settings. */
export function estimateMarkup(e) {
  if (!e) return '';
  const saved = e.frozen_evaluations_per_control > e.evaluations_per_control ? `，是训练时时间层的 1/${(e.frozen_evaluations_per_control / e.evaluations_per_control).toFixed(1)}` : '';
  return `<table class="wb-table"><tr><th>记录</th><td>${e.record === 'compact' ? '精简记录：接收点保存全部采样点（float64），动画用抽稀的表面网格，另存 4 个完整快照' : '完整记录：全部节点、全部采样点（float64）'}，约 ${fmtBytes(e.disk_bytes)}</td></tr>
    <tr><th>时间层</th><td>${e.scheme === 'verlet2' ? '二阶单级' : '三级四阶'}，Δt ≈ ${(e.layer_step_us * 1000).toFixed(2)} ns；每组控制约 ${e.evaluations_per_control.toLocaleString()} 次单元消息${saved}</td></tr>
    <tr><th>预计耗时</th><td>${fmtDuration(e.seconds)}（${e.rate_measured ? '按本机上次计算的速度' : '按典型台式机速度，首次计算后改用本机实测'}）</td></tr>
    <tr><th>预计内存</th><td>${fmtBytes(e.memory_bytes)}${e.physical_memory_bytes ? ` / 本机 ${fmtBytes(e.physical_memory_bytes)}` : ''}</td></tr></table>`;
}

/** Message rows from a preview result: [{kind: ok|error|warn|info, text, detail}] */
export function previewMessages(p) {
  if (!p) return [{ kind: 'info', text: '正在检查模型…' }];
  const rows = [];
  if (p.ok) rows.push({ kind: 'ok', text: '模型检查通过，可以计算。' });
  for (const e of p.errors || []) rows.push({ kind: 'error', text: e });
  if (p.engine) rows.push({ kind: p.engine.out_of_training_range.length ? 'warn' : 'info', text: `快速引擎 ${p.engine.label || p.engine.name}：留出测试 ${p.engine.test_passed}/${p.engine.test_cases} 通过${p.engine.out_of_training_range.length ? `；超出训练范围 ${p.engine.out_of_training_range.length} 项：${p.engine.out_of_training_range.join('；')}` : '；在训练范围内'}` });
  if (p.mesh) rows.push({ kind: 'info', text: `${p.engine ? '粗网格（模型自带）' : p.mesh.auto ? '自动网格' : '网格'}：${p.mesh.elements.toLocaleString()} 个单元，${p.mesh.nodes.toLocaleString()} 个节点，阶次 p = ${p.mesh.degree}`
    + (p.mesh.auto && !p.engine ? `，单元 ${+p.mesh.auto.h_mm.toFixed(3)} mm（${p.mesh.auto.reason}）${p.mesh.auto.split_nodes ? `；缺陷面分开 ${p.mesh.auto.split_nodes} 个节点${p.mesh.auto.contact_pairs ? `，${p.mesh.auto.contact_pairs} 对接触` : ''}` : ''}` : '') });
  if (p.mesh?.auto?.warning && !p.engine) rows.push({ kind: 'warn', text: p.mesh.auto.warning });
  if (p.record) rows.push({ kind: 'info', text: `记录：${p.record.samples.toLocaleString()} 个采样点，${p.record.mode === 'compact' ? '精简记录（接收点完整波形、抽稀表面动画、4 个完整快照）' : '完整记录'}约 ${fmtBytes(p.record.uncompressed_bytes)}${p.record.free_bytes ? `，磁盘可用 ${fmtBytes(p.record.free_bytes)}` : ''}`,
    ...(p.record.mode === 'compact' ? { detail: '模型较大，全部节点的完整记录超过 2 GiB，自动改用精简记录：结果窗口只能在接收点提取波形，请先把关心的位置设为接收点。' } : {}) });
  if (p.estimate) rows.push({ kind: 'info', text: `计算量：${p.estimate.scheme === 'verlet2' ? '二阶单级' : '三级四阶'}时间层，Δt ≈ ${(p.estimate.layer_step_us * 1000).toFixed(2)} ns；预计 ${fmtDuration(p.estimate.seconds)}，内存约 ${fmtBytes(p.estimate.memory_bytes)}`,
    detail: p.estimate.rate_measured ? '耗时按本机上次计算的速度估计。' : '耗时按典型台式机速度估计；本机完成一次计算后改用实测速度。' });
  for (const r of p.resolution || []) {
    const n = r.nodes_per_2f_shear_wavelength;
    rows.push({ kind: n >= 8 ? 'info' : 'warn', text: `2f 分辨率（${r.name}）：${n.toFixed(1)} 个节点 / 2f 剪切波长${n < 4 ? '，二次谐波分辨不足' : n < 8 ? '，适合短记录；长记录需复核' : '，分辨充分'}` });
  }
  if (p.scope?.length) rows.push({ kind: 'warn', text: `超出原八例验收范围 ${p.scope.length} 项：${p.scope.join('；')}`, detail: '可以计算，但结果属于研究结果，需要用细网格参考逐例核对。' });
  else if (p.ok) rows.push({ kind: 'ok', text: '在原八例验收范围内。' });
  for (const w of p.warnings || []) rows.push({ kind: 'warn', text: w });
  return rows;
}

export function renderMessages(root, rows) {
  root.innerHTML = rows.map(r => `<div class="msg ${r.kind}">${icon(r.kind === 'error' ? 'error' : r.kind === 'warn' ? 'warning' : r.kind === 'ok' ? 'check' : 'info')}<div><span>${escapeHtml(r.text)}</span>${r.detail ? `<small>${escapeHtml(r.detail)}</small>` : ''}</div></div>`).join('');
}
