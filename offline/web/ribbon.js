// Ribbon toolbar of the desktop shell: a File menu, tabs, and groups of large/small command buttons.
// Commands are plain ids handled by the application (`run`, `sel:geometry`, `add:source`, `view:iso`, …).
import { icon } from './icons.js';
import { escapeHtml } from './inspector-util.js';

const L = (id, label, ic, extra = {}) => ({ id, label, icon: ic, size: 'L', ...extra });
const S = (id, label, ic, extra = {}) => ({ id, label, icon: ic, size: 'S', ...extra });

export const RIBBON = [
  { id: 'home', label: '主屏幕', groups: [
    { label: '模型', items: [L('new', '新建', 'new', { menu: 'templates', tip: '从模板新建模型' }), L('open', '打开', 'open', { tip: '打开模型文件（Ctrl+O）' }), L('save', '保存', 'save', { tip: '保存模型文件（Ctrl+S）' })] },
    { label: '编辑', items: [S('undo', '撤销', 'undo', { tip: '撤销（Ctrl+Z）' })] },
    { label: '定义', items: [L('sel:geometry', '几何', 'geometry'), L('sel:material', '材料', 'material')] },
    { label: '物理场', items: [L('add:source', '激励源', 'source', { tip: '添加一个激励源' }), L('add:boundary', '边界条件', 'boundary', { menu: 'boundary', tip: '添加边界条件' }), L('add:receiver', '接收点', 'receiver', { tip: '添加一个接收点' })] },
    { label: '网格', items: [L('sel:mesh', '网格', 'mesh')] },
    { label: '研究', items: [L('run', '计算', 'compute', { primary: true, tip: '用物理 AI 网络计算当前模型（F8）' }), S('pause', '暂停', 'pause', { tip: '当前分块保存后暂停（5 ms 及以上的长记录）' }), S('resume', '继续', 'resume', { tip: '从检查点继续计算' }), S('sel:study', '研究设置', 'study')] },
    { label: '结果', items: [L('tab:result', '结果', 'results'), S('tab:signal', '激励信号', 'signal'), S('export-result', '导出结果', 'export', { tip: '下载所选计算的全部结果 ZIP' })] },
  ] },
  { id: 'geometry', label: '几何', groups: [
    { label: '结构类型', items: [L('kind:plate', '平板', 'geometry'), L('kind:pipe', '圆管', 'geometry'), L('kind:hole', '带通孔板', 'geometry'), L('kind:notch', '带缺口板', 'geometry')] },
    { label: '设置', items: [L('sel:geometry', '几何设置', 'geometry')] },
    { label: '视图', items: [S('view:iso', '等轴视图', 'viewIso'), S('view:front', '正视 XZ', 'viewFront'), S('view:top', '俯视 XY', 'viewTop'), S('view:side', '侧视 YZ', 'viewSide'), S('toggle:thin', '放大薄方向', 'thin')] },
  ] },
  { id: 'material', label: '材料', groups: [
    { label: '材料', items: [L('sel:material', '材料设置', 'material', { tip: '多晶铝、各向同性 Murnaghan 或自定义应变能 W(E)' })] },
  ] },
  { id: 'physics', label: '物理场', groups: [
    { label: '激励', items: [L('add:source', '添加激励', 'source'), S('sel:sources', '激励列表', 'source')] },
    { label: '边界条件', items: [L('add-boundary:symmetry', '对称面', 'symmetry'), L('add-boundary:antisymmetry', '反对称面', 'antisymmetry'), L('add-boundary:absorbing', '吸收层', 'absorbing'), L('add-boundary:fixed', '固定面', 'fixed')] },
    { label: '接收', items: [L('add:receiver', '添加接收点', 'receiver'), S('sel:receivers', '接收点列表', 'receiver')] },
  ] },
  { id: 'mesh', label: '网格', groups: [
    { label: '网格', items: [L('sel:mesh', '网格设置', 'mesh'), S('toggle:grid', '显示单元网格', 'grid')] },
  ] },
  { id: 'study', label: '研究', groups: [
    { label: '计算', items: [L('run', '计算', 'compute', { primary: true }), S('pause', '暂停', 'pause'), S('resume', '继续', 'resume')] },
    { label: '设置', items: [L('sel:study', '瞬态记录', 'study'), L('sel:engine', '求解器', 'engine')] },
  ] },
  { id: 'results', label: '结果', groups: [
    { label: '结果', items: [L('tab:result', '结果窗口', 'results'), L('tab:signal', '激励信号', 'signal'), L('export-result', '导出 ZIP', 'export')] },
    { label: '计算记录', items: [S('refresh', '刷新', 'log'), S('sel:results', '全部记录', 'folder')] },
  ] },
  { id: 'view', label: '视图', groups: [
    { label: '方向', items: [L('view:iso', '等轴', 'viewIso'), S('view:front', '正视 XZ', 'viewFront'), S('view:top', '俯视 XY', 'viewTop'), S('view:side', '侧视 YZ', 'viewSide'), L('zoom-fit', '缩放到窗口', 'zoomFit')] },
    { label: '显示', items: [S('toggle:grid', '单元网格', 'grid'), S('toggle:thin', '放大薄方向', 'thin'), S('screenshot', '保存截图', 'camera')] },
    { label: '窗口', items: [S('toggle:dock', '消息窗口', 'message'), S('tab:model', '图形窗口', 'geometry')] },
  ] },
  { id: 'help', label: '帮助', groups: [
    { label: '帮助', items: [L('help', '使用说明', 'help'), L('about', '关于', 'about')] },
  ] },
];

const FILE_MENU = [
  { id: 'new', label: '新建', icon: 'new', sub: 'templates' }, { id: 'open', label: '打开…', icon: 'open', key: 'Ctrl+O' }, { id: 'save', label: '保存…', icon: 'save', key: 'Ctrl+S' },
  { sep: true }, { id: 'export-result', label: '导出计算结果', icon: 'export' }, { sep: true },
  { id: 'help', label: '使用说明', icon: 'help', key: 'F1' }, { id: 'about', label: '关于 WorkAI', icon: 'about' }, { sep: true }, { id: 'exit', label: '退出', icon: 'exit' },
];

function button(item) {
  const caret = item.menu ? `<span class="rb-caret">${icon('caret')}</span>` : '';
  return `<button type="button" class="rb-btn ${item.size === 'L' ? 'big' : 'small'}${item.primary ? ' primary' : ''}" data-cmd="${item.id}" ${item.menu ? `data-menu="${item.menu}"` : ''} title="${escapeHtml(item.tip || item.label)}">${icon(item.icon)}<span class="rb-label">${escapeHtml(item.label)}${caret}</span></button>`;
}

function group(g) {
  // small buttons stack three to a column, large ones stand alone
  const cols = []; let stack = [];
  for (const item of g.items) {
    if (item.size === 'L') { if (stack.length) { cols.push(stack); stack = []; } cols.push([item]); }
    else { stack.push(item); if (stack.length === 3) { cols.push(stack); stack = []; } }
  }
  if (stack.length) cols.push(stack);
  return `<div class="rb-group"><div class="rb-items">${cols.map(c => `<div class="rb-col${c[0].size === 'L' ? ' big' : ''}">${c.map(button).join('')}</div>`).join('')}</div><div class="rb-group-label">${escapeHtml(g.label)}</div></div>`;
}

/**
 * Mount the ribbon in `root`. `app` supplies: command(id), menu(kind) → [{id, label}], disabled(id), active(id).
 * Returns { refresh() } to re-apply enabled/active states.
 */
export function mountRibbon(root, app) {
  let current = 'home';
  root.innerHTML = `<div class="rb-bar">
      <button type="button" class="rb-file" data-file-menu>文件<span class="rb-caret">${icon('caret')}</span></button>
      <nav class="rb-tabs" role="tablist">${RIBBON.map(t => `<button type="button" role="tab" data-ribbon-tab="${t.id}">${escapeHtml(t.label)}</button>`).join('')}</nav>
      <div class="rb-title"><span id="rb-doc"></span></div>
      <button type="button" class="rb-quick" data-cmd="undo" title="撤销（Ctrl+Z）">${icon('undo')}</button>
      <button type="button" class="rb-quick" data-cmd="save" title="保存（Ctrl+S）">${icon('save')}</button>
      <button type="button" class="rb-quick" data-cmd="help" title="使用说明（F1）">${icon('help')}</button>
    </div>
    <div class="rb-panel" role="tabpanel"></div>
    <div class="rb-pop" hidden></div>`;
  const panel = root.querySelector('.rb-panel'), pop = root.querySelector('.rb-pop');
  const closePop = () => { pop.hidden = true; pop.innerHTML = ''; };
  const openPop = (anchor, html, onClick) => {
    pop.innerHTML = html; pop.hidden = false;
    const r = anchor.getBoundingClientRect(), host = root.getBoundingClientRect();
    pop.style.left = `${Math.max(4, r.left - host.left)}px`; pop.style.top = `${r.bottom - host.top + 2}px`;
    pop.querySelectorAll('[data-pick]').forEach(b => b.addEventListener('click', ev => { ev.stopPropagation(); closePop(); onClick(b.dataset.pick); }));
  };
  const menuHtml = rows => `<div class="rb-menu">${rows.map(r => r.sep ? '<hr>' : `<button type="button" data-pick="${escapeHtml(r.id)}" ${r.disabled ? 'disabled' : ''}>${r.icon ? icon(r.icon) : '<span class="ic"></span>'}<span>${escapeHtml(r.label)}</span>${r.key ? `<kbd>${r.key}</kbd>` : ''}${r.sub ? `<span class="rb-sub">${icon('chevron')}</span>` : ''}</button>`).join('')}</div>`;

  const showTab = id => {
    current = id; root.querySelectorAll('[data-ribbon-tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.ribbonTab === id)));
    panel.innerHTML = RIBBON.find(t => t.id === id).groups.map(group).join('<span class="rb-sep"></span>');
    panel.querySelectorAll('[data-cmd]').forEach(bind); refresh();
  };
  const bind = b => b.addEventListener('click', ev => {
    ev.stopPropagation();
    if (b.dataset.menu) {
      const rows = app.menu(b.dataset.menu);
      return openPop(b, menuHtml(rows), pick => app.command(b.dataset.menu === 'boundary' ? `add-boundary:${pick}` : `template:${pick}`));
    }
    closePop(); app.command(b.dataset.cmd);
  });
  root.querySelectorAll('.rb-bar [data-cmd]').forEach(bind);
  root.querySelectorAll('[data-ribbon-tab]').forEach(b => b.addEventListener('click', () => { closePop(); showTab(b.dataset.ribbonTab); }));
  root.querySelector('[data-file-menu]').addEventListener('click', ev => {
    ev.stopPropagation(); const anchor = ev.currentTarget;
    const rows = FILE_MENU.map(r => ({ ...r, disabled: r.id && app.disabled(r.id) }));
    openPop(anchor, menuHtml(rows), pick => {
      if (pick === 'new') return setTimeout(() => openPop(anchor, menuHtml(app.menu('templates')), t => app.command(`template:${t}`)), 0);
      app.command(pick);
    });
  });
  document.addEventListener('click', ev => { if (!ev.target.closest('.rb-pop')) closePop(); });
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape') closePop(); });

  function refresh() {
    root.querySelectorAll('[data-cmd]').forEach(b => { b.disabled = app.disabled(b.dataset.cmd); b.classList.toggle('on', app.active(b.dataset.cmd)); });
  }
  showTab(current);
  return { refresh, showTab, current: () => current };
}
