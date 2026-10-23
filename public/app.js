import { createWaveformUI } from './waveform-ui.js';
import { createModelUI } from './model-ui.js';
import { buildOfflineReport } from './offline-report.js';
import { renderTraining } from './training-ui.js';
import { createSolidUI } from './solid-ui.js';
const $ = selector => document.querySelector(selector);
const names = { model: '建模', physics: '物理场', study: '研究', overview: '研究总览', jobs: '仿真任务', results: '结果', connection: '服务连接', training: 'AI 训练', solid: '三维超弹性' };
const state = { view: 'overview', bootstrap: null, health: null, jobs: [], selected: '', job: null, series: null, seriesError: '', jobError: '', metric: 'reflection', loadingSeries: false };
let pollTimer, toastTimer, selectionVersion = 0, previewVersion = 0, refreshBusy = false, submitBusy = false;
let trainingTimer, trainingBusy = false;
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const shortId = value => value ? `${value.slice(0, 15)}…${value.slice(-5)}` : '—';
const date = value => value && Number.isFinite(Date.parse(value)) ? new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Taipei' }).format(new Date(value)) : '—';
const size = bytes => bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
const kind = value => ({ smoke: '连接验证', plate_smoke: '平板快速验证', plate_reference: '平板参考计算', benchmark: '基准计算', research_reference: '二维导波研究' }[value] || '仿真任务');
const final = job => ['completed', 'failed'].includes(job?.phase);
function pill(phase) {
  const labels = { completed: ['已完成', 'good'], failed: ['失败', 'bad'], starting: ['启动中', 'running'], running: ['计算中', 'running'], queued: ['排队中', 'running'], unknown: ['待查询', 'warn'] };
  const [text, css] = labels[phase] || ['待查询', 'warn'];
  return `<span class="pill ${css}">${text}</span>`;
}
function toast(message) {
  clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').hidden = false;
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 6500);
}
async function api(path, body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? { cache: 'no-store' } : {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-WorkAI-CSRF': state.bootstrap.csrf_token }, body: JSON.stringify(body),
    });
  } catch { throw new Error('无法连接工作台，请检查网络后刷新。'); }
  const value = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(value.error || `请求失败（${response.status}）。`);
    error.status = response.status; throw error;
  }
  return value;
}
function heading(title, subtitle, action = '') {
  return `<div class="page-heading"><div><div class="eyebrow">ULTRASOUND RESEARCH</div><h1>${title}</h1><p>${subtitle}</p></div>${action}</div>`;
}
const newButton = () => '<button class="button primary" data-action="new-job"><span aria-hidden="true">＋</span> 新建验证任务</button>';
function errorNotice(message) { return message ? `<div class="notice error-notice" role="alert">${escape(message)}</div>` : ''; }
function duration(job) {
  if (!job?.state?.started_at_utc || !final(job)) return '—';
  const seconds = (Date.parse(job.state.updated_at_utc) - Date.parse(job.state.started_at_utc)) / 1000;
  return Number.isFinite(seconds) && seconds >= 0 ? `${Math.round(seconds)} 秒` : '—';
}
function servicePanel() {
  const services = state.health?.services || {};
  const research = services.research;
  const researchKnown = research?.ok && research.data.source_sha256 === state.bootstrap.migration.research_source_sha256;
  return `<div class="service-row"><div class="service-icon">⌁</div><div class="service-info"><h3>COMSOL 计算服务</h3><p>${services.base?.ok ? '远程连接与身份验证已通过' : services.base?.error ? '暂时无法验证远程连接' : '等待状态检查'}</p></div><span class="pill ${services.base?.ok ? 'good' : 'warn'}">${services.base?.ok ? '已连接' : services.base ? '未连接' : '检查中'}</span></div>
    <div class="service-row"><div class="service-icon">▱</div><div class="service-info"><h3>三维平板模型</h3><p>${services.plate?.ok ? '有限平板 · 表面开口缺口' : '等待远程模型状态'}</p></div><span class="pill ${services.plate?.ok ? 'good' : 'warn'}">${services.plate?.ok ? '接口可用' : '待检查'}</span></div>
    <div class="service-row"><div class="service-icon">∿</div><div class="service-info"><h3>二维研究模型</h3><p>${researchKnown ? '已核对源码 · 线性 / 材料非线性' : research?.ok ? '服务器版本与已恢复源码不同' : '等待研究接口检查'}</p></div><span class="pill ${researchKnown ? 'good' : 'warn'}">${researchKnown ? '版本一致' : '待检查'}</span></div>
    <div class="service-row"><div class="service-icon">◇</div><div class="service-info"><h3>大体积模型</h3><p>${services.volume?.ok ? services.volume.data.model_ready ? '兼容模型已就绪' : '协议已安装，兼容模型尚未就绪' : '等待远程模型状态'}</p></div><span class="pill ${services.volume?.data?.model_ready ? 'good' : 'warn'}">${services.volume?.data?.model_ready ? '已就绪' : '待配置'}</span></div>`;
}
function chartPanel() {
  return `<section class="panel"><div class="panel-heading"><div><h2>频率响应预览</h2><p>连接验证算例 · ${state.metric === 'reflection' ? '反射系数幅值 |R|' : '应力透射系数幅值 |T|'}（无量纲）</p></div><button class="button text small" data-action="toggle-metric">${state.metric === 'reflection' ? '查看透射' : '查看反射'} ↗</button></div><div class="panel-body"><div class="legend"><span><i></i>粗网格 · 8 单元</span><span class="fine"><i></i>细网格 · 16 单元</span></div><div class="chart-wrap">${chart()}</div><div class="chart-caption">3 MHz 与 6 MHz 是两次独立的频率计算，不是一次激励产生的基频和二次谐波；不能据此计算非线性系数。此图也不代表完整 A 扫或收敛结论。</div></div></section>`;
}
function chart() {
  if (state.loadingSeries) return '<div class="chart-empty">正在校验并读取结果…</div>';
  if (state.seriesError) return `<div class="chart-empty">${escape(state.seriesError)}<br><button class="button text" data-action="reload-series">重新读取</button></div>`;
  if (!state.series) return '<div class="chart-empty">选择一个已完成的连接验证任务后<br>即可查看真实计算结果。</div>';
  const { coarse, fine } = state.series;
  const all = [...coarse, ...fine];
  const minX = Math.min(...all.map(p => p.frequency_mhz)), maxX = Math.max(...all.map(p => p.frequency_mhz));
  const maxY = Math.max(.2, Math.ceil(Math.max(...all.map(p => p[state.metric])) * 1.15 * 5) / 5);
  const x = value => 58 + (value - minX) / (maxX - minX || 1) * 492;
  const y = value => 213 - value / maxY * 165;
  const grid = Array.from({ length: 5 }, (_, i) => {
    const value = maxY * i / 4, yy = y(value);
    return `<line x1="58" x2="550" y1="${yy}" y2="${yy}" stroke="#edf1f3"/><text x="46" y="${yy + 4}" text-anchor="end" fill="#98a6ae" font-size="11">${value.toFixed(2)}</text>`;
  }).join('');
  const frequencies = [...new Set(all.map(p => p.frequency_mhz))].sort((a, b) => a - b);
  const axis = frequencies.filter((_, i) => i % Math.max(1, Math.ceil(frequencies.length / 8)) === 0).map(value => `<text x="${x(value)}" y="238" text-anchor="middle" fill="#98a6ae" font-size="11">${value.toFixed(2)}</text>`).join('');
  const series = (points, color) => `<polyline points="${points.map(p => `${x(p.frequency_mhz)},${y(p[state.metric])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.2"/>${points.map(p => `<circle cx="${x(p.frequency_mhz)}" cy="${y(p[state.metric])}" r="4" fill="white" stroke="${color}" stroke-width="2"><title>${p.frequency_mhz} MHz · ${p[state.metric].toFixed(6)}</title></circle>`).join('')}`;
  return `<svg viewBox="0 0 590 265" role="img" aria-label="粗网格和细网格的${state.metric === 'reflection' ? '反射' : '透射'}系数幅值对比"><title>连接验证任务的频率结果</title>${grid}${axis}${series(coarse, '#148778')}${series(fine, '#729fc9')}<text x="304" y="262" text-anchor="middle" fill="#9babb3" font-size="10">频率 / MHz</text></svg>`;
}
function jobTable(limit) {
  const records = limit ? state.jobs.slice(0, limit) : state.jobs;
  if (!records.length) return '<div class="empty">还没有恢复的任务记录。可用任务编号查询已有结果，或新建连接验证任务。</div>';
  return `<div class="table-scroll"><table><thead><tr><th>任务</th><th>状态</th><th>创建时间</th><th>操作</th></tr></thead><tbody>${records.map(record => `<tr><td><div class="job-title">${escape(kind(record.kind))}</div><div class="job-id mono" title="${escape(record.job_id)}">${escape(shortId(record.job_id))}</div></td><td>${pill(record.last_status?.phase)}</td><td class="muted">${date(record.created_at)}</td><td><div class="row-actions"><button class="table-action" data-job="${escape(record.job_id)}" data-target="results">查看结果</button><button class="table-action" data-query="${escape(record.job_id)}">查询</button></div></td></tr>`).join('')}</tbody></table></div>`;
}
function renderOverview() {
  const base = state.health?.services.base;
  const volume = state.health?.services.volume;
  const completed = state.jobs.filter(record => record.last_status?.phase === 'completed').length;
  return `${heading('超声仿真工作台', '当前聚焦材料非线性导波中的二次谐波产生与累积。', newButton())}
    ${state.health && !base?.ok ? errorNotice(base?.error || '远程服务暂时不可达。保留的任务记录仍可查询。') : ''}
    <div class="cards"><section class="metric"><span class="metric-label">计算服务</span><span class="metric-mark">⌁</span><div class="metric-value ${base?.ok ? 'good' : ''}">${base?.ok ? '已连接' : base ? '待连接' : '检查中'}</div><p class="metric-meta">${base?.ok ? 'COMSOL · 签名验证通过' : '远程 COMSOL'}</p></section><section class="metric"><span class="metric-label">已完成任务</span><span class="metric-mark">✓</span><div class="metric-value">${completed}<span class="muted small-text"> / ${state.jobs.length}</span></div><p class="metric-meta">当前工作区保留的任务</p></section><section class="metric"><span class="metric-label">当前任务结果</span><span class="metric-mark">▤</span><div class="metric-value">${state.job ? state.job.artifacts.length : '—'} <span class="muted small-text">份文件</span></div><p class="metric-meta">以远程任务清单为准</p></section><section class="metric"><span class="metric-label">大体积模型</span><span class="metric-mark">◇</span><div class="metric-value ${volume?.data?.model_ready ? 'good' : 'warn'}">${volume?.ok ? volume.data.model_ready ? '已就绪' : '待配置' : '待检查'}</div><p class="metric-meta">${volume?.ok ? '已读取模型状态' : '等待连接检查'}</p></section></div>
    <div class="two-col">${state.job?.kind === 'research_reference' ? waveformUI.render() : chartPanel()}<section class="panel"><div class="panel-heading"><h2>研究环境</h2><a href="#connection" class="button text small">查看详情 ↗</a></div><div class="panel-body">${servicePanel()}<div class="section-note">模型接口可用不代表已完成求解；请以任务状态和输出文件判断计算结果。</div></div></section></div>
    <section class="panel"><div class="panel-heading"><h2>最近任务</h2><a href="#study" class="button text small">全部任务 ↗</a></div>${jobTable(4)}<div class="panel-footer">${state.health ? `最近服务检查：${date(state.health.checked_at)}` : '正在检查服务连接'}</div></section>`;
}
function renderJobs() {
  return `${heading('仿真任务', '查询已有任务，或运行一次固定算例来验证远程计算流程。', newButton())}${errorNotice(state.jobError)}<section class="panel"><div class="panel-heading"><h2>任务记录</h2><span class="download-count">${state.jobs.length} 个任务</span></div>${jobTable()}<form id="import-form" class="import-form"><input id="import-id" aria-label="已有任务编号" placeholder="输入已有任务编号：comsol-…" required pattern="comsol-[a-f0-9]{32}" autocomplete="off"><button class="button secondary" type="submit">查询并加入列表</button></form><div class="panel-footer">这里只显示当前工作区保留或手动加入的任务。服务器没有提供全部历史任务的列表接口。</div></section>`;
}
function fileTable(job) {
  if (!job?.artifacts?.length) return '<div class="empty">任务完成后，结果文件会出现在这里。</div>';
  return `<div class="table-scroll"><table><thead><tr><th>结果文件</th><th>大小</th><th>SHA-256</th><th>操作</th></tr></thead><tbody>${job.artifacts.map(file => {
    const tooLarge = file.bytes > state.bootstrap.max_artifact_bytes;
    return `<tr><td><span class="file-mark">${escape(file.name.split('.').at(-1).toUpperCase())}</span>${escape(file.name)}</td><td class="muted">${size(file.bytes)}</td><td class="hash-cell" title="${escape(file.sha256)}">${escape(file.sha256.slice(0, 12))}…</td><td><div class="row-actions">${!file.name.endsWith('.mph') && !tooLarge ? `<button class="table-action" data-preview="${escape(file.name)}">预览</button>` : ''}${tooLarge ? '<span class="muted small-text">超过当前下载限制</span>' : `<button class="table-action" data-download="${escape(file.name)}">下载 ↓</button>`}</div></td></tr>`;
  }).join('')}</tbody></table></div>`;
}
function renderResults() {
  const job = state.job;
  const record = state.jobs.find(record => record.job_id === state.selected);
  return `${heading('结果文件', '查看求解结果、日志与模型文件；下载前会自动校验文件。', state.waveforms?.job_id === state.selected && !state.loadingWaveforms && !state.waveformError ? '<button class="button secondary" data-action="export-offline">导出离线结果 HTML</button>' : '')}
    <div class="toolbar"><label for="job-select">当前任务</label><select id="job-select" ${state.jobs.length ? '' : 'disabled'}>${state.jobs.length ? state.jobs.map(record => `<option value="${escape(record.job_id)}" ${record.job_id === state.selected ? 'selected' : ''}>${escape(kind(record.kind))} · ${escape(shortId(record.job_id))} · ${date(record.created_at)}</option>`).join('') : '<option>暂无任务</option>'}</select><button class="button secondary" data-action="query-current" ${state.selected ? '' : 'disabled'}>查询状态</button></div>
    ${errorNotice(state.jobError)}
    ${record?.submission_state === 'unknown' && !job ? '<div class="progress-note">任务提交结果尚未确认。请保留当前任务编号，查询状态；不要重新提交。</div>' : ''}
    ${job && !final(job) ? '<div class="progress-note" role="status">任务正在执行，每 5 秒查询一次状态。可以离开此页面，任务会继续在远程服务器运行。</div>' : ''}
    <section class="panel"><div class="panel-heading"><h2>${job ? escape(kind(job.kind)) : '任务详情'}</h2>${pill(job?.phase)}</div><div class="panel-body"><div class="metadata"><div><span>任务编号</span><strong><button class="copy-button mono" data-action="copy-id" title="复制完整编号">${escape(shortId(state.selected))}</button></strong></div><div><span>计算耗时</span><strong>${duration(job)}</strong></div><div><span>最近状态查询</span><strong>${date(record?.checked_at)}</strong></div></div>${job?.kind === 'smoke' ? '<div class="notice">连接验证算例仅用于验证求解与导出流程。MPH 模型的解状态已清除，查看场结果需要重新求解。</div>' : ''}</div>${fileTable(job)}<div class="panel-footer">单个文件支持至 1 MiB；只提供服务器清单内、通过签名和内容校验的文件。</div></section>
    ${job?.kind === 'smoke' && job.phase === 'completed' ? `<div class="status-line">当前任务的频率结果</div>${chartPanel()}` : ''}${job?.kind === 'research_reference' && job.phase === 'completed' ? waveformUI.render() : ''}`;
}
function renderConnection() {
  const connection = state.bootstrap.connection;
  return `${heading('服务连接', '使用当前已配对的计算服务，保留原有服务器与客户端身份。')}
    <div class="connection-grid"><section class="panel"><div class="panel-heading"><h2>远程计算服务</h2><span class="pill ${state.health?.services.base?.ok ? 'good' : 'warn'}">${state.health?.services.base?.ok ? '已验证' : '待验证'}</span></div><div class="panel-body">${servicePanel()}<div class="section-note">请保持 Windows 服务器上的 COMSOL HTTP 服务运行。服务中断后，恢复连接即可继续查询已有任务。</div><p class="status-line">最近检查：${date(state.health?.checked_at)}</p><dl class="definition"><dt>API 地址</dt><dd class="mono">${escape(connection.base_url)}</dd><dt>连接网络</dt><dd>Tailscale</dd><dt>服务端指纹</dt><dd class="mono">${escape(connection.server_fingerprint)}</dd><dt>客户端指纹</dt><dd class="mono">${escape(connection.client_fingerprint)}</dd></dl></div></section>
    <section class="panel"><div class="panel-heading"><h2>恢复进度</h2></div><div class="panel-body"><ul class="check-list"><li><span class="check">✓</span>远程连接、签名验证与结果回传</li><li><span class="check">✓</span>任务查询、日志预览与文件下载</li><li><span class="check">✓</span>连接验证任务与频率结果预览</li><li><span class="check">✓</span>铝材料有限应变本构、五组对照与二次谐波分析已接入</li><li><span class="pending">○</span>原网站源码、布局和项目历史尚未取得</li><li><span class="pending">○</span>二维研究 A 扫已接入，场分布动画待恢复</li><li><span class="pending">○</span>大体积计算所需的兼容模型待安装</li></ul><div class="section-note">当前界面依据已恢复的服务重建。取得原站的页面资料或源码后，可继续对齐原有功能与布局。</div></div></section></div>`;
}
function render() {
  if (!state.bootstrap) return;
  const view = names[location.hash.slice(1)] ? location.hash.slice(1) : 'model';
  state.view = view;
  document.title = `${names[view]} · workAI`;
  $('#page-name').textContent = names[view];
  document.querySelectorAll('[data-view]').forEach(link => {
    const active = link.dataset.view === view;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  solidUI.leave(view!=='solid');
  $('#main').innerHTML = { model: modelUI.renderModel, physics: modelUI.renderPhysics, study: modelUI.renderStudy, overview: renderOverview, jobs: renderJobs, results: renderResults, connection: renderConnection, training: () => renderTraining(state, escape, heading), solid: solidUI.markup }[view]();
  if(view==='solid')solidUI.mount();
  if (view !== 'training') clearTimeout(trainingTimer);
  else if (!state.training && !state.trainingError && !trainingBusy) void loadTraining();
  if (state.savingModel) document.querySelectorAll('#main input, #main select, #main [data-model-action], #main [data-model-add], #main [data-model-remove]').forEach(control => { control.disabled = true; });
  const base = state.health?.services.base;
  $('#sidebar-status').innerHTML = `${base?.ok ? '计算服务已连接' : base ? '计算服务待连接' : '正在检查服务'}<small>远程 COMSOL</small>`;
  $('#sidebar-dot').className = `dot ${base?.ok ? 'good' : base ? 'bad' : ''}`;
}
async function loadTraining() {
  if (trainingBusy) return;
  trainingBusy = true; clearTimeout(trainingTimer); state.trainingError = '';
  try { state.training = await api('/api/training'); }
  catch (error) { state.trainingError = error.message; }
  finally {
    trainingBusy = false;
    if (state.view === 'training') {
      render();
      if (['preparing', 'training', 'evaluating', 'stale'].includes(state.training?.status)) trainingTimer = setTimeout(() => void loadTraining(), 5000);
    }
  }
}
async function loadWaveforms(id, version) {
  if (state.job?.kind !== 'research_reference' || state.job.phase !== 'completed') return;
  state.loadingWaveforms = true; state.waveformError = ''; render();
  try {
    const result = await api(`/api/jobs/${id}/waveforms`);
    if (version === selectionVersion && id === state.selected) { state.waveforms = result; if (result.constitutive === 'murnaghan_green_lagrange') state.waveComponent = 'u_nm'; }
  } catch (error) { if (version === selectionVersion) state.waveformError = error.message; }
  finally { if (version === selectionVersion) { state.loadingWaveforms = false; render(); } }
}
async function loadSeries(id, version) {
  if (state.job?.kind === 'research_reference') return loadWaveforms(id, version);
  if (state.job?.kind !== 'smoke' || state.job.phase !== 'completed') return;
  state.loadingSeries = true; state.seriesError = ''; render();
  try {
    const series = await api(`/api/jobs/${id}/series`);
    if (version === selectionVersion && id === state.selected) state.series = series;
  } catch (error) { if (version === selectionVersion) state.seriesError = error.message; }
  finally { if (version === selectionVersion) { state.loadingSeries = false; render(); } }
}
async function loadJob(id, { showToast = false, reloadSeries = true } = {}) {
  clearTimeout(pollTimer);
  const version = ++selectionVersion;
  const changed = id !== state.selected;
  state.selected = id; state.jobError = '';
  if (changed) { state.waveforms = null; state.waveformError = ''; state.loadingWaveforms = false; state.job = null; state.series = null; state.seriesError = ''; state.loadingSeries = false; }
  render();
  try {
    const result = await api(`/api/jobs/${id}`);
    if (version !== selectionVersion) return;
    state.job = result.job;
    const record = state.jobs.find(record => record.job_id === id);
    if (record) { record.last_status = result.job; record.checked_at = result.checked_at; }
    render();
    if (showToast) toast('任务状态已更新。');
    if (reloadSeries || (!state.series && final(state.job))) void loadSeries(id, version);
    if (!final(state.job)) pollTimer = setTimeout(() => void loadJob(id, { reloadSeries: false }), 5000);
  } catch (error) {
    if (version !== selectionVersion) return;
    state.jobError = error.message; render();
    if (showToast) toast(error.message);
  }
}
async function refresh() {
  if (refreshBusy) return;
  refreshBusy = true; $('#refresh').disabled = true;
  try {
    const [health, records] = await Promise.all([api('/api/health'), api('/api/jobs')]);
    state.health = health; state.jobs = records.jobs; render();
    const id = state.selected || state.jobs[0]?.job_id;
    if (id) await loadJob(id, { reloadSeries: !state.series });
  } catch (error) { toast(error.message); }
  finally { refreshBusy = false; $('#refresh').disabled = false; }
}
async function download(name, button) {
  button.disabled = true; const label = button.textContent; button.textContent = '校验中…';
  const id = state.selected;
  try {
    const response = await fetch(`/api/jobs/${id}/files/${name}`);
    if (!response.ok) throw new Error((await response.json()).error || '文件下载失败。');
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a'); link.href = url; link.download = name; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000); toast(`${name} 已验证并下载。`);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; button.textContent = label; }
}
async function preview(name) {
  const version = ++previewVersion;
  $('#preview-title').textContent = name; $('#preview-meta').textContent = '正在校验文件…'; $('#preview-content').textContent = '读取中…';
  $('#preview-dialog').showModal();
  try {
    const result = await api(`/api/jobs/${state.selected}/preview/${name}`);
    if (version !== previewVersion) return;
    $('#preview-meta').textContent = `校验通过 · SHA-256 ${result.sha256}`;
    $('#preview-content').textContent = result.text || '（空文件）';
  } catch (error) { if (version === previewVersion) { $('#preview-meta').textContent = '读取未完成'; $('#preview-content').textContent = error.message; } }
}
async function submitJob(study = false) {
  if (submitBusy) return;
  submitBusy = true; $('#submit-job').disabled = true; $('#submit-study').disabled = true;
  const id = `comsol-${crypto.randomUUID().replaceAll('-', '')}`;
  const studyKind = ['workai.recovered.guided_plate.scene.v1', 'workai.murnaghan.guided_plate.scene.v1'].includes(state.model?.case.schema) ? 'research_reference' : state.studyKind || 'plate_smoke';
  try {
    const result = await api(study ? '/api/studies' : '/api/jobs', study ? { job_id: id, model_revision: state.model.revision, kind: studyKind } : { job_id: id, kind: 'smoke' });
    state.jobs = (await api('/api/jobs')).jobs;
    $('#new-job-dialog').close(); $('#study-dialog').close(); location.hash = '#results';
    toast(result.accepted ? '任务已提交，正在查询计算状态。' : result.message);
    await loadJob(id);
  } catch (error) {
    if (error.status && error.status < 500) { toast(error.message); return; }
    // The browser may lose the response after the server submitted successfully.
    // Preserve the chosen ID visibly and query it; do not issue another POST.
    $('#new-job-dialog').close(); $('#study-dialog').close();
    toast(`${error.message} 请查询编号 ${id}`);
    try { state.jobs = (await api('/api/jobs')).jobs; } catch { /* retain current list */ }
    if (!state.jobs.some(record => record.job_id === id)) state.jobs.unshift({ job_id: id, kind: study ? studyKind : 'smoke', submission_state: 'unknown' });
    location.hash = '#results'; await loadJob(id);
  } finally { submitBusy = false; $('#submit-job').disabled = false; $('#submit-study').disabled = false; }
}
document.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button || button.disabled) return;
  if (button.dataset.close) { $(`#${button.dataset.close}`).close(); return; }
  if (button.dataset.job) { location.hash = `#${button.dataset.target}`; void loadJob(button.dataset.job); return; }
  if (button.dataset.query) { void loadJob(button.dataset.query, { showToast: true }); return; }
  if (button.dataset.preview) { void preview(button.dataset.preview); return; }
  if (button.dataset.download) { void download(button.dataset.download, button); return; }
  switch (button.dataset.action) {
    case 'export-offline': {
      if (!state.waveforms?.verified || state.waveforms.job_id !== state.selected || state.loadingWaveforms || state.waveformError) break;
      try {
        const css = Array.from(document.styleSheets).flatMap(sheet => Array.from(sheet.cssRules, rule => rule.cssText)).join('\n');
        const html = buildOfflineReport(state.waveforms, { css });
        const url = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' })), link = document.createElement('a');
        link.href = url; link.download = `${state.selected}-offline.html`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        toast('离线结果已导出，双击 HTML 文件即可查看。');
      } catch (error) { toast(`未能导出：${error.message}`); }
      break;
    }
    case 'export-waveforms': {
      if (!state.waveforms?.verified || state.waveforms.job_id !== state.selected) break;
      const blob = new Blob([JSON.stringify(state.waveforms, null, 2) + '\n'], { type: 'application/json' });
      const url = URL.createObjectURL(blob), link = document.createElement('a');
      link.href = url; link.download = `${state.selected}-analysis.json`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30000); break;
    }
    case 'new-job': $('#new-job-dialog').showModal(); break;
    case 'toggle-metric': state.metric = state.metric === 'reflection' ? 'transmission' : 'reflection'; render(); break;
    case 'reload-waveforms': void loadWaveforms(state.selected, selectionVersion); break;
    case 'reload-series': void loadSeries(state.selected, selectionVersion); break;
    case 'query-current': void loadJob(state.selected, { showToast: true }); break;
    case 'copy-id': if (state.selected) void navigator.clipboard.writeText(state.selected).then(() => toast('已复制完整任务编号。')).catch(() => toast(state.selected)); break;
  }
});
document.addEventListener('change', event => { if (event.target.id === 'job-select') void loadJob(event.target.value); });
document.addEventListener('submit', async event => {
  if (event.target.id !== 'import-form') return;
  event.preventDefault();
  const input = $('#import-id'), button = event.target.querySelector('button'); button.disabled = true;
  try {
    const record = await api('/api/jobs/import', { job_id: input.value.trim() });
    state.jobs = (await api('/api/jobs')).jobs; render();
    await loadJob(record.job_id); toast('已有任务已加入工作区。');
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
});
window.addEventListener('hashchange', render);
document.addEventListener('change', event => {
  if (event.target.id === 'trace-select') { state.waveTrace = event.target.value; render(); }
  if (event.target.id === 'component-select') { state.waveComponent = event.target.value; render(); }
});
const waveformUI = createWaveformUI({ state, e: escape });
const modelUI = createModelUI({ state, api, render, toast, escape, heading, jobTable, submitStudy: () => submitJob(true) });
const solidUI = createSolidUI({state,api,render,toast,e:escape,heading});
$('#refresh').addEventListener('click', () => void (state.view === 'solid' ? solidUI.refresh() : state.view === 'training' ? loadTraining() : refresh()));
$('#submit-job').addEventListener('click', () => void submitJob());
try {
  state.bootstrap = await api('/api/bootstrap');
  const modelData = await api('/api/model'); state.model = modelData.model; state.templates = modelData.templates; state.materials = modelData.materials;
  render(); await refresh();
} catch (error) {
  $('#main').innerHTML = `<div class="error-block"><h2>工作台暂时无法载入</h2><p>${escape(error.message)}</p><p>请确认网站服务运行后重新加载此页面。</p></div>`;
}
