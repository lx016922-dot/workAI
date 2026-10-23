import { createWaveformUI } from './waveform-ui.js';
import { renderHarmonics } from './nonlinear-ui.js';

const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const jsonForHTML = value => JSON.stringify(value).replace(/[<>&\u2028\u2029]/g, c => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`);

// Export only the result contract: never the page's bootstrap, connection or session.
function snapshotResult(data) {
  if (data?.verified !== true || !/^comsol-[a-f0-9]{32}$/.test(data.job_id) ||
      data.receipt?.job_id !== data.job_id || data.receipt?.status !== 'completed' ||
      !/^[a-f0-9]{64}$/.test(data.source_sha256) || data.receipt.source_sha256 !== data.source_sha256) {
    throw new Error('仅支持导出已校验且身份一致的完成任务。');
  }
  let keys;
  for (const level of ['coarse', 'fine', 'timecheck']) {
    const traces = data.levels?.[level];
    if (!Array.isArray(traces) || !traces.length || traces.length > 100) throw new Error('三组位移数据必须完整。');
    const current = traces.map(t => `${t.shot}:${t.rx}`);
    if (new Set(current).size !== current.length || (keys && JSON.stringify(keys) !== JSON.stringify(current))) throw new Error('三组对照或接收点不一致。');
    keys = current;
    for (const t of traces) {
      if (![t.shot, t.rx].every(v => Number.isInteger(v) && v >= 0) || !Number.isFinite(t.x_mm) || !Array.isArray(t.points) || t.points.length < 2 || t.points.length > 4000) throw new Error('位移数据格式不完整。');
      for (let i = 0; i < t.points.length; i++) {
        const p = t.points[i];
        if (!['time_us', 'u_nm', 'v_nm', 'w_nm'].every(k => Number.isFinite(p[k])) || p.time_us < 0 || (i && p.time_us <= t.points[i - 1].time_us)) throw new Error('位移数据含无效数值或时间。');
      }
      if (level !== 'coarse') {
        const other = data.levels.coarse.find(v => v.shot === t.shot && v.rx === t.rx);
        if (other.x_mm !== t.x_mm || other.points.length !== t.points.length || t.points.some((p, i) => p.time_us !== other.points[i].time_us)) throw new Error('三组采样时间或点位不一致。');
      }
    }
  }
  const result = Object.fromEntries(['verified', 'job_id', 'parameters', 'receipt', 'scene', 'constitutive', 'controls', 'source_sha256', 'levels', 'artifacts', 'harmonics'].map(k => [k, data[k]]));
  if (JSON.stringify(result).length > 8 * 1024 * 1024) throw new Error('结果过大，暂不支持单页导出。');
  return result;
}

/** A read-only single HTML file. No server, network requests or credentials. */
export function buildOfflineReport(data, { css = '', generatedAt = new Date().toISOString() } = {}) {
  const result = snapshotResult(data);
  if (!Number.isFinite(Date.parse(generatedAt)) || /<\/style/i.test(css)) throw new Error('导出设置无效。');
  const nonce = Array.from(crypto.getRandomValues(new Uint8Array(18)), v => v.toString(16).padStart(2, '0')).join('');
  const time = value => Number.isFinite(Date.parse(value)) ? new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Taipei', dateStyle: 'medium', timeStyle: 'medium', hour12: false }).format(new Date(value)) + '（台北时间）' : '未记录';
  const state = { waveforms: result, waveComponent: result.constitutive === 'murnaghan_green_lagrange' ? 'u_nm' : 'v_nm' };
  const initial = createWaveformUI({ state, e: escapeHTML }).render();
  const g = result.scene.geometry, source = result.scene.source;
  const extraCSS = `
    body{min-width:0}.offline-main{max-width:1180px;margin:auto;padding:30px 24px}
    .offline-header{margin-bottom:24px}.offline-header h1{margin:8px 0}.offline-header p{color:#657984}
    .offline-actions{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}.offline-note{border-left:3px solid #148778;padding:12px 16px;background:#edf7f4;margin:18px 0;line-height:1.8}
    .offline-meta{display:flex;flex-wrap:wrap;gap:10px 24px;margin:14px 0;color:#526975}
    .offline-main pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;padding:16px;background:#f5f8f9}
    .offline-main code,.offline-main td{overflow-wrap:anywhere}.offline-main details>summary{cursor:pointer;padding:14px 0}
    .offline-main .panel{margin-bottom:22px}.offline-main .toolbar{flex-wrap:wrap}.offline-main select{max-width:100%}
    .offline-caption{font-size:12px;line-height:1.8;color:#71858e}.offline-footer{margin:24px 0;color:#71858e;font-size:12px}
    @media(max-width:600px){.offline-main{padding:20px 12px}.offline-header h1{font-size:24px}.offline-main .panel-heading{align-items:flex-start;gap:12px;flex-wrap:wrap}.offline-main .metadata{display:block}}
    @media print{.offline-actions,.offline-main button{display:none!important}.offline-main{max-width:none;padding:0}.offline-main .panel{break-inside:avoid}.offline-main .table-scroll{overflow:visible}.offline-main pre{max-height:none}}
  `;
  const script = `
    const renderHarmonics = ${renderHarmonics.toString()};
    const createWaveformUI = ${createWaveformUI.toString()};
    const escapeHTML = ${escapeHTML.toString()};
    const result = JSON.parse(document.getElementById('workai-result').textContent);
    const state = {waveforms:result,waveComponent:result.constitutive==='murnaghan_green_lagrange'?'u_nm':'v_nm'};
    const ui = createWaveformUI({state,e:escapeHTML});
    document.addEventListener('change', event => {
      if(event.target.id==='trace-select') state.waveTrace=event.target.value;
      else if(event.target.id==='component-select') state.waveComponent=event.target.value;
      else return;
      document.getElementById('result-view').innerHTML=ui.render();
    });
    document.addEventListener('click', event => {
      const button=event.target.closest('button'); if(!button) return;
      if(button.dataset.action==='print-report') window.print();
      if(button.dataset.action==='export-waveforms') {
        const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
        const link=document.createElement('a');link.href=url;link.download=result.job_id+'-analysis.json';link.click();
        setTimeout(()=>URL.revokeObjectURL(url),30000);
      }
    });
  `;
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-${nonce}'; style-src 'nonce-${nonce}'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>workAI · 离线研究结果 · ${escapeHTML(result.job_id)}</title><style nonce="${nonce}">${css}\n${extraCSS}</style></head>
<body><main class="offline-main"><header class="offline-header"><div class="eyebrow">WORKAI · RESEARCH RESULTS</div><h1>导波研究结果</h1><p>离线结果副本 · 可切换接收位置、本构对照与位移分量</p>
<div class="offline-meta"><span>板截面 ${escapeHTML(g.length_mm)} × ${escapeHTML(g.thickness_mm)} mm</span><span>载频 ${escapeHTML(source.frequency_mhz)} MHz</span><span>基准应力 ${escapeHTML(source.amplitude_pa / 1e6)} MPa</span><span>${escapeHTML(source.cycles)} 周期</span></div>
<p class="mono">${escapeHTML(result.job_id)}</p><p class="offline-caption">求解完成：${escapeHTML(time(result.receipt.completed_at_utc))}<br>副本生成：${escapeHTML(time(generatedAt))}</p>
<div class="offline-actions"><button class="button primary" data-action="export-waveforms">下载波形与分析 JSON</button><button class="button secondary" data-action="print-report">打印 / 保存 PDF</button></div>
<div class="offline-note">本页保存已完成任务的结果，不连接服务器或启动计算。导出前的文件签名、摘要和模型身份由工作台校验；本页不重新验签。数值检查状态保留原分析结论，文献参考材料尚未实验标定。</div></header>
<div id="result-view">${initial}</div>
<section class="panel"><div class="panel-heading"><h2>模型参数与数据来源</h2></div><div class="panel-body"><p class="offline-caption">源码 SHA-256：<code>${escapeHTML(result.source_sha256)}</code><br>场景 SHA-256：<code>${escapeHTML(result.receipt.scene_sha256)}</code></p>
<details><summary>查看本任务参数（只读）</summary><pre>${escapeHTML(JSON.stringify(result.scene, null, 2))}</pre></details>
<details><summary>文件摘要与计算回执</summary><pre>${escapeHTML(JSON.stringify({ artifacts: result.artifacts, receipt: result.receipt }, null, 2))}</pre></details></div></section>
<p class="offline-footer">本页可保存并直接用浏览器打开。页面不含登录信息、客户端私钥或远程连接配置。</p>
<noscript><p class="offline-note">当前浏览器未运行脚本，显示默认对照的静态结果；切换波形需要启用本地脚本。</p></noscript>
</main><script id="workai-result" type="application/json" nonce="${nonce}">${jsonForHTML(result)}</script><script nonce="${nonce}">${script}</script></body></html>`;
}
