import {harmonicControls,projectHarmonicProbe,analyzeSolidHarmonics} from './solid-harmonics.js';
import {harmonicViewerMarkup} from './solid-harmonic-viewer.js';

export function buildSolidHarmonicReport(input,resources){
  projectHarmonicProbe(input,input.direction);
  const window=input.analysis_window_us??[input.times_us[0],input.times_us.at(-1)];analyzeSolidHarmonics(input,...window);
  const probe={schema:input.schema,study_id:input.study_id,scene:input.scene,point_mm:input.point_mm,direction:input.direction,times_us:input.times_us,analysis_window_us:window,same_mesh_and_time_step:input.same_mesh_and_time_step,numerical_reference:input.numerical_reference??null,
    controls:Object.fromEntries(harmonicControls.map(k=>{const c=input.controls[k];return [k,{job_id:c.job_id,result_sha256:c.result_sha256,xyz_m:c.xyz_m,max_export_roundoff_m:c.max_export_roundoff_m}];}))};
  const nonce=Array.from(crypto.getRandomValues(new Uint8Array(18)),v=>v.toString(16).padStart(2,'0')).join('');
  const source=[resources.data,resources.dispersion,resources.harmonics,resources.propagation,resources.viewer].map(s=>s.replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'')).join('\n');
  const code=source+`\nconst savedProbe=JSON.parse(document.getElementById('harmonic-record').textContent);mountHarmonicViewer(document.getElementById('harmonic-report'),savedProbe);`;
  if(code.toLowerCase().includes('</script'))throw new Error('报告脚本含不允许的闭合标签。');
  const json=JSON.stringify(probe).replace(/</g,'\\u003c');
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'nonce-${nonce}';style-src 'unsafe-inline';img-src data: blob:;connect-src 'none';base-uri 'none';form-action 'none'"><title>workAI · 三维材料非线性对照</title><style>${resources.css}\nbody{min-width:0}main{max-width:1200px;margin:auto}@media print{.panel{break-inside:avoid}}</style></head><body><main><div class="page-heading"><div><div class="eyebrow">WORKAI · MATERIAL NONLINEARITY</div><h1>三维材料非线性五组对照</h1><p>同一网格与时间步 · 真实有限元 · 复数谐波分离</p></div></div><div class="notice">离线报告包含当前一个点位的五组三分量波形，可修改方向和分析窗。其他点位需回工作台从完整场提取。该文件保留导出时核对过的结果摘要，不重新验证服务器回执。</div><div id="harmonic-report">${harmonicViewerMarkup(probe)}</div></main><script id="harmonic-record" type="application/json">${json}</script><script type="module" nonce="${nonce}">${code}</script></body></html>`;
}
