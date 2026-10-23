import {comparePropagationProbes} from './solid-propagation.js';
import {harmonicControls} from './solid-harmonics.js';

export function buildPropagationReport(input,resources){
  comparePropagationProbes(input);
  const probes=input.map(p=>({schema:p.schema,study_id:p.study_id,scene:p.scene,point_mm:p.point_mm,direction:p.direction,times_us:p.times_us,same_mesh_and_time_step:p.same_mesh_and_time_step,numerical_reference:p.numerical_reference??null,waveguide_reference:p.waveguide_reference??null,
    controls:Object.fromEntries(harmonicControls.map(k=>{const c=p.controls[k];return [k,{job_id:c.job_id,result_sha256:c.result_sha256,xyz_m:c.xyz_m,max_export_roundoff_m:c.max_export_roundoff_m}];}))}));
  const nonce=Array.from(crypto.getRandomValues(new Uint8Array(18)),v=>v.toString(16).padStart(2,'0')).join('');
  const source=[resources.data,resources.dispersion,resources.harmonics,resources.propagation,resources.viewer,resources.propagationViewer].map(s=>s.replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'')).join('\n');
  const code=source+`\nmountPropagationViewer(document.getElementById('propagation-report'),JSON.parse(document.getElementById('propagation-record').textContent));`;
  if(code.toLowerCase().includes('</script'))throw new Error('报告脚本含不允许的闭合标签。');
  const json=JSON.stringify(probes).replace(/</g,'\\u003c');
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'nonce-${nonce}';style-src 'unsafe-inline';img-src data: blob:;connect-src 'none';base-uri 'none';form-action 'none'"><title>workAI · 铝板材料二次谐波传播</title><style>${resources.css}\nbody{min-width:0}main{max-width:1200px;margin:auto}.panel{margin-bottom:20px}</style></head><body><main><div class="page-heading"><div><div class="eyebrow">WORKAI · HYPERELASTIC WAVE PROPAGATION</div><h1>铝板材料二次谐波传播</h1><p>真实三维有限元 · 等长移动窗 · 相位与回波检查</p></div></div><div class="notice">包含已保存点位的五组三分量波形，可切换点位、方向、时间窗和频谱。其他坐标需要完整位移场；本报告不包含完整场，也不使用 AI 代理。结果摘要在导出时核对。</div><div id="propagation-report"></div></main><script id="propagation-record" type="application/json">${json}</script><script type="module" nonce="${nonce}">${code}</script></body></html>`;
}
