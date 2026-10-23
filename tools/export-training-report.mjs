import { readFile, writeFile } from 'node:fs/promises';
import { readTrainingStatus } from '../server/training.mjs';
import { renderTraining } from '../public/training-ui.js';
const [root, output, diagnostics] = process.argv.slice(2);
if (!root || !output) throw new Error('Usage: node tools/export-training-report.mjs remote-root new-report.html');
const data = await readTrainingStatus(root); if (!data.available) throw new Error('No training record.');
const escape = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const heading = (a,b) => `<div class="page-heading"><div><div class="eyebrow">WORKAI · PHYSICS-CONSTRAINED TRAINING</div><h1>${escape(a)}</h1><p>${escape(b)}</p></div></div>`;
const css = await readFile(new URL('../public/styles.css', import.meta.url), 'utf8');
const time = new Intl.DateTimeFormat('zh-CN',{timeZone:'UTC',dateStyle:'medium',timeStyle:'medium',hour12:false}).format(new Date());
let figure = '';
if (diagnostics) {
  const png = await readFile(diagnostics);
  if (png.length > 2 * 1024 * 1024 || png.subarray(0,8).toString('hex') !== '89504e470d0a1a0a') throw new Error('Expected a bounded local PNG.');
  figure = data.kind==='mesh_conditioned_3d'?`<section class="panel"><div class="panel-heading"><h2>预测与三维有限元对照</h2></div><div class="panel-body"><img style="width:100%;height:auto" alt="四类未见几何的波形及材料增量对照" src="data:image/png;base64,${png.toString('base64')}"><p class="section-note">完整测试工况未参与标签拟合或权重选择。材料增量来自完整响应减去几何非线性控制。</p></div></section>`:`<section class="panel"><div class="panel-heading"><h2>预测与 COMSOL 对照</h2></div><div class="panel-body"><img style="width:100%;height:auto" alt="训练损失、40 mm 线性及材料增量波形、材料二次谐波随距离的对照" src="data:image/png;base64,${png.toString('base64')}"><p class="section-note">图中的 40 mm 波形未用于标签拟合，但参与检查点选择；该位置不能作为完全独立的测试集。材料增量为完整非线性响应减去几何非线性响应。</p></div></section>`;
}
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>workAI · 物理约束训练记录</title><style>${css}\nbody{min-width:0}main{max-width:1200px}.panel{margin-bottom:24px}.notice{margin-bottom:24px}.panel-heading{flex-wrap:wrap;gap:12px}.mono{overflow-wrap:anywhere}.metric-value{font-size:24px}.metric-value small{font-size:14px}@media print{main{padding:0}.panel,.cards{break-inside:avoid}.table-scroll{overflow:visible}.metric-value{font-size:18px}}</style></head><body><main><p class="muted small-text">训练记录快照 · ${escape(time)}（UTC） · 本页不启动训练或计算</p>${renderTraining({training:data},escape,heading)}${figure}</main></body></html>`;
await writeFile(output,html,{flag:'wx'});console.log(`Saved training report: ${output}`);
