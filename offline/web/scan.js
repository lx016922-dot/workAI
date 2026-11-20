// Parameter scans: which fields of a model can be scanned, the values typed for them, the definition form in the
// settings window and the page of a scan (progress, summary table, curve). The server side is offline/scan.py.
import { escapeHtml } from './inspector-util.js';
import { icon } from './icons.js';
import { sourcesOf, getPath, defectName } from './scene-model.js';
import { fmtBytes, fmtDuration } from './shell.js';

export const MAX_AXES = 3, MAX_VALUES = 100, MAX_CASES = 200;
export const SCAN_STATES = { running: '进行中', paused: '已暂停', completed: '已完成' };
const CASE_PHASES = { queued: '排队', running: '计算中', completed: '已完成', failed: '未完成', paused: '已暂停' };
// Series colours of the curve: the first six slots of a categorical palette checked for colour-blind separation on a white surface.
export const SCAN_SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'];
const NAMES = { frequency_mhz: ['频率', 'MHz'], amplitude_mpa: ['幅值', 'MPa'], cycles: ['周期数', ''], delay_us: ['延迟', 'μs'], phase_deg: ['相位', '°'], radius_mm: ['半径', 'mm'], sigma_mm: ['高斯半径 σ', 'mm'],
  pitch_mm: ['齿距', 'mm'], tooth_width_mm: ['齿宽', 'mm'], length_mm: ['长度', 'mm'], width_mm: ['宽度', 'mm'], thickness_mm: ['厚度', 'mm'], outer_radius_mm: ['外半径', 'mm'], inner_radius_mm: ['内半径', 'mm'],
  hole_radius_mm: ['孔半径', 'mm'], hole_x_mm: ['孔中心 x', 'mm'], hole_y_mm: ['孔中心 y', 'mm'], notch_x_mm: ['缺口中心 x', 'mm'], notch_width_mm: ['缺口宽度', 'mm'], notch_depth_mm: ['缺口深度', 'mm'],
  x_mm: ['位置 x', 'mm'], y_mm: ['位置 y', 'mm'], z_mm: ['位置 z', 'mm'], depth_mm: ['深度', 'mm'], x_start_mm: ['起点 x', 'mm'], x_end_mm: ['终点 x', 'mm'], rho_kg_m3: ['密度 ρ', 'kg/m³'], lambda_pa: ['λ', 'Pa'], mu_pa: ['μ', 'Pa'],
  l_pa: ['l', 'Pa'], m_pa: ['m', 'Pa'], n_pa: ['n', 'Pa'], attenuation_db: ['目标衰减', 'dB'] };
// Columns of the summary table and quantities of the curve: key in a receiver's row, heading, factor from metres.
export const QUANTITIES = [
  { key: 'a2_over_a1_squared_per_m', label: 'A₂/A₁²', unit: '1/m', scale: 1, tip: '相对非线性参数：A₂ 取非线性增量（完整 − 线性）；没有算线性控制时取完整响应' },
  { key: 'a1_m', label: 'A₁', unit: 'nm', scale: 1e9, tip: '完整响应在 f 处的幅值' },
  { key: 'a2_nonlinear_m', label: 'A₂ 非线性增量', unit: 'pm', scale: 1e12, tip: '完整 − 线性 在 2f 处的幅值：几何与材料非线性合起来产生的二次谐波' },
  { key: 'a2_geometric_m', label: 'A₂ 几何', unit: 'pm', scale: 1e12, tip: 'StVK − 线性 在 2f 处的幅值' },
  { key: 'a2_material_m', label: 'A₂ 材料', unit: 'pm', scale: 1e12, tip: '完整 − StVK 在 2f 处的幅值' },
  { key: 'a2_m', label: 'A₂ 完整', unit: 'pm', scale: 1e12, tip: '完整响应在 2f 处的幅值：还含有猝发音自身的线性频谱，不能单独当作非线性的度量' },
];

/** Readable name of a scanned field from its path: `source.frequency_mhz` → 主激励 · 频率 (MHz). */
export function pathLabel(path, scene = null) {
  const keys = path.split('.'), last = keys.at(-1), [name, unit] = NAMES[last] || [last, ''];
  let owner = '';
  if (keys[0] === 'source') owner = scene?.source?.name || '主激励';
  else if (keys[0] === 'sources') owner = scene?.sources?.[+keys[1]]?.name || `激励 ${+keys[1] + 2}`;
  else if (keys[0] === 'defects') owner = scene?.defects?.[+keys[1]] ? defectName(scene.defects[+keys[1]], +keys[1]) : `缺陷 ${+keys[1] + 1}`;
  else if (keys[0] === 'geometry') owner = '几何';
  else if (keys[0] === 'material') owner = '材料';
  else if (keys[0] === 'boundaries') owner = `吸收层 ${+keys[1] + 1}`;
  return { owner, name, unit, text: `${owner ? owner + ' · ' : ''}${name}${unit ? ` (${unit})` : ''}` };
}

/** The numeric fields of a model that can be scanned: [{path, owner, name, unit, value}]. */
export function scanParameters(scene) {
  const rows = [], add = path => { const value = getPath(scene, path); if (typeof value === 'number' && Number.isFinite(value)) rows.push({ path, ...pathLabel(path, scene), value }); };
  sourcesOf(scene).forEach((s, i) => {
    const base = i ? `sources.${i - 1}` : 'source';
    for (const key of ['frequency_mhz', 'amplitude_mpa', 'cycles', 'delay_us']) add(`${base}.${key}`);
    if (s.radius_mm > 0) add(`${base}.radius_mm`);
    for (const key of ['sigma_mm', 'pitch_mm']) add(`${base}.profile.${key}`);
  });
  const g = scene.geometry;
  for (const key of g.kind === 'pipe' ? ['length_mm', 'outer_radius_mm', 'inner_radius_mm'] : ['length_mm', 'width_mm', 'thickness_mm']) add(`geometry.${key}`);
  if (g.kind === 'hole') for (const key of ['hole_radius_mm', 'hole_x_mm', 'hole_y_mm']) add(`geometry.${key}`);
  if (g.kind === 'notch') for (const key of ['notch_x_mm', 'notch_width_mm', 'notch_depth_mm']) add(`geometry.${key}`);
  (scene.defects || []).forEach((d, i) => { for (const key of ['x_mm', 'x_start_mm', 'x_end_mm', 'depth_mm', 'width_mm', 'radius_mm', 'y_mm', 'z_mm']) add(`defects.${i}.${key}`); });
  if (scene.material?.model === 'murnaghan') for (const key of ['l_pa', 'm_pa', 'n_pa', 'lambda_pa', 'mu_pa', 'rho_kg_m3']) add(`material.${key}`);
  (scene.boundaries || []).forEach((b, i) => { if (b.type === 'absorbing') add(`boundaries.${i}.length_mm`); });
  return rows;
}

/** Values typed for a parameter: numbers separated by commas or spaces, and ranges start:step:end (both ends included). */
export function parseValues(text) {
  const values = [];
  for (const token of String(text ?? '').split(/[\s,，;；、]+/).filter(Boolean)) {
    const range = token.split(/[:：]/);
    if (range.length === 3) {
      const [a, step, b] = range.map(Number);
      if (![a, step, b].every(Number.isFinite) || !(step > 0) || b < a) throw Error(`“${token}” 不是“起点:步长:终点”（步长为正，终点不小于起点）。`);
      const n = Math.floor((b - a) / step + 1e-9); if (n >= MAX_VALUES) throw Error(`“${token}” 超过 ${MAX_VALUES} 个取值。`);
      for (let i = 0; i <= n; i++) values.push(+(a + i * step).toPrecision(12));
    } else {
      const v = Number(token); if (range.length !== 1 || !Number.isFinite(v)) throw Error(`“${token}” 不是数值。`);
      values.push(v);
    }
  }
  if (!values.length) throw Error('请输入取值，例如 0.2, 0.3, 0.4 或 0.2:0.05:0.4。');
  if (new Set(values).size !== values.length) throw Error('取值有重复。');
  if (values.length > MAX_VALUES) throw Error(`一个参数最多 ${MAX_VALUES} 个取值。`);
  return values;
}

/** The axes of a scan definition {axes: [{path, text}]} checked against the model: {axes: [{path, values}], errors: [per axis], count, ready}. */
export function scanAxes(scan, scene) {
  const known = new Set(scanParameters(scene).map(p => p.path)), axes = [], errors = []; let count = 1;
  (scan?.axes || []).forEach((axis, i) => {
    let error = '';
    if (!axis.path) error = '请选择参数。';
    else if (!known.has(axis.path)) error = '当前模型里没有这个参数。';
    else if ((scan.axes || []).findIndex(a => a.path === axis.path) !== i) error = '这个参数已经选过。';
    let values = [];
    if (!error) { try { values = parseValues(axis.text); } catch (e) { error = e.message; } }
    if (!error) { const p = pathLabel(axis.path, scene), several = sourcesOf(scene).length > 1 || !axis.path.startsWith('source.');
      axes.push({ path: axis.path, values, label: `${several && p.owner && p.owner !== '几何' ? p.owner + ' ' : ''}${p.name}`.slice(0, 60), ...(p.unit ? { unit: p.unit } : {}) }); count *= values.length; }
    errors.push(error);
  });
  const ready = axes.length > 0 && axes.length === (scan?.axes || []).length;
  return { axes, errors, count: ready ? count : 0, ready: ready && count <= MAX_CASES, tooMany: ready && count > MAX_CASES };
}

/** A scan definition that still fits the model: axes whose parameter no longer exists are dropped. */
export function keepScan(scan, scene) {
  const known = new Set(scanParameters(scene).map(p => p.path));
  return { axes: (scan?.axes || []).filter(a => a && typeof a.path === 'string' && typeof a.text === 'string' && (!a.path || known.has(a.path))).slice(0, MAX_AXES).map(a => ({ path: a.path, text: a.text })) };
}

/** Heading of an axis of a stored scan: the name it was given when the scan was defined, else from its path. */
export const axisText = a => a.label ? `${a.label}${a.unit ? ` (${a.unit})` : ''}` : pathLabel(a.path).text;
const axisName = a => a.label || pathLabel(a.path).name;
/** The values of combination `index` of a grid (the first axis varies slowest). */
export function gridValues(axes, index) { const out = []; for (let i = axes.length - 1; i >= 0; i--) { const n = axes[i].values.length; out.unshift(axes[i].values[index % n]); index = Math.floor(index / n); } return out; }
/** A combination in words: 频率 0.3 MHz，幅值 2 MPa. */
export const caseText = (axes, values) => axes.map((a, i) => `${axisName(a)} ${short(values[i])}${a.unit ? ' ' + a.unit : ''}`).join('，');
const num = v => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? String(+v.toPrecision(4)) : v.toExponential(3));
const short = v => String(+(+v).toPrecision(6));

/** The definition form shown in the settings window. `plan`: the server's answer for these axes, or null while it is asked. */
export function scanFormMarkup(scene, scan, plan, { busy = false, engine = 'network' } = {}) {
  const parameters = scanParameters(scene), checked = scanAxes(scan, scene), owners = [...new Set(parameters.map(p => p.owner))];
  const options = selected => `<option value="">选择参数…</option>` + owners.map(o => `<optgroup label="${escapeHtml(o)}">${parameters.filter(p => p.owner === o).map(p =>
    `<option value="${p.path}" ${p.path === selected ? 'selected' : ''}>${escapeHtml(p.name)}${p.unit ? ` (${escapeHtml(p.unit)})` : ''}</option>`).join('')}</optgroup>`).join('');
  const axes = (scan.axes || []).map((axis, i) => {
    const p = parameters.find(x => x.path === axis.path), error = checked.errors[i];
    let hint = error ? `<small class="scan-err">${escapeHtml(error)}</small>` : '';
    if (!error) { const v = parseValues(axis.text); hint = `<small>${v.length} 个值：${escapeHtml(v.slice(0, 8).map(short).join('、'))}${v.length > 8 ? ' …' : ''}</small>`; }
    return `<div class="scan-axis"><label class="wb-field"><span>参数 ${i + 1}</span><select data-scan-path="${i}">${options(axis.path)}</select></label>
      <label class="wb-field"><span>取值${p ? ` <em>当前 ${short(p.value)}${p.unit ? ' ' + escapeHtml(p.unit) : ''}</em>` : ''}</span><input type="text" data-scan-values="${i}" value="${escapeHtml(axis.text || '')}" placeholder="0.2, 0.3, 0.4 或 0.2:0.05:0.4" spellcheck="false">${hint}</label>
      <button type="button" class="wb-button small" data-scan-remove="${i}" title="删除这个参数">${icon('remove')}</button></div>`;
  }).join('');
  const define = `<section class="wb-section"><h3>扫描的参数</h3>${axes || '<p class="wb-note">还没有参数。点“添加参数”，选一个要变化的量并填入取值。</p>'}
    <div class="st-actions"><button type="button" class="wb-button" data-scan-add ${(scan.axes || []).length >= MAX_AXES ? 'disabled' : ''}>${icon('add')}添加参数</button></div>
    <p class="wb-note">每个组合都是当前模型的一次完整计算（同样的自动网格、时间层和结果页），一个接一个地算。最多 ${MAX_AXES} 个参数、${MAX_CASES} 个组合。
      取值用逗号或空格分隔，也可以写“起点:步长:终点”。扫描板的尺寸时，位于移动的表面或中面上的激励和接收点跟着移动。${engine === 'coarse' ? '当前是快速引擎，每个组合都用它计算。' : ''}</p></section>`;
  let body = '<p class="wb-note">选好参数和取值后，这里列出每个组合的网格和预计耗时。</p>', start = '';
  if (checked.tooMany) body = `<p class="wb-note warn">组合数 ${checked.count} 超过上限 ${MAX_CASES}：请减少取值，或分几次扫描。</p>`;
  else if (checked.ready && !plan) body = `<p class="wb-note">正在检查 ${checked.count} 个组合…</p>`;
  else if (checked.ready && plan?.error) body = `<p class="wb-note warn">${escapeHtml(plan.error)}</p>`;
  else if (checked.ready && plan) {
    const bad = plan.count - plan.valid, rows = plan.cases.slice(0, 60).map(c => `<tr class="${c.ok ? '' : 'bad'}"><td>${c.index + 1}</td><td>${c.values.map(short).join(' · ')}</td>
      <td>${c.ok ? `${(c.nodes || 0).toLocaleString()}` : ''}</td><td>${c.ok ? (c.seconds ? fmtDuration(c.seconds) : '—') : `<span class="scan-err" title="${escapeHtml(c.errors.join('；'))}">${escapeHtml(c.errors[0] || '不能计算')}</span>`}</td></tr>`).join('');
    body = `<table class="wb-table"><tr><th>组合</th><td>${plan.count} 个${bad ? `，其中 <b class="scan-err">${bad} 个不能计算</b>` : '，全部通过检查'}</td></tr>
      <tr><th>预计总耗时</th><td>${fmtDuration(plan.seconds_total)}（按本机速度估计）</td></tr>
      <tr><th>内存 / 磁盘</th><td>最大 ${fmtBytes(plan.memory_bytes_max)} · 记录与结果共约 ${fmtBytes(plan.disk_bytes_total)}${plan.free_bytes !== null && plan.free_bytes !== undefined ? `（可用 ${fmtBytes(plan.free_bytes)}）` : ''}</td></tr></table>
      ${plan.disk_short ? `<p class="wb-note warn">数据目录的可用空间不够放下全部组合：空间用完后的组合会失败（可以之后重算）。${plan.full_records ? '把“求解器配置 → 记录方式”改为“精简记录”可以大幅减小：每个组合只保存接收点的完整波形和动画，汇总表不受影响。' : '请先清理磁盘或减少组合。'}</p>` : ''}
      <div class="scan-plan"><table class="wb-table matrix"><thead><tr><th>#</th><th>${plan.axes.map(a => escapeHtml(axisName(a))).join(' · ')}</th><th>节点</th><th>预计</th></tr></thead><tbody>${rows}</tbody></table>
      ${plan.count > 60 ? `<p class="wb-note">只列出前 60 个组合。</p>` : ''}</div>`;
    start = `<div class="st-actions"><button type="button" class="wb-button primary" data-scan-start ${bad || busy ? 'disabled' : ''}>${icon('compute')}开始扫描（${plan.count} 个组合）</button></div>
      ${bad ? '<p class="wb-note warn">有组合不能计算：把鼠标放在红字上看原因，改掉对应的取值后才能开始。</p>' : busy ? '<p class="wb-note">已有计算或扫描在进行，等它完成或暂停后再开始。</p>' : ''}`;
  }
  return define + `<section class="wb-section"><h3>组合与预计</h3>${body}${start}</section>`;
}

/** Bind the definition form: `onChange()` after an edit of `scan` (re-render and ask for the plan), `onStart()` for the start button. */
export function bindScanForm(root, scan, onChange, onStart) {
  root.querySelectorAll('[data-scan-path]').forEach(el => el.addEventListener('change', () => { scan.axes[+el.dataset.scanPath].path = el.value; onChange(); }));
  root.querySelectorAll('[data-scan-values]').forEach(el => el.addEventListener('change', () => { scan.axes[+el.dataset.scanValues].text = el.value.trim(); onChange(); }));
  root.querySelectorAll('[data-scan-remove]').forEach(el => el.addEventListener('click', () => { scan.axes.splice(+el.dataset.scanRemove, 1); onChange(); }));
  root.querySelector('[data-scan-add]')?.addEventListener('click', () => { if (scan.axes.length < MAX_AXES) { scan.axes.push({ path: '', text: '' }); onChange(); } });
  root.querySelector('[data-scan-start]')?.addEventListener('click', onStart);
}

/** One or more curves over one axis: series [{name, points: [[x, y, tip]]}]. Lines 2 px, markers with a surface ring, a legend for two or more.
 *  A series may give its own `colour`, `markers: false` with `dash: true` for a line that is not data (a fit), and
 *  `small: true` for many points (small dots without a ring). */
export function seriesChart(series, labelX, labelY, width = 900) {
  const height = 210, left = 70, right = 20, top = 28, bottom = 40, all = series.flatMap(s => s.points);
  if (!all.length) return '';
  let xmin = Math.min(...all.map(p => p[0])), xmax = Math.max(...all.map(p => p[0])), ymin = Math.min(0, ...all.map(p => p[1])), ymax = Math.max(0, ...all.map(p => p[1]));
  if (xmax === xmin) { xmin -= .5; xmax += .5; } if (ymax - ymin < 1e-300) ymax = ymin + 1;
  const sx = v => left + (v - xmin) / (xmax - xmin) * (width - left - right), sy = v => height - bottom - (v - ymin) / (ymax - ymin) * (height - top - bottom);
  const ticks = Array.from({ length: 5 }, (_, i) => { const xv = xmin + (xmax - xmin) * i / 4, yv = ymin + (ymax - ymin) * i / 4;
    return `<path d="M${left} ${sy(yv).toFixed(1)}H${width - right}" stroke="#e6edef"/><text x="${left - 8}" y="${(sy(yv) + 4).toFixed(1)}" text-anchor="end">${num(yv)}</text><text x="${sx(xv).toFixed(1)}" y="${height - bottom + 18}" text-anchor="middle">${num(xv)}</text>`; }).join('');
  const colourOf = (s, k) => s.colour || SCAN_SERIES[k % SCAN_SERIES.length];
  const marks = series.map((s, k) => { const colour = colourOf(s, k), points = [...s.points].sort((a, b) => a[0] - b[0]);
    return `<polyline points="${points.map(p => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ')}" fill="none" stroke="${colour}" stroke-width="2" stroke-linejoin="round"${s.dash ? ' stroke-dasharray="6 4"' : ''}/>` +
      (s.markers === false ? [] : points).map(p => `<g><circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" ${s.small ? 'r="2.2"' : 'r="4.5" stroke="#fff" stroke-width="2"'} fill="${colour}"/><circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="${s.small ? 5 : 11}" fill="transparent"><title>${escapeHtml(p[2] || `${num(p[0])}：${num(p[1])}`)}</title></circle></g>`).join(''); }).join('');
  const legend = series.length > 1 ? `<ul class="scan-legend">${series.map((s, k) => `<li><i style="background:${colourOf(s, k)}"></i>${escapeHtml(s.name)}</li>`).join('')}</ul>` : '';
  return `${legend}<svg class="solid-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(labelY)}随${escapeHtml(labelX)}的变化"><g font-size="11" fill="#748995">${ticks}<text x="${width / 2}" y="${height - 4}" text-anchor="middle">${escapeHtml(labelX)}</text><text x="7" y="11">${escapeHtml(labelY)}</text></g>${marks}</svg>`;
}

/** Series of a quantity over the first axis that has two or more values; one series per combination of the other axes (at most six). */
export function scanSeries(table, receiver, quantity, wanted = null) {
  // the horizontal axis: the one asked for, else the axis with the most values
  const several = table.axes.map((a, i) => i).filter(i => table.axes[i].values.length > 1);
  const axis = several.includes(wanted) ? wanted : several.sort((a, b) => table.axes[b].values.length - table.axes[a].values.length)[0] ?? -1;
  if (axis < 0) return { axis: -1, series: [], dropped: 0, choices: [] };
  const q = QUANTITIES.find(x => x.key === quantity) || QUANTITIES[0], groups = new Map();
  for (const row of table.rows) {
    const value = row.receivers?.[receiver]?.[q.key]; if (value === null || value === undefined) continue;
    const others = row.values.filter((_, i) => i !== axis), key = JSON.stringify(others);
    if (!groups.has(key)) groups.set(key, { name: caseText(table.axes.filter((_, i) => i !== axis), others) || q.label, points: [] });
    groups.get(key).points.push([row.values[axis], value * q.scale, `${caseText([table.axes[axis]], [row.values[axis]])}：${q.label} ${num(value * q.scale)} ${q.unit}`]);
  }
  const series = [...groups.values()].filter(s => s.points.length);
  return { axis, series: series.slice(0, SCAN_SERIES.length), dropped: Math.max(0, series.length - SCAN_SERIES.length), quantity: q, choices: [...several].sort((a, b) => a - b) };
}

/** The page of a scan in the results window. `view`: {receiver, quantity, window: [start, end] | null}. */
export function scanPageMarkup(detail, table, view = {}) {
  const c = detail.counts, done = c.completed + c.failed, pct = detail.count ? 100 * done / detail.count : 0, current = (detail.cases || []).find(x => x.phase === 'running');
  const head = `<div class="wb-result-title"><h2>${icon('scan')}${escapeHtml(detail.name)}</h2><span class="wb-phase ${detail.state === 'completed' ? (c.failed ? 'paused' : 'completed') : detail.state}">${SCAN_STATES[detail.state] || escapeHtml(detail.state)}</span>
      <span class="muted">${c.completed} / ${detail.count} 个组合已完成${c.failed ? `，${c.failed} 个未完成` : ''}${c.queued ? `，${c.queued} 个排队` : ''}${detail.message && detail.state !== 'completed' ? ' · ' + escapeHtml(detail.message) : ''}</span></div>
    <p class="muted small-text">参数：${detail.axes.map(a => `${escapeHtml(axisText(a))} × ${a.values.length}`).join('　')}${current ? `　|　正在计算第 ${current.index + 1} 个组合（${escapeHtml(caseText(detail.axes, current.values))}）` : ''}</p>
    ${detail.state === 'completed' && !c.failed ? '' : `<progress max="100" value="${pct.toFixed(1)}"></progress>`}
    <div class="wb-result-tools">
      ${detail.state === 'running' ? `<button type="button" class="wb-button" data-scan-act="pause">${icon('pause')}暂停扫描</button>` : ''}
      ${detail.state === 'paused' ? `<button type="button" class="wb-button primary" data-scan-act="resume">${icon('resume')}继续扫描</button>` : ''}
      ${c.failed ? `<button type="button" class="wb-button" data-scan-act="retry">${icon('compute')}重算未完成的 ${c.failed} 个组合</button>` : ''}
      ${c.completed ? `<a class="wb-button" data-scan-csv href="/api/scans/${detail.scan_id}/summary.csv${view.window ? `?start_us=${view.window[0]}&end_us=${view.window[1]}` : ''}">${icon('export')}导出汇总表 CSV</a>` : ''}</div>`;
  if (!table) return { head, body: '<p class="wb-empty">正在读取汇总表…</p>' };
  const first = table.rows.find(r => r.receivers?.length), receivers = first ? first.receivers : [], receiver = Math.min(view.receiver || 0, Math.max(0, receivers.length - 1));
  const quantity = QUANTITIES.find(q => q.key === view.quantity) || QUANTITIES[0], basis = first?.receivers[receiver]?.a2_basis, curve = scanSeries(table, receiver, quantity.key, view.axis);
  const controls = `<div class="scan-controls">
      <label class="wb-field inline"><span>接收点</span><select id="scan-receiver">${receivers.map((r, i) => `<option value="${i}" ${i === receiver ? 'selected' : ''}>${escapeHtml(r.name)}（${r.point_mm.map(short).join(', ')}）</option>`).join('') || '<option>—</option>'}</select></label>
      <label class="wb-field inline"><span>曲线</span><select id="scan-quantity">${QUANTITIES.map(q => `<option value="${q.key}" ${q.key === quantity.key ? 'selected' : ''}>${escapeHtml(q.label)} (${q.unit})</option>`).join('')}</select></label>
      ${curve.choices.length > 1 ? `<label class="wb-field inline"><span>横轴</span><select id="scan-axis">${curve.choices.map(i => `<option value="${i}" ${i === curve.axis ? 'selected' : ''}>${escapeHtml(axisText(table.axes[i]))}</option>`).join('')}</select></label>` : ''}
      <label class="wb-field inline"><span>分析时间窗 (μs)</span><input type="number" step="any" id="scan-start" value="${view.window ? view.window[0] : ''}" placeholder="起点"><input type="number" step="any" id="scan-end" value="${view.window ? view.window[1] : ''}" placeholder="终点">
        <button type="button" class="wb-button small" id="scan-window">应用</button>${view.window ? '<button type="button" class="wb-button small" id="scan-window-clear">整段记录</button>' : ''}</label></div>`;
  const cells = r => QUANTITIES.map(q => `<td title="${escapeHtml(q.tip)}">${num(r?.[q.key] === null || r?.[q.key] === undefined ? null : r[q.key] * q.scale)}</td>`).join('');
  const rows = table.rows.map(row => { const r = row.receivers?.[receiver];
    return `<tr class="${row.phase}${row.job_id ? ' open' : ''}" ${row.job_id ? `data-job="${row.job_id}" title="点击打开这个组合的结果"` : ''}><td>${row.index + 1}</td>${row.values.map(v => `<td>${short(v)}</td>`).join('')}
      <td><span class="wb-phase ${row.phase}">${CASE_PHASES[row.phase] || escapeHtml(row.phase)}</span></td>${r ? cells(r) : `<td colspan="${QUANTITIES.length}" class="scan-msg">${escapeHtml(row.error || row.message || '')}</td>`}
      <td>${row.nodes ? row.nodes.toLocaleString() : ''}</td><td>${row.elapsed_s ? fmtDuration(row.elapsed_s) : ''}</td></tr>`; }).join('');
  const chart = curve.series.some(s => s.points.length > 1) ? `<h3 class="scan-h">${escapeHtml(quantity.label)} 随 ${escapeHtml(axisText(table.axes[curve.axis]))} 的变化</h3>
      ${seriesChart(curve.series, axisText(table.axes[curve.axis]), `${quantity.label} (${quantity.unit})`)}${curve.dropped ? `<p class="wb-note">其余 ${curve.dropped} 条曲线没有画出；全部数值在下面的表里和 CSV 中。</p>` : ''}` : '';
  const note = `<p class="wb-note">${escapeHtml(table.note)}${basis === 'full' ? ' <b>这次扫描没有算线性控制，表中的 A₂/A₁² 用的是完整响应。</b>' : ''}</p>`;
  return { head, body: `${controls}${chart}<div class="scan-wrap"><table class="wb-table matrix scan-table"><thead><tr><th>#</th>${table.axes.map(a => `<th title="${escapeHtml(a.path)}">${escapeHtml(axisText(a))}</th>`).join('')}<th>状态</th>
      ${QUANTITIES.map(q => `<th title="${escapeHtml(q.tip)}">${escapeHtml(q.label)} (${q.unit})</th>`).join('')}<th>节点</th><th>用时</th></tr></thead><tbody>${rows}</tbody></table></div>${note}` };
}
