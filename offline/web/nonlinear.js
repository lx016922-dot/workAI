// Post-processing for nonlinear guided waves on the page of a finished computation: the second harmonic along a line
// with the fit of A₂/A₁², the theoretical dispersion curves and the f–k spectrum drawn over them, and the convergence
// check. The numbers come from the server (offline/nonlinear.py, offline/guided.py, offline/scan.py); this module holds
// the forms, the tables and the charts.
import { escapeHtml } from './inspector-util.js';
import { seriesChart, SCAN_SERIES } from './scan.js';
import { fmtBytes, fmtDuration } from './shell.js';

// One colour per family of modes, the same in every chart (the first slots of the validated categorical palette).
export const FAMILY_COLOURS = { S: SCAN_SERIES[0], Sq: SCAN_SERIES[0], L: SCAN_SERIES[0], A: SCAN_SERIES[1], Aq: SCAN_SERIES[1], SH: SCAN_SERIES[2], M: SCAN_SERIES[3] };
const PART_LABELS = { full: '完整位移', linear: '线性位移', nonlinear: '非线性增量（完整 − 线性）', geometric: '几何非线性增量', material: '材料非线性增量', stvk: 'StVK 位移' };
const INK = '#173147', MUTED = '#748995', GRID = '#e6edef';
const SCAN_WORDS = { running: '进行中', paused: '已暂停', completed: '已完成' };

const num = (v, digits = 4) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? String(+v.toPrecision(digits)) : v.toExponential(digits - 1));
const pct = (v, digits = 2) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v === 0 ? '0%' : `${v >= 0 ? '+' : '−'}${Math.abs(100 * v) < .5 * 10 ** -digits ? Math.abs(100 * v).toExponential(1) : Math.abs(100 * v).toFixed(digits)}%`);
const size = (v, digits = 2) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : Math.abs(v) < 5e-5 ? `${(100 * Math.abs(v)).toExponential(1)}%` : `${(100 * Math.abs(v)).toFixed(digits)}%`);

/** Round tick values covering [lo, hi]: steps of 1, 2 or 5 times a power of ten, about `count` of them. */
export function ticksFor(lo, hi, count = 5) {
  const raw = (hi - lo) / count, power = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map(m => m * power).find(v => v >= raw - 1e-12 * power), out = [];
  for (let i = Math.ceil(lo / step - 1e-9); i * step <= hi + 1e-9 * step; i++) out.push(+(i * step).toPrecision(12));
  return out;
}
const FAMILY_ORDER = ['S', 'Sq', 'L', 'A', 'Aq', 'SH', 'M'];
const familyRank = key => { const i = FAMILY_ORDER.indexOf(key); return i < 0 ? FAMILY_ORDER.length : i; };

// ---------- requests from the forms ----------
function numbers(texts, what) {
  if (texts.some(t => String(t).trim() === '' || !Number.isFinite(Number(t)))) throw new Error(`请填写完整的${what}。`);
  return texts.map(Number);
}
const optional = (text, what) => { if (String(text ?? '').trim() === '') return null; const v = Number(text); if (!Number.isFinite(v)) throw new Error(`${what}需要是数值。`); return v; };

/** The points and the receiving direction of an analysis from the form: {points, direction}. */
export function lineBody(form) {
  const direction = numbers(form.direction, '接收方向');
  if (!direction.some(v => v !== 0)) throw new Error('接收方向不能为零向量。');
  if (form.mode === 'receivers') return { points: { mode: 'receivers' }, direction };
  const count = optional(form.count, '采样点数');
  if (count !== null && (!Number.isInteger(count) || count < 2)) throw new Error('采样点数需要是不小于 2 的整数；留空则自动选取。');
  return { points: { mode: 'line', start_mm: numbers(form.start, '起点坐标'), end_mm: numbers(form.end, '终点坐标'), count: count ?? 'auto' }, direction };
}

/** The request of the accumulation analysis from the form. */
export function accumulationBody(form) {
  const body = lineBody(form);
  if (form.window === 'fixed') {
    const [start, end] = numbers([form.windowStart, form.windowEnd], '时间窗起点和终点');
    if (!(start < end)) throw new Error('时间窗的起点需要早于终点。');
    body.window = { mode: 'fixed', start_us: start, end_us: end };
  } else {
    const speed = optional(form.speed, '跟随速度'), length = optional(form.length, '时间窗长度');
    body.window = { mode: 'follow', ...(speed === null ? {} : { speed_m_s: speed }), ...(length === null ? {} : { length_us: length }) };
  }
  const from = optional(form.fitFrom, '拟合范围'), to = optional(form.fitTo, '拟合范围');
  if ((from === null) !== (to === null)) throw new Error('拟合范围需要同时填写起点和终点，或者都留空（用全部点）。');
  if (from !== null) { if (!(from < to)) throw new Error('拟合范围的起点需要小于终点。'); body.fit_mm = [from, to]; }
  return body;
}

/** The request of the f–k spectrum from the form. */
export function fkBody(form) {
  const body = { ...lineBody(form), part: form.part || 'full' };
  const start = optional(form.rangeStart, '时间范围'), end = optional(form.rangeEnd, '时间范围'), top = optional(form.fmax, '频率上限');
  if ((start === null) !== (end === null)) throw new Error('时间范围需要同时填写起点和终点，或者都留空（整段记录）。');
  if (start !== null) { if (!(start < end)) throw new Error('时间范围的起点需要早于终点。'); body.range_us = [start, end]; }
  if (top !== null) body.frequency_max_mhz = top;
  return body;
}

/** Angle (degrees from x) of the line of an analysis in the plane of the plate, or null when it has a thickness component. */
export function lineAngle(e) { return Math.abs(e[2]) > 1e-6 ? null : +(Math.atan2(e[1], e[0]) * 180 / Math.PI).toFixed(4); }

// ---------- accumulation ----------
const usable = a => a.rows.filter(r => r.usable);
const inFit = (a, r) => r.usable && r.distance_mm >= a.fit.range_mm[0] - 1e-9 && r.distance_mm <= a.fit.range_mm[1] + 1e-9;

/** The three charts of an accumulation analysis: A₂/A₁² with its fit, A₁, and the second harmonics. */
export function accumulationCharts(a, width = 900) {
  const rows = usable(a), fit = a.fit.ratio, [x0, x1] = a.fit.range_mm, where = r => `${r.name ? r.name + ' · ' : ''}${num(r.distance_mm)} mm`, small = rows.length > 40, half = width >= 860 ? Math.floor((width - 18) / 2) : width;
  const ratio = seriesChart([
    { name: 'A₂/A₁²（各点）', small, points: rows.map(r => [r.distance_mm, r.ratio_per_m, `${where(r)}：A₂/A₁² ${num(r.ratio_per_m)} 1/m${inFit(a, r) ? '' : '（未参与拟合）'}`]) },
    { name: `直线拟合（${num(x0)}–${num(x1)} mm）`, points: [[x0, fit.intercept_per_m + fit.slope_per_m2 * x0 * 1e-3], [x1, fit.intercept_per_m + fit.slope_per_m2 * x1 * 1e-3]], markers: false, dash: true, colour: INK }],
    '沿程距离 (mm)', 'A₂/A₁² (1/m)', width);
  const first = seriesChart([{ name: 'A₁', small, points: rows.map(r => [r.distance_mm, r.a1_m * 1e9, `${where(r)}：A₁ ${num(r.a1_m * 1e9)} nm`]) }], '沿程距离 (mm)', 'A₁ 完整响应 (nm)', half);
  const parts = [['a2_nonlinear_m', 'A₂ 非线性增量'], ['a2_geometric_m', 'A₂ 几何'], ['a2_material_m', 'A₂ 材料'], ['a2_m', 'A₂ 完整响应']].filter(([key]) => rows.some(r => Number.isFinite(r[key])));
  const second = seriesChart(parts.map(([key, name]) => ({ name, small, points: rows.filter(r => Number.isFinite(r[key])).map(r => [r.distance_mm, r[key] * 1e12, `${where(r)}：${name} ${num(r[key] * 1e12)} pm`]) })), '沿程距离 (mm)', '2f 幅值 (pm)', half);
  return { ratio, first, second };
}

function waveRow(label, w) {
  const t = w.theory, forced = w.forced;
  return `<tr><th>${label}</th><td>${w.reliable ? `k = ${num(w.wavenumber_rad_mm, 5)} rad/mm，相速度 ${w.phase_velocity_m_s ? num(w.phase_velocity_m_s, 5) + ' m/s' : '—'}（相位直线拟合 R² ${w.r_squared === null ? '—' : w.r_squared.toFixed(5)}${t ? '' : `；点距只能唯一确定 |k| &lt; ${num(w.unambiguous_below_rad_mm, 3)} rad/mm 的波数`}）` : '点距过大，波数不可靠'}
    ${forced ? `<br><span class="muted">受迫波 2k(f) = ${num(forced.wavenumber_rad_mm, 5)} rad/mm；实测相差 ${pct(forced.difference, 3)}</span>` : ''}
    ${t ? `<br><span class="muted">${t.matched ? '理论' : '最接近的理论模态'} ${escapeHtml(t.mode)}：k = ${num(t.wavenumber_rad_mm, 5)} rad/mm，相速度 ${num(t.phase_velocity_m_s, 5)} m/s；实测相差 ${pct(t.difference, 3)}${t.matched ? '' : '（相差超过 2%，不是这个模态）'}</span>` : ''}</td></tr>`;
}

/** Results of an accumulation analysis: the fit, the wavenumbers, the charts, the notes and the table of points. */
export function accumulationMarkup(a, width = 900) {
  const f = a.fit, w = a.window, charts = accumulationCharts(a, width), mode = f.f.theory?.matched ? f.f.theory : null, pair = f.pair;
  const basis = a.a2_basis === 'nonlinear' ? '非线性增量（完整 − 线性）' : '完整响应';
  const summary = `<table class="wb-table nl-summary">
    <tr><th>A₂/A₁² 的斜率</th><td><b>${num(f.ratio.slope_per_m2)} m⁻²</b>（每毫米 ${num(f.ratio.slope_per_m2 * 1e-3)} m⁻¹）；截距 ${num(f.ratio.intercept_per_m)} m⁻¹；R² ${f.ratio.r_squared === null ? '—' : f.ratio.r_squared.toFixed(4)}；
      拟合 ${num(f.range_mm[0])}–${num(f.range_mm[1])} mm 内的 ${f.points} 个点；A₂ 取${basis}</td></tr>
    <tr><th>A₂ 的斜率</th><td>${num(f.a2.slope_m_per_m * 1e9)} pm/mm；R² ${f.a2.r_squared === null ? '—' : f.a2.r_squared.toFixed(4)}；拟合范围内 A₁ 变化 ${(100 * f.a1_variation).toFixed(1)}%</td></tr>
    ${waveRow('f 处的波数', f.f)}${waveRow('2f 处的相位斜率', f['2f'])}
    <tr><th>相位匹配（理论）</th><td>${pair ? `${escapeHtml(pair.primary)} → ${escapeHtml(pair.secondary)}：相速度失配 ${pct(pair.mismatch, 3)}，Δk = k(2f) − 2k(f) = ${num(pair.delta_k_rad_mm, 3)} rad/mm${pair.coherence_length_mm === null ? '（匹配）' : `，π/|Δk| = ${num(pair.coherence_length_mm, 3)} mm`}
      <br><span class="muted">按实测到的一次模态 ${escapeHtml(pair.primary)} 查理论；二次谐波在 π/|Δk| 处最大，之后回落。全部模态对见“频散与 f–k 谱”。</span>` : '没有认出一次模态（非平板，或实测波数不在任何理论模态上）'}</td></tr>
    <tr><th>群速度</th><td>${a.group.velocity_m_s ? `${num(a.group.velocity_m_s, 5)} m/s（包络到达时刻对距离的直线拟合，R² ${a.group.r_squared === null ? '—' : a.group.r_squared.toFixed(4)}）` : '没有量出沿这条线向前传播的波包'}
      ${mode ? `<br><span class="muted">理论 ${escapeHtml(mode.mode)}：${num(mode.group_velocity_m_s, 5)} m/s${a.group.velocity_m_s ? `；实测相差 ${pct(a.group.velocity_m_s / mode.group_velocity_m_s - 1)}` : ''}</span>` : ''}</td></tr>
    <tr><th>时间窗</th><td>${w.mode === 'fixed' ? `各点同一时间窗 ${num(w.start_us)}–${num(w.end_us)} μs` : `跟随波包：长 ${num(w.length_us)} μs，第一个点以 ${num(w.centre_first_us)} μs 为中心，按 ${num(w.speed_m_s, 5)} m/s（${w.speed_source === 'measured' ? '实测群速度' : '手动填写'}）随距离后移`}</td></tr></table>`;
  const notes = a.notes.length ? `<ul class="nl-notes">${a.notes.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul>` : '';
  const extra = [['a2_nonlinear_m', 'A₂ 非线性 / pm'], ['a2_geometric_m', 'A₂ 几何 / pm'], ['a2_material_m', 'A₂ 材料 / pm']].filter(([key]) => a.rows.some(r => Number.isFinite(r[key])));
  const table = `<div class="table-scroll nl-rows"><table><thead><tr><th>点</th><th>坐标 / mm</th><th>沿程 / mm</th><th>时间窗 / μs</th><th>A₁ / nm</th>${extra.map(([, h]) => `<th>${h}</th>`).join('')}<th>A₂ 完整 / pm</th><th>A₂/A₁² / m⁻¹</th><th>f 相位 / °</th><th>2f 相位 / °</th></tr></thead><tbody>
    ${a.rows.map(r => `<tr class="${inFit(a, r) ? '' : 'nl-out'}"><td>${escapeHtml(r.name || String(r.index))}</td><td>${r.point_mm.map(v => num(v, 5)).join(', ')}</td><td>${num(r.distance_mm, 5)}</td><td>${r.window_us.map(v => v.toFixed(2)).join('–')}</td>
      ${r.usable ? `<td>${num(r.a1_m * 1e9, 5)}</td>${extra.map(([key]) => `<td>${num(r[key] * 1e12, 5)}</td>`).join('')}<td>${num(r.a2_m * 1e12, 5)}</td><td>${num(r.ratio_per_m, 5)}</td><td>${num(r.phase1_deg, 5)}</td><td>${num(r.phase2_deg, 5)}</td>`
        : `<td colspan="${5 + extra.length}">${escapeHtml(r.reason || '不可用')}</td>`}</tr>`).join('')}</tbody></table></div>`;
  return `${summary}${notes}<h3 class="scan-h">A₂/A₁² 随沿程距离的变化</h3>${charts.ratio}
    <div class="nl-pair"><div><h3 class="scan-h">基频幅值 A₁</h3>${charts.first}</div><div><h3 class="scan-h">二次谐波幅值</h3>${charts.second}</div></div>
    <p class="section-note">${escapeHtml(a.note)}灰色的行没有参与拟合。</p>${table}`;
}

/** The accumulation analysis as CSV: what was asked, the fit, then one row per point. */
export function accumulationCsv(a) {
  const f = a.fit, blank = v => (v === null || v === undefined ? '' : v);
  const lines = ['# schema,' + a.schema, '# job_id,' + a.job_id, '# frequency_mhz,' + a.frequency_mhz, '# receiving_direction,' + a.direction.join(','), '# line_direction,' + a.line_direction.join(','),
    '# a2_taken_from,' + a.a2_basis, '# window,' + Object.entries(a.window).map(([k, v]) => `${k}=${v}`).join(';'), '# group_velocity_m_s,' + blank(a.group.velocity_m_s),
    '# fit_range_mm,' + f.range_mm.join(','), '# fit_points,' + f.points, '# ratio_slope_per_m2,' + f.ratio.slope_per_m2, '# ratio_intercept_per_m,' + f.ratio.intercept_per_m, '# ratio_r_squared,' + blank(f.ratio.r_squared),
    '# a2_slope_m_per_m,' + f.a2.slope_m_per_m, '# wavenumber_f_rad_mm,' + (f.f.reliable ? f.f.wavenumber_rad_mm : ''), '# wavenumber_2f_rad_mm,' + (f['2f'].reliable ? f['2f'].wavenumber_rad_mm : ''),
    '# theory_mode_f,' + (f.f.theory?.matched ? f.f.theory.mode : ''), '# theory_mode_2f,' + (f['2f'].theory?.matched ? f['2f'].theory.mode : ''),
    '# theory_pair,' + (f.pair ? `${f.pair.primary}-${f.pair.secondary}` : ''), '# theory_delta_k_rad_mm,' + blank(f.pair?.delta_k_rad_mm), '# theory_coherence_length_mm,' + blank(f.pair?.coherence_length_mm),
    'point,x_mm,y_mm,z_mm,distance_mm,window_start_us,window_end_us,in_fit,a1_m,a2_complete_m,a2_nonlinear_m,a2_geometric_m,a2_material_m,a2_over_a1_squared_per_m,phase_f_deg,phase_2f_deg,arrival_us'];
  for (const r of a.rows) lines.push([r.name || r.index, ...r.point_mm, r.distance_mm, ...r.window_us, inFit(a, r) ? 1 : 0,
    ...['a1_m', 'a2_m', 'a2_nonlinear_m', 'a2_geometric_m', 'a2_material_m', 'ratio_per_m', 'phase1_deg', 'phase2_deg'].map(k => (r.usable ? blank(r[k]) : '')), r.arrival_us].join(','));
  return lines.join('\n');
}

// ---------- dispersion ----------
export const DISPERSION_QUANTITIES = { cp: ['相速度', 'km/s'], cg: ['群速度', 'km/s'], k: ['波数', 'rad/mm'] };
let clipCount = 0;

/** Value of a quantity at a point (f MHz, k rad/mm, group velocity m/s) of a branch. */
const quantityOf = (quantity, f, k, group) => (quantity === 'k' ? k : quantity === 'cg' ? (group === null ? NaN : group / 1000) : k > 0 ? 2 * Math.PI * f / k : NaN);

/** Theoretical curves of a plate: one line per branch coloured by family, the modes at f and 2f as labelled dots, and
 *  measured points ([{frequency_mhz, wavenumber_rad_mm, group_velocity_m_s?, label}]) as diamonds. */
export function dispersionChart(d, quantity = 'cp', measured = [], width = 900) {
  const height = 320, left = 62, right = 18, top = 24, bottom = 40, fmax = d.frequency_max_mhz, [name, unit] = DISPERSION_QUANTITIES[quantity] || DISPERSION_QUANTITIES.cp;
  const groups = d.curves.flatMap(c => c.group_velocity_m_s.filter((v, i) => v !== null && c.frequency_mhz[i] <= fmax));
  let ymin = 0, ymax = 1;
  if (quantity === 'cp') ymax = 2 * Math.max(...groups.filter(v => v > 0), 1000) / 1000;
  else if (quantity === 'cg') { ymax = 1.05 * Math.max(...groups, 1000) / 1000; ymin = Math.min(0, 1.05 * Math.min(...groups) / 1000); }
  else ymax = 1.05 * Math.max(...d.curves.flatMap(c => c.wavenumber_rad_mm.filter((_, i) => c.frequency_mhz[i] <= fmax)), 1e-9);
  const sx = v => left + v / fmax * (width - left - right), sy = v => height - bottom - (v - ymin) / (ymax - ymin) * (height - top - bottom), id = `nl-clip-${++clipCount}`;
  const ticks = ticksFor(ymin, ymax, 5).map(yv => `<path d="M${left} ${sy(yv).toFixed(1)}H${width - right}" stroke="${GRID}"/><text x="${left - 8}" y="${(sy(yv) + 4).toFixed(1)}" text-anchor="end">${num(yv, 3)}</text>`).join('')
    + ticksFor(0, fmax, 6).map(xv => `<text x="${sx(xv).toFixed(1)}" y="${height - bottom + 18}" text-anchor="middle">${num(xv, 3)}</text>`).join('');
  const lines = d.curves.map(c => {
    const points = [];
    c.frequency_mhz.forEach((f, i) => { const y = quantityOf(quantity, f, c.wavenumber_rad_mm[i], c.group_velocity_m_s[i]); if (Number.isFinite(y) && f <= fmax * 1.02 && y <= ymax * 3 && y >= ymin - (ymax - ymin)) points.push(`${sx(f).toFixed(1)},${sy(y).toFixed(1)}`); });
    return points.length > 1 ? `<polyline points="${points.join(' ')}" fill="none" stroke="${FAMILY_COLOURS[c.family] || MUTED}" stroke-width="1.6" stroke-linejoin="round"><title>${escapeHtml(c.name)}</title></polyline>` : '';
  }).join('');
  const guides = [[d.frequency_mhz, 'f'], [2 * d.frequency_mhz, '2f']].filter(([f]) => f <= fmax).map(([f, label]) =>
    `<path d="M${sx(f).toFixed(1)} ${top}V${height - bottom}" stroke="${MUTED}" stroke-dasharray="4 4"/><text x="${sx(f).toFixed(1)}" y="${top - 8}" text-anchor="middle" fill="${INK}" font-size="12">${label} = ${num(f)} MHz</text>`).join('');
  const dots = [['f', d.frequency_mhz], ['2f', 2 * d.frequency_mhz]].flatMap(([key, f]) => {
    // labels to the right of the dots; one that would sit on the label above it goes to the left instead
    const shown = (f <= fmax ? d.modes[key] : []).map(m => ({ m, y: quantityOf(quantity, f, m.wavenumber_rad_mm, m.group_velocity_m_s) })).filter(p => p.y >= ymin && p.y <= ymax).sort((a, b) => sy(a.y) - sy(b.y));
    let last = -Infinity, left = false;
    return shown.map(({ m, y }) => {
      left = sy(y) - last < 12 ? !left : false; last = sy(y);
      const tip = `${m.name}（${key}）：相速度 ${num(m.phase_velocity_m_s, 5)} m/s，群速度 ${num(m.group_velocity_m_s, 5)} m/s，波长 ${num(m.wavelength_mm)} mm${m.backward ? '，后向波' : ''}`;
      return `<g><circle cx="${sx(f).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="4.5" fill="${FAMILY_COLOURS[m.family] || MUTED}" stroke="#fff" stroke-width="2"/><text x="${(sx(f) + (left ? -8 : 8)).toFixed(1)}" y="${(sy(y) + 4).toFixed(1)}"${left ? ' text-anchor="end"' : ''} fill="${INK}" font-size="11">${escapeHtml(m.name)}</text>
        <circle cx="${sx(f).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="11" fill="transparent"><title>${escapeHtml(tip)}</title></circle></g>`; });
  }).join('');
  const diamonds = measured.map(p => {
    const y = quantityOf(quantity, p.frequency_mhz, p.wavenumber_rad_mm, p.group_velocity_m_s ?? null); if (!(y >= ymin && y <= ymax) || p.frequency_mhz > fmax) return '';
    const x = sx(p.frequency_mhz), yy = sy(y);
    return `<g><path d="M${x.toFixed(1)} ${(yy - 7).toFixed(1)}l6 7l-6 7l-6 -7z" fill="${INK}" stroke="#fff" stroke-width="2"/><circle cx="${x.toFixed(1)}" cy="${yy.toFixed(1)}" r="12" fill="transparent"><title>${escapeHtml(`${p.label}：${name} ${num(y, 5)} ${unit}`)}</title></circle></g>`; }).join('');
  const families = Object.entries(d.families).sort((a, b) => familyRank(a[0]) - familyRank(b[0])).map(([key, words]) => `<li><i style="background:${FAMILY_COLOURS[key] || MUTED}"></i>${escapeHtml(words)}</li>`).join('');
  const legend = `<ul class="scan-legend">${families}<li><i class="nl-dot"></i>f、2f 处的理论模态</li>${diamonds ? '<li><i class="nl-diamond"></i>本次计算实测（沿程分析）</li>' : ''}</ul>`;
  return `${legend}<svg class="solid-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="平板导波的${name}随频率的变化"><defs><clipPath id="${id}"><rect x="${left}" y="${top}" width="${width - left - right}" height="${height - top - bottom}"/></clipPath></defs>
    <g font-size="11" fill="${MUTED}">${ticks}<text x="${width / 2}" y="${height - 4}" text-anchor="middle">频率 (MHz)</text><text x="7" y="11">${name} (${unit})</text></g>
    <g clip-path="url(#${id})">${lines}</g>${guides}${dots}${diamonds}</svg>`;
}

/** The modes at f and 2f and the phase-matching pairs as tables. */
export function dispersionTables(d, pairs = 8) {
  const surface = m => { const [a, b, c] = m.surface; return `${a.toFixed(2)} : ${b.toFixed(2)} : ${c.toFixed(2)}`; };
  const modes = key => `<table class="wb-table matrix"><thead><tr><th>模态</th><th>相速度 m/s</th><th>群速度 m/s</th><th>波长 mm</th><th>k rad/mm</th><th title="上表面位移的相对大小：沿传播方向 : 面内横向 : 厚度方向">表面位移 x : y : z</th>${d.damping ? '<th title="模型的材料阻尼使这个模态沿传播方向每米衰减的奈培数：衰减率 γ 除以群速度">衰减 Np/m</th>' : ''}</tr></thead><tbody>
    ${d.modes[key].map(m => `<tr><td><i class="nl-swatch" style="background:${FAMILY_COLOURS[m.family] || MUTED}"></i>${escapeHtml(m.name)}${m.backward ? '（后向）' : ''}</td><td>${num(m.phase_velocity_m_s, 5)}</td><td>${num(m.group_velocity_m_s, 5)}</td><td>${num(m.wavelength_mm)}</td><td>${num(m.wavenumber_rad_mm)}</td><td>${surface(m)}</td>${d.damping ? `<td>${m.attenuation_np_m === null || m.attenuation_np_m === undefined ? '—' : num(m.attenuation_np_m, 3)}</td>` : ''}</tr>`).join('')}</tbody></table>`;
  const rows = d.pairs.slice(0, pairs).map(p => `<tr class="${Math.abs(p.mismatch) < .01 ? 'nl-match' : ''}"><td>${escapeHtml(p.primary)} → ${escapeHtml(p.secondary)}</td><td>${pct(p.mismatch, 3)}</td><td>${num(p.delta_k_rad_mm, 3)}</td>
    <td>${p.coherence_length_mm === null ? '匹配' : num(p.coherence_length_mm, 3)}</td><td>${num(p.group_velocity_primary_m_s, 5)}</td><td>${num(p.group_velocity_secondary_m_s, 5)}</td></tr>`).join('');
  return `<h3 class="scan-h">f = ${num(d.frequency_mhz)} MHz 处的模态</h3>${modes('f')}<h3 class="scan-h">2f = ${num(2 * d.frequency_mhz)} MHz 处的模态</h3>${modes('2f')}
    <h3 class="scan-h">相位匹配：f 处的一次模态 → 2f 处的对称模态</h3>
    <table class="wb-table matrix"><thead><tr><th>模态对</th><th>相速度失配</th><th>Δk rad/mm</th><th title="二次谐波先增后减，在这个距离达到最大">π/|Δk| mm</th><th>一次群速度 m/s</th><th>二次群速度 m/s</th></tr></thead><tbody>${rows}</tbody></table>
    ${d.pairs.length > pairs ? `<p class="wb-note">按失配从小到大列出前 ${pairs} 对，共 ${d.pairs.length} 对。</p>` : ''}<p class="section-note">${escapeHtml(d.note)} 加底色的行相速度失配小于 1%。</p>`;
}

// ---------- f–k spectrum ----------
/** The image of an f–k spectrum: {values (0…255 for −60…0 dB), rows, columns, frequency(i), wavenumber(j)}. */
export function fkGrid(fk) {
  const text = atob(fk.image), values = new Uint8Array(text.length); for (let i = 0; i < text.length; i++) values[i] = text.charCodeAt(i);
  const rows = fk.frequency.count, columns = fk.wavenumber.count;
  if (values.length !== rows * columns) throw new Error('f–k 谱的数据大小不一致。');
  return { values, rows, columns, frequency: i => fk.frequency.first_mhz + i * fk.frequency.step_mhz, wavenumber: j => fk.wavenumber.first_rad_mm + j * fk.wavenumber.step_rad_mm };
}

/** Where the largest value of an f–k image is: {frequency_mhz, wavenumber_rad_mm}. */
export function fkPeak(fk) { const g = fkGrid(fk); let best = 0; for (let i = 1; i < g.values.length; i++) if (g.values[i] > g.values[best]) best = i; return { frequency_mhz: g.frequency(Math.floor(best / g.columns)), wavenumber_rad_mm: g.wavenumber(best % g.columns) }; }

/** Draw an f–k spectrum on a canvas: magnitude in grey (light = −60 dB, dark = the maximum), the theoretical curves in
 *  the family colours, dashed lines at f and 2f. Returns {at(x, y)} for the read-out under the pointer. */
export function drawFk(canvas, fk, { curves = true } = {}) {
  const g = fkGrid(fk), ratio = Math.min(globalThis.devicePixelRatio || 1, 2), box = canvas.getBoundingClientRect(), width = Math.max(360, box.width), height = 340;
  canvas.width = width * ratio; canvas.height = height * ratio; canvas.style.height = height + 'px';
  const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  const left = 62, right = 18, top = 14, bottom = 40, w = width - left - right, h = height - top - bottom;
  const k0 = g.wavenumber(0) - fk.wavenumber.step_rad_mm / 2, k1 = g.wavenumber(g.columns - 1) + fk.wavenumber.step_rad_mm / 2, f0 = 0, f1 = g.frequency(g.rows - 1) + fk.frequency.step_mhz / 2;
  const sx = k => left + (k - k0) / (k1 - k0) * w, sy = f => top + h - (f - f0) / (f1 - f0) * h;
  const image = new ImageData(g.columns, g.rows);
  for (let i = 0; i < g.rows; i++) for (let j = 0; j < g.columns; j++) {
    const v = g.values[i * g.columns + j] / 255, grey = Math.round(250 - 222 * v ** 1.25), o = ((g.rows - 1 - i) * g.columns + j) * 4;   // frequency upwards
    image.data[o] = grey; image.data[o + 1] = grey; image.data[o + 2] = grey; image.data[o + 3] = 255;
  }
  const buffer = document.createElement('canvas'); buffer.width = g.columns; buffer.height = g.rows; buffer.getContext('2d').putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(buffer, sx(g.wavenumber(0) - fk.wavenumber.step_rad_mm / 2), sy(f1), sx(k1) - sx(k0), sy(g.frequency(0) - fk.frequency.step_mhz / 2) - sy(f1));
  ctx.save(); ctx.beginPath(); ctx.rect(left, top, w, h); ctx.clip();
  if (curves && fk.curves) for (const c of fk.curves) for (const sign of [1, -1]) {
    ctx.beginPath(); c.wavenumber_rad_mm.forEach((k, i) => { const x = sx(sign * k), y = sy(c.frequency_mhz[i]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.strokeStyle = FAMILY_COLOURS[c.family] || MUTED; ctx.lineWidth = 1.6; ctx.stroke();
  }
  ctx.setLineDash([4, 4]); ctx.strokeStyle = MUTED; ctx.lineWidth = 1;
  for (const f of [fk.frequency_mhz, 2 * fk.frequency_mhz]) if (f < f1) { ctx.beginPath(); ctx.moveTo(left, sy(f)); ctx.lineTo(left + w, sy(f)); ctx.stroke(); }
  if (k0 < 0) { ctx.beginPath(); ctx.moveTo(sx(0), top); ctx.lineTo(sx(0), top + h); ctx.stroke(); }
  ctx.restore(); ctx.setLineDash([]);
  ctx.strokeStyle = '#c9d4da'; ctx.strokeRect(left + .5, top + .5, w, h); ctx.fillStyle = MUTED; ctx.font = '11px sans-serif';
  ctx.textAlign = 'center'; for (const k of ticksFor(k0, k1, 8)) ctx.fillText(num(k, 3), sx(k), top + h + 16);
  ctx.textAlign = 'right'; for (const f of ticksFor(f0, f1, 5)) ctx.fillText(num(f, 3), left - 8, sy(f) + 4);
  ctx.fillStyle = INK; ctx.textAlign = 'left';
  for (const [f, label] of [[fk.frequency_mhz, 'f'], [2 * fk.frequency_mhz, '2f']]) if (f < f1) ctx.fillText(label, left + 5, sy(f) - 4);
  ctx.fillStyle = MUTED; ctx.textAlign = 'center'; ctx.fillText('波数 (rad/mm)　正：从第一个点传向最后一个点', left + w / 2, height - 6);
  ctx.save(); ctx.translate(13, top + h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('频率 (MHz)', 0, 0); ctx.restore();
  return { at(x, y) {
    if (x < left || x > left + w || y < top || y > top + h) return null;
    const k = k0 + (x - left) / w * (k1 - k0), f = f0 + (top + h - y) / h * (f1 - f0), i = Math.round((f - g.frequency(0)) / fk.frequency.step_mhz), j = Math.round((k - g.wavenumber(0)) / fk.wavenumber.step_rad_mm);
    if (i < 0 || i >= g.rows || j < 0 || j >= g.columns) return null;
    return { frequency_mhz: f, wavenumber_rad_mm: k, db: g.values[i * g.columns + j] / 255 * 60 - 60 };
  } };
}

// ---------- convergence ----------
/** The convergence check of a job: the plan of a new check, the newest check and its comparison table. */
export function convergenceMarkup({ plan = null, error = '', check = null, table = null, busy = false } = {}) {
  let start = '<p class="section-note">正在检查加密网格的大小和预计耗时…</p>';
  if (error) start = `<p class="section-note model-error">${escapeHtml(error)}</p>`;
  else if (plan) {
    const bad = plan.cases.filter(c => !c.ok);
    start = `<div class="scan-wrap"><table class="wb-table matrix"><thead><tr><th>网格</th><th>节点</th><th>预计耗时</th><th>内存</th><th>磁盘</th></tr></thead><tbody>
      <tr><td>${escapeHtml(plan.base.label)}（已算）</td><td>${(plan.base.nodes || 0).toLocaleString()}</td><td>${plan.base.seconds ? '用时 ' + fmtDuration(plan.base.seconds) : '—'}</td><td></td><td></td></tr>
      ${plan.cases.map(c => `<tr class="${c.ok ? '' : 'bad'}"><td>${escapeHtml(c.label)}</td>${c.ok ? `<td>${(c.nodes || 0).toLocaleString()}</td><td>${c.seconds ? fmtDuration(c.seconds) : '—'}</td><td>${c.memory_bytes ? fmtBytes(c.memory_bytes) : '—'}</td><td>${c.disk_bytes ? fmtBytes(c.disk_bytes) : '—'}</td>`
        : `<td colspan="4" class="scan-err">${escapeHtml(c.errors[0] || '不能计算')}</td>`}</tr>`).join('')}</tbody></table></div>
      ${plan.disk_short ? `<p class="section-note model-error">数据目录的可用空间（${fmtBytes(plan.free_bytes)}）不够放下加密网格的记录（约 ${fmtBytes(plan.disk_bytes_total)}）。</p>` : ''}
      <div class="solid-path-actions"><label class="field nl-inline"><span>加密级数</span><select id="nl-levels"><option value="1" ${plan.levels === 1 ? 'selected' : ''}>一级</option><option value="2" ${plan.levels === 2 ? 'selected' : ''}>两级</option></select></label>
      <button class="button primary small" id="nl-converge" ${bad.length || busy ? 'disabled' : ''}>开始收敛检查（再算 ${plan.cases.length} 次，约 ${fmtDuration(plan.seconds_total)}）</button></div>
      ${busy ? '<p class="section-note">已有计算或扫描在进行，等它完成或暂停后再开始。</p>' : ''}`;
  }
  let result = '';
  if (check) {
    const head = `<h3 class="scan-h">最近一次检查：${escapeHtml(check.name)} <span class="wb-phase ${check.state}">${SCAN_WORDS[check.state] || escapeHtml(check.state)}</span></h3>
      <div class="solid-path-actions"><button class="button secondary small" id="nl-open-check" data-scan="${check.scan_id}">打开这次检查的汇总页</button><span class="muted small-text">${check.counts.completed} / ${check.count} 套网格已完成${check.counts.failed ? `，${check.counts.failed} 套未完成` : ''}${check.message ? ' · ' + escapeHtml(check.message) : ''}</span></div>`;
    result = head + (table ? convergenceTable(table) : '<p class="section-note">正在读取比较结果…</p>');
  }
  return `<p class="section-note">同一个模型在更细的网格上重算，再比较各接收点的 A₁、A₂ 和波形。每加密一级：单元阶次低于 5 时阶次加一（各方向的节点都加密），已是 5 阶时单元尺寸乘 0.75。加密后的计算排队运行，和参数扫描一样可以暂停、继续。</p>${start}${result}`;
}

/** The comparison table of a convergence check (offline/scan.py `Scans.convergence`). */
export function convergenceTable(t) {
  if (!t.rows.length) return '<p class="section-note">还没有完成的网格。</p>';
  const v = t.verdict, reference = t.rows.at(-1);
  const verdict = v ? `<p class="nl-verdict ${v.converged ? 'ok' : 'warn'}"><b>${v.converged ? '已收敛' : '未达到判据'}</b>　${escapeHtml(v.text)}</p>` : '<p class="section-note">只有一套网格完成，还不能比较。</p>';
  const meshes = `<div class="scan-wrap"><table class="wb-table matrix"><thead><tr><th>网格</th><th>节点</th><th>时间步 ns</th><th>用时</th><th title="各接收点中最大的变化，相对最细网格">|ΔA₁| 最大</th><th>|ΔA₂| 最大</th><th>f 相位差最大 °</th><th title="分析时间窗内，线性（或完整）位移波形的相对 L2 差">波形差最大</th><th title="非线性增量波形的相对 L2 差">增量波形差最大</th></tr></thead><tbody>
    ${t.rows.map(r => `<tr><td>${escapeHtml(r.label)}${r.is_reference ? '（参考）' : ''}</td><td>${(r.nodes || 0).toLocaleString()}</td><td>${r.layer_step_us ? num(r.layer_step_us * 1000) : '—'}</td><td>${r.elapsed_s ? fmtDuration(r.elapsed_s) : '—'}</td>
      ${r.is_reference ? '<td colspan="5" class="muted">其他网格与它比较</td>' : `<td>${size(r.worst.a1_change)}</td><td>${size(r.worst.a2_change)}</td><td>${num(r.worst.phase1_change_deg, 3)}</td><td>${size(r.worst.waveform_change)}</td><td>${size(r.worst.increment_waveform_change)}</td>`}</tr>`).join('')}
    ${t.pending.map(p => `<tr><td>${escapeHtml(p.label)}</td><td colspan="8" class="muted">${{ queued: '排队', running: '计算中', failed: '未完成', paused: '已暂停' }[p.phase] || escapeHtml(p.phase)}${p.message ? '：' + escapeHtml(p.message) : ''}</td></tr>`).join('')}</tbody></table></div>`;
  const base = t.rows[0], detail = t.rows.length > 1 ? `<h3 class="scan-h">${escapeHtml(base.label)} 各接收点相对 ${escapeHtml(reference.label)} 的变化</h3>
    <div class="scan-wrap"><table class="wb-table matrix"><thead><tr><th>接收点</th><th>A₁ nm</th><th>ΔA₁</th><th>A₂ pm</th><th>ΔA₂</th><th>f 相位差 °</th><th>波形差</th><th>增量波形差</th></tr></thead><tbody>
    ${base.receivers.map(r => `<tr><td>${escapeHtml(r.name)}</td><td>${num(r.a1_m * 1e9, 5)}</td><td>${pct(r.a1_change, 3)}</td><td>${num(r.a2_m * 1e12, 5)}</td><td>${pct(r.a2_change, 3)}</td><td>${num(r.phase1_change_deg, 3)}</td><td>${size(r.waveform_change)}</td><td>${size(r.increment_waveform_change)}</td></tr>`).join('')}</tbody></table></div>` : '';
  return `${verdict}${meshes}${detail}<p class="section-note">A₂ 取${t.a2_basis === 'nonlinear' ? '非线性增量（完整 − 线性）' : '完整响应（这次计算没有线性控制）'}${t.window_us ? `；分析时间窗 ${num(t.window_us[0])}–${num(t.window_us[1])} μs` : '；整段记录'}。${escapeHtml(t.note)}</p>`;
}

// ---------- the panel ----------
const TABS = [['accumulation', '沿程累积（A₂/A₁²）'], ['dispersion', '频散与 f–k 谱'], ['convergence', '收敛检查']];

export function nonlinearMarkup(result) {
  const compact = result.recording?.record_mode === 'compact', receivers = result.scene.receivers, laws = (result.controls || []).map(c => c.law);
  const parts = ['full', ...(laws.includes('linear') ? ['linear', 'nonlinear'] : []), ...(laws.length === 3 ? ['geometric', 'material'] : [])];
  const vector = (id, label) => ['X', 'Y', 'Z'].map((axis, j) => `<label class="field"><span>${label} ${axis}</span><input id="${id}-${j}" type="number" step="any"></label>`).join('');
  return `<section class="panel" id="nl-panel"><div class="panel-heading"><h2>非线性导波分析</h2><span class="muted small-text">读取已保存的记录，不重新求解</span></div><div class="panel-body">
    <div class="nl-tabs" role="tablist">${TABS.map(([key, label]) => `<button type="button" role="tab" class="nl-tab" data-nl-tab="${key}">${label}</button>`).join('')}</div>
    <div id="nl-line" class="nl-form">
      <div class="form-grid"><label class="field"><span>分析的点</span><select id="nl-mode">${compact ? '' : '<option value="line">线段等距采样</option>'}<option value="receivers">使用接收点（${receivers.length} 个）</option></select></label>
        <label class="field" id="nl-count-field"><span>采样点数（留空自动：按 2f 处最短波长的 1/5）</span><input id="nl-count" type="number" min="2" max="256" step="1" placeholder="自动"></label></div>
      <div id="nl-segment" class="form-grid solid-vectors">${vector('nl-start', '起点')}${vector('nl-end', '终点')}</div>
      <div class="form-grid solid-vectors">${vector('nl-direction', '接收方向')}</div>
      ${compact ? '<p class="section-note">这次计算保存的是精简记录：只有接收点有完整波形，所以分析的点就是接收点。要沿任意线段采样，请在“求解器配置”里选“完整记录”。</p>' : ''}
    </div>
    <div data-nl-page="accumulation">
      <div class="form-grid"><label class="field"><span>分析时间窗</span><select id="nl-window"><option value="follow">跟随波包（按群速度随距离后移）</option><option value="fixed">各点同一时间窗</option></select></label>
        <label class="field" data-nl-follow><span>跟随速度 / m·s⁻¹（留空用实测群速度）</span><input id="nl-speed" type="number" step="any" placeholder="实测"></label>
        <label class="field" data-nl-follow><span>时间窗长度 / μs（留空为激励时长的 1.5 倍）</span><input id="nl-length" type="number" step="any" placeholder="自动"></label>
        <label class="field" data-nl-fixed hidden><span>时间窗起点 / μs</span><input id="nl-window-start" type="number" step="any"></label>
        <label class="field" data-nl-fixed hidden><span>时间窗终点 / μs</span><input id="nl-window-end" type="number" step="any"></label>
        <label class="field"><span>拟合范围起点 / mm（留空用全部点）</span><input id="nl-fit-from" type="number" step="any" placeholder="全部"></label>
        <label class="field"><span>拟合范围终点 / mm</span><input id="nl-fit-to" type="number" step="any" placeholder="全部"></label></div>
      <div class="solid-path-actions"><button class="button primary small" id="nl-run">计算 A₁、A₂ 和 A₂/A₁² 拟合</button><button class="button secondary small" id="nl-csv" disabled>下载 CSV</button></div>
      <p id="nl-message" class="section-note" role="status">沿一条线提取基频和二次谐波，拟合 A₂/A₁² 随传播距离的增长，并由相位斜率得到 f、2f 处的波数。</p>
      <div id="nl-accumulation"></div>
    </div>
    <div data-nl-page="dispersion" hidden>
      <div class="form-grid"><label class="field"><span>纵轴</span><select id="nl-quantity">${Object.entries(DISPERSION_QUANTITIES).map(([key, [name, unit]]) => `<option value="${key}">${name} (${unit})</option>`).join('')}</select></label>
        <label class="field"><span>传播方向与 x 轴的夹角 / °（各向异性材料有影响）</span><input id="nl-angle" type="number" step="any" value="0"></label>
        <label class="field"><span>频率上限 / MHz（留空为 2.6 f）</span><input id="nl-theory-fmax" type="number" step="any" min="0" placeholder="2.6 f"></label></div>
      <p id="nl-theory-message" class="section-note" role="status"></p><div id="nl-theory"></div>
      <h3 class="scan-h">f–k 谱（沿上面设置的点列）</h3>
      <div class="form-grid"><label class="field"><span>响应</span><select id="nl-part">${parts.map(p => `<option value="${p}">${PART_LABELS[p]}</option>`).join('')}</select></label>
        <label class="field"><span>时间范围起点 / μs（留空为整段记录）</span><input id="nl-range-start" type="number" step="any" placeholder="整段"></label>
        <label class="field"><span>时间范围终点 / μs</span><input id="nl-range-end" type="number" step="any" placeholder="整段"></label>
        <label class="field"><span>频率上限 / MHz（留空为 3 f）</span><input id="nl-fk-fmax" type="number" step="any" min="0" placeholder="3 f"></label></div>
      <div class="solid-path-actions"><button class="button primary small" id="nl-fk-run">计算 f–k 谱</button><label class="nl-check"><input type="checkbox" id="nl-fk-curves" checked>叠加理论曲线</label></div>
      <p id="nl-fk-message" class="section-note" role="status">f–k 谱需要等间距的点，至少 8 个；点距决定能分辨的最大波数。</p>
      <div id="nl-fk" hidden><canvas id="nl-fk-canvas" class="nl-canvas" aria-label="频率–波数谱"></canvas><p id="nl-fk-readout" class="muted small-text">把鼠标移到图上读取频率、波数和相速度。</p><ul class="scan-legend" id="nl-fk-legend"></ul><div id="nl-fk-notes"></div></div>
    </div>
    <div data-nl-page="convergence" hidden><div id="nl-convergence"></div></div>
  </div></section>`;
}

const FORM_IDS = ['nl-mode', 'nl-count', 'nl-window', 'nl-speed', 'nl-length', 'nl-window-start', 'nl-window-end', 'nl-fit-from', 'nl-fit-to', 'nl-quantity', 'nl-angle', 'nl-theory-fmax', 'nl-part', 'nl-range-start', 'nl-range-end', 'nl-fk-fmax',
  ...['nl-start', 'nl-end', 'nl-direction'].flatMap(id => [0, 1, 2].map(j => `${id}-${j}`))];

/** Bind the panel. `ctx`: {job, result, api(url, body), busy(), openScan(id), refresh()}; `saved`: an earlier snapshot(). */
export function mountNonlinear(container, ctx, saved = null) {
  const $ = id => container.querySelector('#' + id), result = ctx.result, base = '/api/jobs/' + ctx.job, receivers = result.scene.receivers, compact = result.recording?.record_mode === 'compact';
  let disposed = false, tab = saved?.tab || 'accumulation', accumulation = saved?.accumulation || null, theory = saved?.theory || null, fk = saved?.fk || null, reader = null, timer = 0, stamp = 0;
  let convergence = { plan: null, error: '', check: null, table: null, levels: saved?.levels || 1 };
  const listeners = [], listen = (element, event, callback) => { if (!element) return; element.addEventListener(event, callback); listeners.push(() => element.removeEventListener(event, callback)); };
  const widthOf = element => Math.max(360, Math.round(element.getBoundingClientRect().width) || 900);
  const say = (id, text, bad = false) => { const el = $(id); el.textContent = text; el.classList.toggle('model-error', bad); };

  // ----- the form -----
  const first = receivers[0].point_mm, last = receivers.at(-1).point_mm, g = result.scene.geometry, apart = Math.hypot(...first.map((v, j) => v - last[j])) > 1e-6;
  const start = apart ? first : [+(g.length_mm * .1).toFixed(4), first[1], first[2]], end = apart ? last : [+(g.length_mm * .9).toFixed(4), first[1], first[2]];
  [0, 1, 2].forEach(j => { $('nl-start-' + j).value = start[j]; $('nl-end-' + j).value = end[j]; $('nl-direction-' + j).value = receivers[0].direction[j]; });
  if (compact || receivers.length < 2) $('nl-mode').value = compact ? 'receivers' : 'line';
  if (saved?.inputs) for (const id of FORM_IDS) if (saved.inputs[id] !== undefined && $(id)) $(id).value = saved.inputs[id];
  const form = () => ({ mode: $('nl-mode').value, count: $('nl-count').value, start: [0, 1, 2].map(j => $('nl-start-' + j).value), end: [0, 1, 2].map(j => $('nl-end-' + j).value), direction: [0, 1, 2].map(j => $('nl-direction-' + j).value),
    window: $('nl-window').value, speed: $('nl-speed').value, length: $('nl-length').value, windowStart: $('nl-window-start').value, windowEnd: $('nl-window-end').value, fitFrom: $('nl-fit-from').value, fitTo: $('nl-fit-to').value,
    part: $('nl-part').value, rangeStart: $('nl-range-start').value, rangeEnd: $('nl-range-end').value, fmax: $('nl-fk-fmax').value });
  function layout() {
    const line = $('nl-mode').value === 'line'; $('nl-segment').hidden = !line; $('nl-count-field').hidden = !line;
    const fixed = $('nl-window').value === 'fixed';
    container.querySelectorAll('[data-nl-fixed]').forEach(el => { el.hidden = !fixed; }); container.querySelectorAll('[data-nl-follow]').forEach(el => { el.hidden = fixed; });
    container.querySelectorAll('[data-nl-tab]').forEach(b => { b.classList.toggle('active', b.dataset.nlTab === tab); b.setAttribute('aria-selected', b.dataset.nlTab === tab); });
    container.querySelectorAll('[data-nl-page]').forEach(page => { page.hidden = page.dataset.nlPage !== tab; });
    $('nl-line').hidden = tab === 'convergence';
  }

  // ----- accumulation -----
  function showAccumulation() {
    if (!accumulation) { $('nl-accumulation').replaceChildren(); $('nl-csv').disabled = true; return; }
    $('nl-accumulation').innerHTML = accumulationMarkup(accumulation, widthOf($('nl-accumulation'))); $('nl-csv').disabled = false;
  }
  async function runAccumulation() {
    const token = ++stamp; $('nl-run').disabled = true; say('nl-message', '正在读取记录并计算…');
    try {
      const body = accumulationBody(form()), next = await ctx.api(base + '/accumulation', body); if (disposed || token !== stamp) return;
      accumulation = next; showAccumulation();
      say('nl-message', `已分析 ${next.rows.length} 个点（${next.points_mode === 'receivers' ? '接收点' : '线段采样'}），接收方向 [${next.direction.map(v => v.toFixed(3)).join(', ')}]，f = ${num(next.frequency_mhz)} MHz。改动设置后请重新计算。`);
      if (theory) showTheory();
    } catch (error) { if (!disposed && token === stamp) say('nl-message', error.message, true); } finally { if (!disposed) $('nl-run').disabled = false; }
  }

  // ----- dispersion -----
  const measured = () => {
    if (!accumulation) return [];
    const f = accumulation.frequency_mhz, fit = accumulation.fit, out = [];
    if (fit.f.reliable && fit.f.wavenumber_rad_mm > 0) out.push({ frequency_mhz: f, wavenumber_rad_mm: fit.f.wavenumber_rad_mm, group_velocity_m_s: accumulation.group.velocity_m_s, label: '实测 f' });
    if (fit['2f'].reliable && fit['2f'].wavenumber_rad_mm > 0) out.push({ frequency_mhz: 2 * f, wavenumber_rad_mm: fit['2f'].wavenumber_rad_mm, group_velocity_m_s: null, label: '实测 2f（相位斜率）' });
    return out;
  };
  function showTheory() {
    if (!theory) return;
    if (theory.error) { $('nl-theory').replaceChildren(); say('nl-theory-message', theory.error, true); return; }
    say('nl-theory-message', `板厚 ${num(theory.thickness_mm)} mm 的自由平板，传播方向与 x 轴夹角 ${num(theory.angle_deg)}°。${theory.validity.note}`, !theory.validity.plane_strain);
    $('nl-theory').innerHTML = dispersionChart(theory, $('nl-quantity').value, measured(), widthOf($('nl-theory'))) + dispersionTables(theory);
  }
  async function loadTheory() {
    const token = ++stamp; say('nl-theory-message', '正在计算理论频散曲线…');
    try {
      const angle = Number($('nl-angle').value) || 0, top = $('nl-theory-fmax').value.trim();
      const next = await ctx.api(`${base}/dispersion?angle_deg=${angle}${top ? `&frequency_max_mhz=${Number(top)}` : ''}`); if (disposed || token !== stamp) return;
      theory = next;
    } catch (error) { if (disposed || token !== stamp) return; theory = { error: error.message }; }
    showTheory();
  }
  function showFk() {
    if (!fk) { $('nl-fk').hidden = true; return; }
    $('nl-fk').hidden = false; reader = drawFk($('nl-fk-canvas'), fk, { curves: $('nl-fk-curves').checked });
    const families = [...new Set((fk.curves || []).map(c => c.family))].sort((a, b) => familyRank(a) - familyRank(b));
    $('nl-fk-legend').innerHTML = (fk.curves && $('nl-fk-curves').checked ? families.map(key => `<li><i style="background:${FAMILY_COLOURS[key] || MUTED}"></i>${escapeHtml(theory?.families?.[key] || { S: '对称 Lamb', A: '反对称 Lamb', SH: '水平剪切 SH' }[key] || key)}（理论）</li>`).join('') : '')
      + '<li><i class="nl-grey"></i>谱幅值：浅 −60 dB，深 0 dB（相对最大值）</li>';
    $('nl-fk-notes').innerHTML = `${fk.notes.length ? `<ul class="nl-notes">${fk.notes.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul>` : ''}<p class="section-note">${escapeHtml(fk.note)}${fk.validity && !fk.validity.plane_strain ? ' ' + escapeHtml(fk.validity.note) : ''}</p>`;
  }
  async function runFk() {
    const token = ++stamp; $('nl-fk-run').disabled = true; say('nl-fk-message', '正在读取记录并计算 f–k 谱…');
    try {
      const next = await ctx.api(base + '/fk', fkBody(form())); if (disposed || token !== stamp) return;
      fk = next; showFk(); const peak = fkPeak(next);
      say('nl-fk-message', `${PART_LABELS[next.part] || next.part}：${next.points} 个点，点距 ${num(next.spacing_mm)} mm（波数上限 ${num(next.wavenumber.nyquist_rad_mm, 3)} rad/mm），${num(next.range_us[0])}–${num(next.range_us[1])} μs 的 ${next.samples} 个采样点。最大值在 ${num(peak.frequency_mhz, 3)} MHz、${num(peak.wavenumber_rad_mm, 3)} rad/mm${peak.frequency_mhz < .5 * next.frequency_mhz && ['nonlinear', 'geometric', 'material'].includes(next.part) ? '：非线性增量里零频附近的成分是随波包一起走的准静态位移，二次谐波在 2f 处' : ''}。`);
    } catch (error) { if (!disposed && token === stamp) say('nl-fk-message', error.message, true); } finally { if (!disposed) $('nl-fk-run').disabled = false; }
  }

  // ----- convergence -----
  function showConvergence() {
    $('nl-convergence').innerHTML = convergenceMarkup({ ...convergence, busy: ctx.busy() });
    listen($('nl-levels'), 'change', ev => { convergence.levels = Number(ev.target.value); convergence.plan = null; showConvergence(); loadConvergence(); });
    listen($('nl-converge'), 'click', startConvergence);
    listen($('nl-open-check'), 'click', ev => ctx.openScan(ev.currentTarget.dataset.scan));
  }
  async function loadConvergence() {
    clearTimeout(timer); const token = ++stamp;
    try {
      const plan = await ctx.api(base + '/converge-plan', { levels: convergence.levels }); if (disposed || token !== stamp) return;
      convergence.plan = plan; convergence.error = ''; convergence.check = plan.existing[0] || null; showConvergence();
      if (convergence.check) {
        convergence.table = await ctx.api(`/api/scans/${convergence.check.scan_id}/convergence`); if (disposed || token !== stamp) return;
        showConvergence();
        if (convergence.check.state === 'running') timer = setTimeout(() => { if (!disposed && tab === 'convergence') loadConvergence(); }, 4000);
      }
    } catch (error) { if (disposed || token !== stamp) return; convergence.error = error.message; showConvergence(); }
  }
  async function startConvergence() {
    $('nl-converge').disabled = true;
    try { await ctx.api(base + '/converge', { levels: convergence.levels }); await ctx.refresh?.(); if (!disposed) await loadConvergence(); }
    catch (error) { if (!disposed) { convergence.error = error.message; showConvergence(); } }
  }

  function open(next) {
    tab = next; layout();
    if (tab === 'accumulation') showAccumulation();
    if (tab === 'dispersion') { if (theory) showTheory(); else loadTheory(); showFk(); }
    if (tab === 'convergence') { showConvergence(); loadConvergence(); }
  }
  container.querySelectorAll('[data-nl-tab]').forEach(b => listen(b, 'click', () => open(b.dataset.nlTab)));
  listen($('nl-mode'), 'change', layout); listen($('nl-window'), 'change', layout);
  listen($('nl-run'), 'click', runAccumulation); listen($('nl-fk-run'), 'click', runFk);
  listen($('nl-quantity'), 'change', showTheory); listen($('nl-angle'), 'change', loadTheory); listen($('nl-theory-fmax'), 'change', loadTheory);
  listen($('nl-fk-curves'), 'change', showFk);
  listen($('nl-fk-canvas'), 'mousemove', ev => {
    const box = ev.currentTarget.getBoundingClientRect(), p = reader?.at(ev.clientX - box.left, ev.clientY - box.top);
    $('nl-fk-readout').textContent = p ? `${num(p.frequency_mhz, 4)} MHz，${num(p.wavenumber_rad_mm, 4)} rad/mm${Math.abs(p.wavenumber_rad_mm) > 1e-9 ? `，相速度 ${num(2 * Math.PI * p.frequency_mhz / Math.abs(p.wavenumber_rad_mm) * 1000, 5)} m/s` : ''}，${p.db.toFixed(1)} dB` : '把鼠标移到图上读取频率、波数和相速度。';
  });
  listen($('nl-csv'), 'click', () => {
    if (!accumulation) return;
    const link = document.createElement('a'), url = URL.createObjectURL(new Blob(['﻿' + accumulationCsv(accumulation)], { type: 'text/csv;charset=utf-8' }));
    link.href = url; link.download = ctx.job + '-accumulation.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  let previous = null;
  const resize = new ResizeObserver(entries => { const width = Math.round(entries[0].contentRect.width); if (disposed || width === previous) return; previous = width; if (tab === 'accumulation') showAccumulation(); if (tab === 'dispersion') { showTheory(); showFk(); } });
  resize.observe($('nl-panel'));
  open(tab);
  return { snapshot: () => ({ tab, accumulation, theory: theory?.error ? null : theory, fk, levels: convergence.levels, inputs: Object.fromEntries(FORM_IDS.filter(id => $(id)).map(id => [id, $(id).value])) }),
    cleanup: () => { disposed = true; stamp++; clearTimeout(timer); resize.disconnect(); listeners.forEach(remove => remove()); } };
}
