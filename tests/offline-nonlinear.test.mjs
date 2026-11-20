import test from 'node:test';
import assert from 'node:assert/strict';
import { FAMILY_COLOURS, ticksFor, lineBody, accumulationBody, fkBody, lineAngle, accumulationCharts, accumulationMarkup, accumulationCsv, dispersionChart, dispersionTables, fkGrid, fkPeak,
  convergenceMarkup, convergenceTable, nonlinearMarkup } from '../offline/web/nonlinear.js';
import { SCAN_SERIES, seriesChart } from '../offline/web/scan.js';

const form = (extra = {}) => ({ mode: 'line', count: '', start: ['10', '1.25', '1'], end: ['140', '1.25', '1'], direction: ['1', '0', '0'], window: 'follow', speed: '', length: '', windowStart: '', windowEnd: '', fitFrom: '', fitTo: '',
  part: 'full', rangeStart: '', rangeEnd: '', fmax: '', ...extra });

test('the forms become requests, and what is incomplete is refused with the reason', () => {
  assert.deepEqual(lineBody(form()), { points: { mode: 'line', start_mm: [10, 1.25, 1], end_mm: [140, 1.25, 1], count: 'auto' }, direction: [1, 0, 0] });
  assert.deepEqual(lineBody(form({ mode: 'receivers', start: ['', '', ''] })), { points: { mode: 'receivers' }, direction: [1, 0, 0] });
  assert.equal(lineBody(form({ count: '41' })).points.count, 41);
  assert.deepEqual(accumulationBody(form()).window, { mode: 'follow' }); assert.equal('fit_mm' in accumulationBody(form()), false);
  assert.deepEqual(accumulationBody(form({ speed: '5400', length: '30', fitFrom: '0', fitTo: '90' })), { ...lineBody(form()), window: { mode: 'follow', speed_m_s: 5400, length_us: 30 }, fit_mm: [0, 90] });
  assert.deepEqual(accumulationBody(form({ window: 'fixed', windowStart: '5', windowEnd: '40' })).window, { mode: 'fixed', start_us: 5, end_us: 40 });
  assert.deepEqual(fkBody(form({ part: 'nonlinear', rangeStart: '0', rangeEnd: '50', fmax: '0.6' })), { ...lineBody(form()), part: 'nonlinear', range_us: [0, 50], frequency_max_mhz: .6 });
  assert.deepEqual(fkBody(form()), { ...lineBody(form()), part: 'full' });
  for (const [bad, text] of [[{ start: ['10', '', '1'] }, /起点/], [{ direction: ['0', '0', '0'] }, /零向量/], [{ count: '1' }, /采样点数/], [{ count: '2.5' }, /采样点数/], [{ window: 'fixed', windowStart: '5', windowEnd: '5' }, /早于/],
    [{ window: 'fixed' }, /时间窗/], [{ fitFrom: '10' }, /同时填写/], [{ fitFrom: '90', fitTo: '10' }, /小于/], [{ speed: 'abc' }, /数值/]]) assert.throws(() => accumulationBody(form(bad)), text);
  assert.throws(() => fkBody(form({ rangeStart: '5' })), /同时填写/); assert.throws(() => fkBody(form({ rangeStart: '9', rangeEnd: '3' })), /早于/);
  assert.equal(lineAngle([1, 0, 0]), 0); assert.equal(lineAngle([0, 1, 0]), 90); assert.equal(lineAngle([.6, 0, .8]), null);
});

test('ticks are round numbers that cover the range', () => {
  assert.deepEqual(ticksFor(0, .52, 6), [0, .1, .2, .3, .4, .5]); assert.deepEqual(ticksFor(-2.74, 2.74, 8), [-2, -1, 0, 1, 2]); assert.deepEqual(ticksFor(0, 10.9, 5), [0, 5, 10]);
  assert.deepEqual(ticksFor(0, 1, 5), [0, .2, .4, .6, .8, 1]); assert.deepEqual(ticksFor(-.5, 5.7, 5), [0, 2, 4]);
});

const wave = (k, mode, difference) => ({ wavenumber_rad_mm: k, phase_velocity_m_s: 5429.8, r_squared: 1, alias_turns: 0, reliable: true, unambiguous_below_rad_mm: .157,
  theory: mode ? { mode, wavenumber_rad_mm: k / (1 + difference), phase_velocity_m_s: 5429.6, group_velocity_m_s: 5422.8, difference, matched: Math.abs(difference) < .02 } : null });
const accumulation = (extra = {}) => ({ schema: 'workai.nonlinear.accumulation.v1', job_id: 'local-' + 'a'.repeat(32), frequency_mhz: .2, direction: [1, 0, 0], line_direction: [1, 0, 0], points_mode: 'receivers', record_mode: 'full',
  controls: ['linear', 'stvk', 'murnaghan'], a2_basis: 'nonlinear', window: { mode: 'follow', speed_m_s: 5441, speed_source: 'measured', length_us: 37.5, centre_first_us: 14.35 }, group: { velocity_m_s: 5441, r_squared: .99996 },
  rows: [0, 20, 40, 60].map((d, i) => ({ index: i + 1, name: `P${i + 1}`, point_mm: [20 + d, 1.25, 1], distance_mm: d, window_us: [-2.5 + d / 5.441, 35 + d / 5.441], arrival_us: 16 + d / 5.441, usable: i < 3,
    ...(i < 3 ? { a1_m: 1.588e-7, phase1_deg: 40 - 30 * i, a2_m: 9e-11 + 4.5e-11 * i, a2_basis_m: 4.5e-11 * (i + 1), a2_nonlinear_m: 4.5e-11 * (i + 1), a2_geometric_m: 1.3e-11 * (i + 1), a2_material_m: 5.8e-11 * (i + 1), phase2_deg: -70 - 60 * i,
      ratio_per_m: 1800 * (i + 1) } : { reason: '时间窗超出记录的末尾' }) })),
  fit: { range_mm: [0, 20], points: 2, ratio: { slope_per_m2: 90000, intercept_per_m: 1800, r_squared: null }, a2: { slope_m_per_m: 2.25e-9, intercept_m: 4.5e-11, r_squared: null }, f: wave(.23145, 'S0', 3e-5), '2f': { ...wave(.4640, 'S0', 5e-4), forced: { wavenumber_rad_mm: .4629, difference: .0024 } },
    pair: { primary: 'S0', secondary: 'S0', mismatch: -.0019, delta_k_rad_mm: 8.84e-4, coherence_length_mm: 3555, group_velocity_primary_m_s: 5422.8, group_velocity_secondary_m_s: 5391.4 }, a1_variation: .001 },
  notes: ['1 个点的时间窗超出记录的末尾，未参与分析。'], note: 'A₂/A₁² 是相对非线性参数。', ...extra });

test('accumulation: the fit, the wavenumbers against the theory, the charts and the table', () => {
  const a = accumulation(), html = accumulationMarkup(a, 900);
  for (const text of ['90000 m⁻²', '每毫米 90 m⁻¹', '拟合 0–20 mm 内的 2 个点', '非线性增量（完整 − 线性）', '理论 S0', '实测相差 +0.003%', '受迫波 2k(f) = 0.4629 rad/mm；实测相差 +0.240%', 'S0 → S0：相速度失配 −0.190%', 'π/|Δk| = 3560 mm', '5441 m/s', '实测群速度', '时间窗超出记录的末尾', 'A₂ 几何 / pm', 'nl-notes']) assert.ok(html.includes(text), text);
  assert.equal((html.match(/<tr class="nl-out">/g) || []).length, 2);   // the point outside the fit range and the unusable one
  assert.equal((html.match(/<tr class="">/g) || []).length, 2);
  const charts = accumulationCharts(a, 900);
  assert.ok(charts.ratio.includes('stroke-dasharray="6 4"') && charts.ratio.includes('直线拟合（0–20 mm）') && charts.ratio.includes('未参与拟合'));
  assert.equal((charts.ratio.match(/r="4.5"/g) || []).length, 3); assert.ok(charts.ratio.includes('viewBox="0 0 900 '));
  for (const name of ['A₂ 非线性增量', 'A₂ 几何', 'A₂ 材料', 'A₂ 完整响应']) assert.ok(charts.second.includes(name), name);
  assert.ok(charts.second.includes('viewBox="0 0 441 ') && charts.first.includes('viewBox="0 0 441 '));   // two charts side by side
  assert.ok(accumulationCharts(a, 700).first.includes('viewBox="0 0 700 '));
  const fixed = accumulationMarkup(accumulation({ a2_basis: 'full', window: { mode: 'fixed', start_us: 0, end_us: 60 }, fit: { ...a.fit, f: { ...wave(-.118, null, 0), phase_velocity_m_s: null }, '2f': { ...wave(.1, null, 0), reliable: false }, pair: null } }), 900);
  for (const text of ['A₂ 取完整响应', '各点同一时间窗 0–60 μs', '点距过大，波数不可靠', '点距只能唯一确定 |k| &lt; 0.157 rad/mm', '没有认出一次模态']) assert.ok(fixed.includes(text), text);
  assert.ok(!fixed.includes('理论 S0'));
  // a mismatched pair: the 2f phase follows the forced wave and is on no mode
  const off = accumulationMarkup(accumulation({ fit: { ...a.fit, '2f': { ...wave(1.178, 'S0', -.112), forced: { wavenumber_rad_mm: 1.1777, difference: 3e-4 } }, pair: { ...a.fit.pair, mismatch: -.1126, delta_k_rad_mm: .1494, coherence_length_mm: 21.02 } } }), 900);
  for (const text of ['最接近的理论模态 S0', '不是这个模态', '实测相差 +0.030%', 'π/|Δk| = 21 mm']) assert.ok(off.includes(text), text);
});

test('accumulation CSV: the fit in the header, one row per point, unusable points left blank', () => {
  const lines = accumulationCsv(accumulation()).split('\n'), head = lines.findIndex(l => l.startsWith('point,')), columns = lines[head].split(',');
  assert.ok(lines.includes('# ratio_slope_per_m2,90000') && lines.includes('# theory_mode_f,S0') && lines.includes('# theory_pair,S0-S0') && lines.includes('# theory_coherence_length_mm,3555') && lines.includes('# a2_taken_from,nonlinear') && lines.includes('# fit_range_mm,0,20'));
  assert.equal(lines.length, head + 5); assert.ok(lines.slice(head).every(l => l.split(',').length === columns.length));
  const row = Object.fromEntries(columns.map((c, i) => [c, lines[head + 2].split(',')[i]])), last = Object.fromEntries(columns.map((c, i) => [c, lines[head + 4].split(',')[i]]));
  assert.deepEqual([row.point, row.distance_mm, row.in_fit, row.a2_over_a1_squared_per_m, row.a2_nonlinear_m], ['P2', '20', '1', '3600', '9e-11']);
  assert.deepEqual([last.point, last.in_fit, last.a1_m, last.a2_over_a1_squared_per_m], ['P4', '0', '', '']);
});

const mode = (name, family, k, cp, cg) => ({ name, family, wavenumber_rad_mm: k, phase_velocity_m_s: cp, group_velocity_m_s: cg, wavelength_mm: 2 * Math.PI / k, backward: false, secondary: family === 'S', surface: family === 'SH' ? [0, 1, 0] : [1, 0, .06] });
const branch = (name, family, speed) => { const k = Array.from({ length: 30 }, (_, i) => i * .05); return { name, family, secondary: family === 'S', wavenumber_rad_mm: k, frequency_mhz: k.map(v => v * speed / (2 * Math.PI)), group_velocity_m_s: k.map(() => speed * 1000) }; };
const theory = () => ({ schema: 'workai.plate_dispersion.v1', thickness_mm: 1, angle_deg: 0, frequency_mhz: .2, frequency_max_mhz: .52, symmetry: { mid_plane: true, sagittal_plane: true },
  families: { A: '反对称 Lamb', S: '对称 Lamb', SH: '水平剪切 SH' }, validity: { plane_strain: true, note: '两个 y 面都是对称面。' }, curves: [branch('S0', 'S', 5.43), branch('SH0', 'SH', 3.109), branch('A0', 'A', 1.5)],
  modes: { f: [mode('A0', 'A', .96, 1308, 2306), mode('SH0', 'SH', .404, 3109, 3109), mode('S0', 'S', .2314, 5429.6, 5422.8)], '2f': [mode('A0', 'A', 1.448, 1735, 2786), mode('S0', 'S', .4638, 5419.2, 5391.4)] },
  pairs: [{ primary: 'S0', secondary: 'S0', mismatch: -.0019, delta_k_rad_mm: 8.84e-4, coherence_length_mm: 3555, phase_velocity_primary_m_s: 5429.6, phase_velocity_secondary_m_s: 5419.2, group_velocity_primary_m_s: 5422.8, group_velocity_secondary_m_s: 5391.4 },
    { primary: 'A0', secondary: 'S0', mismatch: 3.14, delta_k_rad_mm: -1.46, coherence_length_mm: 2.16, phase_velocity_primary_m_s: 1308, phase_velocity_secondary_m_s: 5419.2, group_velocity_primary_m_s: 2306, group_velocity_secondary_m_s: 5391.4 }],
  note: '相位匹配是必要条件。' });

test('dispersion chart: one colour per family, the modes at f and 2f, measured points as diamonds', () => {
  assert.deepEqual([FAMILY_COLOURS.S, FAMILY_COLOURS.A, FAMILY_COLOURS.SH], SCAN_SERIES.slice(0, 3));
  const d = theory(), plain = dispersionChart(d, 'cp', [], 900), marked = dispersionChart(d, 'cp', [{ frequency_mhz: .2, wavenumber_rad_mm: .2315, group_velocity_m_s: 5441, label: '实测 f' }, { frequency_mhz: .4, wavenumber_rad_mm: .464, group_velocity_m_s: null, label: '实测 2f' }], 900);
  assert.equal((plain.match(/<polyline/g) || []).length, 3); for (const colour of SCAN_SERIES.slice(0, 3)) assert.ok(plain.includes(`stroke="${colour}"`), colour);
  assert.ok(plain.indexOf('对称 Lamb') < plain.indexOf('反对称 Lamb') && plain.indexOf('反对称 Lamb') < plain.indexOf('水平剪切 SH'));   // a fixed order in the legend
  assert.ok(plain.includes('f = 0.2 MHz') && plain.includes('2f = 0.4 MHz') && plain.includes('相速度 (km/s)') && plain.includes('clip-path="url(#nl-clip-'));
  assert.equal((plain.match(/r="4.5"/g) || []).length, 5); assert.ok(!plain.includes('本次计算实测')); assert.ok(plain.includes('S0（2f）：相速度 5419.2 m/s'));
  assert.equal((marked.match(/l6 7l-6 7l-6 -7z/g) || []).length, 2); assert.ok(marked.includes('本次计算实测') && marked.includes('实测 f：相速度 5.4282 km/s'));
  const group = dispersionChart(d, 'cg', [{ frequency_mhz: .2, wavenumber_rad_mm: .2315, group_velocity_m_s: 5441, label: '实测 f' }, { frequency_mhz: .4, wavenumber_rad_mm: .464, group_velocity_m_s: null, label: '实测 2f' }], 900);
  assert.equal((group.match(/l6 7l-6 7l-6 -7z/g) || []).length, 1); assert.ok(group.includes('群速度 (km/s)'));   // no group velocity was measured at 2f
  assert.ok(dispersionChart(d, 'k', [], 900).includes('波数 (rad/mm)'));
  const tables = dispersionTables(d);
  for (const text of ['f = 0.2 MHz 处的模态', '2f = 0.4 MHz 处的模态', 'S0 → S0', '−0.190%', '3560', 'A0 → S0', '+314.000%', '1.00 : 0.00 : 0.06', '相位匹配是必要条件。']) assert.ok(tables.includes(text), text);
  assert.equal((tables.match(/class="nl-match"/g) || []).length, 1);
});

test('f–k image: decoded with its axes, the peak found', () => {
  const rows = 4, columns = 5, values = new Uint8Array(rows * columns); values[2 * columns + 3] = 255; values[7] = 90;
  const fk = { image: Buffer.from(values).toString('base64'), frequency: { first_mhz: 0, step_mhz: .1, count: rows }, wavenumber: { first_rad_mm: -1, step_rad_mm: .5, count: columns, nyquist_rad_mm: 1.2 }, frequency_mhz: .2 };
  const grid = fkGrid(fk); assert.equal(grid.values[13], 255); assert.equal(grid.frequency(2), .2); assert.equal(grid.wavenumber(3), .5);
  assert.deepEqual(fkPeak(fk), { frequency_mhz: .2, wavenumber_rad_mm: .5 });
  assert.throws(() => fkGrid({ ...fk, frequency: { ...fk.frequency, count: 3 } }), /大小不一致/);
});

const check = () => ({ scan_id: 'scan-' + 'b'.repeat(24), name: 'S0 板 · 收敛检查', state: 'completed', message: '全部组合已完成。', count: 2, counts: { queued: 0, running: 0, completed: 2, failed: 0, paused: 0 }, kind: 'convergence' });
const receiver = (name, change) => ({ name, a1_m: 4.33e-8, a2_m: 1.19e-11, a1_change: change, a2_change: 60 * change, phase1_change_deg: 2e-5, waveform_change: 5e-7, increment_waveform_change: 1.6e-6 });
const compared = (change = -2.2e-7) => ({ window_us: null, frequency_mhz: .2, a2_basis: 'nonlinear', tolerance: { a1: .01, a2: .05 }, ratio: .75, pending: [], note: '每加密一级：阶次加一。',
  rows: [{ index: 0, degree: 4, h_mm: 2.303, node_spacing_mm: .57575, label: '4 阶 · 单元 2.303 mm', job_id: 'local-' + 'a'.repeat(32), nodes: 14265, elapsed_s: 56, layer_step_us: .008333, scheme: 'verlet2', is_reference: false,
    receivers: [receiver('P1', change), receiver('P2', change / 2)], worst: { a1_change: Math.abs(change), a2_change: 60 * Math.abs(change), phase1_change_deg: 2e-5, waveform_change: 5e-7, increment_waveform_change: 1.6e-6 } },
  { index: 1, degree: 5, h_mm: 2.303, node_spacing_mm: .4606, label: '5 阶 · 单元 2.303 mm', job_id: 'local-' + 'c'.repeat(32), nodes: 26136, elapsed_s: 95, layer_step_us: .008333, scheme: 'verlet2', is_reference: true,
    receivers: [receiver('P1', 0), receiver('P2', 0)], worst: { a1_change: 0, a2_change: 0, phase1_change_deg: 0, waveform_change: 0, increment_waveform_change: 0 } }],
  verdict: { converged: Math.abs(change) < .01, reference: '5 阶 · 单元 2.303 mm', a1_change: Math.abs(change), a2_change: 60 * Math.abs(change), text: '原网格与最细网格相比，变化很小。' } });

test('convergence: the plan before the start, then the verdict and the comparison of the meshes', () => {
  const plan = { job_id: 'local-' + 'a'.repeat(32), levels: 1, ratio: .75, tolerance: { a1: .01, a2: .05 }, base: { degree: 4, h_mm: 2.303, label: '4 阶 · 单元 2.303 mm', nodes: 14265, seconds: 56 },
    cases: [{ degree: 5, h_mm: 2.303, label: '5 阶 · 单元 2.303 mm', ok: true, errors: [], nodes: 26136, seconds: 66, memory_bytes: 869582112, disk_bytes: 3532750848, record: 'full' }], ok: true, seconds_total: 66, memory_bytes_max: 869582112,
    disk_bytes_total: 3532750848, free_bytes: 8e10, disk_short: false, existing: [] };
  assert.ok(convergenceMarkup().includes('正在检查加密网格'));
  const ready = convergenceMarkup({ plan });
  for (const text of ['4 阶 · 单元 2.303 mm（已算）', '26,136', '开始收敛检查（再算 1 次，约 66 秒）', 'id="nl-levels"', '3.29 GiB']) assert.ok(ready.includes(text), text);
  assert.ok(!/id="nl-converge"[^>]*disabled/.test(ready)); assert.ok(/id="nl-converge"[^>]*disabled/.test(convergenceMarkup({ plan, busy: true })));
  const blocked = convergenceMarkup({ plan: { ...plan, cases: [{ ...plan.cases[0], ok: false, errors: ['模型太大：需要 40 GiB 内存。'] }], ok: false, disk_short: true } });
  assert.ok(/id="nl-converge"[^>]*disabled/.test(blocked) && blocked.includes('模型太大：需要 40 GiB 内存。') && blocked.includes('可用空间'));
  assert.ok(convergenceMarkup({ error: '快速引擎在模型自带的粗网格上运行。' }).includes('快速引擎') && !convergenceMarkup({ error: 'x' }).includes('nl-converge'));
  const done = convergenceMarkup({ plan, check: check(), table: compared() });
  for (const text of ['最近一次检查：S0 板 · 收敛检查', 'data-scan="scan-' + 'b'.repeat(24), '<b>已收敛</b>', '5 阶 · 单元 2.303 mm（参考）', '其他网格与它比较', '2.2e-5%', '−2.2e-5%', '各接收点相对 5 阶 · 单元 2.303 mm 的变化', '8.333', 'A₂ 取非线性增量']) assert.ok(done.includes(text), text);
  assert.ok(convergenceMarkup({ plan, check: check() }).includes('正在读取比较结果'));
  const coarse = convergenceTable(compared(.03)); assert.ok(coarse.includes('nl-verdict warn') && coarse.includes('未达到判据') && coarse.includes('+3.000%') && coarse.includes('3.00%'));
  const waiting = convergenceTable({ ...compared(), rows: compared().rows.slice(0, 1), verdict: null, pending: [{ index: 1, label: '5 阶 · 单元 2.303 mm', phase: 'failed', message: '内存不足' }] });
  assert.ok(waiting.includes('还不能比较') && waiting.includes('未完成：内存不足')); assert.ok(convergenceTable({ ...compared(), rows: [] }).includes('还没有完成的网格'));
});

test('the panel offers what the record allows', () => {
  const result = (laws, compact) => ({ scene: { receivers: [{ name: 'P1', point_mm: [20, 1, 1], direction: [1, 0, 0] }, { name: 'P2', point_mm: [40, 1, 1], direction: [1, 0, 0] }], geometry: { kind: 'plate', length_mm: 100 } },
    controls: laws.map(law => ({ law })), recording: { record_mode: compact ? 'compact' : 'full' } });
  const full = nonlinearMarkup(result(['linear', 'stvk', 'murnaghan'], false)), compact = nonlinearMarkup(result(['murnaghan'], true)), two = nonlinearMarkup(result(['linear', 'murnaghan'], false));
  assert.ok(full.includes('<option value="line">') && full.includes('使用接收点（2 个）') && full.includes('data-nl-tab="convergence"'));
  assert.deepEqual([...full.matchAll(/<option value="(full|linear|nonlinear|geometric|material)">/g)].map(m => m[1]), ['full', 'linear', 'nonlinear', 'geometric', 'material']);
  assert.deepEqual([...two.matchAll(/<option value="(full|linear|nonlinear|geometric|material)">/g)].map(m => m[1]), ['full', 'linear', 'nonlinear']);
  assert.ok(!compact.includes('<option value="line">') && compact.includes('精简记录') && [...compact.matchAll(/<option value="(full|linear|nonlinear)">/g)].length === 1);
});

test('the shared chart draws a fit as a dashed line without markers and many points as small dots', () => {
  const svg = seriesChart([{ name: '数据', small: true, points: Array.from({ length: 50 }, (_, i) => [i, i * i]) }, { name: '拟合', points: [[0, 0], [49, 2401]], markers: false, dash: true, colour: '#173147' }], 'x', 'y');
  assert.equal((svg.match(/r="2.2"/g) || []).length, 50); assert.equal((svg.match(/r="4.5"/g) || []).length, 0); assert.ok(svg.includes('stroke="#173147"') && svg.includes('background:#173147') && svg.includes('stroke-dasharray'));
  assert.ok(seriesChart([{ name: 'a', points: [[0, 0], [1, 2]] }], 'x', 'y').includes('>0</text>'));   // zero is written as 0
});
