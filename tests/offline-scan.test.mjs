import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { upgrade, addSource, addBoundary, addDefect, treeNodes } from '../offline/web/scene-model.js';
import { MAX_CASES, QUANTITIES, SCAN_SERIES, parseValues, scanParameters, scanAxes, keepScan, pathLabel, axisText, caseText, gridValues, scanFormMarkup, scanSeries, seriesChart, scanPageMarkup } from '../offline/web/scan.js';

const v1 = JSON.parse(await readFile(new URL('../offline/example-request.json', import.meta.url), 'utf8')).scene;
const plate = () => { const s = upgrade(structuredClone(v1)); s.geometry.kind = 'plate'; return s; };

test('values: lists, ranges with both ends, and what is refused', () => {
  assert.deepEqual(parseValues('0.2, 0.3 0.4；0.5'), [.2, .3, .4, .5]);
  assert.deepEqual(parseValues('0.3:0.1:0.5'), [.3, .4, .5]); assert.deepEqual(parseValues('1:2:6, 10'), [1, 3, 5, 10]); assert.deepEqual(parseValues('2:1:2'), [2]);
  for (const [text, message] of [['', /请输入取值/], ['0.2, x', /不是数值/], ['1:0:2', /起点:步长:终点/], ['2:1:1', /起点:步长:终点/], ['1, 1', /重复/], ['0:1:500', /超过 100/], ['1:2', /不是数值/]]) assert.throws(() => parseValues(text), message);
});

test('parameters: the numeric fields of the model, named with their owner and unit', () => {
  const s = plate(); addSource(s, 'zmax'); addBoundary(s, 'absorbing'); addDefect(s, { type: 'crack', x_mm: 5, depth_mm: .5, from: 'top', closed: false });
  const rows = scanParameters(s), paths = rows.map(r => r.path);
  for (const path of ['source.frequency_mhz', 'source.amplitude_mpa', 'source.cycles', 'sources.0.frequency_mhz', 'geometry.thickness_mm', 'defects.0.depth_mm', 'defects.0.x_mm', 'boundaries.0.length_mm']) assert.ok(paths.includes(path), path);
  assert.ok(!paths.some(p => p.startsWith('material.')));   // the reference aluminium is not a set of free constants
  assert.ok(rows.every(r => Number.isFinite(r.value) && r.name && r.owner));
  assert.equal(pathLabel('source.frequency_mhz', s).text, `${s.source.name || '主激励'} · 频率 (MHz)`); assert.equal(pathLabel('defects.0.depth_mm', s).owner, '裂纹 1'); assert.equal(pathLabel('geometry.thickness_mm').text, '几何 · 厚度 (mm)');
  s.material = { ...s.material, model: 'murnaghan', name: 'x' }; assert.ok(scanParameters(s).some(r => r.path === 'material.l_pa'));
  s.geometry.kind = 'pipe'; assert.ok(scanParameters(s).some(r => r.path === 'geometry.outer_radius_mm')); assert.ok(!scanParameters(s).some(r => r.path === 'geometry.thickness_mm'));
});

test('a definition is checked against the model and carries the names of its axes', () => {
  const s = plate(), scan = { axes: [{ path: 'source.frequency_mhz', text: '0.2:0.05:0.3' }, { path: 'geometry.thickness_mm', text: '1.5, 2' }] };
  const checked = scanAxes(scan, s); assert.equal(checked.ready, true); assert.equal(checked.count, 6); assert.deepEqual(checked.errors, ['', '']);
  assert.deepEqual(checked.axes, [{ path: 'source.frequency_mhz', values: [.2, .25, .3], label: '频率', unit: 'MHz' }, { path: 'geometry.thickness_mm', values: [1.5, 2], label: '厚度', unit: 'mm' }]);
  addSource(s, 'zmax'); assert.match(scanAxes(scan, s).axes[0].label, /频率$/); assert.notEqual(scanAxes(scan, s).axes[0].label, '频率');   // two sources: the owner is named
  for (const [axes, error] of [[[{ path: '', text: '1' }], /请选择参数/], [[{ path: 'source.frequency_mhz', text: 'a' }], /不是数值/], [[{ path: 'defects.0.depth_mm', text: '1' }], /没有这个参数/],
    [[{ path: 'source.cycles', text: '2, 3' }, { path: 'source.cycles', text: '4' }], /已经选过/]]) { const c = scanAxes({ axes }, s); assert.equal(c.ready, false); assert.match(c.errors.join(' '), error); }
  const many = scanAxes({ axes: [{ path: 'source.frequency_mhz', text: '0.1:0.001:0.15' }, { path: 'source.amplitude_mpa', text: '1:1:5' }] }, s);
  assert.equal(many.count, 255); assert.equal(many.ready, false); assert.equal(many.tooMany, true); assert.ok(many.count > MAX_CASES);
  assert.equal(scanAxes({ axes: [] }, s).ready, false);
  assert.deepEqual(keepScan({ axes: [{ path: 'defects.0.depth_mm', text: '1' }, { path: 'source.cycles', text: '2, 3' }, { path: '', text: '' }, null] }, s), { axes: [{ path: 'source.cycles', text: '2, 3' }, { path: '', text: '' }] });
  assert.deepEqual(keepScan(undefined, s), { axes: [] });
});

test('the grid is named and indexed as the server orders it', () => {
  const axes = [{ path: 'source.amplitude_mpa', values: [1, 2], label: '幅值', unit: 'MPa' }, { path: 'source.frequency_mhz', values: [.3, .4, .5] }];
  assert.deepEqual([0, 1, 2, 3, 4, 5].map(i => gridValues(axes, i)), [[1, .3], [1, .4], [1, .5], [2, .3], [2, .4], [2, .5]]);
  assert.equal(caseText(axes, [2, .4]), '幅值 2 MPa，频率 0.4'); assert.equal(axisText(axes[0]), '幅值 (MPa)'); assert.equal(axisText(axes[1]), '主激励 · 频率 (MHz)');
});

test('the form lists the combinations and only starts a scan that passed the checks', () => {
  const s = plate(), scan = { axes: [{ path: 'source.amplitude_mpa', text: '1, 2' }] }, axes = scanAxes(scan, s).axes;
  assert.match(scanFormMarkup(s, { axes: [] }, null), /还没有参数/); assert.match(scanFormMarkup(s, scan, null), /正在检查 2 个组合/);
  const row = (index, ok) => ({ index, values: [index + 1], label: '', ok, errors: ok ? [] : ['幅值超出范围'], warnings: [], nodes: 5000, seconds: 30, memory_bytes: 3e8, disk_bytes: 1e8, record: 'full' });
  const plan = { axes, name: 'x', count: 2, valid: 2, cases: [row(0, true), row(1, true)], seconds_total: 60, memory_bytes_max: 3e8, disk_bytes_total: 2e8, free_bytes: 1e10, disk_short: false, full_records: 2 };
  const good = scanFormMarkup(s, scan, plan); assert.match(good, /data-scan-start >/); assert.match(good, /开始扫描（2 个组合）/); assert.match(good, /全部通过检查/); assert.doesNotMatch(good, /可用空间不够/);
  assert.match(scanFormMarkup(s, scan, plan, { busy: true }), /data-scan-start disabled/);
  const bad = scanFormMarkup(s, scan, { ...plan, valid: 1, cases: [row(0, true), row(1, false)] }); assert.match(bad, /data-scan-start disabled/); assert.match(bad, /1 个不能计算/); assert.match(bad, /幅值超出范围/);
  const short = scanFormMarkup(s, scan, { ...plan, disk_short: true, free_bytes: 1e8 }); assert.match(short, /可用空间不够/); assert.match(short, /精简记录/);
  assert.match(scanFormMarkup(s, scan, { error: '已有任务' }), /已有任务/);
  assert.match(scanFormMarkup(s, { axes: [{ path: 'source.frequency_mhz', text: '0.1:0.001:0.15' }, { path: 'source.amplitude_mpa', text: '1:1:5' }] }, null), /超过上限 200/);
});

const receiver = (a1, a2n) => ({ name: 'P1', point_mm: [8, 1, 1.6], direction: [0, 0, 1], a1_m: a1, a2_m: 3 * a2n, a2_nonlinear_m: a2n, a2_geometric_m: .3 * a2n, a2_material_m: .8 * a2n, a2_over_a1_squared_per_m: a2n / a1 ** 2, a2_basis: 'nonlinear' });
function table() {
  const axes = [{ path: 'source.amplitude_mpa', values: [1, 2], label: '幅值', unit: 'MPa' }, { path: 'source.frequency_mhz', values: [.3, .4, .5], label: '频率', unit: 'MHz' }];
  const rows = [0, 1, 2, 3, 4, 5].map(i => { const values = gridValues(axes, i), done = i !== 5;
    return { index: i, values, phase: done ? 'completed' : 'failed', job_id: `local-${String(i).repeat(32)}`, message: done ? '' : '磁盘空间不足', label: '', ...(done ? { nodes: 12000, elapsed_s: 40, frequency_mhz: values[1], receivers: [receiver(1e-9 * values[0], 1e-14 * values[0] ** 2 * (i % 3 + 1))] } : {}) }; });
  return { scan_id: 'scan-' + 'a'.repeat(24), name: '幅值与频率', state: 'completed', message: '1 个组合未完成，可重算。', axes, count: 6, counts: { queued: 0, running: 0, completed: 5, failed: 1, paused: 0 }, rows, window_us: null, note: '说明' };
}

test('curves: one series per combination of the other axes, over the axis with the most values or the one asked for', () => {
  const t = table(), byFrequency = scanSeries(t, 0, 'a2_nonlinear_m');
  assert.equal(byFrequency.axis, 1); assert.deepEqual(byFrequency.choices, [0, 1]); assert.deepEqual(byFrequency.series.map(s => s.name), ['幅值 1 MPa', '幅值 2 MPa']);
  assert.deepEqual(byFrequency.series.map(s => s.points.length), [3, 2]);   // the failed combination has no point
  assert.ok(Math.abs(byFrequency.series[0].points[1][1] - .02) < 1e-12); assert.match(byFrequency.series[0].points[1][2], /频率 0.4 MHz：A₂ 非线性增量 0.02 pm/);
  const byAmplitude = scanSeries(t, 0, 'a1_m', 0); assert.equal(byAmplitude.axis, 0); assert.equal(byAmplitude.series.length, 3); assert.equal(byAmplitude.series[0].points[1][1], 2);
  const svg = seriesChart(byFrequency.series, '频率 (MHz)', 'A₂ (pm)');
  assert.equal((svg.match(/<polyline/g) || []).length, 2); assert.equal((svg.match(/<title>/g) || []).length, 5); assert.match(svg, /scan-legend/); assert.ok(svg.includes(SCAN_SERIES[0]) && svg.includes(SCAN_SERIES[1]));
  assert.doesNotMatch(seriesChart([byFrequency.series[0]], 'x', 'y'), /scan-legend/); assert.equal(seriesChart([], 'x', 'y'), '');
  const one = { ...t, axes: [{ ...t.axes[0], values: [1] }, { ...t.axes[1], values: [.3] }] }; assert.equal(scanSeries(one, 0, 'a1_m').axis, -1);
});

test('the scan page: commands by state, the table with a row per combination, the reason of a failure', () => {
  const t = table(), detail = { ...t, cases: t.rows.map(r => ({ index: r.index, values: r.values, phase: r.phase, job_id: r.job_id, message: r.message, label: '' })) };
  const page = scanPageMarkup(detail, t, { receiver: 0, quantity: 'a2_nonlinear_m', window: null });
  assert.match(page.head, /5 \/ 6 个组合已完成，1 个未完成/); assert.match(page.head, /data-scan-act="retry"/); assert.match(page.head, /summary\.csv"/); assert.doesNotMatch(page.head, /data-scan-act="pause"/);
  assert.equal((page.body.match(/<tr class="completed open"/g) || []).length, 5); assert.match(page.body, /<tr class="failed open"[^>]*>[\s\S]*磁盘空间不足/); assert.match(page.body, /id="scan-axis"/);
  for (const q of QUANTITIES) assert.ok(page.body.includes(`${q.label} (${q.unit})`));
  assert.match(page.body, /A₂ 非线性增量 随 频率 \(MHz\) 的变化/); assert.match(page.body, /<th title="source\.amplitude_mpa">幅值 \(MPa\)<\/th>/);
  const running = scanPageMarkup({ ...detail, state: 'running', counts: { queued: 3, running: 1, completed: 2, failed: 0, paused: 0 }, cases: detail.cases.map((c, i) => ({ ...c, phase: i < 2 ? 'completed' : i === 2 ? 'running' : 'queued' })) }, null, {});
  assert.match(running.head, /data-scan-act="pause"/); assert.match(running.head, /正在计算第 3 个组合（幅值 1 MPa，频率 0.5 MHz）/); assert.match(running.head, /<progress/); assert.match(running.body, /正在读取汇总表/);
  assert.match(scanPageMarkup({ ...detail, state: 'paused' }, t, { window: [5, 20] }).head, /data-scan-act="resume"[\s\S]*start_us=5&end_us=20/);
  const full = { ...t, rows: t.rows.map(r => r.receivers ? { ...r, receivers: r.receivers.map(x => ({ ...x, a2_basis: 'full', a2_nonlinear_m: null })) } : r) };
  assert.match(scanPageMarkup(detail, full, {}).body, /没有算线性控制/);
});

test('the model tree: the scan node, and a scan as a folder of its combinations', () => {
  const s = plate(), t = table(), job = (i, scan) => ({ job_id: `local-${String(i).repeat(32)}`, name: `模型 · tail${i}`, phase: 'completed', ...(scan ? { scan: { id: t.scan_id, index: i, count: 6 } } : {}) });
  const jobs = [job(7, false), job(1, true), job(0, true)], scans = [{ scan_id: t.scan_id, name: t.name, state: 'running', axes: t.axes, count: 6, counts: t.counts }];
  const root = treeNodes(s, { jobs, scans, scan: '6 个组合', scanCase: (x, index) => caseText(x.axes, gridValues(x.axes, index)) });
  const study = root.children.find(n => n.id === 'study'), results = root.children.find(n => n.id === 'results');
  assert.deepEqual(study.children.map(n => n.id), ['engine', 'scan']); assert.equal(study.children[1].detail, '6 个组合');
  assert.deepEqual(results.children.map(n => n.id), [`scanrun:${t.scan_id}`, `job:${jobs[0].job_id}`]); assert.equal(results.count, 3);
  assert.deepEqual(results.children[0].children.map(n => n.label), ['1. 幅值 1 MPa，频率 0.3 MHz', '2. 幅值 1 MPa，频率 0.4 MHz']); assert.equal(results.children[0].count, '5/6');
  assert.deepEqual(treeNodes(s, { jobs, scans }).children.find(n => n.id === 'results').children[0].children.map(n => n.label), ['1. tail0', '2. tail1']);
  assert.equal(treeNodes(s, { jobs }).children.find(n => n.id === 'results').children.length, 3);   // without the scans every job is listed
});
