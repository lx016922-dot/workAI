import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  V2, upgrade, addSource, removeSource, addBoundary, removeBoundary, addReceiver, removeReceiver, setSourceFace, setSignalType, setProfileType,
  parseSamples, tree, workbenchTemplates, faceCenter, inwardNormal, extent, defaultSignal, sourcesOf,
} from '../offline/web/scene-model.js';
import { displayScale, rotation, pick } from '../offline/web/model-view.js';
import { plot, signalsMarkup } from '../offline/web/signal-view.js';

const v1 = JSON.parse(await readFile(new URL('../offline/example-request.json', import.meta.url), 'utf8')).scene;

test('version-1 scenes upgrade to equivalent version-2 scenes', () => {
  const hann = upgrade({ ...v1, signal: { kind: 'hann' } });
  assert.equal(hann.schema, V2); assert.equal(hann.signal, undefined);
  assert.deepEqual(hann.source.signal, { type: 'tone_burst', window: 'hann' }); assert.deepEqual(hann.sources, []); assert.deepEqual(hann.boundaries, []);
  const chirp = upgrade({ ...v1, signal: { kind: 'chirp', end_frequency_mhz: .28 } });
  assert.equal(chirp.source.signal.type, 'chirp');
  assert.ok(Math.abs(chirp.source.signal.duration_us - v1.source.cycles / ((v1.source.frequency_mhz + .28) / 2)) < 1e-6);
  assert.equal(upgrade({ ...v1, signal: { kind: 'gaussian' } }).source.signal.window, 'gaussian');
  assert.equal(upgrade(hann).source.signal.window, 'hann');
});

test('sources, boundaries and receivers can be added and removed within the schema limits', () => {
  const s = upgrade(v1);
  const index = addSource(s, 'zmax'); assert.equal(index, 1); assert.equal(sourcesOf(s).length, 2);
  const extra = s.sources[0]; assert.deepEqual(extra.direction, [0, 0, -1]); assert.deepEqual(extra.center_mm, faceCenter(s.geometry, 'zmax'));
  setSourceFace(s, extra, 'ymin'); assert.deepEqual(extra.direction, [0, 1, 0]); assert.equal(extra.center_mm[1], 0);
  setProfileType(extra, 'comb'); assert.notEqual(extra.profile.along, 'y');
  for (const type of ['gaussian_pulse', 'ricker', 'chirp', 'multi_tone', 'samples', 'tone_burst']) { setSignalType(extra, type); assert.equal(extra.signal.type, type); }
  assert.throws(() => removeSource(s, 0)); removeSource(s, 1); assert.equal(s.sources, undefined);
  const b = addBoundary(s, 'absorbing'); assert.equal(s.boundaries[b].type, 'absorbing'); assert.ok(s.boundaries[b].length_mm > 0);
  assert.notEqual(s.boundaries[b].face, s.source.face);
  for (let i = 1; i < 8; i++) addBoundary(s, 'symmetry');
  assert.throws(() => addBoundary(s, 'symmetry'), /8/); removeBoundary(s, 0); assert.equal(s.boundaries.length, 7);
  const r = addReceiver(s); assert.equal(s.receivers[r].name, `P${r + 1}`);
  while (s.receivers.length > 1) removeReceiver(s, 0);
  assert.throws(() => removeReceiver(s, 0));
});

test('face geometry helpers follow the plate and pipe coordinate conventions', () => {
  const plate = { kind: 'plate', length_mm: 10, width_mm: 4, thickness_mm: 2 };
  assert.deepEqual(faceCenter(plate, 'xmax'), [10, 2, 1]); assert.deepEqual(faceCenter(plate, 'zmin'), [5, 2, 0]);
  assert.deepEqual(inwardNormal('xmax'), [-1, 0, 0]); assert.deepEqual(inwardNormal('zmin'), [0, 0, 1]);
  const pipe = { kind: 'pipe', length_mm: 10, outer_radius_mm: 2.5, inner_radius_mm: 1.5 };
  assert.deepEqual(extent(pipe), { size: [10, 5, 5], min: [0, -2.5, -2.5] }); assert.deepEqual(faceCenter(pipe, 'xmin'), [0, 0, 0]);
});

test('sampled signals parse from one or two columns and reject uneven time', () => {
  const values = Array.from({ length: 10 }, (_, i) => Math.sin(i));
  assert.deepEqual(parseSamples(values.join('\n')), { sample_step_us: null, values });
  const two = parseSamples('# t, v\n' + values.map((v, i) => `${(i * .02).toFixed(2)},${v}`).join('\n'));
  assert.equal(two.sample_step_us, .02); assert.equal(two.values.length, 10);
  assert.throws(() => parseSamples('1\n2\n3'), /8/);
  assert.throws(() => parseSamples(values.map((v, i) => `${i * i},${v}`).join('\n')), /等间隔/);
  const s = defaultSignal('samples', { frequency_mhz: .2 }); assert.ok(s.values.length >= 8 && s.values.length <= 20000); assert.ok(Math.max(...s.values.map(Math.abs)) <= 1);
});

test('the model tree lists every item and the templates are version-2 scenes', () => {
  const templates = workbenchTemplates({ plate: v1 });
  assert.deepEqual(Object.keys(templates), ['plate', 'gaussian', 'twopatch']);
  for (const t of Object.values(templates)) assert.equal(t.scene.schema, V2);
  const c = templates.twopatch.scene; assert.equal(sourcesOf(c).length, 2); assert.equal(c.sources[0].delay_us, 1); assert.deepEqual(c.sources[0].direction, [0, 0, 1]);
  const rows = tree(c), ids = rows.map(r => r.id);
  for (const id of ['geometry', 'material', 'source:0', 'source:1', 'boundary:0', 'boundary:2', 'receiver:0', 'study']) assert.ok(ids.includes(id), id);
  assert.equal(rows.find(r => r.id === 'source:0').removable, false); assert.equal(rows.find(r => r.id === 'source:1').removable, true);
});

test('view helpers: rotations are orthonormal, thin directions are magnified, picks choose the nearest target', () => {
  for (const [yaw, pitch] of [[0, 0], [-.62, .52], [1.2, -.4]]) {
    const R = rotation(yaw, pitch);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assert.ok(Math.abs(R[i].reduce((s, v, k) => s + v * R[j][k], 0) - (i === j ? 1 : 0)) < 1e-12);
  }
  assert.deepEqual(displayScale([48, 2, 1.6], false), [1, 1, 1]);
  const scale = displayScale([48, 2, 1.6]); assert.equal(scale[0], 1); assert.ok(Math.abs(2 * scale[1] - 48 * .22) < 1e-9);
  assert.equal(pick([{ id: 'a', x: 0, y: 0, r: 10 }, { id: 'b', x: 6, y: 0, r: 10 }], 5, 0), 'b');
  assert.equal(pick([{ id: 'a', x: 0, y: 0, r: 4 }], 9, 0), null);
});

test('signal plots render the preview with f, 2f and band markers', () => {
  const svg = plot([0, 1, 2], [0, 1, 0], { markers: [{ x: 1, label: 'f', color: '#000' }, { x: 5, label: 'out', color: '#000' }] });
  assert.match(svg, /<svg/); assert.match(svg, />f</); assert.doesNotMatch(svg, />out</);
  const preview = { signals: [{ name: '主激励', type: 'tone_burst', frequency_mhz: .2, support_us: [0, 25], band_30db_mhz: .27, times_us: [0, 1, 2], values_mpa: [0, 1, 0], spectrum_mhz: [0, .2, .4], spectrum_db: [-60, 0, -40] }],
    resolution: [{ name: '主激励', nodes_per_2f_shear_wavelength: 9.5 }] };
  const html = signalsMarkup(preview, 1000); assert.match(html, /9\.5 个节点/); assert.equal((html.match(/<svg/g) || []).length, 2);
  assert.match(signalsMarkup({ signals: [], errors: ['bad'] }), /bad/);
});

test('material section: engineering constants, summaries and the derived panel', async () => {
  const { tiFromEngineering, materialDerivedMarkup, materialMarkup } = await import('../offline/web/material-form.js');
  const { materialKind, materialSummary } = await import('../offline/web/scene-model.js');
  const C = tiFromEngineering({ E1: 135, E2: 9, G12: 5, nu12: .3, nu23: .45 });
  assert.ok(Math.abs(C.C11 - 138.01115) < 1e-4 && Math.abs(C.C12 - 5.0185874) < 1e-6 && Math.abs(C.C22 - 11.467761) < 1e-5 && Math.abs(C.C23 - 5.260864) < 1e-5 && C.C55 === 5);
  assert.throws(() => tiFromEngineering({ E1: 1, E2: 1, G12: 1, nu12: 0, nu23: 1 }));
  assert.equal(materialKind(v1.material), 'reference'); assert.equal(materialKind({ model: 'energy' }), 'energy'); assert.equal(materialKind({ model: 'murnaghan' }), 'murnaghan');
  assert.match(materialSummary({ model: 'energy', name: 'CFRP', fiber_angle_deg: 30 }), /CFRP.*30°/);
  const energy = { model: 'energy', name: 'CFRP', rho_kg_m3: 1560, fiber_angle_deg: 0, energy: '0.5*C11*E11**2', parameters: { C11: 138 } };
  const form = materialMarkup({ material: energy }, { cfrp_ud: { label: 'CFRP', material: energy } });
  assert.match(form, /data-path="material.energy"/); assert.match(form, /data-path="material.parameters.C11"/); assert.match(form, /material-derived/);
  assert.match(materialDerivedMarkup({ ok: false, error: '参考构形必须无应力' }), /无应力/);
  const panel = materialDerivedMarkup({ ok: true, voigt_c2_gpa: Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => (i === j ? 10 : 0))), eigenvalues_gpa: [10], c3_nonzero_gpa: [],
    speeds_m_s: { fiber: [9406, 1790, 1790], in_plane_transverse: [2711, 1790, 1410], thickness: [2711, 1790, 1410] }, fastest_m_s: 9406, slowest_shear_m_s: 1410,
    truncation: { remainder_relative: 1e-16, relative_to: 'quadratic' }, check_stress_mpa: [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]], engine: '物理 AI' });
  assert.match(panel, /9406/); assert.match(panel, /全部为 0/); assert.match(panel, /COMSOL/);
});

test('result validation accepts the fast engine and nothing else new', async () => {
  const { validateSpectralResult } = await import('../offline/web/spectral-data.js');
  const base = { schema: 'workai.offline.graphwave.result.v1', neural_model: { numerical_correction: false, weights_sha256: 'a'.repeat(64) } };
  for (const engine of ['workai_graphwave_offline', 'workai_coarse_graph_offline'])
    assert.throws(() => validateSpectralResult({ ...base, engine }), err => !/离线网络结果格式不正确/.test(err.message));  // passes the engine gate, fails later on the empty layout
  assert.throws(() => validateSpectralResult({ ...base, engine: 'other' }), /离线网络结果格式不正确/);
});

test('defects: one line each, add / remove on plates only, tree folder, variant switches and the 3D view', async () => {
  const { defectSummary, addDefect, removeDefect, treeNodes, DEFECT_LABELS, defectName } = await import('../offline/web/scene-model.js');
  const { reshapeDefect, inspectorMarkup } = await import('../offline/web/inspector.js');
  const { drawScene } = await import('../offline/web/model-view.js');
  const s = workbenchTemplates({ plate: v1 }).gaussian.scene;
  assert.deepEqual(Object.keys(DEFECT_LABELS), ['crack', 'delamination', 'contact', 'notch', 'hole']);
  const crack = { type: 'crack', x_mm: 24, depth_mm: .5, from: 'top', closed: true };
  assert.equal(addDefect(s, crack), 0); assert.equal(addDefect(s, { type: 'hole', x_mm: 30, y_mm: 1, radius_mm: .5 }), 1);
  assert.match(defectSummary(crack), /x 24 · 上表面开口 · 深 0.5 mm · 闭合/); assert.equal(defectName(crack, 0), '裂纹 1');
  assert.match(defectSummary(s.defects[1]), /通孔 Φ1/);
  const folder = treeNodes(s, { mesh: 'h 1.2 mm' }).children[0].children.find(n => n.id === 'defects');
  assert.deepEqual(folder.children.map(n => n.id), ['defect:0', 'defect:1']); assert.equal(folder.add, 'defect');
  assert.equal(treeNodes(s, { mesh: 'h 1.2 mm' }).children[0].children.find(n => n.id === 'mesh').detail, 'h 1.2 mm');
  assert.match(inspectorMarkup(s, 'defect:0', 4, {}), /闭合（接触，呼吸裂纹）/);
  assert.match(inspectorMarkup(s, 'mesh', 4, { meshAuto: { h_mm: 1.2, degree: 4, node_spacing_mm: .3, min_node_spacing_mm: .1, nodes_per_2f_shear_wavelength: 10, target: 10, band_top_mhz: 1, nodes: 9000, elements: 100, reason: '按 2f', split_nodes: 36, contact_pairs: 36 } }), /36 对接触/);
  assert.doesNotMatch(inspectorMarkup(s, 'mesh', 4, {}), /data-path="numerics.h_mm"/);   // no mesh inputs: the software chooses
  const contact = reshapeDefect({ type: 'contact', orientation: 'horizontal', x_mm: 20 }, s);
  assert.equal(contact.x_mm, undefined); assert.ok(contact.x_end_mm > contact.x_start_mm); assert.ok(contact.depth_mm > 0);
  const hole = reshapeDefect({ type: 'hole', axis: 'y', x_mm: 30, y_mm: 1, radius_mm: .5 }, s); assert.equal(hole.y_mm, undefined); assert.equal(hole.z_mm, .8);
  const ctx = new Proxy({}, { get: (_, k) => (k === 'measureText' ? () => ({ width: 10 }) : () => {}), set: () => true });
  const canvas = { clientWidth: 800, clientHeight: 500, width: 800, height: 500, getContext: () => ctx };
  const picks = drawScene(canvas, s, { yaw: -.6, pitch: .5, zoom: 1, exaggerate: true, grid: false, selected: 'defect:0' });
  assert.ok(picks.some(p => p.id === 'defect:0') && picks.some(p => p.id === 'defect:1'));
  removeDefect(s, 0); removeDefect(s, 0); assert.equal(s.defects, undefined);
  const pipe = structuredClone(s); pipe.geometry.kind = 'pipe'; assert.throws(() => addDefect(pipe, crack), /平板/);
});

test('large models: display results validate, compact records are told apart, estimates and solver settings are shown', async () => {
  const { validateSpectralResult } = await import('../offline/web/spectral-data.js');
  const { solidResultMarkup } = await import('../offline/web/solid-viewer.js');
  const { fmtDuration, estimateMarkup, previewMessages } = await import('../offline/web/shell.js');
  const { inspectorMarkup, RECORD_MODES, TIME_LAYERS } = await import('../offline/web/inspector.js');
  // a display result: 4 surface nodes of a 500 000-node computation, float32 frames, receivers in the compact record
  const frames = 21, nodes = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];
  const scene = { name: '大板', geometry: { kind: 'plate' }, source: { frequency_mhz: .4 }, acquisition: { duration_us: 30, sample_step_us: .05 }, receivers: [{ name: 'top_0', point_mm: [1, 1, 0], direction: [0, 0, 1] }] };
  const result = () => ({ schema: 'workai.offline.graphwave.result.v1', engine: 'workai_graphwave_offline', neural_model: { numerical_correction: false, weights_sha256: 'a'.repeat(64) },
    response_part: 'full', scene, nodes_mm: nodes, tetra: [], surface: [[0, 1, 2], [0, 2, 3]], loaded_faces: [0], times_us: Array.from({ length: frames }, (_, i) => i * 1.5),
    diagnostics: { output_dtype: 'float32', internal_step_us: .005, external_work_balance_relative: 1e-5, max_strain: 1e-5, min_J: .99999 },
    spectral: { degree: 4, gll: [-1, -.65, 0, .65, 1], mapping_kind: 'cartesian', node_count: 500000, element_count: 7000 },
    display: { nodes: 4, triangles: 2, stride: 4, cell_mm: 2.5, surface_faces: 9000, computation_nodes: 500000, computation_elements: 7000 },
    field: { layout: 'time,node,xyz', dtype: 'little_endian_float32', unit: 'm', shape: [frames, 4, 3], uncompressed_bytes: frames * 4 * 3 * 4 },
    recording: { schema: 'workai.graphnet.long_record.v1', duration_us: 30, sample_step_us: .05, frame_count: 601, animation_only: true, analysis_source: 'receiver_float64_probes', record_mode: 'compact',
      receivers: [{ name: 'top_0', point_mm: [1, 1, 0] }], snapshot_frames: [150, 300, 450, 600] } });
  assert.equal(validateSpectralResult(result()).display.nodes, 4);
  const markup = solidResultMarkup(result());
  assert.match(markup, /精简记录/); assert.match(markup, /500,000 节点 \/ 7,000 高阶六面体/); assert.match(markup, /显示网格：2 个表面三角形/); assert.match(markup, /按 2\.50 mm 合并/); assert.match(markup, /4 个时刻的完整 float64/);
  for (const tamper of [r => { r.display.nodes = 5; }, r => { r.spectral.node_count = 3; }, r => { r.field.dtype = 'little_endian_float64'; r.field.uncompressed_bytes *= 2; }, r => { delete r.recording; },
    r => { r.recording.analysis_source = 'other'; }, r => { r.recording.record_mode = 'full'; }, r => { r.surface.push([0, 1, 9]); r.display.triangles = 3; }]) {
    const bad = result(); tamper(bad); assert.throws(() => validateSpectralResult(bad));
  }
  // durations and the cost table
  assert.deepEqual([20, 600, 7200, 400000].map(fmtDuration), ['20 秒', '10 分钟', '2.0 小时', '4.6 天']);
  const estimate = { record: 'compact', scheme: 'verlet2', layer_step_us: .005, evaluations_per_control: 6000, frozen_evaluations_per_control: 23400, seconds: 420, rate_measured: false, memory_bytes: 2 ** 30, physical_memory_bytes: 2 ** 34, disk_bytes: 3 * 2 ** 28 };
  const table = estimateMarkup(estimate);
  assert.match(table, /精简记录/); assert.match(table, /二阶单级/); assert.match(table, /5\.00 ns/); assert.match(table, /1\/3\.9/); assert.match(table, /7 分钟/); assert.match(table, /典型台式机/); assert.equal(estimateMarkup(null), '');
  assert.match(estimateMarkup({ ...estimate, record: 'full', scheme: 'composition4', rate_measured: true }), /完整记录[\s\S]*三级四阶[\s\S]*本机上次/);
  const rows = previewMessages({ ok: true, errors: [], record: { samples: 601, uncompressed_bytes: 3 * 2 ** 28, mode: 'compact' }, estimate });
  assert.ok(rows.some(r => /精简记录/.test(r.text) && /接收点/.test(r.detail))); assert.ok(rows.some(r => /预计 7 分钟/.test(r.text)));
  assert.ok(previewMessages({ ok: true, errors: [], record: { samples: 601, uncompressed_bytes: 1e6, mode: 'full' } }).some(r => /完整记录约/.test(r.text) && !r.detail));
  // solver settings: record mode and time layer for the network engine only
  const s = upgrade(v1), network = inspectorMarkup(s, 'engine', 4, { engine: 'network', record: 'compact', timeStepping: 'frozen' });
  assert.match(network, /data-action="record-mode"/); assert.match(network, /value="compact" selected/); assert.match(network, /data-action="time-layer"/); assert.match(network, /value="frozen" selected/);
  assert.doesNotMatch(inspectorMarkup(s, 'engine', 4, { engine: 'coarse' }), /record-mode|time-layer/);
  assert.deepEqual([Object.keys(RECORD_MODES), Object.keys(TIME_LAYERS)], [['auto', 'full', 'compact'], ['auto', 'frozen']]);
});

test('selectable controls: the setting, the responses a job offers and the notes', async () => {
  const { inspectorMarkup, CONTROL_SETS } = await import('../offline/web/inspector.js');
  const { responseLabels, controlsNote, solidResultMarkup } = await import('../offline/web/solid-viewer.js');
  const { estimateMarkup } = await import('../offline/web/shell.js');
  assert.deepEqual(Object.keys(CONTROL_SETS), ['all', 'full_linear', 'full']);
  const s = upgrade(v1);
  for (const engine of ['network', 'coarse']) {   // both engines compute the selected controls
    const form = inspectorMarkup(s, 'engine', 4, { engine, controls: 'full_linear' });
    assert.match(form, /data-action="control-set"/); assert.match(form, /value="full_linear" selected/);
  }
  assert.match(inspectorMarkup(s, 'engine', 4, { engine: 'network' }), /value="all" selected/);
  assert.equal(responseLabels.nonlinear, '非线性增量（完整 − 线性）');
  const rows = laws => ({ controls: laws.map(law => ({ law })) });
  assert.match(controlsNote(rows(['linear', 'stvk', 'murnaghan'])), /材料增量 = 完整材料 − StVK/); assert.match(controlsNote({}), /三个冻结网络分支/);   // older results list no controls
  assert.match(controlsNote(rows(['linear', 'murnaghan'])), /非线性增量 = 完整 − 线性/); assert.match(controlsNote(rows(['murnaghan'])), /只计算了完整/);
  const result = { scene: { name: '板', geometry: { kind: 'plate' }, source: { frequency_mhz: .3 }, acquisition: { sample_step_us: .05 }, receivers: [{ name: 'P1' }] }, response_part: 'nonlinear', times_us: [0, 1], nodes_mm: [[0, 0, 0]],
    spectral: { elements: [[0]] }, controls: [{ law: 'linear' }, { law: 'murnaghan' }], diagnostics: { output_dtype: 'float64', internal_step_us: .005, external_work_balance_relative: 1e-5, max_strain: 1e-5, min_J: .99999 } };
  const markup = solidResultMarkup(result);
  assert.match(markup, /非线性增量（完整 − 线性）/); assert.match(markup, /几何与材料增量之和/); assert.match(markup, /位移差不是单独的物理状态/);
  const estimate = { record: 'full', scheme: 'verlet2', layer_step_us: .005, evaluations_per_control: 6000, frozen_evaluations_per_control: 23400, seconds: 30, rate_measured: true, memory_bytes: 2 ** 29, disk_bytes: 2 ** 28 };
  assert.match(estimateMarkup({ ...estimate, controls: ['murnaghan'], cpu_relative_to_all: .54 }), /1 组（完整）[^<]*54%/);
  assert.match(estimateMarkup({ ...estimate, controls: ['linear', 'murnaghan'], cpu_relative_to_all: .76 }), /2 组（线性、完整）[^<]*76%/);
  assert.doesNotMatch(estimateMarkup({ ...estimate, controls: ['linear', 'stvk', 'murnaghan'], cpu_relative_to_all: 1 }), /CPU 用量/);
});

test('the cost table says how the cores are used', async () => {
  const { estimateMarkup } = await import('../offline/web/shell.js');
  const estimate = { record: 'full', scheme: 'verlet2', layer_step_us: .005, evaluations_per_control: 6000, frozen_evaluations_per_control: 23400, seconds: 30, rate_measured: true, memory_bytes: 2 ** 29, disk_bytes: 2 ** 28,
    controls: ['linear', 'stvk', 'murnaghan'], cpu_relative_to_all: 1 };
  assert.match(estimateMarkup({ ...estimate, threads: 4, spread: true }), /4 个线程，各组控制分摊到全部线程/);
  assert.match(estimateMarkup({ ...estimate, threads: 2, spread: false }), /2 个线程，每组控制一个线程/);
  assert.doesNotMatch(estimateMarkup(estimate), /个线程/);   // older previews carry no thread information
});
