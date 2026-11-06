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
