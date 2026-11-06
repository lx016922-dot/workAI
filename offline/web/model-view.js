// Canvas view of a scene: geometry, load patches, boundary faces, absorbing layers, receivers and element grid.
import { extent, sourcesOf } from './scene-model.js';

const COLORS = {
  body: [226, 233, 238], edge: '#5d7383', grid: 'rgba(60,90,110,.18)', source: '#d9480f', sourceFill: 'rgba(217,72,15,.22)',
  symmetry: 'rgba(28,110,214,.28)', antisymmetry: 'rgba(132,64,196,.28)', fixed: 'rgba(40,52,60,.45)', absorbing: [235, 151, 30],
  receiver: '#0b7a6f', selected: '#f2b705', axis: ['#d03c3c', '#2f9e44', '#1c6ed6'],
};

/** Display scale per axis: thin directions are magnified so plates stay readable. */
export function displayScale(size, exaggerate = true) {
  if (!exaggerate) return [1, 1, 1];
  const longest = Math.max(...size);
  return size.map(s => (s > 0 && s < longest * .22 ? longest * .22 / s : 1));
}

/** Rotation for yaw (about the vertical z axis) then pitch; returns 3×3 rows. */
export function rotation(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  // screen x right, screen y up, depth toward viewer
  return [[cy, sy, 0], [-sy * sp, cy * sp, cp], [sy * cp, -cy * cp, sp]];
}

export function makeProjector(scene, view, width, height) {
  const { size, min } = extent(scene.geometry), scale = displayScale(size, view.exaggerate !== false), R = rotation(view.yaw, view.pitch);
  const center = size.map((s, i) => min[i] + s / 2);
  const toWorld = p => p.map((v, i) => (v - center[i]) * scale[i]);
  const raw = p => { const w = toWorld(p); return R.map(r => r[0] * w[0] + r[1] * w[1] + r[2] * w[2]); };
  const corners = [0, 1].flatMap(a => [0, 1].flatMap(b => [0, 1].map(c => raw([min[0] + a * size[0], min[1] + b * size[1], min[2] + c * size[2]]))));
  const span = Math.max(...corners.map(c => Math.abs(c[0]))) * 2 || 1, spanY = Math.max(...corners.map(c => Math.abs(c[1]))) * 2 || 1;
  const fit = Math.min(width * .8 / span, height * .74 / spanY) * (view.zoom || 1);
  const project = p => { const [x, y, z] = raw(p); return [width / 2 + x * fit + (view.panX || 0), height / 2 - y * fit + (view.panY || 0), z]; };
  return { project, scale, size, min, R, fit, geometry: scene.geometry };
}

function polygon(ctx, points, fill, stroke, width = 1) {
  ctx.beginPath(); points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke(); }
}

/** Corner points of a bounding face, in order. */
function faceQuad(min, size, face) {
  const axis = 'xyz'.indexOf(face[0]), at = face.endsWith('min') ? min[axis] : min[axis] + size[axis];
  const [u, v] = [0, 1, 2].filter(i => i !== axis);
  return [[0, 0], [1, 0], [1, 1], [0, 1]].map(([a, b]) => { const p = [0, 0, 0]; p[axis] = at; p[u] = min[u] + a * size[u]; p[v] = min[v] + b * size[v]; return p; });
}

/** Whole-face region: the bounding quad, or the annulus of a pipe end. */
function wholeFace(ctx, P, face, fill, stroke, width) {
  const g = P.geometry;
  if (g.kind === 'pipe' && face[0] === 'x') {
    const x = face === 'xmin' ? 0 : g.length_mm, ring = r => Array.from({ length: 48 }, (_, k) => P.project([x, r * Math.cos(2 * Math.PI * k / 48), r * Math.sin(2 * Math.PI * k / 48)]));
    polygon(ctx, ring(g.outer_radius_mm), fill, stroke, width); polygon(ctx, ring(g.inner_radius_mm), '#f4f7f9', stroke, width); return;
  }
  polygon(ctx, faceQuad(P.min, P.size, face).map(P.project), fill, stroke, width);
}

function faceNormal(face) { const n = [0, 0, 0]; n['xyz'.indexOf(face[0])] = face.endsWith('min') ? -1 : 1; return n; }

/** Point on `face` (plane coordinate fixed) from in-plane offsets. */
function onFace(face, min, size, center, du, dv) {
  const axis = 'xyz'.indexOf(face[0]), [u, v] = [0, 1, 2].filter(i => i !== axis), p = [...center];
  p[axis] = face.endsWith('min') ? min[axis] : min[axis] + size[axis]; p[u] += du; p[v] += dv; return p;
}

function clampToFace(points, face, min, size) {
  const axis = 'xyz'.indexOf(face[0]);
  return points.map(p => p.map((x, i) => (i === axis ? x : Math.min(min[i] + size[i], Math.max(min[i], x)))));
}

function drawArrow(ctx, from, to, color) {
  const [x0, y0] = from, [x1, y1] = to, a = Math.atan2(y1 - y0, x1 - x0), L = Math.hypot(x1 - x0, y1 - y0);
  if (L < 2) return;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 9 * Math.cos(a - .4), y1 - 9 * Math.sin(a - .4)); ctx.lineTo(x1 - 9 * Math.cos(a + .4), y1 - 9 * Math.sin(a + .4)); ctx.closePath(); ctx.fill();
}

/**
 * Draw `scene` on `canvas`. `view` = {yaw, pitch, zoom, exaggerate, grid, selected, gridMm}; returns pick targets
 * [{id, x, y, r}] for click selection.
 */
export function drawScene(canvas, scene, view) {
  const dpr = globalThis.devicePixelRatio || 1, width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  const P = makeProjector(scene, view, width, height), { project, size, min, R } = P, g = scene.geometry, picks = [];
  const facing = face => { const n = faceNormal(face); return R[2][0] * n[0] + R[2][1] * n[1] + R[2][2] * n[2] > 1e-9; };
  const faces = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'];
  const boundaryOf = face => (scene.boundaries || []).find(b => b.face === face && b.type !== 'absorbing') || (scene.boundary?.fixed_face === face ? { type: 'fixed' } : null);
  const shade = face => { const n = faceNormal(face), light = [.35, -.45, .82], d = Math.max(0, n[0] * light[0] + n[1] * light[1] + n[2] * light[2]); const k = .78 + .22 * d; return `rgb(${COLORS.body.map(c => Math.round(c * k)).join(',')})`; };

  if (g.kind === 'pipe') drawPipe(ctx, P, g, view);
  else {
    // back faces first (dashed outline), then front faces filled
    for (const face of faces.filter(f => !facing(f))) polygon(ctx, faceQuad(min, size, face).map(project), null, 'rgba(93,115,131,.25)', 1);
    for (const face of faces.filter(facing)) {
      const quad = faceQuad(min, size, face).map(project); polygon(ctx, quad, shade(face), COLORS.edge, 1.2);
      const b = boundaryOf(face); if (b) polygon(ctx, quad, COLORS[b.type], null);
      if (view.grid && view.gridMm) drawGrid(ctx, project, face, min, size, view.gridMm);
    }
    if (g.kind === 'hole') drawHole(ctx, project, g, min, size, facing);
    if (g.kind === 'notch') drawNotch(ctx, project, g, size, facing);
  }

  // fibre direction of an anisotropic strain-energy material, drawn on the top face
  if (scene.material?.model === 'energy' && g.kind !== 'pipe' && facing('zmax')) drawFibres(ctx, project, min, size, scene.material.fiber_angle_deg || 0);

  // absorbing layers: translucent gradient slabs
  (scene.boundaries || []).forEach((b, i) => {
    if (b.type !== 'absorbing') return;
    const axis = 'xyz'.indexOf(b.face[0]), steps = 6, L = Math.min(b.length_mm, size[axis]);
    for (let s = 0; s < steps; s++) {
      const sz = [...size], m = [...min];
      const depth0 = L * s / steps, depth1 = L * (s + 1) / steps;
      m[axis] = b.face.endsWith('min') ? min[axis] + depth0 : min[axis] + size[axis] - depth1; sz[axis] = depth1 - depth0;
      const alpha = .06 + .3 * ((steps - s) / steps) ** 2;
      for (const face of faces.filter(facing)) { if (face[0] === b.face[0] && face !== b.face) continue; polygon(ctx, faceQuad(m, sz, face).map(project), `rgba(${COLORS.absorbing.join(',')},${alpha})`, null); }
    }
    const selected = view.selected === `boundary:${i}`;
    polygon(ctx, faceQuad(min, size, b.face).map(project), null, selected ? COLORS.selected : `rgba(${COLORS.absorbing.join(',')},.9)`, selected ? 3 : 1.5);
    const c = project(faceQuad(min, size, b.face).reduce((a, p) => a.map((v, k) => v + p[k] / 4), [0, 0, 0])); picks.push({ id: `boundary:${i}`, x: c[0], y: c[1], r: 14 });
  });
  (scene.boundaries || []).forEach((b, i) => {
    if (b.type === 'absorbing') return;
    const quad = faceQuad(min, size, b.face).map(project), selected = view.selected === `boundary:${i}`;
    if (!facing(b.face) && !selected) ctx.setLineDash([5, 4]);
    polygon(ctx, quad, null, selected ? COLORS.selected : b.type === 'symmetry' ? '#1c6ed6' : '#8440c4', selected ? 3 : 1.6);
    ctx.setLineDash([]);
    const c = quad.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4], [0, 0]); picks.push({ id: `boundary:${i}`, x: c[0], y: c[1], r: 14 });
  });
  if (scene.boundary?.fixed_face && scene.boundary.fixed_face !== 'none') {
    const selected = view.selected === 'fixed';
    wholeFace(ctx, P, scene.boundary.fixed_face, g.kind === 'pipe' ? COLORS.fixed : null, selected ? COLORS.selected : '#28343c', selected ? 3 : 2);
  }

  // sources
  sourcesOf(scene).forEach((src, i) => drawSource(ctx, P, src, i, view.selected === `source:${i}`, picks));

  // receivers
  scene.receivers.forEach((r, i) => {
    const p = project(r.point_mm), selected = view.selected === `receiver:${i}`, d = r.direction, n = Math.hypot(...d) || 1;
    const tip = project(r.point_mm.map((v, k) => v + d[k] / n * Math.max(...size) * .06 / P.scale[k]));
    ctx.strokeStyle = COLORS.receiver; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(tip[0], tip[1]); ctx.stroke();
    ctx.beginPath(); ctx.arc(p[0], p[1], selected ? 7 : 5, 0, 2 * Math.PI); ctx.fillStyle = selected ? COLORS.selected : COLORS.receiver; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = '#0b4f48'; ctx.font = '600 11px Inter, "Microsoft YaHei", sans-serif'; ctx.fillText(r.name, p[0] + 8, p[1] - 8);
    picks.push({ id: `receiver:${i}`, x: p[0], y: p[1], r: 10 });
  });

  drawAxes(ctx, R, height);
  if (P.scale.some(s => s > 1.0001)) {
    ctx.fillStyle = '#6a7f8c'; ctx.font = '11px Inter, "Microsoft YaHei", sans-serif';
    ctx.fillText(`显示放大：${['X', 'Y', 'Z'].map((a, i) => P.scale[i] > 1.0001 ? `${a} ×${P.scale[i].toFixed(1)}` : '').filter(Boolean).join('  ')}`, width - 170, height - 14);
  }
  return picks;
}

/** Line segments of direction angle θ (deg, from x about z) clipped to the top rectangle. */
export function fibreSegments(min, size, angleDeg, count = 9) {
  const t = angleDeg * Math.PI / 180, d = [Math.cos(t), Math.sin(t)], n = [-d[1], d[0]], c = [min[0] + size[0] / 2, min[1] + size[1] / 2];
  const half = (Math.abs(size[0] * n[0]) + Math.abs(size[1] * n[1])) / 2, out = [];  // extent across the fibres
  for (let k = 0; k < count; k++) {
    const offset = (k - (count - 1) / 2) / (count - 1) * 2 * half * .92, p = [c[0] + n[0] * offset, c[1] + n[1] * offset];
    let lo = -Infinity, hi = Infinity;
    for (const axis of [0, 1]) {
      const a = min[axis], b = min[axis] + size[axis];
      if (Math.abs(d[axis]) < 1e-12) { if (p[axis] < a || p[axis] > b) { lo = 1; hi = 0; } continue; }
      const s0 = (a - p[axis]) / d[axis], s1 = (b - p[axis]) / d[axis]; lo = Math.max(lo, Math.min(s0, s1)); hi = Math.min(hi, Math.max(s0, s1));
    }
    if (hi > lo) out.push([[p[0] + d[0] * lo, p[1] + d[1] * lo], [p[0] + d[0] * hi, p[1] + d[1] * hi]]);
  }
  return out;
}

function drawFibres(ctx, project, min, size, angle) {
  const z = min[2] + size[2];
  ctx.strokeStyle = 'rgba(30,30,30,.32)'; ctx.lineWidth = 1.2; ctx.setLineDash([7, 4]); ctx.beginPath();
  for (const [a, b] of fibreSegments(min, size, angle, 11)) { const p = project([a[0], a[1], z]), q = project([b[0], b[1], z]); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); }
  ctx.stroke(); ctx.setLineDash([]);
}

function drawGrid(ctx, project, face, min, size, grid) {
  const axis = 'xyz'.indexOf(face[0]), at = face.endsWith('min') ? min[axis] : min[axis] + size[axis], [u, v] = [0, 1, 2].filter(i => i !== axis);
  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1; ctx.beginPath();
  for (const [a, b] of [[u, v], [v, u]]) for (const value of grid[a] || []) {
    if (value <= min[a] + 1e-9 || value >= min[a] + size[a] - 1e-9) continue;
    const p0 = [0, 0, 0], p1 = [0, 0, 0]; p0[axis] = p1[axis] = at; p0[a] = p1[a] = value; p0[b] = min[b]; p1[b] = min[b] + size[b];
    const s0 = project(p0), s1 = project(p1); ctx.moveTo(s0[0], s0[1]); ctx.lineTo(s1[0], s1[1]);
  }
  ctx.stroke();
}

function ellipse(project, face, min, size, center, ru, rv, n = 48) {
  return clampToFace(Array.from({ length: n }, (_, k) => onFace(face, min, size, center, ru * Math.cos(2 * Math.PI * k / n), rv * Math.sin(2 * Math.PI * k / n))), face, min, size).map(project);
}

function drawSource(ctx, P, src, index, selected, picks) {
  const { project, min, size, scale } = P, face = src.face, prof = src.profile || { type: 'uniform' }, stroke = selected ? COLORS.selected : COLORS.source, lw = selected ? 2.6 : 1.5;
  if (!/^[xyz](min|max)$/.test(face)) return;
  const center = src.center_mm, axis = 'xyz'.indexOf(face[0]), [u, v] = [0, 1, 2].filter(i => i !== axis);
  const radius = src.radius_mm || 0;
  if (prof.type === 'gaussian') {
    for (const k of [2, 1]) polygon(ctx, ellipse(project, face, min, size, center, k * prof.sigma_mm, k * prof.sigma_mm), k === 1 ? COLORS.sourceFill : 'rgba(217,72,15,.10)', stroke, k === 1 ? lw : 1);
  } else if (prof.type === 'comb') {
    const along = 'xyz'.indexOf(prof.along), other = along === u ? v : u, half = (prof.teeth - 1) / 2;
    for (let t = 0; t < prof.teeth; t++) {
      const offset = (t - half) * prof.pitch_mm, w = prof.tooth_width_mm / 2, span = radius > 0 ? radius : size[other] / 2;
      const corners = [[-w, -span], [w, -span], [w, span], [-w, span]].map(([a, b]) => { const p = [...center]; p[axis] = face.endsWith('min') ? min[axis] : min[axis] + size[axis]; p[along] += offset + a; p[other] += b; return p; });
      polygon(ctx, clampToFace(corners, face, min, size).map(project), prof.alternating_sign && t % 2 ? 'rgba(28,110,214,.28)' : COLORS.sourceFill, stroke, lw);
    }
  } else if (radius > 0) polygon(ctx, ellipse(project, face, min, size, center, radius, radius), COLORS.sourceFill, stroke, lw);
  else wholeFace(ctx, P, face, COLORS.sourceFill, stroke, lw);
  const c = project(center), d = src.direction, n = Math.hypot(...d) || 1, L = Math.max(...size) * .1;
  const base = project(center.map((x, k) => x - d[k] / n * L / scale[k]));
  drawArrow(ctx, base, c, stroke);
  ctx.fillStyle = '#9c330a'; ctx.font = '600 11px Inter, "Microsoft YaHei", sans-serif'; ctx.fillText(src.name || (index ? `激励 ${index + 1}` : '主激励'), base[0] + 6, base[1] - 6);
  picks.push({ id: `source:${index}`, x: c[0], y: c[1], r: 14 }, { id: `source:${index}`, x: base[0], y: base[1], r: 12 });
}

function drawHole(ctx, project, g, min, size, facing) {
  for (const face of ['zmax', 'zmin']) {
    if (!facing(face)) continue;
    const ring = ellipse(project, face, min, size, [g.hole_x_mm, g.hole_y_mm, 0], g.hole_radius_mm, g.hole_radius_mm);
    polygon(ctx, ring, '#f7f9fa', '#5d7383', 1.2);
  }
}

function drawNotch(ctx, project, g, size, facing) {
  const a = g.notch_x_mm - g.notch_width_mm / 2, b = g.notch_x_mm + g.notch_width_mm / 2, top = size[2], bottom = size[2] - g.notch_depth_mm;
  if (facing('zmax')) polygon(ctx, [[a, 0, top], [b, 0, top], [b, size[1], top], [a, size[1], top]].map(project), '#f7f9fa', '#5d7383', 1.2);
  for (const y of [0, size[1]]) if (facing(y ? 'ymax' : 'ymin')) polygon(ctx, [[a, y, top], [b, y, top], [b, y, bottom], [a, y, bottom]].map(project), '#f7f9fa', '#5d7383', 1.2);
}

function drawPipe(ctx, P, g, view) {
  const { project } = P, n = 48, ring = (x, r) => Array.from({ length: n }, (_, k) => [x, r * Math.cos(2 * Math.PI * k / n), r * Math.sin(2 * Math.PI * k / n)]);
  const outer0 = ring(0, g.outer_radius_mm).map(project), outer1 = ring(g.length_mm, g.outer_radius_mm).map(project);
  const quads = outer0.map((p, k) => [p, outer0[(k + 1) % n], outer1[(k + 1) % n], outer1[k]]).map(q => ({ q, z: q.reduce((s, p) => s + p[2], 0) }));
  quads.sort((a, b) => a.z - b.z).forEach(({ q, z }) => polygon(ctx, q, `rgba(205,216,224,${z > 0 ? .95 : .55})`, null));
  for (const [x, r] of [[0, g.outer_radius_mm], [0, g.inner_radius_mm], [g.length_mm, g.outer_radius_mm], [g.length_mm, g.inner_radius_mm]]) polygon(ctx, ring(x, r).map(project), null, COLORS.edge, 1.1);
  void view;
}

function drawAxes(ctx, R, height) {
  const origin = [44, height - 40], L = 26;
  ['X', 'Y', 'Z'].forEach((label, i) => {
    const e = [0, 0, 0]; e[i] = 1; const x = R[0][0] * e[0] + R[0][1] * e[1] + R[0][2] * e[2], y = R[1][0] * e[0] + R[1][1] * e[1] + R[1][2] * e[2];
    drawArrow(ctx, origin, [origin[0] + x * L, origin[1] - y * L], COLORS.axis[i]);
    ctx.fillStyle = COLORS.axis[i]; ctx.font = '700 11px Inter, sans-serif'; ctx.fillText(label, origin[0] + x * (L + 9) - 3, origin[1] - y * (L + 9) + 4);
  });
}

/** Nearest pick target within its radius, or null. */
export function pick(targets, x, y) {
  let best = null, bestD = Infinity;
  for (const t of targets) { const d = Math.hypot(t.x - x, t.y - y); if (d <= t.r && d < bestD) { best = t; bestD = d; } }
  return best?.id ?? null;
}
