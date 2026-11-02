// Signal preview: time waveform and amplitude spectrum per source, with f / 2f / band markers.
import { escapeHtml } from './inspector.js';

function ticks(lo, hi, count = 5) {
  const span = hi - lo || 1, raw = span / count, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map(k => k * mag).find(s => span / s <= count) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(10)); return out;
}

/** Small SVG line plot. markers: [{x, label, color}]. */
export function plot(xs, ys, { width = 620, height = 210, xLabel = '', yLabel = '', color = '#137d72', markers = [], yRange = null } = {}) {
  const m = { l: 56, r: 14, t: 14, b: 38 }, w = width - m.l - m.r, h = height - m.t - m.b;
  const x0 = Math.min(...xs), x1 = Math.max(...xs); let [y0, y1] = yRange || [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) { y0 -= 1; y1 += 1; } if (!yRange) { const pad = (y1 - y0) * .08; y0 -= pad; y1 += pad; }
  const X = x => m.l + (x - x0) / (x1 - x0 || 1) * w, Y = y => m.t + (1 - (Math.max(y0, Math.min(y1, y)) - y0) / (y1 - y0)) * h;
  const path = xs.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join('');
  const xt = ticks(x0, x1), yt = ticks(y0, y1, 4);
  return `<svg class="wb-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(yLabel)}">
    ${yt.map(v => `<line x1="${m.l}" x2="${m.l + w}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`).join('')}
    ${xt.map(v => `<text x="${X(v)}" y="${m.t + h + 16}" text-anchor="middle">${v}</text>`).join('')}
    ${markers.filter(k => k.x >= x0 && k.x <= x1).map(k => `<line x1="${X(k.x)}" x2="${X(k.x)}" y1="${m.t}" y2="${m.t + h}" stroke="${k.color}" stroke-dasharray="4 3"/><text x="${X(k.x) + 4}" y="${m.t + 12}" fill="${k.color}" class="mark">${escapeHtml(k.label)}</text>`).join('')}
    <rect x="${m.l}" y="${m.t}" width="${w}" height="${h}" class="frame"/>
    <path d="${path}" fill="none" stroke="${color}" stroke-width="1.6"/>
    <text x="${m.l + w / 2}" y="${height - 4}" text-anchor="middle" class="label">${escapeHtml(xLabel)}</text>
    <text x="12" y="${m.t + h / 2}" text-anchor="middle" class="label" transform="rotate(-90 12 ${m.t + h / 2})">${escapeHtml(yLabel)}</text></svg>`;
}

const TYPE = { tone_burst: '猝发音', gaussian_pulse: '高斯脉冲', ricker: 'Ricker', chirp: '线性调频', multi_tone: '多频合成', samples: '任意采样' };

export function signalsMarkup(preview, available = 900) {
  const columns = available >= 860 ? 2 : 1, width = Math.max(300, Math.floor((available - 34 - (columns - 1) * 12) / columns)), height = Math.round(Math.min(230, Math.max(170, width * .34)));
  if (!preview) return '<p class="wb-empty">正在计算信号预览…</p>';
  if (!preview.signals?.length) return `<p class="wb-empty">${escapeHtml(preview.errors?.[0] || '没有可预览的信号。')}</p>`;
  const resolution = Object.fromEntries((preview.resolution || []).map(r => [r.name, r.nodes_per_2f_shear_wavelength]));
  return preview.signals.map(s => {
    const f = s.frequency_mhz, nodes = resolution[s.name];
    return `<article class="wb-card"><header><h3>${escapeHtml(s.name)}</h3><span class="wb-tag">${TYPE[s.type] || s.type}</span>
      <span class="muted">载频 ${f} MHz · 持续 ${s.support_us[0].toFixed(2)}–${s.support_us[1].toFixed(2)} μs · −30 dB 上限 ${s.band_30db_mhz.toFixed(3)} MHz${nodes ? ` · 2f 剪切波长内 ${nodes.toFixed(1)} 个节点` : ''}</span></header>
      <div class="wb-plots" style="grid-template-columns:repeat(${columns},minmax(0,1fr))">${plot(s.times_us, s.values_mpa, { width, height, xLabel: '时间 / μs', yLabel: '牵引 / MPa', color: '#d9480f' })}
      ${plot(s.spectrum_mhz, s.spectrum_db, { width, height, xLabel: '频率 / MHz', yLabel: '幅值 / dB', yRange: [-60, 3], markers: [{ x: f, label: 'f', color: '#137d72' }, { x: 2 * f, label: '2f', color: '#1c6ed6' }, { x: s.band_30db_mhz, label: '−30 dB', color: '#b26b00' }] })}</div></article>`;
  }).join('');
}
