// Figures and a report out of the page: every chart as PNG or SVG, and the open result as one self-contained HTML file.
import { escapeHtml } from './inspector-util.js';

const STYLE_PROPERTIES = ['fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-dasharray', 'stroke-opacity', 'stroke-linecap', 'stroke-linejoin', 'opacity', 'font-family', 'font-size', 'font-weight', 'font-style',
  'text-anchor', 'dominant-baseline', 'letter-spacing', 'visibility', 'display'];

/** A file name from free text: letters, digits and CJK kept, everything else one hyphen; at most 80 characters. */
export function fileName(text, extension) {
  const base = String(text ?? '').trim().replace(/[^\p{L}\p{N}._-]+/gu, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'figure';
  return extension ? `${base}.${extension}` : base;
}

/** The size in pixels a chart is exported at: its drawn size, or its viewBox when it is not laid out. */
export function exportSize(box, viewBox) {
  const width = Math.round(box?.width || viewBox?.width || 0), height = Math.round(box?.height || viewBox?.height || 0);
  if (!(width > 0 && height > 0)) throw new Error('图没有尺寸，无法导出。');
  return { width, height };
}

/** A chart as a stand-alone SVG document: the styles the page's stylesheet gives it are written into the elements, on a white ground. */
export function standaloneSvg(svg) {
  const copy = svg.cloneNode(true), originals = [svg, ...svg.querySelectorAll('*')], copies = [copy, ...copy.querySelectorAll('*')];
  originals.forEach((node, i) => {
    const computed = getComputedStyle(node), rules = STYLE_PROPERTIES.map(p => [p, computed.getPropertyValue(p)]).filter(([, v]) => v && v !== 'normal').map(([p, v]) => `${p}:${v}`).join(';');
    if (rules) copies[i].setAttribute('style', rules);
  });
  const view = svg.viewBox?.baseVal, { width, height } = exportSize(svg.getBoundingClientRect(), view);
  copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg'); copy.setAttribute('width', width); copy.setAttribute('height', height);
  if (!copy.getAttribute('viewBox')) copy.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const ground = document.createElementNS('http://www.w3.org/2000/svg', 'rect'), box = (copy.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number);
  ground.setAttribute('x', box[0] || 0); ground.setAttribute('y', box[1] || 0); ground.setAttribute('width', box[2] || width); ground.setAttribute('height', box[3] || height); ground.setAttribute('fill', '#ffffff');
  copy.insertBefore(ground, copy.firstChild);
  return { text: '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(copy), width, height };
}

/** PNG of an SVG document at `scale` device pixels per pixel. */
export function pngFromSvg({ text, width, height }, scale = 2) {
  return new Promise((resolve, reject) => {
    const image = new Image(), url = URL.createObjectURL(new Blob([text], { type: 'image/svg+xml;charset=utf-8' }));
    image.onload = () => {
      const canvas = document.createElement('canvas'); canvas.width = width * scale; canvas.height = height * scale;
      const context = canvas.getContext('2d'); context.fillStyle = '#ffffff'; context.fillRect(0, 0, canvas.width, canvas.height); context.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url); canvas.toBlob(blob => (blob ? resolve(blob) : reject(new Error('图片生成失败。'))), 'image/png');
    };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('图片生成失败。')); };
    image.src = url;
  });
}

export function saveBlob(blob, name) {
  const link = document.createElement('a'), url = URL.createObjectURL(blob);
  link.href = url; link.download = name; document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const canvasBlob = canvas => new Promise((resolve, reject) => canvas.toBlob(blob => (blob ? resolve(blob) : reject(new Error('图片生成失败。'))), 'image/png'));

/** The heading a figure stands under: the nearest h3 or h2 before it inside its panel. */
export function captionOf(node) {
  for (let at = node; at && at !== document.body; at = at.parentElement) {
    for (let previous = at.previousElementSibling; previous; previous = previous.previousElementSibling) {
      if (/^H[23]$/.test(previous.tagName)) return previous.textContent.trim();
      const inner = previous.querySelector?.('h3, h2'); if (inner && /scan-h|^H3$/.test(inner.className + inner.tagName)) return inner.textContent.trim();
    }
    const heading = at.matches?.('section, .panel') ? at.querySelector('h2') : null; if (heading) return heading.textContent.trim();
  }
  return '';
}

const isFigure = node => (node instanceof SVGSVGElement || node instanceof HTMLCanvasElement) && node.getBoundingClientRect().width >= 180 && node.getBoundingClientRect().height >= 100;

/** A small tool bar that follows the pointer onto any chart of the page: save it as PNG, and as SVG when it is a vector drawing.
 *  `prefix()` gives the first part of the file name (the open job or scan). Returns a function that removes it. */
export function mountFigureTools(root, prefix, report = () => {}) {
  const bar = document.createElement('div'); bar.className = 'figure-tools'; bar.hidden = true;
  bar.innerHTML = '<button type="button" data-kind="png" title="保存这张图为 PNG（2 倍分辨率）">PNG</button><button type="button" data-kind="svg" title="保存这张图为 SVG（矢量）">SVG</button>';
  document.body.appendChild(bar); let target = null;
  const place = () => { const box = target.getBoundingClientRect(); bar.style.top = `${window.scrollY + box.top + 4}px`; bar.style.left = `${window.scrollX + box.right - bar.offsetWidth - 4}px`; };
  const over = event => {
    const node = event.target.closest?.('svg, canvas');
    if (node && root.contains(node) && isFigure(node)) { target = node; bar.querySelector('[data-kind="svg"]').hidden = !(node instanceof SVGSVGElement); bar.hidden = false; place(); }
    else if (!bar.contains(event.target)) bar.hidden = true;
  };
  const click = async event => {
    const kind = event.target.dataset?.kind; if (!kind || !target) return;
    try {
      const name = fileName([prefix(), captionOf(target)].filter(Boolean).join('-'));
      if (target instanceof HTMLCanvasElement) saveBlob(await canvasBlob(target), name + '.png');
      else { const drawing = standaloneSvg(target); saveBlob(kind === 'svg' ? new Blob([drawing.text], { type: 'image/svg+xml;charset=utf-8' }) : await pngFromSvg(drawing), `${name}.${kind}`); }
    } catch (error) { report(error); }
  };
  document.addEventListener('mouseover', over); bar.addEventListener('click', click);
  return () => { document.removeEventListener('mouseover', over); bar.remove(); };
}

const REPORT_STYLE = `body{font:14px/1.6 "Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;color:#173147;max-width:1080px;margin:24px auto;padding:0 20px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid #d6e0e4}h3{font-size:14px;margin:18px 0 6px}
table{border-collapse:collapse;margin:8px 0;font-size:13px}th,td{border:1px solid #d6e0e4;padding:3px 8px;text-align:left;vertical-align:top}th{background:#f3f7f8;font-weight:600}
.meta td:first-child{color:#5a7480;white-space:nowrap}.muted,.section-note,.wb-note,small{color:#5a7480}svg,img{max-width:100%;height:auto}figure{margin:10px 0}
ul{margin:6px 0 6px 20px;padding:0}footer{margin-top:32px;color:#5a7480;font-size:12px;border-top:1px solid #d6e0e4;padding-top:8px}@media print{body{margin:0}h2{break-after:avoid}figure,table{break-inside:avoid}}`;

/** One self-contained HTML document: a title, a table of facts and sections of prepared HTML. */
export function reportHtml({ title, facts = [], sections = [], footer = '' }) {
  const rows = facts.filter(([, value]) => value !== undefined && value !== null && value !== '').map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(String(value))}</td></tr>`).join('');
  return `<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escapeHtml(title)}</title><style>${REPORT_STYLE}</style></head><body>
<h1>${escapeHtml(title)}</h1>${rows ? `<table class="meta">${rows}</table>` : ''}
${sections.filter(s => s.html && s.html.trim()).map(s => `<section><h2>${escapeHtml(s.heading)}</h2>${s.html}</section>`).join('\n')}
<footer>${escapeHtml(footer)}</footer></body></html>\n`;
}

/** What a panel of the page shows, without its controls: charts as SVG with their styles, canvases as images, tables and notes as they are. */
export function panelContent(panel) {
  const copy = panel.cloneNode(true), live = [...panel.querySelectorAll('svg, canvas')], clones = [...copy.querySelectorAll('svg, canvas')];
  live.forEach((node, i) => {
    const clone = clones[i]; if (!clone?.isConnected && !copy.contains(clone)) return;
    if (!isFigure(node)) { if (node instanceof HTMLCanvasElement) clone.remove(); return; }
    const figure = document.createElement('figure');
    if (node instanceof HTMLCanvasElement) { const image = document.createElement('img'); image.src = node.toDataURL('image/png'); image.width = Math.round(node.getBoundingClientRect().width); image.alt = captionOf(node); figure.appendChild(image); }
    else figure.innerHTML = standaloneSvg(node).text.replace(/^<\?xml[^>]*>\n/, '');
    clone.replaceWith(figure);
  });
  copy.querySelectorAll('[hidden], button, input, select, textarea, label.field, label.wb-field, label.nl-check, .solid-path-actions, .nl-tabs, .panel-heading, .wb-result-tools, progress, script, style').forEach(node => node.remove());
  copy.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
  copy.querySelectorAll('*').forEach(node => { if (!node.children.length && !node.textContent.trim() && !/^(IMG|svg|path|line|rect|circle|polyline|polygon|text|g|TD|TH|BR|HR)$/i.test(node.tagName)) node.remove(); });
  return copy.innerHTML;
}

/** The report of what is open: every panel under `root` that shows something, under its own heading. */
export function buildReport(root, { title, facts, footer }) {
  const sections = [...root.querySelectorAll('section.panel')].map(panel => ({ heading: panel.querySelector('.panel-heading h2, h2')?.textContent.trim() || '结果', html: panelContent(panel) }));
  return reportHtml({ title, facts, sections, footer });
}
