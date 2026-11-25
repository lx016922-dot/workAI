"""Assemble the demonstration document: screenshots and text → HTML → PDF (the PDF is printed by pdf.cjs with Chromium).

    python docs/demo/build.py <shots dir> <out dir>      # writes <out dir>/demo.html and the images it uses
    node docs/demo/pdf.cjs <out dir>/demo.html docs/demo/WorkAI-1.5.0-功能演示.pdf

The text is content.py: a list of ('h1' | 'h2' | 'p' | 'ul' | 'table' | 'fig' | 'note', ...) items. A screenshot taller than a page is cut at blank rows.
"""
import html, sys
from pathlib import Path
import numpy as np
from PIL import Image

WIDTH_MM, HEIGHT_MM = 178., 232.          # the room a figure has on an A4 page
WINDOW_PX, PANEL_PX = 1560, 1100           # widths the screenshots are stored at (taken at twice these or so)


def pieces(image, limit):
    """Cut a tall image into parts no taller than `limit`, at the blank row nearest below each limit (searched over the last third of the part)."""
    grey = np.asarray(image.convert('L'), dtype=np.int16); flat = grey.max(1) - grey.min(1) < 12; out = []; top = 0
    while image.height - top > limit:
        window = np.flatnonzero(flat[top + int(limit * .6):top + limit]); cut = top + int(limit * .6) + int(window[-1]) if len(window) else top + limit
        out.append(image.crop((0, top, image.width, cut))); top = cut
    out.append(image.crop((0, top, image.width, image.height))); return out


def trimmed(image):
    """Without the blank rows at its bottom."""
    grey = np.asarray(image.convert('L'), dtype=np.int16); busy = np.flatnonzero(grey.max(1) - grey.min(1) >= 12)
    return image.crop((0, 0, image.width, min(image.height, int(busy[-1]) + 24))) if len(busy) else image


def figure(shots, out, name, caption, width=None):
    source = trimmed(Image.open(shots / f'{name}.jpg').convert('RGB')); window = source.width > 2200
    width = width or (.88 if window else .6); target = WINDOW_PX if window else min(PANEL_PX, source.width)
    if source.width > target: source = source.resize((target, round(source.height * target / source.width)), Image.LANCZOS)
    parts = pieces(source, int(source.width * HEIGHT_MM / (WIDTH_MM * width))); rows = []
    for i, part in enumerate(parts):
        file = out / 'img' / f'{name}-{i}.jpg'; part.save(file, quality=70, optimize=True)
        rows.append(f'<figure style="width:{width * 100:.0f}%"><img src="img/{file.name}">' + (f'<figcaption>{caption}{"（续）" if i else ""}</figcaption>' if caption else '') + '</figure>')
    return '\n'.join(rows)


def render(items, shots, out):
    body = []
    for kind, *rest in items:
        if kind in ('h1', 'h2', 'h3'): body.append(f'<{kind}>{rest[0]}</{kind}>')
        elif kind == 'p': body.append(f'<p>{rest[0]}</p>')
        elif kind == 'note': body.append(f'<p class="note">{rest[0]}</p>')
        elif kind == 'ul': body.append('<ul>' + ''.join(f'<li>{row}</li>' for row in rest[0]) + '</ul>')
        elif kind == 'table':
            head, rows = rest; body.append('<table><thead><tr>' + ''.join(f'<th>{c}</th>' for c in head) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table>')
        elif kind == 'fig': body.append(figure(shots, out, *rest))
        elif kind == 'row': body.append('<div class="row">' + ''.join(figure(shots, out, name, caption, 1.) for name, caption in rest[0]) + '</div>')
        elif kind == 'pre': body.append(f'<pre>{html.escape(rest[0])}</pre>')
        elif kind == 'break': body.append('<div class="break"></div>')
        else: raise ValueError(kind)
    return '\n'.join(body)


STYLE = '''
@page { size: A4; margin: 16mm 16mm 18mm 16mm; }
body { font-family: "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1c2430; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 4mm; padding-bottom: 2mm; border-bottom: 1.5pt solid #1f6fb5; break-before: page; break-after: avoid; }
h1.first { break-before: auto; }
h2 { font-size: 12.5pt; margin: 6mm 0 2mm; color: #174f85; break-after: avoid; }
h3 { font-size: 11pt; margin: 4mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.5mm; } ul { margin: 0 0 2.5mm; padding-left: 5.5mm; } li { margin-bottom: 1mm; }
p.note { background: #f3f6fa; border-left: 3pt solid #8aa6c2; padding: 2mm 3mm; font-size: 9.5pt; }
table { border-collapse: collapse; width: 99.5%; margin: 1mm 0 4mm; font-size: 9.5pt; break-inside: avoid; }
th, td { border: .6pt solid #c3ccd6; padding: 1.2mm 2mm; text-align: left; vertical-align: top; } th { background: #eef2f6; }
figure { margin: 2mm auto 4mm; break-inside: avoid; } figure img { width: 100%; border: .6pt solid #c3ccd6; display: block; }
figcaption { font-size: 9pt; color: #4a5766; margin-top: 1.2mm; text-align: center; }
pre { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; background: #f6f8fa; border: .6pt solid #d5dbe2; padding: 2mm 3mm; white-space: pre-wrap; break-inside: avoid; }
.cover { height: 235mm; display: flex; flex-direction: column; justify-content: center; } .cover h0 { font-size: 26pt; font-weight: bold; display: block; margin-bottom: 4mm; }
.cover .sub { font-size: 13pt; color: #174f85; margin-bottom: 14mm; } .cover .meta { font-size: 10pt; color: #4a5766; line-height: 1.9; }
.break { break-after: page; }
.row { display: flex; gap: 3mm; align-items: flex-start; break-inside: avoid; } .row figure { flex: 1; margin: 2mm 0 4mm; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; background: #f1f3f5; padding: 0 1mm; }
'''

if __name__ == '__main__':
    shots, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / 'img').mkdir(parents=True, exist_ok=True)
    exported = shots / 'files' / 'accumulation.png'   # the figure the PNG button saved: shown as it is, on white
    if exported.exists(): picture = Image.open(exported).convert('RGBA'); ground = Image.new('RGB', picture.size, 'white'); ground.paste(picture, mask=picture.split()[3]); ground.save(shots / 'e01-exported-png.jpg', quality=92)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import content
    page = f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>{content.TITLE}</title><style>{STYLE}</style></head><body>{content.COVER}{render(content.ITEMS, shots, out)}</body></html>'
    (out / 'demo.html').write_text(page, encoding='utf-8'); print(out / 'demo.html', sum(f.stat().st_size for f in (out / 'img').iterdir()) // 1024, 'KiB of images')
