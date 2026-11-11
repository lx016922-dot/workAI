"""Draw the WorkAI application icon (same design as the in-app logo) and save a multi-size .ico. Needs Pillow."""
import math
from pathlib import Path
from PIL import Image, ImageDraw

def draw(size):
    s = 4 * size   # supersample, then reduce
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    mask = Image.new('L', (s, s), 0); ImageDraw.Draw(mask).rounded_rectangle([s * .047, s * .047, s * .953, s * .953], radius=s * .2, fill=255)
    grad = Image.new('RGBA', (s, s))
    top, bottom = (47, 127, 208), (23, 74, 134)
    px = grad.load()
    for y in range(s):
        for x in range(s):
            t = (x + y) / (2 * s)
            px[x, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    pts = [(s * (0.156 + 0.7 * k / 400), s * (0.547 - 0.2 * math.sin(2 * math.pi * 2 * k / 400) * (1 - 0.15 * k / 400))) for k in range(401)]
    w = s * .069 / 2   # a round brush stamped along the curve gives a smooth thick line
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        for t in (0, .25, .5, .75):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            d.ellipse([x - w, y - w, x + w, y + w], fill=(255, 255, 255, 255))
    r = s * .075; cx, cy = s * .797, s * .266
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(127, 224, 201, 255))
    return img.resize((size, size), Image.LANCZOS)

if __name__ == '__main__':
    here = Path(__file__).resolve().parents[1]
    big = draw(256)
    big.save(here / 'workai.ico', sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    big.save(here / 'workai.png')
    print('wrote', here / 'workai.ico')
