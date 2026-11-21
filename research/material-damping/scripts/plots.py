"""Figures of the material-damping verification. Run with a Python that has matplotlib; reads results/*.json only."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 160, 'axes.grid': True, 'grid.alpha': .3, 'axes.unicode_minus': False, 'font.sans-serif': ['WenQuanYi Zen Hei', 'DejaVu Sans'], 'axes.spines.top': False, 'axes.spines.right': False})
INK, MUTED = '#173147', '#748995'; COLOURS = {'B-none': MUTED, 'B-mass': '#2a78d6', 'B-stiffness': '#eb6834', 'B-both': '#1baf7a'}
LABELS = {'B-none': '无阻尼', 'B-mass': '只有质量比例 α', 'B-stiffness': '只有刚度比例 β', 'B-both': 'α 和 β 各一半'}


def bar():
    d = json.loads((HERE / 'results/bar.json').read_text()); fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for name, colour in COLOURS.items():
        r = d[name]; x = np.array(r['x_mm']); a1 = np.array(r['a1_m']) * 1e9; a2 = np.array(r['a2_nonlinear_m']) * 1e12
        axes[0].plot(x, a1, 'o', color=colour, ms=5, label=LABELS[name])
        a = r['attenuation_np_m']['exact_f']; axes[0].plot(x, a1[0] * np.exp(-a * (x - x[0]) * 1e-3), '-', color=colour, lw=1.2)
        axes[1].plot(x, a2, 'o', color=colour, ms=5, label=LABELS[name])
        axes[1].plot(x, a2 / (1 + np.array(r['a2_nonlinear_against_reference'])), '-', color=colour, lw=1.2)
    axes[0].set(xlabel='到加载端的距离 (mm)', ylabel='基频幅值 A₁ (nm)'); axes[0].set_title('基频：点为软件，线为精确衰减 exp(−Im k·x)', loc='left')
    axes[1].set(xlabel='到加载端的距离 (mm)', ylabel='二次谐波幅值 A₂（非线性增量）(pm)'); axes[1].set_title('二次谐波：点为软件，线为独立的一维参考解', loc='left')
    axes[0].legend(frameon=False, fontsize=8); fig.suptitle('有阻尼的平面纵波（0.5 MHz，10 MPa）：三种阻尼在 f 处的衰减率相同，在 2f 处不同', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(OUT / 'fig1-bar.png'); print(OUT / 'fig1-bar.png')


def plate():
    d = json.loads((HERE / 'results/plate.json').read_text())['rows']; offset = d['source_offset_mm']; fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for rows, key, colour, marker, label in ((d['undamped'], 'ratio_per_m', MUTED, 's', '无阻尼'), (d['damped'], 'ratio_per_m', '#eb6834', 'o', '有阻尼，直接取 A₂/A₁²'), (d['damped'], 'ratio_corrected_per_m', '#2a78d6', 'D', '有阻尼，扣除衰减后')):
        ax.plot([r['distance_mm'] + offset for r in rows], [r[key] for r in rows], marker, color=colour, ms=5, label=label)
    fit = d['undamped_fit']; x = np.array([d['undamped'][0]['distance_mm'], d['undamped'][-1]['distance_mm']]); ax.plot(x + offset, fit['intercept_per_m'] + fit['slope_per_m2'] * x * 1e-3, '-', color=MUTED, lw=1.2)
    ax.set(xlabel='到激励端的距离 (mm)', ylabel='A₂/A₁² (1/m)'); ax.legend(frameon=False, fontsize=8); ax.set_title('1 mm 铝板中的 S0（0.2 MHz）：扣除衰减后回到无阻尼的直线', loc='left', fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / 'fig2-plate.png'); print(OUT / 'fig2-plate.png')


if __name__ == '__main__':
    bar(); plate()
