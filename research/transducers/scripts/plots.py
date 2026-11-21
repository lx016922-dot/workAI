"""Figure of the transducer verification: every amplitude ratio of results/transducers.json, software against theory.
Run with a Python that has matplotlib."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 160, 'axes.grid': True, 'grid.alpha': .3, 'axes.unicode_minus': False, 'font.sans-serif': ['WenQuanYi Zen Hei', 'DejaVu Sans'], 'axes.spines.top': False, 'axes.spines.right': False})
INK, BLUE, ORANGE = '#173147', '#2a78d6', '#eb6834'; FLOOR = 3e-4   # ratios below the floor are drawn on it (they are zeros of the theory)


def main():
    d = json.loads((HERE / 'results/transducers.json').read_text()); fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for prefix, colour, marker, label in (('W-', BLUE, 'o', '斜楔（32 mm 底面，6 种迹线速度和 Hann 分布）'), ('P-', ORANGE, 'D', '压电片（4 种长度）')):
        rows = [r for r in d['ratios'] if r['case'].startswith(prefix)]; theory = np.array([r['theory_ideal'] for r in rows]); software = np.array([r['software'] for r in rows])
        ax.scatter(np.maximum(theory, FLOOR), np.maximum(software, FLOOR), s=34, color=colour, marker=marker, label=label, zorder=3, edgecolor='white', linewidth=.6)
    ax.plot([FLOOR, 2], [FLOOR, 2], color=INK, lw=1, zorder=1)
    ax.set(xscale='log', yscale='log', xlim=(FLOOR * .8, 2), ylim=(FLOOR * .8, 2), xlabel='理论：同一模态在两个算例里的幅值之比', ylabel='软件'); ax.set_aspect('equal')
    ticks = [FLOOR, 1e-3, 1e-2, 1e-1, 1]; names = ['0', '0.001', '0.01', '0.1', '1']; ax.set_xticks(ticks); ax.set_xticklabels(names); ax.set_yticks(ticks); ax.set_yticklabels(names); ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.set_title('换能器激发的模态幅值：软件与简正模态展开', loc='left', fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / 'fig1-ratios.png'); print(OUT / 'fig1-ratios.png')


if __name__ == '__main__': main()
