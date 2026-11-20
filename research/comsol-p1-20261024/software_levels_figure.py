"""Figure of `software_levels.py`: per receiver, the software's difference to COMSOL in the second harmonic of the
nonlinear increment beside the difference between COMSOL's two linear runs of the same model.
Run with a Python that has matplotlib; reads software-levels-20261120/levels.json only."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent / 'software-levels-20261120'; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
plt.rcParams.update({'font.size': 9, 'figure.dpi': 160, 'axes.grid': True, 'grid.alpha': .3, 'axes.unicode_minus': False, 'font.sans-serif': ['WenQuanYi Zen Hei', 'DejaVu Sans'], 'axes.spines.top': False, 'axes.spines.right': False})
INK, BLUE, ORANGE = '#173147', '#2a78d6', '#eb6834'


def main():
    cases = json.loads((HERE / 'levels.json').read_text(encoding='utf-8'))['cases']; fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True)
    for ax, (cid, entry) in zip(axes, cases.items()):
        r = entry['receivers']; x = 100 * np.array(r['increment_2f_over_largest'])
        ax.scatter(x, 100 * np.array(r['comsol_two_linear_runs_2f_difference']), s=34, color=ORANGE, marker='D', label='COMSOL 两次线性运行之间', zorder=3)
        ax.scatter(x, 100 * np.array(r['software_amplitude_difference']), s=34, color=BLUE, label='软件与 COMSOL 之间', zorder=4)
        ax.axhline(5, color=INK, lw=1, ls='--'); ax.text(100, 5.6, '5% 门槛', ha='right', va='bottom', fontsize=7, color=INK)
        ax.set(yscale='log', xlim=(0, 104), ylim=(5e-4, 12), xlabel='该接收点增量 2f 的大小（占最大接收点的 %）'); ax.set_title(f'工况 {cid}', loc='left')
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10]); ax.set_yticklabels(['0.001', '0.01', '0.1', '1', '10']); ax.minorticks_off()
    axes[0].set_ylabel('差异（占该点增量 2f 的 %）'); handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.suptitle('非线性增量 2f：软件与 COMSOL 的差异，与 COMSOL 自身重复性的对比（13 个接收点）', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, .07, 1, .95), w_pad=1.5); fig.savefig(OUT / 'receivers.png'); print(OUT / 'receivers.png')


if __name__ == '__main__': main()
