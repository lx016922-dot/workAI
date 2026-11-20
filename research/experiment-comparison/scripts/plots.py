"""Figure of the comparison with Ding et al. (2018): A2/A1² along the plate in the software's models, with the distance
of its maximum from the paper's experiment, from the paper's simulation and from the continuous-wave theory.
Run with a Python that has matplotlib; reads results/results.json only."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 160, 'axes.grid': True, 'grid.alpha': .3, 'axes.unicode_minus': False, 'font.sans-serif': ['WenQuanYi Zen Hei', 'DejaVu Sans'], 'axes.spines.top': False, 'axes.spines.right': False})
INK, MUTED, BLUE, ORANGE, GREEN = '#173147', '#748995', '#2a78d6', '#eb6834', '#1baf7a'
TITLES = {'a-300kHz-2.0mm': '(a) 300 kHz，2.0 mm', 'b-300kHz-2.5mm': '(b) 300 kHz，2.5 mm', 'c-240kHz-2.5mm': '(c) 240 kHz，2.5 mm', 'd-200kHz-2.0mm': '(d) 200 kHz，2.0 mm'}


def main():
    results = json.loads((HERE / 'results/results.json').read_text(encoding='utf-8'))['cases']; fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    for ax, (name, title) in zip(axes.ravel(), TITLES.items()):
        r = results[name]; x = np.array(r['x_mm'])
        for key, colour, label in (('increment', BLUE, '软件：A₂ 取非线性增量'), ('spectrum', ORANGE, '软件：A₂ 取完整响应（测量能得到的）')):
            y = np.array([np.nan if v is None else v for v in r[f'beta_{key}_per_m']], dtype=float); ax.plot(x, y / np.nanmax(y), color=colour, lw=2, label=label)
        lo, hi = r['measured_mm']; ax.axvspan(lo, hi, color=MUTED, alpha=.10, lw=0, label='实验的测量范围')
        ax.axvline(r['theory']['half_beat_mm'], color=GREEN, lw=1.5, ls='-.', label='连续波理论 π/|Δk|')
        ax.axvline(r['paper_simulation_mm'], color=MUTED, lw=1.5, ls=':', label='论文的有限元仿真')
        if 'experiment_minimum_mm' in r:   # case b: the experiment starts beyond the first maximum; its table entry is the spacing of these two points
            for value, word in ((r['experiment_minimum_mm'], '实验最小'), (r['experiment_maximum_mm'], '实验最大')):
                ax.axvline(value, color=INK, lw=1.5, ls='--'); ax.text(value, 1.115, word, ha='center', va='bottom', fontsize=7, color=INK)
        elif name.startswith('d-'):        # case d: still rising at the last measured point
            ax.axvline(r['experiment_mm'], color=INK, lw=1.5, ls='--'); ax.text(r['experiment_mm'], 1.115, '实验的最后一个测点（仍在上升）', ha='center', va='bottom', fontsize=7, color=INK)
        else: ax.axvline(r['experiment_mm'], color=INK, lw=1.5, ls='--', label='论文的实验（表 2）')
        ax.set(xlabel='传播距离 (mm)', ylabel='A₂/A₁²（各自归一化）', xlim=(0, x[-1]), ylim=(0, 1.1)); ax.set_title(title, loc='left', pad=14)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('A₂/A₁² 最大处的传播距离：软件、连续波理论与 Ding 等（2018）的实验和仿真', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, .08, 1, .96), w_pad=2.5, h_pad=2.); fig.savefig(OUT / 'fig1-dispersion-distance.png'); print(OUT / 'fig1-dispersion-distance.png')


if __name__ == '__main__': main()
