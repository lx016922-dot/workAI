"""Figures for the 1 ms accuracy report (needs matplotlib; reads results/*.json only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': .3, 'axes.unicode_minus': False, 'mathtext.fontset': 'dejavusans'})
for f in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei'):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
COLORS = ['#2b6cb0', '#dd6b20', '#2f855a', '#b83280']
from matplotlib.ticker import FuncFormatter, LogLocator
SCI = FuncFormatter(lambda v, _: f'1e{round(np.log10(v))}' if v > 0 else '0')  # ASCII exponents (the CJK font has no U+2212)


def logaxes(*axes):
    for a in axes: a.yaxis.set_major_locator(LogLocator(base=10)); a.yaxis.set_major_formatter(SCI); a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))

bar = json.loads((HERE / 'results/bar-1ms.json').read_text())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for (name, r), col in zip(sorted(bar.items()), COLORS):
    t = np.array(r['windows_us']); label = f"h {r['h_mm']:g} mm，{r['f_mhz']:g} MHz（{r['nodes_per_wavelength_2f']:.0f} 点/2f 波长）"
    exact_win = r['linear']['window_rel_l2']  # vs the 1D reference (equals the exact solution to 1e-9)
    ax[0].semilogy(t, exact_win, color=col, lw=1, label=label); ax[1].semilogy(t, r['material']['window_rel_l2'], color=col, lw=1, label=label)
ax[0].set(xlabel='时间 (μs)', ylabel='50 μs 窗口相对 L2 误差', title='一维杆：线性响应'); ax[1].set(xlabel='时间 (μs)', title='一维杆：材料非线性增量')
ax[0].legend(fontsize=6); logaxes(*ax); fig.tight_layout(); fig.savefig(OUT / 'fig1-bar-error-vs-time.png'); plt.close(fig)

plate_path = HERE / 'results/plate-1ms.json'
if plate_path.exists():
    pl = json.loads(plate_path.read_text()); t = np.array(pl['windows_us'])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for part, a in (('full', ax[0]), ('material', ax[1])):
        a.semilogy(t, pl['reference_self_check'][part]['window_rel_l2'], color='gray', ls='--', lw=1, label='参考解自检（h 0.75 对 0.5 mm）')
        for h, col in zip(('2', '1.5', '1'), COLORS):
            if f'h{h}' in pl: a.semilogy(t, pl[f'h{h}'][part]['window_rel_l2'], color=col, lw=1, label=f'软件 h {h} mm')
    ax[0].set(xlabel='时间 (μs)', ylabel='50 μs 窗口相对 L2 误差', title='平板：完整响应'); ax[1].set(xlabel='时间 (μs)', title='平板：材料非线性增量')
    ax[0].legend(fontsize=6); logaxes(*ax); fig.tight_layout(); fig.savefig(OUT / 'fig2-plate-error-vs-time.png'); plt.close(fig)
    ex = pl.get('excerpt')
    if ex:
        fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True); te = np.array(ex['t_us'])
        for i, kind in enumerate(('full', 'material')):
            ax[i].plot(te, np.array(ex[f'reference_{kind}_top24']) * 1e9, color='black', lw=1.4, label='参考解')
            for h, col in zip(('2', '1'), COLORS):
                if f'h{h}_{kind}_top24' in ex: ax[i].plot(te, np.array(ex[f'h{h}_{kind}_top24']) * 1e9, color=col, lw=.8, ls='--', label=f'软件 h {h} mm')
            ax[i].set(ylabel='nm', title=('完整响应' if kind == 'full' else '材料非线性增量') + '，顶面 x = 24 mm，900–1000 μs'); ax[i].legend(fontsize=6, loc='upper right')
        ax[1].set(xlabel='时间 (μs)'); fig.tight_layout(); fig.savefig(OUT / 'fig3-plate-late-waveform.png'); plt.close(fig)
print(sorted(p.name for p in OUT.glob('*.png')))
