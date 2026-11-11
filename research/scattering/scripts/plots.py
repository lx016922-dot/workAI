"""Figures for the scattering report (needs matplotlib; reads results/*.json only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': .3, 'axes.unicode_minus': False, 'mathtext.fontset': 'dejavusans'})
for f in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei'):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
REF, NET, NET2 = '#3a3a3a', '#2b6cb0', '#dd6b20'   # reference ink, network h 1 mm, network h 1.5 mm (also told apart by line style)

notch = json.loads((HERE / 'results/notch.json').read_text()); cases = {n: v for n, v in notch.items() if n not in ('plate', 'models')}

# fig1: scattered waveforms of one notch, reference vs network
name = 'd0.8-w4' if 'trace' in notch.get('d0.8-w4', {}) else next(n for n, r in notch.items() if 'trace' in r)
tr = notch[name]['trace']; t = np.array(tr['t_us']); r = notch[name]
fig, ax = plt.subplots(2, 2, figsize=(9, 4.6), sharex=True)
for a, (key, title) in zip(ax.flat, (('S30', 'x = 30 mm 反射 S0（对称部分）'), ('A30', 'x = 30 mm 转换 A0（反对称部分）'), ('S75', 'x = 75 mm 透射 S0 的散射部分'), ('A75', 'x = 75 mm 透射 A0'))):
    s = 1e9
    a.plot(t, np.array(tr[f'ref_{key}']) * s, color=REF, lw=1.6, label='二维谱元参考（h 0.25 mm）')
    a.plot(t, np.array(tr[f'net_{key}']) * s, color=NET, lw=1, ls='--', label='软件网络（h 1 mm）')
    a.set_title(title, fontsize=8.5); a.set_ylabel('散射场 u_x (nm)')
for a in ax[1]: a.set_xlabel('时间 (μs)')
ax[0, 0].legend(fontsize=7, loc='upper left')
fig.suptitle(f"缺口宽 {r['width_mm']:g} mm、深 {r['depth_mm']:g} mm（板厚 2 mm，0.3 MHz S0 入射）：散射场为有缺口与无缺口两次计算之差", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'fig1-notch-waveforms.png'); plt.close(fig)

# fig2: reflection and mode conversion vs depth and width, and |R(f)| for each width
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={'width_ratios': [1, 1, 1.25]})
for a, rows, xkey, xlabel in ((ax[0], sorted([v for v in cases.values() if v['width_mm'] == 1.], key=lambda v: v['depth_mm']), 'depth_mm', '缺口深度 (mm)，宽 1 mm'),
                              (ax[1], sorted([v for v in cases.values() if v['depth_mm'] == .8], key=lambda v: v['width_mm']), 'width_mm', '缺口宽度 (mm)，深 0.8 mm')):
    x = [v[xkey] for v in rows]
    for coef, marker, label in (('R_S0_spectral', 'o', '|R| S0（0.3 MHz）'), ('R_A0', 's', '反射中转换的 A0'), ('T_A0', '^', '透射中转换的 A0')):
        a.plot(x, [v['reference'][coef] for v in rows], color=REF, lw=1, marker=marker, ms=7, mfc='none', label=f'{label} · 参考')
        a.plot(x, [v['network_h1'][coef] for v in rows], color=NET, lw=0, marker=marker, ms=3.5, label=f'{label} · 软件')
    a.set_xlabel(xlabel); a.set_ylim(0, .65)
ax[0].set_ylabel('幅值 / 入射 S0 幅值'); ax[0].legend(fontsize=6, loc='upper left')
a = ax[2]; rows = sorted([v for v in cases.values() if v['depth_mm'] == .8 and 'reflection_spectrum_x50_network' in v], key=lambda v: v['width_mm'])
blues = plt.get_cmap('Blues')(np.linspace(.35, .95, len(rows)))
for v, col in zip(rows, blues):
    sp = v['reflection_spectrum_x50_network']; a.plot(sp['f_mhz'], sp['R'], color=col, lw=1.4, label=f"宽 {v['width_mm']:g} mm")
m = notch.get('models', {}).get('two_step_max_2r_over_1_plus_r2')
if m: a.axhline(m, color='#8a8a8a', lw=.8, ls=':'); a.text(.152, m + .012, f'两台阶模型上限 2r/(1+r²) = {m:.2f}', fontsize=6.5, color='#555555')
a.set(xlabel='频率 (MHz)', ylabel='|R(f)| S0', title='深 0.8 mm 缺口的反射谱（软件 h 1 mm，x = 50 mm）'); a.title.set_fontsize(8.5); a.set_ylim(0, .55); a.legend(fontsize=6.5, loc='lower left')
fig.tight_layout(); fig.savefig(OUT / 'fig2-notch-coefficients.png'); plt.close(fig)

# fig4: mesh convergence of the scattered field (d0.8-w1), 2D solver and network on the same mesh sizes
row = cases.get('d0.8-w1', {}); conv = row.get('convergence_vs_extrapolated')
if conv:
    fig, a = plt.subplots(figsize=(4.6, 3.3)); key = 'S30'
    h2 = sorted(float(h) for h in conv['2d']); a.loglog(h2, [100 * conv['2d'][f'{h:g}'][key] for h in h2], color=REF, marker='o', mfc='none', ms=7, lw=1, label='二维谱元（独立求解器）')
    hn = sorted(float(h) for h in conv['network']); a.loglog(hn, [100 * conv['network'][f'{h:g}'][key] for h in hn], color=NET, marker='o', ms=3.5, lw=0, label='软件网络')
    g = np.array([.25, 1.5]); a.loglog(g, 100 * conv['2d']['1'][key] * g, color='#8a8a8a', ls=':', lw=.8, label='一阶收敛 ∝ h')
    from matplotlib.ticker import FuncFormatter
    for ax_ in (a.xaxis, a.yaxis): ax_.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax_.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    a.set_xticks([.25, .5, 1, 1.5]); a.set_yticks([.25, .5, 1, 1.5])
    a.set(xlabel='单元尺寸 h (mm)', ylabel='散射场误差 (%)', title='缺口 宽 1、深 0.8 mm：反射 S0 散射场的网格收敛'); a.title.set_fontsize(8.5); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig4-notch-convergence.png'); plt.close(fig)

# fig3: through hole, scattered field vs the analytical solution
hole = json.loads((HERE / 'results/hole.json').read_text()); recs = hole['receivers']
pick = [x for x in recs if abs(x['r_mm'] - 4.02) < 1e-6 and x['theta_deg'] in (180., 90., 0.)]
fig, ax = plt.subplots(1, len(pick) + 1, figsize=(11, 3), gridspec_kw={'width_ratios': [1] * len(pick) + [1.1]})
names = {180.: '孔前沿 θ = 180°（迎波面）', 90.: '孔侧 θ = 90°', 0.: '孔后沿 θ = 0°'}
for a, x in zip(ax, pick):
    t = np.array(x['trace']['t_us']); comp = 'y' if x['theta_deg'] == 90. else 'x'
    a.plot(t, np.array(x['trace'][f'pred_scattered_{comp}']) * 1e9, color=REF, lw=1.6, label='平面应力解析解')
    a.plot(t, np.array(x['trace'][f'sim_scattered_{comp}']) * 1e9, color=NET, lw=1, ls='--', label='软件网络')
    a.set_title(f"{names[x['theta_deg']]}，散射 u_{comp}", fontsize=8.5); a.set_xlabel('时间 (μs)')
ax[0].set_ylabel('散射场 (nm)'); ax[0].legend(fontsize=7, loc='upper left')
a = ax[-1]; th = sorted({x['theta_deg'] for x in recs})
for rr, col, mk in ((4.02, NET, 'o'), (8., NET2, 's')):
    xs = [x for x in recs if abs(x['r_mm'] - rr) < 1e-6]
    a.plot([x['theta_deg'] for x in xs], [100 * x['scattered_rel_l2'] for x in xs], color=col, marker=mk, ms=5, lw=1, label=f'r = {rr:g} mm')
a.set(xlabel='θ (°)', ylabel='散射场相对 L2 误差 (%)', title='各接收点误差', xticks=th); a.set_ylim(bottom=0); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'fig3-hole.png'); plt.close(fig)
print('figures in', OUT)
