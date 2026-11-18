"""Figures for the verification report (run with a Python that has matplotlib; reads the results JSON only)."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1] / 'results'; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': .3, 'axes.unicode_minus': False})
INK, BLUE, ORANGE, GREEN = '#222222', '#2b6cb0', '#dd6b20', '#2f855a'


def load(name):
    p = HERE / name; return json.loads(p.read_text()) if p.exists() else None


def fig_1d(r):
    tr = r['L1-al-10MPa']['traces']; t = np.array(tr['t_us']); xs = tr['x_mm']; c = r['L1-al-10MPa']['c_theory_m_s']
    lin = np.array(tr['linear']); geo = np.array(tr['geometric']); mat = np.array(tr['material'])
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    for j, x in enumerate(xs): ax[0].plot(t, lin[j] * 1e9, lw=.8, label=f'{x:g} mm')
    ax[0].set(xlabel='t (μs)', ylabel=r'$u_1$ (nm)', title='线性响应（各接收点）'); ax[0].legend(fontsize=6, ncol=2)
    dt = (t[1] - t[0]) * 1e-6
    def deriv(u):
        n = len(u); N = 1 << int(math.ceil(math.log2(4 * n))); U = np.fft.rfft(u, N); return np.fft.irfft(U * 2j * math.pi * np.fft.rfftfreq(N, dt), N)[:n]
    j = len(xs) - 1; dx = (xs[j] - xs[0]) * 1e-3; q = dx * deriv(lin[j]) ** 2 / (4 * c * c)
    for k, (inc, beta, col, label) in enumerate([(geo, -3., BLUE, '几何 β = -3'), (mat, r['L1-al-10MPa']['beta_theory']['material'], ORANGE, f"材料 β = {r['L1-al-10MPa']['beta_theory']['material']:.2f}")]):
        back = np.interp(t - dx / c * 1e6, t, inc[0], left=0, right=0); D = inc[j] - back
        a = ax[1 + k]; a.plot(t, D * 1e12, color=col, lw=1.2, label='软件：增量差'); a.plot(t, beta * q * 1e12, color=INK, lw=.8, ls='--', label=r'理论 $\beta\,\Delta x\,(\partial_t u_1)^2/4c^2$')
        a.set(xlabel='t (μs)', ylabel='pm', title=f'{label}，Δx = {dx * 1e3:g} mm'); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / 'fig1-longitudinal-shg.png'); plt.close(fig)


def fig_lamb(r):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for name, col in (('L2-S0', BLUE), ('L2-A0', ORANGE)):
        if name not in r: continue
        rows = r[name]['rows']; f = [x['f_mhz'] for x in rows]
        ax[0].plot(f, [x['cp_theory'] for x in rows], color=INK, lw=.8); ax[0].plot(f, [x['cp_measured'] for x in rows], 'o', ms=3, color=col, label=name[3:] + ' 软件')
        ax[1].plot(f, [x['cg_theory'] for x in rows], color=INK, lw=.8); ax[1].plot(f[2:-2], [x['cg_measured'] for x in rows][2:-2], 'o', ms=3, color=col, label=name[3:] + ' 软件')
    ax[0].set(xlabel='f (MHz)，板厚 2 mm', ylabel='相速度 (m/s)', title='相速度：点 = 软件，线 = Rayleigh–Lamb'); ax[0].legend(fontsize=7)
    ax[1].set(xlabel='f (MHz)，板厚 2 mm', ylabel='群速度 (m/s)', title='群速度：点 = 软件，线 = Rayleigh–Lamb'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig2-lamb-dispersion.png'); plt.close(fig)


def fig_shg(s):
    x = np.array(s['x_mm']); fig, ax = plt.subplots(figsize=(4.5, 3))
    for part, col in (('total', INK), ('geometric', BLUE), ('material', ORANGE)):
        v = np.array(s['parts'][part]['A2_over_A1sq']); ax.plot(x, v / np.abs(s['parts']['total']['A2_over_A1sq']).max(), 'o-', ms=3, color=col, lw=.8, label={'total': '总增量', 'geometric': '几何', 'material': '材料'}[part])
    dk = s['dk_theory_rad_m']; env = np.abs(np.sin(dk * x * 1e-3 / 2)); ax.plot(x, env / env.max(), ls='--', color='gray', lw=.8, label=f'|sin(Δk x/2)|，理论 Δk={dk:.1f} rad/m')
    ax.set(xlabel='传播距离 x (mm)', ylabel=r'$|A_2|/|A_1|^2$（归一化）', title=f"S0 二次谐波，{s['f_mhz']:g} MHz，板厚 2 mm"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / 'fig3-s0-second-harmonic.png'); plt.close(fig)


def fig_echo(e):
    t = np.array(e['t_us']); fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(t, np.array(e['reference']) * 1e9, color=INK, lw=1, label='长杆（无回波）')
    for name, col in (('B1-free', 'gray'), ('B1-abs6', '#b83280'), ('B1-abs13', ORANGE), ('B1-abs26', BLUE), ('B1-abs39', GREEN)):
        if name not in e: continue
        v = e[name]; label = '自由端（全反射）' if not v['layer_mm'] else f"吸收层 {v['layer_mm']:g} mm = {v['layer_mm'] / v['wavelength_mm']:.1f}λ（{v['reflection_db']:.1f} dB）"
        ax.plot(t, np.array(v['trace']) * 1e9, color=col, lw=.8, label=label)
    ax.set(xlabel='t (μs)', ylabel='u (nm)，x = 40 mm', title='吸收层回波'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / 'fig4-absorbing-echo.png'); plt.close(fig)


def fig_s1s2(r):
    """The synchronous pair: growth at the receivers (comb), growth along the wave train (end face), mode shapes."""
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3)); labels = {'total': '完整', 'geometric': '几何', 'material': '材料'}; colours = {'total': INK, 'geometric': BLUE, 'material': ORANGE}
    comb = r.get('N3-e4') or r['N3-e3']; x = np.array(comb['x_mm']); k2 = comb['k_squared']; a = ax[0]
    for part in ('material', 'total', 'geometric'):
        p = comb['parts'][part]; col = colours[part]; software = np.array(p['r_per_m'][0]) + 1j * np.array(p['r_per_m'][1]); burst = np.array(p['burst_r_per_m'][0]) + 1j * np.array(p['burst_r_per_m'][1])
        # the comb is ten wavelengths long and the theory starts every pair at its centre: the two differ by a constant, which the fit leaves in its intercept
        M = np.stack([x, np.ones_like(x)], 1).astype(complex); shift = complex(*p['intercept_per_m']) - np.linalg.lstsq(M, burst, rcond=None)[0][1]
        a.plot(x, np.abs(software) / k2 * 1e3, 'o', ms=3, color=col, label=f'软件：{labels[part]}'); a.plot(x, np.abs(burst + shift) / k2 * 1e3, '-', lw=1, color=col)
        a.plot(x, np.abs(complex(*p['theory_per_m2']).conjugate() * x * 1e-3 + complex(*p['intercept_per_m'])) / k2 * 1e3, ':', lw=.9, color=col)
    a.plot([], [], '-', color='gray', lw=1, label='理论：本次脉冲的频谱'); a.plot([], [], ':', color='gray', lw=.9, label='理论：单一频率')
    a.set(xlabel='接收点 x (mm)', ylabel=r'$|u_2/u_1^2|/k^2$ (mm)', title='梳状激励：32 个接收点（截距取拟合值）'); a.legend(fontsize=6)
    end = r.get('N4-e4') or r.get('N4-e3'); a = ax[1]
    if end:
        k2 = end['k_squared']
        for s, marker in zip(end['snapshots'], ('o', 's')):
            for part in ('material', 'total', 'geometric'):
                p = s['parts'][part]; xs = np.array(p['local_x_mm']); col = colours[part]
                a.plot(xs[::4], np.array(p['local_ratio_abs_per_m'])[::4] / k2 * 1e3, marker, ms=2.5, color=col, label=f"软件：{labels[part]}" if marker == 'o' else None)
                a.plot(xs, np.array(p['burst_local_ratio_abs_per_m']) / k2 * 1e3, '-', lw=.9, color=col); a.plot(xs, np.hypot(*p['theory_per_m2']) * xs * 1e-3 / k2 * 1e3, ':', lw=.9, color=col)
        a.plot([], [], '-', color='gray', lw=.9, label='理论：本次脉冲的频谱'); a.plot([], [], ':', color='gray', lw=.9, label=r'理论：单一频率 $\kappa_2 x$')
        a.set(xlabel='沿波包 x (mm)，x = 0 为加载端', ylabel=r'$|u_2/u_1^2|/k^2$ (mm)', title=f"端面激励：{end['snapshots'][0]['time_us']:.0f} 与 {end['snapshots'][-1]['time_us']:.0f} μs 的整场"); a.legend(fontsize=6)
    a = ax[2]; shapes = (end or comb)['snapshots'][0]['mode_shapes']
    for label, col in (('S1', BLUE), ('S2', ORANGE)):
        m = shapes[label]; z = np.array(m['z_mm'])
        for key, style, name, axis in (('ux', '-', '$u_x$', 0), ('uz', '--', '$u_z$', 1)):   # u_z is a quarter period from u_x: its imaginary part
            a.plot(np.array(m[key + '_theory'])[:, axis], z, style, color=col, lw=1, label=f'{label} {name} 理论'); a.plot(np.array(m[key])[:, axis], z, 'o', ms=3, color=col)
    a.set(xlabel='位移（上表面 $u_x$ = 1）', ylabel='z (mm)，0 为中面', title='板厚方向模态形状（点 = 软件）'); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / 'fig5-s1s2-synchronous-pair.png'); plt.close(fig)


if __name__ == '__main__':
    for f in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei'):
        if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
    r1, r2, s = load('results-1d.json'), load('results-2d.json'), load('results-shg.json')
    if r1 and 'L1-al-10MPa' in r1: fig_1d(r1)
    if r2: fig_lamb(r2)
    if r2 and 'echo' in r2: fig_echo(r2['echo'])
    if s: fig_shg(s)
    s1s2 = load('results-s1s2.json')
    if s1s2 and 'N3-e3' in s1s2: fig_s1s2(s1s2)
    print(sorted(p.name for p in OUT.glob('*.png')))
