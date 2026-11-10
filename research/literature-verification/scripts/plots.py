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
    for name, col in (('B1-free', 'gray'), ('B1-abs13', ORANGE), ('B1-abs26', BLUE), ('B1-abs39', GREEN)):
        if name in e: ax.plot(t, np.array(e[name]['trace']) * 1e9, color=col, lw=.8, label=f"{name}（{e[name]['reflection_db']:.1f} dB）")
    ax.set(xlabel='t (μs)', ylabel='u (nm)，x = 40 mm', title='吸收层回波'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / 'fig4-absorbing-echo.png'); plt.close(fig)


if __name__ == '__main__':
    for f in ('Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei'):
        if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist): plt.rcParams['font.family'] = f; break
    r1, r2, s = load('results-1d.json'), load('results-2d.json'), load('results-shg.json')
    if r1 and 'L1-al-10MPa' in r1: fig_1d(r1)
    if r2: fig_lamb(r2)
    if r2 and 'echo' in r2: fig_echo(r2['echo'])
    if s: fig_shg(s)
    print(sorted(p.name for p in OUT.glob('*.png')))
