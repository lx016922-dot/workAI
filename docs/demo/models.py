"""The models of the feature demonstration (docs/demo): written as model files the workbench opens with 文件 → 打开.

    python docs/demo/models.py <dir> [http://127.0.0.1:8765]

With the address of a running workbench every model is sent to its pre-flight check and the mesh size and estimate printed.
"""
import copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = json.loads((REPO / 'coarsegraph/specs/labels-b-end-axial.json').read_text(encoding='utf-8'))['base_scene']


def scene(name, length, width, thickness, duration, source, receivers, boundaries, h=1., step=.05, **extra):
    s = copy.deepcopy(BASE); s['name'] = name; s['geometry'].update(length_mm=float(length), width_mm=float(width), thickness_mm=float(thickness))
    s['boundaries'] = boundaries; s['receiver_lines'] = []; s['source'] = dict(s['source'], **source); s['acquisition'] = {'duration_us': float(duration), 'sample_step_us': step}
    s['numerics'] = {'h_mm': h, 'cfl': .3, 'max_elements': 40000}; s['receivers'] = receivers; s.update(extra); return s


def absorbing(face, length=40.): return {'type': 'absorbing', 'face': face, 'length_mm': length, 'ramp': 'quadratic', 'attenuation_db': 60.}
def symmetry(*faces): return [{'type': 'symmetry', 'face': f} for f in faces]
def line(xs, y, z, direction, prefix='x'): return [{'name': f'{prefix}{x:g}', 'point_mm': [float(x), y, z], 'direction': direction} for x in xs]


def s0_plate(name, damping=None):
    """S0 in a 1 mm plate at 0.2 MHz (upper half, plane strain): the second harmonic is nearly phase matched and grows with distance."""
    s = scene(name, 310., 1., .5, 105., {'face': 'xmin', 'center_mm': [0., .5, .25], 'direction': [1., 0., 0.], 'amplitude_mpa': 5., 'frequency_mhz': .2, 'cycles': 10.},
              line(range(30, 211, 20), .5, .5, [1., 0., 0.]), symmetry('ymin', 'ymax', 'zmin') + [absorbing('xmax', 70.)])
    if damping: s['damping'] = damping
    return s


def transducer(name, profile, direction, amplitude=1.):
    """A 2 mm plate (whole thickness, plane strain) at 0.5 MHz with a transducer on its top face and receivers on both sides."""
    s = scene(name, 240., 1., 2., 70., {'face': 'zmax', 'center_mm': [120., .5, 2.], 'radius_mm': 0., 'direction': direction, 'amplitude_mpa': amplitude, 'frequency_mhz': .5, 'cycles': 10., 'delay_us': 0., 'profile': profile},
              line([150. + 2. * i for i in range(16)], .5, 2., [0., 0., 1.], 'R') + line([90. - 2. * i for i in range(16)], .5, 2., [0., 0., 1.], 'L'), symmetry('ymin', 'ymax') + [absorbing('xmin'), absorbing('xmax')])
    return s


def two_tone(name):
    """Two collinear plane longitudinal waves (0.4 and 0.5 MHz) in a bar: the case of research/wave-mixing."""
    signal = {'type': 'multi_tone', 'duration_us': 40., 'window': 'hann', 'components': [{'frequency_mhz': .4, 'weight': 1., 'phase_deg': 0.}, {'frequency_mhz': .5, 'weight': 1., 'phase_deg': 0.}]}
    return scene(name, 300., .8, .8, 72., {'face': 'xmin', 'center_mm': [0., .4, .4], 'direction': [1., 0., 0.], 'amplitude_mpa': 10., 'frequency_mhz': .5, 'cycles': 1., 'signal': signal},
                 line([20. + 3.2 * i for i in range(32)], .4, .4, [1., 0., 0.]), symmetry('ymin', 'ymax', 'zmin', 'zmax'), h=.8)


def cracked(name, closed):
    """S0 at 0.3 MHz meets a surface crack half the thickness deep in a 2 mm plate (plane strain)."""
    s = scene(name, 205., 1., 2., 75., {'face': 'xmin', 'center_mm': [0., .5, 1.], 'direction': [1., 0., 0.], 'amplitude_mpa': 5., 'frequency_mhz': .3, 'cycles': 5.},
              line([40., 60., 80.], .5, 2., [1., 0., 0.], 'before') + line([120., 140.], .5, 2., [1., 0., 0.], 'after'), symmetry('ymin', 'ymax') + [absorbing('xmax', 45.)])
    s['defects'] = [{'type': 'crack', 'name': '表面裂纹', 'x_mm': 100., 'depth_mm': 1., 'closed': closed}]
    return s


def disc(name):
    """A quarter of a plate with a piezoelectric disc at its corner: the waves spread in the plane (three-dimensional)."""
    profile = {'type': 'patch', 'shape': 'disc', 'length_mm': 10., 'width_mm': 0., 'edge_mm': 1.5}
    return scene(name, 84., 84., 1.6, 30., {'face': 'zmax', 'center_mm': [0., 0., 1.6], 'radius_mm': 0., 'direction': [1., 0., 0.], 'amplitude_mpa': 1., 'frequency_mhz': .3, 'cycles': 3., 'delay_us': 0., 'profile': profile},
                 [{'name': f'r{r:g}', 'point_mm': [r * .7071, r * .7071, 1.6], 'direction': [.7071, .7071, 0.]} for r in (20., 30., 40.)], symmetry('xmin', 'ymin') + [absorbing('xmax', 24.), absorbing('ymax', 24.)], h=1.2)


DAMPING = {'mass_per_s': 8000., 'stiffness_s': 2e-9}
WEDGE = {'type': 'wedge', 'along': 'x', 'length_mm': 32., 'width_mm': 0., 'trace_velocity_m_s': 2331., 'reverse': False, 'apodization': 'hann'}
PATCH = {'type': 'patch', 'shape': 'strip', 'length_mm': 5.3, 'width_mm': 0., 'edge_mm': .5}
MODELS = {
    '01-S0-二次谐波累积': ({'scene': s0_plate('1 mm 板 S0 · 二次谐波累积'), 'degree': 4, 'record': 'full'}),
    '02-S0-有材料阻尼': ({'scene': s0_plate('1 mm 板 S0 · 有材料阻尼', DAMPING), 'degree': 4, 'record': 'full'}),
    '03-斜楔-A0': ({'scene': transducer('2 mm 板 · 斜楔换能器激发 A0', WEDGE, [0., 0., -1.]), 'degree': 4, 'controls': 'full_linear', 'record': 'full'}),
    '04-压电片-条形': ({'scene': transducer('2 mm 板 · 条形压电片', PATCH, [1., 0., 0.]), 'degree': 4, 'controls': 'full_linear', 'record': 'full'}),
    '05-双频混频': ({'scene': two_tone('细杆 · 0.4 与 0.5 MHz 混频'), 'degree': 4}),
    '06-表面裂纹-张开': ({'scene': cracked('2 mm 板 · S0 遇到表面裂纹', False), 'degree': 4, 'controls': 'full_linear', 'record': 'full'}),
    '07-表面裂纹-闭合': ({'scene': cracked('2 mm 板 · S0 遇到闭合裂纹', True), 'degree': 4, 'controls': 'full_linear', 'record': 'full'}),
    '08-压电圆片-三维': ({'scene': disc('三维板 · 压电圆片（四分之一模型）'), 'degree': 4, 'controls': 'full_linear'}),
}

if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for name, document in MODELS.items(): (out / f'{name}.json').write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding='utf-8')
    if len(sys.argv) > 2:
        from urllib.request import Request, build_opener, ProxyHandler
        opener = build_opener(ProxyHandler({})); token = json.loads(opener.open(sys.argv[2] + '/api/info').read())['token']
        for name, document in MODELS.items():
            body = dict(document, mesh='auto'); headers = {'Content-Type': 'application/json', 'X-WorkAI-Token': token}
            try: r = json.loads(opener.open(Request(sys.argv[2] + '/api/preview', json.dumps(body).encode(), headers), timeout=300).read())
            except Exception as e: print(name, 'FAILED', getattr(e, 'read', lambda: b'')()[:400].decode('utf-8', 'replace') or e); continue
            auto = (r.get('mesh') or {}).get('auto') or {}; est = r.get('estimate') or {}
            print(name, 'ok' if r.get('ok') else 'NOT OK', auto.get('nodes'), auto.get('h_mm'), {k: est.get(k) for k in ('seconds', 'memory_bytes', 'record_mode')}, [m.get('text', '')[:90] for m in r.get('messages', []) if m.get('level') in ('error', 'warn')][:4])
