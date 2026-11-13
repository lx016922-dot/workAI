"""Software scenes for the defect COMSOL runs (docs/comsol/defects/README.md). Run: python docs/comsol/defects/make_cases.py"""
import copy
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATERIAL = {'rho_kg_m3': 2700., 'lambda_pa': 58.1e9, 'mu_pa': 26.1e9, 'l_pa': -143e9, 'm_pa': -297e9, 'n_pa': -345e9}
RECEIVERS = [(f'top_{x:g}', [x, 2., 1.6]) for x in (20., 25., 35., 40., 45.)] + [(f'mid_{x:g}', [x, 2., .8]) for x in (20., 25., 35., 40., 45.)]


def base(name, constitutive='linear', amplitude=1.):
    return {'schema': 'workai.hyperelastic.solid.v2', 'name': name, 'constitutive': constitutive, 'material': dict(MATERIAL),
            'geometry': {'kind': 'plate', 'length_mm': 60., 'width_mm': 2., 'thickness_mm': 1.6, 'outer_radius_mm': 3., 'inner_radius_mm': 2., 'hole_radius_mm': .7,
                         'hole_x_mm': 6., 'hole_y_mm': 2., 'notch_x_mm': 6., 'notch_width_mm': 1., 'notch_depth_mm': .6},
            'source': {'name': 'top', 'face': 'zmax', 'center_mm': [15., 2., 1.6], 'radius_mm': 0., 'direction': [0., 0., -1.], 'amplitude_mpa': amplitude,
                       'frequency_mhz': .3, 'cycles': 5., 'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}, 'profile': {'type': 'gaussian', 'sigma_mm': 2.}},
            'sources': [], 'boundary': {'fixed_face': 'none'},
            'boundaries': [{'type': 'symmetry', 'face': 'ymax'}, {'type': 'absorbing', 'face': 'xmin', 'length_mm': 10., 'ramp': 'cubic', 'attenuation_db': 60.},
                           {'type': 'absorbing', 'face': 'xmax', 'length_mm': 10., 'ramp': 'cubic', 'attenuation_db': 60.}],
            'acquisition': {'duration_us': 35., 'sample_step_us': .05}, 'numerics': {'h_mm': 1., 'cfl': .3, 'max_elements': 40000},
            'receivers': [{'name': n, 'point_mm': p, 'direction': [0., 0., 1.]} for n, p in RECEIVERS]}


CASES = {
    'D0-intact': (base('D0 无缺陷参考'), []),
    'D1-open-crack': (base('D1 开口表面裂纹'), [{'type': 'crack', 'x_mm': 30., 'depth_mm': .8, 'from': 'top'}]),
    'D2-open-delamination': (base('D2 开口分层'), [{'type': 'delamination', 'x_start_mm': 25., 'x_end_mm': 35., 'depth_mm': .8}]),
    'D3-side-drilled-hole': (base('D3 横孔'), [{'type': 'hole', 'axis': 'y', 'x_mm': 30., 'z_mm': .8, 'radius_mm': .4}]),
    'D4-closed-crack-1MPa': (base('D4 闭合裂纹 1 MPa'), [{'type': 'crack', 'x_mm': 30., 'depth_mm': .8, 'from': 'top', 'closed': True}]),
    'D5-closed-crack-2MPa': (base('D5 闭合裂纹 2 MPa', amplitude=2.), [{'type': 'crack', 'x_mm': 30., 'depth_mm': .8, 'from': 'top', 'closed': True}]),
    'D6-closed-delamination': (base('D6 闭合分层'), [{'type': 'delamination', 'x_start_mm': 25., 'x_end_mm': 35., 'depth_mm': .8, 'closed': True}]),
    # Covers only part of the width: the crack front has ends in y (3D corners), which the plane-strain check cannot reach.
    'D7-part-width-crack': (base('D7 部分宽度裂纹'), [{'type': 'crack', 'x_mm': 30., 'depth_mm': .8, 'from': 'top', 'y_range_mm': [1., 2.]}]),
}


def main():
    (HERE / 'cases').mkdir(exist_ok=True)
    for name, (scene, defects) in CASES.items():
        s = copy.deepcopy(scene)
        if defects: s['defects'] = defects
        (HERE / 'cases' / f'{name}.json').write_text(json.dumps({'scene': s, 'mesh': 'auto'}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    c = math.sqrt((MATERIAL['lambda_pa'] + 2 * MATERIAL['mu_pa']) / MATERIAL['rho_kg_m3'])
    print(json.dumps({'cases': list(CASES), 'alpha_max_per_s': 4 * c * 60 * math.log(10) / (20 * .01), 'c_l_m_s': c}, indent=1))


if __name__ == '__main__': main()
