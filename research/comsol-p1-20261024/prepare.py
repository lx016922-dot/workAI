"""Prepare immutable P1 inputs for the existing paired, signed COMSOL API.

The legacy free-boundary transport token means natural boundary defaults. The
signed scene and dedicated Java source explicitly add ymax symmetry and volume
mass damping. No private key or credential is read by this script.
"""
import argparse
import csv
import hashlib
import json
import math
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='ascii')


def prepare(args):
    rows = {r['case_id']: r for r in csv.DictReader((REPO/'docs/comsol/cases.csv').open())}
    case = rows[args.case_id]
    if case['priority'] != 'P1':
        raise ValueError('This campaign is limited to the three mandatory P1 cases.')
    family, thickness, frequency = case['family'], float(case['thickness_mm']), float(case['frequency_mhz'])
    receivers = [dict(name='P_far', point_mm=[36, 1, thickness])]
    receivers += [dict(name=f'top_{i}', point_mm=[22+3*i, 1, thickness]) for i in range(6)]
    receivers += [dict(name=f'mid_{i}', point_mm=[22+3*i, 1, thickness/2]) for i in range(6)]
    material = dict(rho_kg_m3=2700, lambda_pa=58.1e9, mu_pa=26.1e9, l_pa=-143e9, m_pa=-297e9, n_pa=-345e9)
    geometry = dict(length_mm=48, width_mm=2, thickness_mm=thickness)
    if args.material_check:
        geometry = dict(length_mm=1, width_mm=1, thickness_mm=1)
    h = 1.0 if args.material_check else .56 if args.variant == 'mesh2' else .8
    dt = .005 if args.variant == 'dt2' else .01
    scene = dict(schema='workai.comsol.p1.v1', analysis='material_check' if args.material_check else 'transient',
                 family=family, geometry=geometry, material=material, receivers=receivers,
                 symmetry='ymax_uy_zero_ux_uz_free', absorbing_faces=case['absorbing_faces'],
                 alpha_max_per_s=22075615.536638446, amplitude_pa=1e6, cycles=5, frequency_mhz=frequency,
                 initial_displacement_m=[0, 0, 0], initial_velocity_m_s=[0, 0, 0],
                 load_type='nominal_dead_traction_per_reference_area', receiver_frame='reference',
                 constitutive_energy='0.5*(lambda+2*mu)*i1^2-2*mu*i2+(l+2*m)/3*i1^3-2*m*i1*i2+n*i3')
    if family == 'B':
        scene['sources'] = [dict(face='xmin', spatial='uniform', direction=[1,0,0], delay_us=0)]
    else:
        scene['sources'] = [dict(face='zmax', spatial='full_face_gaussian', center_mm=[16,1], sigma_mm=2, direction=[0,0,-1], delay_us=0)] if family == 'A' else [
            dict(face='zmax', spatial='full_face_gaussian', center_mm=[13,1], sigma_mm=1.5, direction=[0,0,-1], delay_us=0),
            dict(face='zmax', spatial='full_face_gaussian', center_mm=[19,1], sigma_mm=1.5, direction=[0,0,1], delay_us=1)]
    level = {'base':'coarse', 'mesh2':'fine', 'dt2':'timecheck'}[args.variant]
    execution = dict(level=level, h_mm=h, max_step_us=dt, duration_us=30, sample_step_us=.05,
                     quadrature_order=20, degree=4, relative_tolerance=1e-6, absolute_tolerance_m=1e-18,
                     laws=args.laws, rho_infinity=1, guard_output_times=601)
    reference = dict(schema='workai.comsol.graphwave3d.scene.v1', case_id='material-check' if args.material_check else args.case_id,
                     scene=scene, execution=execution)
    p = dict(schema='workai.comsol.research.case.v1', scenario='guided_plate', boundary='free',
             length_mm=geometry['length_mm'], height_mm=geometry['thickness_mm'], axisymmetric=0, outer_radius_mm=0,
             rho=2700, cl=math.sqrt(110.3e9/2700), ct=math.sqrt(26.1e9/2700), conductivity=0, heat_capacity=1,
             expansion=0, thermal=0, duration_us=30, sample_step_us=.05, relative_tolerance=1e-6,
             element_order=4, element_budget=40000, field_nx=3, field_ny=3, field_frames=601,
             notch_enabled=0, notch_x_mm=0, notch_width_mm=0, notch_depth_mm=0, notch_bottom=0,
             shot_count=len(args.laws), receiver_count=13, coarse_h_mm=h, fine_h_mm=h,
             max_step_us=dt, bandwidth_estimate_mhz=frequency)
    for j, r in enumerate(receivers):
        p[f'rx{j}_x_mm'] = r['point_mm'][0]
    for j, law in enumerate(args.laws):
        values = dict(kind=0, x_mm=0 if family=='B' else 16, radius_mm=0, delay_us=0, phase_deg=0,
                      cycles=5, amplitude_pa=1e6, direction=1, frequency_count=1, f0=frequency,
                      elements=1, pitch_mm=0, angle_deg=0, wedge_incidence_deg=0, wedge_cl=0,
                      pulse_us=0, absorbed_j_per_m=0, gaussian_integral_m=0)
        p.update({f'tx{j}_{k}': v for k,v in values.items()})
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'scene.json', reference)
    write(args.output/'parameters.json', p)
    identity = dict(job_id='comsol-'+uuid.uuid4().hex, source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    scene_sha256=hashlib.sha256((args.output/'scene.json').read_bytes()).hexdigest(),
                    parameters_sha256=hashlib.sha256((args.output/'parameters.json').read_bytes()).hexdigest(),
                    case_id=reference['case_id'], variant=args.variant, laws=args.laws,
                    purpose='constitutive_verification' if args.material_check else 'independent_reference_validation',
                    label_split=case['label_split'], uses_test_data_for_training=False)
    write(args.output/'identity.json', identity)
    print(json.dumps(identity))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case-id', default='A-f0.40-T1.6')
    p.add_argument('--material-check', action='store_true')
    p.add_argument('--variant', choices=['base','mesh2','dt2'], default='base')
    p.add_argument('--laws', type=int, choices=[0,1,2], nargs='+', default=[0,1,2])
    p.add_argument('--source', type=Path, default=HERE/'UltraResearch.java')
    p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args())
