"""Prepare a bounded, signed-transport-compatible COMSOL 3D reference case.

The scene is explicit about its 3D geometry, essential boundary and constitutive
constants. The legacy transport's ``free`` token means natural boundaries only;
it is not used to disguise a fixed end or to exceed its time/export limits.
"""
import argparse
import copy
import hashlib
import json
import math
import uuid
from pathlib import Path


def write(path, value):
    # The existing Windows PowerShell 5.1 worker reads JSON through the system
    # code page. ASCII JSON escapes preserve Unicode labels without changing
    # the signed bytes or altering the established server protocol.
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='ascii')


def prepare(args):
    payload = json.loads(args.input.read_text())
    scene = copy.deepcopy(payload.get('scene', payload))
    scene.setdefault('signal', {'kind': 'hann', 'end_frequency_mhz': scene['source']['frequency_mhz']})
    if args.signal:
        scene['signal']['kind'] = args.signal
    if args.cycles:
        scene['source']['cycles'] = args.cycles
    if args.duration:
        scene['acquisition']['duration_us'] = args.duration
    if args.sample:
        scene['acquisition']['sample_step_us'] = args.sample
    if args.frequency:
        scene['source']['frequency_mhz'] = args.frequency
    if args.end_frequency:
        scene['signal']['end_frequency_mhz'] = args.end_frequency
    if args.amplitude:
        scene['source']['amplitude_mpa'] = args.amplitude
    acq, geom, mat, tx = (scene[k] for k in ('acquisition', 'geometry', 'material', 'source'))
    if tx['radius_mm'] != 0:
        raise ValueError('Current 3D reference uses uniform whole-face traction only.')
    duration, sample = acq['duration_us'], acq['sample_step_us']
    receivers = scene['receivers']
    laws = args.laws
    if len(set(laws)) != len(laws) or not 1 <= len(receivers) <= 8:
        raise ValueError('Invalid control/receiver count.')
    if duration / (args.step * .5) > 100000.1 or args.step > sample:
        raise ValueError('Existing server time budget exceeded; do not bypass it.')
    if len(laws) * len(receivers) * (duration / sample + 1) > 2_000_000:
        raise ValueError('Existing server export budget exceeded.')
    execution = dict(level=args.level, coarse_h_mm=args.coarse_h, fine_h_mm=args.fine_h,
                     max_step_us=args.step, relative_tolerance=args.rtol,
                     absolute_tolerance_m=args.atol, quadrature_order=args.quadrature,
                     solver=args.solver, laws=laws, field_nx=3, field_ny=3,
                     field_nz=3, field_frames=args.guard_frames)
    execution.update(mesh_type=args.mesh_type, analysis=args.analysis,artifact_encoding=args.artifact_encoding)
    if args.analysis=='eigen':
        if laws!=[0] or geom['kind']!='plate':
            raise ValueError('Eigen label scope: linear rectangular plate only.')
        execution.update(eigenmodes=args.eigenmodes,sample_degree=4,sample_h_mm=args.sample_h,
                         field_nx=4*math.ceil(geom['length_mm']/args.sample_h)+1,
                         field_ny=4*math.ceil(geom['width_mm']/args.sample_h)+1,
                         field_nz=4*math.ceil(geom['thickness_mm']/args.sample_h)+1,
                         field_frames=args.eigenmodes)
    reference = dict(schema='workai.comsol.graphwave3d.scene.v1', case_id=args.case_id,
                     scene=scene, execution=execution)
    p = dict(schema='workai.comsol.research.case.v1', scenario='guided_plate', boundary='free',
             length_mm=geom['length_mm'], height_mm=geom['thickness_mm'], axisymmetric=0,
             outer_radius_mm=geom['outer_radius_mm'], rho=mat['rho_kg_m3'],
             cl=math.sqrt((mat['lambda_pa']+2*mat['mu_pa'])/mat['rho_kg_m3']),
             ct=math.sqrt(mat['mu_pa']/mat['rho_kg_m3']), conductivity=0,
             heat_capacity=1, expansion=0, thermal=0, duration_us=duration,
             sample_step_us=sample, relative_tolerance=args.rtol, element_order=4,
             element_budget=args.element_budget, field_nx=3, field_ny=3,
             field_frames=args.guard_frames, notch_enabled=int(geom['kind']=='notch'),
             notch_x_mm=geom['notch_x_mm'], notch_width_mm=geom['notch_width_mm'],
             notch_depth_mm=geom['notch_depth_mm'], notch_bottom=0, shot_count=len(laws),
             receiver_count=len(receivers), coarse_h_mm=args.coarse_h, fine_h_mm=args.fine_h,
             max_step_us=args.step, bandwidth_estimate_mhz=max(tx['frequency_mhz'], scene['signal']['end_frequency_mhz']))
    for key in ('field_nx','field_ny','field_frames'):
        p[key]=execution[key]
    for j, r in enumerate(receivers):
        p[f'rx{j}_x_mm'] = r['point_mm'][0]
    for j, _law in enumerate(laws):
        values = dict(kind=0, x_mm=0 if tx['face']=='xmin' else geom['length_mm'],
                      radius_mm=0, delay_us=tx['delay_us'], phase_deg=tx['phase_deg'],
                      cycles=tx['cycles'], amplitude_pa=tx['amplitude_mpa']*1e6,
                      direction=1, frequency_count=1, f0=tx['frequency_mhz'], elements=1,
                      pitch_mm=0, angle_deg=0, wedge_incidence_deg=0, wedge_cl=0,
                      pulse_us=0, absorbed_j_per_m=0, gaussian_integral_m=0)
        p.update({f'tx{j}_{key}': value for key, value in values.items()})
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'scene.json', reference)
    write(args.output/'parameters.json', p)
    write(args.output/'input.json', dict(scene=scene, degree=4, h_mm=args.coarse_h))
    write(args.output/'identity.json', dict(job_id='comsol-'+uuid.uuid4().hex,
          source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
          scene_sha256=hashlib.sha256((args.output/'scene.json').read_bytes()).hexdigest(),
          parameters_sha256=hashlib.sha256((args.output/'parameters.json').read_bytes()).hexdigest(),
          purpose=args.purpose, uses_test_data_for_training=False))
    print((args.output/'identity.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('research/comsol-validation/UltraResearch.java'))
    p.add_argument('--case-id', required=True)
    p.add_argument('--purpose', choices=['engineering_pilot', 'reference_convergence', 'training', 'validation', 'heldout_test'], required=True)
    p.add_argument('--level', choices=['coarse', 'fine', 'timecheck'], default='coarse')
    p.add_argument('--laws', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    p.add_argument('--coarse-h', type=float, default=2.5)
    p.add_argument('--fine-h', type=float, default=1.5)
    p.add_argument('--step', type=float, required=True)
    p.add_argument('--duration', type=float)
    p.add_argument('--sample', type=float)
    p.add_argument('--signal', choices=['hann', 'gaussian', 'rectangular', 'chirp'])
    p.add_argument('--cycles', type=int, choices=range(1, 21))
    p.add_argument('--frequency', type=float)
    p.add_argument('--end-frequency', type=float)
    p.add_argument('--amplitude', type=float)
    p.add_argument('--solver', choices=['genalpha', 'bdf', 'dopri5'], default='genalpha')
    p.add_argument('--rtol', type=float, default=1e-8)
    p.add_argument('--atol', type=float, default=1e-18)
    p.add_argument('--quadrature', type=int, default=10)
    p.add_argument('--guard-frames', type=int, default=17)
    p.add_argument('--element-budget', type=int, default=6000)
    p.add_argument('--mesh-type', choices=['tet','hex'], default='hex')
    p.add_argument('--analysis', choices=['transient','eigen'], default='transient')
    p.add_argument('--eigenmodes', type=int, default=48)
    p.add_argument('--sample-h', type=float, default=2.5)
    p.add_argument('--artifact-encoding',choices=['csv','gzip_csv'],default='gzip_csv')
    prepare(p.parse_args())
