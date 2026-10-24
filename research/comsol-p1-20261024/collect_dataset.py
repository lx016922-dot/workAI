"""Convert only signed, completed COMSOL jobs into the uploaded CSV contract."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LAWS = ['linear', 'stvk', 'murnaghan']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def properties(path):
    result = {}
    for line in path.read_text(encoding='latin1').splitlines():
        if not line or line.startswith(('#','!')):
            continue
        key, value = line.split('=',1)
        value = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1],16)), value)
        result[key] = re.sub(r'\\([:=\\])', r'\1', value)
    return result


def verified(folder):
    collection = json.loads((folder/'collection.json').read_text())
    status_path = folder/'status.json'
    status = json.loads(status_path.read_text())
    identity = json.loads((folder/'identity.json').read_text())
    if collection['phase']!='completed' or status['body']['phase']!='completed' or not status['server_signature_verified']:
        raise ValueError(f'Not a verified completed job: {folder}')
    if digest(status_path)!=collection['signed_status_sha256']:
        raise ValueError('Changed signed status receipt')
    for item in collection['artifacts']:
        path=folder/item['name']
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError(f'Artifact digest mismatch: {path}')
    receipt=properties(folder/'completed.properties')
    for key in ('job_id','source_sha256','scene_sha256','parameters_sha256'):
        if receipt[key]!=identity[key]:
            raise ValueError(f'COMSOL receipt identity mismatch: {key}')
    if digest(folder/'UltraResearch.java')!=identity['source_sha256']:
        raise ValueError('Changed model source')
    for name in ('scene','parameters'):
        if digest(folder/f'{name}.json')!=identity[f'{name}_sha256']:
            raise ValueError('Changed input')
    return identity, receipt


def step_counts(folder, expected_laws, max_step_us):
    log=(folder/'solver.log').read_text(encoding='utf-8-sig')
    sections=re.split(r'Step\s+Time\s+Stepsize[^\n]*\n',log)[1:]
    if len(sections)!=len(expected_laws):
        raise ValueError('Unexpected transient solver log sections')
    result={}
    for law, section in zip(expected_laws, sections):
        steps=[]
        for match in re.finditer(r'^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+|-)\s+(out\s+)?(.*)$', section, re.M):
            count, when, dt, output, _=match.groups()
            steps.append((int(count),float(when),None if dt=='-' else float(dt),bool(output)))
        if not steps or abs(steps[-1][1]-30e-6)>1e-12:
            raise ValueError('Solver log does not reach 30 microseconds')
        max_dt=max(row[2] or 0 for row in steps)
        if max_dt>max_step_us*1e-6*1.001:
            raise ValueError('Internal time step exceeded contract')
        result[law]=dict(time_steps=max(row[0] for row in steps),
                         maximum_logged_step_us=max_dt*1e6,
                         logged_output_hits=sum(row[3] for row in steps),
                         step_count_source='full COMSOL solver.log, final accepted step index')
    return result


def material_check(folder):
    identity, receipt=verified(folder)
    if receipt['analysis']!='material_check' or int(receipt['volume_hexahedra'])!=1:
        raise ValueError('Single-element material check required')
    values=np.genfromtxt(folder/'coarse.csv',delimiter=',',names=True)
    if len(values)!=27:
        raise ValueError('Three 3x3 stress tensors required')
    expected=np.array([[[109.563,0,0],[0,57.957,0],[0,0,57.957]],
                       [[-.594,52.2,0],[52.2,-.594,0],[0,0,-.249]],
                       [[120.89172,20.71218,-10.30434],[20.71218,43.28142,15.57726],[-10.30434,15.57726,105.52146]]])
    actual=values['S_MPa'].reshape(3,3,3)
    if not np.allclose(actual,expected,atol=1e-5,rtol=0):
        raise ValueError('COMSOL constitutive self-check failed')
    return dict(uniaxial_E11_1e_3_S_MPa=actual[0].tolist(), shear_E12_1e_3_S_MPa=actual[1].tolist(),
                general_S_MPa=actual[2].tolist(), job_id=identity['job_id'], source_sha256=identity['source_sha256'],
                max_absolute_error_MPa=float(np.max(abs(actual-expected))), passed=True)


def archive(folder, target):
    target.mkdir(parents=True, exist_ok=True)
    allowed=['identity.json','scene.json','parameters.json','UltraResearch.java','collection.json','status.json',
             'submitted.json','request.json','case.properties','completed.properties','compile.log','solver.log','worker.log',
             'coarse.csv','fine.csv','timecheck.csv','coarse.field.csv','fine.field.csv','timecheck.field.csv']
    for name in allowed:
        if (folder/name).exists():
            shutil.copyfile(folder/name,target/name)


def collect(args):
    check=material_check(args.material_check)
    args.output.mkdir(parents=True,exist_ok=True)
    evidence=args.output.parent/'evidence'
    archive(args.material_check,evidence/'material-check')
    write(args.output/'material-check.json',check)
    header=(REPO/'docs/comsol/header-template.csv').read_text().strip()
    if len(header.split(','))!=40:
        raise ValueError('Expected 40-column contract')
    for folder in args.runs:
        identity, receipt=verified(folder)
        if receipt['analysis']!='transient':
            raise ValueError('Transient run required')
        if identity['source_sha256']!=check['source_sha256']:
            raise ValueError('Material self-check must cover the same source revision')
        reference=json.loads((folder/'scene.json').read_text())
        cfg,scene=reference['execution'],reference['scene']
        level=cfg['level']
        raw=np.genfromtxt(folder/f'{level}.csv',delimiter=',',names=True)
        guards=np.genfromtxt(folder/f'{level}.field.csv',delimiter=',',names=True)
        if len(raw)!=len(identity['laws'])*13*601 or len(guards)!=len(identity['laws'])*601:
            raise ValueError('Incomplete output/guard histories')
        steps=step_counts(folder,identity['laws'],cfg['max_step_us'])
        meta_path=args.output/f"{identity['case_id']}.json"
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
        else:
            meta=dict(case_id=identity['case_id'],comsol_version=receipt['comsol_version'],
                      element=dict(shape='hexahedral, swept',order='quartic Lagrange',degree=4,
                                   discretization='displacement; linear Hooke or total Lagrangian P=F*S'),
                      integration_order=20, time_stepping=dict(method='generalized-alpha',rho_infinity=1.,
                      max_step_us=.01,output_step_us=.05,steps='strict'),
                      tolerances=dict(time_relative=cfg['relative_tolerance'],
                                      absolute_m=cfg['absolute_tolerance_m'],
                                      newton='solution-change termination; time relative tolerance times 0.001 = 1e-9',
                                      notes='Unscaled absolute tolerance 1e-18 m; no residual-only Newton termination'),
                      load=dict(type='nominal traction per reference area (dead load, fixed direction)',sources=scene['sources']),
                      absorbing=dict(method='body load -rho0*alpha(x)*du/dt per reference volume in all three components',
                                     alpha_max_per_s=scene['alpha_max_per_s'],ramp_power=3,faces=scene['absorbing_faces']),
                      geometry=scene['geometry'],material=scene['material'],receivers=scene['receivers'],
                      material_check={'uniaxial_E11_1e-3_S_MPa':check['uniaxial_E11_1e_3_S_MPa'],
                                      'shear_E12_1e-3_S_MPa':check['shear_E12_1e_3_S_MPa'],
                                      'general_S_MPa':check['general_S_MPa'],'job_id':check['job_id'],'passed':True},
                      runs={}, notes='All receiver coordinates fixed in reference configuration. All 601 output times checked with COMSOL MaxVolume. The user waived the external coarsegraph checker and full spectral-label comparison; checks cover the uploaded contract and COMSOL convergence.')
        for law in identity['laws']:
            rows=raw[raw['law']==law]
            if not np.array_equal(rows['rx'],np.tile(np.arange(13),601)):
                raise ValueError('Receiver order changed')
            times=rows['time_s'].reshape(601,13)
            if not np.allclose(times,np.arange(601)[:,None]*.05e-6,atol=1e-15,rtol=0):
                raise ValueError('Time grid mismatch')
            displacements=np.column_stack([rows[k] for k in ('u_m','v_m','w_m')]).reshape(601,39)
            if not np.isfinite(displacements).all() or np.max(abs(displacements[0]))>1e-18:
                raise ValueError('Finite/zero initial displacement check failed')
            g=guards[guards['law']==law]
            if not np.isfinite(g['max_green_strain']).all() or not np.isfinite(g['min_J']).all() or np.max(g['max_green_strain'])>.001 or np.min(g['min_J'])<=0:
                raise ValueError('Green strain/J check failed')
            name=LAWS[law]+('' if identity['variant']=='base' else '_'+identity['variant'])
            path=args.output/f"{identity['case_id']}_{name}.csv"
            matrix=np.column_stack([np.arange(601)*.05,displacements])
            np.savetxt(path,matrix,delimiter=',',header=header,comments='',fmt='%.17e')
            run=dict(solve_time_s=float(receipt[f'law{law}_solve_seconds']),
                     max_green_strain=float(np.max(g['max_green_strain'])),min_J=float(np.min(g['min_J'])),
                     guard_output_times=601, elements=int(receipt['volume_hexahedra']),dofs=int(receipt['dofs']),
                     max_element_size_mm={dim:float(receipt[f'h_axis_{i}_mm']) for i,dim in enumerate('xyz')},
                     mesh_requested_h_mm=cfg['h_mm'],max_step_us=cfg['max_step_us'],
                     job_id=identity['job_id'],source_sha256=identity['source_sha256'],
                     scene_sha256=identity['scene_sha256'],csv_sha256=digest(path),server_signature_verified=True,**steps[law])
            meta['runs'][name]=run
            if identity['variant']=='base':
                for key in ('max_element_size_mm','elements','dofs'):
                    meta[key]=run[key]
        write(meta_path,meta)
        archive(folder,evidence/folder.name)
        print(f"Collected {identity['case_id']} {identity['variant']} {identity['laws']}")


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--material-check',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=HERE/'data')
    collect(p.parse_args())
