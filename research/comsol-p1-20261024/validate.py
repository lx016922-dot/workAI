"""Independent checks of the uploaded P1 contract and COMSOL convergence.

No spectral labels or coarsegraph package are required. The user explicitly
waived those external comparisons on 2026-10-24. Harmonic comparisons below use
the same rectangular 0..30 us window for both COMSOL runs and are diagnostics.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
RUNS=['linear','stvk','murnaghan','linear_mesh2','murnaghan_mesh2','linear_dt2','murnaghan_dt2']


def relative(a,b):
    denominator=float(np.linalg.norm(b))
    return None if denominator<1e-30 else float(np.linalg.norm(a-b)/denominator)


def harmonic(u,f_mhz,multiple):
    t=np.arange(601)*.05
    return (2/601)*np.exp(-2j*np.pi*f_mhz*multiple*t)@u


def validate(folder,partial=False):
    cases=[r for r in csv.DictReader((REPO/'docs/comsol/cases.csv').open()) if r['priority']=='P1']
    header=(REPO/'docs/comsol/header-template.csv').read_text().strip()
    summary=dict(schema='workai.comsol.p1.validation.v1',complete=False,expected_runs=21,
                 completed_runs=0,errors=[],warnings=[],cases={},
                 comparison_scope='COMSOL mesh/time convergence only; external checker and spectral-label comparisons waived by user',
                 harmonic_definition='2/N sum u(t) exp(-2*pi*i*k*f*t), k=1 or 2; identical rectangular 601-sample 0..30us window')
    for case in cases:
        cid=case['case_id'];meta_path=folder/f'{cid}.json';data={}
        if not meta_path.exists():
            if not partial:summary['errors'].append(f'{cid}: metadata missing')
            continue
        meta=json.loads(meta_path.read_text());metrics=dict(runs={},convergence={})
        for run in RUNS:
            path=folder/f'{cid}_{run}.csv'
            if not path.exists():
                if not partial:summary['errors'].append(f'{cid}/{run}: missing')
                continue
            lines=path.read_text().splitlines()
            if len(lines)!=602 or lines[0]!=header:
                raise ValueError(f'Header/row count: {path}')
            if not all(re.fullmatch(r'[+-]?\d\.\d{12,}[eE][+-]\d+',cell) for line in lines[1:] for cell in line.split(',')[1:]):
                raise ValueError(f'Insufficient scientific output precision: {path}')
            a=np.loadtxt(path,delimiter=',',skiprows=1)
            if a.shape!=(601,40) or not np.isfinite(a).all() or not np.allclose(a[:,0],np.arange(601)*.05,rtol=0,atol=1e-12):
                raise ValueError(f'Time grid/dimensions/finite values: {path}')
            r=meta['runs'][run]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=r['csv_sha256'] or not r['server_signature_verified']:
                raise ValueError('Data provenance mismatch')
            if r['guard_output_times']!=601 or r['max_green_strain']>.001 or r['min_J']<=0 or r['dofs']<=0 or r['elements']<=0:
                raise ValueError('Physical guard/actual metadata check failed')
            if r['max_step_us']>(.005 if run.endswith('_dt2') else .01)+1e-12 or max(r['max_element_size_mm'].values())>(.56 if run.endswith('_mesh2') else .8)+1e-12:
                raise ValueError('Discretization budget exceeded')
            if meta['time_stepping']['steps']!='strict' or meta['time_stepping']['rho_infinity']!=1 or meta['integration_order']<20 or meta['tolerances']['time_relative']>1e-6:
                raise ValueError('Solver contract mismatch')
            u=a[:,1:];data[run]=u
            projections=np.column_stack([u[:,3*i+(2 if i<7 else 0)] for i in range(13)])
            far=u[:,2]*1e9;idx=int(np.argmax(abs(far)))
            metrics['runs'][run]=dict(P_far_projected_peak_nm=float(far[idx]),P_far_peak_time_us=float(a[idx,0]),
                max_abs_projected_nm=float(np.max(abs(projections))*1e9),max_green_strain=r['max_green_strain'],min_J=r['min_J'],
                solve_time_s=r['solve_time_s'],time_steps=r['time_steps'],dofs=r['dofs'])
            summary['completed_runs']+=1
        for variant in ('mesh2','dt2'):
            for law in ('linear','murnaghan'):
                if law not in data or law+'_'+variant not in data:continue
                ref,u=data[law],data[law+'_'+variant];freq=float(case['frequency_mhz'])
                metrics['convergence'][law+'_'+variant]=dict(waveform_relative_l2=relative(u,ref),
                    waveform_relative_linf=float(np.max(abs(u-ref))/max(np.max(abs(ref)),1e-30)),
                    f_relative_complex_l2=relative(harmonic(u,freq,1),harmonic(ref,freq,1)),
                    two_f_relative_complex_l2=relative(harmonic(u,freq,2),harmonic(ref,freq,2)))
            if all(k in data for k in ['linear','murnaghan','linear_'+variant,'murnaghan_'+variant]):
                inc=data['murnaghan']-data['linear'];inc2=data['murnaghan_'+variant]-data['linear_'+variant]
                metrics['convergence']['material_plus_geometry_increment_'+variant]=dict(
                    relative_l2=relative(inc2,inc),two_f_relative_complex_l2=relative(harmonic(inc2,float(case['frequency_mhz']),2),harmonic(inc,float(case['frequency_mhz']),2)))
        if all(k in data for k in ('linear','murnaghan')):
            ratio=float(np.max(abs(data['murnaghan']-data['linear']))/np.max(abs(data['linear'])))
            metrics['murnaghan_minus_linear_relative_max']=ratio
            if ratio<1e-6 or ratio>1e-2:summary['warnings'].append(f'{cid}: nonlinear increment relative max {ratio:.6g}; inspect')
        if 'linear' in metrics['runs']:
            actual=metrics['runs']['linear'];ref=float(case['ref_P_far_projected_peak_nm']);ref_time=float(case['ref_P_far_peak_time_us'])
            metrics['reference_peak_sanity_only']=dict(reference_nm=ref,reference_time_us=ref_time,source='uploaded cases.csv; no full spectral-label comparison')
            if actual['P_far_projected_peak_nm']*ref<0 or not .5<abs(actual['P_far_projected_peak_nm']/ref)<2 or abs(actual['P_far_peak_time_us']-ref_time)>2:
                summary['warnings'].append(f'{cid}: uploaded reference peak sanity check flagged; see actual peaks')
        for name,comparison in metrics['convergence'].items():
            for metric,value in comparison.items():
                if value is not None and value>.05:summary['warnings'].append(f'{cid}/{name}/{metric}: {value:.3%}, above 5% diagnostic threshold')
        summary['cases'][cid]=metrics
    summary['complete']=summary['completed_runs']==21 and not summary['errors']
    summary['contract_checks_passed_for_available_runs']=not summary['errors']
    (folder/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['complete','expected_runs','completed_runs','errors','warnings']},ensure_ascii=False,indent=2))
    if summary['errors']:raise SystemExit(1)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--partial',action='store_true')
    a=p.parse_args();validate(a.folder,a.partial)
