"""Compare a SAFE candidate with independently verified 3-D transient fields.

Fixed previously defined windows; no fitting of eigenvalues to transient data.
Shape residual is a mass-L2 compatibility check, not modal power decomposition.
"""
import argparse,gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from waveguide.safe import assemble,shape_fit

def interpolate_plane(record,field,x_mm,points_m):
    nodes=np.array(record['nodes_mm'])*.001;tetra=np.array(record['tetra']);vertices=nodes[tetra];x=x_mm*.001
    candidates=np.flatnonzero((vertices[:,:,0].min(axis=1)<=x+1e-10)&(vertices[:,:,0].max(axis=1)>=x-1e-10));v=vertices[candidates];lower=v.min(axis=1);upper=v.max(axis=1)
    cells=[];weights=[]
    for yz in points_m:
        point=np.r_[x,yz];near=np.flatnonzero(np.all(point>=lower-1e-10,axis=1)&np.all(point<=upper+1e-10,axis=1));found=False
        for index in near:
            w=np.linalg.solve((v[index,1:]-v[index,0]).T,point-v[index,0]);w=np.r_[1-w.sum(),w]
            if w.min()>=-1e-7 and w.max()<=1+1e-7:
                w=np.clip(w,0,1);w/=w.sum();cells.append(candidates[index]);weights.append(w);found=True;break
        if not found:raise ValueError('Cross-section point is outside the verified 3-D mesh.')
    return np.einsum('tpac,pa->tpc',field[:,tetra[cells]],weights)

def coefficient(times,values,f_mhz,window):
    start,end=window;w=np.where((times>=start)&(times<=end),.5*(1-np.cos(2*np.pi*(times-start)/(end-start))),0)
    if start<times[0] or end>times[-1] or w.sum()<=2:raise ValueError('Incomplete analysis window.')
    flattened=values.reshape(len(times),-1);mean=w@flattened/w.sum()
    return 2*(w*np.exp(-2j*np.pi*f_mhz*times))@(flattened-mean)/w.sum()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));parser.add_argument('--study',required=True);parser.add_argument('--dispersion',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import re
    if not re.fullmatch(r'study-[a-f0-9]{32}',args.study):raise ValueError('Invalid study ID.')
    study_dir=args.root/'solid/studies'/args.study;raw=(study_dir/'result.json').read_bytes();study=json.loads(raw);status=json.loads((study_dir/'status.json').read_text());sha=hashlib.sha256(raw).hexdigest()
    assert status['phase']=='completed' and sha==status['result_sha256'] and study['study_id']==args.study
    data=args.dispersion.read_bytes();dispersion=json.loads(data);finest=dispersion['levels'][-1];mode_file=args.dispersion.parent/finest['mode_file'];assert hashlib.sha256(mode_file.read_bytes()).hexdigest()==finest['mode_file_sha256']
    scene=study['scene'];g=scene['geometry'];m=scene['material'];f=scene['source']['frequency_mhz'];assert g['kind']=='plate' and g['width_mm']==dispersion['cross_section_mm']['width'] and g['thickness_mm']==dispersion['cross_section_mm']['thickness']
    assert all(m[k]==v for k,v in dispersion['material'].items()) and finest['branches'][0]['frequency_hz']==f*1e6
    assert scene['source']['face']=='xmin' and scene['source']['radius_mm']==0 and scene['source']['direction']==[1,0,0] and scene['boundary']['fixed_face']=='none'
    with np.load(mode_file,allow_pickle=False) as saved:points=saved['points_m'];tri=saved['triangles'];fundamental=saved['fundamental'];second=saved['second']
    system=assemble(points,tri,m['rho_kg_m3'],m['lambda_pa'],m['mu_pa'])
    # Call the pre-existing window planner to preserve its exact choice. All
    # field interpolation and harmonic projection below are independent NumPy.
    source="import {planPlatePropagation} from './public/solid-propagation.js';let s='';for await(const b of process.stdin)s+=b;const scene=JSON.parse(s);console.log(JSON.stringify(scene.receivers.map(r=>planPlatePropagation(scene,r.point_mm))));"
    plans=json.loads(subprocess.check_output(['node','--input-type=module','-e',source],input=json.dumps(scene).encode(),cwd=REPO))
    assert all(p['available'] and p['passes_end_echo_screen'] for p in plans)
    planes={};records={};references=[]
    for control in study['controls'][:3]:
        job=args.root/'solid/jobs'/control['job_id'];rraw=(job/'result.json').read_bytes();r=json.loads(rraw);completed=json.loads((job/'status.json').read_text());assert completed['phase']=='completed' and hashlib.sha256(rraw).hexdigest()==control['result_sha256']==completed['result_sha256']
        packed=(job/'field.bin.gz').read_bytes();assert hashlib.sha256(packed).hexdigest()==r['field']['sha256'];raw=gzip.decompress(packed);assert hashlib.sha256(raw).hexdigest()==r['field']['uncompressed_sha256']
        assert r['field']['dtype']=='little_endian_float64';field=np.frombuffer(raw,dtype='<f8').reshape(r['field']['shape']);assert np.isfinite(field).all()
        if records:
            base=next(iter(records.values()));assert r['nodes_mm']==base['nodes_mm'] and r['tetra']==base['tetra'] and r['times_us']==base['times_us'] and r['source_sha256']==base['source_sha256']
        records[control['role']]=r;planes[control['role']]=[interpolate_plane(r,field,rx['point_mm'][0],points) for rx in scene['receivers']];references.append(control)
    rows=[];times=np.array(records['linear_A']['times_us'])
    for i,rx in enumerate(scene['receivers']):
        window=plans[i]['window_us'];linear=coefficient(times,planes['linear_A'][i],f,window);material=coefficient(times,planes['material_A'][i]-planes['geometric_A'][i],2*f,window)
        rows.append({'point_mm':rx['point_mm'],'window_us':window,'linear_fundamental_shape':shape_fit(system,fundamental,linear),'material_second_shape':shape_fit(system,second,material),
            'same_branch_group_walkoff_us':rx['point_mm'][0]*1000*abs(1/finest['branches'][1]['group_velocity_m_s']-1/finest['branches'][0]['group_velocity_m_s'])})
    phase=[];k=finest['branches'][0]['wavenumber_rad_m']
    for a,b in zip(rows,rows[1:]):
        dx=(b['point_mm'][0]-a['point_mm'][0])*.001;za=complex(*a['linear_fundamental_shape']['complex_coefficient']);zb=complex(*b['linear_fundamental_shape']['complex_coefficient']);expected=-k*dx
        residual=np.angle(np.exp(1j*(np.angle(zb/za)-expected)));phase.append({'from_mm':a['point_mm'][0],'to_mm':b['point_mm'][0],'relative_wavenumber_difference':float(-residual/(k*dx)),'spacing_below_half_wavelength':bool(k*dx<np.pi)})
    record={'schema':'workai.rectangular-waveguide.validation.v1','study_id':args.study,'study_result_sha256':sha,'transient_references':references,'dispersion_sha256':hashlib.sha256(data).hexdigest(),'source_sha256':{'comparison':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'safe':hashlib.sha256((REPO/'waveguide/safe.py').read_bytes()).hexdigest()},
        'cross_section_mm':dispersion['cross_section_mm'],'material':dispersion['material'],'frequency_mhz':f,'branches':[{k:v for k,v in r.items() if k!='tracking_path'} for r in finest['branches']],
        'successive_relative_k_differences':dispersion['successive_relative_k_differences'],'dispersion_finest_pair_within_one_percent':dispersion['finest_pair_within_one_percent'],'mismatch_rad_m':dispersion['mismatch_rad_m'],'coherence_length_mm':dispersion['coherence_length_mm'],
        'rows':rows,'phase_checks':phase,'fundamental_candidate_checks_within_5_percent':dispersion['finest_pair_within_one_percent'] and all(p['spacing_below_half_wavelength'] and abs(p['relative_wavenumber_difference'])<=.05 for p in phase) and all(r['linear_fundamental_shape']['relative_mass_L2_shape_mismatch']<=.05 for r in rows),
        'modal_purity_verified':False,'nonlinear_coupling_integral_computed':False,'other_second_harmonic_modes_excluded':False,'window_changed_for_this_fit':False,'scope':'Mass-L2 single-shape compatibility and phase, not modal energy decomposition. Same continued branch at 2f is only one candidate; all possible nonlinear receiving modes were not enumerated.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:json.dump(record,file,indent=2)
    print(json.dumps({'study_id':args.study,'shape_f1':[r['linear_fundamental_shape']['relative_mass_L2_shape_mismatch'] for r in rows],'shape_material_2f':[r['material_second_shape']['relative_mass_L2_shape_mismatch'] for r in rows],'phase_checks':phase,'fundamental_candidate_checks':record['fundamental_candidate_checks_within_5_percent']}))

if __name__=='__main__':main()
