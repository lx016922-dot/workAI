"""Compare entire retained fields using high-order interpolation, not probes.

Integrals use the finer GLL mass quadrature. This is a discrete mass-weighted
RMS check, not a pointwise upper bound or a guarantee for every projection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .mesh import interpolation
from .compare import digest,metric,complex_coefficient,FLOOR_M


def field(root,case,law):
    control=next(c for c in case['controls'] if c['law']==law);path=root/control['directory']/'field.npz'
    assert digest(path)==control['field_sha256']
    with np.load(path,allow_pickle=False) as data:value=data['displacement_m']
    assert np.isfinite(value).all()
    return value


def weighted(a,b,mass):
    weight=mass/mass.sum();absolute=float(np.sqrt(np.sum(np.mean(abs(a-b)**2,axis=(0,2))*weight)));scale=float(np.sqrt(np.sum(np.mean(abs(b)**2,axis=(0,2))*weight)))
    above=scale>FLOOR_M;rel=absolute/scale if above else None
    return {'relative_l2':rel,'difference_rms_nm':absolute*1e9,'finer_rms_nm':scale*1e9,'above_reporting_floor':above,'pass':bool(rel<=.05 if above else absolute<=FLOOR_M)}


def compare_fields(left_path,right_path):
    left=json.loads(left_path.read_text());right=json.loads(right_path.read_text());assert left['phase']==right['phase']=='completed';rows=[]
    for case in right['cases']:
        prior=next(c for c in left['cases'] if c['case_id']==case['case_id'])
        assert prior['scene']==case['scene']
        for path,item in [(left_path,prior),(right_path,case)]:assert digest(path.parent/item['mesh_file'])==item['mesh_sha256']
        with np.load(left_path.parent/prior['mesh_file'],allow_pickle=False) as data:coarse=dict(data)
        with np.load(right_path.parent/case['mesh_file'],allow_pickle=False) as data:fine=dict(data)
        r=[];c=[];values=[];extrapolated=0
        for i,point in enumerate(fine['points']*1000):
            try:ids,w=interpolation(coarse,point)
            except ValueError:
                if str(coarse.get('mapping_kind',''))!='isoparametric':raise
                # Different polynomial representations of the same curved surface
                # can put a finer boundary node just outside the coarse domain.
                # Only this comparison permits tightly bounded extrapolation;
                # receiver queries keep the strict material-domain check.
                from .curved_mesh import interpolation as curved_interpolation
                ids,w=curved_interpolation(coarse,point,local_tolerance=.002);extrapolated+=1
            keep=abs(w)>1e-15;r.extend([i]*int(keep.sum()));c.extend(ids[keep]);values.extend(w[keep])
        transfer=csr_matrix((values,(r,c)),shape=(len(fine['points']),len(coarse['points'])))
        np.testing.assert_allclose(transfer@coarse['points'],fine['points'],rtol=1e-12,atol=1e-15)
        checks={};material_a=None;material_b=None
        for law,part in [('murnaghan','full'),('stvk','material')]:
            a=field(left_path.parent,prior,law);b=field(right_path.parent,case,law)
            if part=='full':material_a=a.copy();material_b=b.copy()
            else:a=material_a-a;b=material_b-b
            mapped=np.empty_like(b)
            for start in range(0,len(a),16):
                chunk=a[start:start+16];mapped[start:start+len(chunk)]=(transfer@chunk.transpose(1,0,2).reshape(len(coarse['points']),-1)).reshape(len(fine['points']),len(chunk),3).transpose(1,0,2)
            checks[part+'_vector']=weighted(mapped,b,fine['mass'])
            for axis,name in enumerate('xyz'):checks[part+'_'+name]=weighted(mapped[:,:,axis:axis+1],b[:,:,axis:axis+1],fine['mass'])
            if part=='material':
                times=np.arange(len(a))*case['scene']['acquisition']['sample_step_us'];frequency=2*case['scene']['source']['frequency_mhz']
                ac=complex_coefficient(times,mapped,frequency);bc=complex_coefficient(times,b,frequency)
                checks['material_2f_vector']=weighted(ac[None],bc[None],fine['mass'])
                for axis,name in enumerate('xyz'):checks['material_2f_'+name]=weighted(ac[None,:,axis:axis+1],bc[None,:,axis:axis+1],fine['mass'])
        row={'case_id':case['case_id'],'checks':checks,'all_checks_pass':all(c['pass'] for c in checks.values()),'coarse_nodes':len(coarse['points']),'finer_nodes':len(fine['points']),'frames':len(a),'curved_boundary_extrapolated_nodes':extrapolated,'maximum_allowed_reference_coordinate_excess':.002};rows.append(row);print(json.dumps(row),flush=True)
    return {'schema':'workai.spectral_field_comparison.v1','left_manifest_sha256':digest(left_path),'right_manifest_sha256':digest(right_path),
            'cases':rows,'all_checks_pass':all(c['all_checks_pass'] for c in rows),'threshold':.05,'reporting_floor_nm':FLOOR_M*1e9,
            'scope':'All stored times; coarse high-order interpolation at finer nodes; finer GLL mass quadrature. Curved geometry discrepancies allow reference-coordinate extrapolation up to .002, reported separately. Not a continuous-domain or pointwise error bound.',
            'accepted_for_solver_routing':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=compare_fields(a.left,a.right)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
