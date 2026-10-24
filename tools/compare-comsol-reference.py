"""Compare verified COMSOL waveforms with the frozen/continued graph network.

No phase shift, rescaling, or reference response enters the network forward pass.
Completion of this comparison is distinct from reference convergence admission.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.comsol_artifacts import numeric_csv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def props(path):
    values={}
    for line in path.read_text().splitlines():
        if line and not line.startswith(('#','!')):
            k,v=line.split('=',1);values[k]=v.replace('\\:',':')
    return values


def load_case(folder):
    collection=json.loads((folder/'collection.json').read_text())
    if collection['phase']!='completed':
        raise ValueError('Failed COMSOL job cannot be used as a reference.')
    identity=collection['identity']
    for item in collection['artifacts']:
        path=folder/item['name']
        if not item.get('verified_against_signed_manifest') or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('Unverified/changed COMSOL artifact: '+item['name'])
    config=json.loads((folder/'scene.json').read_text())
    receipt=props(folder/'completed.properties')
    for key in ['job_id','source_sha256','scene_sha256','parameters_sha256']:
        if receipt[key]!=identity[key]:
            raise ValueError('COMSOL identity mismatch: '+key)
    if receipt['method']!='comsol_3d_reference_weakform_full_murnaghan':
        raise ValueError('Not a three-dimensional transient reference.')
    if digest(folder/'scene.json')!=identity['scene_sha256']:
        raise ValueError('Scene changed')
    scene=config['scene'];acq=scene['acquisition'];laws=config['execution']['laws']
    frames=round(acq['duration_us']/acq['sample_step_us'])+1
    nr=len(scene['receivers'])
    data=np.genfromtxt(numeric_csv(folder/(receipt['level']+'.csv')),delimiter=',',names=True)
    if len(data)!=len(laws)*frames*nr or not all(np.isfinite(data[n]).all() for n in data.dtype.names):
        raise ValueError('Incomplete/nonfinite COMSOL trace')
    traces={}
    for law in laws:
        selected=data[data['law']==law]
        if len(selected)!=frames*nr or not np.array_equal(selected['rx'].reshape(frames,nr),np.tile(np.arange(nr),(frames,1))):
            raise ValueError('Receiver order or repeated/missing samples')
        times=selected['time_s'].reshape(frames,nr)
        if not np.allclose(times,np.arange(frames)[:,None]*acq['sample_step_us']*1e-6,rtol=0,atol=1e-15):
            raise ValueError('Actual COMSOL time grid differs')
        traces[law]=np.stack([selected[k].reshape(frames,nr) for k in ['u_m','v_m','w_m']],-1)
    return config,receipt,traces


def metrics(actual, expected, floor=1e-13):
    result=[]
    for a,b in zip(actual.transpose(1,0,2),expected.transpose(1,0,2)):
        row={}
        for axis,key in enumerate(['x','y','z','vector']):
            x,y=(a[:,axis],b[:,axis]) if axis<3 else (a,b)
            error=float(np.sqrt(np.sum((x-y)**2)/len(x)))
            scale=float(np.sqrt(np.sum(y*y)/len(y)))
            row[key]={'error_rms_m':error,'reference_rms_m':scale,'relative_l2':error/scale if scale>floor else None,
                      'passed':error<=(.05*scale if scale>floor else floor)}
        result.append(row)
    return result


def parts(traces):
    out={name:traces[i] for i,name in enumerate(['linear','stvk','full']) if i in traces}
    if 0 in traces and 1 in traces:out['geometric']=traces[1]-traces[0]
    if 1 in traces and 2 in traces:out['material']=traces[2]-traces[1]
    if 0 in traces and 2 in traces:out['total_nonlinear']=traces[2]-traces[0]
    return out


def frequency_checks(actual,expected,scene):
    from scipy.signal import stft
    sample=scene['acquisition']['sample_step_us'];frames=len(actual)
    carrier=scene['source']['frequency_mhz'];rows=[]
    def score(a,b,count=None):
        count=a.size if count is None else count
        error=float(np.sqrt(np.sum(abs(a-b)**2)/count));scale=float(np.sqrt(np.sum(abs(b)**2)/count))
        return {'error_rms_m':error,'reference_rms_m':scale,'relative_l2':error/scale if scale>1e-13 else None,
                'passed':bool(error<=(.05*scale if scale>1e-13 else 1e-13))}
    for receiver in range(actual.shape[1]):
        a,b=actual[:,receiver],expected[:,receiver]
        if scene.get('signal',{}).get('kind','hann')=='chirp':
            length=min(512,frames);overlap=min(384,length*3//4)
            frequency,_,x=stft(a,fs=1/sample,nperseg=length,noverlap=overlap,axis=0,boundary=None)
            _,_,y=stft(b,fs=1/sample,nperseg=length,noverlap=overlap,axis=0,boundary=None)
            low,high=sorted([carrier,scene['signal']['end_frequency_mhz']]);row={}
            for name,factor in [('drive_band',1),('second_band',2)]:
                mask=(frequency>=factor*.9*low)&(frequency<=factor*(high+.02))
                if not np.any(mask):raise ValueError('Insufficient time span for requested chirp band check')
                row[name]=score(x[mask],y[mask])
                for axis,key in enumerate(['x','y','z']):
                    row[name+'_'+key]=score(x[mask,axis,:],y[mask,axis,:])
            rows.append(row)
        else:
            window=np.hanning(frames);window/=window.sum();times=np.arange(frames)*sample;row={}
            for label,order in [('f',1),('2f',2)]:
                kernel=window*np.exp(-2j*np.pi*order*carrier*times)
                x=2*np.einsum('t,tc->c',kernel,a-np.einsum('t,tc->c',window,a))
                y=2*np.einsum('t,tc->c',kernel,b-np.einsum('t,tc->c',window,b))
                for axis,key in enumerate(['x','y','z']):row[label+'_'+key]=score(x[axis:axis+1],y[axis:axis+1])
                row[label+'_vector']=score(x,y,count=1)
            rows.append(row)
    return rows


def run(args):
    config,receipt,reference=load_case(args.case)
    scene=config['scene']
    if args.other:
        other,other_receipt,candidate=load_case(args.other)
        if scene!=other['scene']:
            raise ValueError('Refinement must keep scene and output times identical')
        mode='COMSOL_reference_refinement'
        candidate_identity={'job_id':other_receipt['job_id']}
    else:
        from numba import set_num_threads
        from graphwave.long_campaign import Driver
        set_num_threads(3)
        with np.load(args.weights,allow_pickle=False) as saved:weights=dict(saved)
        temporal=json.loads(args.temporal.read_text())
        driver=Driver(scene,args.degree,args.h,weights,temporal)
        frames=round(scene['acquisition']['duration_us']/scene['acquisition']['sample_step_us'])+1
        rows=[]
        for start in range(0,frames,256):
            _,probes,_=driver.advance(start,min(frames-1,start+255));rows.append(probes)
        probes=np.concatenate(rows,axis=1)
        candidate={i:probes[i] for i in reference}
        mode='neural_prediction_vs_COMSOL'
        candidate_identity={'weights_sha256':digest(args.weights),'temporal_sha256':digest(args.temporal),
                            'degree':args.degree,'h_mm':args.h,'internal_step_us':driver.dt*1e6}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output.with_suffix('.npz'),predicted=probes,
                            reference=np.stack([reference[k] for k in sorted(reference)]))
    comparisons={key:metrics(parts(candidate)[key],expected) for key,expected in parts(reference).items()}
    spectra={key:frequency_checks(parts(candidate)[key],expected,scene) for key,expected in parts(reference).items()}
    checks=[m for rows in comparisons.values() for row in rows for m in row.values()]
    spectral_checks=[m for rows in spectra.values() for row in rows for m in row.values()]
    result={'schema':'workai.comsol_graphwave_comparison.v1','method':mode,'reference_job_id':receipt['job_id'],
            'comsol_version':receipt['comsol_version'],'source_sha256':receipt['source_sha256'],
            'duration_us':scene['acquisition']['duration_us'],'sample_step_us':scene['acquisition']['sample_step_us'],
            'receiver_count':len(scene['receivers']),'laws':sorted(reference),'threshold':.05,'floor_m':1e-13,
            'candidate':candidate_identity,'all_waveform_checks_passed':all(m['passed'] for m in checks),
            'all_spectral_checks_passed':all(m['passed'] for m in spectral_checks),
            'all_checks_passed':all(m['passed'] for m in checks+spectral_checks),
            'maximum_reported_relative_l2':max((m['relative_l2'] for m in checks+spectral_checks if m['relative_l2'] is not None),default=None),
            'comparisons':comparisons,'spectral_comparisons':spectra,
            'reference_convergence_verified':False,'scientific_accuracy_certified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','spectral_comparisons']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--other',type=Path)
    p.add_argument('--weights',type=Path,default=Path('offline/model/continued-weights.npz'))
    p.add_argument('--temporal',type=Path,default=Path('offline/model/time-coefficients.json'))
    p.add_argument('--degree',type=int,default=4)
    p.add_argument('--h',type=float,default=2.5)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
