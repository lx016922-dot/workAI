"""Predeclared 20-case, 5 ms validation; no fitting to evaluation responses.

Full-field errors are accumulated over every sample, volume weighted. Raw
receiver histories and per-block checksums remain inspectable; memory is bounded.
Numerical reference code is used only in this research command.
"""
import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from numba import set_num_threads
from scipy.signal import stft
from spectral3d.mesh import make_mesh, boundary_data, interpolation
from spectral3d.long_reference import three_segments as reference_segment, signal_samples, COMPOSITION
from .inference import inverse_mapping
from .long_inference import three_segments as network_segment
from .long_rollout import atomic, digest, save_arrays
from .signals import pulse_values
from .continued_validation import transfer, map_field

PARTS = ('linear','stvk','full','geometric','material')
STATE = ('positions','velocities','accelerations','externals','works','potentials','strains','jacobians')


def parts(x): return (x[0],x[1],x[2],x[1]-x[0],x[2]-x[1])


def score(error2, reference2, count):
    error, rms = np.sqrt(error2/count), np.sqrt(reference2/count)
    return {'error_rms_m':float(error),'reference_rms_m':float(rms),
            'relative_l2':float(error/rms) if rms>1e-13 else None,
            'passed':bool(error <= (.05*rms if rms>1e-13 else 1e-13)),
            'criterion':'relative_l2_5_percent' if rms>1e-13 else 'absolute_rms_1e-13_m'}


class Driver:
    def __init__(self, scene, degree, h, weights, temporal, reference=False):
        self.scene=scene; self.mesh=make_mesh(scene,degree,h); self.reference=reference
        self.load,self.fixed,_=boundary_data(self.mesh,scene)
        self.inverse=inverse_mapping(self.mesh)
        self.weights=weights; self.composition=COMPOSITION if reference else np.array(temporal['weights'])
        sample=scene['acquisition']['sample_step_us']*1e-6
        self.subdivisions=math.ceil(sample/(.3*self.mesh['min_node_spacing']/10000))
        self.dt=sample/self.subdivisions
        shape=(3,len(self.load),3)
        self.state={k:np.zeros(shape) for k in STATE[:4]}
        self.state.update(works=np.zeros(3),potentials=np.zeros(3),strains=np.zeros(3),jacobians=np.ones(3))
        self.receivers=[interpolation(self.mesh,r['point_mm']) for r in scene['receivers']]
        self.constants=np.array([scene['material'][k] for k in ('lambda_pa','mu_pa','l_pa','m_pa','n_pa')])

    def advance(self, first, last):
        start=max(0,first-1)*self.subdivisions; end=last*self.subdivisions
        t=(np.arange(start,end)[:,None]+np.cumsum(self.composition)[None])*self.dt*1e6
        if self.reference:
            field,energy,guards=reference_segment(*(self.state[k] for k in STATE[:5]),
                self.mesh['elements'],self.mesh['derivative'],self.inverse,self.mesh['volume_weight'],
                self.mesh['mass'],self.constants,self.load,self.fixed,self.dt,self.subdivisions,start,end,first==0,
                signal_samples(self.scene,t))
            self.state['strains']=np.maximum(self.state['strains'],guards[:,0])
            self.state['jacobians']=np.minimum(self.state['jacobians'],guards[:,1])
        else:
            field,energy=network_segment(*(self.state[k] for k in STATE),self.mesh['elements'],
                self.weights['derivative_'+str(int(self.mesh['degree']))],self.inverse,self.mesh['volume_weight'],
                self.mesh['mass'],self.load,self.fixed,self.weights['quadratic_tensor'],self.weights['cubic_tensor'],
                self.composition,self.dt,self.subdivisions,start,end,first==0,pulse_values(self.scene,t))
        probes=np.stack([np.stack([np.einsum('n,tnc->tc',w,f[:,ids]) for ids,w in self.receivers],1) for f in field])
        return field,probes,energy


def plan(args):
    source=Path(__file__).resolve().parents[1]
    files=['graphwave/long_campaign.py','graphwave/long_inference.py','graphwave/buffered_inference.py','graphwave/signals.py',
           'graphwave/inference.py','graphwave/continued_validation.py','spectral3d/long_reference.py',
           'spectral3d/long_reference_force.py','spectral3d/mesh.py']
    if (args.output/'plan.json').exists():
        p=json.loads((args.output/'plan.json').read_text())
        assert all(digest(source/f)==v for f,v in p['source_sha256'].items()), 'Frozen evaluation source changed'
        assert digest(args.weights)==p['weights_sha256'] and digest(args.temporal)==p['temporal_sha256']
        return p
    scene=json.loads(args.input.read_text())['scene']
    scene['acquisition']={'duration_us':5000.,'sample_step_us':.2}
    scene['receivers']=[{'name':f'P{i+1}','point_mm':point,'direction':[1.,0.,0.]} for i,point in enumerate(
        [[1.1,.7,.4],[2.,1.5,.75],[3.1,2.3,1.1],[4.2,.9,1.2],[5.3,2.1,.5],[6.,1.5,.75],[6.9,.6,.9],[7.6,2.4,.3]])]
    cases=[]
    for kind in ('hann','gaussian','rectangular','chirp'):
        for cycles in (1,3,5,10,20):
            s=copy.deepcopy(scene); s['source'].update(cycles=cycles,frequency_mhz=.2)
            s['signal']={'kind':kind,'end_frequency_mhz':.28}
            cases.append({'id':f'{kind}-{cycles:02d}','scene':s,'candidate_degree':4,'reference_degree':5,'h_mm':2.5})
    p={'created_utc':datetime.now(timezone.utc).isoformat(),'weights_sha256':digest(args.weights),
       'temporal_sha256':digest(args.temporal),'source_sha256':{f:digest(source/f) for f in files},'cases':cases,
       'policy':{'relative_l2_limit':.05,'absolute_floor_m':1e-13,'all_5_responses_and_8_receivers':True,
                 'full_field_every_sample':True,'duration_us':5000.,'trained_on_these_transients':False,
                 'reference':'analytic Murnaghan/StVK/linear, degree 5, independent smaller time step',
                 'scope':'one fixed-end 8 x 3 x 1.5 mm aluminum plate; no universal geometric certification',
                 'reference_convergence_requires_separate_checks':True}}
    atomic(args.output/'plan.json',p)
    for f in files:
        target=args.output/'source_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((source/f).read_bytes())
    return p


def run_case(args,case,weights,temporal):
    directory=args.output/case['id'];directory.mkdir(exist_ok=True)
    metric=directory/'metrics.json'
    if metric.exists(): return json.loads(metric.read_text())
    scene=case['scene'];frames=25001;sample=.2;start_time=time.monotonic()
    candidate=Driver(scene,case['candidate_degree'],case['h_mm'],weights,temporal)
    reference=Driver(scene,case['reference_degree'],case['h_mm'],weights,temporal,True)
    # Evaluate the candidate interpolant on the independent reference grid.
    matrix,_=transfer(candidate.mesh,reference.mesh,directory/'transfer.npz')
    mass=reference.mesh['mass']/reference.mesh['mass'].sum()
    sums=np.zeros((5,2,3)); scale=np.zeros((2,3)); balance=np.zeros((2,3)); first=0
    statefile=directory/'state.npz'
    if statefile.exists():
        with np.load(statefile,allow_pickle=False) as data:
            first=int(data['next_frame']);sums=data['sums'];scale=data['scale'];balance=data['balance']
            for prefix,driver in [('candidate',candidate),('reference',reference)]:
                driver.state={k:data[prefix+'_'+k] for k in STATE}
    for at in range(first,frames,256):
        last=min(frames-1,at+255)
        actual,ap,ae=candidate.advance(at,last)
        expected,rp,re=reference.advance(at,last)
        for i,(a,b) in enumerate(zip(parts(actual),parts(expected))):
            a=map_field(matrix,a)
            sums[i,0]+=np.einsum('tnc,n->c',(a-b)**2,mass)
            sums[i,1]+=np.einsum('tnc,n->c',b*b,mass)
        for j,energy in enumerate((ae,re)):
            total=energy[:,:,1]+energy[:,:,2]
            scale[j]=np.maximum(scale[j],np.maximum(np.max(abs(total),axis=1),np.max(abs(energy[:,:,3]),axis=1)))
            balance[j]=np.maximum(balance[j],np.max(abs(total-energy[:,:,3]),axis=1))
        block=directory/f'probes-{at:06d}.npz'
        save_arrays(block,candidate=ap,reference=rp)
        save_arrays(statefile,next_frame=np.array(last+1),sums=sums,scale=scale,balance=balance,
                    **{prefix+'_'+k:v for prefix,driver in [('candidate',candidate),('reference',reference)] for k,v in driver.state.items()})
        atomic(directory/'progress.json',{'completed_time_us':last*sample,'duration_us':5000.,'elapsed_s':time.monotonic()-start_time})
        print(json.dumps({'case':case['id'],'time_us':last*sample,'elapsed_s':round(time.monotonic()-start_time,2)}),flush=True)
    ap=[];rp=[];receipts=[]
    for block in sorted(directory.glob('probes-*.npz')):
        with np.load(block) as data: ap.append(data['candidate']);rp.append(data['reference'])
        receipts.append({'file':block.name,'sha256':digest(block)})
    actual=np.concatenate(ap,axis=1);expected=np.concatenate(rp,axis=1)
    fields={part:{axis:score(sums[i,0,j],sums[i,1,j],frames) for j,axis in enumerate('xyz')} for i,part in enumerate(PARTS)}
    for i,part in enumerate(PARTS): fields[part]['vector']=score(sums[i,0].sum(),sums[i,1].sum(),frames)
    probes={}
    spectral={}
    for part,a,b in zip(PARTS,parts(actual),parts(expected)):
        probes[part]=[]
        spectral[part]=[]
        for j in range(a.shape[1]):
            probes[part].append({axis:score(np.sum((a[:,j,k]-b[:,j,k])**2),np.sum(b[:,j,k]**2),frames) for k,axis in enumerate('xyz')})
            if scene['signal']['kind']=='chirp':
                f,t,za=stft(a[:,j],fs=1/sample,nperseg=512,noverlap=384,axis=0,boundary=None)
                _,_,zb=stft(b[:,j],fs=1/sample,nperseg=512,noverlap=384,axis=0,boundary=None)
                bands={}
                for name,lo,hi in [('drive_band',.18,.3),('second_band',.36,.6)]:
                    chosen=(f>=lo)&(f<=hi);x,y=za[chosen],zb[chosen]
                    bands[name]=score(np.sum(abs(x-y)**2),np.sum(abs(y)**2),x.size)
                spectral[part].append(bands)
            else:
                window=np.hanning(frames);window/=window.sum();times=np.arange(frames)*sample
                coeff=[]
                for order in (1,2):
                    def value(x):
                        return 2*np.einsum('t,tc->c',window*np.exp(-2j*np.pi*order*.2*times),x-np.einsum('t,tc->c',window,x))
                    x,y=value(a[:,j]),value(b[:,j]);coeff.append(score(np.sum(abs(x-y)**2),np.sum(abs(y)**2),1))
                spectral[part].append({'f':coeff[0],'2f':coeff[1]})
    checks=[v for row in fields.values() for v in row.values()]+[v for rows in probes.values() for row in rows for v in row.values()]+[v for rows in spectral.values() for row in rows for v in row.values()]
    result={'id':case['id'],'duration_us':5000.,'frame_count':frames,'field':fields,'receivers':probes,'spectral':spectral,
            'all_comparisons_under_5_percent':all(x['passed'] for x in checks),
            'maximum_reported_relative_l2':max(x['relative_l2'] or 0 for x in checks),
            'energy_balance_relative':(balance/np.maximum(scale,1e-30)).tolist(),
            'energy_within_1_percent':bool(np.all(balance<=.01*np.maximum(scale,1e-30))),
            'reference_spatial_convergence_verified':False,'accuracy_certified':False,
            'candidate_dt_us':candidate.dt*1e6,'reference_dt_us':reference.dt*1e6,
            'candidate_max_strain':candidate.state['strains'].tolist(),'candidate_min_J':candidate.state['jacobians'].tolist(),
            'probe_blocks':receipts}
    atomic(metric,result)
    return result


def main(args):
    args.output.mkdir(parents=True,exist_ok=True);set_num_threads(3)
    p=plan(args)
    with np.load(args.weights) as data: weights=dict(data)
    temporal=json.loads(args.temporal.read_text())
    results=[]
    for case in p['cases']:
        try: results.append(run_case(args,case,weights,temporal))
        except Exception as error:
            atomic(args.output/case['id']/'failure.json',{'error':str(error),'time_utc':datetime.now(timezone.utc).isoformat()})
            raise
        atomic(args.output/'summary.json',{'completed_cases':len(results),'required_cases':20,'results':results,
               'all_comparisons_under_5_percent':len(results)==20 and all(r['all_comparisons_under_5_percent'] for r in results),
               'accuracy_certified':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('input','weights','temporal','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
