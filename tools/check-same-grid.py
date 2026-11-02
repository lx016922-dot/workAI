"""Compare a completed CLI graph record to an analytic, identical-grid control."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from numba import get_num_threads, set_num_threads
from graphwave.long_campaign import Driver, STATE, parts, score
from graphwave.long_rollout import atomic, digest, save_arrays

parser=argparse.ArgumentParser()
parser.add_argument('--record',type=Path,required=True)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
manifest=json.loads((args.record/'recording.json').read_text())
request=json.loads(args.input.read_text());scene=request['scene']
assert manifest['phase']=='completed' and manifest['frame_count']==25001 and manifest['duration_us']==5000
assert manifest['identity']['input_sha256']==digest(args.input), 'Use the exact input file supplied to graphwave.long_rollout'
protocol={'source_recording_sha256':digest(args.record/'recording.json'),
          'input_sha256':digest(args.input),'reference':'analytic material; exact derivative; identical mesh and time step',
          'weights_not_fitted':True,'threshold':.05,'floor_m':1e-13}
args.output.mkdir(parents=True,exist_ok=True)
if (args.output/'protocol.json').exists():
    assert json.loads((args.output/'protocol.json').read_text())==protocol
else:atomic(args.output/'protocol.json',protocol)
set_num_threads(min(3,get_num_threads()))
driver=Driver(scene,request['degree'],request.get('h_mm',scene['numerics']['h_mm']),None,None,reference=True)
assert digest(args.record/'mesh.npz')==manifest['mesh_sha256']
with np.load(args.record/'mesh.npz',allow_pickle=False) as data:
    np.testing.assert_array_equal(data['points'],driver.mesh['points'])
assert abs(driver.dt*1e6-manifest['layer_step_us'])<1e-15
statefile=args.output/'state.npz';first=0;sums=np.zeros((5,2,3))
if statefile.exists():
    with np.load(statefile,allow_pickle=False) as data:
        first=int(data['next_frame']);sums=data['sums'];driver.state={k:data[k] for k in STATE}
mass=driver.mesh['mass']/driver.mesh['mass'].sum()
for row in manifest['chunks']:
    at=row['first_frame']
    if at<first:continue
    path=args.record/row['file'];assert digest(path)==row['sha256']
    last=at+row['frame_count']-1
    expected,probes,energy=driver.advance(at,last)
    with np.load(path,allow_pickle=False) as data:
        actual=np.stack([data[k] for k in ['linear','stvk','murnaghan']])
    for i,(a,b) in enumerate(zip(parts(actual),parts(expected))):
        sums[i,0]+=np.einsum('tnc,n->c',(a-b)**2,mass)
        sums[i,1]+=np.einsum('tnc,n->c',b*b,mass)
    save_arrays(args.output/f'probes-{at:06d}.npz',reference=probes,energy=energy)
    save_arrays(statefile,next_frame=np.array(last+1),sums=sums,**driver.state)
    print(json.dumps({'time_us':last*.2}),flush=True)
fields={k:score(sums[i,0].sum(),sums[i,1].sum(),25001) for i,k in enumerate(['linear','stvk','full','geometric','material'])}
result={'duration_us':5000,'full_samples':25001,'same_grid':True,'same_time_step':True,
        'fields':fields,'all_passed':all(x['passed'] for x in fields.values()),'scientific_accuracy_certified':False}
atomic(args.output/'metrics.json',result)
print(json.dumps(result))
