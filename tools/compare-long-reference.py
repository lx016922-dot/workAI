"""Compare every preserved p5/p6 reference sample; this never fits model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=Path, required=True, help='Completed rectangular-01 CI case directory')
parser.add_argument('--higher', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((args.higher / 'progress.json').read_text())['phase'] == 'completed'
protocol=json.loads((args.higher / 'protocol.json').read_text())
baseline=json.loads((args.baseline / 'metrics.json').read_text())
assert protocol['case_id']==baseline['id']=='rectangular-01' and protocol['degree']==6
assert protocol['source_plan_sha256']==sha(args.baseline.parent/'plan.json')
assert baseline['frame_count']==25001 and baseline['duration_us']==5000
expected={row['file']:row['sha256'] for row in baseline['probe_blocks']}
left = [];right = [];receipts = [];scale=np.zeros(3);balance=np.zeros(3)
for path in sorted(args.higher.glob('probes-*.npz')):
    other = args.baseline / path.name
    assert sha(other)==expected[path.name]
    with np.load(path, allow_pickle=False) as high, np.load(other, allow_pickle=False) as low:
        left.append(low['reference']);right.append(high['reference'])
        energy=high['energy'];total=energy[:,:,1]+energy[:,:,2]
        scale=np.maximum(scale,np.maximum(np.max(abs(total),axis=1),np.max(abs(energy[:,:,3]),axis=1)))
        balance=np.maximum(balance,np.max(abs(total-energy[:,:,3]),axis=1))
    receipts.append({'file':path.name,'baseline_sha256':sha(other),'higher_sha256':sha(path)})
a = np.concatenate(left,axis=1);b = np.concatenate(right,axis=1)
assert a.shape == b.shape == (3,25001,8,3)
parts = lambda v:dict(linear=v[0],stvk=v[1],full=v[2],geometric=v[1]-v[0],material=v[2]-v[1])
rows = {}
for part, x in parts(a).items():
    y = parts(b)[part];rows[part] = []
    for point in range(8):
        row = {}
        for component in range(4):
            u,v = (x[:,point],y[:,point]) if component == 3 else (x[:,point,component],y[:,point,component])
            error = float(np.sqrt(np.sum((u-v)**2)/25001))
            rms = float(np.sqrt(np.sum(v*v)/25001))
            row['xyz'[component] if component < 3 else 'vector'] = {
                'error_rms_m':error,'reference_rms_m':rms,
                'relative_l2':error/rms if rms>1e-13 else None,
                'passed':error <= (.05*rms if rms>1e-13 else 1e-13)}
        rows[part].append(row)
checks = [s for group in rows.values() for point in group for s in point.values()]
progress=json.loads((args.higher/'progress.json').read_text())
with np.load(args.higher/'state.npz',allow_pickle=False) as state:
    assert int(state['next_frame'])==25001
    strains=state['strains'].tolist();jacobians=state['jacobians'].tolist()
energy_relative=(balance/np.maximum(scale,1e-30)).tolist()
guards=max(energy_relative)<=.01 and max(strains)<=.001 and min(jacobians)>0
report = {'case_id':'rectangular-01','duration_us':5000,'full_samples':25001,
    'baseline_degree':5,'higher_degree':6,'h_mm':2.5,'threshold':.05,'floor_m':1e-13,
    'checks':'all five response waveforms, eight receivers, XYZ and vector; not a full-field convergence certificate',
    'probe_convergence_passed':all(s['passed'] for s in checks) and guards,
    'higher_reference_guards_passed':guards,'higher_energy_balance_relative':energy_relative,
    'higher_max_strain':strains,'higher_min_J':jacobians,
    'baseline_dt_us':baseline['reference_dt_us'],'higher_dt_us':progress['dt_us'],
    'maximum_reported_relative_l2':max(s['relative_l2'] or 0 for s in checks),
    'maximum_full_vector_relative_l2':max(p['vector']['relative_l2'] for p in rows['full']),
    'scientific_accuracy_certified':False,'receivers':rows,'probe_blocks':receipts}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['receivers','probe_blocks']}))
