"""Verify downloaded CI evidence and write a bounded, explicit accuracy report."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from scipy.signal import stft


def verify_score(saved, error, reference, count):
    error_rms=float(np.sqrt(error/count)); reference_rms=float(np.sqrt(reference/count))
    relative=error_rms/reference_rms if reference_rms>1e-13 else None
    np.testing.assert_allclose([saved['error_rms_m'],saved['reference_rms_m']],
                               [error_rms,reference_rms],rtol=2e-9,atol=1e-24)
    assert saved['passed']==(error_rms <= (.05*reference_rms if relative is not None else 1e-13))
    assert saved['criterion']==('relative_l2_5_percent' if relative is not None else 'absolute_rms_1e-13_m')
    if relative is None: assert saved['relative_l2'] is None
    else: np.testing.assert_allclose(saved['relative_l2'],relative,rtol=2e-9,atol=1e-14)


def response_parts(x):
    return {'linear':x[0],'stvk':x[1],'full':x[2],
            'geometric':x[1]-x[0],'material':x[2]-x[1]}

parser=argparse.ArgumentParser()
parser.add_argument('--downloads',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[1]
plan_path=repo/'research/long-validation/plan.json';plan=json.loads(plan_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
seen=set()
for receipt in json.loads(args.downloads.read_text()):
    archive=Path(receipt['path']);assert sha(archive)==receipt['sha256']
    case_id=receipt['name'].removeprefix('long-')
    assert case_id not in seen;seen.add(case_id)
    directory=args.downloads.parent/case_id
    if not (directory/'summary.json').exists():
        directory.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            assert sum(f.file_size for f in z.infolist())<256*1024**2
            assert all((directory/f.filename).resolve().is_relative_to(directory.resolve()) for f in z.infolist())
            z.extractall(directory)
    assert sha(directory/'plan.json')==sha(plan_path)
    case=next(c for c in plan['cases'] if c['id']==case_id)
    metrics_path=directory/case_id/'metrics.json';m=json.loads(metrics_path.read_text())
    assert m['id']==case_id and m['duration_us']==5000 and m['frame_count']==25001
    frames=0;candidate=[];reference=[]
    for block in m['probe_blocks']:
        p=directory/case_id/block['file'];assert sha(p)==block['sha256']
        assert p.name==f'probes-{frames:06d}.npz'
        with np.load(p,allow_pickle=False) as data:
            a,b=data['candidate'],data['reference']
            assert a.shape==b.shape==(3,min(256,25001-frames),8,3)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            candidate.append(a);reference.append(b)
            frames+=a.shape[1]
    assert frames==25001
    with np.load(directory/case_id/'state.npz',allow_pickle=False) as state:
        assert int(state['next_frame'])==25001
        sums=state['sums']
        for i,part in enumerate(('linear','stvk','full','geometric','material')):
            verify_score(m['field'][part]['vector'],sums[i,0].sum(),sums[i,1].sum(),25001)
            for j,axis in enumerate('xyz'):
                verify_score(m['field'][part][axis],sums[i,0,j],sums[i,1,j],25001)
        np.testing.assert_allclose(m['energy_balance_relative'],state['balance']/np.maximum(state['scale'],1e-30),rtol=1e-13)
        assert m['energy_within_1_percent']==bool(np.all(state['balance']<=.01*np.maximum(state['scale'],1e-30)))
        np.testing.assert_array_equal(m['candidate_max_strain'],state['candidate_strains'])
        np.testing.assert_array_equal(m['candidate_min_J'],state['candidate_jacobians'])
    candidate=response_parts(np.concatenate(candidate,axis=1));reference=response_parts(np.concatenate(reference,axis=1))
    for part,a in candidate.items():
        b=reference[part]
        for point in range(8):
            x,y=a[:,point],b[:,point]
            for j,axis in enumerate('xyz'):
                verify_score(m['receivers'][part][point][axis],np.square(x[:,j]-y[:,j]).sum(),np.square(y[:,j]).sum(),25001)
            if case['scene']['signal']['kind']=='chirp':
                freq,_,u=stft(x,fs=5,nperseg=512,noverlap=384,axis=0,boundary=None)
                _,_,v=stft(y,fs=5,nperseg=512,noverlap=384,axis=0,boundary=None)
                for name,lo,hi in [('drive_band',.18,.30),('second_band',.36,.60)]:
                    use=(freq>=lo)&(freq<=hi)
                    verify_score(m['spectral'][part][point][name],np.square(abs(u[use]-v[use])).sum(),np.square(abs(v[use])).sum(),u[use].size)
            else:
                w=np.hanning(25001);w/=w.sum()
                x=x-np.sum(w[:,None]*x,axis=0);y=y-np.sum(w[:,None]*y,axis=0)
                for order,name in [(1,'f'),(2,'2f')]:
                    times=np.arange(25001)*.2
                    kernel=(2*w*np.exp(-2j*np.pi*order*.2*times))[:,None]
                    u=np.sum(kernel*x,axis=0);v=np.sum(kernel*y,axis=0)
                    verify_score(m['spectral'][part][point][name],np.square(abs(u-v)).sum(),np.square(abs(v)).sum(),1)
    checks=[s for field in m['field'].values() for s in field.values()]
    checks += [s for group in [m['receivers'],m['spectral']] for response in group.values() for point in response for s in point.values()]
    assert m['all_comparisons_under_5_percent']==all(s['passed'] for s in checks)
    assert m['maximum_reported_relative_l2']==max(s['relative_l2'] or 0 for s in checks)
    value=lambda name:100*m['field'][name]['vector']['relative_l2'] if m['field'][name]['vector']['relative_l2'] is not None else None
    rows.append({'case_id':case_id,'signal':case['scene']['signal']['kind'],'cycles':case['scene']['source']['cycles'],
                 'duration_us':5000,'samples':25001,'full_relative_l2_percent':value('full'),
                 'material_relative_l2_percent':value('material'),'geometric_relative_l2_percent':value('geometric'),
                 'maximum_reported_relative_l2_percent':100*m['maximum_reported_relative_l2'],
                 'all_comparisons_under_5_percent':m['all_comparisons_under_5_percent'],
                 'energy_balance_within_1_percent':m['energy_within_1_percent'],
                 'reference_spatial_convergence_verified':m['reference_spatial_convergence_verified'],
                 'accuracy_certified':m['accuracy_certified'],'artifact_id':receipt['id'],
                 'artifact_sha256':receipt['sha256'],'metrics_sha256':sha(metrics_path)})
order={c['id']:i for i,c in enumerate(plan['cases'])};rows.sort(key=lambda r:order[r['case_id']])
summary={'schema':'workai.long_validation_summary.v1','source_commit':'511d06a23a6a0108c4d8753df4aaf3e97ff85d70',
         'workflow_run_url':'https://github.com/lx016922-dot/workAI/actions/runs/40058525287',
         'plan_sha256':sha(plan_path),'required_cases':20,'completed_cases':len(rows),
         'comparison_pass_count':sum(r['all_comparisons_under_5_percent'] for r in rows),
         'energy_pass_count':sum(r['energy_balance_within_1_percent'] for r in rows),
         'all_comparisons_passed':len(rows)==20 and all(r['all_comparisons_under_5_percent'] for r in rows),
         'scientific_accuracy_certified':False,'scope':plan['policy']['scope'],'rows':rows}
summary['raw_receiver_and_spectral_metrics_recomputed']=True
summary['field_metrics_checked_against_saved_full_time_accumulators']=True
(args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
if rows:
    with (args.output/'cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
for row in rows:print(json.dumps({k:row[k] for k in ['case_id','full_relative_l2_percent','material_relative_l2_percent','geometric_relative_l2_percent','all_comparisons_under_5_percent']}))
