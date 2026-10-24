"""Compare frozen actual NN traces against COMSOL-sourced linear modal responses.

Keeps this linear modal diagnostic separate from native nonlinear COMSOL
verification and from training. Every source trace chunk is hash checked.
"""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.comsol_modal_response import response
from graphwave.modal_calibration import read_modes,digest

spec=importlib.util.spec_from_file_location('comsol_compare',Path(__file__).with_name('compare-comsol-reference.py'))
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def check(a,b,scene):
    waves=comparison.metrics(a,b);spectra=comparison.frequency_checks(a,b,scene)
    checks=[m for rows in [waves,spectra] for row in rows for m in row.values()]
    return {'waveform':waves,'spectral':spectra,'passed':all(m['passed'] for m in checks),
            'maximum_relative_l2':max((m['relative_l2'] for m in checks if m['relative_l2'] is not None),default=None),
            'maximum_vector_waveform_relative_l2':max((row['vector']['relative_l2'] for row in waves if row['vector']['relative_l2'] is not None),default=None)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--check-reference',type=Path);p.add_argument('--truncation',type=int,default=32)
    p.add_argument('--ci-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receiver-indices',nargs='+',type=int)
    args=p.parse_args();plan=json.loads(args.plan.read_text());reference=read_modes(args.reference)
    if not 0<args.truncation<len(reference['frequency_hz']):raise ValueError('Truncation check needs strictly fewer modes')
    refinement=read_modes(args.check_reference) if args.check_reference else None
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in plan['cases']:
        scene=copy.deepcopy(case['scene']);indices=args.receiver_indices or list(range(len(scene['receivers'])))
        scene['receivers']=[scene['receivers'][i] for i in indices]
        directory=args.ci_root/case['id']/case['id']
        old=json.loads((directory/'metrics.json').read_text());chunks=[]
        for block in old['probe_blocks']:
            path=directory/block['file']
            if digest(path)!=block['sha256']:raise ValueError('Changed baseline neural trace: '+str(path))
            with np.load(path,allow_pickle=False) as f:chunks.append(f['candidate'][0][:,indices,:])
        actual=np.concatenate(chunks)
        times,expected,method=response(reference,scene)
        if actual.shape!=expected.shape or len(times)!=25001 or times[-1]!=5000:
            raise ValueError('Incomplete actual 5 ms NN or modal trace')
        _,truncated,_=response(reference,scene,count=args.truncation)
        _,tighter,_=response(reference,scene,rtol=1e-12)
        row={'id':case['id'],'duration_us':5000,'samples':len(times),'receiver_indices':indices,
             'reference_identity':reference['identity'],'method':method,
             'baseline_weights_sha256':plan['weights_sha256'],
             'baseline_metrics_sha256':digest(directory/'metrics.json'),
             'comparison':check(actual,expected,scene),'modal_truncation':check(truncated,expected,scene),
             'time_integration_refinement':check(expected,tighter,scene),
             'modal_truncation_counts':[args.truncation,len(reference['frequency_hz'])],
             'native_comsol_time_dependent_solve':False,'nonlinear_reference':False,
             'long_record_accuracy_certified':False}
        if refinement:
            _,refined,_=response(refinement,scene)
            row['mesh_refinement']=check(expected,refined,scene)
            row['refinement_identity']=refinement['identity']
        np.savez_compressed(args.output/(case['id']+'.npz'),times_us=times,actual_linear=actual,reference_linear=expected)
        row['traces_sha256']=digest(args.output/(case['id']+'.npz'))
        (args.output/(case['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row);print(json.dumps({'id':case['id'],'vector_error':row['comparison']['maximum_vector_waveform_relative_l2'],
                                         'passed':row['comparison']['passed'],'truncation_passed':row['modal_truncation']['passed']}),flush=True)
    summary={'schema':'workai.comsol_linear_modal_matrix.v1','cases':len(rows),
             'plan_sha256':digest(args.plan),'reference_identity':reference['identity'],
             'comparison_passes':sum(row['comparison']['passed'] for row in rows),
             'truncation_passes':sum(row['modal_truncation']['passed'] for row in rows),
             'time_refinement_passes':sum(row['time_integration_refinement']['passed'] for row in rows),
             'native_comsol_time_dependent_solve':False,'nonlinear_reference':False,
             'long_record_accuracy_certified':False,'results':[{k:row[k] for k in ['id','receiver_indices','comparison','modal_truncation','time_integration_refinement']} for row in rows]}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'cases.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['case','duration_us','receivers','max_vector_waveform_relative_l2','all_checks_passed','modal_truncation_passed','certified'])
        for row in rows:writer.writerow([row['id'],5000,len(row['receiver_indices']),row['comparison']['maximum_vector_waveform_relative_l2'],row['comparison']['passed'],row['modal_truncation']['passed'],False])


if __name__=='__main__':main()
