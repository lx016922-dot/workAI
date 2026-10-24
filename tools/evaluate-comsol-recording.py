"""Evaluate a completed, hash-verified full-field NN record against COMSOL modes."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import read_modes,digest
from tools.comsol_modal_response import response


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--receiver-indices',nargs='+',type=int);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=json.loads((args.record/'recording.json').read_text())
    if manifest['phase']!='completed' or manifest['completed_frames']!=manifest['frame_count']:
        raise ValueError('Full actual record is required')
    if digest(args.input)!=manifest['identity']['input_sha256']:raise ValueError('Input changed')
    scene=json.loads(args.input.read_text())['scene'];rows=[];energy=[];next_frame=0
    for block in manifest['chunks']:
        path=args.record/block['file']
        if digest(path)!=block['sha256'] or path.stat().st_size!=block['bytes'] or block['first_frame']!=next_frame:
            raise ValueError('Changed or discontinuous prediction data')
        with np.load(path,allow_pickle=False) as file:
            rows.append(file['probes'].transpose(2,0,1,3));energy.append(file['energies'])
        next_frame+=block['frame_count']
    if next_frame!=manifest['frame_count']:raise ValueError('Record incomplete')
    actual=np.concatenate(rows,axis=1);energies=np.concatenate(energy,axis=1)
    total=energies[:,:,1]+energies[:,:,2];work=energies[:,:,3]
    balance=np.max(abs(total-work),axis=1)/np.maximum(np.max(np.maximum(abs(total),abs(work)),axis=1),1e-30)
    indices=args.receiver_indices or list(range(len(scene['receivers'])))
    actual=actual[:,:,indices];scene['receivers']=[scene['receivers'][i] for i in indices]
    reference=read_modes(args.reference);times,expected,method=response(reference,scene)
    spec=importlib.util.spec_from_file_location('matrix',Path(__file__).with_name('validate-comsol-linear-matrix.py'))
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    result={'schema':'workai.comsol_recording_linear_validation.v1','recording_sha256':digest(args.record/'recording.json'),
            'prediction_identity':manifest['identity'],'reference_identity':reference['identity'],
            'duration_us':manifest['duration_us'],'samples':manifest['frame_count'],'receiver_indices':indices,
            'actual_neural_time_steps':True,'reference_method':method,
            'linear_comparison':compare.check(actual[0],expected,scene),'energy_balance_relative':balance.tolist(),
            'energy_within_1_percent':bool(np.all(balance<=.01)),'max_strain':manifest['max_strain'],'min_J':manifest['min_J'],
            'native_comsol_time_dependent_reference':False,'nonlinear_comsol_comparison_completed':False,
            'reference_spatial_convergence_verified':False,'accepted_for_offline_release':False,'long_record_accuracy_certified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'),times_us=times,predicted_controls=actual,comsol_linear_modal_reference=expected)
    result['traces_sha256']=digest(args.output.with_suffix('.npz'))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='linear_comparison'}),flush=True)
    print(json.dumps({'maximum_linear_vector_relative_error':result['linear_comparison']['maximum_vector_waveform_relative_l2'],
                      'all_linear_checks_passed':result['linear_comparison']['passed']}),flush=True)


if __name__=='__main__':main()
