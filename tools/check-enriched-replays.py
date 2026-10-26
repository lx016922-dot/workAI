"""Check algebraic replay or nested learned-space convergence of saved records."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest,save
from graphwave.quarter_symmetry import make_quarter_mesh


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lower',type=Path,required=True);p.add_argument('--higher',type=Path,required=True)
    p.add_argument('--kind',choices=['operator','enrichment'],required=True)
    p.add_argument('--lower-model',type=Path);p.add_argument('--higher-model',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    loader=module('admission','check-modal-refinements.py')
    compare=module('metrics','validate-comsol-linear-matrix.py')
    lower,higher=[loader.load_case(folder) for folder in [args.lower,args.higher]]
    a,b=lower[0],higher[0]
    for key in ['case','weights_sha256','step_us','reference_arrays_loaded_before_prediction']:
        if a[key]!=b[key]:raise ValueError('Comparison identity differs: '+key)
    if a['reference_arrays_loaded_before_prediction'] is not False:
        raise ValueError('A saved independent prediction is required')
    for name in a['source_sha256'].keys() & b['source_sha256'].keys():
        if a['source_sha256'][name]!=b['source_sha256'][name]:
            raise ValueError('Shared inference source changed: '+name)
    nesting=None
    if args.kind=='operator':
        for key in ['model_sha256','basis_sha256','candidate_mode_limit','symmetry']:
            if a[key]!=b[key]:raise ValueError('Force replay requires the same candidate: '+key)
    else:
        if not args.lower_model or not args.higher_model:
            p.error('Both compiled models are required for nested-space admission')
        configs=[];bases=[]
        for folder,identity in [(args.lower_model,a),(args.higher_model,b)]:
            config=json.loads((folder/'model.json').read_text())
            if digest(folder/'model.json')!=identity['model_sha256'] or digest(folder/'basis.npz')!=identity['basis_sha256']:
                raise ValueError('Compiled model does not match its saved prediction')
            if identity['candidate_mode_limit']!=config['exported_modes']:
                raise ValueError('Compare both complete trained spaces')
            configs.append(config)
            with np.load(folder/'basis.npz',allow_pickle=False) as z:bases.append(z['fields'])
        for key in ['scene','degree','h_mm','axis_elements','grading_ratios','weights_sha256','quarter_domain']:
            if configs[0][key]!=configs[1][key]:raise ValueError('Enrichment changed the underlying graph: '+key)
        for key in ['input_basis_sha256','input_model_sha256','source_sha256','train_seed','holdout_seed','training_fields','gradient_norm_range']:
            if configs[0]['nonlinear_basis_training'][key]!=configs[1]['nonlinear_basis_training'][key]:
                raise ValueError('Enrichment training identity differs: '+key)
        if not configs[0]['exported_modes']<configs[1]['exported_modes']:
            raise ValueError('A strictly larger trained space is required')
        c=configs[0];mesh=make_quarter_mesh(c['scene'],c['degree'],c['h_mm'],c['grading_ratios'],c['axis_elements'])
        mass=np.sqrt(np.repeat(mesh['mass'],3))
        first,second=[field.reshape(len(field),-1)*mass for field in bases]
        for field in [first,second]:
            np.testing.assert_allclose(field@field.T,np.eye(len(field)),rtol=0,atol=1e-8)
        projection=(first@second.T)@second
        nesting=float(np.linalg.norm(first-projection,axis=1).max())
        if nesting>1e-8:raise ValueError('The larger basis does not contain the smaller trained space')
    scene=a['case']['scene']
    first,second=[compare.comparison.parts(dict(enumerate(record[3]))) for record in [lower,higher]]
    metrics={name:compare.check(first[name],second[name],scene) for name in first}
    limit=.0001 if args.kind=='operator' else .05
    absolute=1e-18 if args.kind=='operator' else 1e-13
    maximum=0.;passed=True
    for group in metrics.values():
        for category in ['waveform','spectral']:
            for receiver in group[category]:
                for value in receiver.values():
                    relative=value['relative_l2']
                    if relative is None:passed&=value['error_rms_m']<=absolute
                    else:maximum=max(maximum,relative);passed&=relative<=limit
    args.output.mkdir(parents=True,exist_ok=False)
    report={'kind':args.kind,'duration_us':scene['acquisition']['duration_us'],
            'lower_identity':a,'higher_identity':b,'maximum_mass_norm_nesting_residual':nesting,
            'relative_limit':limit,'small_signal_absolute_limit_m':absolute,
            'metrics':metrics,'maximum_relative_change':maximum,'passed':bool(passed),
            'native_reference_job_ids':[record[2].get('native_reference_job_id') for record in [lower,higher]],
            'scientific_accuracy_certified':False,'full_5ms_nonlinear_accuracy_verified':False}
    save(args.output/'summary.json',report)
    print(json.dumps({k:report[k] for k in ['kind','duration_us','maximum_mass_norm_nesting_residual','maximum_relative_change','passed']}),flush=True)


if __name__=='__main__':main()
