"""Independent transient checks for neural-only predictions, never a corrector."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import torch
from spectral3d.compare import spectral,compare,complex_coefficient,digest
from spectral3d.field_compare import field,weighted
from spectralop.data import tensors
from .physics_checks import physics_checks
from operator3d.train import atomic,now


def evaluate_case(record,directory,scales):
    base=Path(record['base_manifest']);assert digest(base)==record['base_manifest_sha256']
    teacher=json.loads(base.read_text());item=next(c for c in teacher['cases'] if c['case_id']==record['case_id']);cid=item['case_id']
    report=json.loads((directory/'prediction.json').read_text())
    assert report['neural_forward_only'] and not report['numerical_correction'] and not report['reference_responses_read_for_inference']
    assert digest(directory/'fields.npz')==report['field_sha256'];assert digest(directory/'mesh.npz')==report['mesh_sha256'];assert digest(directory/'probes.npz')==report['probe_sha256']
    with np.load(directory/'mesh.npz',allow_pickle=False) as saved:mesh=dict(saved)
    with np.load(base.parent/item['mesh_file'],allow_pickle=False) as saved:
        assert set(mesh)==set(saved)
        for key in mesh:np.testing.assert_array_equal(mesh[key],saved[key])
    with np.load(directory/'fields.npz',allow_pickle=False) as saved:pred=[saved[k] for k in ['linear','stvk','murnaghan']];times=saved['times_us']
    true=[field(base.parent,item,law) for law in ['linear','stvk','murnaghan']]
    pparts=np.stack([pred[0],pred[1]-pred[0],pred[2]-pred[1]],2);tparts=np.stack([true[0],true[1]-true[0],true[2]-true[1]],2)
    checks={}
    for name,p,t in [('full',pred[2],true[2]),('material',pparts[:,:,2],tparts[:,:,2]),('geometric',pparts[:,:,1],tparts[:,:,1])]:
        checks[name+'_vector']=weighted(p,t,mesh['mass'])
        for axis,label in enumerate('xyz'):checks[name+'_'+label]=weighted(p[:,:,axis:axis+1],t[:,:,axis:axis+1],mesh['mass'])
    pc=complex_coefficient(times,pparts[:,:,2],2*item['scene']['source']['frequency_mhz']);tc=complex_coefficient(times,tparts[:,:,2],2*item['scene']['source']['frequency_mhz'])
    checks['material_2f_vector']=weighted(pc[None],tc[None],mesh['mass'])
    for axis,label in enumerate('xyz'):checks['material_2f_'+label]=weighted(pc[None,:,axis:axis+1],tc[None,:,axis:axis+1],mesh['mass'])
    with np.load(directory/'probes.npz',allow_pickle=False) as saved:controls=saved['controls']
    probes=np.stack([controls[:,:,0],controls[:,:,1]-controls[:,:,0],controls[:,:,2]-controls[:,:,1]],2)
    summary=spectral(base);pred_summary={cid:{'scene':item['scene'],'times':times,'parts':probes,'diagnostics':report['diagnostics']}}
    trace=compare(pred_summary,summary)['cases'][0]
    case=tensors(item['scene'],mesh);case['times']=times
    physical=physics_checks(case,pparts,torch.as_tensor(scales,dtype=torch.float64))
    initial=bool(np.all(pparts[0]==0));fixed=bool(np.all(pparts[:,case['fixed'].numpy()]==0))
    guards=all(d['max_strain']<=.001 and d['min_J']>0 and d['energy_balance_within_1_percent'] for d in report['diagnostics'])
    return {'case_id':cid,'split':record.get('split','test'),'geometry':item['scene']['geometry'],'field_checks':checks,'probes':trace['probes'],'probe_checks_pass':trace['all_probe_checks_pass'],
            'physical_checks':physical,'energy_strain_J_pass':guards,'exact_initial_and_fixed':initial and fixed,'reference_manifest_sha256':digest(base),
            'prediction_sha256':digest(directory/'prediction.json'),'weights_sha256':report['weights_sha256'],'all_gates_passed':all(c['pass'] for c in checks.values()) and trace['all_probe_checks_pass'] and physical['passed'] and guards and initial and fixed}


def run(args):
    torch.set_num_threads(2);data=json.loads(args.development.read_text());rows=[]
    for item in data['cases']:
        if item['split']!='validation' or (args.case and item['case_id']!=args.case):continue
        row=evaluate_case(item,args.predictions/item['case_id'],data['branch_scales_in_100nm']);rows.append(row)
        atomic(args.predictions/(item['case_id']+'-metrics.json'),row)
        print(json.dumps({'case_id':row['case_id'],'full_relative_l2':row['field_checks']['full_vector']['relative_l2'],'material_relative_l2':row['field_checks']['material_vector']['relative_l2'],
            'physics':row['physical_checks']['passed'],'probes':row['probe_checks_pass'],'failed_fields':[k for k,v in row['field_checks'].items() if not v['pass']], 'all_passed':row['all_gates_passed']}),flush=True)
        gc.collect()
    atomic(args.predictions/('development-metrics.json' if not args.case else args.case+'-summary.json'),{'schema':'workai.graphwave_development.v1','cases':rows,'all_gates_passed':all(r['all_gates_passed'] for r in rows),'final_test':False,'threshold':.05,'evaluated_at_utc':now()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--case');run(p.parse_args())
