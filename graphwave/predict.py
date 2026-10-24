"""Direct neural recurrence from public geometry/excitation plus frozen weights."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np
from spectral3d.mesh import make_mesh,boundary_data,interpolation
from .inference import inverse_mapping,source_values,recurrent_forward


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(scene,mesh,weights,cfl=.3,callback=None):
    if not .05<=cfl<=.3:raise ValueError('Forward layer spacing outside validated bounds.')
    if int(mesh['degree']) not in [3,4]:raise ValueError('The frozen network supports graph degrees 3 and 4.')
    # This is a conservative bound for the fixed aluminum model, not an
    # iterative time-step adjustment or residual-based numerical correction.
    sample=scene['acquisition']['sample_step_us']*1e-6
    subdivisions=int(np.ceil(sample/(cfl*float(mesh['min_node_spacing'])/10000.)))
    dt=sample/subdivisions;frames=round(scene['acquisition']['duration_us']/scene['acquisition']['sample_step_us'])+1
    steps=(frames-1)*subdivisions
    if steps>150000 or frames*len(mesh['mass'])*3*8>512*1024**2:raise ValueError('Bounded neural forward budget exceeded.')
    load,fixed,area=boundary_data(mesh,scene);inverse=inverse_mapping(mesh)
    pulse=source_values(scene['source'],np.arange(steps+1)*dt*1e6);fields=[];diagnostics=[]
    for law in range(3):
        started=time.monotonic()
        field,energy,strain,jac=recurrent_forward(mesh['elements'],weights['derivative_'+str(int(mesh['degree']))],inverse,mesh['volume_weight'],mesh['mass'],load,fixed,
            weights['quadratic_tensor'],weights['cubic_tensor'],weights['state_weights'],law,dt,subdivisions,frames,pulse)
        scale=max(np.max(abs(energy[:,3])),np.max(abs(energy[:,1]+energy[:,2])),1e-30)
        balance=float(np.max(abs(energy[:,1]+energy[:,2]-energy[:,3]))/scale)
        fields.append(field);diagnostics.append({'law':['linear','stvk','murnaghan'][law],'elapsed_s':time.monotonic()-started,'recurrent_layers':steps,'layer_step_us':dt*1e6,
            'max_strain':float(strain),'min_J':float(jac),'external_work_balance_relative':balance,'energy_balance_within_1_percent':balance<=.01,'energy_samples':energy.tolist(),
            'neural_forward_only':True,'numerical_correction':False,'reference_responses_read':False})
        if callback:callback(diagnostics[-1])
    return fields,diagnostics


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);request=json.loads(args.input.read_text());scene=request['scene']
    expected=request['weights_sha256'];assert digest(args.weights)==expected
    with np.load(args.weights,allow_pickle=False) as saved:weights=dict(saved)
    if set(weights)!={'quadratic_tensor','cubic_tensor','state_weights','derivative_3','derivative_4'} or not all(np.isfinite(v).all() for v in weights.values()):raise ValueError('Invalid network weights.')
    mesh=make_mesh(scene,request['degree'],request['h_mm'])
    fields,diagnostics=predict(scene,mesh,weights,cfl=request.get('cfl',.3),callback=lambda row:print(json.dumps({k:v for k,v in row.items() if k!='energy_samples'}),flush=True))
    np.savez_compressed(args.output/'mesh.npz',**mesh)
    np.savez_compressed(args.output/'fields.npz',linear=fields[0],stvk=fields[1],murnaghan=fields[2],times_us=np.arange(len(fields[0]))*scene['acquisition']['sample_step_us'])
    probes=[]
    for receiver in scene['receivers']:
        ids,w=interpolation(mesh,receiver['point_mm']);probes.append(np.stack([np.einsum('n,tnc->tc',w,field[:,ids]) for field in fields],1))
    np.savez_compressed(args.output/'probes.npz',controls=np.stack(probes,1))
    forbidden=['spectral3d.solver','spectral3d.curved_solver','spectral3d.implicit','solid.fem','energyai.predict']
    if any(name in sys.modules for name in forbidden):raise RuntimeError('A numerical solver module entered the neural inference process.')
    report={'schema':'workai.graphwave_prediction.v1','weights_sha256':expected,'input_sha256':digest(args.input),'field_sha256':digest(args.output/'fields.npz'),
        'mesh_sha256':digest(args.output/'mesh.npz'),'probe_sha256':digest(args.output/'probes.npz'),'diagnostics':diagnostics,'neural_forward_only':True,
        'reference_responses_read_for_inference':False,'numerical_correction':False,'spatially_discretized_recurrent_network':True,'solver_modules_absent':forbidden,'accepted_for_solver_routing':False}
    (args.output/'prediction.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
