"""Offline COMSOL-supervised graph-filter calibration and modal diagnostics.

This module is a training tool, never an inference fallback. It assembles the
linearization of the *learned* energy/graph layer for modal training. The physical
energy tensors, nodal mass, fixed constraints, and time layer remain unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment, least_squares
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from spectral3d.mesh import make_mesh, boundary_data
from .inference import inverse_mapping, graph_message
from tools.comsol_artifacts import properties, numeric_csv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def read_modes(folder):
    collection=json.loads((folder/'collection.json').read_text())
    if collection['phase']!='completed':raise ValueError('Unfinished COMSOL labels')
    for item in collection['artifacts']:
        if not item.get('verified_against_signed_manifest') or digest(folder/item['name'])!=item['sha256']:
            raise ValueError('Changed/unverified COMSOL labels')
    identity=collection['identity'];receipt=properties(folder/'completed.properties')
    for key in ['job_id','source_sha256','scene_sha256','parameters_sha256']:
        if identity[key]!=receipt[key]:raise ValueError('COMSOL label identity mismatch')
    if receipt['method']!='comsol_3d_linear_eigenfrequency_weakform':raise ValueError('Wrong label type')
    scene=json.loads((folder/'scene.json').read_text())
    if digest(folder/'scene.json')!=identity['scene_sha256']:raise ValueError('Scene changed')
    modes,points=int(receipt['modes']),int(receipt['sample_points'])
    rows=np.genfromtxt(numeric_csv(folder/(receipt['level']+'.csv')),delimiter=',',names=True)
    if len(rows)!=modes*points:raise ValueError('Incomplete modal export')
    if not np.array_equal(rows['mode'],np.repeat(np.arange(modes),points)) or not np.array_equal(rows['point'],np.tile(np.arange(points),modes)):
        raise ValueError('Modal index order')
    for key in rows.dtype.names:
        if not np.isfinite(rows[key]).all():raise ValueError('Nonfinite mode data')
    coordinates=np.stack([rows[k].reshape(modes,points)[0] for k in ['x_m','y_m','z_m']],-1)
    freq=rows['frequency_hz'].reshape(modes,points)
    if not np.allclose(freq,freq[:,0,None],rtol=0,atol=1e-8):raise ValueError('Inconsistent eigenfrequency')
    fields=np.stack([rows[k].reshape(modes,points) for k in ['u_m','v_m','w_m']],-1)
    integral=np.genfromtxt(numeric_csv(folder/(receipt['level']+'.field.csv')),delimiter=',',names=True)
    energy_frequency=np.sqrt(integral['stiffness_integral']/integral['mass_integral'])/(2*np.pi)
    if not np.allclose(energy_frequency,freq[:,0],rtol=1e-8,atol=1e-5):
        raise ValueError('COMSOL modal energy does not match its frequency')
    return {'scene':scene['scene'],'coordinates':coordinates,'frequency_hz':freq[:,0],
            'fields':fields,'integrals':integral,'receipt':receipt,'identity':identity,'folder':str(folder)}


def stiffness(mesh, derivative, quadratic):
    """Exact tangent of graphwave.inference.graph_message at zero displacement."""
    n=int(mesh['degree'])+1;q=n**3
    gradient=np.zeros((n,n,n,n,n,n,3))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for a in range(n):
                    gradient[i,j,k,a,j,k,0]=derivative[i,a]
                    gradient[i,j,k,i,a,k,1]=derivative[j,a]
                    gradient[i,j,k,i,j,a,2]=derivative[k,a]
    gradient=gradient.reshape(q,q,3);inverse=inverse_mapping(mesh).reshape(-1,q,3,3)
    rows=[];columns=[];entries=[];c=2e11*quadratic
    for e,element in enumerate(mesh['elements']):
        grad=np.einsum('qaA,qAi->qai',gradient,inverse[e])
        strain=np.zeros((q,6,q*3))
        strain[:,0,0::3]=grad[:,:,0];strain[:,1,1::3]=grad[:,:,1];strain[:,2,2::3]=grad[:,:,2]
        strain[:,3,1::3]=grad[:,:,2]/np.sqrt(2);strain[:,3,2::3]=grad[:,:,1]/np.sqrt(2)
        strain[:,4,0::3]=grad[:,:,2]/np.sqrt(2);strain[:,4,2::3]=grad[:,:,0]/np.sqrt(2)
        strain[:,5,0::3]=grad[:,:,1]/np.sqrt(2);strain[:,5,1::3]=grad[:,:,0]/np.sqrt(2)
        block=np.einsum('qra,rs,qsb,q->ab',strain,c,strain,mesh['volume_weight'][e].ravel(),optimize=True)
        dofs=(element.ravel()[:,None]*3+np.arange(3)).ravel()
        rows.append(np.repeat(dofs,len(dofs)));columns.append(np.tile(dofs,len(dofs)));entries.append(block.ravel())
    count=len(mesh['points'])*3
    result=coo_matrix((np.concatenate(entries),(np.concatenate(rows),np.concatenate(columns))),shape=(count,count)).tocsr()
    result.eliminate_zeros()
    return result


def modes(mesh, scene, derivative, quadratic, count):
    matrix=stiffness(mesh,derivative,quadratic)
    _,fixed,_=boundary_data(mesh,scene)
    free=np.ones(len(mesh['points'])*3,dtype=bool);free[(fixed[:,None]*3+np.arange(3)).ravel()]=False
    mass=np.repeat(mesh['mass'],3);scale=1/np.sqrt(mass[free])
    normalized=diags(scale)@matrix[free][:,free]@diags(scale)
    values,vectors=eigsh(normalized,k=count,sigma=0,which='LM',tol=1e-11,v0=np.ones(free.sum()),maxiter=5000)
    order=np.argsort(values);values=values[order];vectors=vectors[:,order]
    if np.any(values<=0):raise ValueError('Nonpositive neural stiffness mode')
    fields=np.zeros((count,len(mass)));fields[:,free]=(vectors*scale[:,None]).T
    return np.sqrt(values)/(2*np.pi),fields.reshape(count,-1,3)


def match(reference, actual, mass):
    a=reference.reshape(len(reference),-1)*np.sqrt(np.repeat(mass,3))[None]
    b=actual.reshape(len(actual),-1)*np.sqrt(np.repeat(mass,3))[None]
    a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
    mac=(a@b.T)**2
    left,right=linear_sum_assignment(-mac)
    if not np.array_equal(left,np.arange(len(reference))):raise ValueError('Incomplete mode matching')
    return right,mac[left,right]


def filter_basis(nodes):
    """Preserve derivatives of constant, affine, and quadratic fields and parity."""
    constraints=np.stack([nodes**k for k in range(3)],1)
    null=null_space(constraints.T);raw=[]
    for row in range(len(nodes)):
        for direction in null.T:
            value=np.zeros((len(nodes),len(nodes)));value[row]=direction
            raw.append((value-value[::-1,::-1])*.5)
    u,s,_=np.linalg.svd(np.stack(raw,-1).reshape(len(nodes)**2,-1),full_matrices=False)
    basis=u[:,s>1e-10].T.reshape(-1,len(nodes),len(nodes))
    if np.max(abs(basis@constraints))>1e-12:raise ValueError('Filter consistency constraint')
    return basis


def audit(args):
    reference=read_modes(args.reference)
    scene=reference['scene'];mesh=make_mesh(scene,4,2.5)
    if not np.allclose(mesh['points'],reference['coordinates'],rtol=0,atol=1e-14):raise ValueError('Reference/node grid mismatch')
    with np.load(args.weights,allow_pickle=False) as f:weights=dict(f)
    derivative=weights['derivative_4'];matrix=stiffness(mesh,derivative,weights['quadratic_tensor'])
    displacement=np.random.default_rng(2026102405).normal(size=mesh['points'].shape)*1e-9
    message,_,_,_=graph_message(displacement,mesh['elements'],derivative,inverse_mapping(mesh),mesh['volume_weight'],weights['quadratic_tensor'],weights['cubic_tensor'],0)
    tangent_error=np.linalg.norm(matrix@displacement.ravel()-message.ravel())/np.linalg.norm(message)
    if tangent_error>1e-12:raise ValueError('Modal training tangent differs from actual neural forward pass')
    frequency,fields=modes(mesh,scene,derivative,weights['quadratic_tensor'],len(reference['frequency_hz']))
    mapping,mac=match(reference['fields'],fields,mesh['mass'])
    actual=frequency[mapping];delta=actual-reference['frequency_hz']
    load,_,_=boundary_data(mesh,scene)
    participation=np.abs(np.einsum('mnc,nc->m',reference['fields'],load))
    participation/=np.sqrt(np.einsum('mnc,mnc,n->m',reference['fields'],reference['fields'],mesh['mass']))
    result={'schema':'workai.comsol_neural_modal_audit.v1','reference_job_id':reference['receipt']['job_id'],
            'reference_scene_sha256':reference['identity']['scene_sha256'],'weights_sha256':digest(args.weights),
            'neural_tangent_forward_relative_error':float(tangent_error),'nodes':len(mesh['points']),
            'filter_trainable_basis_dimension':len(filter_basis(mesh['gll'])),
            'modes':[{'index':i,'comsol_frequency_hz':float(f),'neural_frequency_hz':float(a),
                      'difference_hz':float(d),'phase_drift_degrees_at_5ms':float(360*.005*d),
                      'modal_assurance':float(m),'load_participation':float(p)}
                     for i,(f,a,d,m,p) in enumerate(zip(reference['frequency_hz'],actual,delta,mac,participation))],
            'reference_convergence_verified':False,'long_record_accuracy_certified':False}
    save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='modes'}),flush=True)
    for row in result['modes']:
        if row['load_participation']>participation.max()*1e-4:
            print(json.dumps(row),flush=True)


def train(args):
    reference=read_modes(args.reference);check=read_modes(args.check_reference)
    scene=reference['scene'];mesh=make_mesh(scene,4,2.5)
    # Excitation metadata is irrelevant to a zero-load eigenproblem, but every
    # physical input and every spatial sampling coordinate must agree.
    for key in ['geometry','material','boundary','constitutive']:
        if scene[key]!=check['scene'][key]:raise ValueError('Refinement changes physical problem')
    if not np.allclose(mesh['points'],reference['coordinates'],rtol=0,atol=1e-14) or not np.allclose(reference['coordinates'],check['coordinates'],rtol=0,atol=1e-14):
        raise ValueError('Modal sampling grid mismatch')
    correspondence,ref_mac=match(reference['fields'],check['fields'],mesh['mass'])
    target=check['frequency_hz'][correspondence]
    uncertainty=abs(reference['frequency_hz']-target)
    load,_,_=boundary_data(mesh,scene)
    participation=abs(np.einsum('mnc,nc->m',reference['fields'],load))
    participation/=np.sqrt(np.einsum('mnc,mnc,n->m',reference['fields'],reference['fields'],mesh['mass']))
    # Admission is per mode, with its actual refinement uncertainty retained.
    # 0.01% frequency convergence permits research training; it is explicitly
    # NOT sufficient by itself to certify a phase-sensitive 5 ms waveform.
    selected=(target<=900000)&(participation>=.01*participation.max())&(ref_mac>=.99)&(uncertainty/target<=1e-4)
    indices=np.flatnonzero(selected)
    if len(indices)<2:raise ValueError('Insufficient converged COMSOL development modes')
    with np.load(args.weights,allow_pickle=False) as f:weights=dict(f)
    baseline=weights['derivative_4'];basis=filter_basis(mesh['gll'])
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    physical_weight=np.sqrt((participation/participation.max())*(target.min()/target)**2)
    physical_weight/=physical_weight[indices].max()
    config={'schema':'workai.comsol_modal_training.v1','reference_job_id':reference['receipt']['job_id'],
            'refinement_job_id':check['receipt']['job_id'],'reference_identity':reference['identity'],
            'refinement_identity':check['identity'],'baseline_weights_sha256':digest(args.weights),
            'training_mode_indices':indices.tolist(),'mode_refinement_uncertainty_hz':uncertainty.tolist(),
            'mode_refinement_mac':ref_mac.tolist(),'frequency_admission_relative':1e-4,
            'long_phase_reference_verified':bool(np.max(uncertainty[indices])<=.5),
            'training_parameters':len(basis),'parameter_bounds':[-.25,.25],
            'preserved':['constant/affine/quadratic polynomial derivatives','centroskew filter parity',
                         'all material energy tensors','all other polynomial degree filters','nodal mass','time weights'],
            'training_labels':'Actual signed COMSOL eigenfrequencies; no transient waveform is loaded by optimization',
            'phase_sensitivity_horizon_s':.005,'frequency_band_upper_hz':900000,
            'inference_reference_lookup':False,'inference_numerical_corrector':False,
            'scope':'Research candidate for the bounded fixed-end aluminum plate. Requires independent long nonlinear waveforms before promotion.',
            'accepted_for_offline_release':False,'long_record_accuracy_certified':False,
            'training_source_sha256':digest(Path(__file__))}
    save(output/'config.json',config)
    history=[];start=time.monotonic();count=len(reference['frequency_hz'])

    def evaluate(theta):
        derivative=baseline+np.einsum('a,aij->ij',theta,basis)
        frequency,fields=modes(mesh,scene,derivative,weights['quadratic_tensor'],count)
        mapping,mac=match(reference['fields'],fields,mesh['mass'])
        return frequency[mapping],mac,derivative

    initial,initial_mac,_=evaluate(np.zeros(len(basis)))
    def objective(theta):
        frequency,mac,_=evaluate(theta)
        if np.min(mac[indices])<.9:raise ValueError('Training mode correspondence lost')
        residual=(frequency[indices]-target[indices])*2*np.pi*.005*physical_weight[indices]
        row={'evaluation':len(history)+1,'elapsed_s':time.monotonic()-start,'parameters':theta.tolist(),
             'weighted_phase_loss':float(np.sum(residual**2)),
             'training_frequency_errors_hz':(frequency[indices]-target[indices]).tolist(),
             'minimum_training_mac':float(mac[indices].min())}
        history.append(row)
        if len(history)%20==0:print(json.dumps(row),flush=True)
        return np.r_[residual,.01*theta]
    fitted=least_squares(objective,np.zeros(len(basis)),bounds=(-.25,.25),jac='3-point',
                         x_scale='jac',max_nfev=100,ftol=1e-12,xtol=1e-12,gtol=1e-10)
    final,final_mac,derivative=evaluate(fitted.x)
    exported=dict(weights);exported['derivative_4']=derivative
    for key in weights:
        if key!='derivative_4' and not np.array_equal(exported[key],weights[key]):raise AssertionError('Unexpected parameter change')
    np.savez(output/'candidate-weights.npz',**exported)
    save(output/'history.json',history)
    save(output/'training.json',{'optimizer':'scipy_least_squares_three_point','optimizer_success':bool(fitted.success),
        'optimizer_message':str(fitted.message),'optimizer_evaluations':int(fitted.nfev),'total_forward_evaluations':len(history),
        'parameters':fitted.x.tolist(),'weights_sha256':digest(output/'candidate-weights.npz'),
        'initial_frequency_errors_hz':(initial-target).tolist(),'final_frequency_errors_hz':(final-target).tolist(),
        'training_mode_indices':indices.tolist(),'initial_training_error_norm_hz':float(np.linalg.norm((initial-target)[indices])),
        'final_training_error_norm_hz':float(np.linalg.norm((final-target)[indices])),
        'initial_mode_mac':initial_mac.tolist(),'final_mode_mac':final_mac.tolist(),
        'material_parameters_changed':False,'test_waveforms_loaded':False,
        'long_phase_reference_verified':config['long_phase_reference_verified'],
        'accepted_for_offline_release':False,'long_record_accuracy_certified':False,'elapsed_s':time.monotonic()-start})
    print(json.dumps({'candidate':str(output/'candidate-weights.npz'),'training_modes':indices.tolist(),
                      'initial_error_norm_hz':float(np.linalg.norm((initial-target)[indices])),
                      'final_error_norm_hz':float(np.linalg.norm((final-target)[indices])),
                      'long_record_accuracy_certified':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path('offline/model/continued-weights.npz'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--action',choices=['audit','train'],default='audit')
    p.add_argument('--check-reference',type=Path)
    args=p.parse_args()
    if args.action=='train':
        if not args.check_reference:p.error('--check-reference is required for training')
        train(args)
    else:audit(args)
