"""Offline COMSOL-supervised graph-filter calibration and modal diagnostics.

This module is a training tool, never an inference fallback. It assembles the
linearization of the *learned* energy/graph layer for modal training. The physical
energy tensors, nodal mass, fixed constraints, and time layer remain unchanged.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment, least_squares, minimize
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
    grid_points=int(receipt.get('grid_sample_points',points))
    receiver_points=int(receipt.get('receiver_sample_points',0))
    if grid_points+receiver_points!=points:raise ValueError('Modal sampling count mismatch')
    if receiver_points and (receiver_points!=len(scene['scene']['receivers']) or not np.allclose(
            coordinates[grid_points:],np.array([r['point_mm'] for r in scene['scene']['receivers']])*1e-3,rtol=0,atol=1e-14)):
        raise ValueError('COMSOL receiver modal coordinates mismatch')
    return {'scene':scene['scene'],'coordinates':coordinates[:grid_points],'frequency_hz':freq[:,0],
            'fields':fields[:,:grid_points],'receiver_fields':fields[:,grid_points:],
            'integrals':integral,'receipt':receipt,'identity':identity,'folder':str(folder)}


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


def project_filter_consistency(derivative,nodes,consistency_order=1):
    """Project the learned starting filter, not only its trainable increments.

    An approximately affine-consistent parent can otherwise create spurious
    strain under finite rigid rotation even when every increment is consistent.
    """
    if consistency_order not in [1,2]:raise ValueError('Consistency order must be one or two')
    coordinates=np.stack([nodes**k for k in range(consistency_order+1)],1)
    exact=np.stack([np.zeros_like(nodes)]+[k*nodes**(k-1) for k in range(1,consistency_order+1)],1)
    result=np.array(derivative,copy=True)
    result+=np.linalg.solve(coordinates.T@coordinates,(exact-result@coordinates).T).T@coordinates.T
    result=(result-result[::-1,::-1])*.5
    if np.max(abs(result@coordinates-exact))>2e-14:raise ValueError('Polynomial consistency projection failed')
    return result


def filter_basis(nodes, consistency_order=2):
    """Preserve polynomial consistency and parity, always including rigid motions."""
    if consistency_order not in [1,2]:raise ValueError('Consistency order must be one or two')
    constraints=np.stack([nodes**k for k in range(consistency_order+1)],1)
    null=null_space(constraints.T);raw=[]
    for row in range(len(nodes)):
        for direction in null.T:
            value=np.zeros((len(nodes),len(nodes)));value[row]=direction
            raw.append((value-value[::-1,::-1])*.5)
    u,s,_=np.linalg.svd(np.stack(raw,-1).reshape(len(nodes)**2,-1),full_matrices=False)
    basis=u[:,s>1e-10].T.reshape(-1,len(nodes),len(nodes))
    if np.max(abs(basis@constraints))>1e-12:raise ValueError('Filter consistency constraint')
    return basis


def restore_physical_feasibility(theta, baseline, basis, weights, dataset, limit):
    """Offline backtracking of learned parameters using training physics only.

    Numerical optimization may end just outside an active inequality. Preserve
    its status, but never export that point as satisfying the hard constraint.
    Re-evaluate every trial rather than assume the residual varies monotonically.
    """
    from . import nonlinear_constraints as physics
    target=limit*(1-1e-3);trials=[]
    for attempt in range(1002):
        scale=.999**attempt if attempt<=1000 else 0.
        trial=np.asarray(theta)*scale
        derivative=baseline+np.einsum('a,aij->ij',trial,basis)
        _,report=physics.evaluate(derivative,weights,dataset,details=True)
        maximum=max(report['maximum_normalized_residual'].values())
        trials.append({'scale':scale,'maximum_training_residual':maximum})
        if maximum<=target:
            return trial,{'method':'offline parameter backtracking toward the physically verified baseline',
                          'training_limit':limit,'target_with_margin':target,'trials':trials,
                          'validation_used_for_selection':False,'inference_correction':False}
    raise ValueError('No physically feasible baseline; no candidate may be released')


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
    from . import nonlinear_constraints as physics
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
    baseline=project_filter_consistency(weights['derivative_4'],mesh['gll'],args.consistency_order)
    basis=filter_basis(mesh['gll'],args.consistency_order)
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    (output/'source_snapshot').mkdir()
    shutil.copyfile(Path(__file__),output/'source_snapshot/modal_calibration.py')
    shutil.copyfile(Path(physics.__file__),output/'source_snapshot/nonlinear_constraints.py')
    physical_training=physics.make_dataset(scene['material'],args.physics_samples,args.physics_seed)
    physical_validation=physics.make_dataset(scene['material'],args.physics_validation_samples,args.physics_seed+1)
    initial_restoration=None
    if args.optimizer=='slsqp':
        # A previously valid candidate may violate a newly expanded training
        # set. Restore a feasible starting point using training fields only;
        # post-fit backtracking must not head toward an infeasible parent.
        _,starting_report=physics.evaluate(baseline,weights,physical_training,details=True)
        if max(starting_report['maximum_normalized_residual'].values())>args.physics_limit*(1-1e-3):
            exact=project_filter_consistency(mesh['derivative'],mesh['gll'],args.consistency_order)
            coefficients=np.einsum('aij,ij->a',basis,baseline-exact)
            if np.linalg.norm(exact+np.einsum('a,aij->ij',coefficients,basis)-baseline)>1e-12:
                raise ValueError('Parent filter cannot be represented in the declared consistent basis')
            coefficients,initial_restoration=restore_physical_feasibility(coefficients,exact,basis,weights,physical_training,args.physics_limit)
            initial_restoration['parent_maximum_training_residual']=max(starting_report['maximum_normalized_residual'].values())
            initial_restoration['scope']='Warm-start feasibility on the declared training set; independent validation is not consulted'
            baseline=exact+np.einsum('a,aij->ij',coefficients,basis)
    rigid_parent=physics.rigid_motion_checks(weights['derivative_4'],weights)
    rigid_initial=physics.rigid_motion_checks(baseline,weights)
    if not rigid_initial['passed']:raise ValueError('Projected starting filter fails rigid-motion objectivity')
    save(output/'rigid-parent.json',rigid_parent);save(output/'rigid-initial.json',rigid_initial)
    for split,dataset in [('training',physical_training),('validation',physical_validation)]:
        numeric={key:np.stack([item[key] for item in dataset]) for key in dataset[0] if isinstance(dataset[0][key],np.ndarray)}
        np.savez_compressed(output/('physics-'+split+'-patches.npz'),**numeric)
    physical_weight=np.sqrt((participation/participation.max())*(target.min()/target)**2)
    physical_weight/=physical_weight[indices].max()
    dominant_index=int(indices[np.argmax(participation[indices])])
    physical_weight[dominant_index]*=args.dominant_phase_weight
    config={'schema':'workai.comsol_modal_training.v1','reference_job_id':reference['receipt']['job_id'],
            'refinement_job_id':check['receipt']['job_id'],'reference_identity':reference['identity'],
            'refinement_identity':check['identity'],'baseline_weights_sha256':digest(args.weights),
            'training_mode_indices':indices.tolist(),'mode_refinement_uncertainty_hz':uncertainty.tolist(),
            'mode_refinement_mac':ref_mac.tolist(),'frequency_admission_relative':1e-4,
            'long_phase_reference_verified':bool(np.max(uncertainty[indices])<=.5),
            'training_parameters':len(basis),'parameter_bounds':[-args.parameter_bound,args.parameter_bound],
            'dominant_phase_mode_index':dominant_index,
            'dominant_phase_residual_multiplier':args.dominant_phase_weight,
            'phase_residual_weights':physical_weight.tolist(),
            'dominant_mode_selection':'largest load participation among admitted COMSOL development modes; no test trajectory selection',
            'initial_physics_feasibility_restoration':initial_restoration,
            'polynomial_consistency_order':args.consistency_order,
            'initial_filter_consistency_projection':{
                'frobenius_correction':float(np.linalg.norm(baseline-weights['derivative_4'])),
                'parent_constant_error':float(np.max(abs(weights['derivative_4']@np.ones(5)))),
                'parent_affine_error':float(np.max(abs(weights['derivative_4']@mesh['gll']-1))),
                'projected_constant_error':float(np.max(abs(baseline@np.ones(5)))),
                'projected_affine_error':float(np.max(abs(baseline@mesh['gll']-1))),
                'scope':'offline weight projection before training; no inference correction'},
            'preserved':['polynomial derivatives through the declared consistency order','centroskew filter parity',
                         'all material energy tensors','all other polynomial degree filters','nodal mass','time weights'],
            'training_labels':'Actual signed COMSOL eigenfrequencies plus independent analytic nonlinear momentum, traction and energy constraints; no transient test waveform is loaded',
            'physics_constraints':{'equation':'rho*u_tt = Div(F*S) + b; full Murnaghan Green strain; all mixed derivatives retained',
                'branches':list(physics.BRANCHES),'projections':list(physics.DIRECTIONS),
                'families':list(physics.FAMILIES),'training_samples':args.physics_samples,
                'validation_samples':args.physics_validation_samples,'seed':args.physics_seed,
                'loss_weight':args.physics_weight,'maximum_normalized_component_residual':.05,
                'optimizer':args.optimizer,'hard_training_limit':args.physics_limit if args.optimizer=='slsqp' else None,
                'constraint_form':'1 - (per-sample component RMS / hard_training_limit)^2 >= 0',
                'finite_difference_step':args.gradient_step,'objective_scaling':'fixed initial objective; never candidate-dependent',
                'slsqp_tolerance':args.optimizer_tolerance,'maximum_optimizer_iterations':args.max_iterations,
                'postfit_feasibility':'re-evaluate training constraints after parameter backtracking; 0.1% interior margin; preserve raw optimizer status',
                'physical_equations_source_sha256':digest(Path(physics.__file__)),
                'training_patches_sha256':digest(output/'physics-training-patches.npz'),
                'validation_patches_sha256':digest(output/'physics-validation-patches.npz'),
                'hard_constraints':['zero initial state','zero fixed displacement and velocity','constant/affine filter consistency','conservative transpose-tied force','unchanged objective material energy'],
                'training_only':True,'validation_patches_used_in_objective':False},
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
    parent_frequency,parent_fields=modes(mesh,scene,weights['derivative_4'],weights['quadratic_tensor'],count)
    parent_mapping,_=match(reference['fields'],parent_fields,mesh['mass'])
    parent_frequency=parent_frequency[parent_mapping]
    _,initial_physical=physics.evaluate(baseline,weights,physical_training,details=True)
    save(output/'physics-initial.json',initial_physical)
    def objective(theta):
        frequency,mac,derivative=evaluate(theta)
        if np.min(mac[indices])<.9:raise ValueError('Training mode correspondence lost')
        residual=(frequency[indices]-target[indices])*2*np.pi*.005*physical_weight[indices]
        physical_residual=physics.evaluate(derivative,weights,physical_training)
        row={'evaluation':len(history)+1,'elapsed_s':time.monotonic()-start,'parameters':theta.tolist(),
             'weighted_phase_loss':float(np.sum(residual**2)),
             'weighted_physics_loss':float(args.physics_weight**2*np.sum(physical_residual**2)),
             'physics_normalized_residual_rms':float(np.sqrt(np.sum(physical_residual**2)/len(physics.CONSTRAINT_GROUPS))),
             'training_frequency_errors_hz':(frequency[indices]-target[indices]).tolist(),
             'minimum_training_mac':float(mac[indices].min())}
        history.append(row)
        if len(history)%20==0:print(json.dumps(row),flush=True)
        return np.r_[residual,args.physics_weight*physical_residual,.01*theta]
    if args.optimizer=='least_squares':
        fitted=least_squares(objective,np.zeros(len(basis)),bounds=(-args.parameter_bound,args.parameter_bound),jac='3-point',
                             x_scale='jac',max_nfev=100,ftol=1e-12,xtol=1e-12,gtol=1e-10)
    else:
        # Enforce every sample/branch/projection gate, so an average physics
        # penalty cannot hide a locally poor boundary or transverse equation.
        cache={}
        def constrained(theta):
            key=np.asarray(theta,dtype=np.float64).tobytes()
            if key not in cache:
                frequency,mac,derivative=evaluate(theta)
                if np.min(mac[indices])<.9:raise ValueError('Training mode correspondence lost')
                residual=(frequency[indices]-target[indices])*2*np.pi*.005*physical_weight[indices]
                r,report=physics.evaluate(derivative,weights,physical_training,details=True)
                values=np.concatenate([np.asarray(row[group]).ravel() for row in report['samples'] for group in physics.CONSTRAINT_GROUPS])
                value=float(residual@residual+args.physics_weight**2*(r@r)+.0001*(theta@theta))
                # Dimensionless squared inequalities avoid the cusp of the
                # RMS near zero and put every physical gate on the same scale.
                cache[key]=(value,1-(values/args.physics_limit)**2)
                if len(cache)>96:cache.pop(next(iter(cache)))
                row={'evaluation':len(history)+1,'elapsed_s':time.monotonic()-start,'parameters':theta.tolist(),
                     'weighted_phase_loss':float(residual@residual),'weighted_physics_loss':float(args.physics_weight**2*(r@r)),
                     'maximum_physics_training_residual':float(values.max()),
                     'training_frequency_errors_hz':(frequency[indices]-target[indices]).tolist(),
                     'minimum_training_mac':float(mac[indices].min())}
                history.append(row)
                if len(history)%20==0:print(json.dumps(row),flush=True)
            return cache[key]
        def jacobian(theta,which):
            # A declared absolute stencil avoids cancellation in the small
            # material-force increment when differentiating the training loss.
            step=args.gradient_step;columns=[]
            for j in range(len(theta)):
                direction=np.zeros_like(theta);direction[j]=step
                columns.append((constrained(theta+direction)[which]-constrained(theta-direction)[which])/(2*step))
            return np.stack(columns,axis=-1)
        objective_scale=max(1.,constrained(np.zeros(len(basis)))[0])
        fitted=minimize(lambda theta:constrained(theta)[0]/objective_scale,np.zeros(len(basis)),
                        jac=lambda theta:jacobian(theta,0)/objective_scale,method='SLSQP',bounds=[(-args.parameter_bound,args.parameter_bound)]*len(basis),
                        constraints=[{'type':'ineq','fun':lambda theta:constrained(theta)[1],
                                      'jac':lambda theta:jacobian(theta,1)}],
                        options={'maxiter':args.max_iterations,'ftol':args.optimizer_tolerance})
    optimized_theta=fitted.x.copy();restoration=None
    if args.optimizer=='slsqp':
        fitted.x,restoration=restore_physical_feasibility(fitted.x,baseline,basis,weights,physical_training,args.physics_limit)
    final,final_mac,derivative=evaluate(fitted.x)
    _,physical_train_report=physics.evaluate(derivative,weights,physical_training,details=True)
    _,physical_validation_report=physics.evaluate(derivative,weights,physical_validation,details=True)
    rigid_final=physics.rigid_motion_checks(derivative,weights)
    save(output/'rigid-final.json',rigid_final)
    save(output/'physics-training.json',physical_train_report)
    save(output/'physics-validation.json',physical_validation_report)
    exported=dict(weights);exported['derivative_4']=derivative
    for key in weights:
        if key!='derivative_4' and not np.array_equal(exported[key],weights[key]):raise AssertionError('Unexpected parameter change')
    np.savez(output/'candidate-weights.npz',**exported)
    save(output/'history.json',history)
    save(output/'training.json',{'optimizer':'scipy_'+args.optimizer,'optimizer_success':bool(fitted.success),
        'optimizer_message':str(fitted.message),'optimizer_evaluations':int(fitted.nfev),'total_forward_evaluations':len(history),
        'parameters':fitted.x.tolist(),'weights_sha256':digest(output/'candidate-weights.npz'),
        'raw_optimizer_parameters':optimized_theta.tolist(),'feasibility_restoration':restoration,
        'initial_frequency_errors_hz':(initial-target).tolist(),'final_frequency_errors_hz':(final-target).tolist(),
        'unmodified_parent_frequency_errors_hz':(parent_frequency-target).tolist(),
        'initial_physics_feasibility_restoration':initial_restoration,
        'training_mode_indices':indices.tolist(),'initial_training_error_norm_hz':float(np.linalg.norm((initial-target)[indices])),
        'final_training_error_norm_hz':float(np.linalg.norm((final-target)[indices])),
        'initial_mode_mac':initial_mac.tolist(),'final_mode_mac':final_mac.tolist(),
        'material_parameters_changed':False,'test_waveforms_loaded':False,
        'nonlinear_longitudinal_transverse_physics_in_loss':True,
        'physics_training_passed':physical_train_report['all_physical_checks_within_5_percent'],
        'hard_training_limit':args.physics_limit if args.optimizer=='slsqp' else None,
        'hard_training_limit_satisfied':all(v<=args.physics_limit for v in physical_train_report['maximum_normalized_residual'].values()) if args.optimizer=='slsqp' else None,
        'physics_training_maxima':physical_train_report['maximum_normalized_residual'],
        'physics_validation_passed':physical_validation_report['all_physical_checks_within_5_percent'],
        'physics_validation_maxima':physical_validation_report['maximum_normalized_residual'],
        'finite_rigid_motion_checks_passed':rigid_final['passed'],'rigid_motion_maxima':rigid_final['maxima'],
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
    p.add_argument('--consistency-order',type=int,choices=[1,2],default=2)
    p.add_argument('--physics-samples',type=int,default=48)
    p.add_argument('--physics-validation-samples',type=int,default=24)
    p.add_argument('--physics-seed',type=int,default=2026102413)
    p.add_argument('--physics-weight',type=float,default=30.)
    p.add_argument('--optimizer',choices=['least_squares','slsqp'],default='slsqp')
    p.add_argument('--physics-limit',type=float,default=.04)
    p.add_argument('--gradient-step',type=float,default=1e-5)
    p.add_argument('--dominant-phase-weight',type=float,default=1.,help='Residual multiplier for the largest admitted load participation mode')
    p.add_argument('--parameter-bound',type=float,default=.25,help='Bound on each affine-consistent filter increment; physical gates remain unchanged')
    p.add_argument('--optimizer-tolerance',type=float,default=1e-10,help='SLSQP stopping tolerance; exported parameters still require exact physical rechecks')
    p.add_argument('--max-iterations',type=int,default=150)
    args=p.parse_args()
    if args.action=='train':
        if not args.check_reference:p.error('--check-reference is required for training')
        if args.physics_samples<6 or args.physics_validation_samples<6 or not np.isfinite(args.physics_weight) or args.physics_weight<=0:
            p.error('Nonlinear physics constraints require at least six patches per split and positive finite loss weight')
        if not 0<args.physics_limit<=.05:p.error('Hard training limit must be positive and no looser than the 5% physical validation gate')
        if not np.isfinite(args.gradient_step) or not 0<args.gradient_step<=1e-3:p.error('Gradient stencil must be finite and in (0, 1e-3]')
        if not np.isfinite(args.dominant_phase_weight) or not 1<=args.dominant_phase_weight<=100:p.error('Dominant phase weight must be finite and in [1, 100]')
        if not np.isfinite(args.parameter_bound) or not 0<args.parameter_bound<=1:p.error('Parameter bound must be finite and in (0, 1]')
        if not np.isfinite(args.optimizer_tolerance) or not 1e-12<=args.optimizer_tolerance<=1e-5:p.error('Optimizer tolerance must be finite and in [1e-12, 1e-5]')
        if not 1<=args.max_iterations<=500:p.error('Optimizer iterations must be in [1, 500]')
        train(args)
    else:audit(args)
