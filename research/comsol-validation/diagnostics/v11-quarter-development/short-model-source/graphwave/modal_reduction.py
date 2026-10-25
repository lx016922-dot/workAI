"""Research modal coordinates of the trained conservative graph operator.

The basis is computed from the candidate's own graph tangent. No COMSOL data is
read here. Nonlinear forces retain every learned energy-tensor entry and every
GLL quadrature point. Modal truncation and time convergence must be checked
independently before this research architecture can support an accuracy claim.
"""
import numpy as np
from numba import njit

from spectral3d.mesh import boundary_data, interpolation
from .buffered_inference import _activation
from .modal_calibration import modes
from .modal_sensitivity import modal_gradients


def axial_reflection_subspace(mesh,scene,weights,fields):
    """Select the invariant reflection class for uniform axial plate loading.

    Transverse displacement is retained with its physical odd parity. This
    optional reduction is rejected for asymmetric loads/materials/meshes or
    mixed eigenvectors; it never selects modes by a small response amplitude.
    """
    source=scene['source']
    direction=np.asarray(source['direction'],dtype=float)
    if (scene['geometry']['kind']!='plate' or source['radius_mm']!=0 or
            source['face'] not in ['xmin','xmax'] or scene['boundary']['fixed_face'] not in ['xmin','xmax'] or
            not np.array_equal(direction[1:],[0.,0.]) or direction[0]==0):
        raise ValueError('Axial reflection reduction requires uniform axial loading of a fixed plate')
    quadratic,cubic=weights['quadratic_tensor'],weights['cubic_tensor']
    material_errors=[]
    for signs in [np.array([1,1,1,-1,1,-1]),np.array([1,1,1,-1,-1,1])]:
        material_errors.extend([np.linalg.norm(quadratic-quadratic*signs[:,None]*signs[None,:])/np.linalg.norm(quadratic),
            np.linalg.norm(cubic-cubic*signs[:,None,None]*signs[None,:,None]*signs[None,None,:])/np.linalg.norm(cubic)])
    if max(material_errors)>1e-12:raise ValueError('Learned material tensors do not preserve axial reflection symmetry')
    points=mesh['points'];axes=[np.unique(points[:,i]) for i in range(3)]
    if np.prod([len(axis) for axis in axes])!=len(points):raise ValueError('Complete tensor grid required for reflection reduction')
    index=np.arange(len(points)).reshape(*[len(axis) for axis in axes])
    reflected=[index[:,::-1,:].ravel(),index[:,:,::-1].ravel()]
    projection=fields.copy()
    for axis,mapping in zip([1,2],reflected):
        length=points[:,axis].max()
        expected=points.copy();expected[:,axis]=length-points[:,axis]
        if not np.allclose(points[mapping],expected,rtol=0,atol=1e-14) or not np.allclose(mesh['mass'][mapping],mesh['mass'],rtol=1e-12,atol=0):
            raise ValueError('Mesh and mass do not preserve reflection symmetry')
        signs=np.ones(3);signs[axis]=-1
        projection=.5*(projection+projection[:,mapping]*signs)
    mass=mesh['mass']
    norm=np.einsum('mnc,mnc,n->m',fields,fields,mass)
    fractions=np.einsum('mnc,mnc,n->m',projection,projection,mass)/norm
    selected=fractions>1-1e-12;excluded=fractions<1e-12
    if not np.all(selected|excluded):raise ValueError('Mixed reflection classes need a degenerate-eigenspace decomposition')
    if not np.any(selected):raise ValueError('No axial reflection modes found')
    load,_,_=boundary_data(mesh,scene)
    coupling=np.einsum('mnc,nc->m',fields,load)
    leakage=float(np.linalg.norm(coupling[excluded])/np.linalg.norm(coupling))
    if leakage>1e-8:raise ValueError('Excluded symmetry classes have nonzero applied loading')
    return np.flatnonzero(selected),{'method':'two physical reflection symmetries, including transverse component signs',
        'original_modes':len(fields),'retained_modes':int(selected.sum()),'selected_indices':np.flatnonzero(selected).tolist(),
        'projected_modal_mass_fraction':fractions.tolist(),'maximum_material_symmetry_residual':float(max(material_errors)),
        'relative_excluded_load':leakage,'response_amplitude_selection':False,'requires_full_basis_nonlinear_check':True}


def compile_basis(mesh, scene, weights, count):
    derivative = weights['derivative_'+str(mesh['degree'])]
    frequency, fields = modes(mesh, scene, derivative, weights['quadratic_tensor'], count)
    return attach_basis(mesh, scene, weights, frequency, fields)


def attach_basis(mesh, scene, weights, frequency, fields):
    frequency, fields = np.asarray(frequency), np.asarray(fields)
    if fields.shape != (len(frequency), len(mesh['points']), 3):
        raise ValueError('Modal basis shape mismatch')
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0) or not np.all(np.isfinite(fields)):
        raise ValueError('Finite positive frequencies and finite fields required')
    load, fixed, _ = boundary_data(mesh, scene)
    if np.max(abs(fields[:, fixed])) > 1e-12:
        raise ValueError('Modal basis violates the essential boundary')
    if 'reflection_planes_m' in mesh:
        from .quarter_symmetry import free_dofs
        if np.max(abs(fields.reshape(len(fields),-1)[:,~free_dofs(mesh,scene)]))>1e-12:
            raise ValueError('Modal basis violates a symmetry-plane displacement constraint')
    gram = np.einsum('mnc,knc,n->mk', fields, fields, mesh['mass'], optimize=True)
    if np.max(abs(gram-np.eye(len(frequency)))) > 1e-8:
        raise ValueError('Modal basis is not mass orthonormal')
    from .modal_calibration import physical_sampling
    receiver=physical_sampling(mesh,np.array([row['point_mm'] for row in scene['receivers']])*1e-3)(fields)
    gradient = modal_gradients(mesh, weights['derivative_'+str(mesh['degree'])], fields)
    return {'frequency_hz': frequency, 'omega': 2*np.pi*frequency, 'fields': fields,
            'gradient': np.ascontiguousarray(gradient.reshape(len(fields), -1)),
            'volume': np.ascontiguousarray(mesh['volume_weight'].ravel()),
            'load': np.einsum('mnc,nc->m', fields, load),
            'receiver_fields': receiver,
            'quadratic': weights['quadratic_tensor'], 'cubic': weights['cubic_tensor'],
            'mass_orthonormality_error': float(np.max(abs(gram-np.eye(len(frequency)))))}


@njit(cache=True)
def _nonlinear_stresses(gradients, volume, quadratic, cubic, laws):
    batch, points = gradients.shape[:2]
    residual = np.empty_like(gradients)
    energies = np.zeros(batch)
    maximum_strain = np.zeros(batch)
    minimum_j = np.ones(batch)
    strain, second, stress, linear_stress = [np.empty((3, 3)) for _ in range(4)]
    z, linear, nonlinear = [np.empty(6) for _ in range(3)]
    for row in range(batch):
        for point in range(points):
            g = gradients[row, point]
            linear_energy, _, _ = _activation(g, quadratic, cubic, 0, strain, z, linear, nonlinear, second, linear_stress)
            energy, norm, jac = _activation(g, quadratic, cubic, laws[row], strain, z, linear, nonlinear, second, stress)
            residual[row, point] = (stress-linear_stress)*volume[point]
            energies[row] += (energy-linear_energy)*volume[point]
            maximum_strain[row] = max(maximum_strain[row], norm)
            minimum_j[row] = min(minimum_j[row], jac)
    return residual, energies, maximum_strain, minimum_j


def nonlinear_force_pointwise(model, coordinates, laws):
    coordinates, laws = np.atleast_2d(coordinates), np.asarray(laws, dtype=np.int64)
    if coordinates.shape != (len(laws), len(model['omega'])) or np.any((laws < 0) | (laws > 2)):
        raise ValueError('Matching modal states and constitutive branches required')
    gradients = (coordinates @ model['gradient']).reshape(len(laws), -1, 3, 3)
    stress, energy, strain, jac = _nonlinear_stresses(gradients, model['volume'],
                                                    model['quadratic'], model['cubic'], laws)
    force = stress.reshape(len(laws), -1) @ model['gradient'].T
    if not np.isfinite(force).all() or not np.isfinite(energy).all() or np.any(strain > .001) or np.any(jac <= 0):
        raise ValueError('Modal graph violates strain/J/finite-energy guard')
    return force, energy, strain, jac


def _tensor(mandel):
    result=np.empty(mandel.shape[:-1]+(3,3))
    result[...,0,0],result[...,1,1],result[...,2,2]=[mandel[...,i] for i in range(3)]
    result[...,1,2]=result[...,2,1]=mandel[...,3]/np.sqrt(2)
    result[...,0,2]=result[...,2,0]=mandel[...,4]/np.sqrt(2)
    result[...,0,1]=result[...,1,0]=mandel[...,5]/np.sqrt(2)
    return result


def nonlinear_force_factored(model,coordinates,laws):
    """Algebraic full-energy increments evaluated in batches without cancellation.

    Every quadratic and cubic tensor entry participates. Factoring the residual
    into its geometric and material terms changes evaluation order only; it
    neither drops higher powers of displacement nor substitutes material data.
    """
    from .modal_sensitivity import mandel_strain
    coordinates,laws=np.atleast_2d(coordinates),np.asarray(laws,dtype=np.int64)
    if coordinates.shape!=(len(laws),len(model['omega'])) or np.any((laws<0)|(laws>2)):
        raise ValueError('Matching modal states and constitutive branches required')
    g=(coordinates@model['gradient']).reshape(len(laws),-1,3,3)
    linear=mandel_strain(g)
    geometric=mandel_strain(.5*np.swapaxes(g,-1,-2)@g)
    geometric[laws==0]=0.
    strain=linear+geometric
    q=model['quadratic'];t=model['cubic']
    linear_second=2e11*(linear@q.T)
    geometric_second=2e11*(geometric@q.T)
    stress=_tensor(geometric_second)+g@_tensor(linear_second+geometric_second)
    energy=1e11*np.sum(linear*(geometric@q.T)+geometric*(linear@q.T)+geometric*(geometric@q.T),axis=-1)
    for row in np.flatnonzero(laws==2):
        material=3e11*np.einsum('abc,pb,pc->pa',t,strain[row],strain[row],optimize=True)
        tensor=_tensor(material)
        stress[row]+=tensor+g[row]@tensor
        energy[row]+=np.sum(strain[row]*material,axis=-1)/3
    stress[laws==0]=0.;energy[laws==0]=0.
    weighted=stress*model['volume'][None,:,None,None]
    force=weighted.reshape(len(laws),-1)@model['gradient'].T
    potential=np.sum(energy*model['volume'][None,:],axis=1)
    maximum=np.sqrt(np.sum(strain*strain,axis=-1)).max(axis=1)
    f=g+np.eye(3)
    determinant=(f[...,0,0]*(f[...,1,1]*f[...,2,2]-f[...,1,2]*f[...,2,1])
                 -f[...,0,1]*(f[...,1,0]*f[...,2,2]-f[...,1,2]*f[...,2,0])
                 +f[...,0,2]*(f[...,1,0]*f[...,2,1]-f[...,1,1]*f[...,2,0]))
    minimum=determinant.min(axis=1)
    if not np.isfinite(force).all() or not np.isfinite(potential).all() or np.any(maximum>.001) or np.any(minimum<=0):
        raise ValueError('Modal graph violates strain/J/finite-energy guard')
    return force,potential,maximum,minimum



def nonlinear_force(model,coordinates,laws):
    return nonlinear_force_pointwise(model,coordinates,laws)


def oscillator_flow(position, velocity, omega, step):
    angle = omega*step
    cosine, sine = np.cos(angle), np.sin(angle)
    return position*cosine+velocity*(sine/omega), velocity*cosine-position*(omega*sine)


def propagate(model, scene, *, step_us, composition=None, keep_coordinates=False, progress=None):
    """Conservative nonlinear kicks around exact candidate linear evolution.

    All three constitutive controls start at zero. The same declared fourth
    order symmetric composition is applied to each. Output covers the complete
    requested interval; no phase alignment, amplitude fit or repeated tail.
    """
    from .signals import pulse_values
    if composition is None:
        a = 1/(2-2**(1/3)); composition = np.array([a, 1-2*a, a])
    composition = np.asarray(composition, dtype=float)
    if composition.shape != (3,) or not np.isfinite(composition).all() or abs(composition.sum()-1) > 1e-12 or abs(composition[0]-composition[-1]) > 1e-12:
        raise ValueError('A consistent symmetric three-stage composition is required')
    acq = scene['acquisition']; sample_us = acq['sample_step_us']; duration_us = acq['duration_us']
    if not np.isfinite(step_us) or not 0 < step_us <= sample_us:
        raise ValueError('Finite positive step no longer than the output interval required')
    subdivisions = round(sample_us/step_us)
    if abs(subdivisions*step_us-sample_us) > 1e-10 or abs(round(duration_us/sample_us)*sample_us-duration_us) > 1e-8:
        raise ValueError('Complete integer time grid required')
    steps = round(duration_us/step_us); frames = round(duration_us/sample_us)+1
    omega, load = model['omega'], model['load']
    q = np.zeros((3, len(omega))); v = np.zeros_like(q); work = np.zeros(3)
    nonlinear = np.zeros_like(q); nonlinear_energy = np.zeros(3)
    max_strain, min_j = np.zeros(3), np.ones(3)
    traces = np.zeros((3, frames, len(scene['receivers']), 3))
    energy_log = np.zeros((3, frames, 3))
    retained = np.zeros((3, frames, len(omega))) if keep_coordinates else None
    # Pulse values are computed once at the actual signed composition times.
    stage_offsets = np.r_[0., np.cumsum(composition)]
    pulse_times = (np.arange(steps)[:, None]+stage_offsets[None, :])*step_us
    pulses = pulse_values(scene, pulse_times)
    laws = np.array([1, 2], dtype=np.int64)
    for index in range(steps):
        for stage, weight in enumerate(composition):
            h = weight*step_us*1e-6
            external = load*pulses[index, stage]
            force = external[None, :]-nonlinear
            new_v = v+.5*h*force
            work += np.sum(.5*h*external[None, :]*.5*(v+new_v), axis=1)
            q, v = oscillator_flow(q, new_v, omega, h)
            nforce, nenergy, strain, jac = nonlinear_force(model, q[1:], laws)
            nonlinear[1:], nonlinear_energy[1:] = nforce, nenergy
            max_strain[1:] = np.maximum(max_strain[1:], strain); min_j[1:] = np.minimum(min_j[1:], jac)
            external = load*pulses[index, stage+1]
            new_v = v+.5*h*(external[None, :]-nonlinear)
            work += np.sum(.5*h*external[None, :]*.5*(v+new_v), axis=1)
            v = new_v
        if (index+1) % subdivisions == 0:
            at = (index+1)//subdivisions
            traces[:, at] = np.einsum('lm,mrc->lrc', q, model['receiver_fields'], optimize=True)
            kinetic = np.sum(.5*v*v, axis=1)
            potential = np.sum(.5*(q*omega)**2, axis=1)+nonlinear_energy
            energy_log[:, at] = np.stack([kinetic, potential, work], axis=-1)
            if retained is not None:retained[:, at] = q
            if progress is not None and (at % 100 == 0 or at == frames-1):
                progress({'completed_us':at*sample_us,'duration_us':duration_us})
    # Check the original energy budget per branch, with its own fixed history.
    scale = np.maximum(np.max(abs(energy_log[:, :, 2]), axis=1),
                       np.max(abs(energy_log[:, :, :2].sum(axis=-1)), axis=1))
    error = np.max(abs(energy_log[:, :, :2].sum(axis=-1)-energy_log[:, :, 2]), axis=1)/np.maximum(scale, 1e-30)
    if not np.isfinite(traces).all() or np.any(error > .01):
        raise ValueError('Modal propagation exceeds the 1% energy-balance guard')
    return {'times_us':np.arange(frames)*sample_us, 'traces':traces, 'energy':energy_log,
            'coordinates':retained, 'energy_relative_error':error,
            'maximum_nonlinear_strain':max_strain[1:], 'minimum_nonlinear_j':min_j[1:],
            'modes':len(omega), 'step_us':step_us, 'modal_truncation_convergence_verified':False,
            'time_convergence_verified':False, 'scientific_accuracy_certified':False}
