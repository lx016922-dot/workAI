"""Independent 3D nonlinear momentum constraints used only in offline training.

rho*u_tt = Div(P) + b, P = F*S, F = I + Grad(u), E=(F.T*F-I)/2.
Longitudinal and both transverse projections retain every mixed derivative.
Analytic manufactured fields supply body forces and tractions; no transient
validation waveform is used to construct these labels.
"""
from itertools import product
import numpy as np
from numba import njit
from spectral3d.mesh import gll
from .buffered_inference import graph_message,energy_activation

BRANCHES=('linear','stvk','full','geometric','material')
DIRECTIONS=('longitudinal','transverse_1','transverse_2')
FAMILIES=('longitudinal','transverse_1','transverse_2','mixed','torsion','mixed_shear')
CONSTRAINT_GROUPS=('momentum_interior','momentum_boundary','traction','strain_energy','internal_power')


def rigid_motion_checks(derivative,weights):
    """Finite-rotation objectivity of the actual learned graph layer, in SI units."""
    nodes,w,_,_=gll(4);size=np.array([.0017,.0021,.0013])
    coordinates=np.stack(np.meshgrid(nodes,nodes,nodes,indexing='ij'),-1).reshape(-1,3)*size/2
    cells=np.arange(125,dtype=np.int32).reshape(1,5,5,5)
    inverse=np.broadcast_to(np.diag(2/size),(1,5,5,5,3,3)).copy()
    volume=np.einsum('i,j,k->ijk',w,w,w)[None]*np.prod(size)/8
    fields=[]
    for axis in range(3):
        i,j=[k for k in range(3) if k!=axis]
        for angle in [.15,.73,1.5]:
            rotation=np.eye(3);rotation[i,i]=rotation[j,j]=np.cos(angle)
            rotation[i,j]=-np.sin(angle);rotation[j,i]=np.sin(angle)
            fields.append((f'rotation_axis_{axis}_angle_{angle}',coordinates@(rotation-np.eye(3)).T))
    for axis in range(3):fields.append((f'translation_axis_{axis}',np.broadcast_to(np.eye(3)[axis]*.001,(125,3)).copy()))
    records=[]
    for name,field in fields:
        for law in [1,2]:
            force,energy,strain,jacobian=graph_message(np.ascontiguousarray(field),cells,derivative,inverse,volume,
                                                     weights['quadratic_tensor'],weights['cubic_tensor'],law)
            records.append({'field':name,'law':law,'force_l2_N':float(np.linalg.norm(force)),
                            'absolute_energy_J':float(abs(energy)),'strain_norm':float(strain),'absolute_J_minus_one':float(abs(jacobian-1))})
    limits={'force_l2_N':1e-7,'absolute_energy_J':1e-16,'strain_norm':1e-12,'absolute_J_minus_one':1e-12}
    maxima={key:max(row[key] for row in records) for key in limits}
    return {'schema':'workai.learned_rigid_motion_check.v1','states':len(fields),'control_checks':len(records),
            'scope':'StVK and full finite-strain learned operator; finite rotations are not a linear-elastic invariance test',
            'limits':limits,'maxima':maxima,'passed':all(maxima[key]<=limit for key,limit in limits.items()),'checks':records}


def _trace(a):return np.trace(a,axis1=-2,axis2=-1)


def stress_energy_parts(g,m):
    """Analytic Piola stress and energy, with stable nonlinear increments."""
    eye=np.eye(3);eps=(g+g.swapaxes(-1,-2))/2;q=g.swapaxes(-1,-2)@g/2;e=eps+q;f=eye+g
    lam,mu=m['lambda_pa'],m['mu_pa'];a=m['n_pa'];b=m['m_pa']-a/2;c=m['l_pa']-m['m_pa']+a/2
    def linear(x):return lam*_trace(x)[...,None,None]*eye+2*mu*x
    le,lq=linear(eps),linear(q);tr,esq=_trace(e),e@e
    nonlinear=(b*_trace(esq)+c*tr**2)[...,None,None]*eye+2*b*tr[...,None,None]*e+a*esq
    geometric=lq+g@(le+lq);material=f@nonlinear
    wl=.5*lam*_trace(eps)**2+mu*np.sum(eps*eps,axis=(-2,-1))
    wg=lam*_trace(eps)*_trace(q)+.5*lam*_trace(q)**2+2*mu*np.sum(eps*q,axis=(-2,-1))+mu*np.sum(q*q,axis=(-2,-1))
    wm=a/3*_trace(esq@e)+b*tr*_trace(esq)+c/3*tr**3
    return np.stack([le,le+geometric,le+geometric+material,geometric,material]),np.stack([wl,wl+wg,wl+wg+wm,wg,wm])


def divergence_parts(g,hessian,m):
    """Exact chain rule Div(P); hessian[...,i,J,K] = u_i,JK."""
    eye=np.eye(3);eps=(g+g.swapaxes(-1,-2))/2;q=g.swapaxes(-1,-2)@g/2;e=eps+q;f=eye+g
    lam,mu=m['lambda_pa'],m['mu_pa'];a=m['n_pa'];b=m['m_pa']-a/2;c=m['l_pa']-m['m_pa']+a/2
    def linear(x):return lam*_trace(x)[...,None,None]*eye+2*mu*x
    tr,esq=_trace(e),e@e
    sn=(b*_trace(esq)+c*tr**2)[...,None,None]*eye+2*b*tr[...,None,None]*e+a*esq
    result=np.zeros((5,*g.shape[:-2],3),dtype=np.result_type(g,hessian))
    for j in range(3):
        dg=hessian[...,j];de0=(dg+dg.swapaxes(-1,-2))/2
        dq=(dg.swapaxes(-1,-2)@g+g.swapaxes(-1,-2)@dg)/2;de=de0+dq;dtr=_trace(de)
        dsl=linear(de0);dsg=linear(dq)+dg@linear(e)+g@linear(de)
        dsn=(2*b*np.sum(e*de,axis=(-2,-1))+2*c*tr*dtr)[...,None,None]*eye
        dsn=dsn+2*b*(dtr[...,None,None]*e+tr[...,None,None]*de)+a*(de@e+e@de)
        dsm=dg@sn+f@dsn
        result+=np.stack([dsl,dsl+dsg,dsl+dsg+dsm,dsg,dsm])[...,j]
    return result


def polynomial_field(coefficients,exponents,coordinates,size):
    def values(orders):
        answer=np.zeros((len(coordinates),3))
        for coef,exponent in zip(coefficients.T,exponents):
            factor=1.;power=exponent.copy()
            for axis in orders:
                factor*=power[axis]*2/size[axis];power[axis]-=1
            if factor==0 or np.any(power<0):continue
            answer+=np.prod(coordinates**power,axis=1)[:,None]*(factor*coef)
        return answer
    u=values([]);gradient=np.stack([values([j]) for j in range(3)],-1)
    hessian=np.stack([np.stack([values([j,k]) for j in range(3)],-1) for k in range(3)],-1)
    return u,gradient,hessian


def make_dataset(material,count=48,seed=2026102413):
    rng=np.random.default_rng(seed);nodes,w,_,_=gll(4);n=len(nodes)
    xyz=np.stack(np.meshgrid(nodes,nodes,nodes,indexing='ij'),-1).reshape(-1,3)
    exponents=np.array([p for p in product(range(4),repeat=3) if 1<=sum(p)<=3])
    cells=np.arange(n**3,dtype=np.int32).reshape(1,n,n,n);items=[]
    interior=np.all(abs(xyz)<1-1e-12,axis=1)
    for index in range(count):
        family=FAMILIES[index%len(FAMILIES)];size=rng.uniform(1.2,2.5,3)*1e-3
        coefficients=np.zeros((3,len(exponents)))
        if family in FAMILIES[:3]:
            axis=FAMILIES.index(family)
            for exponent,value in [((2,0,0),1.),((3,0,0),.35)]:
                coefficients[axis,np.flatnonzero(np.all(exponents==exponent,axis=1))[0]]=value
        elif family=='torsion':
            coefficients[1,np.flatnonzero(np.all(exponents==(1,0,1),axis=1))[0]]=-1
            coefficients[2,np.flatnonzero(np.all(exponents==(1,1,0),axis=1))[0]]=1
        else:coefficients=rng.normal(size=coefficients.shape)/np.maximum(1,exponents.sum(1))
        amplitude=10**rng.uniform(-6,-4)
        _,gradient,_=polynomial_field(coefficients,exponents,xyz,size)
        coefficients*=amplitude/np.max(np.linalg.norm(gradient,axis=(-2,-1)))
        u,gradient,hessian=polynomial_field(coefficients,exponents,xyz,size)
        stress,energy=stress_energy_parts(gradient,material);divergence=divergence_parts(gradient,hessian,material)
        volume=np.einsum('i,j,k->ijk',w,w,w)[None]*np.prod(size)/8
        inverse=np.broadcast_to(np.diag(2/size),(1,n,n,n,3,3)).copy()
        face_load=np.zeros((5,n**3,3));face_nodes=[];normals=[];traction=[]
        for axis in range(3):
            tangential=[a for a in range(3) if a!=axis]
            area=np.outer(w,w)*np.prod(size[tangential])/4
            for end in [-1,1]:
                selected=np.flatnonzero(abs(xyz[:,axis]-end)<1e-14)
                load=end*stress[:,selected][...,axis]
                face_load[:,selected]+=load*area.ravel()[None,:,None]
                face_nodes.extend(selected.tolist());normals.extend([np.eye(3)[axis]*end]*len(selected));traction.append(load)
        target=face_load-volume.reshape(1,-1,1)*divergence
        # u(X,t)=shape(X)*(1-cos(omega*t)) has exactly zero initial
        # displacement/velocity. The sampled snapshot is u above; its analytic
        # acceleration and body force make the complete motion equation exact.
        omega=2*np.pi*rng.uniform(.18,.28)*1e6;phase=rng.uniform(.35,2.7)
        velocity_factor=omega*np.sin(phase)/(1-np.cos(phase))
        acceleration_factor=omega**2*np.cos(phase)/(1-np.cos(phase))
        acceleration=np.stack([u*acceleration_factor]*3+[u*0,u*0])
        body_force=material['rho_kg_m3']*acceleration-divergence
        velocity=u*velocity_factor
        frame=np.eye(3)
        if family in ['mixed','mixed_shear']:
            frame=np.linalg.qr(rng.normal(size=(3,3)))[0]
        # Each nonlinear increment gets its own physical scale; linear forces
        # cannot hide transverse nonlinear or weak coupled terms.
        modulus=material['lambda_pa']+2*material['mu_pa']
        characteristic=modulus*np.array([amplitude,amplitude,amplitude,amplitude**2,amplitude**2])
        projected=divergence@frame
        density_scale=np.maximum(np.sqrt(np.mean(projected**2,axis=1)),.05*characteristic[:,None]/size.min())
        target_traction=np.concatenate(traction,axis=1)
        traction_scale=np.maximum(np.sqrt(np.mean((target_traction@frame)**2,axis=1)),.05*characteristic[:,None])
        energy_target=np.einsum('bn,n->b',energy,volume.ravel())
        energy_scale=np.maximum(abs(energy_target),.05*characteristic*amplitude*np.prod(size))
        power_target=np.einsum('bnij,nij,n->b',stress,gradient*velocity_factor,volume.ravel())
        power_scale=np.maximum(abs(power_target),.05*energy_scale*omega)
        items.append({'family':family,'u':np.ascontiguousarray(u),'cells':cells,'inverse':inverse,'volume':volume,
                      'target_force':target,'density_scale':density_scale,'interior':interior,'frame':frame,
                      'face_nodes':np.array(face_nodes),'normals':np.array(normals),'traction':target_traction,
                      'traction_scale':traction_scale,'energy_target':energy_target,'energy_scale':energy_scale,
                      'amplitude':amplitude,'size':size,'analytic_gradient':gradient,'analytic_hessian':hessian,
                      'acceleration':acceleration,'velocity':velocity,'body_force_density':body_force,
                      'boundary_load':face_load,'rho':np.array(material['rho_kg_m3']),
                      'sample_time_s':np.array(phase/omega),'angular_frequency':np.array(omega),
                      'power_target':power_target,'power_scale':power_scale})
    return items


@njit(cache=True)
def _stresses(gradient,quadratic,cubic):
    out=np.empty((3,len(gradient),3,3))
    for law in range(3):
        for i in range(len(gradient)):out[law,i]=energy_activation(gradient[i],quadratic,cubic,law)[0]
    return out


def _parts(x):return np.stack([x[0],x[1],x[2],x[1]-x[0],x[2]-x[1]])


def evaluate(derivative,weights,dataset,*,details=False):
    residuals=[];reports=[]
    for item in dataset:
        messages=[];energies=[]
        for law in range(3):
            force,potential,_,_=graph_message(item['u'],item['cells'],derivative,item['inverse'],item['volume'],
                                             weights['quadratic_tensor'],weights['cubic_tensor'],law)
            messages.append(force);energies.append(potential)
        force_parts=_parts(np.array(messages))
        error=(item['rho']*item['acceleration']+(force_parts-item['boundary_load'])/item['volume'].reshape(1,-1,1)
               -item['body_force_density'])
        normalized=(error@item['frame'])/item['density_scale'][:,None,:]
        groups={}
        for name,mask in [('momentum_interior',item['interior']),('momentum_boundary',~item['interior'])]:
            values=normalized[:,mask];groups[name]=np.sqrt(np.mean(values**2,axis=1))
            residuals.append(values.ravel()/np.sqrt(values.size*len(dataset)))
        local=item['u'].reshape(5,5,5,3)
        grad=np.stack([np.einsum('ia,ajkc->ijkc',derivative,local),np.einsum('ja,iakc->ijkc',derivative,local),
                       np.einsum('ka,ijac->ijkc',derivative,local)],-1).reshape(-1,3,3)@item['inverse'][0,0,0,0]
        stress=_parts(_stresses(np.ascontiguousarray(grad),weights['quadratic_tensor'],weights['cubic_tensor']))
        predicted_traction=np.einsum('bnij,nj->bni',stress[:,item['face_nodes']],item['normals'])
        tr=((predicted_traction-item['traction'])@item['frame'])/item['traction_scale'][:,None,:]
        groups['traction']=np.sqrt(np.mean(tr**2,axis=1));residuals.append(tr.ravel()/np.sqrt(tr.size*len(dataset)))
        en=(_parts(np.array(energies))-item['energy_target'])/item['energy_scale']
        groups['strain_energy']=abs(en);residuals.append(en/np.sqrt(en.size*len(dataset)))
        power=(np.einsum('bnc,nc->b',force_parts,item['velocity'])-item['power_target'])/item['power_scale']
        groups['internal_power']=abs(power);residuals.append(power/np.sqrt(power.size*len(dataset)))
        if details:reports.append({'family':item['family'],'gradient_amplitude':float(item['amplitude']),
                                   'element_size_m':item['size'].tolist(),'time_s':float(item['sample_time_s']),
                                   **{key:value.tolist() for key,value in groups.items()}})
    residual=np.concatenate(residuals)
    if not np.isfinite(residual).all():raise ValueError('Nonfinite nonlinear physical constraint')
    if not details:return residual
    maxima={key:float(max(np.max(row[key]) for row in reports)) for key in CONSTRAINT_GROUPS}
    return residual,{'branches':list(BRANCHES),'projections':list(DIRECTIONS),'sample_count':len(dataset),
                     'families':list(FAMILIES),'maximum_normalized_residual':maxima,'samples':reports,
                     'normalized_residual_rms':float(np.sqrt(np.sum(residual**2)/len(CONSTRAINT_GROUPS))),
                     'all_physical_checks_within_5_percent':all(value<=.05 for value in maxima.values()),
                     'manufactured_body_force':'rho*manufactured_acceleration - Div(P); analytic chain rule',
                     'manufactured_time_function':'shape(X)*(1-cos(omega*t)); exact zero initial displacement and velocity',
                     'increment_scope':'constitutive-force increments on the same manufactured field, not fitted transient response differences',
                     'boundary_force':'P*N on all six faces; initial/fixed state constraints are exact in the recurrence',
                     'normalization':'per-branch and per-projection reference RMS, floor 5% of material/gradient characteristic; no candidate-dependent scaling',
                     'transient_accuracy_certified':False}
