"""Average-acceleration Newmark with converged full hyperelastic forces.

The constant elastic tangent is only a modified-Newton preconditioner; the
accepted equation uses the full nonlinear Piola force. No mass scaling.
"""
import math
import time
import numpy as np
from scipy.sparse import coo_matrix,diags,eye
from scipy.sparse.linalg import splu
from solid.fem import pulse
from .mesh import boundary_data,interpolation
from .solver import internal_force


def assemble(mesh,material):
    ids=mesh['elements'];n=mesh['degree']+1;q=n**3;reference=np.zeros((n,n,n,n,n,n,3));d=mesh['derivative']
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for a in range(n):
                    reference[i,j,k,a,j,k,0]=d[i,a];reference[i,j,k,i,a,k,1]=d[j,a];reference[i,j,k,i,j,a,2]=d[k,a]
    reference=reference.reshape(q,q,3);lam=material['lambda_pa'];mu=material['mu_pa'];entries=[];rows=[];cols=[]
    for e,element in enumerate(ids):
        inv=mesh['inverse_jacobian'][e].reshape(q,3,3) if 'inverse_jacobian' in mesh else np.broadcast_to(np.diag(2/mesh['size'][e]),(q,3,3))
        grad=np.einsum('qaA,qAi->qai',reference,inv);weight=mesh['volume_weight'][e].ravel()
        # Pairwise gradient integrals indexed by local node and physical axis.
        gram=(grad.reshape(q,-1).T*weight)@grad.reshape(q,-1);gram=gram.reshape(q,3,q,3)
        block=lam*gram+mu*gram.transpose(0,3,2,1)
        scalar=np.einsum('aibi->ab',gram)
        for axis in range(3):block[:,axis,:,axis]+=mu*scalar
        dofs=(element.ravel()[:,None]*3+np.arange(3)).ravel();rows.append(np.repeat(dofs,len(dofs)));cols.append(np.tile(dofs,len(dofs)));entries.append(block.ravel())
    size=len(mesh['points'])*3
    stiffness=coo_matrix((np.concatenate(entries),(np.concatenate(rows),np.concatenate(cols))),shape=(size,size)).tocsr();stiffness.eliminate_zeros()
    return stiffness


def prepare(scene,mesh,substeps=4,excitation=None):
    if type(substeps) is not int or not 1<=substeps<=32:raise ValueError('Use 1–32 internal steps per output.')
    start=time.monotonic()
    if excitation is None:
        load,fixed,area=boundary_data(mesh,scene);loads=[(load,scene['source'])];constrained=np.zeros((len(mesh['mass']),3),dtype=bool);constrained[fixed]=True;alpha=None
    else:loads=excitation['loads'];constrained=excitation['constrained'];alpha=excitation['alpha'];area=excitation['area_m2']
    mass=np.repeat(mesh['mass'],3);free=np.flatnonzero(~constrained.ravel())
    stiffness=assemble(mesh,scene['material']);scale=1/np.sqrt(mass[free]);dt=scene['acquisition']['sample_step_us']*1e-6/substeps
    # The mass-normalized elastic matrix is I plus a positive semidefinite
    # Gram operator; velocity-implicit damping adds a nonnegative diagonal.
    # Symmetric ordering avoids unnecessary fill and pivoting.
    identity=eye(len(free),format='csc') if alpha is None else diags(1+.5*dt*np.repeat(alpha,3)[free],format='csc')
    matrix=identity+.25*dt*dt*(diags(scale)@stiffness[free][:,free]@diags(scale))
    factor=splu(matrix.tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
    return {'loads':loads,'constrained':constrained,'alpha':alpha,'area':area,'free':free,'scale':scale,'mass':mass,'stiffness':stiffness,'factor':factor,'dt':dt,'substeps':substeps,
            'summary':{'excitation_model':'legacy_single_source'} if excitation is None else excitation['summary'],'preparation_seconds':time.monotonic()-start}


def solve(scene,mesh,substeps=4,keep_fields=False,callback=None,prepared=None,excitation=None):
    start=time.monotonic();p=prepared or prepare(scene,mesh,substeps,excitation);loads=p['loads'];constrained=p['constrained'];alpha=p['alpha'];mass=mesh['mass'];dt=p['dt'];free=p['free'];scale=p['scale'];factor=p['factor'];substeps=p['substeps'];force_function=internal_force;geometry=mesh.get('size')
    if 'inverse_jacobian' in mesh:
        from .curved_solver import internal_force as force_function
        geometry=mesh['inverse_jacobian']
    constants=np.array([scene['material'][k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']]);law={'linear':0,'stvk':1,'murnaghan':2}[scene['constitutive']]
    frames=round(scene['acquisition']['duration_us']/scene['acquisition']['sample_step_us'])+1;steps=(frames-1)*substeps
    if keep_fields and frames*len(mass)*24>512*1024**2:raise ValueError('Field exceeds 512 MiB.')
    if p['summary']['excitation_model']=='legacy_single_source':values=[np.array([pulse(step*dt*1e6,loads[0][1]) for step in range(steps+1)])]
    else:
        from physics.signals import evaluate
        times=np.arange(steps+1)*dt*1e6;values=[evaluate(src,times) for _,src in loads]
    damping=None if alpha is None else mass*alpha;inertia=mass if damping is None else mass+.5*dt*damping
    fields=np.zeros((frames,len(mass),3)) if keep_fields else None;probes=np.zeros((frames,len(scene['receivers']),3));receivers=[interpolation(mesh,r['point_mm']) for r in scene['receivers']]
    u=np.zeros((len(mass),3));v=u.copy();acc=u.copy();previous_ext=u.copy();energies=[];work=0.;dissipated=0.;max_strain=0.;min_j=1.;max_iterations=0;max_correction=0.;avoided_solves=0;scale_bound=2*float(scale.max())
    for step in range(1,steps+1):
        predictor=u+dt*v+.25*dt*dt*acc;next_u=predictor.copy();next_u[constrained]=0
        ext=loads[0][0]*values[0][step]
        for k in range(1,len(loads)):ext=ext+loads[k][0]*values[k][step]
        drag=0. if damping is None else damping[:,None]*(v+.5*dt*acc)
        for iteration in range(8):
            fint,energy,strain,jac=force_function(next_u,mesh['elements'],mesh['derivative'],geometry,mesh['volume_weight'],constants,law)
            residual=(inertia[:,None]*(next_u-predictor)+.25*dt*dt*(fint-ext+drag)).ravel()[free]*scale
            tolerance=1e-20+1e-11*float(np.max(abs(next_u)))
            # ||A^-1||_2 <= 1 for A=D+dt^2/4 M^-1/2 K M^-1/2 with D >= I.
            # The factor of two leaves margin for floating-point rounding.
            error=scale_bound*float(np.linalg.norm(residual))
            if error<=tolerance:avoided_solves+=1;break
            correction=factor.solve(residual)*scale;error=float(np.max(abs(correction)))
            if error<=tolerance:break
            next_u.ravel()[free]-=correction
        else:raise ValueError('Modified Newton failed to converge; no valid result.')
        max_iterations=max(max_iterations,iteration+1);max_correction=max(max_correction,error)
        if not np.isfinite(energy) or strain>.001 or jac<=0:raise ValueError(f'Elastic guard failed: {strain}, {jac}')
        max_strain=max(max_strain,strain);min_j=min(min_j,jac);next_acc=4*(next_u-predictor)/(dt*dt);next_acc[constrained]=0;next_v=v+.5*dt*(acc+next_acc);next_v[constrained]=0
        if damping is not None:dissipated+=float(np.sum(damping[:,None]*.5*(v+next_v)*(next_u-u)))
        work+=float(np.sum(.5*(ext+previous_ext)*(next_u-u)));u,v,acc=next_u,next_v,next_acc;previous_ext=ext
        if step%substeps==0:
            frame=step//substeps
            if fields is not None:fields[frame]=u
            for i,(ids,w) in enumerate(receivers):probes[frame,i]=w@u[ids]
            energies.append([step*dt*1e6,float(np.sum(.5*mass[:,None]*v*v)),energy,work]+([dissipated] if damping is not None else []))
            if callback and (frame%40==0 or frame==frames-1):callback({'frame':frame,'frames':frames,'elapsed_s':time.monotonic()-start})
    energies=np.array(energies);stored=energies[:,1]+energies[:,2]+(energies[:,4] if damping is not None else 0.)
    energy_scale=max(np.max(abs(energies[:,3])),np.max(abs(stored)),1e-30);balance=float(np.max(abs(stored-energies[:,3]))/energy_scale)
    diagnostics={'method':'isoparametric_gll_full_hyperelastic_newmark_v1','degree':mesh['degree'],'h_mm':mesh['h_mm'],'nodes':len(mass),'elements':len(mesh['elements']),'quadrature_points':int(mesh['elements'].size),'internal_steps':steps,'internal_step_us':dt*1e6,'newmark_beta':.25,'newmark_gamma':.5,'max_newton_iterations':max_iterations,'max_final_displacement_correction_m':max_correction,'preparation_seconds':p['preparation_seconds'],'elapsed_s':time.monotonic()-start,'loaded_area_mm2':p['area']*1e6,'volume_mm3':float(mesh['volume_weight'].sum()*1e9),'max_strain':max_strain,'min_J':min_j,'external_work_balance_relative':balance,'energy_balance_within_1_percent':balance<=.01,'calculation_dtype':'float64','output_dtype':'float64','accepted_for_solver_routing':False,'spatial_convergence_verified':False,'temporal_convergence_verified':False,'energy_samples':energies.tolist()}
    diagnostics.update(method='isoparametric_gll_full_hyperelastic_newmark_v2',linear_solver='symmetric_ordered_lu',avoided_converged_linear_solves=avoided_solves,correction_check='Exact preconditioned correction or conservative residual upper bound')
    if p['summary']['excitation_model']!='legacy_single_source':
        diagnostics.update(p['summary'],energy_columns=['time_us','kinetic_j','strain_j','external_work_j']+(['dissipated_j'] if damping is not None else []),dissipated_energy_j=dissipated,signal_peaks=[float(np.max(abs(v))) for v in values])
    return {'fields_m':fields,'probes_m':probes,'times_us':np.arange(frames)*scene['acquisition']['sample_step_us'],'diagnostics':diagnostics}
