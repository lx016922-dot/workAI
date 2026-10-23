"""Semi-analytical linear FEM for a free rectangular waveguide along X.

u = Re[phi(y,z) exp(i omega t - i k x)]. P1 triangular cross-section,
consistent mass, engineering-shear Voigt order xx,yy,zz,yz,xz,xy.
Natural boundaries impose zero lateral traction. No end boundaries or
nonlinear coupling are represented by this linear infinite-guide model.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import brentq

def rectangle(width_m,thickness_m,h_m):
    if not all(np.isfinite(v) and v>0 for v in [width_m,thickness_m,h_m]):raise ValueError('Positive finite dimensions required.')
    ny=max(2,int(np.ceil(width_m/h_m)));nz=max(2,int(np.ceil(thickness_m/h_m)))
    if (ny+1)*(nz+1)>1500:raise ValueError('Cross-section node budget exceeded.')
    points=np.array([[y,z] for y in np.linspace(0,width_m,ny+1) for z in np.linspace(0,thickness_m,nz+1)])
    triangles=[]
    for i in range(ny):
        for j in range(nz):
            a=i*(nz+1)+j;b=(i+1)*(nz+1)+j;c=b+1;d=a+1
            # Alternating diagonals avoid a single preferred shear orientation.
            triangles.extend([[a,b,c],[a,c,d]] if (i+j)%2==0 else [[a,b,d],[b,c,d]])
    return points,np.array(triangles,dtype=int)

def assemble(points,triangles,rho,lam,mu):
    if not all(np.isfinite(v) and v>0 for v in [rho,mu,lam+2*mu/3]):raise ValueError('Material must have positive density and elastic energy.')
    C=np.zeros((6,6));C[:3,:3]=lam;C[np.arange(3),np.arange(3)]+=2*mu;C[3:,3:]=np.eye(3)*mu
    rows=[];cols=[];values=[[],[],[],[]];area_total=0.
    quadrature=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    for tri in triangles:
        yz=points[tri];area=abs(np.linalg.det((yz[1:]-yz[0]).T))/2
        if area<=0:raise ValueError('Degenerate cross-section triangle.')
        gradients=np.linalg.inv(np.column_stack([np.ones(3),yz]))[1:].T
        B0=np.zeros((6,9));
        for a,(dy,dz) in enumerate(gradients):
            B0[1,3*a+1]=dy;B0[2,3*a+2]=dz;B0[3,3*a+1]=dz;B0[3,3*a+2]=dy;B0[4,3*a]=dz;B0[5,3*a]=dy
        k0=area*(B0.T@C@B0);g=np.zeros((9,9));k2=np.zeros((9,9));mass=np.zeros((9,9))
        for N in quadrature:
            B1=np.zeros((6,9))
            for a,n in enumerate(N):B1[0,3*a]=n;B1[4,3*a+2]=n;B1[5,3*a+1]=n
            cross=B0.T@C@B1;g+=area/3*(cross-cross.T);k2+=area/3*(B1.T@C@B1);mass+=rho*area/3*np.kron(np.outer(N,N),np.eye(3))
        dofs=(3*tri[:,None]+np.arange(3)).ravel();rows.extend(np.repeat(dofs,9));cols.extend(np.tile(dofs,9));area_total+=area
        for storage,block in zip(values,[k0,g,k2,mass]):storage.extend(block.ravel())
    shape=(3*len(points),)*2
    K0,G,K2,M=[coo_matrix((v,(rows,cols)),shape=shape).tocsc() for v in values]
    axial=np.zeros(shape[0]);axial[::3]=1
    return {'points_m':points,'triangles':triangles,'K0':K0,'G':G,'K2':K2,'M':M,'rho':rho,'area_m2':area_total,'axial':axial}

def modes(system,k,count=12):
    if not np.isfinite(k) or k<=0:raise ValueError('Positive wavenumber required.')
    K=system['K0']-1j*k*system['G']+k*k*system['K2'];M=system['M'];size=K.shape[0]
    # Deterministic starting vector; no measured transient trace enters the solve.
    v0=np.sin(np.arange(size)+.7)+1j*np.cos(np.arange(size)*.37)
    # A negative spectral shift avoids factoring the nearly singular flexural
    # stiffness at long wavelength. It changes only the eigensolver, not K/M.
    speed_squared_scale=float(np.max(system['K2'].diagonal()/M.diagonal()))
    shift=-speed_squared_scale*k*k
    vals,vec=eigsh(K,k=min(count,size-2),M=M,sigma=shift,which='LM',tol=1e-12,ncv=min(size,max(4*count+1,30)),v0=v0)
    order=np.argsort(vals);vals=vals[order];vec=vec[:,order];positive=vals>1e-6;vals=vals[positive];vec=vec[:,positive]
    for i in range(vec.shape[1]):
        norm=np.sqrt(np.real(np.vdot(vec[:,i],M@vec[:,i])));vec[:,i]/=norm
        phase=np.vdot(system['axial'],M@vec[:,i]);vec[:,i]*=np.exp(-1j*np.angle(phase))
    coupling=abs(system['axial']@(M@vec))**2/(system['rho']*system['area_m2'])
    residual=[float(np.linalg.norm(K@v-w*M@v)/max(np.linalg.norm(K@v),np.linalg.norm(w*M@v),1e-30)) for w,v in zip(vals,vec.T)]
    return {'frequency_hz':np.sqrt(vals)/(2*np.pi),'vectors':vec,'axial_uniform_overlap':coupling,'relative_residual':residual}

def extensional_branch(system,target_hz,young_modulus,steps=40):
    """Track from long wavelength by mass-weighted MAC, then bracket frequency.

    This is a tracked candidate, not an assertion that the transient contains
    only this mode. Near-degenerate/poorly tracked branches are rejected.
    """
    if not np.isfinite(target_hz) or target_hz<=0:raise ValueError('Positive frequency required.')
    rod_speed=np.sqrt(young_modulus/system['rho']);estimate=2*np.pi*target_hz/rod_speed;k0=max(estimate/100,0.1)
    first=modes(system,k0);index=int(np.argmax(first['axial_uniform_overlap']));previous=first['vectors'][:,index];path=[];minimum_mac=1.
    def selected(k,reference):
        all_modes=modes(system,k);overlap=abs(reference.conj()@(system['M']@all_modes['vectors']))**2;i=int(np.argmax(overlap))
        return all_modes,i,float(overlap[i])
    last_k=k0;last_f=first['frequency_hz'][index];last_vector=previous
    for k in np.linspace(k0,estimate*2.5,steps+1)[1:]:
        current,i,mac=selected(k,previous);f=float(current['frequency_hz'][i]);minimum_mac=min(minimum_mac,mac)
        if mac<.8:raise ValueError('Mode tracking overlap too small; refine continuation before using the branch.')
        path.append({'k_rad_m':float(k),'frequency_hz':f,'MAC':mac,'axial_uniform_overlap':float(current['axial_uniform_overlap'][i])})
        if f>=target_hz:
            reference=last_vector
            def gap(q):r,j,_=selected(q,reference);return float(r['frequency_hz'][j])-target_hz
            root=brentq(gap,last_k,float(k),xtol=1e-8);r,j,mac=selected(root,reference);vector=r['vectors'][:,j]
            eps=root*1e-4;lo,il,_=selected(root-eps,vector);hi,ih,_=selected(root+eps,vector);vg=2*np.pi*(hi['frequency_hz'][ih]-lo['frequency_hz'][il])/(2*eps)
            if vg<=0 or abs(r['frequency_hz'][j]/target_hz-1)>1e-7 or r['relative_residual'][j]>1e-6:raise ValueError('Branch root or eigen residual invalid.')
            return {'frequency_hz':target_hz,'wavenumber_rad_m':root,'phase_velocity_m_s':2*np.pi*target_hz/root,'group_velocity_m_s':float(vg),
                'axial_uniform_overlap':float(r['axial_uniform_overlap'][j]),'relative_eigen_residual':r['relative_residual'][j],
                'minimum_tracking_MAC':min(minimum_mac,mac),'tracking_path':path,'vector':vector}
        last_k=float(k);last_f=f;previous=current['vectors'][:,i];last_vector=previous
    raise ValueError('Tracked branch did not reach the target frequency within the bounded search.')

def shape_fit(system,mode,displacement):
    """Mass-L2 best-fit shape mismatch; NOT modal energy or power fraction."""
    M=system['M'];v=np.asarray(mode);u=np.asarray(displacement).reshape(-1)
    scale=np.real(np.vdot(u,M@u))
    if scale<=0:raise ValueError('Zero displacement cannot identify a mode.')
    coefficient=np.vdot(v,M@u)/np.vdot(v,M@v);residual=u-coefficient*v
    return {'complex_coefficient':[float(coefficient.real),float(coefficient.imag)],'relative_mass_L2_shape_mismatch':float(np.sqrt(max(0,np.real(np.vdot(residual,M@residual))/scale))),'not_a_modal_energy_fraction':True}
