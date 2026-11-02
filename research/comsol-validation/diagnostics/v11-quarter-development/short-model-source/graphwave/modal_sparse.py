"""Memory-bounded assembly of the exact learned graph tangent for research.

Only algebraically zero entries are omitted. No learned coefficient or small
nonzero stiffness entry is thresholded.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import eigsh

from spectral3d.mesh import boundary_data
from .inference import inverse_mapping


def stiffness(mesh,derivative,quadratic):
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
        dofs=(element.ravel()[:,None]*3+np.arange(3)).ravel().astype(np.int32)
        left,right=np.nonzero(block)
        rows.append(dofs[left]);columns.append(dofs[right]);entries.append(block[left,right])
    count=len(mesh['points'])*3
    result=coo_matrix((np.concatenate(entries),(np.concatenate(rows),np.concatenate(columns))),shape=(count,count)).tocsr()
    result.eliminate_zeros()
    return result


def modes(mesh,scene,derivative,quadratic,count):
    matrix=stiffness(mesh,derivative,quadratic)
    _,fixed,_=boundary_data(mesh,scene)
    free=np.ones(len(mesh['points'])*3,dtype=bool);free[(fixed[:,None]*3+np.arange(3)).ravel()]=False
    mass=np.repeat(mesh['mass'],3);scale=1/np.sqrt(mass[free])
    normalized=diags(scale)@matrix[free][:,free]@diags(scale)
    start=np.random.default_rng(2026102524).normal(size=free.sum())
    values,vectors=eigsh(normalized,k=count,sigma=0,which='LM',tol=1e-11,v0=start,maxiter=5000)
    order=np.argsort(values);values=values[order];vectors=vectors[:,order]
    if np.any(values<=0):raise ValueError('Nonpositive graph stiffness mode')
    fields=np.zeros((count,len(mass)));fields[:,free]=(vectors*scale[:,None]).T
    return np.sqrt(values)/(2*np.pi),fields.reshape(count,-1,3)
