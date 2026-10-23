"""Energy-consistent curved GLL force, sharing the original constitutive law."""
import numpy as np
from numba import njit
from .solver import constitutive


@njit(cache=True)
def internal_force(u,elements,derivative,inverse,volume_weight,constants,law):
    force=np.zeros_like(u);energy=0.;max_strain=0.;min_j=1e100;n=elements.shape[1]
    flux=np.empty((n,n,n,3,3));local=np.empty((n,n,n,3));reference_gradient=np.empty((3,3))
    for e in range(len(elements)):
        ids=elements[e]
        for i in range(n):
            for j in range(n):
                for k in range(n):local[i,j,k]=u[ids[i,j,k]]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    reference_gradient[:]=0.
                    for a in range(n):
                        for c in range(3):
                            reference_gradient[c,0]+=derivative[i,a]*local[a,j,k,c]
                            reference_gradient[c,1]+=derivative[j,a]*local[i,a,k,c]
                            reference_gradient[c,2]+=derivative[k,a]*local[i,j,a,c]
                    inv=inverse[e,i,j,k];g=reference_gradient@inv;p,w,strain,jac=constitutive(g,constants,law)
                    energy+=w*volume_weight[e,i,j,k];max_strain=max(max_strain,strain);min_j=min(min_j,jac)
                    flux[i,j,k]=(p@inv.T)*volume_weight[e,i,j,k]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    node=ids[i,j,k]
                    for a in range(n):
                        for c in range(3):
                            force[node,c]+=derivative[a,i]*flux[a,j,k,c,0]+derivative[a,j]*flux[i,a,k,c,1]+derivative[a,k]*flux[i,j,a,c,2]
    return force,energy,max_strain,min_j
