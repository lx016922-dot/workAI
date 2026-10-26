"""Complete graph forces assembled at nodes before modal projection.

This is an algebraic reordering, with all quadrature points and learned tensor
entries retained. It avoids storing/multiplying the dense modal-gradient matrix
at every step. Equivalence and trajectory replay are required before use.
"""
import numpy as np
from numba import njit
from .buffered_inference import _activation
from .inference import inverse_mapping


def prepare(model, mesh, weights):
    model = dict(model)
    model['nodal_operator'] = (mesh['elements'], weights['derivative_'+str(mesh['degree'])],
                               inverse_mapping(mesh), mesh['volume_weight'])
    model['nodal_fields'] = np.ascontiguousarray(model['fields'].reshape(len(model['omega']), -1))
    # The exact same derivative is now applied to reconstructed nodal fields.
    # Removing this derived dense cache does not discard a graph coefficient.
    model.pop('gradient', None)
    return model


@njit(cache=True)
def residual_messages(current, cells, derivative, inverse, volume, quadratic, cubic, laws):
    message = np.zeros_like(current)
    potential = np.zeros(len(laws))
    maximum = np.zeros(len(laws))
    minimum = np.ones(len(laws))
    n = cells.shape[1]
    local, flux = np.empty((n,n,n,3)), np.empty((n,n,n,3,3))
    ref, g = np.empty((3,3)), np.empty((3,3))
    stress, linear_stress, strain, second = [np.empty((3,3)) for _ in range(4)]
    z, linear, nonlinear = np.empty(6), np.empty(6), np.empty(6)
    for row in range(len(laws)):
        for element in range(len(cells)):
            ids = cells[element]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for axis in range(3):
                            local[i,j,k,axis] = current[row,ids[i,j,k],axis]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        ref[:] = 0.
                        for neighbor in range(n):
                            for axis in range(3):
                                ref[axis,0] += derivative[i,neighbor]*local[neighbor,j,k,axis]
                                ref[axis,1] += derivative[j,neighbor]*local[i,neighbor,k,axis]
                                ref[axis,2] += derivative[k,neighbor]*local[i,j,neighbor,axis]
                        transform = inverse[element,i,j,k]
                        for a in range(3):
                            for b in range(3):
                                g[a,b] = 0.
                                for c in range(3):
                                    g[a,b] += ref[a,c]*transform[c,b]
                        linear_energy,_,_ = _activation(g,quadratic,cubic,0,strain,z,linear,nonlinear,second,linear_stress)
                        energy,norm,jac = _activation(g,quadratic,cubic,laws[row],strain,z,linear,nonlinear,second,stress)
                        for a in range(3):
                            for b in range(3):
                                value = 0.
                                for c in range(3):
                                    value += (stress[a,c]-linear_stress[a,c])*transform[b,c]
                                flux[i,j,k,a,b] = value*volume[element,i,j,k]
                        potential[row] += (energy-linear_energy)*volume[element,i,j,k]
                        maximum[row] = max(maximum[row],norm)
                        minimum[row] = min(minimum[row],jac)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        node = ids[i,j,k]
                        for neighbor in range(n):
                            for axis in range(3):
                                message[row,node,axis] += (derivative[neighbor,i]*flux[neighbor,j,k,axis,0]
                                    + derivative[neighbor,j]*flux[i,neighbor,k,axis,1]
                                    + derivative[neighbor,k]*flux[i,j,neighbor,axis,2])
    return message,potential,maximum,minimum


def nonlinear_force(model, coordinates, laws):
    coordinates, laws = np.atleast_2d(coordinates), np.asarray(laws, dtype=np.int64)
    if coordinates.shape != (len(laws),len(model['omega'])) or np.any((laws<0)|(laws>2)):
        raise ValueError('Matching modal states and constitutive branches required')
    fields = model['nodal_fields']
    current = (coordinates @ fields).reshape(len(laws),-1,3)
    message,potential,strain,jac = residual_messages(current,*model['nodal_operator'],model['quadratic'],model['cubic'],laws)
    force = message.reshape(len(laws),-1) @ fields.T
    if not np.isfinite(force).all() or not np.isfinite(potential).all() or np.any(strain>.001) or np.any(jac<=0):
        raise ValueError('Modal graph violates strain/J/finite-energy guard')
    return force,potential,strain,jac
