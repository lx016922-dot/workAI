"""Trainable graph filters, polynomial energy network, and recurrent state layer.

This is a spatially discretized recurrent network, not a one-shot coordinate
network. Geometry supplies connectivity, volume and inverse mappings. The
derivative filters, all 77 energy coefficients and two state-update weights
are learned. The transpose graph layer is tied to the first layer so internal
messages are the gradient of one energy. Inference has no residual minimizer,
linear/nonlinear solve, finite-element fallback, or reference-response input.
"""
from itertools import combinations_with_replacement
import math
import torch
from torch import nn

STRAIN_SCALE = 1e-4
PRESSURE_SCALE = 1e11
PAIRS = list(combinations_with_replacement(range(6), 2))
TRIPLES = list(combinations_with_replacement(range(6), 3))


def mandel(matrix):
    return torch.stack([matrix[...,0,0], matrix[...,1,1], matrix[...,2,2],
                        math.sqrt(2)*matrix[...,1,2], math.sqrt(2)*matrix[...,0,2],
                        math.sqrt(2)*matrix[...,0,1]], -1)


def symmetric(vector):
    a,b,c,d,e,f = vector.unbind(-1)
    return torch.stack([a,f/math.sqrt(2),e/math.sqrt(2),f/math.sqrt(2),b,
                        d/math.sqrt(2),e/math.sqrt(2),d/math.sqrt(2),c], -1).reshape(*vector.shape[:-1],3,3)


def derivative_features(z, monomials):
    """Derivative of the polynomial activation, without a numerical derivative."""
    columns=[]
    for term in monomials:
        values=[]
        for i in range(6):
            value=torch.zeros_like(z[...,0])
            for j,axis in enumerate(term):
                if axis==i:
                    product=torch.ones_like(value)
                    for k,other in enumerate(term):
                        if k!=j:product=product*z[...,other]
                    value=value+product
            values.append(value)
        columns.append(torch.stack(values,-1))
    return torch.stack(columns,-1)


class GraphWave(nn.Module):
    def __init__(self, degrees=(3,4)):
        super().__init__()
        self.filters=nn.ParameterDict({str(p):nn.Parameter(torch.randn(p+1,p+1,dtype=torch.float64)*.1) for p in degrees})
        self.quadratic=nn.Parameter(torch.randn(len(PAIRS),dtype=torch.float64)*.1)
        self.cubic=nn.Parameter(torch.randn(len(TRIPLES),dtype=torch.float64)*.1)
        self.state_weights=nn.Parameter(torch.tensor([.8,.7],dtype=torch.float64))

    def derivative(self, degree):
        raw=self.filters[str(degree)]
        return raw-raw.mean(-1,keepdim=True)

    def stress_parts(self, z):
        return derivative_features(z,PAIRS)@self.quadratic, derivative_features(z,TRIPLES)@self.cubic

    def stress(self, gradient, law):
        eye=torch.eye(3,dtype=gradient.dtype,device=gradient.device)
        strain=.5*(gradient+gradient.transpose(-1,-2))
        if law:strain=strain+.5*gradient.transpose(-1,-2)@gradient
        z=mandel(strain)/STRAIN_SCALE
        a,b=self.stress_parts(z)
        second=symmetric(PRESSURE_SCALE*STRAIN_SCALE*(a+(STRAIN_SCALE*b if law==2 else 0)))
        return second if law==0 else (eye+gradient)@second

    def update(self, current, previous, scaled_message):
        return current+self.state_weights[0]*(current-previous)+self.state_weights[1]*scaled_message

    def graph_message(self, current, cells, inverse, volume, law, degree):
        """Gather -> learned edge filters -> energy NN -> tied scatter."""
        local=current[cells];d=self.derivative(degree)
        gradient=torch.stack([torch.einsum('ia,eajkc->eijkc',d,local),
                              torch.einsum('ja,eiakc->eijkc',d,local),
                              torch.einsum('ka,eijac->eijkc',d,local)],-1)@inverse
        stress=self.stress(gradient,law)
        flux=(stress@inverse.transpose(-1,-2))*volume[...,None,None]
        local_message=(torch.einsum('ai,eajkc->eijkc',d,flux[...,0])+
                       torch.einsum('aj,eiakc->eijkc',d,flux[...,1])+
                       torch.einsum('ak,eijac->eijkc',d,flux[...,2]))
        return torch.zeros_like(current).index_add(0,cells.reshape(-1),local_message.reshape(-1,3))

    def export(self):
        # Dense symmetric tensor coefficients are an exact folding of the
        # polynomial activation and linear output layer, not fitted at export.
        h=torch.zeros(6,6,dtype=torch.float64);t=torch.zeros(6,6,6,dtype=torch.float64)
        from itertools import permutations
        for coefficient,term in zip(self.quadratic.detach(),PAIRS):
            unique=set(permutations(term))
            for indices in unique:h[indices]=coefficient/len(unique)
        for coefficient,term in zip(self.cubic.detach(),TRIPLES):
            unique=set(permutations(term))
            for indices in unique:t[indices]=coefficient/len(unique)
        return {'derivatives':{p:self.derivative(int(p)).detach().numpy() for p in self.filters},
                'quadratic_tensor':h.numpy(),'cubic_tensor':t.numpy(),
                'state_weights':self.state_weights.detach().numpy()}
