"""Three trainable, objective cubic-energy coupling multipliers.

The wave propagation is a classical reduced physical model. This small learned
layer is intentionally not described as a general black-box neural operator.
"""
import torch

class EnergyCoupling(torch.nn.Module):
    def __init__(self):
        super().__init__();self.raw=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64))

    def forward(self):return 1+.1*torch.tanh(self.raw)

    def cubic_energy(self,gradient,material):
        f=torch.eye(3,dtype=gradient.dtype)+gradient;e=(f.transpose(-1,-2)@f-torch.eye(3,dtype=gradient.dtype))*.5;e2=e@e;tr=e.diagonal(dim1=-2,dim2=-1).sum(-1);tr2=e2.diagonal(dim1=-2,dim2=-1).sum(-1);tr3=(e2@e).diagonal(dim1=-2,dim2=-1).sum(-1)
        reference=torch.tensor([material['n_pa'],material['m_pa']-material['n_pa']/2,material['l_pa']-material['m_pa']+material['n_pa']/2],dtype=gradient.dtype);a,b,c=self()*reference
        return a/3*tr3+b*tr*tr2+c/3*tr**3
