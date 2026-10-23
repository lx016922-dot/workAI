"""A shared mesh-conditioned neural field, with no per-case learned embedding.

The branch encoder sees boundary coordinates/normals/load/fixity. The decoder
queries 3-D coordinates and time and predicts three additive vector fields.
"""
import torch

def make_model(scales,width=64,architecture='baseline'):
    if architecture=='baseline':return GeometryConditionedField(scales,width)
    if architecture=='propagation':
        from .propagation_model import PropagationConditionedField
        return PropagationConditionedField(scales,width)
    if architecture=='propagation_separate':
        from .propagation_model import PropagationSeparatedField
        return PropagationSeparatedField(scales,width)
    raise ValueError('Unknown neural field architecture.')

class GeometryConditionedField(torch.nn.Module):
    def __init__(self,scales,width=64):
        super().__init__();self.width=width;self.register_buffer('scales',torch.as_tensor(scales,dtype=torch.float64))
        self.geometry=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh())
        self.condition=torch.nn.Sequential(torch.nn.Linear(64+17,48),torch.nn.Tanh())
        layers=[torch.nn.Linear(34+48,width),torch.nn.Tanh()]
        for _ in range(2):layers.extend([torch.nn.Linear(width,width),torch.nn.Tanh()])
        layers.append(torch.nn.Linear(width,9));self.decoder=torch.nn.Sequential(*layers);self.double()
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):torch.nn.init.xavier_normal_(layer.weight,gain=.7);torch.nn.init.zeros_(layer.bias)
        torch.nn.init.normal_(self.decoder[-1].weight,std=.002)

    def encode(self,case):
        boundary=self.geometry(case['surface_features']);weighted=(boundary*case['surface_weights'][:,None]).sum(0);maximum=boundary.max(0).values
        return self.condition(torch.cat([weighted,maximum,case['context']]))

    def forward(self,case,indices,time):
        points=case['points'][indices];source=case['scene']['source'];feature=[points,time[:,None]/14]
        for frequency in [.5,1.,2.,4.]:
            phase=2*torch.pi*frequency*points;feature.extend([torch.sin(phase),torch.cos(phase)])
        for harmonic in [1.,2.,3.]:
            phase=2*torch.pi*source['frequency_mhz']*harmonic*time[:,None];feature.extend([torch.sin(phase),torch.cos(phase)])
        encoded=self.encode(case).expand(len(points),-1);raw=self.decoder(torch.cat(feature+[encoded],1)).reshape(-1,3,3)
        elapsed=torch.clamp(time-source['delay_us'],min=0);gate=elapsed**2/(1+elapsed**2)
        return raw*self.scales[None,:,None]*gate[:,None,None]*(~case['fixed'][indices])[:,None,None]
