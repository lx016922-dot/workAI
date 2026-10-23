"""Source-aligned features and geometry-modulated neural field.

Bulk and rod travel times are coordinate features, not imposed wave speeds,
modal labels or a replacement for the learned three-dimensional solution.
No FEM response, fitted mode or held-out displacement enters these features.
"""
import math
import torch
from .physics import L0

def setup_features(case):
    if 'propagation_features' in case:return case['propagation_features']
    p=case['points'];lower=p.min(0).values;size=p.max(0).values-lower;s=case['scene']['source'];g=case['scene']['geometry'];m=case['scene']['material']
    if s['face'] not in ['xmin','xmax']:raise ValueError('This experimental encoding requires an axial end-face source.')
    right=s['face']=='xmax';q=(p-lower)/size
    q=torch.stack([1-q[:,0] if right else q[:,0],q[:,1]-.5,q[:,2]-.5],1)
    surfaces=case['surface_features'].clone();centers=(surfaces[:,:3]-lower)/size;centers[:,1:]-=.5
    if right:centers[:,0]=1-centers[:,0];surfaces[:,3]*=-1
    surfaces[:,:3]=centers
    kind=['plate','pipe','hole','notch'].index(g['kind']);shape=[float(kind==i) for i in range(4)]
    void=[0.]*4
    if kind==1:void=[g['inner_radius_mm']/g['outer_radius_mm'],0.,0.,0.]
    if kind==2:void=[g['hole_radius_mm']/g['width_mm'],g['hole_x_mm']/g['length_mm'],g['hole_y_mm']/g['width_mm'],0.]
    if kind==3:void=[g['notch_depth_mm']/g['thickness_mm'],g['notch_x_mm']/g['length_mm'],g['notch_width_mm']/g['length_mm'],0.]
    if right and kind in [2,3]:void[1]=1-void[1]
    young=m['mu_pa']*(3*m['lambda_pa']+2*m['mu_pa'])/(m['lambda_pa']+m['mu_pa'])
    speeds=[math.sqrt((m['lambda_pa']+2*m['mu_pa'])/m['rho_kg_m3'])*.001,math.sqrt(m['mu_pa']/m['rho_kg_m3'])*.001,math.sqrt(young/m['rho_kg_m3'])*.001] # mm/us
    extra=torch.tensor(shape+void+speeds,dtype=torch.float64)
    context=torch.cat([case['context'],extra]) # 28
    factor=torch.ones(len(p),dtype=torch.float64);fixed=case['scene']['boundary']['fixed_face']
    if fixed!='none':
        axis='xyz'.index(fixed[0]);coord=(p[:,axis]-lower[axis])/size[axis];factor=coord if fixed.endswith('min') else 1-coord
    # Enforce exact fixed nodes even with roundoff in the geometric extent.
    factor=torch.where(case['fixed'],0.,factor)
    values={'q':q,'surfaces':surfaces,'context':context,'length_mm':float(size[0]*L0*1000),'speeds':speeds,'fixed_factor':factor}
    case['propagation_features']=values;return values

def query_features(case,indices,time):
    f=setup_features(case);q=f['q'][indices];s=case['scene']['source'];tau=time-s['delay_us'];features=[q,tau[:,None]/14]
    for frequency in [.5,1.,2.,4.]:
        phase=2*torch.pi*frequency*q;features.extend([torch.sin(phase),torch.cos(phase)])
    length=f['length_mm'];distance=q[:,0]*length;phase0=s['phase_deg']*math.pi/180
    for harmonic in [1.,2.,3.]:
        phase=harmonic*(2*torch.pi*s['frequency_mhz']*tau+phase0);features.extend([torch.sin(phase)[:,None],torch.cos(phase)[:,None]])
    for speed in f['speeds']:
        for path in [distance,2*length-distance,2*length+distance]:
            retarded=tau-path/speed
            for harmonic in [1.,2.]:
                phase=harmonic*(2*torch.pi*s['frequency_mhz']*retarded+phase0);features.extend([torch.sin(phase)[:,None],torch.cos(phase)[:,None]])
            duration=s['cycles']/s['frequency_mhz'];a=retarded/duration
            envelope=torch.where((a>=0)&(a<=1),.5*(1-torch.cos(2*torch.pi*a)),0.)
            features.append(envelope[:,None])
    return torch.cat(features,1) # 79

class PropagationConditionedField(torch.nn.Module):
    def __init__(self,scales,width=96):
        super().__init__();self.width=width;self.register_buffer('scales',torch.as_tensor(scales,dtype=torch.float64))
        self.geometry=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh())
        self.condition=torch.nn.Sequential(torch.nn.Linear(64+28,64),torch.nn.Tanh(),torch.nn.Linear(64,32),torch.nn.Tanh())
        self.input=torch.nn.Linear(79+32,width);self.layers=torch.nn.ModuleList([torch.nn.Linear(width,width) for _ in range(3)])
        self.modulation=torch.nn.Linear(32,6*width);self.output=torch.nn.Linear(width,9);self.double()
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):torch.nn.init.xavier_normal_(layer.weight,gain=.8);torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.modulation.weight);torch.nn.init.normal_(self.output.weight,std=.002)

    def encode(self,case):
        f=setup_features(case);b=self.geometry(f['surfaces']);pooled=torch.cat([(b*case['surface_weights'][:,None]).sum(0),b.max(0).values,f['context']])
        return self.condition(pooled)

    def forward(self,case,indices,time):
        f=setup_features(case);encoded=self.encode(case);h=torch.tanh(self.input(torch.cat([query_features(case,indices,time),encoded.expand(len(indices),-1)],1)))
        modulation=self.modulation(encoded).reshape(3,2,self.width)
        for i,layer in enumerate(self.layers):
            h=(h+torch.tanh(layer(h)*(1+.5*torch.tanh(modulation[i,0]))+modulation[i,1]))/math.sqrt(2)
        raw=self.output(h).reshape(-1,3,3);tau=torch.clamp(time-case['scene']['source']['delay_us'],min=0);gate=tau**2/(1+tau**2)
        return raw*self.scales[None,:,None]*gate[:,None,None]*f['fixed_factor'][indices,None,None]

class PropagationSeparatedField(PropagationConditionedField):
    """Separate decoders prevent a common hidden basis dominating weak branches."""
    def __init__(self,scales,width=96):
        super().__init__(scales,width);del self.layers;del self.output
        self.branches=torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Linear(width,width),torch.nn.Linear(width,width)]) for _ in range(3)])
        self.heads=torch.nn.ModuleList([torch.nn.Linear(width,3) for _ in range(3)]);self.modulation=torch.nn.Linear(32,12*width);self.double()
        for branch in self.branches:
            for layer in branch:torch.nn.init.xavier_normal_(layer.weight,gain=.8);torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.modulation.weight);torch.nn.init.zeros_(self.modulation.bias)
        for head in self.heads:torch.nn.init.normal_(head.weight,std=.002);torch.nn.init.zeros_(head.bias)

    def forward(self,case,indices,time):
        f=setup_features(case);encoded=self.encode(case);base=torch.tanh(self.input(torch.cat([query_features(case,indices,time),encoded.expand(len(indices),-1)],1)));modulation=self.modulation(encoded).reshape(3,2,2,self.width);parts=[]
        for k,branch in enumerate(self.branches):
            h=base
            for i,layer in enumerate(branch):h=(h+torch.tanh(layer(h)*(1+.5*torch.tanh(modulation[k,i,0]))+modulation[k,i,1]))/math.sqrt(2)
            parts.append(self.heads[k](h))
        raw=torch.stack(parts,1);tau=torch.clamp(time-case['scene']['source']['delay_us'],min=0);gate=tau**2/(1+tau**2)
        return raw*self.scales[None,:,None]*gate[:,None,None]*f['fixed_factor'][indices,None,None]
