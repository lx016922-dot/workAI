import torch
from .physics import L0, T0

class GuidedWaveField(torch.nn.Module):
    """Fixed-case field surrogate with hard IC, mirror parity and weak A² scaling.

    Fourier features are an optimization aid, not proof of S0 modal purity.
    The quadratic response truncation is an explicit small-amplitude assumption.
    """
    def __init__(self, scene, scales, width=48):
        super().__init__()
        self.scene, self.width = scene, width
        self.register_buffer('scales', torch.as_tensor(scales, dtype=torch.float64))
        self.length = scene['geometry']['length_mm'] * .001 / L0
        self.half = scene['geometry']['thickness_mm'] * .0005 / L0
        self.duration = scene['acquisition']['duration_us']
        cl, ct = scene['material']['cl_m_s'], scene['material']['ct_m_s']
        self.s0_hint = 2 * ct * (1 - ct**2/cl**2)**.5
        self.frequency = scene['source']['frequency_mhz']
        self.net = torch.nn.Sequential(torch.nn.Linear(16, width), torch.nn.Tanh(), torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 6)).double()
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=.7); torch.nn.init.zeros_(layer.bias)

    def forward(self, q):
        x, y, time = q.unbind(1)
        s0 = time - x * L0 / self.s0_hint / T0
        fast = time - x * L0 / self.scene['material']['cl_m_s'] / T0
        features = [2*x/self.length-1, .02*(y/self.half)**2, 2*time/self.duration-1, s0/self.duration]
        for tau in [s0, fast]:
            for harmonic in [1, 2, 3]:
                phase = 2 * torch.pi * self.frequency * harmonic * tau
                features.extend([torch.sin(phase), torch.cos(phase)])
        raw = self.net(torch.stack(features, 1)).reshape(-1, 3, 2)
        parity = torch.stack([torch.ones_like(y), y/self.half], 1)
        initial_gate = (-torch.expm1(-time))**2
        return raw * self.scales * parity[:, None, :] * initial_gate[:, None, None]

    def controls(self, q):
        p = self(q); linear, geo, mat = p.unbind(1)
        return torch.stack([linear, linear+geo, linear+geo+mat, 2*linear+4*geo, 2*linear+4*(geo+mat)], 1)
