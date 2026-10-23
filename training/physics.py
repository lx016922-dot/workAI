"""Total-Lagrangian plane strain, with the same Murnaghan convention as COMSOL.

q = (X / 0.01 m, (Y-h/2) / 0.01 m, t / 1 us), U = displacement / 100 nm.
Residuals are dimensionless. All automatic differentiation uses float64.
"""
import torch

L0, T0, U0 = .01, 1e-6, 1e-7
EPS = U0 / L0

def derivative(value, q):
    if not value.requires_grad:
        return q * 0
    out = torch.autograd.grad(value.sum(), q, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return q * 0 if out is None else out

def first_piola(g, material, law='material'):
    """g is the physical 3x3 displacement gradient, not F or strain."""
    eye = torch.eye(3, dtype=g.dtype, device=g.device)
    lam, mu = material['lambda_pa'], material['mu_pa']
    if law == 'linear':
        e = (g + g.transpose(-1, -2)) / 2
        p = lam * e.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * eye + 2 * mu * e
        return p, e, torch.linalg.det(eye + g)
    e = (g + g.transpose(-1, -2) + g.transpose(-1, -2) @ g) / 2
    tr = e.diagonal(dim1=-2, dim2=-1).sum(-1)
    e2 = e @ e
    a = material['n_pa'] if law == 'material' else 0.
    b = material['m_pa'] - material['n_pa'] / 2 if law == 'material' else 0.
    c = material['l_pa'] - material['m_pa'] + material['n_pa'] / 2 if law == 'material' else 0.
    s = (lam * tr + b * e2.diagonal(dim1=-2, dim2=-1).sum(-1) + c * tr**2)[..., None, None] * eye + (2 * mu + 2 * b * tr)[..., None, None] * e + a * e2
    f = eye + g
    return f @ s, e, torch.linalg.det(f)

def energy(g, material):
    e = (g + g.transpose(-1, -2) + g.transpose(-1, -2) @ g) / 2
    tr = e.diagonal(dim1=-2, dim2=-1).sum(-1)
    tr2 = (e @ e).diagonal(dim1=-2, dim2=-1).sum(-1)
    tr3 = (e @ e @ e).diagonal(dim1=-2, dim2=-1).sum(-1)
    a, b, c = material['n_pa'], material['m_pa'] - material['n_pa']/2, material['l_pa'] - material['m_pa'] + material['n_pa']/2
    return material['lambda_pa']/2 * tr**2 + material['mu_pa'] * tr2 + a/3 * tr3 + b * tr * tr2 + c/3 * tr**3

def stresses(parts, q, scene):
    # parts are additive linear, geometry-only increment, material increment.
    gradients = torch.stack([torch.stack([derivative(parts[:, k, j], q)[:, :2] for j in range(2)], 1) for k in range(3)], 1)
    total = gradients.cumsum(1)
    c0 = scene['material']['rho_kg_m3'] * (L0 / T0)**2
    ps, strains, jacobians = [], [], []
    for k, law in enumerate(['linear', 'geometry', 'material']):
        g = torch.nn.functional.pad(total[:, k] * EPS, (0, 1, 0, 1))
        p, strain, jac = first_piola(g, scene['nonlinear'], law)
        ps.append(p[:, :2, :2] / (c0 * EPS)); strains.append(strain); jacobians.append(jac)
    p = torch.stack(ps, 1)
    increments = torch.stack([p[:, 0], p[:, 1] - p[:, 0], p[:, 2] - p[:, 1]], 1)
    return increments, torch.stack(strains, 1), torch.stack(jacobians, 1)

def momentum(parts, q, scene):
    p, strain, jac = stresses(parts, q, scene)
    residual = []
    for k in range(3):
        axes = []
        for j in range(2):
            acceleration = derivative(derivative(parts[:, k, j], q)[:, 2], q)[:, 2]
            divergence = derivative(p[:, k, j, 0], q)[:, 0] + derivative(p[:, k, j, 1], q)[:, 1]
            axes.append(acceleration - divergence)
        residual.append(torch.stack(axes, 1))
    return torch.stack(residual, 1), strain, jac

def pulse(time_us, scene):
    s = scene['source']; time = time_us - s['delay_us']; duration = s['cycles'] / s['frequency_mhz']
    hann = .5 * (1 - torch.cos(2 * torch.pi * time / duration))
    return torch.where((time >= 0) & (time <= duration), hann * torch.sin(2 * torch.pi * s['frequency_mhz'] * time + s['phase_deg'] * torch.pi / 180), 0.)

def boundary_residual(parts, q, normals, source_edge, scene, amplitude=1.):
    p, strain, jac = stresses(parts, q, scene)
    traction = torch.einsum('nkij,nj->nki', p, normals)
    c0 = scene['material']['rho_kg_m3'] * (L0/T0)**2
    load = source_edge * amplitude * scene['source']['amplitude_pa'] / (c0 * EPS) * pulse(q[:, 2], scene)
    target = torch.zeros_like(traction)
    target[:, 0, 0] = load
    return traction - target, strain, jac
