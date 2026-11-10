"""Material module vs closed-form literature relations (no simulation)."""
import json, math, subprocess, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from physics.materials import murnaghan_tensors, taylor_constants, compile_energy, christoffel, presets, tensors, rotation_z, mandel_rotation, PAIRS

out = {}
SQ2 = math.sqrt(2)
def voigt3(c3):  # Mandel c3 -> Brugger Voigt constants (engineering shear strains): divide by sqrt2 per shear index
    w = np.array([1, 1, 1, SQ2, SQ2, SQ2]); return c3 / np.einsum('i,j,k->ijk', w, w, w)
def voigt2(c2):
    w = np.array([1, 1, 1, SQ2, SQ2, SQ2]); return c2 / np.outer(w, w)

# --- M1: isotropic Murnaghan -> second/third-order constants (Thurston & Brugger 1964; Landau mapping in Destrade & Ogden 2013)
al = presets()['murnaghan']['material']; lam, mu, l, m, n = (al[k] for k in ('lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa'))
brugger = {'C11': lam + 2 * mu, 'C12': lam, 'C44': mu, 'C111': 2 * l + 4 * m, 'C112': 2 * l, 'C123': 2 * l - 2 * m + n, 'C144': m - n / 2, 'C155': m, 'C456': n / 4}
idx = {'C11': (0, 0), 'C12': (0, 1), 'C44': (3, 3), 'C111': (0, 0, 0), 'C112': (0, 0, 1), 'C123': (0, 1, 2), 'C144': (0, 3, 3), 'C155': (0, 4, 4), 'C456': (3, 4, 5)}
expr = ('0.5*(lam+2*mu)*i1**2 - 2*mu*i2 + (l+2*m)/3*i1**3 - 2*m*i1*i2 + n*i3')
c2e, c3e, s0 = taylor_constants(compile_energy(expr, {'lam': lam / 1e9, 'mu': mu / 1e9, 'l': l / 1e9, 'm': m / 1e9, 'n': n / 1e9}))
c2x, c3x = murnaghan_tensors(al)
rows = []
for key, ref in brugger.items():
    i = idx[key]; get = (lambda c2, c3: voigt2(c2)[i] if len(i) == 2 else voigt3(c3)[i])
    rows.append({'constant': key, 'literature_GPa': ref / 1e9, 'software_exact_GPa': get(c2x, c3x) / 1e9, 'software_custom_W_GPa': get(c2e, c3e) / 1e9})
out['M1_murnaghan_brugger'] = {'rows': rows, 'max_rel_diff_exact': max(abs(r['software_exact_GPa'] / r['literature_GPa'] - 1) for r in rows),
                               'max_rel_diff_custom_W': max(abs(r['software_custom_W_GPa'] / r['literature_GPa'] - 1) for r in rows)}

# --- M2: transversely isotropic stiffness from engineering constants (Jones 1999, compliance of a TI solid), via the workbench JS helper
def jones(E1, E2, G12, nu12, nu23):
    S = np.zeros((6, 6)); S[0, 0] = 1 / E1; S[1, 1] = S[2, 2] = 1 / E2; S[0, 1] = S[1, 0] = S[0, 2] = S[2, 0] = -nu12 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2; S[3, 3] = 2 * (1 + nu23) / E2; S[4, 4] = S[5, 5] = 1 / G12
    C = np.linalg.inv(S); return {'C11': C[0, 0], 'C12': C[0, 1], 'C22': C[1, 1], 'C23': C[1, 2], 'C55': C[4, 4], 'C44': C[3, 3]}
mats = {'example-1': dict(E1=138., E2=11., G12=5.5, nu12=.28, nu23=.4), 'example-2': dict(E1=142., E2=10.3, G12=7.2, nu12=.27, nu23=.54),
        'example-3': dict(E1=135., E2=10., G12=5., nu12=.3, nu23=.45)}  # formula checks; not claimed as specific laminates
module = (REPO / 'offline/web/material-form.js').as_uri()  # the workbench's own helper
js = subprocess.run(['node', '--input-type=module', '-e', f"import {{ tiFromEngineering }} from '{module}';"
                     f"const m = {json.dumps(mats)}; console.log(JSON.stringify(Object.fromEntries(Object.entries(m).map(([k, v]) => [k, tiFromEngineering(v)]))));"],
                    capture_output=True, text=True)
jsout = json.loads(js.stdout) if js.returncode == 0 else {'error': js.stderr[-500:]}
m2 = []
for name, e in mats.items():
    ref = jones(**e); got = jsout.get(name, {})
    m2.append({'material': name, 'engineering': e, 'jones': ref, 'workbench': got, 'max_rel_diff': max(abs(got[k] / ref[k] - 1) for k in got) if got else None,
               'C44_isotropy_check': abs(ref['C44'] - (ref['C22'] - ref['C23']) / 2) / ref['C44']})
out['M2_TI_engineering'] = m2

# --- M3: Christoffel phase speeds of the CFRP preset vs closed forms for a TI solid (Auld 1973; Rokhlin et al. 2011)
cf = presets()['cfrp_ud']['material']; P = cf['parameters']; rho = cf['rho_kg_m3']
C11, C12, C22, C23, C55 = (P[k] * 1e9 for k in ('C11', 'C12', 'C22', 'C23', 'C55')); C44 = (C22 - C23) / 2; C66 = C55
c2, _, _ = tensors(cf)
def closed_12(theta):  # phase speeds in the 1-2 plane at angle theta from the fibre (the 1-3 plane is identical): qL, qT (in-plane), SH (out of plane)
    c, s = math.cos(theta), math.sin(theta); a = C11 * c * c + C66 * s * s; b = C66 * c * c + C22 * s * s; d = (C12 + C66) * c * s
    return sorted([math.sqrt(((a + b) / 2 + math.sqrt(((a - b) / 2) ** 2 + d * d)) / rho), math.sqrt(((a + b) / 2 - math.sqrt(((a - b) / 2) ** 2 + d * d)) / rho),
                   math.sqrt((C55 * c * c + C44 * s * s) / rho)], reverse=True)
m3 = []
for deg in (0, 30, 45, 60, 90):
    t = math.radians(deg); v, _ = christoffel(c2, rho, [math.cos(t), math.sin(t), 0.]); ref = closed_12(t)
    m3.append({'angle_deg': deg, 'software_m_s': v.tolist(), 'closed_form_m_s': ref, 'max_rel_diff': float(max(abs(a / b - 1) for a, b in zip(sorted(v, reverse=True), ref)))})
# fibre rotated 90 deg about z: speed along global x must equal the transverse speeds
rot = dict(cf, fiber_angle_deg=90.); c2r, _, _ = tensors(rot); v, _ = christoffel(c2r, rho, [1., 0., 0.])
m3.append({'angle_deg': 'fibre 90°, along x', 'software_m_s': v.tolist(), 'closed_form_m_s': closed_12(math.pi / 2), 'max_rel_diff': float(max(abs(a / b - 1) for a, b in zip(sorted(v, reverse=True), closed_12(math.pi / 2))))})
out['M3_TI_christoffel'] = m3
(REPO / 'research/literature-verification/results/materials.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print('M1', out['M1_murnaghan_brugger']['max_rel_diff_exact'], out['M1_murnaghan_brugger']['max_rel_diff_custom_W'], 'M2', max(r['max_rel_diff'] for r in out['M2_TI_engineering']), 'M3', max(r['max_rel_diff'] for r in out['M3_TI_christoffel']))
