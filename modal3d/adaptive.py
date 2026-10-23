"""Bounded development predictor with explicit residual rejection.

Passing sampled momentum checks is necessary but does not certify accuracy.
There is deliberately no application solver routing in this module.
"""
import time
import numpy as np
from .basis import build_basis
from .enrichment import enrich
from .nonlinear import nonlinear_response
from .residual_enrichment import residual_enrich
from .differential import exact_checks

def predict(mesh,scene,scales,multipliers=(1.,1.,1.),max_corrections=2):
    if not 0<=max_corrections<=2:raise ValueError('At most two residual corrections in this CPU prototype.')
    started=time.monotonic();basis=enrich(mesh,scene,build_basis(mesh,scene,160));history=[]
    for correction in range(max_corrections+1):
        prediction=nonlinear_response(mesh,scene,basis,multipliers);checks=exact_checks(prediction['parts_m'],prediction['acceleration_m_us2'],prediction['times_us'],mesh,scene,scales);worst=max(max(c['interior_rms'],c['boundary_rms']) for c in checks);history.append({'correction':correction,'modes':len(basis['lambda_us2']),'worst_exact_momentum_rms':worst,'waveform_relaxation':prediction['iteration_history']})
        if worst<=.05 or correction==max_corrections:break
        basis=residual_enrich(mesh,scene,basis,prediction,scales)
    prediction.update(sampled_physics=checks,correction_history=history,total_seconds=time.monotonic()-started,sampled_momentum_passed=worst<=.05,accepted_for_solver_routing=False,scope='Development-only numerical hybrid; geometry/mode preparation and nonlinear propagation are classical. Sampled residual is not a continuous-domain accuracy certificate.')
    return prediction
