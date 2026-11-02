"""Verify complete-force algebraic equivalence and measure both implementations."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, save
from graphwave.modal_reduction import attach_basis, nonlinear_force as dense_force
from graphwave.modal_nodal_force import prepare, nonlinear_force
from graphwave.quarter_symmetry import make_quarter_mesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fields', type=int, default=16)
    args = p.parse_args()
    config = json.loads((args.model/'model.json').read_text())
    for name, key in [('basis.npz','basis_sha256'),('weights.npz','weights_sha256')]:
        if digest(args.model/name) != config[key]:
            raise ValueError('Candidate identity changed')
    with np.load(args.model/'basis.npz',allow_pickle=False) as z:
        frequency,fields = z['frequency_hz'],z['fields']
    with np.load(args.model/'weights.npz',allow_pickle=False) as z:
        weights = dict(z)
    mesh = make_quarter_mesh(config['scene'],config['degree'],config['h_mm'],config['grading_ratios'],config['axis_elements'])
    dense = attach_basis(mesh,config['scene'],weights,frequency,fields)
    nodal = prepare(dense,mesh,weights)
    laws = np.array([1,2]); rng = np.random.default_rng(2026102605)
    qscale = (abs(dense['load'])+.01*np.linalg.norm(dense['load'])/np.sqrt(len(frequency)))/dense['omega']**2
    zero = np.zeros((2,len(frequency)))
    dense_force(dense,zero,laws);nonlinear_force(nodal,zero,laws)
    timings = {'dense':0.,'nodal':0.}; rows=[];states=[]
    for index in range(args.fields):
        q = rng.normal(size=len(frequency))*qscale
        q *= rng.uniform(2e-5,8e-4)/np.linalg.norm((q@dense['gradient']).reshape(-1,3,3),axis=(1,2)).max()
        states.append(q);state=np.stack([q,q])
        start=time.perf_counter();expected=dense_force(dense,state,laws);timings['dense']+=time.perf_counter()-start
        start=time.perf_counter();actual=nonlinear_force(nodal,state,laws);timings['nodal']+=time.perf_counter()-start
        errors={}
        for name,part in [('geometric',lambda v:v[0]),('material',lambda v:v[1]-v[0]),('total',lambda v:v[1])]:
            for index2,quantity in [(0,'force'),(1,'potential')]:
                a,b=part(actual[index2]),part(expected[index2])
                errors[name+'.'+quantity]=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
        np.testing.assert_allclose(actual[2],expected[2],rtol=1e-10,atol=1e-15)
        np.testing.assert_allclose(actual[3],expected[3],rtol=1e-12,atol=1e-15)
        rows.append(errors)
    maximum=max(max(row.values()) for row in rows)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'states.npz',coordinates=np.asarray(states))
    report={'basis_sha256':config['basis_sha256'],'weights_sha256':config['weights_sha256'],
            'field_count':args.fields,'seed':2026102605,'gradient_norm_range':[2e-5,8e-4],
            'rows':rows,'maximum_relative_error':maximum,'equivalence_limit':5e-9,
            'passed':bool(maximum<=5e-9),'timings_seconds':timings,
            'force_evaluation_speedup':timings['dense']/timings['nodal'],
            'all_original_quadrature_points_retained':True,'scientific_accuracy_certified':False,
            'source_sha256':{name:digest(Path(__file__).resolve().parents[1]/name) for name in
                ['graphwave/modal_nodal_force.py','graphwave/modal_reduction.py','graphwave/buffered_inference.py',
                 'graphwave/inference.py','tools/check-modal-nodal-force.py']}}
    save(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ['passed','maximum_relative_error','force_evaluation_speedup','timings_seconds']}),flush=True)
    if not report['passed']:
        raise RuntimeError('Equivalent-force admission failed; do not use this backend')


if __name__=='__main__':
    main()
