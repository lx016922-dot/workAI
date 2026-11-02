"""Development diagnostic: quartic fields expose an SBP-consistent high-order fault."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from numpy.polynomial.legendre import Legendre

root=next(p for p in Path(__file__).resolve().parents if (p/'graphwave').is_dir())
sys.path.insert(0,str(root))
from graphwave import nonlinear_constraints as physics
from graphwave.modal_calibration import sbp_checks
from spectral3d.mesh import gll

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
material=json.loads((root/'research/comsol-validation/data/eigen-035-v7/scene.json').read_text())['scene']['material']
path=root/'research/comsol-validation/training/v10-untrained/candidate-weights.npz'
with np.load(path,allow_pickle=False) as saved:weights=dict(saved)
x,w,exact,_=gll(6);a,b=Legendre.basis(5)(x),Legendre.basis(6)(x)
fault=exact+10*(np.outer(a,w*b)-np.outer(b,w*a))
records={}
for degree in [3,4]:
 fields=physics.make_dataset(material,24,2026102510,degree=6,polynomial_degree=degree)
 for name,operator in [('analytic',exact),('deliberate_fault',fault)]:
  _,report=physics.evaluate(operator,weights,fields,details=True)
  records[f'max_degree_{degree}_{name}']={'maximum_normalized_residual':report['maximum_normalized_residual'],
    'within_training_limit_2_5_percent':max(report['maximum_normalized_residual'].values())<=.025,
    'within_independent_limit_5_percent':report['all_physical_checks_within_5_percent']}
assert records['max_degree_3_deliberate_fault']['within_training_limit_2_5_percent']
assert not records['max_degree_4_deliberate_fault']['within_training_limit_2_5_percent']
assert records['max_degree_4_analytic']['within_training_limit_2_5_percent']
result={'schema':'workai.physical_field_coverage_diagnostic.v1','degree':6,'seed':2026102510,'samples':24,
 'fault_definition':'D + 10 * (outer(P5, w*P6) - outer(P6, w*P5)); Pk are Legendre polynomials at GLL nodes',
 'fault_sbp':sbp_checks(fault,w),'records':records,
 'baseline_weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'diagnostic_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'physics_source_sha256':hashlib.sha256(Path(physics.__file__).read_bytes()).hexdigest(),
 'scope':'Pretraining sensitivity diagnostic, not an optimized candidate or independent candidate acceptance dataset',
 'scientific_accuracy_certified':False}
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
