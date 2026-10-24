"""Run one immutable case on an independent validation runner."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from numba import get_num_threads,set_num_threads
from graphwave.long_campaign import plan,run_case
from graphwave.long_rollout import atomic
parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/'research/long-validation'
args.input=root/'input.json';args.weights=root/'weights.npz';args.temporal=root/'time-coefficients.json'
args.output.mkdir(parents=True,exist_ok=True)
if not (args.output/'plan.json').exists():
    (args.output/'plan.json').write_bytes((root/'plan.json').read_bytes())
p=plan(args)
case=next(c for c in p['cases'] if c['id']==args.case)
set_num_threads(min(3,get_num_threads()))
with np.load(args.weights,allow_pickle=False) as data:weights=dict(data)
result=run_case(args,case,weights,json.loads(args.temporal.read_text()))
# Completion of a scientific comparison is distinct from passing its criterion.
atomic(args.output/'summary.json',{'completed_cases':1,'case':args.case,
       'all_comparisons_under_5_percent':result['all_comparisons_under_5_percent'],
       'accuracy_certified':result['accuracy_certified'],'result':result})
print(json.dumps({'case':args.case,'complete':True,'maximum_relative_l2':result['maximum_reported_relative_l2'],
                  'all_comparisons_under_5_percent':result['all_comparisons_under_5_percent']}),flush=True)
