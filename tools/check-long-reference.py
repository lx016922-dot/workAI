"""Reproduce a higher-degree analytic probe reference without learning from it."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numba import set_num_threads
from graphwave.long_campaign import Driver, STATE
from graphwave.long_rollout import atomic, digest, save_arrays

parser = argparse.ArgumentParser()
parser.add_argument('--plan', type=Path, default=Path('research/long-validation/plan.json'))
parser.add_argument('--case', default='rectangular-01')
parser.add_argument('--degree', type=int, default=6)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--threads', type=int, default=3)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
case = next(c for c in plan['cases'] if c['id'] == args.case)
repo = Path(__file__).resolve().parents[1]
assert all(digest(repo / p) == expected for p, expected in plan['source_sha256'].items())
protocol = {'purpose':'independent higher-degree probe convergence diagnostic',
            'case_id':args.case, 'source_plan_sha256':digest(args.plan),
            'degree':args.degree, 'h_mm':case['h_mm'], 'duration_us':5000,
            'candidate_weights_read':False, 'threshold':.05, 'absolute_floor_m':1e-13}
args.output.mkdir(parents=True, exist_ok=True)
protocol_file = args.output / 'protocol.json'
if protocol_file.exists():
    assert json.loads(protocol_file.read_text()) == protocol, 'Changed diagnostic identity'
else:
    atomic(protocol_file, protocol)
set_num_threads(args.threads)
driver = Driver(case['scene'], args.degree, case['h_mm'], None, None, reference=True)
statefile = args.output / 'state.npz'
first = 0
if statefile.exists():
    with np.load(statefile, allow_pickle=False) as data:
        first = int(data['next_frame'])
        driver.state = {k:data[k] for k in STATE}
for at in range(first, 25001, 256):
    last = min(25000, at + 255)
    _, probes, energy = driver.advance(at, last)
    save_arrays(args.output / f'probes-{at:06d}.npz', reference=probes, energy=energy)
    save_arrays(statefile, next_frame=np.array(last + 1), **driver.state)
    atomic(args.output / 'progress.json', {'phase':'completed' if last == 25000 else 'running',
           'completed_time_us':last * .2, 'dt_us':driver.dt * 1e6, 'nodes':len(driver.mesh['points'])})
    print(json.dumps({'time_us':last * .2}), flush=True)
