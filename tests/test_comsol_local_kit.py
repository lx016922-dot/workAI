"""The local COMSOL run kit reproduces the archived requests and round-trips results into the delivery contract."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
P1 = REPO / 'research' / 'comsol-p1-20261024'
spec = importlib.util.spec_from_file_location('local_kit', P1 / 'local_kit.py'); kit = importlib.util.module_from_spec(spec); spec.loader.exec_module(kit)


class LocalKit(unittest.TestCase):
    def test_build_matches_archived_request_and_collect_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'kit'
            kit.build(type('A', (), {'output': str(out), 'tiers': []})())
            jobs = json.loads((out / 'jobs.json').read_text(encoding='utf-8'))
            self.assertEqual(len(jobs), 22)
            self.assertEqual([j['tier'] for j in jobs], sorted(j['tier'] for j in jobs))
            check = out / 'runs' / '00-material-check'
            ours, archived = json.loads((check / 'request.json').read_text()), json.loads((P1 / 'evidence/material-check/request.json').read_text(encoding='utf-8-sig'))
            for key in ('job_id', 'source_sha256', 'scene_sha256', 'parameters_sha256', 'scene_json', 'parameters_json'): self.assertEqual(ours[key], archived[key])
            props = lambda p: {k: float(v) if k not in ('job_id', 'schema', 'scenario', 'boundary') and not k.endswith('sha256') else v for k, v in kit.properties(p).items()}
            self.assertEqual(props(check / 'case.properties'), props(P1 / 'evidence/material-check/case.properties'))
            # finished jobs: archived self-check and the archived A linear record written in the model's long format
            for name in ('coarse.csv', 'completed.properties'): (check / name).write_bytes((P1 / 'evidence/material-check' / name).read_bytes())
            run = out / 'runs' / '04-A-f0.40-T1.6-linear'; identity = json.loads((run / 'identity.json').read_text())
            U = np.loadtxt(P1 / 'recovery-rebuild-20261029/data/A-f0.40-T1.6_linear.csv', delimiter=',', skiprows=1)
            rows = [f'0,{r},{i * 0.05e-6:.17g},{U[i, 1 + 3 * r]:.17g},{U[i, 2 + 3 * r]:.17g},{U[i, 3 + 3 * r]:.17g}' for i in range(601) for r in range(13)]
            (run / 'coarse.csv').write_text('law,rx,time_s,u_m,v_m,w_m\n' + '\n'.join(rows) + '\n')
            (run / 'coarse.field.csv').write_text('law,time_s,max_green_strain,min_J\n' + '\n'.join(f'0,{i * 0.05e-6:.17g},1e-5,0.99999' for i in range(601)) + '\n')
            (run / 'completed.properties').write_text(''.join(f'{k}={identity[k]}\n' for k in ('job_id', 'source_sha256', 'scene_sha256', 'parameters_sha256')) + 'analysis=transient\ndofs=22167\nvolume_hexahedra=360\n')
            delivery = Path(tmp) / 'delivery'
            kit.collect(type('A', (), {'runs': str(out / 'runs'), 'output': str(delivery)})())
            self.assertEqual((delivery / 'A-f0.40-T1.6_linear.csv').read_text().splitlines()[1:],
                             (P1 / 'recovery-rebuild-20261029/data/A-f0.40-T1.6_linear.csv').read_text().splitlines()[1:])
            meta = json.loads((delivery / 'A-f0.40-T1.6.json').read_text())
            self.assertEqual(meta['dofs'], 22167); self.assertTrue(meta['material_check']['passed'])
            # a changed model source is refused
            (run / 'UltraResearch.java').write_text('// edited\n')
            with self.assertRaises(ValueError): kit.checked(run)


if __name__ == '__main__': unittest.main()
