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
    def test_build_reproduces_the_archived_serendipity_kit_and_collect_round_trips(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'kit'
            kit.build(type('A', (), {'output': str(out), 'tiers': []})())
            jobs = json.loads((out / 'jobs.json').read_text(encoding='utf-8'))
            self.assertEqual(len(jobs), 22)
            self.assertEqual([j['tier'] for j in jobs], sorted(j['tier'] for j in jobs))
            # every job file is byte-identical to the corrected kit the completed runs used
            with zipfile.ZipFile(P1 / 'serendipity-54e90010/P1_SERP_20261102.zip') as z:
                archived = {n[len('p1-kit/'):]: z.read(n) for n in z.namelist() if n.startswith('p1-kit/runs/') and not n.endswith('/')}
                self.assertEqual(z.read('p1-kit/jobs.json'), (out / 'jobs.json').read_bytes())
            self.assertEqual(len(archived), 132)
            for name, data in archived.items(): self.assertEqual((out / name).read_bytes(), data, name)
            # finished jobs: the delivered self-check and A linear run, written as the model writes them
            delivered = P1 / 'serendipity-baselines-20261109'
            check = out / 'runs' / '00-material-check'
            (check / 'coarse.csv').write_bytes((delivered / 'evidence/material-check.csv').read_bytes())
            (check / 'completed.properties').write_bytes((delivered / 'evidence/receipts/00-material-check.properties').read_bytes())
            run = out / 'runs' / '04-A-f0.40-T1.6-linear'
            U = np.loadtxt(delivered / 'data/A-f0.40-T1.6_linear.csv', delimiter=',', skiprows=1)
            rows = [f'0,{r},{i * 0.05e-6:.17g},{U[i, 1 + 3 * r]:.17g},{U[i, 2 + 3 * r]:.17g},{U[i, 3 + 3 * r]:.17g}' for i in range(601) for r in range(13)]
            (run / 'coarse.csv').write_text('law,rx,time_s,u_m,v_m,w_m\n' + '\n'.join(rows) + '\n')
            (run / 'coarse.field.csv').write_bytes((delivered / 'data/field/A-f0.40-T1.6_linear.field.csv').read_bytes())
            (run / 'completed.properties').write_bytes((delivered / 'evidence/receipts/04-A-f0.40-T1.6-linear.properties').read_bytes())
            delivery = Path(tmp) / 'delivery'
            kit.collect(type('A', (), {'runs': str(out / 'runs'), 'output': str(delivery)})())
            self.assertTrue(np.array_equal(np.loadtxt(delivery / 'A-f0.40-T1.6_linear.csv', delimiter=',', skiprows=1), U))
            meta = json.loads((delivery / 'A-f0.40-T1.6.json').read_text())
            self.assertEqual(meta['dofs'], 22167); self.assertTrue(meta['material_check']['passed'])
            # a changed model source is refused
            (run / 'UltraResearch.java').write_text('// edited\n')
            with self.assertRaises(ValueError): kit.checked(run)


if __name__ == '__main__': unittest.main()
