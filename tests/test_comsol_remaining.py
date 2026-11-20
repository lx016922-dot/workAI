"""The comparison of the remaining P1 COMSOL runs (stvk, dt2, mesh2) with the baselines and the software's predictions."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
P1 = REPO / 'research' / 'comsol-p1-20261024'
spec = importlib.util.spec_from_file_location('compare_remaining', P1 / 'compare_remaining.py'); tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
from coarsegraph.comsol import export_csv, load_case

CID = 'A-f0.40-T1.6'


def fields():
    """Synthetic displacements frames × 13 receivers × 3: a linear burst at f and two small increments at 2f."""
    case = load_case(CID); out = case['output']; t = out['t_start_us'] + np.arange(out['samples']) * out['step_us']; f = case['sources'][0]['frequency_mhz']
    d = np.array([r['direction'] for r in out['receivers']], dtype=float); delay = 2. + .4 * np.arange(len(d)); gain = 1. + .05 * np.arange(len(d))
    window = lambda centre: np.exp(-((t[:, None] - 12. - centre[None, :]) / 5.) ** 2)
    wave = lambda order, scale, phase: (scale * gain[None, :] * window(delay) * np.sin(2 * np.pi * order * f * (t[:, None] - delay[None, :]) + phase))[:, :, None] * d[None, :, :]
    return case, t, wave(1, 1e-9, 0.), wave(2, 2e-13, .3), wave(2, 5e-13, 1.1)


class Remaining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name); self.delivery, self.baselines, self.predictions = root / 'delivery', root / 'baselines', root / 'predictions'
        for d in (self.delivery, self.baselines, self.predictions): d.mkdir()
        self.case, self.t, self.L, self.G, self.M = fields()

    def tearDown(self): self.tmp.cleanup()

    def write(self, directory, name, U): export_csv(self.case, self.t, U, directory / f'{CID}_{name}.csv')

    def test_the_two_increments_of_a_stvk_run_are_compared_separately(self):
        L, G, M = self.L, self.G, self.M
        self.write(self.baselines, 'linear', L); self.write(self.baselines, 'murnaghan', L + G + M); self.write(self.delivery, 'stvk', L + G)
        for law, U in (('linear', L), ('stvk', L + 1.02 * G), ('murnaghan', L + 1.02 * G + .99 * M)): self.write(self.predictions, law, U)
        report = tool.analyse(self.delivery, self.baselines, self.predictions); row, = report['rows']; tier = row['tier2_stvk']
        self.assertEqual(row['delivered'], [f'{CID}_stvk.csv']); self.assertNotIn('tier3_dt2', row)
        self.assertAlmostEqual(tier['geometric_increment']['worst_a2'], .02, places=6); self.assertAlmostEqual(tier['geometric_increment']['a2_vector_relative'], .02, places=6)
        self.assertAlmostEqual(tier['material_increment']['worst_a2'], .01, places=6); self.assertTrue(tier['passed'])
        self.assertLess(tier['stvk_record']['worst_waveform_l2'], 1e-5); self.assertGreater(tier['comsol_shares_l2']['material'], tier['comsol_shares_l2']['geometric'])
        self.assertEqual(set(report['files_sha256']), {f'{CID}_linear.csv', f'{CID}_stvk.csv', f'{CID}_murnaghan.csv', *(f'prediction {CID}_{law}.csv' for law in ('linear', 'stvk', 'murnaghan'))})
        self.assertIn('within the 5% gate', tool.summary(report)[0])
        # a geometric increment 8 % off is outside the gate
        self.write(self.predictions, 'stvk', L + 1.08 * G); bad = tool.analyse(self.delivery, self.baselines, self.predictions)['rows'][0]['tier2_stvk']
        self.assertFalse(bad['geometric_increment']['passed']); self.assertFalse(bad['passed'])

    def test_a_refined_comsol_run_shows_whose_error_the_difference_was(self):
        L, I = self.L, self.G + self.M   # the software has the increment right; COMSOL's base run has it 3 % too large and its dt2 run 0.75 % too large
        self.write(self.baselines, 'linear', L); self.write(self.baselines, 'murnaghan', L + 1.03 * I)
        self.write(self.delivery, 'linear_dt2', L); self.write(self.delivery, 'murnaghan_dt2', L + 1.0075 * I); self.write(self.delivery, 'linear_mesh2', L * (1 + 1e-4))
        for law, U in (('linear', L), ('stvk', L + self.G), ('murnaghan', L + I)): self.write(self.predictions, law, U)
        row, = tool.analyse(self.delivery, self.baselines, self.predictions)['rows']; dt2 = row['tier3_dt2']
        self.assertAlmostEqual(dt2['comsol_base_against_refined']['worst_increment_a2'], 1.03 / 1.0075 - 1, places=6)
        self.assertAlmostEqual(dt2['software_against_base']['worst_increment_a2'], 1 - 1 / 1.03, places=6); self.assertAlmostEqual(dt2['software_against_refined']['worst_increment_a2'], 1 - 1 / 1.0075, places=6)
        self.assertAlmostEqual(dt2['gap_ratio_worst_increment_a2'], (1 - 1 / 1.0075) / (1 - 1 / 1.03), places=4); self.assertLess(dt2['gap_ratio_worst_increment_a2'], .3)
        self.assertLess(dt2['comsol_linear_base_against_refined']['l2'], 1e-12)
        mesh2 = row['tier3_mesh2']; self.assertEqual(set(mesh2), {'comsol_linear_base_against_refined', 'software_linear'})   # only the linear run of this variant was delivered
        self.assertAlmostEqual(mesh2['comsol_linear_base_against_refined']['l2'], 1e-4 / (1 + 1e-4), places=9); self.assertNotIn('tier2_stvk', row)
        lines = tool.summary({'gate': .05, 'rows': [row]}); self.assertEqual(len(lines), 2); self.assertIn('ratio 0.2', lines[0]); self.assertIn('linear only', lines[1])

    def test_a_delivery_without_new_runs_says_so(self):
        self.write(self.delivery, 'linear', self.L)
        report = tool.analyse(self.delivery, self.baselines, self.predictions); self.assertEqual([set(r) for r in report['rows']], [{'case_id', 'delivered'}]); self.assertEqual(tool.summary(report), [])


if __name__ == '__main__': unittest.main()
