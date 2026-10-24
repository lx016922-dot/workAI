import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from coarsegraph import comsol

REPO = Path(__file__).resolve().parents[1]
CID = 'A-f0.40-T1.6'


def synthetic(case):
    t = np.arange(case['output']['samples']) * case['output']['step_us']; R = len(case['output']['receivers'])
    phase = np.arange(R)[None, :, None] * .3 + np.arange(3)[None, None, :]
    envelope = np.exp(-.5 * ((t[:, None, None] - 14) / 4) ** 2)
    linear = 5e-9 * envelope * np.sin(2 * np.pi * .4 * t[:, None, None] + phase)
    return t, linear, linear + 5e-13 * envelope * np.sin(4 * np.pi * .4 * t[:, None, None] + 2 * phase)


def deliver(directory, case, t, linear, nonlinear, metadata=True):
    comsol.export_csv(case, t, linear, directory / f'{CID}_linear.csv'); comsol.export_csv(case, t, nonlinear, directory / f'{CID}_murnaghan.csv', comment='synthetic test data')
    if metadata: (directory / f'{CID}.json').write_text((REPO / 'docs' / 'comsol' / 'metadata-template.json').read_text(encoding='utf-8'), encoding='utf-8')


class ComsolDataTests(unittest.TestCase):
    def test_case_files_cover_every_label_and_match_the_header_template(self):
        cases = sorted(p.stem for p in (REPO / 'docs' / 'comsol' / 'cases').glob('*.json')); self.assertEqual(len(cases), 17)
        header = (REPO / 'docs' / 'comsol' / 'header-template.csv').read_text(encoding='utf-8').strip().split(',')
        for cid in cases:
            case = comsol.load_case(cid); self.assertEqual(comsol.columns(case), header); self.assertEqual(case['output']['samples'], 601)
            self.assertTrue(all(a['alpha_max_per_s'] > 2.2e7 for a in case['absorbing']))
        self.assertEqual(set(comsol.PRIORITY['P1']) | set(comsol.PRIORITY['P2']) <= set(cases), True)

    def test_check_import_and_compare_round_trip(self):
        case = comsol.load_case(CID); t, linear, nonlinear = synthetic(case)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; data.mkdir(); deliver(data, case, t, linear, nonlinear)
            report, ok = comsol.check(data); self.assertTrue(ok, report['errors']); self.assertEqual(len(report['files']), 2)
            self.assertTrue(all(r['significant_digits'] >= 10 for r in report['files']))
            self.assertAlmostEqual(report['files'][1]['max_increment_over_linear'], 1e-4, delta=5e-5)
            counts = comsol.import_data(data, root / 'imported'); self.assertEqual(counts, {'base': 2})
            record = json.loads((root / 'imported' / 'base' / 'cases' / f'{CID}-murnaghan' / 'out' / 'result.json').read_text())
            self.assertEqual(record['engine'], 'comsol_import'); self.assertTrue(np.allclose(record['receivers'][0]['u_z_m'], nonlinear[:, 0, 2], rtol=1e-10, atol=1e-22))
            labels = root / 'labels-a-top-gaussian'; comsol.import_data(data, labels); (labels / 'base').rename(root / 'tmp-base'); (root / 'tmp-base').rename(root / 'labels'); labels.rmdir()
            rows = comsol.compare(root / 'imported', [root / 'labels'])
            self.assertEqual(rows[0]['case_id'], CID); m = rows[0]['spectral_vs_comsol']; self.assertTrue(m['passed']); self.assertLess(m['record_l2'], 1e-12)

    def test_check_rejects_wrong_files_and_flags_units_and_precision(self):
        case = comsol.load_case(CID); t, linear, nonlinear = synthetic(case)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seconds = root / 'seconds'; seconds.mkdir(); deliver(seconds, case, t, linear, nonlinear)
            for name in (f'{CID}_linear.csv', f'{CID}_murnaghan.csv'):
                path = seconds / name; lines = path.read_text().splitlines(); header = [l for l in lines if not l.startswith('%')][0]
                body = [l for l in lines if not l.startswith('%')][1:]
                path.write_text('\n'.join(['time_s' + header[len('time_us'):]] + [f'{float(r.split(",")[0]) * 1e-6:.12e},' + r.split(',', 1)[1] for r in body]) + '\n')
            self.assertTrue(comsol.check(seconds)[1])
            bad = root / 'bad'; bad.mkdir(); deliver(bad, case, t, linear, nonlinear, metadata=False)
            (bad / f'{CID}_murnaghan.csv').write_text('time_us,ux\n0,0\n'); report, ok = comsol.check(bad)
            self.assertFalse(ok); self.assertTrue(any('表头' in e for e in report['errors'])); self.assertTrue(any('元数据' in e for e in report['errors']))
            mm = root / 'mm'; mm.mkdir(); deliver(mm, case, t, linear * 1e3, nonlinear * 1e3); report, ok = comsol.check(mm)
            self.assertTrue(any('毫米' in w for w in report['warnings']))
            coarse = root / 'coarse'; coarse.mkdir(); deliver(coarse, case, t, linear, nonlinear)
            for name in (f'{CID}_linear.csv', f'{CID}_murnaghan.csv'):
                path = coarse / name; lines = [l for l in path.read_text().splitlines() if not l.startswith('%')]
                path.write_text('\n'.join([lines[0]] + [','.join([r.split(',')[0]] + [f'{float(v):.5g}' for v in r.split(',')[1:]]) for r in lines[1:]]) + '\n')
            report, ok = comsol.check(coarse); self.assertFalse(ok); self.assertTrue(any('有效数字' in e for e in report['errors']))
            wrong = root / 'wrong'; wrong.mkdir(); deliver(wrong, case, t[:-1], linear[:-1], nonlinear[:-1]); self.assertFalse(comsol.check(wrong)[1])


if __name__ == '__main__': unittest.main()
