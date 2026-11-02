import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('refinements', Path(__file__).resolve().parents[1] / 'tools/check-modal-refinements.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RefinementAdmission(unittest.TestCase):
    def fixture(self, folder, times=None, basis='same-basis', modes=8, step=.05):
        folder.mkdir()
        scene = {'acquisition': {'duration_us': .2, 'sample_step_us': .1}, 'receivers': [{}]}
        times = np.array([0., .1, .2] if times is None else times)
        traces = np.zeros((3, len(times), 1, 3))
        energy = np.zeros((3, len(times), 3))
        np.savez(folder / 'prediction.npz', times_us=times, traces=traces, energy=energy,
                 coordinates=np.zeros((3, len(times), modes)))
        identity = {'case': {'scene': scene}, 'model_sha256': 'model', 'basis_sha256': basis,
                    'weights_sha256': 'weights', 'source_sha256': {}, 'plan_sha256': 'plan',
                    'candidate_mode_limit': modes, 'step_us': step, 'symmetry': {'retained_modes': modes}}
        prediction = {'completed': True, 'controls': ['linear', 'stvk', 'full'],
                      'prediction_sha256': module.digest(folder / 'prediction.npz'),
                      'energy_relative_error': [0., 0., 0.], 'mode_limit': modes, 'step_us': step,
                      'active_modes': modes, 'frames': len(times), 'duration_us': .2}
        metrics = {'prediction': prediction, 'native_reference_job_id': 'synthetic-unit-test',
                   'native_comsol_time_dependent_solve': True}
        for name, data in [('identity.json', identity), ('prediction.json', prediction), ('metrics.json', metrics)]:
            (folder / name).write_text(json.dumps(data))
        return folder

    def test_full_saved_record_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            folder = self.fixture(Path(root) / 'case')
            self.assertEqual(module.load_case(folder)[3].shape, (3, 3, 1, 3))

    def test_missing_middle_frame_rejected_even_with_matching_file_hash(self):
        with tempfile.TemporaryDirectory() as root:
            folder = self.fixture(Path(root) / 'case', times=[0., .2])
            with self.assertRaises(AssertionError):
                module.load_case(folder)

    def test_changed_energy_history_cannot_reuse_claimed_balance(self):
        with tempfile.TemporaryDirectory() as root:
            folder = self.fixture(Path(root) / 'case')
            with np.load(folder / 'prediction.npz') as archive:
                values = dict(archive)
            values['energy'][0, 1, 0] = 1.
            np.savez(folder / 'prediction.npz', **values)
            prediction = json.loads((folder / 'prediction.json').read_text())
            prediction['prediction_sha256'] = module.digest(folder / 'prediction.npz')
            (folder / 'prediction.json').write_text(json.dumps(prediction))
            metrics = json.loads((folder / 'metrics.json').read_text())
            metrics['prediction'] = prediction
            (folder / 'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaises(AssertionError):
                module.load_case(folder)

    def test_changed_basis_cannot_be_called_a_same_basis_refinement(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            low = self.fixture(root / 'low')
            high = self.fixture(root / 'high', basis='changed', modes=10)
            fine = self.fixture(root / 'fine', basis='changed', modes=10, step=.025)
            with patch('sys.argv', ['check', '--lower-modes', str(low), '--higher-modes', str(high),
                                   '--finer-time', str(fine), '--output', str(root / 'output')]):
                with self.assertRaisesRegex(ValueError, 'basis_sha256'):
                    module.main()


if __name__ == '__main__':
    unittest.main()
