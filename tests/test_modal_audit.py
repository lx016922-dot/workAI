"""Acceptance checks must reject incomplete or physically inconsistent records."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('modal_audit', Path(__file__).resolve().parents[1] / 'tools/audit-modal-matrix.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SavedTrajectoryAdmission(unittest.TestCase):
    def setUp(self):
        self.scene = {'acquisition': {'duration_us': 5000, 'sample_step_us': .2}, 'receivers': [{}]}
        self.times = np.arange(25001) * .2
        self.traces = np.zeros((3, 25001, 1, 3))
        self.energy = np.zeros((3, 25001, 3))
        self.coordinates = np.zeros((3, 25001, 8))

    def test_complete_conservative_record(self):
        self.energy[:, 1:, 0] = 3e-9
        self.energy[:, 1:, 1] = 2e-9
        self.energy[:, 1:, 2] = 5e-9
        self.assertLess(audit.check_arrays(self.times, self.traces, self.energy, self.coordinates, self.scene, 8).max(), 1e-14)

    def test_short_repeated_or_shifted_record_rejected(self):
        for times in [self.times[:-1], self.times + .2, np.r_[self.times[:-1], self.times[-2]]]:
            with self.subTest(length=len(times)), self.assertRaises(ValueError):
                audit.check_arrays(times, self.traces, self.energy, self.coordinates, self.scene, 8)

    def test_unbalanced_negative_or_nonfinite_record_rejected(self):
        for value in [-1e-9, 1e-9, np.nan]:
            self.energy[1, -1, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.check_arrays(self.times, self.traces, self.energy, self.coordinates, self.scene, 8)

    def test_missing_control_rejected(self):
        with self.assertRaises(ValueError):
            audit.check_arrays(self.times, self.traces[:2], self.energy, self.coordinates, self.scene, 8)


if __name__ == '__main__':
    unittest.main()
