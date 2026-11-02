"""Acceptance checks must reject incomplete or physically inconsistent records."""
import importlib.util
import copy
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


class RecordedMetricReplay(unittest.TestCase):
    def setUp(self):
        self.metrics = {'passed': True, 'maximum_relative_l2': 1.008944e-11,
                        'waveform': [{'z': {'passed': True, 'relative_l2': 1.008944e-11,
                                            'error_rms_m': 1.008944e-23,
                                            'reference_rms_m': 1e-12}}]}

    def test_nearly_zero_reference_residual_can_be_reproduced(self):
        recomputed = copy.deepcopy(self.metrics)
        recomputed['maximum_relative_l2'] = 6.204634e-12
        recomputed['waveform'][0]['z']['relative_l2'] = 6.204634e-12
        recomputed['waveform'][0]['z']['error_rms_m'] = 6.204634e-24
        audit.check_recorded_metrics(self.metrics, recomputed)

    def test_changed_component_is_rejected_even_with_same_maximum(self):
        recomputed = copy.deepcopy(self.metrics)
        recomputed['waveform'][0]['z']['relative_l2'] = .001
        with self.assertRaises(AssertionError):
            audit.check_recorded_metrics(self.metrics, recomputed)

    def test_unchanged_aggregate_cannot_hide_changed_decision_or_floor(self):
        for key, value in [('passed', False), ('relative_l2', None)]:
            recomputed = copy.deepcopy(self.metrics)
            recomputed['waveform'][0]['z'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.check_recorded_metrics(self.metrics, recomputed)

    def test_missing_band_and_nonfinite_diagnostic_are_rejected(self):
        for value in [[], [{'z': {'passed': True, 'relative_l2': float('nan'),
                                 'error_rms_m': 1e-23, 'reference_rms_m': 1e-12}}]]:
            recomputed = copy.deepcopy(self.metrics)
            recomputed['waveform'] = value
            with self.assertRaises(ValueError):
                audit.check_recorded_metrics(self.metrics, recomputed)


if __name__ == '__main__':
    unittest.main()
