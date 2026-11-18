"""Interruptions must preserve the actual LM trajectory and validation choice."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from coarsegraph.checkpoint import (atomic_json, load_checkpoint, operator_snapshot, restore_operators,
                                    save_checkpoint, training_identity)
from coarsegraph.operator import parameterization
from coarsegraph.train import argument_parser, train, training_source_hashes


def example_identity(P):
    config = {'degree': P.degree, 'h_mm': 2., 'cfl': .3, 'axes': 'shared',
              'increment_weight': 1., 'harmonic_weight': 1., 'stall': 0.,
              'source_sha256': {'graphnet/recurrence.py': 'original-forward'}}
    pairs = [{'key': 'small:frequency=0.3', 'split': 'train', 'parameters': {'frequency': .3},
              'scene': {'receivers': [{'name': 'a', 'point_mm': [1., 0., 0.]}]}, 'names': ['a'],
              'frequency_mhz': .3, 'times': np.array([0., 1., 2.]),
              'fine_N': np.array([[1.], [2.], [3.]]), 'fine_I': np.array([[.1], [.2], [.3]]),
              'label_sha256': {'linear': 'linear-result', 'murnaghan': 'nonlinear-result'},
              'input_sha256': {'linear': {'out/lines.npz': 'linear-lines', 'scene.json': 'linear-scene'}}}]
    splits = {'small': {'all': 'train'}}
    return training_identity(config, pairs, splits), config, pairs, splits


class SmallProblem:
    """Deterministic nonlinear least squares using real operator coordinates.

    This exercises the optimizer, line search and checkpoint without launching
    expensive wave simulations. Validation prefers the starting operator, so
    losing the historical best on resume is observable.
    """
    def __init__(self):
        self.parameterization = parameterization(3); self.evaluations = 0; self.seen = []
        self.target = np.linspace(.002, .01, self.parameterization.size)

    def residuals(self, thetas, pairs):
        if pairs not in ('train', 'validation'): raise AssertionError('test data entered optimizer')
        self.evaluations += len(thetas); self.seen.append(pairs)
        target = self.target if pairs == 'train' else np.zeros_like(self.target)
        return [np.r_[np.asarray(t) - target, 10 * (np.asarray(t) ** 2 - target ** 2)] for t in thetas]


class TrainingCheckpoint(unittest.TestCase):
    def setUp(self):
        self.P = parameterization(3)
        self.identity, self.config, self.pairs, self.splits = example_identity(self.P)

    def capture(self, directory, states, interrupt=None, **extra):
        def save(state):
            states.append(copy.deepcopy(state))
            save_checkpoint(Path(directory) / 'checkpoint.json', self.identity, state, self.P,
                            iterations_target=4, **extra)
            if state['iteration'] == interrupt: raise InterruptedError('simulated process interruption')
        return save

    def test_interrupted_run_matches_uninterrupted_optimizer_and_keeps_best(self):
        full = SmallProblem(); whole = []
        train(full, 'train', 'validation', 4, lambda row: None, stall=0., checkpoint=lambda s: whole.append(copy.deepcopy(s)))
        with tempfile.TemporaryDirectory() as tmp:
            interrupted = SmallProblem(); saved = []
            with self.assertRaises(InterruptedError):
                train(interrupted, 'train', 'validation', 4, lambda row: None, stall=0., checkpoint=self.capture(tmp, saved, interrupt=2))
            restored = load_checkpoint(Path(tmp) / 'checkpoint.json', self.identity, self.P, 4)
            self.assertFalse(restored['state']['recompute_residual'])
            self.assertEqual(restored['state']['mu'], saved[-1]['mu'])
            self.assertEqual(restored['state']['evaluations'], interrupted.evaluations)
            self.assertGreater(restored['state']['iteration'], restored['state']['best']['iteration'])
            resumed = SmallProblem(); continued = []
            best, history = train(resumed, 'train', 'validation', 4, lambda row: None, stall=0.,
                                  resume=restored['state'], checkpoint=self.capture(tmp, continued))
        self.assertEqual(history, whole[-1]['history'])
        self.assertEqual(resumed.evaluations, full.evaluations)
        np.testing.assert_array_equal(continued[-1]['current']['theta'], whole[-1]['current']['theta'])
        np.testing.assert_array_equal(continued[-1]['current']['residual'], whole[-1]['current']['residual'])
        np.testing.assert_array_equal(best['theta'], whole[-1]['best']['theta'])
        self.assertEqual(best['iteration'], 0)
        self.assertEqual(best['validation_loss'], 0.)

    def test_stall_count_survives_interruption_and_preserves_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = []
            with self.assertRaises(InterruptedError):
                train(SmallProblem(), 'train', 'validation', 4, lambda row: None, stall=1.1,
                      checkpoint=self.capture(tmp, saved, interrupt=1))
            state = load_checkpoint(Path(tmp) / 'checkpoint.json', self.identity, self.P, 4)['state']
            self.assertEqual(state['stalls'], 1)
            continued = []; problem = SmallProblem()
            train(problem, 'train', 'validation', 4, lambda row: None, stall=1.1,
                  resume=state, checkpoint=self.capture(tmp, continued))
            self.assertEqual(continued[-1]['iteration'], 2)
            self.assertEqual(continued[-1]['stalls'], 2)
            self.assertIsNotNone(continued[-1]['stopped'])
            stopped = load_checkpoint(Path(tmp) / 'checkpoint.json', self.identity, self.P, 5)['state']
            again = SmallProblem()
            train(again, 'train', 'validation', 5, lambda row: None, stall=1.1, resume=stopped)
            self.assertEqual(again.seen, [])
            self.assertEqual(again.evaluations, problem.evaluations)

    def test_changed_configuration_split_or_actual_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            train(SmallProblem(), 'train', 'validation', 0, lambda row: None, checkpoint=self.capture(tmp, []))
            changed = []
            for field, value in (('degree', 4), ('h_mm', 3.), ('cfl', .4), ('axes', 'per_axis'),
                                 ('increment_weight', 2.), ('harmonic_weight', .5), ('stall', .01)):
                config = dict(self.config, **{field: value})
                changed.append((field, training_identity(config, self.pairs, self.splits)))
            for field in ('times', 'fine_N', 'fine_I'):
                pairs = copy.deepcopy(self.pairs); pairs[0][field].flat[0] += .01
                changed.append((field, training_identity(self.config, pairs, self.splits)))
            for field in ('scene', 'names', 'input_sha256', 'split'):
                pairs = copy.deepcopy(self.pairs)
                if field == 'scene': pairs[0]['scene']['receivers'][0]['point_mm'][0] = 2.
                elif field == 'names': pairs[0]['names'] = ['another receiver']
                elif field == 'input_sha256': pairs[0]['input_sha256']['linear']['out/lines.npz'] = 'different-lines'
                else: pairs[0]['split'] = 'validation'
                changed.append((field, training_identity(self.config, pairs, self.splits)))
            changed.append(('split rules', training_identity(self.config, self.pairs, {'small': {'all': 'validation'}})))
            config = copy.deepcopy(self.config); config['source_sha256']['graphnet/recurrence.py'] = 'changed-forward'
            changed.append(('forward implementation', training_identity(config, self.pairs, self.splits)))
            for label, identity in changed:
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, '不兼容'):
                    load_checkpoint(Path(tmp) / 'checkpoint.json', identity, self.P, 4)

    def test_source_identity_includes_forward_input_mesh_and_metric_dependencies(self):
        sources = training_source_hashes()
        for name in ('graphnet/recurrence.py', 'graphnet/inputs.py', 'graphnet/predict.py', 'graphwave/inference.py',
                     'physics/receivers.py', 'spectral3d/mesh.py', 'solid/schema.py', 'dispersion/analysis.py'):
            self.assertIn(name, sources)
        self.assertFalse(any(Path(name).name.startswith('test_') for name in sources))

    def test_portable_operators_are_recovered_when_parameter_basis_changes(self):
        theta = np.linspace(.0001, .001, self.P.size); snapshot = operator_snapshot(self.P, theta)
        original = self.P

        class RotatedBasis:
            degree = original.degree; size = original.size
            Q = np.eye(original.size)[:, ::-1]
            def unpack(self, x): return original.unpack(self.Q @ x)
            def check_operators(self, *ops): return original.check_operators(*ops)
            def theta_from_operators(self, *ops): return self.Q.T @ original.theta_from_operators(*ops)

        changed = RotatedBasis()
        with patch('coarsegraph.run.parameterization', return_value=changed):
            restored, exact = restore_operators(snapshot, changed)
        self.assertFalse(exact)
        self.assertGreater(float(np.linalg.norm(restored - theta)), 1e-4)
        for actual, expected in zip(changed.unpack(restored), original.unpack(theta)):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_atomic_serialization_failure_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.json'; atomic_json(path, {'valid': 1})
            original = path.read_bytes()
            with self.assertRaises(ValueError): atomic_json(path, {'invalid': float('nan')})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_frozen_resume_only_finishes_same_evaluation_and_completed_run_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            states = []
            train(SmallProblem(), 'train', 'validation', 1, lambda row: None, checkpoint=self.capture(tmp, states))
            path = Path(tmp) / 'checkpoint.json'
            save_checkpoint(path, self.identity, states[-1], self.P, phase='frozen', iterations_target=1)
            self.assertEqual(load_checkpoint(path, self.identity, self.P, 1)['phase'], 'frozen')
            with self.assertRaisesRegex(ValueError, '已冻结'): load_checkpoint(path, self.identity, self.P, 2)
            save_checkpoint(path, self.identity, states[-1], self.P, phase='completed', iterations_target=1)
            with self.assertRaisesRegex(ValueError, '--init'): load_checkpoint(path, self.identity, self.P, 2)

    def test_initial_model_snapshot_is_self_contained_and_init_resume_are_exclusive(self):
        initial = {'path': '/no/longer/available/model.json', 'sha256': 'source-hash',
                   'snapshot': operator_snapshot(self.P, np.zeros(self.P.size)), 'evaluation_scope': 'test'}
        with tempfile.TemporaryDirectory() as tmp:
            train(SmallProblem(), 'train', 'validation', 0, lambda row: None,
                  checkpoint=self.capture(tmp, [], initial_model=initial))
            saved = load_checkpoint(Path(tmp) / 'checkpoint.json', self.identity, self.P, 4)
            self.assertEqual(saved['initial_model'], initial)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            argument_parser().parse_args(['--labels', 'data', '--splits', 'splits.json', '--output', 'out',
                                          '--init', 'model.json', '--resume'])


if __name__ == '__main__': unittest.main()
