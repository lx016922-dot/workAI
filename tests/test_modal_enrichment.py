"""Physical invariants of conservative nonlinear basis training."""
import unittest
import numpy as np
from graphwave.modal_enrichment import fit_space, diagonalize, static_metrics


class EnrichmentPhysics(unittest.TestCase):
    def test_enrichment_recovers_missing_coupled_static_response(self):
        rng = np.random.default_rng(19)
        rotation = np.linalg.qr(rng.normal(size=(18,18)))[0]
        values = np.arange(1,19,dtype=float)**2
        tangent = rotation @ np.diag(values) @ rotation.T
        base = rotation[:,:3]
        complement = rotation[:,3:7]
        rhs = tangent @ (complement @ rng.normal(size=(4,24)) + base @ rng.normal(size=(3,24)))
        exact = np.linalg.solve(tangent,rhs)
        space,_ = fit_space(base,exact,4)
        omega,basis,_ = diagonalize(space,tangent)
        metrics = static_metrics(basis,omega,tangent,rhs,exact,np.tile(np.arange(3),6))
        self.assertLess(max(max(v) for v in metrics.values()),1e-12)
        q = rng.normal(size=7)
        self.assertAlmostEqual(float((basis@q)@tangent@(basis@q)),float(np.sum((omega*q)**2)),places=10)

    def test_rank_deficient_training_cannot_silently_invent_directions(self):
        base = np.eye(12)[:,:2]
        states = np.tile(np.eye(12)[:,3:4],(1,8))
        with self.assertRaises(ValueError):
            fit_space(base,states,3)


if __name__ == '__main__':
    unittest.main()
