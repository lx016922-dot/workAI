"""Subnormal numbers flushed to zero inside the layers: what changes and what does not."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numba import njit, prange
from spectral3d.feature_mesh import make_mesh
from graphnet.model import load
from graphnet.inputs import network_scene
from graphnet.rollout import read_record, rollout
from graphnet.subnormal import AVAILABLE, enter, flushing, leave
from graphnet.test_graphnet import plate
from graphnet.test_parallel import same

TINY = np.finfo(float).tiny   # the smallest normal double, 2.2e-308


@njit(parallel=True)
def halves(values, flush):
    """Half of every value, each in its own task; and whether the task saw the flushing mode."""
    out = np.empty(len(values)); seen = np.zeros(len(values), dtype=np.bool_)
    for i in prange(len(values)):
        saved = enter(flush); seen[i] = flushing(); out[i] = values[i] * .5; leave(saved)
    return out, seen


def subnormal(values):
    values = np.abs(values); return (values > 0) & (values < TINY)


class Mode(unittest.TestCase):
    def test_tasks_flush_only_when_asked_and_leave_the_thread_as_it_was(self):
        values = np.array([4 * TINY, 2 * TINY, TINY, 1e-310, 1., -TINY] * 16)
        kept, seen = halves(values, False); self.assertFalse(seen.any()); self.assertTrue(np.array_equal(kept, values * .5)); self.assertTrue(subnormal(kept).any())
        flushed, seen = halves(values, True); self.assertFalse(flushing())   # the calling thread is back in its own mode
        if not AVAILABLE:
            self.assertFalse(seen.any()); self.assertTrue(np.array_equal(flushed, kept)); return
        self.assertTrue(seen.all()); self.assertFalse(subnormal(flushed).any())
        # A result under the smallest normal number is an exact zero, a subnormal operand counts as zero, everything else is untouched.
        self.assertTrue(np.array_equal(flushed[:6], [2 * TINY, TINY, 0., 0., .5, 0.])); self.assertTrue(np.array_equal(halves(values, False)[0], kept))


class Records(unittest.TestCase):
    def test_a_flushed_record_differs_only_in_the_numerical_tail_ahead_of_the_wave(self):
        # A plate much longer than the wave travels during the record: most of it is ahead of the wave, where the displacement
        # falls through the subnormal range before it grows.
        model = load(3, long_record=True); raw = plate(length=220., duration=8.5, v2=True); raw['source'].update(delay_us=0.); raw['acquisition']['sample_step_us'] = .1
        scene = network_scene(raw, 3, model['material']); mesh = make_mesh(scene, 3, 1.6); settings = dict(time_plan='auto', controls=('murnaghan',))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); kept = rollout(scene, mesh, model, root / 'kept', {'t': 1}, **settings); a = read_record(root / 'kept')
            self.assertFalse(kept['flush_subnormals']); self.assertTrue(subnormal(a['murnaghan']).any())   # the case is there without flushing
            b = None
            for spread in (False, True):   # one thread per control, and every control spread over the threads: the same bits
                m = rollout(scene, mesh, model, root / f'flush{spread}', {'t': 1}, spread=spread, flush=True, **settings); record = read_record(root / f'flush{spread}')
                self.assertEqual(m['flush_subnormals'], AVAILABLE); self.assertTrue(b is None or same(b, record)); b = record
            if not AVAILABLE:
                self.assertTrue(same(a, b)); self.assertEqual(m['fingerprint'], kept['fingerprint']); return
            self.assertNotEqual(m['fingerprint'], kept['fingerprint'])
            with self.assertRaises(ValueError): rollout(scene, mesh, model, root / 'flushTrue', {'t': 1}, **settings)   # a record keeps the arithmetic it started with
            x, y = a['murnaghan'], b['murnaghan']; self.assertFalse(subnormal(y).any()); differ = x != y; self.assertTrue(differ.any())
            # What differs is the rounding tail far ahead of the wave: hundreds of orders of magnitude under the signal.
            self.assertGreater(np.abs(y).max(), 1e-9); self.assertLess(np.abs(x - y).max(), 1e-200)
            self.assertLess(max(np.abs(x[differ]).max(), np.abs(y[differ]).max()), 1e-200)
            self.assertTrue(np.array_equal(x[-1], y[-1]))   # and it is gone once the wave has passed its range
            for key in ('probes', 'energies'): self.assertTrue(np.allclose(a[key], b[key], rtol=1e-13, atol=1e-200), key)


if __name__ == '__main__': unittest.main()
