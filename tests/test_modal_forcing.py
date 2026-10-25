"""Closed-form displacement/velocity validation of candidate forcing over 5 ms."""
import unittest
import numpy as np

from graphwave.modal_forcing import driven_states


class DrivenStatesTest(unittest.TestCase):
    def test_rectangular_burst_resonance_high_frequency_and_free_tail(self):
        scene={'signal':{'kind':'rectangular','end_frequency_mhz':.2},
               'source':{'frequency_mhz':.2,'cycles':5,'delay_us':.13,'phase_deg':0}}
        times=np.r_[-.2,np.arange(0,5000.001,.2)]
        frequency=np.array([160801.6,200000.,3700000.]);omega=2*np.pi*frequency
        drive=2*np.pi*.2e6;end=25e-6;local=times*1e-6-.13e-6
        position,velocity,details=driven_states(scene,frequency,times)
        for mode,w in enumerate(omega):
            def exact(t):
                if w==drive:return (np.sin(w*t)/(2*w*w)-t*np.cos(w*t)/(2*w),t*np.sin(w*t)/2)
                return ((np.sin(drive*t)-drive/w*np.sin(w*t))/(w*w-drive*drive),
                        drive*(np.cos(drive*t)-np.cos(w*t))/(w*w-drive*drive))
            q,v=exact(np.clip(local,0,end));tail=local>end;q0,v0=exact(end);angle=w*(local[tail]-end)
            q[tail]=q0*np.cos(angle)+v0/w*np.sin(angle)
            v[tail]=v0*np.cos(angle)-w*q0*np.sin(angle)
            for actual,expected in [(position[:,mode],q),(velocity[:,mode],v)]:
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-9)
        np.testing.assert_array_equal(position[local<0],0.)
        np.testing.assert_array_equal(velocity[local<0],0.)
        self.assertFalse(details['comsol_arrays_used'])


if __name__=='__main__':unittest.main()
