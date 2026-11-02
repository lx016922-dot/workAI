"""Independent closed-form check of the COMSOL modal postprocessor."""
import unittest
import numpy as np
from tools.comsol_modal_response import oscillator_response


class ModalForcingTest(unittest.TestCase):
    def test_finite_sine_burst_including_resonance_and_free_tail(self):
        scene={'signal':{'kind':'rectangular','end_frequency_mhz':.2},
               'source':{'frequency_mhz':.2,'cycles':3,'delay_us':.1,'phase_deg':0}}
        times=np.arange(0,5000.001,.2)
        frequency=np.array([.12e6,.2e6,.7e6]);omega=2*np.pi*frequency
        drive=2*np.pi*.2e6;end=15e-6
        local=np.maximum(0,times*1e-6-.1e-6);forced=np.minimum(local,end)
        expected=np.empty((len(times),len(omega)))
        for j,w in enumerate(omega):
            def exact(t):
                if w==drive:
                    q=np.sin(w*t)/(2*w*w)-t*np.cos(w*t)/(2*w)
                    v=t*np.sin(w*t)/2
                else:
                    q=(np.sin(drive*t)-drive/w*np.sin(w*t))/(w*w-drive*drive)
                    v=drive*(np.cos(drive*t)-np.cos(w*t))/(w*w-drive*drive)
                return q,v
            expected[:,j]=exact(forced)[0]
            tail=local>end;q,v=exact(end);angle=w*(local[tail]-end)
            expected[tail,j]=q*np.cos(angle)+v/w*np.sin(angle)
        actual,_=oscillator_response(scene,frequency,times)
        relative=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        self.assertLess(relative,1e-10)
        self.assertTrue(np.array_equal(actual[times<.1],np.zeros_like(actual[times<.1])))


if __name__=='__main__':unittest.main()
