import unittest
import numpy as np
from tests.test_modal_sensitivity import ModalSensitivityTest
from graphwave import modal_calibration as dense
from graphwave import modal_sparse as sparse


class SparseModalAssemblyTest(unittest.TestCase):
    def test_no_learned_nonzero_is_removed_and_modes_agree(self):
        ModalSensitivityTest.setUpClass();t=ModalSensitivityTest
        derivative=t.mesh['derivative']+np.einsum('a,aij->ij',[.013,-.007],t.basis)
        a=dense.stiffness(t.mesh,derivative,t.quadratic)
        b=sparse.stiffness(t.mesh,derivative,t.quadratic)
        self.assertEqual(a.nnz,b.nnz)
        difference=a-b
        self.assertLess(np.linalg.norm(difference.data)/np.linalg.norm(a.data),2e-15)
        f,x=dense.modes(t.mesh,t.scene,derivative,t.quadratic,10)
        g,y=sparse.modes(t.mesh,t.scene,derivative,t.quadratic,10)
        mapping,mac=dense.match(x,y,t.mesh['mass'])
        np.testing.assert_allclose(f,g[mapping],rtol=1e-10,atol=.0001)
        self.assertGreater(mac.min(),1-1e-10)


if __name__=='__main__':unittest.main()
