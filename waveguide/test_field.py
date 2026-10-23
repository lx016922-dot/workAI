"""Independent analytic checks of the transient-to-section comparison."""
import importlib.util
import itertools
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('field_comparison',Path(__file__).resolve().parents[1]/'tools/compare-waveguide-field.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)

class FieldComparisonTests(unittest.TestCase):
    def test_tetrahedral_section_interpolation_reproduces_an_affine_vector_field(self):
        nodes=np.array([[i&1,(i>>1)&1,(i>>2)&1] for i in range(8)])*np.array([3.,4.,1.])
        tetra=[[0,1<<a,(1<<a)+(1<<b),7] for a,b,c in itertools.permutations(range(3))]
        record={'nodes_mm':nodes.tolist(),'tetra':tetra}
        matrix=np.array([[2.,-3.,4.],[-5.,6.,7.],[8.,9.,-10.]])*1e-9
        offset=np.array([1.,2.,3.])*1e-9
        field=np.array([nodes@matrix.T+offset,-2*(nodes@matrix.T+offset)])
        yz=np.array([[0.,0.],[1.7,.32],[4.,1.],[4.,0.],[0.,1.]])*.001
        sampled=comparison.interpolate_plane(record,field,.74,yz)
        points=np.column_stack([np.full(len(yz),.74),yz*1000])
        expected=points@matrix.T+offset
        np.testing.assert_allclose(sampled,np.array([expected,-2*expected]),rtol=1e-13,atol=1e-22)
        with self.assertRaisesRegex(ValueError,'outside'):
            comparison.interpolate_plane(record,field,3.1,yz)

    def test_hann_complex_projection_preserves_phase_and_rejects_incomplete_windows(self):
        times=np.linspace(0,20,401);amplitude=np.array([2+3j,-4+5j,6-7j])*1e-9
        signal=np.real(np.exp(2j*np.pi*.5*times[:,None])*amplitude)+9e-9
        signal+=np.cos(2*np.pi*1.5*times[:,None])*2e-9
        result=comparison.coefficient(times,signal,.5,[0,20])
        np.testing.assert_allclose(result,amplitude,rtol=1e-13,atol=1e-22)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            comparison.coefficient(times,signal,.5,[0,21])

if __name__=='__main__':unittest.main()
