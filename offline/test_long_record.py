"""Regression checks for real sampling, durable restart, and input bounds."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from spectral3d.mesh import make_mesh
from graphwave.long_rollout import rollout
from .domain import templates,validate_request,MODEL,model_info

class LongRecord(unittest.TestCase):
    def test_twenty_real_configurations(self):
        for kind in ('hann','gaussian','rectangular','chirp'):
            for cycles in (1,3,5,10,20):
                scene=templates()['plate'];scene['signal']['kind']=kind;scene['source']['cycles']=cycles
                result=validate_request({'scene':scene,'degree':5})
                self.assertEqual(result['scene']['acquisition'],{'duration_us':5000.,'sample_step_us':.2})
                self.assertEqual(result['scene']['source']['cycles'],cycles)
        for acquisition in ({'duration_us':50.,'sample_step_us':.2},{'duration_us':5000.,'sample_step_us':1.},
                            {'duration_us':5000.,'sample_step_us':.001},{'duration_us':5000.1,'sample_step_us':.2}):
            scene=templates()['plate'];scene['acquisition']=acquisition
            with self.assertRaises(ValueError):validate_request({'scene':scene,'degree':5})
        scene=templates()['plate'];scene['signal']=[]
        with self.assertRaises(ValueError):validate_request({'scene':scene,'degree':5})

    def test_checkpoint_matches_uninterrupted_record(self):
        request=json.loads((Path(__file__).parent/'example-request.json').read_text())
        scene=request['scene'];scene['acquisition']={'duration_us':60.,'sample_step_us':.2}
        mesh=make_mesh(scene,4,2.5)
        with np.load(MODEL/'continued-weights.npz') as saved:weights=dict(saved)
        temporal=json.loads((MODEL/'time-coefficients.json').read_text())
        identity={'fixture':'durable-restart'}
        with tempfile.TemporaryDirectory() as temporary:
            a,b=Path(temporary)/'continuous',Path(temporary)/'resumed'
            complete=rollout(scene,mesh,weights,temporal,a,identity)
            first=rollout(scene,mesh,weights,temporal,b,identity,stop_after_chunks=1)
            self.assertEqual(first['phase'],'running');self.assertEqual(first['completed_frames'],256)
            resumed=rollout(scene,mesh,weights,temporal,b,identity)
            self.assertEqual(resumed['phase'],'completed');self.assertEqual(resumed['frame_count'],301)
            self.assertEqual(resumed['completed_time_us'],60.)
            for left,right in zip(complete['chunks'],resumed['chunks']):
                with np.load(a/left['file']) as x,np.load(b/right['file']) as y:
                    for key in x.files:np.testing.assert_array_equal(x[key],y[key])
            with self.assertRaises(ValueError):rollout(scene,mesh,weights,temporal,b,{'fixture':'changed'})
            (b/resumed['chunks'][0]['file']).write_bytes(b'corrupted')
            with self.assertRaises(AssertionError):rollout(scene,mesh,weights,temporal,b,identity)

if __name__=='__main__':unittest.main()
