"""Numba export must agree with the actual trained PyTorch forward pass."""
import argparse,json
from pathlib import Path
import numpy as np
from .inference import energy_activation,graph_message
from .predict import digest


def run(args):
    global energy_activation,graph_message
    if args.engine=='buffered':
        from .buffered_inference import energy_activation,graph_message
    with np.load(args.reference,allow_pickle=False) as data:r=dict(data)
    with np.load(args.weights,allow_pickle=False) as data:w=dict(data)
    rows=[]
    for law in range(3):
        stress=np.stack([energy_activation(g,w['quadratic_tensor'],w['cubic_tensor'],law)[0] for g in r['gradients']])
        degree=int(r.get('degree',4))
        message=graph_message(r['current'],r['cells'],w['derivative_'+str(degree)],r['inverse'],r['volume'],w['quadratic_tensor'],w['cubic_tensor'],law)[0]
        np.testing.assert_allclose(stress,r['stress_'+str(law)],rtol=2e-12,atol=1e-8)
        np.testing.assert_allclose(message,r['message_'+str(law)],rtol=2e-12,atol=1e-12)
        rows.append({'law':law,'stress_relative_l2':float(np.linalg.norm(stress-r['stress_'+str(law)])/np.linalg.norm(stress)),
            'message_relative_l2':float(np.linalg.norm(message-r['message_'+str(law)])/np.linalg.norm(message))})
    receipt={'passed':True,'engine':args.engine,'degree':degree,'weights_sha256':digest(args.weights),'torch_output_sha256':digest(args.reference),'checks':rows}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',choices=['original','buffered'],default='original');run(p.parse_args())
