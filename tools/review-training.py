"""Reload this task's saved checkpoint, compare stored predictions, and plot evidence.

This reads local artifacts only. It never retrains or submits a COMSOL job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
os.environ.setdefault('MPLCONFIGDIR', '/tmp/workai-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.model import GuidedWaveField
from training.physics import U0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    run, output = args.run_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run/'config.json').read_text())
    metrics = json.loads((run/'metrics.json').read_text())
    for name, expected in config['source_files'].items():
        for path in [REPO/name, run/'source_snapshot'/name]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    digest = hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    assert digest == metrics['checkpoint_sha256']
    torch.set_num_threads(2)
    saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    assert saved['step'] == metrics['selected_step']
    model = GuidedWaveField(saved['scene'], saved['scales'], saved['width'])
    model.load_state_dict(saved['model_state'])
    model.eval()
    with np.load(run/'predictions.npz', allow_pickle=False) as values:
        q, prediction, truth, rx = [values[k] for k in ['q','predicted','comsol','rx']]
    with torch.no_grad():
        reloaded = model.controls(torch.tensor(q, dtype=torch.float64)).numpy()
    maximum = float(np.max(abs(reloaded-prediction)))
    assert np.allclose(reloaded, prediction, rtol=1e-12, atol=1e-14)
    result = {'passed':True, 'checkpoint_sha256':digest, 'selected_step':saved['step'],
              'max_reload_difference_displacement_over_100nm':maximum,
              'source_snapshot_matches':True, 'accepted_for_solver_routing':False}
    (output/'checkpoint-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), layout='constrained')
    for key, label in [('data','Training labels'),('pde','Momentum'),('traction','Traction'),('validation_mse','40 mm validation')]:
        axes[0,0].semilogy([h['step'] for h in history], [h[key] for h in history], label=label)
    axes[0,0].axvline(saved['step'],color='#777777',ls=':',lw=1)
    axes[0,0].set(xlabel='Optimizer step',ylabel='Normalized mean-square loss',title='Actual optimization history')
    axes[0,0].legend(fontsize=8)
    mask = rx == 1
    time = q[mask,2]
    factor = U0*1e9
    for ax, value, predicted, title, ylabel in [
        (axes[0,1],truth[mask,0,0],prediction[mask,0,0],'40 mm: linear displacement','u / nm'),
        (axes[1,0],truth[mask,2,0]-truth[mask,1,0],prediction[mask,2,0]-prediction[mask,1,0],
         '40 mm: material contribution (full - geometric)','Material increment u / nm')]:
        ax.plot(time,value*factor,label='COMSOL',color='#148778',lw=1.8)
        ax.plot(time,predicted*factor,label='Surrogate',color='#bc6b46',ls='--',lw=1.4)
        ax.set(xlabel='Time / microseconds',ylabel=ylabel,title=title)
        ax.legend(fontsize=8)
    rows = metrics['material_harmonics']
    axes[1,1].plot([r['x_mm'] for r in rows],[r['comsol_material_a2_nm'] for r in rows],'o-',label='COMSOL',color='#148778')
    axes[1,1].plot([r['x_mm'] for r in rows],[r['predicted_material_a2_nm'] for r in rows],'s--',label='Surrogate',color='#bc6b46')
    error40 = next(r['relative_complex_error'] for r in rows if r['x_mm']==40)
    axes[1,1].set(xlabel='Distance / mm (40 mm held out of label fitting)',ylabel='Material A2 / nm',
                  title=f'40 mm complex harmonic error: {error40:.2%} (limit 5%)',xticks=[20,40,60])
    axes[1,1].legend(fontsize=8)
    fig.suptitle(f'Experimental physics-constrained surrogate | 1 mm Al, 0.4 MHz | checkpoint {saved["step"]}\n'
                 'FAILED accuracy acceptance - not enabled as a solver',fontsize=14,color='#8e3e27')
    for ax in axes.flat:
        ax.grid(True,alpha=.16)
    fig.savefig(output/'training-diagnostics.png',dpi=170)
    fig.savefig(output/'training-diagnostics.svg')
    fig.savefig(output/'training-diagnostics.pdf')
    plt.close(fig)
    print(json.dumps(result))

if __name__=='__main__':
    main()
