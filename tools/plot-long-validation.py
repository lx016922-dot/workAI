"""Scientific diagnostic plot from preserved receiver records, no curve fitting to a model."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,hilbert
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--case-directory',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
left=[];right=[]
for p in sorted(args.case_directory.glob('probes-*.npz')):
    with np.load(p,allow_pickle=False) as data:
        left.append(data['candidate'][2,:,0,0]);right.append(data['reference'][2,:,0,0])
a,b=np.concatenate(left),np.concatenate(right);assert len(a)==len(b)==25001
t=np.arange(len(a))*.2
sos=butter(4,[.13,.19],fs=5,output='sos',btype='bandpass')
phases=[np.unwrap(np.angle(hilbert(sosfiltfilt(sos,x)))) for x in [a,b]]
mask=(t>=1000)&(t<=4000)
fits=[np.polyfit(t[mask],p[mask],1) for p in phases]
f=[v[0]/(2*np.pi)*1e6 for v in fits];delta=f[0]-f[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})
fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
for ax,lo,hi,title in [(axes[0],0,50,'Early response: 0–50 μs'),(axes[1],4900,4950,'Late response: 4900–4950 μs')]:
    selected=(t>=lo)&(t<=hi)
    ax.plot(t[selected],a[selected]*1e9,color='#077b83',lw=1.6,label='p4 graph prediction')
    ax.plot(t[selected],b[selected]*1e9,color='#b55a34',lw=1.2,ls='--',label='p5 analytic reference')
    ax.set(xlabel='Time / μs',ylabel='X displacement / nm',title=title,xlim=(lo,hi));ax.legend(loc='upper right',frameon=False)
phase=(phases[0]-phases[1])*180/np.pi
phase-=phase[np.flatnonzero(mask)[0]]
axes[2].plot(t[mask]/1000,phase[mask],color='#573c87',lw=2,label='Band-limited phase difference (1–4 ms)')
axes[2].set(xlabel='Time / ms',ylabel='Phase drift since 1 ms / degrees',title='A small modal frequency shift accumulates over the long record')
axes[2].text(.02,.93,f'Estimated frequencies: {f[0]:.2f} vs {f[1]:.2f} Hz\nDifference: {delta:.2f} Hz → {delta*.005*360:.1f}° over 5 ms',transform=axes[2].transAxes,va='top',bbox=dict(facecolor='white',edgecolor='#dddddd',alpha=.95))
fig.suptitle('5 ms verification: Hann burst, 5 cycles, fixed-end aluminum plate\nOne predeclared receiver; identical 2.5 mm element size, different spatial degree',fontsize=13)
fig.savefig(args.output/'phase-diagnostic.png',dpi=180)
fig.savefig(args.output/'phase-diagnostic.pdf')
plt.close(fig)
(args.output/'phase-plot-data.json').write_text(json.dumps({'case':'hann-05','receiver':1,'component':'x','candidate_frequency_hz':float(f[0]),'reference_frequency_hz':float(f[1]),'difference_hz':float(delta),'phase_accumulation_5ms_degrees':float(delta*.005*360),'analysis':'zero-phase Butterworth bandpass 0.13–0.19 MHz, Hilbert phase regression over1000–4000 us','raw_samples':25001},indent=2)+'\n')
print('Saved phase diagnostic figure and audit data')
