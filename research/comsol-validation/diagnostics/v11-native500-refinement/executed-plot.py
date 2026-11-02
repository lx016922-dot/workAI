import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

repo=Path('/workspace/workAI');sys.path.insert(0,str(repo));out=repo/'research/comsol-validation/diagnostics/v11-native500-refinement'
spec=importlib.util.spec_from_file_location('native',repo/'tools/compare-comsol-reference.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
raw,receipt,truth=native.load_case(repo/'research/comsol-validation/data/native-quarter-075-20us-v11')
summary=json.loads((out/'summary.json').read_text());older=json.loads((repo/'research/comsol-validation/diagnostics/v11-native-quarter075-comparison.json').read_text())
reports=[]
for folder in [repo/'research/comsol-validation/diagnostics/v11-native-modal-truncation/modes200',out/'modes300',out/'modes500']:
 prediction=json.loads((folder/'prediction.json').read_text());assert hashlib.sha256((folder/'prediction.npz').read_bytes()).hexdigest()==prediction['prediction_sha256']
 with np.load(folder/'prediction.npz',allow_pickle=False) as f:times=f['times_us'];parts=native.parts(dict(enumerate(f['traces'])))
 reports.append((folder,parts))
reference=native.parts(truth)
spec=importlib.util.spec_from_file_location('checks',repo/'tools/validate-comsol-linear-matrix.py');checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
metrics=[{key:checks.check(parts[key],reference[key],raw['scene']) for key in parts} for _,parts in reports]
for actual,expected in [(metrics[1],summary['native_comparisons'][0]),(metrics[2],summary['native_comparisons'][1])]:
 for key in actual:np.testing.assert_allclose(actual[key]['maximum_relative_l2'],expected[key]['maximum_relative_l2'],rtol=1e-10,atol=1e-12)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(12,8));grid=fig.add_gridspec(2,2,height_ratios=[1,1.1],hspace=.45,wspace=.25)
ax=fig.add_subplot(grid[0,0]);x=np.arange(3);colors=['#24779d','#d88033','#7670a9']
for offset,(key,label,color) in enumerate(zip(['geometric','material','total_nonlinear'],['Geometric','Material','Total nonlinear'],colors)):
 values=[100*r[key]['maximum_vector_waveform_relative_l2'] for r in metrics]
 ax.bar(x+(offset-1)*.23,values,width=.22,color=color,label=label)
ax.axhline(5,color='#a62b2b',ls=':',lw=1.3);ax.set(xticks=x,xticklabels=['200','300','500'],xlabel='Candidate modes',ylabel='Relative L2 (%)',title='Maximum vector waveform error',ylim=(0,8.5));ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.16)
ax=fig.add_subplot(grid[0,1]);values=[100*max(v['maximum_relative_l2'] for v in r.values()) for r in metrics]
bars=ax.bar(x,values,width=.55,color='#52627c');ax.bar_label(bars,fmt='%.2f%%',padding=3)
ax.axhline(5,color='#a62b2b',ls=':',lw=1.3,label='5% criterion');ax.set(xticks=x,xticklabels=['200','300','500'],xlabel='Candidate modes',ylabel='Relative L2 (%)',title='Maximum across components and bands',ylim=(0,35));ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.16)
ax=fig.add_subplot(grid[1,:]);receiver,axis=6,2
ax.plot(times,1e12*reference['material'][:,receiver,axis],color='#172b4d',lw=1.8,label='Native COMSOL, h = 0.75 mm')
for label,(_,parts),color,style in [('300 modes',reports[1],'#24779d','--'),('500 modes',reports[2],'#d88033','-.')]:
 ax.plot(times,1e12*parts['material'][:,receiver,axis],color=color,ls=style,lw=1.2,label=label)
ax.set(xlim=(0,20),xlabel='Time (microseconds)',ylabel='Material increment (pm)',title='P7, Z component: largest relative error for 500 modes');ax.grid(alpha=.16);ax.legend(frameon=False,ncol=3,fontsize=9)
fig.suptitle('Short native COMSOL comparison: increasing candidate modes',fontsize=16,y=.98)
fig.text(.5,.94,'20 microseconds; Hann burst, 2 cycles, 0.22 MHz; candidate step 0.025 microseconds',ha='center',fontsize=10)
fig.text(.07,.04,'All samples are used without time alignment or amplitude scaling. The transverse residual remains above 5%.\nNative mesh/time convergence and the complete nonlinear 5 ms matrix remain unverified.',fontsize=10)
fig.subplots_adjust(top=.86,bottom=.17)
fig.savefig(out/'short-comparison.png',dpi=160);fig.savefig(out/'short-comparison.pdf');plt.close(fig)
with (out/'short-P7-material-z.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['time_us','native_pm','modes300_pm','modes500_pm'])
 for i,t in enumerate(times):writer.writerow([t,1e12*reference['material'][i,receiver,axis],1e12*reports[1][1]['material'][i,receiver,axis],1e12*reports[2][1]['material'][i,receiver,axis]])
proof={'schema':'workai.short_native_modal_plot.v1','duration_us':20,'reference_job_id':receipt['job_id'],'mode_counts':[200,300,500],'candidate_step_us':.025,'selected_receiver':7,'selected_component':'z','selection_reason':'Largest relative error for the 500-mode material increment, not a favorable receiver','reference_convergence_verified':False,'scientific_accuracy_certified':False,'files_sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['short-comparison.png','short-comparison.pdf','short-P7-material-z.csv']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'plot-provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'plot':str(out/'short-comparison.png'),'strict_errors_percent':values}),flush=True)
