// Read-only: bind completed studies, extract receiver traces, compare complex 2f.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {SolidService} from '../server/solid.mjs';
import {analyzeSolidHarmonics} from '../public/solid-harmonics.js';
import {planPlatePropagation} from '../public/solid-propagation.js';
const [root,evidence,mode]=process.argv.slice(2);if(!root||!evidence||(mode&&mode!=='--direct-windows'))throw new Error('Usage: node tools/review-solid-harmonics.mjs REMOTE_ROOT EVIDENCE_DIR [--direct-windows]');
const manifest=JSON.parse(await readFile(join(evidence,'studies.json')));if(!manifest.complete||manifest.references.length!==3)throw new Error('All three levels must finish first.');
const service=new SolidService(root);await service.init();const levels={},traces=[];
for(const ref of manifest.references){const bytes=await readFile(join(root,'solid/studies',ref.study_id,'result.json'));if(createHash('sha256').update(bytes).digest('hex')!==ref.result_sha256)throw new Error('Study reference changed.');
  const checked=await service.studies.result(ref.study_id);levels[ref.level]=checked;
  for(let i=0;i<checked.study.scene.receivers.length;i++){const rx=checked.study.scene.receivers[i],probe=await service.studies.probe(ref.study_id,{point_mm:rx.point_mm,direction:rx.direction});
    const plan=mode?planPlatePropagation(probe.scene,rx.point_mm):null;if(mode&&(!plan.available||!plan.passes_end_echo_screen))throw new Error('Suggested window unavailable or fails end-echo screen.');
    const window=plan?.window_us??[0,checked.study.scene.acquisition.duration_us],analysis=analyzeSolidHarmonics(probe,...window);
    const name=ref.level+'-P'+(i+1);await writeFile(join(evidence,name+'-probe.json'),JSON.stringify(probe),{flag:'wx'});await writeFile(join(evidence,name+'-analysis.json'),JSON.stringify(analysis),{flag:'wx'});traces.push({level:ref.level,receiver:i,analysis});
  }
}
const physical=s=>{const c=structuredClone(s);delete c.name;delete c.numerics;return JSON.stringify(c);};
const a=levels.coarse.records[0],b=levels.fine.records[0],c=levels.timecheck.records[0];
if(physical(a.scene)!==physical(b.scene)||physical(b.scene)!==physical(c.scene))throw new Error('Physical scenes differ.');
if(JSON.stringify(a.source_sha256)!==JSON.stringify(b.source_sha256)||JSON.stringify(b.source_sha256)!==JSON.stringify(c.source_sha256))throw new Error('Physics source differs.');
if(!(b.tetra.length>a.tetra.length))throw new Error('Mesh was not refined.');
if(JSON.stringify(b.nodes_mm)!==JSON.stringify(c.nodes_mm)||JSON.stringify(b.tetra)!==JSON.stringify(c.tetra)||!(c.diagnostics.internal_step_us<b.diagnostics.internal_step_us))throw new Error('Pure time reference must use the same mesh and a smaller step.');
const relative=(a,b)=>b.amplitude_m>0?Math.hypot(a.re_m-b.re_m,a.im_m-b.im_m)/b.amplitude_m:null;
const rows=a.scene.receivers.map((rx,i)=>{const get=l=>traces.find(t=>t.level===l&&t.receiver===i).analysis,coarse=get('coarse'),fine=get('fine'),time=get('timecheck'),space=relative(coarse.material_second,fine.material_second),temporal=relative(fine.material_second,time.material_second);
  return {name:rx.name,point_mm:rx.point_mm,direction:rx.direction,window_us:time.window_us,material_second_nm:time.material_second.amplitude_m*1e9,linear_second_nm:time.linear_second.amplitude_m*1e9,full_fundamental_nm:time.fundamental.amplitude_m*1e9,spatial_with_changed_stable_step:space,temporal_same_mesh:temporal,amplitude_ratio_2A_A:time.amplitude_ratio_2A_A,quadratic_complex_error:time.relative_complex_quadratic_error,above_export_roundoff:time.above_export_roundoff,selected_point_checks_within_5_percent:coarse.above_export_roundoff&&fine.above_export_roundoff&&time.above_export_roundoff&&space!==null&&temporal!==null&&space<=.05&&temporal<=.05&&time.quadratic_scaling_within_5_percent};});
const report={schema:'workai.solid.harmonic-validation.v1',references:manifest.references,field_precision:a.field.dtype,threshold:.05,mesh_elements:[a.tetra.length,b.tetra.length,c.tetra.length],internal_steps_us:[a.diagnostics.internal_step_us,b.diagnostics.internal_step_us,c.diagnostics.internal_step_us],rows,all_selected_point_checks_within_5_percent:rows.every(r=>r.selected_point_checks_within_5_percent),window_policy:mode?'S0_candidate_arrival_aligned_equal_duration_end_echo_screened':'full_record',scope:mode?'Moving windows pass a distal-end echo speed estimate. Finite-width modes are not separated. No pure-mode accumulation law, arbitrary-shape or independent COMSOL accuracy claim.':'Full finite record may include reflections. No pure-mode, accumulation-law, arbitrary-shape or independent COMSOL accuracy claim.'};
await writeFile(join(evidence,'harmonic-validation.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));
