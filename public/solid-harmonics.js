import {unitVector} from './solid-data.js';

export const harmonicControls=['linear_A','geometric_A','material_A','geometric_2A','material_2A'];
export function projectHarmonicProbe(probe,direction){
  if(probe?.schema!=='workai.solid.harmonic-probe.v1'||!probe.same_mesh_and_time_step||!Array.isArray(probe.times_us)||probe.times_us.length<21||probe.times_us.length>2001)throw new Error('五组同网格对照记录不完整。');
  const d=unitVector(direction),controls={};
  for(const role of harmonicControls){const c=probe.controls?.[role];
    if(!c||!Array.isArray(c.xyz_m)||c.xyz_m.length!==probe.times_us.length||!c.xyz_m.every(v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite))||!Number.isFinite(c.max_export_roundoff_m)||c.max_export_roundoff_m<0)throw new Error('对照波形或存储精度记录无效。');
    controls[role]={...c,projected_m:c.xyz_m.map(v=>v.reduce((s,x,j)=>s+x*d[j],0))};
  }
  return {...probe,direction:d,controls};
}

export function harmonicCoefficient(times,values,frequency,start,end){
  if(times.length!==values.length||!times.every(Number.isFinite)||!values.every(Number.isFinite)||!(frequency>0)||!Number.isFinite(start)||!Number.isFinite(end)||!(end>start)||start<times[0]-1e-9||end>times.at(-1)+1e-9)throw new Error('谐波分析窗必须完整位于已有记录内。');
  const step=times[1]-times[0];if(!(step>0)||frequency>=.5/step||!times.every((t,i)=>Math.abs(t-times[0]-i*step)<1e-7))throw new Error('谐波频率或采样网格不正确。');
  const weights=times.map(t=>t<start||t>end?0:.5*(1-Math.cos(2*Math.PI*(t-start)/(end-start)))),sum=weights.reduce((a,b)=>a+b,0);
  if(sum<=2)throw new Error('分析窗内的有效采样不足。');
  const mean=values.reduce((s,v,i)=>s+weights[i]*v,0)/sum;let re=0,im=0;
  for(let i=0;i<times.length;i++){const v=weights[i]*(values[i]-mean),phase=2*Math.PI*frequency*times[i];re+=v*Math.cos(phase);im-=v*Math.sin(phase);}
  re*=2/sum;im*=2/sum;
  return {re_m:re,im_m:im,amplitude_m:Math.hypot(re,im),phase_deg:Math.atan2(im,re)*180/Math.PI};
}

export function analyzeSolidHarmonics(input,start,end,direction=input.direction){
  const probe=projectHarmonicProbe(input,direction),frequency=probe.scene.source.frequency_mhz;
  if((end-start)*frequency<2-1e-9)throw new Error('分析窗至少覆盖两个基频周期；短窗不用于本次谐波检查。');
  const values=Object.fromEntries(harmonicControls.map(k=>[k,probe.controls[k].projected_m]));
  const delta=(a,b)=>values[a].map((v,i)=>v-values[b][i]),signal={
    linear:values.linear_A,full:values.material_A,geometric_increment:delta('geometric_A','linear_A'),
    material_increment:delta('material_A','geometric_A'),material_double_increment:delta('material_2A','geometric_2A')};
  const coeff=(v,k)=>harmonicCoefficient(probe.times_us,v,k*frequency,start,end);
  const fundamental=coeff(signal.full,1),linearSecond=coeff(signal.linear,2),geometricSecond=coeff(signal.geometric_increment,2),materialSecond=coeff(signal.material_increment,2),materialDouble=coeff(signal.material_double_increment,2),fullSecond=coeff(signal.full,2);
  const norm1=probe.direction.reduce((s,v)=>s+Math.abs(v),0),q=(a,b)=>4*norm1*(probe.controls[a].max_export_roundoff_m+probe.controls[b].max_export_roundoff_m);
  // Barycentric weights are nonnegative. The bound covers quantization and
  // weighted-mean removal in the complex projection, not discretization error.
  const bound=q('material_A','geometric_A'),doubleBound=q('material_2A','geometric_2A');
  const reliable=materialSecond.amplitude_m>0&&materialDouble.amplitude_m>0&&bound/materialSecond.amplitude_m<=.01&&doubleBound/materialDouble.amplitude_m<=.01;
  const ratio=reliable?materialDouble.amplitude_m/materialSecond.amplitude_m:null;
  const scalingError=reliable?Math.hypot(materialDouble.re_m-4*materialSecond.re_m,materialDouble.im_m-4*materialSecond.im_m)/(4*materialSecond.amplitude_m):null;
  const equal=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<1e-10),reference=probe.numerical_reference;
  let selectedReference=null;
  if(reference?.schema==='workai.solid.harmonic-validation.v1'&&reference.threshold===.05&&reference.references?.find(r=>r.level==='timecheck')?.study_id===probe.study_id){
    const row=reference.rows.find(r=>equal(r.point_mm,probe.point_mm)&&equal(r.direction,probe.direction)&&equal(r.window_us,[start,end]));
    if(row&&Math.abs(row.material_second_nm*1e-9-materialSecond.amplitude_m)<=materialSecond.amplitude_m*1e-7){const space=row.spatial_with_changed_stable_step,time=row.temporal_same_mesh;
      if([space,time].every(v=>Number.isFinite(v)&&v>=0))selectedReference={spatial_with_changed_stable_step:space,temporal_same_mesh:time,all_selected_checks_within_5_percent:reliable&&space<=.05&&time<=.05&&scalingError!==null&&scalingError<=.05,scope:'This point, direction and time window only.'};
    }
  }
  return {schema:'workai.solid.harmonic-analysis.v1',study_id:probe.study_id,point_mm:probe.point_mm,direction:probe.direction,window_us:[start,end],frequency_mhz:frequency,
    extraction:'Hann weighted mean removed complex projection; subtract time traces before projection',
    fundamental,linear_second:linearSecond,full_second:fullSecond,geometric_second:geometricSecond,material_second:materialSecond,material_double_second:materialDouble,
    export_roundoff_bound_m:bound,double_export_roundoff_bound_m:doubleBound,above_export_roundoff:reliable,amplitude_ratio_2A_A:ratio,
    relative_complex_quadratic_error:scalingError,quadratic_scaling_within_5_percent:reliable&&scalingError<=.05,
    relative_indicator_per_m:fundamental.amplitude_m>0&&reliable?materialSecond.amplitude_m/fundamental.amplitude_m**2:null,
    selected_reference:selectedReference,harmonic_convergence_verified:false,comsol_agreement_verified:false,modal_separation:false,
    window_scope:'User-selected finite record; contains reflections unless independently excluded. No automatic pure-mode or cumulative-growth claim.',signals_m:signal};
}
