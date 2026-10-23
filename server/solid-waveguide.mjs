import {readFile,lstat} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {planPlatePropagation} from '../public/solid-propagation.js';

const finite=v=>Number.isFinite(v),nonnegative=v=>finite(v)&&v>=0;
const near=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((v,i)=>finite(v)&&Math.abs(v-b[i])<1e-10);
export function validateWaveguideReference(r,study){
  const s=study.scene,m=s.material;
  if(r.schema!=='workai.rectangular-waveguide.validation.v1'||r.study_id!==study.study_id||s.geometry.kind!=='plate'||r.frequency_mhz!==s.source.frequency_mhz||r.cross_section_mm?.width!==s.geometry.width_mm||r.cross_section_mm?.thickness!==s.geometry.thickness_mm||!['rho_kg_m3','lambda_pa','mu_pa'].every(k=>r.material?.[k]===m[k]))throw Error('截面参考与当前研究不一致。');
  if(JSON.stringify(r.transient_references)!==JSON.stringify(study.controls.slice(0,3))||r.modal_purity_verified!==false||r.nonlinear_coupling_integral_computed!==false||r.other_second_harmonic_modes_excluded!==false||r.window_changed_for_this_fit!==false)throw Error('截面参考身份或适用范围不正确。');
  if(!Array.isArray(r.branches)||r.branches.length!==2||r.branches.some((b,i)=>b.frequency_hz!==s.source.frequency_mhz*1e6*(i+1)||!['wavenumber_rad_m','phase_velocity_m_s','group_velocity_m_s'].every(k=>finite(b[k])&&b[k]>0)||!nonnegative(b.relative_eigen_residual)||b.relative_eigen_residual>1e-6||!finite(b.minimum_tracking_MAC)||b.minimum_tracking_MAC<.8||b.minimum_tracking_MAC>1+1e-9))throw Error('截面特征解未通过残差或分支跟踪检查。');
  const delta=r.branches[1].wavenumber_rad_m-2*r.branches[0].wavenumber_rad_m;
  if(!finite(r.mismatch_rad_m)||Math.abs(delta-r.mismatch_rad_m)>1e-8||!finite(r.coherence_length_mm)||Math.abs(r.coherence_length_mm-Math.PI/Math.abs(delta)*1000)>1e-7)throw Error('相位失配与分支波数不一致。');
  if(!Array.isArray(r.successive_relative_k_differences)||r.successive_relative_k_differences.length!==2||!r.successive_relative_k_differences.every(a=>Array.isArray(a)&&a.length===2&&a.every(nonnegative))||r.dispersion_finest_pair_within_one_percent!==r.successive_relative_k_differences[1].every(v=>v<=.01))throw Error('截面网格检查不一致。');
  if(!Array.isArray(r.rows)||r.rows.length!==s.receivers.length||r.rows.some((row,i)=>{
    const rx=s.receivers[i],plan=planPlatePropagation(s,rx.point_mm);
    return !near(row.point_mm,rx.point_mm)||!plan.available||!near(row.window_us,plan.window_us)||!nonnegative(row.same_branch_group_walkoff_us)||['linear_fundamental_shape','material_second_shape'].some(k=>!nonnegative(row[k]?.relative_mass_L2_shape_mismatch)||row[k].relative_mass_L2_shape_mismatch>1+1e-8||row[k].not_a_modal_energy_fraction!==true||!Array.isArray(row[k].complex_coefficient)||row[k].complex_coefficient.length!==2||!row[k].complex_coefficient.every(finite));
  }))throw Error('截面形状比较的点位、时间窗或范数不正确。');
  if(!Array.isArray(r.phase_checks)||r.phase_checks.length!==r.rows.length-1||r.phase_checks.some((p,i)=>!finite(p.relative_wavenumber_difference)||!(p.to_mm>p.from_mm)||p.from_mm!==r.rows[i].point_mm[0]||p.to_mm!==r.rows[i+1].point_mm[0]||p.spacing_below_half_wavelength!==(r.branches[0].wavenumber_rad_m*(p.to_mm-p.from_mm)*.001<Math.PI)))throw Error('截面相位比较不正确。');
  const pass=r.dispersion_finest_pair_within_one_percent&&r.phase_checks.every(p=>p.spacing_below_half_wavelength&&Math.abs(p.relative_wavenumber_difference)<=.05)&&r.rows.every(p=>p.linear_fundamental_shape.relative_mass_L2_shape_mismatch<=.05);
  if(r.fundamental_candidate_checks_within_5_percent!==pass)throw Error('截面候选结论与数值不一致。');
  return r;
}
export function validateWaveguideDispersion(record,dispersion,bytes){
  if(createHash('sha256').update(bytes).digest('hex')!==record.dispersion_sha256||dispersion.schema!=='workai.rectangular-waveguide.dispersion.v1')throw Error('截面色散原始记录摘要不符。');
  const branches=dispersion.levels?.at(-1)?.branches?.map(({tracking_path,...b})=>b);
  if(JSON.stringify(branches)!==JSON.stringify(record.branches)||JSON.stringify(dispersion.successive_relative_k_differences)!==JSON.stringify(record.successive_relative_k_differences)||dispersion.mismatch_rad_m!==record.mismatch_rad_m||dispersion.coherence_length_mm!==record.coherence_length_mm)throw Error('截面摘要与独立特征解不一致。');
}
export async function readWaveguideReference(directory,study){
  const path=join(directory,'waveguide-validation.json');let stat;try{stat=await lstat(path);}catch(error){if(error.code==='ENOENT')return null;throw error;}
  if(!stat.isFile()||stat.isSymbolicLink()||stat.size>262144)throw Error('截面参考文件无效。');
  const record=JSON.parse(await readFile(path)),result=await readFile(join(directory,'result.json')),status=JSON.parse(await readFile(join(directory,'status.json'))),hash=createHash('sha256').update(result).digest('hex');
  if(status.phase!=='completed'||record.study_result_sha256!==hash||status.result_sha256!==hash)throw Error('截面参考未绑定当前完成研究。');
  const source=join(directory,'waveguide-dispersion.json'),sourceStat=await lstat(source);
  if(!sourceStat.isFile()||sourceStat.isSymbolicLink()||sourceStat.size>262144)throw Error('截面色散源记录无效。');
  const bytes=await readFile(source);validateWaveguideDispersion(record,JSON.parse(bytes),bytes);
  return validateWaveguideReference(record,study);
}
