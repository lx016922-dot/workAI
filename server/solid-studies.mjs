import {spawn} from 'node:child_process';
import {mkdir,readdir,readFile,writeFile,open,lstat,copyFile} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash,randomBytes} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {AppError} from './remote.mjs';
import {pointSeries,unitVector,validateSolidResult} from '../public/solid-data.js';
import {readWaveguideReference} from './solid-waveguide.mjs';

const repo=fileURLToPath(new URL('../',import.meta.url)),hash=b=>createHash('sha256').update(b).digest('hex');
export const SOLID_CONTROLS=['linear_A','geometric_A','material_A','geometric_2A','material_2A'];
async function readBounded(path,limit=262144){const stat=await lstat(path);if(!stat.isFile()||stat.isSymbolicLink()||stat.size>limit)throw new AppError(422,'非线性对照记录无效。');return readFile(path);}
function comparable(scene){const s=structuredClone(scene);delete s.name;delete s.constitutive;delete s.source.amplitude_mpa;return s;}
export function validateControlSet(study,records){
  if(study.schema!=='workai.solid.nonlinear-study.v1'||!Array.isArray(study.controls)||study.controls.map(c=>c.role).join(',')!==SOLID_CONTROLS.join(',')||new Set(study.controls.map(c=>c.job_id)).size!==5||records.length!==5)throw new AppError(422,'五组对照不完整。');
  const base=records[0],same=JSON.stringify(comparable(study.scene)),laws=['linear','stvk','murnaghan','stvk','murnaghan'];
  for(let i=0;i<5;i++){
    const r=records[i];validateSolidResult(r);
    if(r.job_id!==study.controls[i].job_id||JSON.stringify(comparable(r.scene))!==same||r.scene.constitutive!==laws[i]||r.scene.source.amplitude_mpa!==study.scene.source.amplitude_mpa*(i>=3?2:1))throw new AppError(422,'对照的物理参数或幅值不一致。');
    if(JSON.stringify(r.nodes_mm)!==JSON.stringify(base.nodes_mm)||JSON.stringify(r.tetra)!==JSON.stringify(base.tetra)||JSON.stringify(r.times_us)!==JSON.stringify(base.times_us)||r.diagnostics.internal_step_us!==base.diagnostics.internal_step_us||JSON.stringify(r.source_sha256)!==JSON.stringify(base.source_sha256))throw new AppError(422,'对照没有使用完全相同的网格、时间步和实现。');
    if(!Number.isFinite(r.diagnostics.max_export_roundoff_m)||r.diagnostics.max_export_roundoff_m<0)throw new AppError(422,'位移场量化误差记录不完整。');
  }
}

export class SolidStudies {
  constructor(owner){this.owner=owner;this.root=join(owner.root,'studies');this.probing=false;}
  async init(){await mkdir(this.root,{recursive:true});}
  path(id){if(!/^study-[a-f0-9]{32}$/.test(id))throw new AppError(400,'非线性对照编号无效。');return join(this.root,id);}
  async status(id){try{const s=JSON.parse(await readBounded(join(this.path(id),'status.json')));if(s.study_id!==id)throw new AppError(422,'对照身份不符。');return s;}catch(e){if(e.code==='ENOENT')throw new AppError(404,'没有找到该对照。');throw e;}}
  async list(){const list=[];for(const id of (await readdir(this.root)).filter(n=>/^study-[a-f0-9]{32}$/.test(n)))list.push(await this.status(id));return list.sort((a,b)=>String(b.started_at_utc).localeCompare(String(a.started_at_utc)));}
  async submit(body){return this.owner.exclusive(async()=>{
    await this.owner.assertIdle();const {scene}=await this.owner.command('validate-study',body),id='study-'+randomBytes(16).toString('hex'),path=this.path(id);await mkdir(path);
    await writeFile(join(path,'study.json'),JSON.stringify({schema:'workai.solid.study-input.v1',study_id:id,scene}),{flag:'wx'});
    // Each study executes its frozen package; later editing cannot change a running control.
    const frozen=join(path,'source_snapshot');await mkdir(join(frozen,'solid'),{recursive:true});
    for(const file of (await readdir(join(repo,'solid'))).filter(f=>f.endsWith('.py')))await copyFile(join(repo,'solid',file),join(frozen,'solid',file));
    await writeFile(join(path,'status.json'),JSON.stringify({study_id:id,phase:'starting',started_at_utc:new Date().toISOString(),completed_controls:0,total_controls:5}),{flag:'wx'});
    const log=await open(join(path,'worker.log'),'wx');
    const child=spawn(this.owner.python,['-m','solid.study','--study-dir',path],{cwd:frozen,detached:true,stdio:['ignore',log.fd,log.fd]});
    const fail=async message=>{try{const status=await this.status(id);if(!['completed','failed'].includes(status.phase))await writeFile(join(path,'status.json'),JSON.stringify({...status,phase:'failed',error:message}));}catch{}};
    child.on('error',()=>void fail('对照进程未能启动。'));child.on('exit',code=>void fail(`对照进程退出（${code}），未发布完整五组结果。`));child.unref();await log.close();return {study_id:id,scene};
  });}
  async result(id){
    const state=await this.status(id);if(state.phase!=='completed')throw new AppError(409,'五组对照尚未完成。');
    const bytes=await readBounded(join(this.path(id),'result.json'));if(hash(bytes)!==state.result_sha256)throw new AppError(422,'对照摘要与完成记录不符。');
    const r=JSON.parse(bytes);if(r.study_id!==id||r.schema!=='workai.solid.nonlinear-study.v1')throw new AppError(422,'对照身份不正确。');
    const config=await readBounded(join(this.path(id),'study.json'));if(hash(config)!==r.config_sha256)throw new AppError(422,'对照输入已改变。');
    const records=[];for(const c of r.controls){const state=await this.owner.status(c.job_id);if(state.result_sha256!==c.result_sha256)throw new AppError(422,'子任务身份已改变。');records.push(await this.owner.result(c.job_id));}
    validateControlSet(r,records);return {study:r,records};
  }
  async validation(id){
    let report;try{report=JSON.parse(await readBounded(join(this.path(id),'harmonic-validation.json')));}catch(error){if(error.code==='ENOENT')return null;throw error;}
    if(report.schema!=='workai.solid.harmonic-validation.v1'||report.threshold!==.05||report.field_precision!=='little_endian_float64'||report.references?.map(r=>r.level).join(',')!=='coarse,fine,timecheck'||report.references[2].study_id!==id||!Array.isArray(report.rows)||report.rows.length>32)throw new AppError(422,'谐波细化参考格式不正确。');
    for(const ref of report.references){const status=await this.status(ref.study_id),bytes=await readBounded(join(this.path(ref.study_id),'result.json'));if(status.phase!=='completed'||hash(bytes)!==ref.result_sha256||status.result_sha256!==ref.result_sha256)throw new AppError(422,'谐波细化参考摘要不匹配。');}
    for(const row of report.rows){const metrics=[row.spatial_with_changed_stable_step,row.temporal_same_mesh,row.quadratic_complex_error];if(!metrics.every(v=>Number.isFinite(v)&&v>=0)||!Number.isFinite(row.material_second_nm)||row.material_second_nm<=0||![row.point_mm,row.direction,row.window_us].every((v,i)=>Array.isArray(v)&&v.length===(i===2?2:3)&&v.every(Number.isFinite)))throw new AppError(422,'谐波细化指标不正确。');
      if(row.selected_point_checks_within_5_percent!==(row.above_export_roundoff===true&&metrics.every(v=>v<=.05)))throw new AppError(422,'谐波细化结论与指标不一致。');
    }
    return report;
  }
  async probe(id,body){
    if(this.probing)throw new AppError(409,'已有对照波形正在读取，请稍后重试。');
    if(!body||Object.keys(body).sort().join(',')!=='direction,point_mm'||!Array.isArray(body.point_mm)||body.point_mm.length!==3||!body.point_mm.every(Number.isFinite))throw new AppError(400,'请提供完整点位和方向。');
    let direction;try{direction=unitVector(body.direction);}catch(error){throw new AppError(400,error.message);}this.probing=true;
    try{
      const {study,records}=await this.result(id),controls={};
      for(let i=0;i<5;i++){
        const r=records[i],packed=await this.owner.field(r.job_id),raw=gunzipSync(packed,{maxOutputLength:r.field.uncompressed_bytes});
        if(raw.length!==r.field.uncompressed_bytes||hash(raw)!==r.field.uncompressed_sha256)throw new AppError(422,'完整位移场摘要不匹配。');
        const FieldType=r.field.dtype==='little_endian_float64'?Float64Array:Float32Array,field=new FieldType(raw.buffer,raw.byteOffset,raw.length/FieldType.BYTES_PER_ELEMENT);if(!field.every(Number.isFinite))throw new AppError(422,'位移场含无效数值。');
        let signal;try{signal=pointSeries(r,field,body.point_mm,direction);}catch(error){throw new AppError(422,error.message);}
        controls[SOLID_CONTROLS[i]]={job_id:r.job_id,result_sha256:study.controls[i].result_sha256,xyz_m:signal.xyz_m,projected_m:signal.projected_m,max_export_roundoff_m:r.diagnostics.max_export_roundoff_m};
      }
      let numerical_reference=null,reference_error=null;try{numerical_reference=await this.validation(id);}catch{reference_error='附加谐波细化参考未通过校验，未将其用于通过结论。';}
      let waveguide_reference=null,waveguide_reference_error=null;try{waveguide_reference=await readWaveguideReference(this.path(id),study);}catch{waveguide_reference_error='有限宽截面参考校验失败，未将其用于候选模态结论。';}
      return {schema:'workai.solid.harmonic-probe.v1',study_id:id,scene:study.scene,point_mm:body.point_mm,direction,times_us:records[0].times_us,controls,same_mesh_and_time_step:true,harmonic_convergence_verified:false,comsol_agreement_verified:false,numerical_reference,reference_error,waveguide_reference,waveguide_reference_error};
    }finally{this.probing=false;}
  }
}
