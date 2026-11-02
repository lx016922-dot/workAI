import { spawn } from 'node:child_process';
import { mkdir, readdir, readFile, writeFile, lstat, open } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomBytes } from 'node:crypto';
import { AppError } from './remote.mjs';
import { readSolidValidation } from './solid-validation.mjs';
import { SolidStudies } from './solid-studies.mjs';

const repository = fileURLToPath(new URL('../', import.meta.url));
const validId = id => /^solid-[a-f0-9]{32}$/.test(id);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function bounded(path, max) {
  const stat = await lstat(path);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > max) throw new AppError(422, '三维任务文件无效或超出大小限制。');
  return readFile(path);
}

export class SolidService {
  constructor(root) { this.root = join(root,'solid'); this.jobs = join(this.root,'jobs'); this.python = join(this.root,'venv/bin/python'); this.submitting = false; this.studies=new SolidStudies(this); }
  async init() { await mkdir(this.jobs,{recursive:true}); await this.studies.init(); }
  async exclusive(action){if(this.submitting)throw new AppError(409,'已有三维任务正在提交。');this.submitting=true;try{return await action();}finally{this.submitting=false;}}
  async assertIdle(){if([...(await this.list()),...(await this.studies.list())].some(j=>!['completed','failed'].includes(j.phase)))throw new AppError(409,'已有三维计算或非线性对照正在运行或状态待确认，请先查看该任务。');if(this.spectral)await this.spectral.assertIdle();if(this.plane)await this.plane.assertIdle();}
  async upload(body) {
    if(!body||Object.keys(body).sort().join(',')!=='data_base64,filename'||typeof body.filename!=='string'||body.filename.length>160||!/^.+\.(step|stp|stl)$/i.test(body.filename)||typeof body.data_base64!=='string'||body.data_base64.length>12*1024*1024||! /^[A-Za-z0-9+/]*={0,2}$/.test(body.data_base64))throw new AppError(400,'请上传不超过 8 MiB 的 STEP 或 STL 几何文件。');
    const bytes=Buffer.from(body.data_base64,'base64');if(bytes.length<20||bytes.length>8*1024*1024||bytes.toString('base64')!==body.data_base64)throw new AppError(400,'几何文件编码或大小无效。');
    const format=body.filename.toLowerCase().endsWith('.stl')?'stl':'step';
    if(format==='step'&&!bytes.subarray(0,256).toString('ascii').includes('ISO-10303-21'))throw new AppError(422,'STEP 文件头不正确。');
    const hash=sha(bytes),id='asset-'+hash,path=join(this.root,'assets',id);await mkdir(path,{recursive:true});
    try{await writeFile(join(path,'geometry.'+format),bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;const existing=await bounded(join(path,'geometry.'+format),8*1024*1024);if(sha(existing)!==hash)throw new AppError(422,'已有几何文件摘要不符。');}
    const metadata={asset_id:id,sha256:hash,format,original_name:body.filename,bytes:bytes.length};
    try{await writeFile(join(path,'metadata.json'),JSON.stringify(metadata),{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;}
    return metadata;
  }
  async command(action, value) {
    return new Promise((resolve,reject) => {
      const child=spawn(this.python,['-m','solid.cli',action,'--root',dirname(this.root)],{cwd:repository,stdio:['pipe','pipe','pipe']});
      let out='',error='';const timer=setTimeout(()=>{ child.kill('SIGTERM'); reject(new AppError(504,'三维参数检查超时。')); },20000);
      child.stdout.on('data',c=>{out+=c;if(out.length>100000)child.kill('SIGTERM');});
      child.stderr.on('data',c=>{error=(error+c).slice(-2000);});
      child.on('error',()=>{clearTimeout(timer);reject(new AppError(503,'三维求解环境尚未安装。'));});
      child.on('close',()=>{clearTimeout(timer);try { const result=JSON.parse(out);if(!result.ok)reject(new AppError(422,result.error));else resolve(result.value); }catch{ reject(new AppError(503,'三维参数检查未完成。')); }});
      child.stdin.on('error',()=>{});child.stdin.end(value===undefined?'':JSON.stringify(value));
    });
  }
  path(id) { if(!validId(id))throw new AppError(400,'三维任务编号格式不正确。');return join(this.jobs,id); }
  async status(id) {
    try { const st=JSON.parse(await bounded(join(this.path(id),'status.json'),256*1024));if(st.job_id!==id)throw new Error();return st; }
    catch(error){if(error.code==='ENOENT')throw new AppError(404,'三维任务不存在。');throw error;}
  }
  async list() {
    const names=(await readdir(this.jobs)).filter(validId),jobs=[];
    for(const id of names)jobs.push(await this.status(id));
    return jobs.sort((a,b)=>String(b.started_at_utc).localeCompare(String(a.started_at_utc)));
  }
  async submit(body) {
    if(this.submitting)throw new AppError(409,'已有三维任务正在提交。');this.submitting=true;
    try {
      await this.assertIdle();
      const scene=await this.command('validate',body),id='solid-'+randomBytes(16).toString('hex'),path=this.path(id);
      await mkdir(path);await writeFile(join(path,'scene.json'),JSON.stringify(scene),{flag:'wx'});
      await writeFile(join(path,'status.json'),JSON.stringify({job_id:id,phase:'starting',started_at_utc:new Date().toISOString()}),{flag:'wx'});
      const log=await open(join(path,'worker.log'),'wx');
      const child=spawn(this.python,['-m','solid.worker','--job-dir',path],{cwd:repository,detached:true,stdio:['ignore',log.fd,log.fd]});
      child.on('error',async()=>{await writeFile(join(path,'status.json'),JSON.stringify({job_id:id,phase:'failed',error:'三维求解进程未能启动。'}));});
      child.on('exit',async code=>{try {const state=await this.status(id);if(!['completed','failed'].includes(state.phase))await writeFile(join(path,'status.json'),JSON.stringify({...state,phase:'failed',error:`求解进程退出（${code}），未发布完整结果。`}));}catch{}});
      child.unref();await log.close();return {job_id:id,scene};
    } finally {this.submitting=false;}
  }
  async result(id) {
    const st=await this.status(id);if(st.phase!=='completed')throw new AppError(409,'该三维计算尚未完成。');
    const bytes=await bounded(join(this.path(id),'result.json'),12*1024*1024);
    if(sha(bytes)!==st.result_sha256)throw new AppError(422,'三维结果摘要与完成记录不符。');
    const result=JSON.parse(bytes);if(result.schema!=='workai.hyperelastic.result.v1'||result.job_id!==id)throw new AppError(422,'三维结果身份不匹配。');
    const scene=await bounded(join(this.path(id),'scene.json'),65536);
    if(sha(scene)!==result.scene_sha256)throw new AppError(422,'三维场景摘要不匹配。');
    try{result.validation=await readSolidValidation(this.jobs,id,st.result_sha256);}catch{result.validation=null;result.validation_error='附加收敛记录未通过身份与文件校验。';}
    return result;
  }
  async field(id) {
    const result=await this.result(id);const bytes=await bounded(join(this.path(id),'field.bin.gz'),192*1024*1024);
    if(bytes.length!==result.field.compressed_bytes||sha(bytes)!==result.field.sha256)throw new AppError(422,'三维位移场摘要不匹配。');
    return bytes;
  }
}
