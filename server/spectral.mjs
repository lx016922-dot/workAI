import { spawn } from 'node:child_process';
import { mkdir, readdir, readFile, writeFile, lstat, open } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomBytes } from 'node:crypto';
import { AppError } from './remote.mjs';

const repository = fileURLToPath(new URL('../', import.meta.url));
const validId = id => /^spectral-[a-f0-9]{32}$/.test(id);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function bounded(path, max) {
  const stat = await lstat(path);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > max) throw new AppError(422, '高阶结果文件无效或超过大小限制。');
  return readFile(path);
}

export class SpectralService {
  constructor(root, solid) { this.jobs = join(root, 'spectral3d/jobs'); this.solid = solid; this.python = join(root, 'solid/venv/bin/python'); }
  async init() { await mkdir(this.jobs, {recursive:true}); }
  path(id) { if (!validId(id)) throw new AppError(400, '高阶任务编号格式不正确。'); return join(this.jobs, id); }
  async status(id) {
    try { const result=JSON.parse(await bounded(join(this.path(id),'status.json'),256*1024)); if(result.job_id!==id)throw new AppError(422,'高阶任务身份不符。');return result; }
    catch(error) { if(error.code==='ENOENT')throw new AppError(404,'高阶任务不存在。');throw error; }
  }
  async list() {
    const rows=[];for(const id of (await readdir(this.jobs)).filter(validId))rows.push(await this.status(id));
    return rows.sort((a,b)=>String(b.started_at_utc).localeCompare(String(a.started_at_utc)));
  }
  async assertIdle() { if((await this.list()).some(r=>!['completed','failed'].includes(r.phase)))throw new AppError(409,'已有高阶计算正在运行或状态待确认。'); }
  async submit(body) {
    if(!body||Object.keys(body).sort().join(',')!=='degree,scene'||![3,4].includes(body.degree))throw new AppError(400,'请提供场景及 3 或 4 阶设置。');
    return this.solid.exclusive(async()=>{
      await this.solid.assertIdle();await this.assertIdle();
      const input=structuredClone(body.scene),h=input?.numerics?.h_mm;
      if(typeof h!=='number'||!Number.isFinite(h)||h<.02||h>100)throw new AppError(422,'高阶单元尺寸必须在 0.02–100 mm 范围内。');
      // The existing prefilter uses linear-element spacing. For p-th order
      // elements use h/p only as an initial resolution screen, not convergence.
      input.numerics.h_mm=Math.max(.02,h/body.degree);
      const scene=await this.solid.command('validate',input);scene.numerics.h_mm=h;
      if(!['plate','pipe','hole','notch'].includes(scene.geometry.kind))throw new AppError(422,'高阶计算支持参数化板、管、通孔和缺口。');
      // Version-2 scenes (signals, extra sources, symmetry/absorbing faces) are admitted by the Python validator and worker.
      if(scene.schema!=='workai.hyperelastic.solid.v2'&&(!['xmin','xmax'].includes(scene.source.face)||scene.source.radius_mm!==0||!['none','xmin','xmax'].includes(scene.boundary.fixed_face)))throw new AppError(422,'当前高阶计算使用完整 X 端面加载，固定面可选另一 X 端面。');
      const id='spectral-'+randomBytes(16).toString('hex'),path=this.path(id);await mkdir(path);
      await writeFile(join(path,'scene.json'),JSON.stringify(scene),{flag:'wx'});await writeFile(join(path,'settings.json'),JSON.stringify({degree:body.degree}),{flag:'wx'});
      await writeFile(join(path,'status.json'),JSON.stringify({job_id:id,phase:'starting',started_at_utc:new Date().toISOString()}),{flag:'wx'});
      const log=await open(join(path,'worker.log'),'wx');
      const child=spawn(this.python,['-m','spectral3d.worker','--job-dir',path],{cwd:repository,env:{...process.env,OPENBLAS_NUM_THREADS:'1',OMP_NUM_THREADS:'1'},detached:true,stdio:['ignore',log.fd,log.fd]});
      const fail=async message=>{try{const prior=await this.status(id);if(!['completed','failed'].includes(prior.phase))await writeFile(join(path,'status.json'),JSON.stringify({...prior,phase:'failed',error:message}));}catch{}};
      child.on('error',()=>void fail('高阶求解进程未能启动。'));child.on('exit',code=>void fail(`高阶求解进程退出（${code}），未发布完整结果。`));
      child.unref();await log.close();return {job_id:id,scene,degree:body.degree};
    });
  }
  async result(id) {
    const status=await this.status(id);if(status.phase!=='completed')throw new AppError(409,'高阶计算尚未完成。');
    const bytes=await bounded(join(this.path(id),'result.json'),24*1024*1024);
    if(sha(bytes)!==status.result_sha256)throw new AppError(422,'高阶结果摘要不匹配。');
    const result=JSON.parse(bytes);
    if(result.schema!=='workai.spectral_reference.result.v1'||result.engine!=='workai_spectral_reference_3d'||result.job_id!==id||result.accepted_for_solver_routing!==false)throw new AppError(422,'高阶结果身份不匹配。');
    if(sha(await bounded(join(this.path(id),'scene.json'),65536))!==result.scene_sha256||sha(await bounded(join(this.path(id),'settings.json'),4096))!==result.settings_sha256)throw new AppError(422,'高阶场景或阶次设置摘要不匹配。');
    return result;
  }
  async field(id) {
    const result=await this.result(id),bytes=await bounded(join(this.path(id),'field.bin.gz'),192*1024*1024);
    if(bytes.length!==result.field.compressed_bytes||sha(bytes)!==result.field.sha256)throw new AppError(422,'高阶位移场摘要不匹配。');
    return bytes;
  }
}
