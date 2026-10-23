import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { readTrainingStatus } from '../server/training.mjs';
import {renderTraining} from '../public/training-ui.js';
import {createHash} from 'node:crypto';

test('training status is read-only, omits process/paths, and does not promote an experimental checkpoint', async t => {
  const root = await mkdtemp(join(tmpdir(), 'workai-training-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  assert.deepEqual(await readTrainingStatus(root), { available: false });
  await mkdir(join(root, 'training/runs/pilot-1'), { recursive: true });
  const date = '2026-10-22T12:00:00Z';
  await writeFile(join(root, 'training/latest.json'), JSON.stringify({ schema: 'workai.physics_training.v1', run_id: 'pilot-1', status: 'training', step: 50, steps_total: 1500, updated_at_utc: date, pid: 123, run_dir: '/not-public', accepted_for_solver_routing: true }));
  await writeFile(join(root, 'training/runs/pilot-1/history.jsonl'), '{"step":50,"data":0.1,"internal":"not-public"}\n{"step":');
  const data = await readTrainingStatus(root, Date.parse(date));
  assert.equal(data.status, 'training'); assert.equal(data.accepted_for_solver_routing, false);
  assert.equal(data.history.length, 1); assert.ok(!JSON.stringify(data).includes('not-public')); assert.ok(!Object.hasOwn(data, 'pid'));
  assert.equal((await readTrainingStatus(root, Date.parse(date) + 121000)).status, 'stale');
});

test('3-D grouped training takes precedence without rewriting or enabling the older model',async t=>{
  const root=await mkdtemp(join(tmpdir(),'workai-operator-training-'));t.after(()=>rm(root,{recursive:true,force:true}));await mkdir(join(root,'training3d/runs/three-d'),{recursive:true});await mkdir(join(root,'training'));
  await writeFile(join(root,'training/latest.json'),'old record retained');const date='2026-10-23T02:00:00Z';const record={schema:'workai.operator_training.v1',kind:'mesh_conditioned_3d',run_id:'three-d',status:'training',step:100,steps_total:8000,physical_case_count:32,split_counts:{train:16,validation:8,test:8},updated_at_utc:date,pid:123,run_dir:'/not-public',accepted_for_solver_routing:true};
  await writeFile(join(root,'training3d/latest.json'),JSON.stringify(record));await writeFile(join(root,'training3d/runs/three-d/history.jsonl'),'{"step":100,"data":0.1,"secret":"not-public"}\n');const result=await readTrainingStatus(root,Date.parse(date));assert.equal(result.kind,'mesh_conditioned_3d');assert.equal(result.physical_case_count,32);assert.equal(result.accepted_for_solver_routing,false);assert.ok(!JSON.stringify(result).includes('not-public'));assert.equal((await readTrainingStatus(root,Date.parse(date)+121000)).status,'stale');
  for(const mutate of [r=>r.run_id='../private',r=>r.split_counts.test=7,r=>r.step=9000]){const bad=structuredClone(record);mutate(bad);await writeFile(join(root,'training3d/latest.json'),JSON.stringify(bad));await assert.rejects(()=>readTrainingStatus(root));}
});

test('3-D training page describes weak 3-D physics and retains failure/read-error states',()=>{
  const escape=v=>String(v??'').replaceAll('<','&lt;'),heading=(a,b)=>`<h1>${escape(a)}</h1><p>${escape(b)}</p>`;
  const training={kind:'mesh_conditioned_3d',status:'completed_experimental',step:2000,steps_total:2000,physical_case_count:16,split_counts:{train:8,validation:4,test:4},run_id:'pilot',history:[],elapsed_s:1,metrics:{selected_step:650,acceptance:{threshold:.05,all_gates_passed:false},cases:[]}};
  const html=renderTraining({training},escape,heading);assert.match(html,/本轮尚未通过代理验收/);assert.match(html,/没有强制纯 S₀ 模态或 A²/);assert.match(html,/弱残差/);assert.ok(!html.includes('40 mm'));assert.ok(!html.includes('w = 0'));
  const failed=renderTraining({training,trainingError:'读取失败'},escape,heading);assert.match(failed,/读取失败/);assert.ok(!failed.includes('本轮训练完成'));
});

test('malformed or path-traversing training records do not become successful training status', async t => {
  const root = await mkdtemp(join(tmpdir(), 'workai-training-'));
  t.after(() => rm(root, { recursive: true, force: true })); await mkdir(join(root, 'training'));
  for (const run_id of ['../private', '/tmp/other', '']) {
    await writeFile(join(root, 'training/latest.json'), JSON.stringify({ schema: 'workai.physics_training.v1', run_id, status: 'training', step: 1, steps_total: 10 }));
    await assert.rejects(() => readTrainingStatus(root));
  }
});

test('modal training is identified as a hybrid and cannot enable solver routing',async t=>{
  const root=await mkdtemp(join(tmpdir(),'workai-modal-training-'));t.after(()=>rm(root,{recursive:true,force:true}));const directory=join(root,'modal3d/runs/hybrid-1');await mkdir(directory,{recursive:true});
  const checkpoint=Buffer.from('fixture checkpoint'),sha=createHash('sha256').update(checkpoint).digest('hex');await writeFile(join(directory,'best.pt'),checkpoint);
  await writeFile(join(directory,'history.json'),JSON.stringify([{step:30,train_relative_mse:.00001,validation_relative_mse:.00002,private_path:'/not-public'}]));await writeFile(join(directory,'config.json'),JSON.stringify({modes:160,trainable_parameters:3}));await writeFile(join(directory,'metrics.json'),JSON.stringify({kind:'modal_energy',checkpoint_sha256:sha,selected_step:30,cases:[],acceptance:{threshold:.05,held_out_full_fields:true,all_gates_passed:false},accepted_for_solver_routing:true}));
  const record={schema:'workai.modal_energy_training.v1',kind:'modal_energy',run_id:'hybrid-1',status:'completed_experimental',step:1000,steps_total:1000,physical_case_count:56,split_counts:{train:48,validation:8,test:0},evaluation_completed:true,multipliers:[1,1,1],elapsed_s:3};await writeFile(join(root,'modal3d/latest.json'),JSON.stringify(record));const data=await readTrainingStatus(root);assert.equal(data.kind,'modal_energy');assert.equal(data.accepted_for_solver_routing,false);assert.equal(data.metrics.accepted_for_solver_routing,false);assert.ok(!JSON.stringify(data).includes('not-public'));
  const html=renderTraining({training:data},String,(a,b)=>`<h1>${a}</h1><p>${b}</p>`);assert.match(html,/三个有界能量耦合系数/);assert.match(html,/不是纯神经网络/);assert.match(html,/尚未通过完整替代验收/);assert.ok(!html.includes('共享网络读取'));
  await mkdir(join(root,'spectral3d'));const upgrade={schema:'workai.spectral_reference_summary.v1',accepted_for_solver_routing:false,training_used_these_fields:false,threshold:.05,completed_solves:45,cases:[{case_id:'notch',h_passed:true,p_passed:true,time_passed:true,all_component_checks_passed:true,max_h_full_vector:.01,max_h_material_vector:.02,max_p_material_vector:.01,max_time_material_vector:.0001,private_path:'/not-public'}]};
  await writeFile(join(root,'spectral3d/latest.json'),JSON.stringify(upgrade));const improved=await readTrainingStatus(root);assert.equal(improved.reference_upgrade.completed_solves,45);assert.equal(improved.accepted_for_solver_routing,false);assert.equal(improved.metrics.acceptance.all_gates_passed,false);assert.ok(!JSON.stringify(improved).includes('not-public'));
  const improvedHtml=renderTraining({training:improved},String,(a,b)=>`<h1>${a}</h1><p>${b}</p>`);assert.match(improvedHtml,/这些新结果尚未加入当前模型训练/);
  await writeFile(join(root,'spectral3d/latest.json'),JSON.stringify({...upgrade,training_used_these_fields:true}));await assert.rejects(()=>readTrainingStatus(root),/高阶参考解记录格式/);await rm(join(root,'spectral3d/latest.json'));
  await writeFile(join(directory,'best.pt'),'changed');await assert.rejects(()=>readTrainingStatus(root),/权重与评估记录不一致/);
  await rm(join(directory,'best.pt'));await assert.rejects(()=>readTrainingStatus(root),{code:'ENOENT'});
});

test('spectral training displays genuine network steps and verifies frozen checkpoint',async t=>{
  const root=await mkdtemp(join(tmpdir(),'workai-spectral-training-'));t.after(()=>rm(root,{recursive:true,force:true}));const directory=join(root,'spectralop/runs/spectral-1');await mkdir(directory,{recursive:true});
  const record={schema:'workai.spectral_operator_training.v1',run_id:'spectral-1',status:'training',step:100,steps_total:8000,physical_case_count:16,split_counts:{train:8,validation:4,test:4},accepted_for_solver_routing:false,updated_at_utc:'2026-10-23T12:00:00Z'};
  await writeFile(join(root,'spectralop/latest.json'),JSON.stringify(record));await writeFile(join(directory,'history.jsonl'),JSON.stringify({step:100,data:.1,interior:.2,boundary:.3,validation:{normalized_mse:.4},private_path:'/not-public'})+'\n');
  let status=await readTrainingStatus(root,Date.parse(record.updated_at_utc));assert.equal(status.kind,'spectral_operator');assert.equal(status.history[0].pde,.2);assert.equal(status.accepted_for_solver_routing,false);assert.ok(!JSON.stringify(status).includes('not-public'));
  const checkpoint=Buffer.from('spectral checkpoint');await writeFile(join(directory,'best.pt'),checkpoint);await writeFile(join(directory,'metrics.json'),JSON.stringify({schema:'workai.spectral_operator_metrics.v1',checkpoint_sha256:createHash('sha256').update(checkpoint).digest('hex'),test_used_for_checkpoint_selection:false,accepted_for_solver_routing:false,acceptance:{threshold:.05,all_gates_passed:false,reference_h_p_time_and_fields:true},cases:[],selected_step:1000}));record.status='completed_experimental';record.step=8000;await writeFile(join(root,'spectralop/latest.json'),JSON.stringify(record));status=await readTrainingStatus(root);
  const page=renderTraining({training:status},String,(a,b)=>`<h1>${a}</h1><p>${b}</p>`);assert.match(page,/高阶 GLL/);assert.match(page,/本轮模型未通过代理验收/);assert.ok(!page.includes('实际四面体'));assert.ok(!page.includes('这些新结果尚未加入'));
  await writeFile(join(directory,'best.pt'),'tampered');await assert.rejects(()=>readTrainingStatus(root),/权重与评估记录不一致/);
  record.run_id='../private';await writeFile(join(root,'spectralop/latest.json'),JSON.stringify(record));await assert.rejects(()=>readTrainingStatus(root),/格式/);
});
