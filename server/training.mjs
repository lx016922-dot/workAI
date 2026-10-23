import { readFile, lstat } from 'node:fs/promises';
import { join } from 'node:path';
import {createHash} from 'node:crypto';

async function boundedFile(path, limit) {
  const st = await lstat(path);
  if (!st.isFile() || st.isSymbolicLink() || st.size > limit) throw new Error('训练记录文件无效或过大。');
  return readFile(path, 'utf8');
}

export async function readTrainingStatus(root, now = Date.now()) {
  let spectralRecord;
  try {spectralRecord=JSON.parse(await boundedFile(join(root,'spectralop/latest.json'),512*1024));}
  catch(error){if(error.code!=='ENOENT')throw error;}
  if(spectralRecord!==undefined)return readSpectralTraining(root,spectralRecord,now);
  let modalRecord;
  try {
    modalRecord=JSON.parse(await boundedFile(join(root,'modal3d/latest.json'),256*1024));
  } catch(error) {if(error.code!=='ENOENT')throw error;}
  if(modalRecord!==undefined)return await readModalTraining(root,modalRecord,now);
  try {
    const record=JSON.parse(await boundedFile(join(root,'training3d/latest.json'),256*1024));
    return await readOperatorTraining(root,record,now);
  } catch(error) {if(error.code!=='ENOENT')throw error;}
  let record;
  try { record = JSON.parse(await boundedFile(join(root, 'training/latest.json'), 256 * 1024)); }
  catch (error) { if (error.code === 'ENOENT') return { available: false }; throw error; }
  if (record.schema !== 'workai.physics_training.v1' || !/^[a-zA-Z0-9-]{1,100}$/.test(record.run_id) ||
      !['training', 'completed_experimental', 'failed'].includes(record.status) ||
      !Number.isInteger(record.step) || !Number.isInteger(record.steps_total) || record.step < 0 || record.step > record.steps_total) throw new Error('训练记录格式不正确。');
  const updated = record.updated_at_utc || record.started_at_utc;
  const stale = record.status === 'training' && (!Number.isFinite(Date.parse(updated)) || now - Date.parse(updated) > 120_000);
  let history = [];
  try {
    const text = await boundedFile(join(root, 'training/runs', record.run_id, 'history.jsonl'), 1024 * 1024);
    const lines = text.split('\n'); lines.pop();
    history = lines.slice(-250).map(line => JSON.parse(line)).map(row => Object.fromEntries(['step', 'total', 'data', 'pde', 'traction', 'validation_mse'].map(k => [k, Number.isFinite(row[k]) ? row[k] : null])));
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  return {
    available: true, run_id: record.run_id, status: stale ? 'stale' : record.status, step: record.step, steps_total: record.steps_total,
    started_at_utc: record.started_at_utc, updated_at_utc: updated, elapsed_s: record.elapsed_s || 0,
    physical_case_count: record.physical_case_count, constraints: record.constraints, losses: record.losses,
    validation: record.validation, metrics: record.metrics, history,
    accepted_for_solver_routing: false,
  };
}

async function readSpectralTraining(root,record,now){
  const id=record.run_id;
  if(record.schema!=='workai.spectral_operator_training.v1'||!/^[a-zA-Z0-9-]{1,100}$/.test(id)||!['preparing','training','evaluating','completed_experimental','failed'].includes(record.status)||record.accepted_for_solver_routing!==false||!Number.isInteger(record.step)||!Number.isInteger(record.steps_total)||record.step<0||record.step>record.steps_total||!['train','validation','test'].every(k=>Number.isInteger(record.split_counts?.[k])&&record.split_counts[k]>0)||Object.values(record.split_counts).reduce((a,b)=>a+b,0)!==record.physical_case_count)throw Error('高阶物理约束训练记录格式不正确。');
  const directory=join(root,'spectralop/runs',id),updated=record.updated_at_utc||record.started_at_utc,active=['preparing','training','evaluating'].includes(record.status),stale=active&&(!Number.isFinite(Date.parse(updated))||now-Date.parse(updated)>300000);
  let history=[],metrics;
  try {const lines=(await boundedFile(join(directory,'history.jsonl'),1024*1024)).split('\n');lines.pop();history=lines.slice(-250).map(line=>JSON.parse(line)).map(r=>({step:r.step,data:r.data,pde:r.interior,traction:r.boundary,validation_mse:r.validation?.normalized_mse}));}
  catch(error){if(error.code!=='ENOENT')throw error;}
  if(record.status==='completed_experimental'){
    metrics=JSON.parse(await boundedFile(join(directory,'metrics.json'),512*1024));const checkpoint=join(directory,'best.pt'),st=await lstat(checkpoint);
    if(!st.isFile()||st.isSymbolicLink()||st.size>32*1024*1024||metrics.schema!=='workai.spectral_operator_metrics.v1'||metrics.accepted_for_solver_routing!==false||metrics.test_used_for_checkpoint_selection!==false||metrics.acceptance?.threshold!==.05||!Array.isArray(metrics.cases)||metrics.cases.length>32||createHash('sha256').update(await readFile(checkpoint)).digest('hex')!==metrics.checkpoint_sha256)throw Error('高阶训练权重与评估记录不一致。');
    for(const row of metrics.cases)if(!['validation','test'].includes(row.split)||typeof row.case_id!=='string'||!Array.isArray(row.probes)||!row.field_checks||!Object.values(row.field_checks).every(v=>typeof v.pass==='boolean'&&(v.relative_l2===null||Number.isFinite(v.relative_l2)&&v.relative_l2>=0)))throw Error('高阶训练指标无效。');
  }
  return {available:true,kind:'spectral_operator',run_id:id,status:stale?'stale':record.status,step:record.step,steps_total:record.steps_total,elapsed_s:record.elapsed_s||0,physical_case_count:record.physical_case_count,split_counts:record.split_counts,trainable_parameters:record.trainable_parameters,validation:record.validation,constraints:record.constraints,history,metrics,accepted_for_solver_routing:false};
}

async function readModalTraining(root,record,now){
  const id=record.run_id;
  if(record.schema!=='workai.modal_energy_training.v1'||record.kind!=='modal_energy'||!/^[a-zA-Z0-9-]{1,100}$/.test(id)||!['training','trained_pending_new_test','completed_experimental','failed'].includes(record.status)||!Number.isInteger(record.step)||!Number.isInteger(record.steps_total)||record.step<0||record.step>record.steps_total||!['train','validation','test'].every(k=>Number.isInteger(record.split_counts?.[k])&&record.split_counts[k]>=0)||record.split_counts.train<1||record.split_counts.validation<1||Object.values(record.split_counts).reduce((a,b)=>a+b,0)!==record.physical_case_count)throw Error('能量耦合训练记录格式不正确。');
  const directory=join(root,'modal3d/runs',id),updated=record.updated_at_utc||record.started_at_utc,stale=record.status==='training'&&(!Number.isFinite(Date.parse(updated))||now-Date.parse(updated)>120000);
  const history=JSON.parse(await boundedFile(join(directory,'history.json'),1024*1024)).map(r=>Object.fromEntries(['step','train_relative_mse','validation_relative_mse'].map(k=>[k,Number.isFinite(r[k])?r[k]:null])));
  const config=JSON.parse(await boundedFile(join(directory,'config.json'),256*1024));let metrics,referenceChecks,referenceUpgrade;
  if(record.evaluation_completed){
    metrics=JSON.parse(await boundedFile(join(directory,'metrics.json'),2*1024*1024));const path=join(directory,'best.pt'),st=await lstat(path);
    if(!st.isFile()||st.isSymbolicLink()||st.size>256*1024||metrics.kind!=='modal_energy'||createHash('sha256').update(await readFile(path)).digest('hex')!==metrics.checkpoint_sha256)throw Error('能量耦合权重与评估记录不一致。');
    try {referenceChecks=JSON.parse(await boundedFile(join(directory,'reference-checks.json'),128*1024));if(referenceChecks.schema!=='workai.modal_reference_summary.v1'||referenceChecks.campaign_sha256!==metrics.evaluation_campaign_sha256)throw Error('参考解核验与测试工况不一致。');}catch(error){if(error.code!=='ENOENT')throw error;}
    metrics={selected_step:metrics.selected_step,checkpoint_sha256:metrics.checkpoint_sha256,acceptance:metrics.acceptance,model_scope:metrics.model_scope,cases:metrics.cases.map(r=>Object.fromEntries(['case_id','split','geometry','source','boundary','full_field_relative_l2','parts_field_relative_l2','probes','sampled_physics','all_saved_frame_guard','timing'].map(k=>[k,r[k]]))),accepted_for_solver_routing:false};
  }
  try {
    const upgrade=JSON.parse(await boundedFile(join(root,'spectral3d/latest.json'),128*1024));
    if(upgrade.schema!=='workai.spectral_reference_summary.v1'||upgrade.accepted_for_solver_routing!==false||upgrade.training_used_these_fields!==false||upgrade.threshold!==.05||!Array.isArray(upgrade.cases)||!upgrade.cases.length||upgrade.cases.length>32||!Number.isInteger(upgrade.completed_solves)||upgrade.completed_solves<1)throw Error('高阶参考解记录格式不正确。');
    const keys=['case_id','h_passed','p_passed','time_passed','max_h_full_vector','max_h_material_vector','max_p_material_vector','max_time_material_vector','all_component_checks_passed'];
    for(const row of upgrade.cases){
      if(typeof row.case_id!=='string'||row.case_id.length>100||!['h_passed','p_passed','time_passed','all_component_checks_passed'].every(k=>typeof row[k]==='boolean')||!['max_h_full_vector','max_h_material_vector','max_p_material_vector','max_time_material_vector'].every(k=>Number.isFinite(row[k])&&row[k]>=0))throw Error('高阶参考解指标无效。');
    }
    referenceUpgrade={completed_solves:upgrade.completed_solves,threshold:upgrade.threshold,training_used_these_fields:false,accepted_for_solver_routing:false,cases:upgrade.cases.map(r=>Object.fromEntries(keys.map(k=>[k,r[k]])))};
  }catch(error){if(error.code!=='ENOENT')throw error;}
  return {available:true,kind:'modal_energy',run_id:id,status:stale?'stale':record.status,step:record.step,steps_total:record.steps_total,elapsed_s:record.elapsed_s,physical_case_count:record.physical_case_count,split_counts:record.split_counts,evaluation_split_counts:record.evaluation_split_counts,multipliers:record.multipliers,history,metrics,reference_checks:referenceChecks,reference_upgrade:referenceUpgrade,trainable_parameters:config.trainable_parameters,modes:config.modes,dynamic_enrichment_vectors:config.dynamic_enrichment_vectors??0,accepted_for_solver_routing:false};
}

async function readOperatorTraining(root,record,now){
  if(record.schema!=='workai.operator_training.v1'||record.kind!=='mesh_conditioned_3d'||!/^[a-zA-Z0-9-]{1,100}$/.test(record.run_id)||!['preparing','training','evaluating','completed_experimental','failed'].includes(record.status)||!Number.isInteger(record.step)||!Number.isInteger(record.steps_total)||record.step<0||record.step>record.steps_total||!Number.isInteger(record.physical_case_count)||record.physical_case_count<3||!['train','validation','test'].every(k=>Number.isInteger(record.split_counts?.[k])&&record.split_counts[k]>0)||Object.values(record.split_counts).reduce((a,b)=>a+b,0)!==record.physical_case_count)throw Error('三维训练记录格式不正确。');
  const updated=record.updated_at_utc||record.started_at_utc,active=['preparing','training','evaluating'].includes(record.status),stale=active&&(!Number.isFinite(Date.parse(updated))||now-Date.parse(updated)>120_000);
  let history=[];
  try {const text=await boundedFile(join(root,'training3d/runs',record.run_id,'history.jsonl'),1024*1024);const lines=text.split('\n');lines.pop();history=lines.slice(-250).map(line=>JSON.parse(line)).map(row=>Object.fromEntries(['step','total','data','pde','traction','validation_mse'].map(k=>[k,Number.isFinite(row[k])?row[k]:null])));}
  catch(error){if(error.code!=='ENOENT')throw error;}
  return {available:true,kind:record.kind,run_id:record.run_id,status:stale?'stale':record.status,step:record.step,steps_total:record.steps_total,started_at_utc:record.started_at_utc,updated_at_utc:updated,elapsed_s:record.elapsed_s||0,physical_case_count:record.physical_case_count,split_counts:record.split_counts,constraints:record.constraints,discretization_verified:record.discretization_verified===true,losses:record.losses,validation:record.validation,metrics:record.metrics,history,accepted_for_solver_routing:false};
}
