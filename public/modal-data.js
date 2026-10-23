import {validateDisplacementLayout,decodeDisplacementField} from './solid-data.js';

export function validateModalResult(r){
  if(r?.schema!=='workai.modal_prediction.result.v1'||r.engine!=='workai_modal_energy_3d'||r.accepted_for_solver_routing!==false||!['full','linear','geometric','material'].includes(r.response_part)||!r.evidence?.run_id||!(/^[a-f0-9]{64}$/).test(r.evidence.checkpoint_sha256)||r.evidence.transient_labels_used_for_prediction!==false)throw Error('实验性降阶结果或权重记录不正确。');
  return validateDisplacementLayout(r);
}

export async function decodeModalField(r,bytes){validateModalResult(r);return decodeDisplacementField(r,bytes);}
