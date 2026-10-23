import {readFile,lstat} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';

// Additional evidence is bound to immutable completed results; it never changes them.
export async function readSolidValidation(jobsRoot,id,resultSha) {
  const path=join(jobsRoot,id,'validation.json');let evidence;
  try {const stat=await lstat(path);if(!stat.isFile()||stat.isSymbolicLink()||stat.size>100000)throw new Error('Invalid comparison record');evidence=JSON.parse(await readFile(path));}
  catch(error){if(error.code==='ENOENT')return null;throw error;}
  if(evidence.schema!=='workai.solid.comparison.v1'||evidence.job_id!==id||evidence.result_sha256!==resultSha||!Array.isArray(evidence.references)||evidence.references.length!==3)throw new Error('Comparison identity mismatch');
  for(const ref of evidence.references){if(!/^solid-[a-f0-9]{32}$/.test(ref.job_id)||!/^[a-f0-9]{64}$/.test(ref.result_sha256))throw new Error('Invalid comparison reference');
    const bytes=await readFile(join(jobsRoot,ref.job_id,'result.json'));
    if(createHash('sha256').update(bytes).digest('hex')!==ref.result_sha256)throw new Error('Comparison reference bytes changed');
  }
  for(const key of ['spatial_with_changed_stable_step','temporal_same_mesh']){
    const c=evidence[key];if(c?.threshold!==.05||!Array.isArray(c.rows)||c.rows.length<1||c.rows.length>32)throw new Error('Invalid comparison metrics');
    const passed=c.rows.every(r=>Number.isFinite(r.vector_relative_l2)&&r.vector_relative_l2>=0&&r.vector_relative_l2<=.05&&Number.isFinite(r.direction_relative_l2)&&r.direction_relative_l2>=0&&r.direction_relative_l2<=.05);
    if(passed!==c.selected_receivers_within_5_percent)throw new Error('Comparison decision does not match errors');
  }
  return evidence;
}
