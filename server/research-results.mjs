import { readFileSync } from 'node:fs';
import { decodeText } from './remote.mjs';
import { verifyResearchRequest, parseResearchCsv } from './research.mjs';
import { analyzeHarmonics } from './physics/harmonics.mjs';
import { attachSpatialReference } from './physics/convergence.mjs';
const referenceRecord = JSON.parse(readFileSync(new URL('../data/research/nonlinear-spatial-reference-20261022.json', import.meta.url)));

export async function readResearchResults(remote, id, withReference = true) {
  const names = ['request.json', 'completed.properties', 'coarse.csv', 'fine.csv', 'timecheck.csv'];
  const files = await Promise.all(names.map(name => remote.artifact(id, name)));
  const metadata = verifyResearchRequest(decodeText(files[0].bytes), decodeText(files[1].bytes), id), levels = {};
  for (const [index, level] of ['coarse', 'fine', 'timecheck'].entries()) levels[level] = parseResearchCsv(decodeText(files[index + 2].bytes), metadata.parameters);
  const data = { verified: true, job_id: id, ...metadata, levels, artifacts: files.map(f => ({ name: f.name, sha256: f.sha256, bytes: f.bytes.length })) };
  data.harmonics = data.constitutive === 'murnaghan_green_lagrange' ? analyzeHarmonics(data.scene, levels, data.receipt) : null;
  if (withReference && id === referenceRecord.target_job_id) {
    const status = await remote.job(referenceRecord.reference_job_id);
    if (status.kind !== 'research_reference' || status.phase !== 'completed') throw new Error('已绑定的空间参考任务尚未完成。');
    const reference = await readResearchResults(remote, referenceRecord.reference_job_id, false);
    data.harmonics = attachSpatialReference(data, reference, referenceRecord);
  }
  return data;
}
