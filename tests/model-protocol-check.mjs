// Local compatibility checks against the original uploaded PowerShell validator.
// Does not read the private key, contact the server, or submit a compute task.
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { defaultCase } from '../server/models.mjs';
import { RemoteClient } from '../server/remote.mjs';

const directory = await mkdtemp(join(tmpdir(), 'workai-model-protocol-'));
const remote = new RemoteClient(process.env.WORKAI_REMOTE_ROOT || '/workspace/workai-remote');
try {
  for (const mode of ['plate', 'volume']) {
    const modelCase = defaultCase(mode), file = join(directory, `${mode}.json`);
    await writeFile(file, JSON.stringify(modelCase));
    let result = await remote.validateCase(file); assert.equal(result.valid, true, JSON.stringify(result));
    modelCase.sources[0].position_mm[2] = 20;
    await writeFile(file, JSON.stringify(modelCase));
    result = await remote.validateCase(file); assert.equal(result.valid, false);
    console.log(`PASS ${mode}: original validator accepts template and rejects a source outside the solid.`);
  }
  const research = defaultCase('research'), researchFile = join(directory, 'research.json');
  await writeFile(researchFile, JSON.stringify(research));
  const result = await remote.validateCase(researchFile);
  assert.equal(result.valid, true, JSON.stringify(result));
  research.material.murnaghan_l_pa = -300e9;
  await writeFile(researchFile, JSON.stringify(research));
  assert.equal((await remote.validateCase(researchFile)).valid, false);
  console.log('PASS research: original research validator accepts compiled parameters; unsupported nonlinear constants are rejected.');
  const nonlinear = defaultCase('nonlinear'), nonlinearFile = join(directory, 'nonlinear.json');
  await writeFile(nonlinearFile, JSON.stringify(nonlinear));
  assert.equal((await remote.validateCase(nonlinearFile)).valid, true);
  nonlinear.nonlinear.l_pa = -143;
  await writeFile(nonlinearFile, JSON.stringify(nonlinear));
  assert.equal((await remote.validateCase(nonlinearFile)).valid, false);
  console.log('PASS nonlinear: five-control transport passes original validator; mismatched material units rejected.');
} finally { await rm(directory, { recursive: true, force: true }); }
