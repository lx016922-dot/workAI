import {readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {SolidService} from '../server/solid.mjs';
const [root,evidence]=process.argv.slice(2);if(!root||!evidence)throw new Error('Usage: node tools/link-solid-harmonic-validation.mjs ROOT EVIDENCE');
const independent=JSON.parse(await readFile(join(evidence,'independent-numpy.json')));if(independent.passed!==true)throw new Error('Independent numerical analysis check has not passed.');
const bytes=await readFile(join(evidence,'harmonic-validation.json')),record=JSON.parse(bytes),service=new SolidService(root);await service.init();
for(const ref of record.references){await service.studies.result(ref.study_id);const source=await readFile(join(service.studies.path(ref.study_id),'result.json'));if(createHash('sha256').update(source).digest('hex')!==ref.result_sha256)throw new Error('Reference identity changed.');}
const id=record.references.find(r=>r.level==='timecheck').study_id;await writeFile(join(service.studies.path(id),'harmonic-validation.json'),bytes,{flag:'wx'});await service.studies.validation(id);
console.log(JSON.stringify({study_id:id,separate_reference_attached:true,original_results_unchanged:true}));
