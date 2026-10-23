import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {buildSolidHarmonicReport} from '../public/solid-harmonic-report.js';
const [input,output]=process.argv.slice(2);if(!input||!output)throw new Error('Usage: node tools/export-solid-harmonics.mjs VERIFIED-PROBE.json NEW.html');
const probe=JSON.parse(await readFile(input)),resources={};
for(const [key,file] of Object.entries({css:'styles.css',data:'solid-data.js',dispersion:'plate-dispersion.js',propagation:'solid-propagation.js',harmonics:'solid-harmonics.js',viewer:'solid-harmonic-viewer.js'}))resources[key]=await readFile(new URL('../public/'+file,import.meta.url),'utf8');
const html=buildSolidHarmonicReport(probe,resources);await writeFile(output,html,{flag:'wx'});console.log(JSON.stringify({file:output,bytes:Buffer.byteLength(html),sha256:createHash('sha256').update(html).digest('hex'),study_id:probe.study_id,point_mm:probe.point_mm}));
