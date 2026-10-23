// Input is an analysis JSON previously verified by WorkAI, not a new signature proof.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { buildOfflineReport } from '../public/offline-report.js';
const [input, output] = process.argv.slice(2);
if (!input || !output || resolve(input) === resolve(output)) throw new Error('Usage: node tools/export-offline-results.mjs verified-analysis.json new-report.html');
const data = JSON.parse(await readFile(input, 'utf8'));
const css = await readFile(new URL('../public/styles.css', import.meta.url), 'utf8');
await writeFile(output, buildOfflineReport(data, { css }), { flag: 'wx' });
console.log(`Saved offline result: ${resolve(output)}`);
