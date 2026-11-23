// Stage C: the nonlinear guided-wave analysis of a finished record (S0 in a 1 mm plate), and the same plate with material damping.
const { open } = require('./lib.cjs');
const fs = require('fs');
(async () => {
  const out = process.argv[2]; const w = await open(out); const { page, shot, wait } = w; const files = out + '/files'; fs.mkdirSync(files, { recursive: true });
  const only = process.argv.slice(3); await w.tall(); await w.openJob('1 mm 板 S0 · 二次谐波累积');
  if (only.includes('convergence')) { await w.nlTab('convergence'); await page.waitForFunction(() => { const t = document.querySelector('#nl-convergence')?.textContent || ''; return /5 阶/.test(t) && !/进行中|计算中|只有一套|正在检查/.test(t); }, null, { timeout: 1500000, polling: 5000 }); await wait(3000); await shot('c05-convergence', '#nl-convergence'); console.log((await page.locator('#nl-convergence').innerText()).slice(0, 900)); await w.close(); return; }
  await page.locator('#nl-run').click(); await page.waitForFunction(() => document.querySelector('#nl-accumulation svg'), null, { timeout: 180000 }); await wait(800);
  await w.summaryShot('c01-accumulation');
  const chart = page.locator('#nl-accumulation svg').first(); await chart.hover(); await wait(400); await w.chartsShot('c02-charts');
  for (const kind of ['png', 'svg']) { await chart.hover(); const file = await w.download(page.locator(`.figure-tools [data-kind="${kind}"]`), files); fs.copyFileSync(file, `${files}/accumulation.${kind}`); }
  fs.copyFileSync(await w.download(page.locator('#nl-csv'), files), files + '/accumulation.csv');
  await w.nlTab('dispersion'); await page.waitForFunction(() => document.querySelector('#nl-theory table'), null, { timeout: 180000 }); await wait(800);
  await page.locator('#nl-part').selectOption('nonlinear'); await page.locator('#nl-fk-run').click();
  await wait(400); await page.waitForFunction(() => !document.querySelector('#nl-fk')?.hidden && !/正在/.test(document.querySelector('#nl-fk-message')?.textContent || ''), null, { timeout: 240000 }); await wait(2500);
  await w.theoryShot('c03-dispersion'); await w.fkShot('c03-fk');
  await w.nlTab('convergence'); await page.waitForFunction(() => document.querySelector('#nl-converge'), null, { timeout: 120000 }); await wait(500); await shot('c04-convergence-plan', '#nl-convergence');
  if (!(await page.locator('#nl-open-check').count())) { await page.locator('#nl-converge').click(); await wait(3000); }
  await page.waitForFunction(() => { const t = document.querySelector('#nl-convergence')?.textContent || ''; return /5 阶/.test(t) && !/进行中|计算中|只有一套/.test(t); }, null, { timeout: 1500000, polling: 5000 }).catch(e => console.log('convergence wait', String(e).slice(0, 80)));
  await wait(3000); await shot('c05-convergence', '#nl-convergence');
  await w.nlTab('accumulation'); const report = await w.download(page.locator('#report'), files); fs.copyFileSync(report, files + '/report.html'); console.log('report', report);
  // the same plate with material damping
  await w.openJob('1 mm 板 S0 · 有材料阻尼'); await page.locator('#nl-run').click(); await page.waitForFunction(() => document.querySelector('#nl-accumulation svg'), null, { timeout: 180000 }); await wait(800);
  await w.clip('c06-accumulation-damped', '#nl-panel', '#nl-accumulation table.nl-summary', '#nl-accumulation table.nl-summary'); await w.chartsShot('c06-charts-damped');
  await w.nlTab('dispersion'); await page.waitForFunction(() => document.querySelector('#nl-theory table'), null, { timeout: 180000 }); await wait(800); await w.theoryShot('c07-dispersion-damped');
  console.log('TEXT', (await page.locator('#nl-convergence').innerText().catch(() => '')).slice(0, 600));
  await w.close();
})().catch(e => { console.error('FAILED', e); process.exit(1); });
