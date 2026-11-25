// Stage B: computing in the workbench and the result page — the template model, the parameter scan, the fast engine, pause and resume, a pipe.
const { open } = require('./lib.cjs');
const fs = require('fs');
(async () => {
  const out = process.argv[2]; const only = process.argv.slice(3); const want = id => !only.length || only.includes(id);
  const w = await open(out); const { page, shot, wait } = w; const files = out + '/files'; fs.mkdirSync(files, { recursive: true });
  const finished = () => page.waitForFunction(() => document.querySelector('#result-head .wb-phase.completed'), null, { timeout: 1800000, polling: 2000 });
  const compute = async () => { await w.ribbon('home'); await w.cmd('run'); };
  if (want('run')) {
    await w.template('gaussian'); await compute(); await page.waitForFunction(() => (document.querySelector('#result-head progress')?.value || 0) >= 25, null, { timeout: 600000, polling: 500 }); await shot('b01-running'); await finished(); await wait(4000); await shot('b02-finished');
    await w.tall(); await w.side(); await w.frame(.5); await w.clip('b03-field', '#results', '#result-head', w.section('三维位移场动画'));
    await page.locator('#solid-component').selectOption('2'); await w.frame(.5); await shot('b04-field-z', w.section('三维位移场动画'));
    await shot('b05-probe', w.section('任意点位与方向')); await w.clip('b05-waveform', '#results', w.section('方向位移波形'), w.section('单边幅值频谱'));
    await page.locator('#part').selectOption('material'); await page.waitForSelector('#solid-canvas'); await wait(3500); await shot('b06-head', '#result-head'); await w.clip('b06-material-increment', '#results', w.section('方向位移波形'), w.section('单边幅值频谱'));
    await page.locator('#part').selectOption('full'); await page.waitForSelector('#solid-canvas'); await wait(3500);
    await page.locator('#solid-path-presets').click(); await wait(500); await page.locator('#solid-path-analyze').click();
    await page.waitForFunction(() => document.querySelector('#solid-path-table tr td, #solid-path-table tbody tr'), null, { timeout: 120000 }); await wait(1500); await shot('b07-path', '#solid-path-panel');
    const id = await page.evaluate(() => document.querySelector('#result-head a[href*="/export"]').getAttribute('href')); const zip = await page.request.get('http://127.0.0.1:8765' + id); fs.writeFileSync(files + '/result.zip', await zip.body());
    await page.setViewportSize({ width: 1440, height: 900 }); await wait(800);
  }
  if (want('scan')) {
    await w.template('gaussian'); await w.node('scan');
    for (const [i, param, values] of [[0, 'source.amplitude_mpa', '1, 2'], [1, 'source.frequency_mhz', '0.3, 0.4']]) {
      await page.locator('[data-scan-add]').click(); await wait(400); await page.locator(`[data-scan-path="${i}"]`).selectOption(param); await wait(400);
      await page.locator(`[data-scan-values="${i}"]`).fill(values); await page.locator(`[data-scan-values="${i}"]`).blur(); await wait(600);
    }
    await page.waitForSelector('[data-scan-start]:not([disabled])', { timeout: 120000 }); await wait(600); await shot('b10-scan-plan');
    await page.locator('[data-scan-start]').click(); await wait(500); await shot('b11-scan-confirm'); await page.getByRole('button', { name: '开始', exact: true }).click(); await wait(15000); await shot('b12-scan-running');
    await page.waitForFunction(() => /已完成|完成 4|4 \/ 4/.test(document.querySelector('#result-head')?.textContent || '') && !document.querySelector('#result-head progress'), null, { timeout: 1800000, polling: 3000 }); await wait(5000);
    await w.tall(2600); await shot('b13-scan-summary', '#tab-result'); fs.writeFileSync(files + '/scan.csv', await (await page.request.get('http://127.0.0.1:8765' + await page.evaluate(() => document.querySelector('[data-scan-csv]').getAttribute('href')))).body());
    await page.setViewportSize({ width: 1440, height: 900 }); await wait(800);
  }
  if (want('fast')) {
    await w.template('plate'); await w.node('study'); await w.set('[data-path="acquisition.duration_us"]', '30'); await w.set('[data-path="acquisition.sample_step_us"]', '0.05');
    await w.node('engine'); await w.set('[data-action="engine"]', 'coarse'); await w.previewed(); await shot('b20-fast-setup'); await compute(); await finished(); await wait(4000);
    await w.tall(); await w.frame(.5); await w.clip('b21-fast-result', '#results', '#result-head', w.section('三维位移场动画')); await page.setViewportSize({ width: 1440, height: 900 }); await wait(800);
  }
  if (want('pause')) {
    await w.template('gaussian'); await w.node('engine'); await w.set('[data-action="engine"]', 'network'); await w.node('study'); await w.set('[data-path="acquisition.duration_us"]', '6000'); await w.set('[data-path="acquisition.sample_step_us"]', '0.1'); await w.previewed(); await shot('b30-long-setup');
    await compute(); await page.waitForSelector('#pause-job', { timeout: 300000 }); await wait(25000); await shot('b31-long-running'); await page.locator('#pause-job').click();
    await page.waitForSelector('#resume-job', { timeout: 300000 }); await wait(1500); await shot('b32-paused'); await page.locator('#resume-job').click(); await wait(12000); await shot('b33-resumed');
    await page.locator('#pause-job').click(); await page.waitForSelector('#resume-job', { timeout: 300000 });   // left paused: the demonstration does not need the rest of the record
  }
  if (want('pipe')) {
    await w.template('pipe'); await w.node('engine'); await w.set('[data-action="engine"]', 'network'); await w.node('study'); await w.set('[data-path="acquisition.duration_us"]', '60'); await w.set('[data-path="acquisition.sample_step_us"]', '0.05'); await w.previewed(); await w.dock('messages'); await shot('b40-pipe-setup');
    await compute(); await finished(); await wait(4000); await w.tall(); await w.frame(.5); await w.clip('b41-pipe-result', '#results', '#result-head', w.section('三维位移场动画'));
  }
  if (want('cfrp')) {
    await w.template('gaussian'); await w.node('engine'); await w.set('[data-action="engine"]', 'network'); await w.set('[data-action="control-set"]', 'full'); await w.set('[data-action="record-mode"]', 'full');
    await w.node('material'); await w.set('[data-action="material-preset"]', 'cfrp_ud'); await w.previewed(); await w.dock('messages'); await shot('b50-cfrp-setup');
    await compute(); await finished(); await wait(4000); await w.tall(); await w.side(); await w.frame(.6); await w.clip('b51-cfrp-result', '#results', '#result-head', w.section('三维位移场动画'));
    for (const [prefix, values] of [['nl-start', [20, 1, 1.6]], ['nl-end', [40, 1, 1.6]], ['nl-direction', [0, 0, 1]]]) for (let i = 0; i < 3; i++) { await page.locator(`#${prefix}-${i}`).fill(String(values[i])); await page.locator(`#${prefix}-${i}`).blur(); }
    await w.nlTab('dispersion'); await page.waitForFunction(() => document.querySelector('#nl-theory table'), null, { timeout: 300000 }); await wait(800);
    await page.locator('#nl-part').selectOption('full'); await page.locator('#nl-fk-run').click(); await wait(400); await page.waitForFunction(() => !document.querySelector('#nl-fk')?.hidden && !/正在/.test(document.querySelector('#nl-fk-message')?.textContent || ''), null, { timeout: 240000 }); await wait(2500);
    await w.theoryShot('b52-cfrp-theory'); await w.fkShot('b53-cfrp-fk'); await page.setViewportSize({ width: 1440, height: 900 }); await wait(800);
  }
  await w.close();
})().catch(e => { console.error('FAILED', e); process.exit(1); });
