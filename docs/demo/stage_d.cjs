// Stage D: transducers, wave mixing, cracks and the three-dimensional plate — results of the other demonstration models.
const { open } = require('./lib.cjs');
(async () => {
  const out = process.argv[2]; const only = process.argv.slice(3); const w = await open(out); const { page, shot, wait } = w; await w.tall();
  const want = id => !only.length || only.includes(id);
  const line = async (from, to, direction) => {
    for (const [prefix, values] of [['nl-start', from], ['nl-end', to], ['nl-direction', direction]]) for (let i = 0; i < 3; i++) { await page.locator(`#${prefix}-${i}`).fill(String(values[i])); await page.locator(`#${prefix}-${i}`).blur(); }
    await wait(400);
  };
  const fk = async part => {
    await w.nlTab('dispersion'); await page.waitForFunction(() => document.querySelector('#nl-theory table') || /圆管|没有理论/.test(document.querySelector('#nl-theory-message')?.textContent || ''), null, { timeout: 180000 }); await wait(600);
    await page.locator('#nl-part').selectOption(part); await page.locator('#nl-fk-run').click(); await wait(400); await page.waitForFunction(() => !document.querySelector('#nl-fk')?.hidden && !/正在/.test(document.querySelector('#nl-fk-message')?.textContent || ''), null, { timeout: 240000 }); await wait(2500);
  };
  if (want('wedge')) {
    await w.openJob('2 mm 板 · 斜楔换能器激发 A0'); await w.side(5, 12000); await w.frame(.3); await shot('d01-wedge-field', w.section('三维位移场动画'));
    await line([140, .5, 2], [200, .5, 2], [0, 0, 1]); await fk('linear'); await w.theoryShot('d02-wedge-theory'); await w.fkShot('d02-wedge-fk');
    await line([100, .5, 2], [40, .5, 2], [0, 0, 1]); await fk('linear'); await w.fkShot('d03-wedge-fk-back');
  }
  if (want('patch')) {
    await w.openJob('2 mm 板 · 条形压电片'); await line([140, .5, 2], [200, .5, 2], [0, 0, 1]); await fk('linear'); await w.fkShot('d04-patch-fk');
  }
  if (want('mixing')) {
    await w.openJob('细杆 · 0.4 与 0.5 MHz 混频'); await w.nlTab('mixing'); await wait(800); await page.locator('#nl-mode').selectOption('receivers').catch(() => {});
    await page.locator('#nl-mix-run').click(); await page.waitForFunction(() => document.querySelector('#nl-mixing svg, #nl-mixing table'), null, { timeout: 240000 }); await wait(1000); await w.clip('d05-mixing', '#nl-panel', '#nl-panel h2', page.locator('#nl-mixing svg').last());
  }
  if (want('crack')) {
    await w.openJob('2 mm 板 · S0 遇到表面裂纹'); await w.side(5, 12000); await w.frame(.42); await shot('d06-crack-field', w.section('三维位移场动画'));
    await line([20, .5, 2], [92, .5, 2], [1, 0, 0]); await fk('linear'); await w.fkShot('d07-crack-fk');
    const increment = async (name, id) => {   // the complete response behind the crack: waveform and spectrum (the contact acts in every control, so its harmonics are here and not in complete − linear)
      await w.openJob(name);
      await page.locator('#solid-receiver').selectOption('3'); await wait(1500); await page.waitForFunction(() => /已提取/.test(document.querySelector('#solid-probe-message')?.textContent || '') && document.querySelector('#solid-waveform')?.children.length, null, { timeout: 120000 }).catch(e => console.log('probe wait failed')); await wait(2500); console.log(await page.locator('#solid-probe-message').innerText());
      await w.clip(id, '#results', w.section('方向位移波形'), w.section('单边幅值频谱'));
    };
    await increment('2 mm 板 · S0 遇到闭合裂纹', 'd08-closed-crack'); await increment('2 mm 板 · S0 遇到表面裂纹', 'd08-open-crack');
  }
  if (want('disc')) {
    await w.openJob('三维板 · 压电圆片（四分之一模型）'); await w.frame(.3); await shot('d09-disc-field', w.section('三维位移场动画')); await w.view(-55, 52, 1.25); await page.locator('#solid-component').selectOption('2'); await w.frame(.4); await shot('d09-disc-top', w.section('三维位移场动画'));
    await shot('d10-disc-wave', w.section('方向位移波形'));
  }
  await w.close();
})().catch(e => { console.error('FAILED', e); process.exit(1); });
