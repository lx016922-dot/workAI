// Stage A: building a model — every panel of the model builder, without computing.
const { open } = require('./lib.cjs');
(async () => {
  const w = await open(process.argv[2]); const { page, shot, wait } = w;
  await w.template('gaussian'); await w.node('geometry'); await shot('a01-home');
  await w.ribbon('home'); await page.locator('.rb-panel [data-cmd="new"]').click(); await shot('a02-templates'); await page.keyboard.press('Escape');
  for (const kind of ['pipe', 'hole', 'notch']) { await w.template(kind); await w.node('geometry'); await w.ribbon('geometry'); await shot('a03-kind-' + kind); }
  // defects: one click, then by text
  await w.template('gaussian'); await w.ribbon('geometry'); await w.cmd('add-defect:crack'); await w.previewed(); await shot('a04-crack');
  await page.locator('#command-input').fill('在 30 mm 处加一个直径 1 mm 的通孔；在 36 到 42 mm 之间添加分层，距上表面 0.8 mm'); await page.locator('#command-run').click();
  await page.waitForSelector('.nl-ops', { timeout: 30000 }); await shot('a05-text-dialog'); await page.getByRole('button', { name: '应用修改' }).click(); await w.previewed(); await w.node('defects'); await shot('a06-defects');
  await w.cmd('add-defect:contact'); await w.previewed(); await shot('a07-contact');
  // material
  await w.template('gaussian'); await w.node('material'); await shot('a08-material');
  await w.set('[data-action="material-preset"]', 'cfrp_ud'); await w.previewed(); await shot('a09-material-cfrp');
  await w.template('gaussian'); await w.node('material'); await page.locator('#inspector details').filter({ hasText: '由 f 和 2f 处实测的衰减换算' }).locator('summary').click();
  await w.set('#damping-f1', '0.4'); await w.set('#damping-a1', '1.0'); await w.set('#damping-c1', '5300'); await w.set('#damping-a2', '2.5'); await w.set('#damping-c2', '5000');
  await page.locator('#damping-c2').scrollIntoViewIfNeeded(); await shot('a10-damping-convert'); await page.locator('[data-action="damping-from-attenuation"]').click(); await w.previewed(); await page.locator('#inspector .wb-section').filter({ hasText: '阻尼在 f 和 2f 处的效果' }).last().scrollIntoViewIfNeeded(); await shot('a10-damping');
  // sources: signal and profiles
  await w.template('gaussian'); await w.node('source:0'); await shot('a11-source');
  await w.tab('signal'); await shot('a12-signal'); await w.set('[data-path="source.signal.type"]', 'chirp'); await w.previewed(); await shot('a13-signal-chirp');
  await w.set('[data-path="source.signal.type"]', 'tone_burst'); await w.tab('model');
  await w.set('[data-path="source.profile.type"]', 'wedge'); await w.previewed(); await page.locator('[data-path="source.profile.type"]').scrollIntoViewIfNeeded(); await shot('a14-wedge');
  await w.set('[data-path="source.profile.type"]', 'patch'); await w.previewed(); await page.locator('[data-path="source.profile.type"]').scrollIntoViewIfNeeded(); await shot('a15-patch');
  await w.set('[data-path="source.profile.type"]', 'comb'); await w.previewed(); await page.locator('[data-path="source.profile.type"]').scrollIntoViewIfNeeded(); await shot('a16-comb');
  await w.template('twopatch'); await w.node('sources'); await shot('a17-two-sources');
  // boundaries and receivers
  await w.template('gaussian'); await w.node('boundary:1'); await shot('a18-boundary');
  await w.ribbon('physics'); await w.cmd('add-boundary:fixed'); await w.previewed(); await shot('a19-boundary-added');
  await w.template('gaussian'); await w.node('receiver:0'); await shot('a20-receiver');
  // mesh, study, solver
  await w.node('mesh'); await w.ribbon('mesh'); await w.cmd('toggle:grid'); await shot('a21-mesh');
  await w.node('study'); await shot('a22-study'); await w.node('engine'); await shot('a23-engine');
  await w.set('[data-action="engine"]', 'coarse'); await w.previewed(); await shot('a24-engine-fast'); await w.set('[data-action="engine"]', 'network');
  // a model the software refuses, and why
  await w.template('gaussian'); await w.node('source:0'); await w.set('[data-path="source.frequency_mhz"]', '30'); await w.previewed(); await w.dock('messages'); await shot('a25-refused');
  await w.ribbon('help'); await w.cmd('help'); await shot('a26-help'); await page.keyboard.press('Escape');
  await w.close();
})().catch(e => { console.error('FAILED', e); process.exit(1); });
