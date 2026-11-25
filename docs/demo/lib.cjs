// Drive the workbench in a browser and take the screenshots of the demonstration (docs/demo).
// The workbench must be running: python -m offline.app --data <dir> --port 8765 --no-browser
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');

exports.open = async function open(out, address = 'http://127.0.0.1:8765/') {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, acceptDownloads: true, locale: 'zh-CN' });
  const page = await context.newPage(); const errors = [];
  page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.addInitScript(() => { try { localStorage.clear(); } catch { /* no storage */ } });
  await page.goto(address); await page.waitForSelector('#tree [data-node="geometry"]'); await page.waitForTimeout(2500);
  const wait = ms => page.waitForTimeout(ms);
  const api = {
    browser, context, page, errors, wait,
    /** A screenshot of the window, or of one element (`sel`), as JPEG. */
    async shot(id, sel) {
      const file = path.join(out, id + '.jpg'); await wait(350);
      if (sel) await (typeof sel === 'string' ? page.locator(sel).first() : sel).screenshot({ path: file, type: 'jpeg', quality: 90 });
      else await page.screenshot({ path: file, type: 'jpeg', quality: 90 });
      console.log('shot', id); return file;
    },
    ribbon: async id => { await page.locator(`[data-ribbon-tab="${id}"]`).click(); await wait(200); },
    cmd: async id => { await page.locator(`.rb-panel [data-cmd="${id}"]`).first().click(); await wait(500); },
    node: async id => { await page.locator(`#tree [data-node="${id}"]`).first().click(); await wait(500); },
    tab: async id => { await page.locator(`[data-tab="${id}"]`).click(); await wait(400); },
    dock: async id => { await page.locator(`[data-dock="${id}"]`).click(); await wait(300); },
    template: async id => { await api.ribbon('home'); await page.locator('.rb-panel [data-cmd="new"]').click(); await wait(200); await page.locator(`.rb-pop [data-pick="${id}"]`).click(); await api.previewed(); },
    /** Wait until the pre-flight check of the present model has come back. */
    previewed: async () => { await wait(1200); await page.waitForFunction(() => !/正在检查|检查中/.test(document.querySelector('#dock-messages')?.textContent || ''), null, { timeout: 120000 }); await wait(800); },
    /** 文件 → 打开 with a model file. */
    openModel: async file => { await page.locator('#import-scene').setInputFiles(file); await api.previewed(); },
    set: async (selector, value) => { const field = page.locator(selector).first(); if (await field.evaluate(n => n.tagName === 'SELECT')) await field.selectOption(value); else { await field.fill(String(value)); await field.blur(); } await wait(500); },
    /** Open a finished record by the name of its model. */
    openJob: async name => {
      const jobs = await (await page.request.get(address + 'api/jobs')).json(); const job = jobs.find(j => j.name === name && j.phase === 'completed'); if (!job) throw Error('no finished record named ' + name);
      await page.evaluate(() => document.querySelector('[data-cmd="refresh"]')?.click()); await api.ribbon('results'); await api.cmd('refresh'); await wait(800);
      await page.locator(`#tree [data-node="job:${job.job_id}"]`).first().click(); await page.waitForSelector('#nl-panel', { timeout: 120000 }); await wait(2500); return job;
    },
    /** A tall window, so that the result page shows all of its sections at once. */
    tall: async (height = 7000, width = 1440) => { await page.setViewportSize({ width, height }); await wait(1200); },
    /** The section of the result page with this heading. */
    section: title => page.locator('#results section').filter({ has: page.locator('h2', { hasText: title }) }).first(),
    /** Move the animation of the displacement field to a fraction of the record. */
    frame: async fraction => { await page.locator('#solid-frame').evaluate((n, f) => { n.value = Math.round(f * Number(n.max)); n.dispatchEvent(new Event('input', { bubbles: true })); }, fraction); await wait(900); },
    /** Turn the displacement field by dragging (0.01 rad a pixel) and zoom it with the wheel. */
    view: async (dx, dy, zoom = 1) => {
      const canvas = page.locator('#solid-canvas'); await canvas.scrollIntoViewIfNeeded(); const b = await canvas.boundingBox(), x = b.x + b.width / 2, y = b.y + b.height / 2;
      await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.move(x + dx, y + dy, { steps: 8 }); await page.mouse.up();
      if (zoom !== 1) await canvas.evaluate((n, d) => n.dispatchEvent(new WheelEvent('wheel', { deltaY: d, cancelable: true })), -1000 * Math.log(zoom)); await wait(600);
    },
    /** The side view (侧视 XZ) of the displacement field; a strip is shown this way from the start. `zoom` by the wheel, `scale` the deformation factor. */
    side: async (zoom = 1, scale = 0) => {
      await page.locator('[data-solid-view="side"]').click(); await wait(300);
      if (zoom !== 1) await page.locator('#solid-canvas').evaluate((n, d) => n.dispatchEvent(new WheelEvent('wheel', { deltaY: d, cancelable: true })), -1000 * Math.log(zoom));
      if (scale) { await page.locator('#solid-scale').fill(String(scale)); await page.locator('#solid-scale').dispatchEvent('change'); }
      await wait(500);
    },
    top: async () => { await page.locator('[data-solid-view="top"]').click(); await wait(400); },
    /** The parts of the analysis panel as separate pictures. */
    fkShot: id => api.clip(id, '#nl-panel', page.locator('#nl-panel h3', { hasText: 'f–k 谱' }), '#nl-fk-notes'),
    theoryShot: id => api.clip(id, '#nl-panel', '#nl-theory-message', '#nl-theory'),
    summaryShot: id => api.clip(id, '#nl-panel', '#nl-panel h2', '#nl-accumulation table.nl-summary'),
    chartsShot: id => api.clip(id, '#nl-panel', page.locator('#nl-accumulation h3.scan-h').first(), page.locator('#nl-accumulation svg').last()),
    nlTab: async id => { await page.locator(`[data-nl-tab="${id}"]`).click(); await wait(700); },
    /** Save what a click downloads; returns the path. */
    download: async (locator, dir) => { const [d] = await Promise.all([page.waitForEvent('download'), locator.click()]); const file = path.join(dir, d.suggestedFilename()); await d.saveAs(file); return file; },
    /** A screenshot of part of the page: across `within`, from the top of `from` to the bottom of `to`. */
    async clip(id, within, from, to = from, pad = 6) {
      const box = async l => (typeof l === 'string' ? page.locator(l).first() : l).boundingBox(); const [frame, top, bottom] = [await box(within), await box(from), await box(to)];
      const y = Math.max(0, top.y - pad), file = path.join(out, id + '.jpg'); await wait(350);
      await page.screenshot({ path: file, type: 'jpeg', quality: 90, clip: { x: frame.x, y, width: frame.width, height: bottom.y + bottom.height + pad - y } }); console.log('shot', id); return file;
    },
    openJobId: async id => { await api.ribbon('results'); await api.cmd('refresh'); await wait(800); await page.locator(`#tree [data-node="job:${id}"]`).first().click(); await page.waitForSelector('#nl-panel', { timeout: 120000 }); await wait(2500); },
    close: async () => { if (errors.length) console.log('PAGE ERRORS', JSON.stringify(errors.slice(0, 6))); await browser.close(); },
  };
  return api;
};
