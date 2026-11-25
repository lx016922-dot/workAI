// The exported HTML report (stage_c saves it as <shots>/files/report.html) as a picture of its beginning: node docs/demo/report_shot.cjs <shots dir>
const { chromium } = require('playwright'); const path = require('path');
(async () => {
  const shots = path.resolve(process.argv[2]); const browser = await chromium.launch(); const context = await browser.newContext({ viewport: { width: 1100, height: 1350 }, deviceScaleFactor: 2 });
  const page = await context.newPage(); await page.goto('file://' + path.join(shots, 'files', 'report.html')); await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(shots, 'e02-report.jpg'), type: 'jpeg', quality: 90 }); await browser.close();
})();
