// Print the demonstration document to PDF: node docs/demo/pdf.cjs <demo.html> <out.pdf>
const { chromium } = require('playwright'); const path = require('path');
(async () => {
  const browser = await chromium.launch(); const page = await browser.newPage(); await page.goto('file://' + path.resolve(process.argv[2])); await page.waitForTimeout(1500);
  await page.pdf({ path: process.argv[3], format: 'A4', printBackground: true, displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-size:8px;color:#667;text-align:center;font-family:sans-serif">WorkAI 1.5.0 功能演示 · <span class="pageNumber"></span> / <span class="totalPages"></span></div>', margin: { top: '16mm', bottom: '18mm', left: '16mm', right: '16mm' } });
  await browser.close();
})();
