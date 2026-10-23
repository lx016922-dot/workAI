"""Nonlinear forms in isolated state; no compute requests are permitted."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
from playwright.async_api import async_playwright, expect

async def check(origin, output):
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = await browser.new_page(viewport={'width': 1440, 'height': 1050}, accept_downloads=True)
        errors, computes = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda r: computes.append(r.url) if r.method == 'POST' and r.url.endswith(('/api/jobs', '/api/studies')) else None)
        await page.route('**/api/studies', lambda route: route.abort())
        await page.goto(origin + '/#model')
        await page.locator('#model-template').select_option('nonlinear')
        await expect(page.get_by_role('heading', name='本构与对照', exact=True)).to_be_visible()
        await expect(page.locator('.material-table')).to_contain_text('-143')
        await page.locator('#model-name').fill('多晶铝材料非线性 · 浏览器验证')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-status')).to_contain_text('已按当前模型规则校验并保存', timeout=30000)
        await page.reload()
        await expect(page.locator('#model-template')).to_have_value('nonlinear')
        saved = (await (await page.request.get(origin + '/api/model')).json())['model']
        assert saved['case']['material_reference']['usage'] == 'murnaghan_total_lagrangian'
        assert saved['case']['nonlinear']['n_pa'] == -345e9
        async with page.expect_download() as event:
            await page.get_by_role('button', name='导出参数', exact=True).click()
        export = output / 'nonlinear-browser-scene.json'
        await (await event.value).save_as(export)
        assert json.loads(export.read_text()) == saved['case']
        await page.get_by_role('link', name='物理场', exact=True).click()
        await expect(page.get_by_role('heading', name='端面激励', exact=True)).to_be_visible()
        await page.locator('[data-model-path="source.amplitude_pa"]').fill('10000000')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-error')).to_contain_text('0.1–5 MPa', timeout=30000)
        await page.reload()
        await expect(page.locator('[data-model-path="source.amplitude_pa"]')).to_have_value('5000000')
        await page.get_by_role('link', name='研究', exact=True).click()
        await expect(page.get_by_role('button', name='开始计算', exact=True)).to_be_enabled(timeout=60000)
        await page.get_by_role('button', name='开始计算', exact=True).click()
        await expect(page.locator('#study-description')).to_contain_text('15 次真实 COMSOL 求解')
        await page.get_by_role('dialog').get_by_role('button', name='取消', exact=True).click()
        await page.screenshot(path=str(output / 'nonlinear-study-desktop.png'), full_page=True)
        await page.set_viewport_size({'width': 390, 'height': 844})
        for view in ['建模', '物理场', '研究']:
            await page.get_by_role('link', name=view, exact=True).click()
            await page.get_by_role('heading', name=view, exact=True).wait_for()
            assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth'), view
        await page.screenshot(path=str(output / 'nonlinear-study-mobile.png'), full_page=True)
        assert not errors, errors
        assert not computes, computes
        report = {'passed': True, 'save_reload_export': True, 'material_identity_bound': True, 'invalid_amplitude_preserves_saved_model': True, 'fifteen_solve_dialog_cancelled': True, 'mobile_overflow': False, 'javascript_errors': errors, 'compute_post_requests': computes}
        (output / 'nonlinear-browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()

output = Path(os.environ.get('WORKAI_TEST_OUTPUT', '/workspace/workai-remote/validation/20261022-nonlinear-comsol'))
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='workai-nonlinear-browser-') as temporary:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    environment = dict(os.environ, PORT=str(port), WORKAI_STATE_DIR=str(Path(temporary) / 'jobs'))
    with open(Path(temporary) / 'server.log', 'w+') as log:
        process = subprocess.Popen(['node', 'server/index.mjs'], cwd=Path(__file__).resolve().parents[1], env=environment, stdout=log, stderr=log)
        try:
            for _ in range(100):
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', port)) == 0: break
                if process.poll() is not None: raise RuntimeError('Test server failed to start')
                time.sleep(.05)
            else: raise RuntimeError('Test server did not become ready')
            asyncio.run(check(f'http://127.0.0.1:{port}', output))
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
