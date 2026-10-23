"""Exercise model editing with real original validation, in isolated local state.

Starts and stops its own workbench process. Never submits COMSOL jobs.
"""
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

from playwright.async_api import async_playwright, expect


async def check(origin, output):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = await browser.new_page(viewport={'width': 1440, 'height': 1050}, accept_downloads=True)
        errors, compute_requests = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: compute_requests.append(request.url) if request.method == 'POST' and request.url.endswith(('/api/jobs', '/api/studies')) else None)
        await page.route('**/api/studies', lambda route: route.abort())
        await page.goto(origin + '/#model')
        await page.get_by_role('heading', name='建模', exact=True).wait_for()
        await page.locator('[data-model-path="geometry.length_mm"]').fill('45')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-status')).to_contain_text('已按当前模型规则校验并保存', timeout=30000)
        await page.reload()
        await expect(page.locator('[data-model-path="geometry.length_mm"]')).to_have_value('45')

        await page.get_by_role('link', name='物理场', exact=True).click()
        await page.locator('[data-model-path="sources.0.position_mm.2"]').fill('20')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-status')).to_contain_text('未通过服务器原有规则', timeout=30000)
        saved = await (await page.request.get(origin + '/api/model')).json()
        assert saved['model']['case']['sources'][0]['position_mm'][2] == 2
        assert saved['model']['case']['geometry']['length_mm'] == 45

        await page.get_by_role('link', name='建模', exact=True).click()
        await page.locator('#model-template').select_option('volume')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-status')).to_contain_text('已按当前模型规则校验并保存', timeout=30000)
        async with page.expect_download() as download_event:
            await page.get_by_role('button', name='导出参数', exact=True).click()
        download = await download_event.value
        export = output / 'browser-volume-case.json'
        await download.save_as(export)
        case = json.loads(export.read_text())
        assert case['schema'] == 'workai.physics_ai.volume.case.v1'
        assert case['geometry'] == {'length_mm': 200, 'width_mm': 100, 'thickness_mm': 2}
        await page.get_by_role('link', name='研究', exact=True).click()
        await expect(page.get_by_role('button', name='开始计算', exact=True)).to_be_disabled()
        await expect(page.locator('.study-notice')).to_contain_text('大体积模型尚未就绪')
        await expect(page.locator('#toast')).to_be_hidden(timeout=15000)
        await page.screenshot(path=str(output / 'study-volume-desktop.png'), full_page=True)

        await page.get_by_role('link', name='建模', exact=True).click()
        await page.set_viewport_size({'width': 390, 'height': 844})
        assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
        await page.screenshot(path=str(output / 'model-mobile.png'), full_page=True)
        assert not errors, errors
        assert not compute_requests, compute_requests
        report = {'passed': True, 'isolated_state': True, 'edit_save_reload': True, 'original_validator_rejected_invalid_source': True, 'invalid_save_preserved_previous_model': True, 'volume_case_export_verified': True, 'unready_volume_compute_disabled': True, 'javascript_errors': errors, 'compute_post_requests': compute_requests}
        (output / 'model-browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()


output = Path(os.environ.get('WORKAI_TEST_OUTPUT', '/workspace/workai-remote/validation/20261022-website'))
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='workai-browser-model-') as directory:
    with socket.socket() as socket_:
        socket_.bind(('127.0.0.1', 0))
        port = socket_.getsockname()[1]
    environment = dict(os.environ, PORT=str(port), WORKAI_STATE_DIR=str(Path(directory) / 'jobs'))
    with open(Path(directory) / 'server.log', 'w+') as log:
        process = subprocess.Popen(['node', 'server/index.mjs'], cwd=Path(__file__).resolve().parents[1], env=environment, stdout=log, stderr=log)
        try:
            import time
            for _ in range(100):
                with socket.socket() as socket_:
                    if socket_.connect_ex(('127.0.0.1', port)) == 0:
                        break
                if process.poll() is not None:
                    raise RuntimeError('Test server failed to start')
                time.sleep(.05)
            else:
                raise RuntimeError('Test server did not become ready')
            asyncio.run(check(f'http://127.0.0.1:{port}', output))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
