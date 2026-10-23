"""Research model interaction in temporary state; never submits a COMSOL job."""
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
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = await browser.new_page(viewport={'width': 1440, 'height': 1050}, accept_downloads=True)
        errors, computes = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: computes.append(request.url) if request.method == 'POST' and request.url.endswith(('/api/jobs', '/api/studies')) else None)
        await page.route('**/api/studies', lambda route: route.abort())
        await page.goto(origin + '/#model')
        await page.locator('#model-template').select_option('research')
        await expect(page.locator('.material-reference-state')).to_contain_text('已关联')
        await expect(page.locator('.material-table')).to_contain_text('-143')
        await expect(page.locator('.material-table')).to_contain_text('± 13')
        await page.locator('[data-model-path="material.rho_kg_m3"]').fill('2705')
        await expect(page.locator('.material-reference-state')).to_contain_text('自定义')
        await page.locator('[data-model-path="geometry.thickness_mm"]').fill('2')
        await page.get_by_role('button', name='应用文献铝参数', exact=True).click()
        await expect(page.locator('[data-model-path="geometry.thickness_mm"]')).to_have_value('2')
        await expect(page.locator('[data-model-path="material.rho_kg_m3"]')).to_have_value('2700')
        await page.locator('[data-model-path="geometry.thickness_mm"]').fill('1')
        async with page.expect_download() as material_download:
            await page.get_by_role('button', name='导出材料卡', exact=True).click()
        material_file = output / 'browser-aluminum-reference.json'
        await (await material_download.value).save_as(material_file)
        material = json.loads(material_file.read_text())
        assert material['record']['third_order']['l_pa'] == -143e9
        assert material['reference']['usage'] == 'linear_constants_only'
        await page.locator('#model-name').fill('二维线性导波参考')
        await page.locator('[data-model-path="geometry.length_mm"]').fill('22')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-status')).to_contain_text('已按当前模型规则校验并保存', timeout=30000)
        await page.reload()
        await expect(page.locator('#model-template')).to_have_value('research')
        await expect(page.locator('[data-model-path="geometry.length_mm"]')).to_have_value('22')
        await expect(page.locator('.material-reference-state')).to_contain_text('已关联')
        await page.locator('.material-details').evaluate('(node) => node.open = true')
        await page.screenshot(path=str(output / 'aluminum-material-desktop.png'), full_page=True)
        saved_material = (await (await page.request.get(origin + '/api/model')).json())['model']['case']['material_reference']
        assert saved_material['record_sha256'] == material['record_sha256']
        await page.get_by_role('link', name='物理场', exact=True).click()
        await expect(page.get_by_role('heading', name='顶面接收点', exact=True)).to_be_visible()
        await page.locator('[data-model-path="source.direction"]').select_option('sh')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-status')).to_contain_text('已按当前模型规则校验并保存', timeout=30000)
        await page.get_by_role('link', name='研究', exact=True).click()
        await expect(page.get_by_role('button', name='开始计算', exact=True)).to_be_enabled(timeout=60000)
        await page.get_by_role('button', name='开始计算', exact=True).click()
        await expect(page.locator('#study-description')).to_contain_text('二维 22 × 1 mm')
        await expect(page.locator('#study-description')).not_to_contain_text('undefined')
        await page.get_by_role('dialog').get_by_role('button', name='取消', exact=True).click()
        await page.locator('[data-model-path="acquisition.duration_us"]').fill('3')
        await page.get_by_role('button', name='校验并保存', exact=True).click()
        await expect(page.locator('.model-error')).to_contain_text('未覆盖完整激励脉冲', timeout=30000)
        saved = (await (await page.request.get(origin + '/api/model')).json())['model']
        assert saved['case']['acquisition']['duration_us'] == 12
        assert saved['case']['source']['direction'] == 'sh'
        await page.reload()
        await expect(page.locator('[data-model-path="acquisition.duration_us"]')).to_have_value('12')
        await expect(page.get_by_role('button', name='开始计算', exact=True)).to_be_enabled(timeout=60000)
        await page.screenshot(path=str(output / 'research-study-desktop.png'), full_page=True)
        # A live interface with a different model hash must not enable compute.
        health = await (await page.request.get(origin + '/api/health')).json()
        health['services']['research']['data']['source_sha256'] = '0' * 64
        await page.route('**/api/health', lambda route: route.fulfill(status=200, content_type='application/json', body=json.dumps(health)))
        await page.get_by_role('button', name='刷新状态').click()
        await expect(page.get_by_role('button', name='开始计算', exact=True)).to_be_disabled()
        await expect(page.locator('.research-notice')).to_contain_text('暂时不能提交计算')
        await page.get_by_role('link', name='建模', exact=True).click()
        async with page.expect_download() as download_event:
            await page.get_by_role('button', name='导出参数', exact=True).click()
        exported = output / 'browser-research-scene.json'
        await (await download_event.value).save_as(exported)
        assert json.loads(exported.read_text()) == saved['case']
        await page.set_viewport_size({'width': 390, 'height': 844})
        for view in ['建模', '物理场', '研究']:
            await page.get_by_role('link', name=view, exact=True).click()
            await page.get_by_role('heading', name=view, exact=True).wait_for()
            assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth'), view
            if view == '建模':
                await page.locator('.material-details').evaluate('(node) => node.open = true')
                assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
                await page.screenshot(path=str(output / 'aluminum-material-mobile.png'), full_page=True)
        await page.screenshot(path=str(output / 'research-study-mobile.png'), full_page=True)
        assert not errors, errors
        assert not computes, computes
        report = {'passed': True, 'material_reference_save_reload_export': True, 'custom_edit_unlinks_reference': True, 'preset_preserves_geometry': True, 'save_reload_export': True, 'direction_mapping_edit': True, 'invalid_pulse_window_preserves_saved_model': True, 'confirmation_cancelled': True, 'changed_source_blocks_compute': True, 'mobile_overflow': False, 'javascript_errors': errors, 'compute_post_requests': computes}
        (output / 'research-browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()

output = Path(os.environ.get('WORKAI_TEST_OUTPUT', '/workspace/workai-remote/validation/20261022-research-source'))
output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='workai-research-browser-') as temporary:
    with socket.socket() as socket_:
        socket_.bind(('127.0.0.1', 0)); port = socket_.getsockname()[1]
    environment = dict(os.environ, PORT=str(port), WORKAI_STATE_DIR=str(Path(temporary) / 'jobs'))
    with open(Path(temporary) / 'server.log', 'w+') as log:
        process = subprocess.Popen(['node', 'server/index.mjs'], cwd=Path(__file__).resolve().parents[1], env=environment, stdout=log, stderr=log)
        try:
            for _ in range(100):
                with socket.socket() as socket_:
                    if socket_.connect_ex(('127.0.0.1', port)) == 0: break
                if process.poll() is not None: raise RuntimeError('Test server failed to start')
                time.sleep(.05)
            else: raise RuntimeError('Test server did not become ready')
            asyncio.run(check(f'http://127.0.0.1:{port}', output))
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
