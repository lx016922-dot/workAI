"""Read-only browser acceptance against the running workbench and real COMSOL.

Uses Playwright and Chromium already supplied by the cloud execution image.
Never clicks the job submission button or sends a remote solve request.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil

from playwright.async_api import async_playwright, expect


async def main():
    origin = os.environ.get('WORKAI_TEST_ORIGIN', 'http://127.0.0.1:3000')
    output = Path(os.environ.get('WORKAI_TEST_OUTPUT', '/workspace/workai-remote/validation/20261022-website'))
    output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox', '--disable-dev-shm-usage'])
        context = await browser.new_context(viewport={'width': 1440, 'height': 1050}, device_scale_factor=1, accept_downloads=True)
        page = await context.new_page()
        errors, writes = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: writes.append(request.url) if request.method == 'POST' else None)
        await page.goto(origin + '/#results', wait_until='domcontentloaded')
        await page.locator('#job-select').select_option('comsol-9f446b8016a041f2bff1d47982eec477')
        await page.get_by_role('link', name='研究总览', exact=True).click()
        await expect(page.locator('#sidebar-status')).to_contain_text('已连接', timeout=60000)
        await page.locator('.chart-wrap svg').wait_for(timeout=60000)
        assert await page.get_by_role('heading', name='超声仿真工作台').count() == 1
        assert await page.get_by_text('已完成', exact=True).count() >= 1
        await page.screenshot(path=str(output / 'overview-desktop.png'), full_page=True)

        await page.get_by_role('link', name='建模', exact=True).click()
        await page.get_by_role('heading', name='建模', exact=True).wait_for()
        assert await page.get_by_label('模型名称', exact=True).input_value()
        await page.screenshot(path=str(output / 'model-desktop.png'), full_page=True)
        await page.get_by_role('link', name='物理场', exact=True).click()
        await page.get_by_role('heading', name='物理场', exact=True).wait_for()
        assert await page.get_by_role('heading', name='接收点', exact=True).count() == 1
        await page.get_by_role('link', name='研究', exact=True).click()
        await page.get_by_role('heading', name='研究', exact=True).wait_for()
        assert await page.locator('tbody tr').count() >= 1
        await page.get_by_role('button', name='连接验证', exact=True).click()
        await page.get_by_role('dialog', name='新建连接验证任务').wait_for()
        await page.get_by_role('button', name='取消', exact=True).click()

        await page.get_by_role('link', name='结果', exact=True).click()
        await page.locator('#job-select').select_option('comsol-9f446b8016a041f2bff1d47982eec477')
        await page.get_by_role('heading', name='结果文件', exact=True).wait_for()
        await page.locator('tr').filter(has_text='worker.log').get_by_role('button', name='预览', exact=True).click()
        await expect(page.locator('#preview-content')).to_contain_text('WORKAI_SOLVE', timeout=60000)
        assert '校验通过' in await page.locator('#preview-meta').inner_text()
        await page.get_by_role('dialog').get_by_role('button', name='关闭', exact=True).click()

        async with page.expect_download(timeout=60000) as event:
            await page.locator('tr').filter(has_text='coarse.csv').get_by_role('button', name='下载 ↓').click()
        download = await event.value
        target = output / 'browser-coarse.csv'
        await download.save_as(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == '5640858a457d463c46d3b65cd8c804a1b64ad7235f86c936e9cc4883935af9d7'
        await page.locator('.chart-wrap svg').wait_for(timeout=60000)
        await expect(page.locator('#toast')).to_be_hidden(timeout=15000)
        await page.screenshot(path=str(output / 'results-desktop.png'), full_page=True)

        await page.get_by_role('link', name='服务连接', exact=True).click()
        await page.get_by_role('heading', name='服务连接', exact=True).wait_for()
        assert await page.get_by_text('http://100.100.205.105:18765', exact=True).count() == 1

        await page.set_viewport_size({'width': 390, 'height': 844})
        await page.get_by_role('link', name='研究总览', exact=True).click()
        await page.get_by_role('heading', name='超声仿真工作台').wait_for()
        assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth'), 'Mobile page overflows the viewport'
        await page.locator('.chart-wrap svg').wait_for(timeout=60000)
        await page.screenshot(path=str(output / 'overview-mobile.png'), full_page=True)

        await page.route('**/api/health', lambda route: route.fulfill(status=200, content_type='application/json', body=json.dumps({'checked_at': '2026-10-22T04:00:00Z', 'services': {name: {'ok': False, 'verified': False, 'error': '测试：服务暂时不可达'} for name in ['base', 'plate', 'volume']}})))
        await page.get_by_role('button', name='刷新状态').click()
        await expect(page.locator('#sidebar-status')).to_contain_text('待连接')
        assert await page.get_by_role('alert').count() == 1
        assert not errors, errors
        assert not writes, writes
        report = {'passed': True, 'desktop': '1440x1050', 'mobile': '390x844', 'real_service_health': True, 'real_job_status': True, 'real_worker_log_preview': True, 'real_csv_download_sha256_verified': True, 'mobile_overflow': False, 'offline_state_simulated': True, 'javascript_errors': errors, 'post_requests': writes}
        (output / 'browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()


asyncio.run(main())
