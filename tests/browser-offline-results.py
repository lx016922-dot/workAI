"""Read-only export from the running workbench, then open the HTML with networking off."""
import asyncio
import json
import os
from pathlib import Path
import shutil
from playwright.async_api import async_playwright, expect

async def check():
    job_id = 'comsol-d62800aa89a8478e85d6478acd71d783'
    origin = os.environ.get('WORKAI_TEST_ORIGIN', 'http://127.0.0.1:3000')
    output = Path('/workspace/workai-remote/validation/20261022-offline-results')
    output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox', '--disable-dev-shm-usage'])
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
        page = await context.new_page()
        errors, posts = [], []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: posts.append(r.url) if r.method == 'POST' else None)
        await page.goto(origin + '/#results')
        await page.locator('#job-select').select_option(job_id)
        button = page.get_by_role('button', name='导出离线结果 HTML', exact=True)
        await expect(button).to_be_visible(timeout=60000)
        expected = await (await context.request.get(origin + '/api/jobs/' + job_id + '/waveforms')).json()
        assert expected['verified'] and expected['harmonics']['all_checks_within_5_percent']
        network_during_export = []
        page.on('request', lambda r: network_during_export.append(r.url))
        async with page.expect_download() as event:
            await button.click()
        html = output / 'workai-nonlinear-results.html'
        await (await event.value).save_as(html)
        assert not network_during_export, network_during_export
        assert not posts and not errors, (posts, errors)
        await context.close()

        offline = await browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True, offline=True)
        page = await offline.new_page()
        network, offline_errors = [], []
        page.on('request', lambda r: network.append(r.url) if r.url.startswith(('http:', 'https:')) else None)
        page.on('pageerror', lambda e: offline_errors.append(str(e)))
        # The managed browser disallows file URLs. Keep that policy intact;
        # exercise the exact exported bytes in an offline in-memory document.
        await page.set_content(html.read_text(), wait_until='load')
        await expect(page.get_by_role('heading', name='导波研究结果', exact=True)).to_be_visible()
        await expect(page.locator('.harmonic-panel .pill')).to_have_text('本算例检查通过')
        await expect(page.locator('#trace-select option')).to_have_count(15)
        await expect(page.locator('.harmonic-panel')).to_contain_text('234/640')
        await expect(page.locator('.harmonic-panel')).to_contain_text('comsol-eec8757166b547e38c8cf07ed1754498')
        before = await page.locator('.waveform-panel svg').inner_html()
        await page.locator('#trace-select').select_option('2:2')
        after = await page.locator('.waveform-panel svg').inner_html()
        assert before != after
        await page.locator('#component-select').select_option('w_nm')
        await expect(page.locator('.waveform-metrics')).to_contain_text('参考分量为零，未定义')
        await page.locator('#component-select').select_option('u_nm')
        async with page.expect_download() as event:
            await page.locator('.offline-actions').get_by_role('button', name='下载波形与分析 JSON', exact=True).click()
        exported = output / 'offline-export.json'
        await (await event.value).save_as(exported)
        assert json.loads(exported.read_text()) == expected
        await page.screenshot(path=str(output / 'offline-desktop.png'), full_page=True)
        await page.pdf(path=str(output / 'workai-nonlinear-results.pdf'), format='A4', print_background=True, margin={'top': '12mm', 'bottom': '12mm', 'left': '12mm', 'right': '12mm'})
        await page.set_viewport_size({'width': 390, 'height': 844})
        assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
        await page.locator('#trace-select').select_option('4:1')
        await expect(page.locator('#trace-select')).to_have_value('4:1')
        await page.screenshot(path=str(output / 'offline-mobile.png'), full_page=True)
        assert not network and not offline_errors, (network, offline_errors)
        report = {'passed': True, 'job_id': job_id, 'live_export_without_new_requests': True,
                  'offline_controls_work': True, 'fifteen_controls': True, 'analysis_export_exact_match': True,
                  'spatial_reference_preserved': True, 'zero_component_undefined': True,
                  'mobile_overflow': False, 'post_requests': posts, 'offline_network_requests': network,
                  'direct_file_opening_tested': False,
                  'offline_render_mode': 'in_memory_html',
                  'javascript_errors': errors + offline_errors}
        (output / 'browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()

asyncio.run(check())
