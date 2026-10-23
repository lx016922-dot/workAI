"""Read-only browser acceptance of a completed real research job."""
import asyncio
import json
import os
from pathlib import Path
import shutil
from playwright.async_api import async_playwright, expect

async def check():
    origin = os.environ.get('WORKAI_TEST_ORIGIN', 'http://127.0.0.1:3000')
    output = Path(os.environ.get('WORKAI_TEST_OUTPUT', '/workspace/workai-remote/validation/20261022-research-source'))
    output.mkdir(parents=True, exist_ok=True)
    job_id = 'comsol-a0faea1fd16642f99831207a326fc705'
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = await browser.new_page(viewport={'width': 1440, 'height': 1050})
        errors, posts = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: posts.append(request.url) if request.method == 'POST' else None)
        await page.goto(origin + '/#results')
        await page.locator('#job-select').select_option(job_id)
        await expect(page.get_by_role('heading', name='时域 A 扫 · 线性导波', exact=True)).to_be_visible(timeout=60000)
        await expect(page.locator('.waveform-panel svg polyline')).to_have_count(3)
        await expect(page.locator('#trace-select option')).to_have_count(3)
        await page.locator('#trace-select').select_option('0:2')
        await expect(page.locator('#trace-select')).to_have_value('0:2')
        await page.locator('#component-select').select_option('u_nm')
        await expect(page.locator('.waveform-panel')).not_to_contain_text('NaN')
        await page.locator('#component-select').select_option('w_nm')
        await expect(page.get_by_text('参考分量为零，未定义', exact=True)).to_have_count(2)
        await page.locator('#component-select').select_option('v_nm')
        await page.locator('#trace-select').select_option('0:0')
        metrics = await page.locator('.waveform-metrics').inner_text()
        await page.locator('.waveform-panel').screenshot(path=str(output / 'ascan-desktop.png'))
        await page.set_viewport_size({'width': 390, 'height': 844})
        assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
        await page.locator('.waveform-panel').screenshot(path=str(output / 'ascan-mobile.png'))
        assert not errors, errors
        assert not posts, posts
        report = {'passed': True, 'job_id': job_id, 'three_real_solve_levels': True, 'three_receivers': True, 'component_selection': True, 'zero_reference_norm_not_reported_as_zero_error': True, 'mobile_overflow': False, 'displayed_first_receiver_metrics': metrics, 'javascript_errors': errors, 'post_requests': posts}
        (output / 'ascan-browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()

asyncio.run(check())
