"""Read-only acceptance of an explicitly selected real nonlinear job."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
from playwright.async_api import async_playwright, expect

async def check():
    job_id = sys.argv[1]
    expected = sys.argv[2] == 'true'
    origin = os.environ.get('WORKAI_TEST_ORIGIN', 'http://127.0.0.1:3000')
    output = Path('/workspace/workai-remote/validation/20261022-nonlinear-comsol') / job_id
    output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = await browser.new_page(viewport={'width': 1440, 'height': 1050}, accept_downloads=True)
        errors, posts = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda r: posts.append(r.url) if r.method == 'POST' else None)
        await page.goto(origin + '/#results')
        await page.locator('#job-select').select_option(job_id)
        await expect(page.get_by_role('heading', name='材料引起的二次谐波', exact=True)).to_be_visible(timeout=60000)
        await expect(page.locator('.harmonic-panel .pill')).to_have_text('本算例检查通过' if expected else '数值检查待通过')
        await expect(page.locator('.harmonic-panel tbody tr')).to_have_count(3)
        await expect(page.locator('.harmonic-panel svg circle')).to_have_count(3)
        await expect(page.locator('#trace-select option')).to_have_count(15)
        await expect(page.locator('.waveform-panel svg polyline')).to_have_count(3)
        await page.locator('#trace-select').select_option('2:2')
        await expect(page.locator('#component-select')).to_have_value('u_nm')
        await expect(page.locator('.harmonic-panel')).not_to_contain_text('NaN')
        await page.locator('.harmonic-panel details').evaluate('(node) => node.open = true')
        async with page.expect_download() as event:
            await page.get_by_role('button', name='下载波形与分析 JSON', exact=True).click()
        exported = output / 'browser-waveforms.json'
        await (await event.value).save_as(exported)
        data = json.loads(exported.read_text())
        assert data['job_id'] == job_id and data['verified'] is True
        assert data['harmonics']['all_checks_within_5_percent'] == expected
        assert data['constitutive'] == 'murnaghan_green_lagrange'
        assert len(data['harmonics']['linear_phase_checks']) == 2
        await page.locator('.harmonic-panel details').evaluate('(node) => node.open = false')
        await page.locator('.harmonic-panel').screenshot(path=str(output / 'harmonics-desktop.png'))
        await page.locator('.waveform-panel').screenshot(path=str(output / 'nonlinear-ascan-desktop.png'))
        await page.set_viewport_size({'width': 390, 'height': 844})
        assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
        await page.locator('.harmonic-panel').screenshot(path=str(output / 'harmonics-mobile.png'))
        assert not errors, errors
        assert not posts, posts
        report = {'passed': True, 'job_id': job_id, 'displayed_numerical_checks_passed': expected, 'fifteen_real_traces': True, 'harmonic_rows': 3, 'verified_analysis_download': True, 'mobile_overflow': False, 'javascript_errors': errors, 'post_requests': posts}
        (output / 'browser-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()

asyncio.run(check())
