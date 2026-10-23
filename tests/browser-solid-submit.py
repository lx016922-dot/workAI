"""Authorized end-to-end test: imports generated STEP and runs one real local solve."""
import asyncio,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect

ROOT=Path('/workspace/workai-remote');ORIGIN='http://127.0.0.1:3000'
async def main():
    evidence=ROOT/'validation/20261022-solid';posts=[];errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        page=await browser.new_page(viewport={'width':1440,'height':1100})
        page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(ORIGIN+'/#solid');await expect(page.locator('#solid-cad-file')).to_be_visible(timeout=30000)
        await page.locator('#solid-cad-file').set_input_files(evidence/'cad-fixtures/L-bracket.step')
        await page.locator('#solid-upload').click();await expect(page.locator('#solid-template')).to_have_value('cad',timeout=30000)
        await page.locator('[data-solid-path="source.face"]').select_option('surface_patch')
        for i,v in enumerate(['1','.2','.1']):
            field=page.locator(f'[data-solid-path="source.direction.{i}"]');await field.fill(v);await field.press('Tab')
        async with page.expect_response(lambda r:r.url==ORIGIN+'/api/solid/jobs' and r.request.method=='POST') as reply:
            await page.locator('#solid-submit').click()
        response=await reply.value;body=await response.json();assert response.status==202,body;job=body['job_id']
        await expect(page.locator('#solid-result-container canvas')).to_be_visible(timeout=120000)
        await expect(page.locator('#solid-waveform svg')).to_be_visible()
        state=await (await page.request.get(ORIGIN+'/api/solid/jobs/'+job)).json();assert state['phase']=='completed'
        result=await (await page.request.get(ORIGIN+'/api/solid/jobs/'+job+'/result')).json()
        assert result['scene']['geometry']['kind']=='cad' and result['scene']['source']['face']=='surface_patch'
        assert result['diagnostics']['surrogate_used'] is False and result['load_area_mm2']>0
        assert len([p for p in posts if p.endswith('/api/solid/jobs')])==1
        for i,v in enumerate(['9','5','1']):await page.locator(f'#solid-point-{i}').fill(v)
        await page.locator('#solid-extract').click();await expect(page.locator('#solid-probe-message')).to_contain_text('实体之外')
        for i,v in enumerate(['9','2','1']):await page.locator(f'#solid-point-{i}').fill(v)
        for i,v in enumerate(['0','0','1']):await page.locator(f'#solid-direction-{i}').fill(v)
        await page.locator('#solid-extract').click();await expect(page.locator('#solid-probe-message')).to_contain_text('未重新求解')
        await page.locator('#solid-frequency-limit').fill('5');await page.locator('#solid-frequency-limit').press('Tab')
        await expect(page.locator('#solid-spectrum-note')).to_contain_text('0–5.00 MHz')
        await page.locator('#solid-result-container').screenshot(path=str(evidence/'cad-live-result.png'))
        assert not errors,errors
        receipt={'passed':True,'job_id':job,'geometry':'locally_generated_L_bracket_STEP','source':'curved_surface_patch_arbitrary_traction_direction','exactly_one_solver_submission':True,'post_requests':posts,'cut_out_not_filled':True,'posthoc_z_displacement_and_spectrum':True,'javascript_errors':errors,'not_comsol_or_ai':True}
        (evidence/'browser-submit-acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));await browser.close()

asyncio.run(main())
