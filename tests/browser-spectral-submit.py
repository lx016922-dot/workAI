"""One real high-order UI solve, followed by read-only numerical extraction checks."""
import argparse,asyncio,csv,gzip,json,shutil,sys
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright,expect
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral3d.mesh import interpolation


async def main(args):
    root=Path('/workspace/workai-remote');args.evidence.mkdir(parents=True,exist_ok=False);posts=[];errors=[]
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        page=await browser.new_page(viewport={'width':1440,'height':1050},accept_downloads=True)
        page.on('pageerror',lambda error:errors.append(str(error)));page.on('request',lambda request:posts.append(request.url) if request.method=='POST' else None)
        await page.goto(args.origin+'/#solid');await expect(page.locator('#solid-method')).to_be_visible(timeout=30000)
        for kind in ['pipe','hole','notch','plate']:await page.locator('#solid-template').select_option(kind)
        await page.locator('#solid-template').select_option(args.kind)
        await page.locator('#solid-method').select_option('spectral');await page.locator('#solid-degree').select_option('3')
        if args.existing:
            job=args.existing;await page.locator('#solid-job-select').select_option(job)
        else:
            for path,value in {'source.frequency_mhz':'.25','source.cycles':'1.5','source.direction.1':'.2','source.direction.2':'-.1','acquisition.duration_us':'8','numerics.h_mm':'2'}.items():
                control=page.locator('[data-solid-path="'+path+'"]');await control.fill(value);await control.press('Tab')
            async with page.expect_response(lambda r:r.url==args.origin+'/api/spectral/jobs' and r.request.method=='POST') as reply:
                await page.locator('#solid-submit').click()
            response=await reply.value;body=await response.json();assert response.status==202,body;job=body['job_id']
            (args.evidence/'submission.json').write_text(json.dumps(body,indent=2))
        await expect(page.locator('#solid-result-container canvas')).to_be_visible(timeout=120000)
        await expect(page.locator('#solid-waveform svg')).to_be_visible();await expect(page.locator('#solid-result-container .notice').first).to_contain_text('高阶超弹性有限元参考场')
        record=await (await page.request.get(args.origin+'/api/spectral/jobs/'+job+'/result')).json()
        assert record['spectral']['degree']==3 and record['accepted_for_solver_routing'] is False and record['scene']['source']['frequency_mhz']==.25
        before=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
        await page.locator('#solid-frame').evaluate('(e)=>{e.value=100;e.dispatchEvent(new Event("input",{bubbles:true}));}')
        assert before!=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
        point=[3.3,2.5,.2] if record['scene']['geometry']['kind']=='pipe' else [3.3,1.7,.8];direction=np.array([2.,-3.,1.]);direction/=np.linalg.norm(direction)
        for i,value in enumerate(point):await page.locator('#solid-point-'+str(i)).fill(str(value))
        for i,value in enumerate([2,-3,1]):await page.locator('#solid-direction-'+str(i)).fill(str(value))
        await page.locator('#solid-extract').click();await expect(page.locator('#solid-spectrum svg')).to_be_visible()
        async with page.expect_download() as downloaded:await page.locator('#solid-csv').click()
        csv_path=args.evidence/'direction.csv';await (await downloaded.value).save_as(csv_path)
        rows=list(csv.reader(line for line in csv_path.read_text().splitlines() if not line.startswith('#')));count=len(record['times_us']);values=np.asarray(rows[1:count+1],float)
        directory=root/'spectral3d/jobs'/job
        with np.load(directory/'mesh.npz',allow_pickle=False) as archive:mesh=dict(archive)
        fields=np.frombuffer(gzip.decompress((directory/'field.bin.gz').read_bytes()),dtype='<f8').reshape(record['field']['shape']);ids,weights=interpolation(mesh,point);truth=np.einsum('tnc,n->tc',fields[:,ids],weights)
        np.testing.assert_allclose(values[:,1:4],truth,rtol=1e-10,atol=1e-21);np.testing.assert_allclose(values[:,4],truth@direction,rtol=1e-10,atol=1e-21)
        w=np.hanning(count);projected=truth@direction;centered=projected-np.dot(w,projected)/w.sum();size=1<<(count-1).bit_length();spectrum=abs(np.fft.rfft(centered*w,n=size))*2/w.sum();spectrum[[0,-1]]/=2
        np.testing.assert_allclose(np.asarray(rows[count+3:],float)[:,1],spectrum,rtol=1e-10,atol=1e-21)
        await page.locator('#solid-result-container').screenshot(path=str(args.evidence/'result-desktop.png'))
        await page.locator('#solid-reuse').click();await expect(page.locator('#solid-method')).to_have_value('spectral');await expect(page.locator('#solid-degree')).to_have_value('3');await expect(page.locator('[data-solid-path="source.frequency_mhz"]')).to_have_value('0.25')
        await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(args.evidence/'result-mobile.png'),full_page=True)
        for i in range(3):await page.locator('#solid-direction-'+str(i)).fill('0')
        await page.locator('#solid-extract').click();await expect(page.locator('#solid-csv')).to_be_disabled()
        # The same UI must still read an earlier tetrahedral job.
        previous=json.loads((root/'validation/20261022-solid/final-cases.json').read_text())['cases'][0]['display_job']
        await page.locator('#solid-job-select').select_option(previous);await expect(page.locator('#solid-waveform svg')).to_be_visible(timeout=30000)
        assert not errors and len(posts)==(0 if args.existing else 1),(errors,posts)
        result={'passed':True,'job_id':job,'geometry':record['scene']['geometry']['kind'],'new_solver_submissions':len(posts),'source':'custom_UI_parameters','numerical_solver_not_AI':True,'csv_xyz_and_direction_match_field':True,'spectrum_matches_numpy':True,'animation_changes':True,'reused_parameters_preserve_solver':True,'previous_tetrahedral_result_readable':True,'mobile_no_overflow':True,'javascript_errors':errors}
        (args.evidence/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--origin',default='http://127.0.0.1:3000');parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--existing');parser.add_argument('--kind',choices=['plate','pipe','hole','notch'],default='plate');asyncio.run(main(parser.parse_args()))
