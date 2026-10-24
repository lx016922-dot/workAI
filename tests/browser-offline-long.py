"""Real 5 ms browser waveforms and CSV vs independent chunk interpolation."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright,expect
parser=argparse.ArgumentParser()
parser.add_argument('--url',required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--evidence',type=Path,required=True)
args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=True)
job='local-20261024000000000000000000000005';directory=args.data/'jobs'/job
point=np.array([3.3,1.1,.8]);direction=np.array([1.,-.6,.2]);direction/=np.linalg.norm(direction)
with np.load(directory/'record/mesh.npz') as mesh:
    e=np.flatnonzero(np.all((point*.001>=mesh['origin']-1e-12)&(point*.001<=mesh['origin']+mesh['size']+1e-12),axis=1))[0]
    local=2*(point*.001-mesh['origin'][e])/mesh['size'][e]-1;nodes=mesh['gll'];ids=mesh['elements'][e].ravel()
    rows=np.array([[np.prod([(x-nodes[j])/(nodes[i]-nodes[j]) for j in range(len(nodes)) if i!=j]) for i in range(len(nodes))] for x in local]);w=np.einsum('i,j,k->ijk',*rows).ravel()
m=json.loads((directory/'record/recording.json').read_text());expected={k:[] for k in ['full','linear','stvk','material','geometric']}
for chunk in m['chunks']:
    with np.load(directory/'record'/chunk['file']) as data:
        linear,stvk,full=[np.einsum('n,tnc->tc',w,data[k][:,ids]) for k in ['linear','stvk','murnaghan']]
        for k,a in [('linear',linear),('stvk',stvk),('full',full),('geometric',stvk-linear),('material',full-stvk)]:expected[k].append(a)
expected={k:np.concatenate(v) for k,v in expected.items()}
errors=[];outside=[];results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
    page.on('pageerror',lambda e:errors.append(str(e)))
    def route(r):
        if r.request.url.startswith(args.url):r.continue_()
        else:outside.append(r.request.url);r.abort()
    page.route('**/*',route);page.goto(args.url)
    expect(page.locator('#solid-canvas')).to_be_visible(timeout=60000)
    expect(page.locator('[data-path="acquisition.duration_us"]')).to_have_value('5')
    expect(page.locator('#solid-window-end')).to_have_value('5000')
    assert float(page.locator('#solid-window-end').get_attribute('max'))==5000
    for kind in ['hann','gaussian','rectangular','chirp']:
        page.select_option('[data-path="signal.kind"]',kind)
        for cycles in [1,3,5,10,20]:page.fill('[data-path="source.cycles"]',str(cycles))
        if kind=='chirp':expect(page.locator('[data-path="signal.end_frequency_mhz"]')).to_be_visible()
    page.select_option('[data-path="signal.kind"]','hann')
    page.select_option('#degree','5')
    with page.expect_download() as pending:page.click('#save-scene')
    config=args.evidence/'config.json';pending.value.save_as(config);assert json.loads(config.read_text())['degree']==5
    for part in ['full','linear','stvk','geometric','material']:
        if part!='full':
            page.select_option('#part',part);expect(page.locator('#result-tools')).to_be_visible(timeout=60000)
        for i,v in enumerate(point):page.fill('#solid-point-'+str(i),str(v))
        for i,v in enumerate([1.,-.6,.2]):page.fill('#solid-direction-'+str(i),str(v))
        page.fill('#solid-window-start','0');page.fill('#solid-window-end','5000');page.click('#solid-extract')
        expect(page.locator('#solid-probe-message')).to_contain_text('已提取 [3.3, 1.1, 0.8]',timeout=120000)
        expect(page.locator('#solid-csv')).to_be_enabled()
        with page.expect_download() as pending:page.click('#solid-csv')
        path=args.evidence/(part+'.csv');pending.value.save_as(path);lines=path.read_text().splitlines()
        a=next(i for i,x in enumerate(lines) if x.startswith('time_us,'))+1;b=next(i for i in range(a,len(lines)) if lines[i]=='')
        values=np.array([[float(v) for v in row.split(',')] for row in lines[a:b]])
        assert len(values)==25001 and values[-1,0]==5000
        np.testing.assert_allclose(values[:,1:4],expected[part],rtol=5e-10,atol=3e-22)
        np.testing.assert_allclose(values[:,4],expected[part]@direction,rtol=5e-10,atol=3e-22)
        projected=values[:,4];window=np.hanning(len(projected));centered=projected-np.dot(window,projected)/sum(window)
        for row in [x for x in lines if x.startswith('# harmonic,')]:
            h=np.array([float(v) for v in row.split(',')[1:]]);c=2*np.dot(window*centered,np.exp(-2j*np.pi*h[1]*values[:,0]))/sum(window)
            np.testing.assert_allclose(h[2:4],[c.real,c.imag],rtol=1e-8,atol=1e-22)
        size=1<<(len(projected)-1).bit_length();fft=abs(np.fft.rfft(window*centered,n=size))*2/sum(window);fft[[0,-1]]/=2
        spectrum=np.array([[float(v) for v in row.split(',')] for row in lines[b+2:]])
        np.testing.assert_allclose(spectrum[:,1],fft,rtol=1e-10,atol=64*np.finfo(float).eps*max(fft))
        fft_error=float(np.linalg.norm(spectrum[:,1]-fft)/np.linalg.norm(fft))
        assert fft_error<1e-12
        results.append({'part':part,'samples':len(values),'last_time_us':values[-1,0],'max_abs_xyz_error_m':float(np.max(abs(values[:,1:4]-expected[part]))),'fft_relative_l2':fft_error})
    page.fill('#solid-point-0','-100');page.click('#solid-extract')
    expect(page.locator('#solid-probe-message')).to_contain_text('outside',timeout=30000)
    expect(page.locator('#solid-csv')).to_be_disabled()
    page.select_option('#solid-receiver','1');page.fill('#solid-path-count','2');page.click('#solid-path-analyze')
    expect(page.locator('#solid-path-csv')).to_be_enabled(timeout=120000)
    with page.expect_download() as pending:page.click('#solid-path-csv')
    pending.value.save_as(args.evidence/'path.csv')
    page.screenshot(path=str(args.evidence/'desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(args.evidence/'mobile.png'),full_page=True)
    assert not errors,errors
    assert not outside,outside
    browser.close()
receipt={'passed':True,'actual_duration_us':5000,'full_samples':25001,'responses':results,'fft_and_complex_coefficients_match_numpy':True,'multi_point_csv':True,'all_four_signal_controls':True,'cycles':[1,3,5,10,20],'config_preserves_degree':True,'outside_requests':outside,'javascript_errors':errors}
(args.evidence/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
