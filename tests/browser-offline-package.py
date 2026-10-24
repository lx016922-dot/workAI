"""Check real browser exports against independent NumPy field interpolation."""
import argparse
import csv
import gzip
import io
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--url', required=True)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--evidence', type=Path, required=True)
args = parser.parse_args()
args.evidence.mkdir(parents=True, exist_ok=True)
errors, outside = [], []
job = 'local-20261024000000000000000000000001'
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path=shutil.which('chromium'), headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
    page = browser.new_page(viewport={'width': 1400, 'height': 1000}, accept_downloads=True)
    page.on('pageerror', lambda error: errors.append(str(error)))
    def route(request):
        if request.request.url.startswith(args.url):
            request.continue_()
        else:
            outside.append(request.request.url)
            request.abort()
    page.route('**/*', route)
    page.goto(args.url, wait_until='networkidle')
    page.wait_for_selector('#solid-canvas')
    rows = []
    for part in ['full', 'linear', 'stvk', 'geometric', 'material']:
        if part != 'full':
            page.select_option('#part', part)
            page.wait_for_selector('#result-tools:not([hidden])')
        page.fill('#solid-direction-0', '1')
        page.fill('#solid-direction-1', '-0.6')
        page.fill('#solid-direction-2', '0.2')
        page.fill('#solid-window-start', '.123')
        page.fill('#solid-window-end', '12.4')
        page.click('#solid-extract')
        assert page.locator('#solid-waveform svg').count() == 1
        with page.expect_download() as pending:
            page.click('#solid-csv')
        path = args.evidence / (part + '.csv')
        pending.value.save_as(path)
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        point = np.array([float(x) for x in next(l for l in lines if l.startswith('# point_mm,')).split(',')[1:]])
        direction = np.array([1., -.6, .2]); direction /= np.linalg.norm(direction)
        start = next(i for i, l in enumerate(lines) if l.startswith('time_us,')) + 1
        stop = next(i for i in range(start, len(lines)) if lines[i] == '')
        actual = np.array([[float(v) for v in line.split(',')] for line in lines[start:stop]])
        directory = args.data / 'jobs' / job
        result = json.loads((directory / (part + '.json')).read_text())
        field = np.frombuffer(gzip.decompress((directory / (part + '.bin.gz')).read_bytes()), dtype='<f8').reshape(result['field']['shape'])
        mesh = result['spectral']
        origins, sizes = np.array(mesh['origins_mm']), np.array(mesh['sizes_mm'])
        e = np.flatnonzero(np.all((point >= origins-1e-9) & (point <= origins+sizes+1e-9), axis=1))[0]
        coords = 2*(point-origins[e])/sizes[e]-1
        nodes = np.array(mesh['gll'])
        weights = np.array([[np.prod([(x-nodes[j])/(nodes[i]-nodes[j]) for j in range(len(nodes)) if j != i]) for i in range(len(nodes))] for x in coords])
        combined = np.einsum('i,j,k->ijk', *weights).ravel()
        expected = np.einsum('n,tnc->tc', combined, field[:, mesh['elements'][e]])
        np.testing.assert_allclose(actual[:, 1:4], expected, rtol=2e-12, atol=1e-22)
        np.testing.assert_allclose(actual[:, 4], expected@direction, rtol=2e-12, atol=1e-22)
        # Exact-frequency complex coefficients with an independently constructed Hann window.
        mask = (actual[:, 0] >= .123) & (actual[:, 0] <= 12.4)
        times, trace = actual[mask, 0], actual[mask, 4]
        window = np.hanning(len(times)); trace -= np.sum(trace*window)/sum(window)
        harmonic_rows = [l for l in lines if l.startswith('# harmonic,')]
        for order, row in enumerate(harmonic_rows, 1):
            values = [float(v) for v in row.split(',')[1:]]
            coefficient = 2*np.sum(window*trace*np.exp(-2j*np.pi*values[1]*times))/sum(window)
            np.testing.assert_allclose(values[2:4], [coefficient.real, coefficient.imag], rtol=2e-10, atol=1e-21)
        rows.append({'part': part, 'samples': len(actual), 'max_abs_error_m': float(np.max(abs(actual[:, 4]-expected@direction)))})
    page.fill('#solid-point-0', '-100')
    page.click('#solid-extract')
    assert page.locator('#solid-csv').is_disabled()
    assert page.locator('#solid-waveform svg').count() == 0
    page.select_option('#solid-receiver', '1')
    page.click('#solid-path-analyze')
    page.wait_for_timeout(800)
    assert not page.locator('#solid-path-csv').is_disabled()
    with page.expect_download() as pending:
        page.click('#solid-path-csv')
    pending.value.save_as(args.evidence / 'path.csv')
    page.screenshot(path=str(args.evidence / 'desktop.png'), full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_timeout(250)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
    page.screenshot(path=str(args.evidence / 'mobile.png'), full_page=True)
    assert not errors, errors
    assert not outside, outside
    browser.close()
(args.evidence / 'verification.json').write_text(json.dumps({'passed': True, 'responses': rows,
    'external_requests': outside, 'browser_errors': errors, 'invalid_point_rejected': True,
    'path_csv_exported': True, 'mobile_no_horizontal_overflow': True}, indent=2)+'\n')
print(json.dumps({'passed': True, 'responses': len(rows), 'external_requests': len(outside)}))
