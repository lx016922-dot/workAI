"""Independent NumPy/SciPy check of already signature-verified CSV artifacts.

Usage: python tools/verify-nonlinear-results.py JOB_ID EVIDENCE_DIRECTORY
No remote requests or computation submissions. Input artifact hashes are rechecked.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import os
import numpy as np
from scipy.optimize import brentq
from scipy.signal import hilbert
os.environ.setdefault('MPLCONFIGDIR', '/tmp/workai-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

job_id, directory = sys.argv[1], Path(sys.argv[2])
report = json.loads((directory / (job_id + '-waveforms.json')).read_text())
assert report['verified'] and report['job_id'] == job_id
artifact_root = directory.parents[1] / 'requests' / job_id / 'artifacts'
for artifact in report['artifacts']:
    data = (artifact_root / artifact['name']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == artifact['sha256'], artifact['name']
case, analysis = report['scene'], report['harmonics']
assert analysis['available']
freq = case['source']['frequency_mhz']
rho, lam, mu = case['material']['rho_kg_m3'], case['nonlinear']['lambda_pa'], case['nonlinear']['mu_pa']
cl, ct = np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho)
half = case['geometry']['thickness_mm'] * .0005
thin = 2 * ct * np.sqrt(1 - (ct / cl) ** 2)
def s0(frequency):
    omega = 2 * np.pi * frequency
    def dispersion(speed):
        k = omega / speed
        alpha, q = np.sqrt(k*k - (omega/cl)**2), np.sqrt((omega/ct)**2 - k*k)
        return np.tan(q*half)/np.tanh(alpha*half) - 4*k*k*alpha*q/(q*q-k*k)**2
    speed = brentq(dispersion, max(thin * .9, np.sqrt(2) * ct * 1.001), cl * .999)
    return omega / speed
independent_k = [s0(freq * 1e6), s0(2 * freq * 1e6)]
for i, key in enumerate(['fundamental', 'second']):
    assert abs(independent_k[i] / analysis['dispersion'][key]['wavenumber_rad_m'] - 1) < 1e-10
tables = {level: np.genfromtxt(artifact_root / (level + '.csv'), delimiter=',', names=True) for level in ['coarse', 'fine', 'timecheck']}
def trace(level, shot, rx):
    data = tables[level]
    return data[(data['shot'] == shot) & (data['rx'] == rx)]
def coefficient(time, values, frequency, window):
    start, end = window
    weights = np.where((time >= start) & (time <= end), .5 * (1 - np.cos(2 * np.pi * (time - start) / (end - start))), 0)
    mean = np.sum(values * weights) / np.sum(weights)
    return 2 * np.sum(weights * (values - mean) * np.exp(-2j * np.pi * frequency * time)) / np.sum(weights)
checks, arrivals = [], []
coefficients = {}
for row in analysis['rows']:
    assert row['available']
    rx = row['rx']
    for level in tables:
        waves = [trace(level, shot, rx) for shot in range(5)]
        time = waves[0]['time_s'] * 1e6
        values = [wave['u_m'] * 1e9 for wave in waves]
        mat = coefficient(time, values[2] - values[1], 2 * freq, row['window_us'])
        double = coefficient(time, values[4] - values[3], 2 * freq, row['window_us'])
        expected = row['levels'][level]['material_second']
        assert abs(mat - complex(expected['re_nm'], expected['im_nm'])) < 2e-11
        assert abs(abs(double/mat) - row['levels'][level]['amplitude_scaling']) < 1e-9
        coefficients[(rx, level)] = mat
        checks.append({'rx': rx, 'level': level, 'material_second_nm': float(abs(mat)), 'double_ratio': float(abs(double/mat))})
    linear = trace('timecheck', 0, rx)
    envelope = abs(hilbert(linear['u_m']))
    index = int(np.argmax(envelope)); assert 0 < index < len(envelope) - 1
    sub = .5 * (envelope[index-1] - envelope[index+1]) / (envelope[index-1] - 2*envelope[index] + envelope[index+1])
    arrival = linear['time_s'][index] + sub * np.median(np.diff(linear['time_s']))
    arrivals.append({'x_mm': row['x_mm'], 'envelope_peak_us': float(arrival * 1e6)})
slope, intercept = np.polyfit([r['x_mm'] * .001 for r in arrivals], [r['envelope_peak_us'] * 1e-6 for r in arrivals], 1)
speed = 1 / slope
spatial_checks = []
if analysis.get('spatial_reference'):
    ref = analysis['spatial_reference']
    ref_root = artifact_root.parent.parent / ref['reference_job_id'] / 'artifacts'
    for artifact in ref['reference_artifacts']:
        assert hashlib.sha256((ref_root / artifact['name']).read_bytes()).hexdigest() == artifact['sha256']
    ref_request = json.loads((ref_root / 'request.json').read_text())
    ref_scene = json.loads(ref_request['scene_json'])
    physical_a, physical_b = dict(case), dict(ref_scene)
    del physical_a['numerics']; del physical_b['numerics']
    assert physical_a == physical_b and ref_request['source_sha256'] == report['source_sha256']
    assert ref['reference_elements'] > ref['target_elements']
    ref_csv = np.genfromtxt(ref_root / (ref['reference_level'] + '.csv'), delimiter=',', names=True)
    for row in analysis['rows']:
        rx = row['rx']; material = ref_csv[(ref_csv['shot'] == 2) & (ref_csv['rx'] == rx)]; geometric = ref_csv[(ref_csv['shot'] == 1) & (ref_csv['rx'] == rx)]
        fine = coefficient(material['time_s'] * 1e6, material['u_m'] * 1e9 - geometric['u_m'] * 1e9, 2 * freq, row['window_us'])
        error = abs(coefficients[(rx, ref['target_level'])] - fine) / abs(fine)
        assert abs(error - row['mesh_relative_complex_difference']) < 1e-9
        time_error = abs(coefficients[(rx, 'fine')] - coefficients[(rx, 'timecheck')]) / abs(coefficients[(rx, 'timecheck')])
        assert abs(time_error - row['time_relative_complex_difference']) < 1e-9
        spatial_checks.append({'rx': rx, 'mesh_and_tolerance_relative_complex_difference': float(error), 'time_relative_complex_difference': float(time_error)})
result = {'job_id': job_id, 'cached_file_hashes_match_signed_download_manifest': True, 'numpy_complex_projection_matches_javascript': True,
          'scipy_s0_wavenumbers_rad_m': independent_k, 'checks': checks, 'linear_envelope_peaks': arrivals,
          'envelope_speed_m_s': float(speed), 'predicted_s0_group_velocity_m_s': analysis['dispersion']['fundamental']['group_velocity_m_s'],
          'independent_cross_job_checks': spatial_checks,
          'scope': 'Independent numerical implementation checks; envelope/phase agreement does not establish modal purity or experimental calibration.'}
(directory / (job_id + '-independent-check.json')).write_text(json.dumps(result, indent=2))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
rows = analysis['rows']; xs = [r['x_mm'] for r in rows]
colors = {'coarse': '#148778', 'fine': '#648fc3', 'timecheck': '#b67e4e'}
for level in colors:
    ax[0, 0].plot(xs, [r['levels'][level]['material_second']['amplitude_nm'] for r in rows], 'o-', color=colors[level], label=level)
ax[0, 0].set(xlabel='Propagation distance / mm', ylabel='Material increment at 2f / nm', title='Generated second harmonic'); ax[0, 0].legend()
ax[0, 1].plot(xs, [r['levels']['timecheck']['amplitude_scaling'] for r in rows], 'o-', color='#148778', label='Simulated')
ax[0, 1].axhline(4, color='#777', linestyle='--', label='Quadratic scaling: 4')
ax[0, 1].set(xlabel='Propagation distance / mm', ylabel='Increment(2A) / increment(A)', ylim=(0, 4.5), title='Amplitude control'); ax[0, 1].legend()
mesh_label = 'Mesh + solver tolerance' if analysis.get('spatial_reference') else 'Mesh'
for key, title, color in [('mesh_relative_complex_difference', mesh_label, '#148778'), ('time_relative_complex_difference', 'Time step', '#b67e4e')]:
    ax[1, 0].plot(xs, [100*r[key] for r in rows], 'o-', color=color, label=title)
ax[1, 0].axhline(5, color='#777', linestyle='--', label='Current check: 5%')
ax[1, 0].set(xlabel='Propagation distance / mm', ylabel='Complex coefficient difference / %', title='Discretization check'); ax[1, 0].legend()
wave, base = trace('timecheck', 2, len(rows)-1), trace('timecheck', 1, len(rows)-1)
ax[1, 1].plot(wave['time_s']*1e6, (wave['u_m']-base['u_m'])*1e9, color='#148778')
ax[1, 1].set(xlabel='Time / microsecond', ylabel='Full minus geometric displacement / nm', title=f'Material residual at {xs[-1]:g} mm')
fig.suptitle(f'Aluminum reference: {freq:g} MHz, {case["geometry"]["thickness_mm"]:g} mm plate\nFinite pulse; no modal separation or experimental calibration', fontsize=12)
if analysis.get('spatial_reference'):
    ref = analysis['spatial_reference']
    fig.supxlabel(
        f'Spatial check: {ref["target_elements"]}/{ref["reference_elements"]} elements from two verified jobs, '
        f'same dt = {ref["internal_step_s"] * 1e6:g} microsecond; finer reference has tighter tolerances.\n'
        'The target job alone does not establish spatial refinement; see the retained cross-job reference.',
        fontsize=8)
fig.savefig(directory / (job_id + '-verification.png'), dpi=180)
fig.savefig(directory / (job_id + '-verification.pdf'))
print(json.dumps({k: v for k, v in result.items() if k not in ['checks']}, indent=2))
