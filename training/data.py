"""Group splits of one physical case; mesh repeats are never extra cases."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .physics import L0, T0, U0

JOB_ID = 'comsol-d62800aa89a8478e85d6478acd71d783'
SOURCE = '528b3b8190866b8f1fbe05f27d093370a2e75c9288e8d91aaf654a8b39a09513'
MATERIAL = '39102235cd2fae341a277e46aa08573ae2294710c79abd104b10491fe5edef0a'

def load_data(remote_root):
    root = Path(remote_root); evidence = root / 'validation/20261022-nonlinear-comsol'
    report = json.loads((evidence / (JOB_ID+'-waveforms.json')).read_text())
    extra = json.loads((evidence / (JOB_ID+'-extra-artifacts.json')).read_text())
    assert report['verified'] and report['job_id'] == extra['job_id'] == JOB_ID
    assert report['source_sha256'] == SOURCE and report['scene']['material_reference']['record_sha256'] == MATERIAL
    assert report['harmonics']['all_checks_within_5_percent']
    manifest = {a['name']: a for a in report['artifacts'] + extra['artifacts']}
    artifacts = root / 'requests' / JOB_ID / 'artifacts'
    checked = []
    for name in ['request.json', 'completed.properties', 'timecheck.csv', 'timecheck.field.csv']:
        p = artifacts / name; assert p.is_file() and not p.is_symlink()
        data = p.read_bytes(); record = manifest[name]
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], name
        checked.append(record)
    request = json.loads((artifacts/'request.json').read_text())
    assert request['job_id'] == JOB_ID and request['source_sha256'] == SOURCE
    assert hashlib.sha256(request['scene_json'].encode()).hexdigest() == request['scene_sha256'] == report['receipt']['scene_sha256']
    assert hashlib.sha256(request['parameters_json'].encode()).hexdigest() == request['parameters_sha256'] == report['receipt']['parameters_sha256']
    scene = json.loads(request['scene_json']); assert scene == report['scene']
    card = Path(__file__).parents[1]/'data/materials/aluminum-polycrystal-blaschke2017.json'
    assert hashlib.sha256(card.read_bytes()).hexdigest() == MATERIAL
    h = scene['geometry']['thickness_mm'] * .001; length = scene['geometry']['length_mm'] * .001
    scans = np.genfromtxt(artifacts/'timecheck.csv', delimiter=',', names=True)
    field = np.genfromtxt(artifacts/'timecheck.field.csv', delimiter=',', names=True)
    def grouped(table, indices):
        groups = [table[table['shot'] == shot] for shot in range(5)]
        reference = groups[0]
        assert all(len(g) == len(reference) for g in groups)
        assert all(np.array_equal(g[list(indices)], reference[list(indices)]) for g in groups)
        raw = np.stack([np.stack([g['u_m'], g['v_m']], 1) / U0 for g in groups], 1)
        assert np.isfinite(raw).all() and all(np.max(abs(g['w_m'])) < 1e-20 for g in groups)
        parts = np.stack([raw[:, 0], raw[:, 1]-raw[:, 0], raw[:, 2]-raw[:, 1]], 1)
        return reference, raw, parts
    scan, raw_scan, scan_parts = grouped(scans, ['rx', 'time_s'])
    f, raw_field, field_parts = grouped(field, ['frame', 'ix', 'iy', 'time_s'])
    x = np.array([scene['receivers'][int(rx)]['x_mm']*.001 for rx in scan['rx']])
    scan_q = np.stack([x/L0, np.full_like(x, (h/2-1e-9)/L0), scan['time_s']/T0], 1)
    nx, ny = scene['numerics']['field_nx'], scene['numerics']['field_ny']
    fx = np.clip(f['ix']/(nx-1)*length, 1e-9, length-1e-9)
    fy = np.clip(f['iy']/(ny-1)*h, 1e-9, h-1e-9)
    field_q = np.stack([fx/L0, (fy-h/2)/L0, f['time_s']/T0], 1)
    scan_train = (scan['rx'] != 1) & (scan['time_s'] > 0)
    scan_val = scan['rx'] == 1
    # Hold the entire 40 mm location (and neighboring field columns) out of labels.
    field_train = (f['frame'] == 1) & (abs(fx-.04) > .0041)
    # Remove duplicate/mirror labels already represented by receiver training.
    field_train &= ~((np.isclose(fx,.02) | np.isclose(fx,.06)) & (f['iy'] != 1))
    field_val = (f['frame'] == 2) & ~(np.isclose(fx,.02) | np.isclose(fx,.06))
    train_q = np.concatenate([scan_q[scan_train], field_q[field_train]])
    train_y = np.concatenate([scan_parts[scan_train], field_parts[field_train]])
    floors = np.array([[.5,.05],[1e-4,2e-5],[1e-4,2e-5]])
    scales = np.maximum(np.max(abs(train_y), axis=0), floors)
    assert not np.any(np.isclose(train_q[:, 0]*L0,.04))
    return dict(scene=scene, report=report, train_q=train_q, train_y=train_y, scales=scales,
                val_q=scan_q[scan_val], val_y=scan_parts[scan_val], field_val_q=field_q[field_val], field_val_y=field_parts[field_val],
                scan_q=scan_q, raw_scan=raw_scan, scan_rx=scan['rx'].astype(int),
                manifest={'schema':'workai.training.dataset.v1','job_id':JOB_ID,'source_sha256':SOURCE,'material_sha256':MATERIAL,'artifacts':checked,
                          'physical_case_count':1,'training_points':len(train_q),'validation_receiver_x_mm':40,'validation_points':int(scan_val.sum()),
                          'late_field_validation_points':int(field_val.sum()),'training_controls':[0,1,2],'held_out_amplitude_controls':[3,4],
                          'no_random_time_split':True,'mesh_repeats_used_as_extra_cases':False,'normalization_from_training_only':True,
                          'scope':'1 mm aluminum, 0.4 MHz, 160 mm plate, 5 MPa training; 10 MPa is an amplitude consistency check, not a new material/frequency case.',
                          'verification':'Cached bytes rehashed against prior signature-verified download records. Training does not itself perform RSA verification.'})
