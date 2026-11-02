"""Research nonlinear evolution in an independently exported COMSOL modal basis.

This is explicitly a COMSOL-derived reference ROM, not a native COMSOL transient.
Native transient, mesh, temporal, modal, and quadrature admission remain separate.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_forcing import driven_states
from graphwave.signals import specification
from tools.comsol_nodal_reference import digest, nonlinear_force


def evolve(model, scene, step_us, progress=None, force=nonlinear_force):
    """Sixth-order symmetric composition around the accurately driven oscillator."""
    sample, duration = [scene['acquisition'][key] for key in ['sample_step_us', 'duration_us']]
    if not np.isfinite(step_us) or not 0 < step_us <= sample:
        raise ValueError('Finite positive step no longer than the output interval required')
    subdivisions, steps = round(sample / step_us), round(duration / step_us)
    if (abs(subdivisions * step_us - sample) > 1e-10
            or abs(steps * step_us - duration) > 1e-10 or steps % subdivisions):
        raise ValueError('Complete integer output/internal time grid required')
    # Yoshida's symmetric seven-composition, order-six solution.
    outer = np.array([.7845136104775573, .23557321335935813, -1.177679984178871])
    weights = np.r_[outer, 1 - 2 * outer.sum(), outer[::-1]]
    stage_times = (np.arange(steps)[:, None] + np.r_[0., np.cumsum(weights)][None, :]) * step_us
    driven_q, driven_v, forcing = driven_states(scene, model['frequency_hz'], stage_times)
    forcing = dict(forcing, method='DOP853 unit-traction forcing; exact COMSOL-frequency free evolution',
                   comsol_arrays_used=True, frequency_source='verified native COMSOL eigenfrequency export')
    # Native force integrals are per pascal of traction; unlike a candidate
    # boundary load, they have not already absorbed the excitation amplitude.
    traction_amplitude_pa = scene['source']['amplitude_mpa'] * 1e6
    driven_q *= model['load'] * traction_amplitude_pa
    driven_v *= model['load'] * traction_amplitude_pa
    omega = model['omega']; frames = steps // subdivisions + 1
    position = np.zeros((3, len(omega))); velocity = np.zeros_like(position)
    residual = np.zeros_like(position); increment_energy = np.zeros(3); work = np.zeros(3)
    traces = np.zeros((3, frames, len(scene['receivers']), 3))
    coordinates = np.zeros((3, frames, len(omega))); energies = np.zeros((3, frames, 3))
    maximum, minimum = np.zeros(2), np.ones(2)
    delay = scene['source']['delay_us']; end = delay + specification(scene)['duration_us']
    rotations = [(np.cos(omega * w * step_us * 1e-6), np.sin(omega * w * step_us * 1e-6)) for w in weights]
    for index in range(steps):
        for stage, weight in enumerate(weights):
            h = weight * step_us * 1e-6
            kicked = velocity - .5 * h * residual
            active = min(stage_times[index, stage:stage + 2]) < end and max(stage_times[index, stage:stage + 2]) > delay
            if active:
                before = .5 * np.sum((kicked + driven_v[index, stage])**2 + ((position + driven_q[index, stage]) * omega)**2, axis=1)
            cosine, sine = rotations[stage]
            rotated_position = position * cosine + kicked * (sine / omega)
            velocity = kicked * cosine - position * (omega * sine)
            position = rotated_position
            full_position = position + driven_q[index, stage + 1]
            if active:
                after = .5 * np.sum((velocity + driven_v[index, stage + 1])**2 + (full_position * omega)**2, axis=1)
                work += after - before
            residual[1:], increment_energy[1:], strain, jac = force(model, full_position[1:], [1, 2])
            maximum = np.maximum(maximum, strain); minimum = np.minimum(minimum, jac)
            velocity -= .5 * h * residual
        if (index + 1) % subdivisions == 0:
            frame = (index + 1) // subdivisions
            full_velocity = velocity + driven_v[index, -1]
            coordinates[:, frame] = full_position
            traces[:, frame] = np.einsum('lm,mrc->lrc', full_position, model['receiver_fields'], optimize=True)
            kinetic = .5 * np.sum(full_velocity**2, axis=1)
            potential = .5 * np.sum((full_position * omega)**2, axis=1) + increment_energy
            energies[:, frame] = np.stack([kinetic, potential, work], axis=-1)
            if progress and (frame % 100 == 0 or frame == frames - 1):
                progress({'completed_us': frame * sample, 'duration_us': duration})
    internal = energies[:, :, :2].sum(axis=-1)
    scale = np.maximum(np.max(abs(internal), axis=1), np.max(abs(energies[:, :, 2]), axis=1))
    error = np.max(abs(internal - energies[:, :, 2]), axis=1) / np.maximum(scale, 1e-30)
    if not np.isfinite(traces).all() or np.any(error > .01):
        raise ValueError('Reference exceeds original finite-value/1% energy guard')
    return {'times_us': np.arange(frames) * sample, 'traces': traces, 'coordinates': coordinates,
            'energy': energies, 'energy_relative_error': error, 'maximum_strain': maximum,
            'minimum_j': minimum, 'composition_weights': weights, 'forcing_integration': forcing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--step-us', type=float, required=True)
    parser.add_argument('--native-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads((args.model / 'model.json').read_text())
    if configuration['basis_sha256'] != digest(args.model / 'basis.npz'):
        raise ValueError('Changed independent reference basis')
    case = next(row for row in json.loads(args.plan.read_text())['cases'] if row['id'] == args.case)
    scene, original = case['scene'], configuration['scene']
    for key in ['geometry', 'material', 'boundary', 'constitutive', 'receivers']:
        if scene[key] != original[key]:
            raise ValueError('Physical problem differs from the exported native basis')
    for key in ['face', 'radius_mm', 'direction']:
        if scene['source'][key] != original['source'][key]:
            raise ValueError('Reference traction coupling differs from the physical problem')
    with np.load(args.model / 'basis.npz', allow_pickle=False) as archive:
        model = dict(archive)
    folder = args.output / args.case; folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def save(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    identity = {'case': case, 'model_sha256': digest(args.model / 'model.json'),
                'basis_sha256': configuration['basis_sha256'], 'plan_sha256': digest(args.plan),
                'native_source_identity': configuration['reference_identity'],
                'source_sha256': {name: digest(Path(__file__).resolve().parents[1] / name) for name in
                    ['tools/run-comsol-nodal-transient.py', 'tools/comsol_nodal_reference.py',
                     'spectral3d/long_reference_force.py', 'graphwave/modal_forcing.py', 'graphwave/signals.py']},
                'step_us': args.step_us, 'native_comsol_transient': False, 'learned_candidate_arrays_used': False}
    save(folder / 'identity.json', identity)
    def progress(value):
        value.update(elapsed_s=time.monotonic() - started, phase='running')
        save(folder / 'progress.json', value)
        if round(value['completed_us'] / scene['acquisition']['sample_step_us']) % 1000 == 0:
            print(json.dumps(value), flush=True)
    result = evolve(model, scene, args.step_us, progress)
    np.savez_compressed(folder / 'prediction.npz', **{key: result[key] for key in ['times_us', 'traces', 'coordinates', 'energy']})
    report = {'schema': 'workai.comsol_derived_nonlinear_rom.v1', 'case_id': case['id'],
              'completed': True, 'duration_us': float(result['times_us'][-1]), 'frames': len(result['times_us']),
              'prediction_sha256': digest(folder / 'prediction.npz'), 'elapsed_s': time.monotonic() - started,
              'energy_relative_error': result['energy_relative_error'].tolist(),
              'maximum_strain': result['maximum_strain'].tolist(), 'minimum_j': result['minimum_j'].tolist(),
              'time_method': 'order-six symmetric composition around accurately driven native-frequency oscillators',
              'composition_weights': result['composition_weights'].tolist(), 'forcing_integration': result['forcing_integration'],
              'native_comsol_transient': False, 'nonlinear_reference_convergence_verified': False,
              'scientific_accuracy_certified': False}
    if args.native_reference:
        spec = importlib.util.spec_from_file_location('checks', Path(__file__).with_name('validate-comsol-linear-matrix.py'))
        checks = importlib.util.module_from_spec(spec); spec.loader.exec_module(checks)
        config, receipt, native = checks.comparison.load_case(args.native_reference)
        if config['scene'] != scene:
            raise ValueError('Native transient scene differs')
        expected = checks.comparison.parts(native)
        actual = checks.comparison.parts({i: result['traces'][i] for i in range(3)})
        report['native_comparison'] = {name: checks.check(actual[name], expected[name], scene) for name in expected}
        report['native_comparison_job_id'] = receipt['job_id']
    save(folder / 'metrics.json', report)
    save(folder / 'progress.json', {'phase': 'completed', 'elapsed_s': time.monotonic() - started})
    print(json.dumps({'case_id': case['id'], 'completed': True, 'scientific_accuracy_certified': False}), flush=True)


if __name__ == '__main__':
    main()
