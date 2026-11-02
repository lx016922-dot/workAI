"""Run the repository's Python unit tests by group (one list for CI and for local use).

    python tools/run-unit-tests.py --group core|solvers|networks|torch|all [--list]

Groups:
  core      scene/signal definitions, dispersion, COMSOL data contract and collector, modal tools (tests/)
  solvers   tetrahedral, spectral 2-D/3-D, cross-section and sweep/convergence tests (needs gmsh)
  networks  frozen graph network v2 input layer, coarse-graph operators, offline software
  torch     training tests of the frozen network and earlier surrogate models (needs PyTorch)

Every module is run in its own interpreter so a failure or a numba compilation cache in one
module cannot affect another; the exit code is non-zero if any module fails.

graphwave.test_network reads `candidate.pt` from WORKAI_GRAPH_MODEL; when that is unset, the
shipped checkpoint offline/model/best.pt is provided under that name in a temporary directory.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GROUPS = {
    'core': ['physics.test_signals', 'physics.test_scene', 'dispersion.test_lamb', 'dispersion.test_analysis', 'coarsegraph.test_comsol'],
    'solvers': ['solid.test_excitation', 'solid.test_physics', 'spectral2d.test_plane_strain', 'spectral3d.test_curved', 'spectral3d.test_excitation',
                'spectral3d.test_implicit', 'spectral3d.test_physics', 'waveguide.test_safe', 'waveguide.test_field', 'sweep.test_sweep', 'sweep.test_convergence'],
    'networks': ['graphnet.test_graphnet', 'coarsegraph.test_coarse', 'offline.test_long_record', 'offline.test_network_v2', 'offline.test_preview'],
    'torch': ['graphwave.test_network', 'modal3d.test_physics', 'operator3d.test_physics', 'spectralop.test_physics', 'training.test_physics'],
}
GROUPS['core'] += sorted(f'tests.{p.stem}' for p in (REPO / 'tests').glob('test_*.py'))


def discovered():
    """Every test module in the repository outside research/ (to catch modules missing from GROUPS)."""
    found = set()
    for path in REPO.rglob('test_*.py'):
        rel = path.relative_to(REPO)
        if rel.parts[0] in ('research', 'node_modules', 'portable', 'build-cache') or any(p.startswith('.') for p in rel.parts): continue
        found.add('.'.join(rel.with_suffix('').parts))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--group', choices=[*GROUPS, 'all'], default='all'); parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    listed = {m for mods in GROUPS.values() for m in mods}; missing = sorted(discovered() - listed)
    if missing:
        print('Test modules not assigned to any group: ' + ', '.join(missing)); sys.exit(2)
    modules = [m for g, mods in GROUPS.items() if args.group in (g, 'all') for m in mods]
    if args.list:
        print('\n'.join(modules)); return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
    with tempfile.TemporaryDirectory() as scratch:
        if 'WORKAI_GRAPH_MODEL' not in env:
            shutil.copyfile(REPO / 'offline' / 'model' / 'best.pt', Path(scratch) / 'candidate.pt'); env['WORKAI_GRAPH_MODEL'] = scratch
        failures = run(modules, env)
    print(f'{len(modules) - len(failures)}/{len(modules)} modules passed'); sys.exit(1 if failures else 0)


def run(modules, env):
    failures = []
    for module in modules:
        started = time.monotonic(); done = subprocess.run([sys.executable, '-m', 'unittest', module], cwd=REPO, env=env, capture_output=True, text=True)
        status = 'ok' if done.returncode == 0 else 'FAILED'; summary = (done.stderr.strip().splitlines() or [''])[-1]
        print(f'{status:6s} {module:40s} {time.monotonic() - started:7.1f} s  {summary}', flush=True)
        if done.returncode: failures.append(module); print(done.stderr[-4000:], flush=True)
    return failures


if __name__ == '__main__': main()
