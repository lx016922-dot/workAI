"""The workbench worker with one rule relaxed: element sizes of 0.05–2.5 mm instead of 0.8–2.5 mm.

Used only for the experiment of cases_s1s2.py N3-2mm (is the lower limit of the element size needed?). The rule is
replaced in this process's copy of offline.domain; no file of the software is changed.

    python relaxed_worker.py --job-dir <job>      # as python -m offline.worker
"""
import argparse, inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import offline.domain as domain

RULE, RELAXED = '.8 <= h <= 2.5', '.05 <= h <= 2.5'
source = inspect.getsource(domain); assert source.count(RULE) == 2
exec(compile(source.replace(RULE, RELAXED), domain.__file__, 'exec'), domain.__dict__)
validate_request = domain.validate_request

if __name__ == '__main__':
    from offline import worker
    parser = argparse.ArgumentParser(); parser.add_argument('--job-dir', required=True, type=Path); worker.run(parser.parse_args().job_dir)
