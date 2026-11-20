"""The remaining P1 COMSOL runs (tiers 2–4) against the baselines and against the software's predictions.

    python research/comsol-p1-20261024/compare_remaining.py <delivery data dir> --output <report.json>
        [--baselines research/comsol-p1-20261024/serendipity-baselines-20261109/data]
        [--predictions research/comsol-p1-20261024/software-levels-20261120/predictions]

The delivery directory holds CSV files in the delivery format, `<case>_<linear|stvk|murnaghan>[_<dt2|mesh2>].csv`
(the output of `local_kit.py collect`). Whatever is there is analysed; a base run that is not in the delivery is
taken from the baselines. Per case:

* tier 2, when `<case>_stvk.csv` exists — the two nonlinear increments separately, the software's prediction against
  COMSOL: geometric (stvk − linear) and material (murnaghan − stvk), each with the 2f amplitude at the worst receiver
  above the reporting floor, the 2f vector error over all receivers and the waveform error;
* tiers 3–4, for `dt2` and `mesh2` — COMSOL against its own refined run (how far the base run is from converged), and
  the software against the base and against the refined run (does the difference between the software and COMSOL
  shrink when COMSOL is refined). A variant with only its linear run gives the linear part of this.

COMSOL is the reference in every comparison with the software; the refined run is the reference in COMSOL against
itself. Metrics: `coarsegraph.evaluate` (Hann window over 0–30 μs, displacement along each receiver's direction,
receivers below 10 % of the largest are not counted in a worst value), gate 5 %.

The predictions were written by `software_levels.py` on 2026-11-20, before any tier 2–4 run existed; their hashes are
in `software-levels-20261120/levels.json`, and the report records the hashes of the files it read.
"""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
GATE = .05


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def increment(times, f, candidate, reference):
    """A nonlinear increment of the software against COMSOL's: frames × receivers each."""
    from coarsegraph.evaluate import compare, vector_relative
    a2 = compare(times, candidate, reference, f, ('a2',))['a2']
    return {'worst_a2': a2['worst'], 'reference_a2_max_m': a2['max_reference_m'], 'a2_vector_relative': vector_relative(times, candidate, reference, f, 2),
            'l2': float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference)), 'passed': bool(a2['worst'] <= GATE)}


def analyse(delivery, baselines, predictions):
    from coarsegraph.comsol import load_case, parse_name, projected, read_csv
    from coarsegraph.evaluate import case_metrics, compare
    delivery, baselines, predictions = Path(delivery), Path(baselines), Path(predictions); found = {}
    for path in sorted(delivery.glob('*.csv')):
        cid, control, variant = parse_name(path); found.setdefault(cid, {})[(control, variant)] = path
    rows = []; used = {}
    for cid, files in sorted(found.items()):
        case = load_case(cid); f = case['sources'][0]['frequency_mhz']; row = {'case_id': cid, 'delivered': sorted(p.name for p in files.values())}
        def comsol(control, variant='base'):
            path = files.get((control, variant)) or (baselines / f'{cid}_{control}.csv' if variant == 'base' and (baselines / f'{cid}_{control}.csv').exists() else None)
            if path is None: return None
            used[Path(path).name] = digest(path); t, U = read_csv(path, case); return t, projected(case, U)
        def software(control):
            path = predictions / f'{cid}_{control}.csv'
            if not path.exists(): return None
            used['prediction ' + path.name] = digest(path); t, U = read_csv(path, case); return projected(case, U)
        L, S, N = comsol('linear'), comsol('stvk'), comsol('murnaghan'); l, s, n = software('linear'), software('stvk'), software('murnaghan')
        if (('stvk', 'base') in files) and L and S and l is not None and s is not None:
            t = S[0]; tier = {'stvk_record': {'worst_waveform_l2': compare(t, s, S[1], f, ('waveform',))['waveform']['worst']}, 'geometric_increment': increment(t, f, s - l, S[1] - L[1])}
            if N and n is not None:
                tier['material_increment'] = increment(t, f, n - s, N[1] - S[1])
                size = lambda x: float(np.linalg.norm(x)); tier['comsol_shares_l2'] = {'geometric': size(S[1] - L[1]) / size(N[1] - L[1]), 'material': size(N[1] - S[1]) / size(N[1] - L[1])}
            tier['passed'] = bool(all(v.get('passed', True) for v in tier.values() if isinstance(v, dict))); row['tier2_stvk'] = tier
        for variant in ('dt2', 'mesh2'):
            Lv, Nv = comsol('linear', variant), comsol('murnaghan', variant)
            if Lv is None and Nv is None: continue
            out = {}
            if Lv and L:
                t = Lv[0]; out['comsol_linear_base_against_refined'] = {'l2': float(np.linalg.norm(L[1] - Lv[1]) / np.linalg.norm(Lv[1])), **{k: compare(t, L[1], Lv[1], f, (k,))[k]['worst'] for k in ('waveform', 'a1')}}
                if l is not None: out['software_linear'] = {'against_base_l2': float(np.linalg.norm(l - L[1]) / np.linalg.norm(L[1])), 'against_refined_l2': float(np.linalg.norm(l - Lv[1]) / np.linalg.norm(Lv[1]))}
            if Lv and Nv and L and N:
                t = Nv[0]; out['comsol_base_against_refined'] = case_metrics(t, f, N[1], L[1], Nv[1], Nv[1] - Lv[1])
                if l is not None and n is not None:
                    base = case_metrics(t, f, n, l, N[1], N[1] - L[1]); refined = case_metrics(t, f, n, l, Nv[1], Nv[1] - Lv[1]); out['software_against_base'] = base; out['software_against_refined'] = refined
                    out['gap_ratio_worst_increment_a2'] = refined['worst_increment_a2'] / base['worst_increment_a2'] if base['worst_increment_a2'] > 0 else None
                    out['gap_ratio_increment_a2_vector'] = refined['increment_a2_vector_relative'] / base['increment_a2_vector_relative'] if base['increment_a2_vector_relative'] > 0 else None
            row[f'tier3_{variant}'] = out
        rows.append(row)
    return {'schema': 'workai.comsol.p1.remaining_comparison.v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'gate': GATE,
            'reference': 'COMSOL in every comparison with the software; the refined COMSOL run in COMSOL against itself', 'files_sha256': used, 'tool_sha256': digest(__file__), 'rows': rows}


def summary(report):
    lines = []; pct = lambda v: '—' if v is None else f'{v:.2%}'
    for row in report['rows']:
        if 'tier2_stvk' in row:
            t = row['tier2_stvk']; g = t['geometric_increment']; m = t.get('material_increment')
            lines.append(f"{row['case_id']} StVK: geometric increment 2f worst {pct(g['worst_a2'])} (vector {pct(g['a2_vector_relative'])})" + (f", material increment 2f worst {pct(m['worst_a2'])} (vector {pct(m['a2_vector_relative'])})" if m else '') + f" — {'within' if t['passed'] else 'outside'} the {report['gate']:.0%} gate")
        for variant in ('dt2', 'mesh2'):
            t = row.get(f'tier3_{variant}')
            if not t: continue
            if 'comsol_base_against_refined' in t:
                c = t['comsol_base_against_refined']; line = f"{row['case_id']} {variant}: COMSOL base against refined, increment 2f worst {pct(c['worst_increment_a2'])}"
                if 'software_against_refined' in t: line += f"; software against COMSOL base {pct(t['software_against_base']['worst_increment_a2'])} → refined {pct(t['software_against_refined']['worst_increment_a2'])} (ratio {t['gap_ratio_worst_increment_a2']:.2f})"
                lines.append(line)
            elif 'comsol_linear_base_against_refined' in t: lines.append(f"{row['case_id']} {variant} (linear only): COMSOL base against refined, waveform {pct(t['comsol_linear_base_against_refined']['waveform'])}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]); p.add_argument('data', type=Path); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baselines', type=Path, default=HERE / 'serendipity-baselines-20261109' / 'data'); p.add_argument('--predictions', type=Path, default=HERE / 'software-levels-20261120' / 'predictions')
    a = p.parse_args(); report = analyse(a.data, a.baselines, a.predictions)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8'); print('\n'.join(summary(report)) or 'nothing of tiers 2–4 in the delivery')


if __name__ == '__main__': main()
