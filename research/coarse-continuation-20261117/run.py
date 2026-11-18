#!/usr/bin/env python3
"""Prepare and run the predeclared short-record coarse-operator continuation pilot.

Only this experiment's output directory is written. Fine references are actual
spectral3d solves in the existing sweep layout, not the trained model's outputs.
The three test pairs are never scored by prepare/labels/converge: the training
entry point evaluates them only after freezing the validation-selected model.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Experiment definition changed; use a new output directory: {path}")
    else:
        atomic(path, value)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def validate_config(config):
    if config.get("schema") != "workai.coarsegraph.continuation-pilot.v1":
        raise ValueError("Unexpected pilot schema")
    original = read(REPO / config["original_splits"])
    if config["reference"]["engine"] != "3d_spectral":
        raise ValueError("Pilot references must be actual spectral3d solves")
    if config["gate"]["relative_tolerance"] != 0.05 or config["gate"]["relative_reporting_floor"] != 0.1:
        raise ValueError("Keep the existing 5% gate and 10% relative reporting floor")
    seen = set()
    identities = set()
    for pair in config["pairs"]:
        if pair["id"] in seen:
            raise ValueError(f"Duplicate pair id: {pair['id']}")
        seen.add(pair["id"])
        identity = (pair["campaign"], json.dumps(pair["parameters"], sort_keys=True))
        if identity in identities:
            raise ValueError(f"Duplicate physical pair: {pair['id']}")
        identities.add(identity)
        rule = original[pair["campaign"]]
        matches = [name for name in ("train", "validation", "test")
                   if any(all(pair["parameters"].get(key) == value for key, value in row.items())
                          for row in rule.get(name, []))]
        if matches != [pair["split"]]:
            raise ValueError(f"Original split identity changed for {pair['id']}: {matches}")
    for split in ("train", "validation", "test"):
        group = [p for p in config["pairs"] if p["split"] == split]
        if len(group) != 3 or {p["campaign"] for p in group} != {
                "labels-a-top-gaussian", "labels-b-end-axial", "labels-c-two-patch"}:
            raise ValueError("This predeclared pilot requires one A, B and C pair in each split")
    for pair_id in config["reference_check"]["pair_ids"]:
        if next(p for p in config["pairs"] if p["id"] == pair_id)["split"] != "train":
            raise ValueError("Reference-check choices must be predeclared training pairs")
    if config["test_used_for_selection"] or config["accepted_for_solver_routing"] or config["accepted_for_offline_release"]:
        raise ValueError("This pilot cannot use test selection or promote a release model")


def prepare(config_path, output):
    from sweep.spec import plan, validate_spec

    config = read(config_path)
    validate_config(config)
    sources = [config["initial_model"], config["original_splits"]]
    campaigns = sorted({p["campaign"] for p in config["pairs"]})
    sources += [f"coarsegraph/specs/{name}.json" for name in campaigns]
    protocol = {
        "schema": "workai.coarsegraph.continuation-protocol.v1",
        "configuration": config,
        "configuration_sha256": sha256(config_path),
        "source_sha256": {name: sha256(REPO / name) for name in sources},
        "reference_identity": "Actual spectral3d.run solutions; regenerated labels, not recovered historical files.",
        "scope": "Pilot: 3 train, 3 validation, 3 test pairs; no D comb cases or long-record certification.",
    }
    immutable(output / "protocol.json", protocol)
    split_rules = {}
    pair_records = []
    for name in campaigns:
        selected = [p for p in config["pairs"] if p["campaign"] == name]
        spec = read(REPO / f"coarsegraph/specs/{name}.json")
        spec.pop("grid", None)
        spec["cases"] = [dict(p["parameters"], constitutive=law)
                         for p in selected for law in ("linear", "murnaghan")]
        spec["name"] += " · 20261117 pilot"
        spec["engine"] = config["reference"]["engine"]
        spec["degree"] = config["reference"]["degree"]
        spec["base_scene"]["numerics"]["h_mm"] = config["reference"]["h_mm"]
        spec["base_scene"]["numerics"]["cfl"] = config["reference"]["cfl"]
        spec["base_scene"]["acquisition"] = {
            "duration_us": config["reference"]["duration_us"],
            "sample_step_us": config["reference"]["sample_step_us"],
        }
        spec["outputs"] = {"fields": False, "csv": False, "browser_record": False}
        spec["budget"] = {"workers": 2, "seconds_per_case": config["reference"]["seconds_per_case"]}
        spec["convergence"] = {
            "levels": ["space"],
            "space_factor": config["reference_check"]["space_factor"],
            "tolerance": {"waveform_relative_l2": config["reference_check"]["relative_tolerance"],
                          "harmonic_relative": config["reference_check"]["relative_tolerance"]},
        }
        spec = validate_spec(spec)
        immutable(output / "specs" / f"{name}.json", spec)
        rows = plan(spec)
        split_rules[name] = {s: [p["parameters"] for p in selected if p["split"] == s]
                             for s in ("train", "validation", "test")}
        for pair in selected:
            cases = {row["parameters"]["constitutive"]: row["case_id"]
                     for row in rows if all(row["parameters"].get(k) == v for k, v in pair["parameters"].items())}
            if set(cases) != {"linear", "murnaghan"}:
                raise ValueError(f"Missing control pair: {pair['id']}")
            pair_records.append(dict(pair, case_ids=cases))
    immutable(output / "splits.json", split_rules)
    immutable(output / "pairs.json", pair_records)
    return config, pair_records


def campaign(output, name, pairs):
    from sweep.runner import Campaign
    from sweep.spec import load_spec

    spec = load_spec(output / "specs" / f"{name}.json")
    directory = output / "labels" / name
    run = Campaign.create(spec, directory, resume=directory.exists())
    by_case = {case_id: pair for pair in pairs if pair["campaign"] == name
               for case_id in pair["case_ids"].values()}
    # sweep's default split is "train"; this experiment preserves the explicit
    # original identities instead of generating a new random split.
    for row in run.manifest["cases"]:
        row["split"] = by_case[row["case_id"]]["split"]
        row["pilot_pair_id"] = by_case[row["case_id"]]["id"]
    run.manifest["split_policy"] = "Explicit identities preserved from coarsegraph/specs/splits.json"
    run.save()
    return spec, run


def chosen_pairs(pairs, selected):
    if not selected:
        return pairs
    unknown = set(selected) - {p["id"] for p in pairs}
    if unknown:
        raise ValueError(f"Unknown pair ids: {sorted(unknown)}")
    return [p for p in pairs if p["id"] in selected]


def generate_labels(output, pairs, selected, workers):
    selected = chosen_pairs(pairs, selected)
    results = []
    for name in sorted({p["campaign"] for p in selected}):
        spec, run = campaign(output, name, pairs)
        pending = [cid for pair in selected if pair["campaign"] == name
                   for cid in pair["case_ids"].values() if not run.completed(cid)]

        def solve(cid):
            phase = run.run_case(cid, spec)
            emit({"campaign": name, "case_id": cid, "phase": phase,
                  "elapsed_s": run.entry(cid).get("elapsed_s")})
            return phase

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results.extend(pool.map(solve, pending))
        run.manifest["summary"] = {phase: sum(c.get("phase") == phase for c in run.manifest["cases"])
                                   for phase in ("planned", "completed", "failed")}
        run.save()
    if any(phase != "completed" for phase in results):
        raise RuntimeError("A fine-reference solve failed; inspect its run.log before continuing")


def reference_hashes(output, pairs, refined=False):
    hashes = {}
    for pair in pairs:
        for law, cid in pair["case_ids"].items():
            case = output / "labels" / pair["campaign"] / "cases" / cid
            if refined:
                case = case / "convergence/space"
            for name in ("scene.json", "out/result.json", "out/lines.npz"):
                path = case / name
                hashes[str(path.relative_to(output))] = sha256(path)
    return hashes


def check_references(config, output, pairs, workers):
    import numpy as np
    from coarsegraph.evaluate import compare
    from coarsegraph.simulate import fine_projected
    from physics.receivers import receiver_points

    selected = chosen_pairs(pairs, config["reference_check"]["pair_ids"])
    for name in sorted({p["campaign"] for p in selected}):
        spec, run = campaign(output, name, pairs)
        wanted = [p for p in selected if p["campaign"] == name]
        for pair in wanted:
            if not all(run.completed(cid) for cid in pair["case_ids"].values()):
                raise ValueError(f"Generate both controls before refinement: {pair['id']}")
        run.converge(spec, [p["case_ids"]["murnaghan"] for p in wanted], workers=workers, progress=emit)

    rows = []
    floor = config["reference_check"]["relative_reporting_floor"]
    tolerance = config["reference_check"]["relative_tolerance"]
    for pair in selected:
        paths = {law: output / "labels" / pair["campaign"] / "cases" / cid
                 for law, cid in pair["case_ids"].items()}
        scene = read(paths["murnaghan"] / "scene.json")
        names = [rx["name"] for rx in receiver_points(scene)]
        arrays = {}
        times = None
        for law, case in paths.items():
            refinement = read(case / "convergence/report.json")
            if any(level.get("error") or level.get("skipped") for level in refinement["levels"]):
                raise ValueError(f"Space refinement was not valid for {pair['id']} / {law}")
            for level, suffix in (("base", "out"), ("space", "convergence/space/out")):
                values, t = fine_projected(case / suffix, names)
                if not np.all(np.isfinite(values)):
                    raise ValueError("Nonfinite reference output")
                if times is None:
                    times = t
                if not np.array_equal(times, t):
                    raise ValueError("Reference samples changed during mesh refinement")
                arrays[(law, level)] = values
        f = scene["source"]["frequency_mhz"]
        linear = compare(times, arrays[("linear", "base")], arrays[("linear", "space")], f,
                         floor=floor, tolerance=tolerance)
        nonlinear = compare(times, arrays[("murnaghan", "base")], arrays[("murnaghan", "space")], f,
                            floor=floor, tolerance=tolerance)
        increment = compare(times, arrays[("murnaghan", "base")] - arrays[("linear", "base")],
                            arrays[("murnaghan", "space")] - arrays[("linear", "space")], f,
                            terms=("a2",), floor=floor, tolerance=tolerance)
        rows.append({"pair_id": pair["id"], "linear": linear, "murnaghan": nonlinear,
                     "increment": increment, "passed": linear["pass"] and nonlinear["pass"] and increment["pass"]})
    result = {"schema": "workai.coarsegraph.pilot-reference-check.v1", "rows": rows,
              "passed": all(row["passed"] for row in rows),
              "reference_sha256": reference_hashes(output, selected),
              "refined_reference_sha256": reference_hashes(output, selected, refined=True),
              "scope": config["reference_check"]["scope"],
              "note": "Relative 10% reporting floor, including the nonlinear increment; no absolute weak-signal waiver.",
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic(output / "reference-check.json", result)
    emit(result)
    if not result["passed"]:
        raise RuntimeError("Reference refinement did not pass the predeclared pilot check")


def training_command(config, output, pairs, workers, dry_run=False):
    # Check every label before launching the optimizer; a partial campaign must
    # not silently change the predeclared training/validation/test experiment.
    for name in sorted({p["campaign"] for p in pairs}):
        _, run = campaign(output, name, pairs)
        if not all(run.completed(cid) for p in pairs if p["campaign"] == name for cid in p["case_ids"].values()):
            raise ValueError(f"Incomplete label campaign: {name}")
    check = read(output / "reference-check.json")
    selected = chosen_pairs(pairs, config["reference_check"]["pair_ids"])
    if (not check["passed"] or check["reference_sha256"] != reference_hashes(output, selected)
            or check["refined_reference_sha256"] != reference_hashes(output, selected, refined=True)):
        raise ValueError("Missing, failed, or stale pilot reference check")
    cfg = config["training"]
    command = [sys.executable, "-m", "coarsegraph.train", "--labels"]
    command += [str(output / "labels" / name) for name in sorted({p["campaign"] for p in pairs})]
    command += ["--splits", str(output / "splits.json"), "--init", str(REPO / config["initial_model"]),
                "--output", str(output / "candidate"), "--workers", str(workers)]
    for key in ("degree", "h_mm", "cfl", "axes", "increment_weight", "harmonic_weight", "iterations", "stall"):
        command += ["--" + key.replace("_", "-"), str(cfg[key])]
    emit({"command": command, "cwd": str(REPO), "dry_run": dry_run})
    if dry_run:
        return
    if (output / "candidate").exists():
        raise FileExistsError("Candidate output already exists; use the trainer's explicit checkpoint resume, not a second test-selection run")
    atomic(output / "training-plan.json", {"command": command, "label_sha256": reference_hashes(output, pairs),
                                           "test_used_for_selection": False,
                                           "created_at_utc": datetime.now(timezone.utc).isoformat()})
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUTF8="1")
    subprocess.run(command, cwd=REPO, env=env, check=True)


def archive_labels(config, output, pairs):
    """Archive only reproducible, compact sweep inputs/results and refinement evidence."""
    # Fail rather than publish an incomplete archive under the full-pilot name.
    reference_hashes(output, pairs)
    selected = chosen_pairs(pairs, config["reference_check"]["pair_ids"])
    reference_hashes(output, selected, refined=True)
    files = [output / name for name in ("protocol.json", "pairs.json", "splits.json", "reference-check.json")]
    for name in sorted({p["campaign"] for p in pairs}):
        directory = output / "labels" / name
        manifest = read(directory / "manifest.json")
        for case in manifest["cases"]:
            result = directory / "cases" / case["case_id"] / "out/result.json"
            if case.get("phase") != "completed" or case.get("result_sha256") != sha256(result):
                raise ValueError(f"Cannot archive an incomplete or changed reference: {name} / {case['case_id']}")
        files += [output / "specs" / f"{name}.json"]
        files += [output / "labels" / name / filename for filename in ("manifest.json", "spec.json")]
    for pair in pairs:
        for cid in pair["case_ids"].values():
            case = output / "labels" / pair["campaign"] / "cases" / cid
            files += [case / name for name in ("scene.json", "settings.json", "out/result.json", "out/lines.npz")]
            if pair["id"] in config["reference_check"]["pair_ids"]:
                files += [case / "convergence" / name for name in ("report.json",)]
                files += [case / "convergence/space" / name
                          for name in ("scene.json", "settings.json", "out/result.json", "out/lines.npz")]
                if (case / "convergence/increment.json").is_file():
                    files.append(case / "convergence/increment.json")
    files = sorted(set(files))
    index = {"schema": "workai.coarsegraph.pilot-label-archive.v1",
             "scope": "Regenerated spectral reference labels and predeclared refinement evidence; no solver caches, logs or full fields.",
             "files": {str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
                       for path in files}}
    index_path = output / "archive-index.json"
    atomic(index_path, index)
    files.append(index_path)
    destination = output / "labels-evidence.tar.gz"
    temporary = destination.with_suffix(".gz.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(files):
                    data = path.read_bytes()
                    info = tarfile.TarInfo(str(path.relative_to(output)))
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))
    temporary.replace(destination)
    checksum = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{checksum}  {destination.name}\n", encoding="utf-8")
    emit({"archive": str(destination), "sha256": checksum, "bytes": destination.stat().st_size,
          "files": len(files), "uncompressed_bytes": sum(path.stat().st_size for path in files)})


def report(config, output):
    from coarsegraph.train import summarize

    model = read(output / "candidate/model.json")
    evaluation = read(output / "candidate/evaluation.json")
    if "evaluation" in evaluation:
        evaluation = evaluation["evaluation"]
    if model.get("training", {}).get("test_used_for_selection") is not False:
        raise ValueError("Candidate does not record test isolation")
    summary = summarize(evaluation)
    if any(block[split]["pairs"] != 3 for block in summary.values()
           for split in ("train", "validation", "test")):
        raise ValueError("Evaluation does not cover the predeclared 3/3/3 pilot")
    result = {
        "schema": "workai.coarsegraph.continuation-pilot-report.v1",
        "pilot": True,
        "protocol_sha256": sha256(output / "protocol.json"),
        "candidate_sha256": sha256(output / "candidate/model.json"),
        "initial_model": config["initial_model"],
        "selected_iteration": model["selected_iteration"],
        "validation_loss": model["validation_loss"],
        "summary": summary,
        "constraints": model["constraints"],
        "reference_check": read(output / "reference-check.json"),
        "test_used_for_selection": False,
        "accepted_for_solver_routing": False,
        "accepted_for_offline_release": False,
        "limitations": [
            "Only 3 training, 3 validation and 3 predeclared test pairs; not a rerun of the original 13/5/16 campaign.",
            "Spatial reference checks cover only the predeclared two training pairs; other pairs and temporal convergence are not certified here.",
            "Only 30 microsecond aluminum plate records; no 5 millisecond, defect, comb-load or general-material accuracy claim.",
            "Neither software defaults nor the shipped model are changed by this experiment.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic(output / "pilot-report.json", result)
    emit(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "labels", "converge", "train", "report", "archive"))
    parser.add_argument("--config", type=Path, default=HERE / "pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pairs", nargs="+", help="labels only: e.g. A-train, for the initial cost measurement")
    parser.add_argument("--dry-run", action="store_true", help="train only: validate inputs and print the command")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.pairs and args.action != "labels":
        parser.error("--pairs is only valid for labels")
    if args.dry_run and args.action != "train":
        parser.error("--dry-run is only valid for train")
    output = args.output.resolve()
    config, pairs = prepare(args.config.resolve(), output)
    if args.action == "prepare":
        emit({"output": str(output), "pairs": len(pairs), "fine_solves": 2 * len(pairs),
              "splits": {s: [p["id"] for p in pairs if p["split"] == s] for s in ("train", "validation", "test")}})
    elif args.action == "labels":
        generate_labels(output, pairs, args.pairs, args.workers)
    elif args.action == "converge":
        check_references(config, output, pairs, args.workers)
    elif args.action == "train":
        training_command(config, output, pairs, args.workers, args.dry_run)
    elif args.action == "archive":
        archive_labels(config, output, pairs)
    else:
        report(config, output)


if __name__ == "__main__":
    main()
