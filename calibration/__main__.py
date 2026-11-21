"""Fit a scheduler service-time envelope or evaluate it on separate runs."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .table import BIN_NAMES, DEFAULT_BINS, Step, evaluate, fit


def json_object(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as fh:
            name = fh.name
            json.dump(data, fh, ensure_ascii=False, allow_nan=False, indent=2)
            fh.write("\n")
        os.replace(name, path)
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def files_manifest(paths):
    seen, result = set(), []
    for path in paths:
        checksum = hashlib.sha256()
        with path.open("rb") as inp:
            for chunk in iter(lambda: inp.read(1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        if digest in seen:
            raise ValueError(f"duplicate trace content: {path}")
        seen.add(digest)
        result.append({"file": path.name, "sha256": digest})
    return result


def traces(paths, warmup):
    counts = Counter()
    for path in paths:
        with path.open(encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    step = Step.from_dict(json.loads(line))
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f"{path}:{line_no}: {error}") from error
                counts[step.run_id] += 1
                if counts[step.run_id] > warmup:
                    yield step


def edges(text):
    try:
        return [int(s) for s in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("give a comma-separated list of integers") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True, dest="mode")
    trainer = modes.add_parser("fit", help="fit empirical p95 service-time envelopes")
    scorer = modes.add_parser("evaluate", help="evaluate on complete, disjoint runs")
    for subparser in (trainer, scorer):
        subparser.add_argument("--traces", nargs="+", required=True, type=Path)
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--skip-first", type=int, default=0,
                               help="skip this many measured rows per run; default: 0")
    trainer.add_argument("--min-samples", type=int, default=20)
    trainer.add_argument("--metadata", type=Path, help="environment/workload manifest JSON")
    for name, default in zip(BIN_NAMES, DEFAULT_BINS):
        trainer.add_argument("--" + name.replace("_", "-"), type=edges, default=default,
                             help=f"inclusive upper edges, default: {','.join(map(str, default))}")
    scorer.add_argument("--profile", type=Path, required=True)
    scorer.add_argument("--target-ms", type=float)
    scorer.add_argument("--min-coverage", type=float, default=0.0,
                        help="write report, then exit 2 if coverage < threshold")
    args = parser.parse_args(argv)
    try:
        if args.skip_first < 0:
            raise ValueError("--skip-first must be nonnegative")
        inputs = list(args.traces)
        if args.mode == "fit" and args.metadata:
            inputs.append(args.metadata)
        if args.mode == "evaluate":
            inputs.append(args.profile)
        if args.out.resolve() in {p.resolve() for p in inputs}:
            raise ValueError("--out must differ from input files")
        manifest = files_manifest(args.traces)
        rows = traces(args.traces, args.skip_first)
        if args.mode == "fit":
            metadata = json_object(args.metadata) if args.metadata else {}
            result = fit(rows, [getattr(args, k) for k in BIN_NAMES], args.min_samples, metadata)
        else:
            if not 0 <= args.min_coverage <= 1:
                raise ValueError("--min-coverage must be in [0,1]")
            profile = json_object(args.profile)
            if {x["sha256"] for x in manifest} & {x["sha256"] for x in profile.get("source_files", [])}:
                raise ValueError("train/holdout trace content overlaps")
            result = evaluate(profile, rows, args.target_ms)
        result["source_files"], result["skip_first_per_run"] = manifest, args.skip_first
        atomic_write(args.out, result)
        if args.mode == "fit":
            print(f"Wrote {len(result['cells'])} supported cells from {result['stats']['rows']} rows to {args.out}")
        else:
            print(f"Covered {result['covered_rows']}/{result['rows']} held-out rows; report: {args.out}")
            if not result["covered_rows"] or result["coverage"] < args.min_coverage:
                print("Insufficient held-out coverage.", file=sys.stderr)
                return 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"calibration: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
