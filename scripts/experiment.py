"""Plan or run a serial, single-GPU mixed-SLO experiment matrix."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from bench.metrics import compare, load_results
from bench.trace import generate, load_trace, write_trace
from calibration.__main__ import atomic_write, main as calibration_cli, traces
from calibration.table import evaluate
from scripts.model_identity import model_identity
from engine.install import RELATIVE, inspect_source, render


def make_plan(quick, rate_scale=1.0):
    """All random seeds, per-arm settings and run order are fixed before fitting."""
    if not math.isfinite(rate_scale) or rate_scale <= 0:
        raise ValueError("rate scale must be positive and finite")
    budgets = [128, 256, 512] if quick else [256, 512, 1024]
    work = {}
    runs = []

    def workload(stage, mix, seed, rate):
        key = f"{stage}-{mix}-{seed}"
        work[key] = dict(mix=mix, seed=seed, rate=rate * rate_scale)
        return key

    def add(stage, trace, arm, mode, budget):
        runs.append(dict(name=f"{trace}-{arm}", stage=stage, trace=trace,
                         arm=arm, mode=mode, budget=budget))

    training = [(0, "burst", 24), (2, "mixed", 24)] if quick else [
        (i, mix, rate) for i in range(3) for mix, rate in [("mixed", 12), ("burst", 24)]]
    for i, (budget_index, mix, rate) in enumerate(training):
        trace = workload("fit", mix, 1100 + i * 10, rate)
        add("fit", trace, "observe", "observe", budgets[budget_index])
    for i, (mix, rate) in enumerate([("burst", 24)] if quick else [("mixed", 12), ("burst", 24)]):
        trace = workload("holdout", mix, 2300 + i * 10, rate)
        add("holdout", trace, "observe", "observe", budgets[-1])

    settings = [("off", "off", budgets[-1]), ("observe", "observe", budgets[-1]),
                ("adaptive", "adaptive", budgets[-1]), ("static-small", "off", budgets[0]),
                ("static-mid", "off", budgets[1])]
    cases = [("burst", 24)] if quick else [
        ("short", 12), ("long", 12), ("mixed", 6), ("mixed", 18), ("burst", 24)]
    for i, (mix, rate) in enumerate(cases):
        for repeat in range(1 if quick else 3):
            seed = 3500 + 100 * i + repeat * 10
            trace = workload("eval", mix, seed, rate)
            # Rotate order between paired repetitions to spread temporal drift.
            shift = (i + repeat) % len(settings)
            for arm, mode, budget in settings[shift:] + settings[:shift]:
                add("eval", trace, arm, mode, budget)
    return dict(schema=1, quick=quick, rate_scale=rate_scale,
                workload_shape=dict(n=16 if quick else 96, short_input=64 if quick else 128,
                                    long_input=512 if quick else 2048, short_output=8 if quick else 32,
                                    long_output=16 if quick else 64),
                min_samples=2 if quick else 20, workloads=work, trials=runs)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hashes(model):
    """Verify that the scheduler installed in this interpreter matches this checkout."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("the matrix requires vllm==0.10.2 with the engine patch applied")
    installed = Path(next(iter(spec.submodule_search_locations))) / RELATIVE
    active = installed.read_bytes()
    status, clean = inspect_source(active)
    if status != "patched" or active != render(clean):
        raise ValueError("apply the repository engine patch to this interpreter before running the matrix")
    files = ["scripts/experiment.py", "scripts/trial.py", "bench/__main__.py", "bench/metrics.py",
             "bench/replay.py", "bench/trace.py", "calibration/__main__.py", "calibration/table.py"]
    return dict(scheduler_sha256=hashlib.sha256(active).hexdigest(),
                model_config_sha256=digest(model / "config.json"), model_snapshot=model_identity(model),
                sources_sha256={name:digest(ROOT / name) for name in files})


def trace_inputs(root, plan):
    """Rebuild the expected byte sequence on resume to detect input drift."""
    dest = root / "traces"
    dest.mkdir(exist_ok=True)
    result = {}
    for label, spec in plan["workloads"].items():
        path = dest / (label + ".jsonl")
        rows = generate(**plan["workload_shape"], **spec)
        if path.exists():
            expected = hashlib.sha256()
            for row in rows:
                expected.update((json.dumps(row, separators=(",", ":")) + "\n").encode())
            if digest(path) != expected.hexdigest():
                raise ValueError(f"trace changed: {path}")
        else:
            write_trace(path, rows)
        result[label] = path
    return result


def trial_command(args, run, trace, target, profile):
    cmd = [sys.executable, str(ROOT / "scripts" / "trial.py"), "--model", str(args.model),
           "--gpu", str(args.gpu), "--out", str(target), "--trace", str(trace),
           "--mode", run["mode"], "--budget", str(run["budget"]), "--target-ms", str(args.target_ms),
           "--short-ttft-ms", str(args.short_ttft_ms), "--long-ttft-ms", str(args.long_ttft_ms),
           "--tpot-ms", str(args.tpot_ms)]
    if run["mode"] == "adaptive":
        cmd.extend(["--profile", str(profile)])
    return cmd


def run_trial(root, args, run, trace, profile):
    expected = digest(trace)
    expected_profile = digest(profile) if run["mode"] == "adaptive" else None
    # trial.py derives the run ID from the directory basename. Preserve this
    # across retries while keeping original logs and incomplete raw data.
    attempt = 1
    while True:
        target = root / "trials" / f"attempt-{attempt}" / run["name"]
        manifest = target / "manifest.json"
        if not target.exists():
            print(f"Running {run['name']} (attempt {attempt})", flush=True)
            subprocess.run(trial_command(args, run, trace, target, profile), cwd=ROOT, check=True)
        if manifest.is_file() and read(manifest).get("status") == "completed":
            info = read(manifest)
            if (info["trace_sha256"] != expected or info["policy_profile_sha256"] != expected_profile
                    or info["mode"] != run["mode"] or info["run_id"] != run["name"]):
                raise ValueError(f"completed trial has conflicting provenance: {target}")
            if not all((target / name).is_file() for name in ["requests.jsonl", "summary.json"]):
                raise ValueError(f"completed trial has missing results: {target}")
            if run["mode"] != "off" and not (target / "measured-steps.jsonl").is_file():
                raise ValueError(f"completed trial has no measured steps: {target}")
            return target
        attempt += 1


def step_coverage(profile, stepfile, target_ms):
    report = evaluate(profile, traces([stepfile], 0), target_ms)
    reasons = Counter()
    mixed_reasons = Counter()
    last_excluded = {}
    with stepfile.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                reasons[row["reason"]] += 1
                if row["decode_count"] and row["prefill_tokens"]:
                    mixed_reasons[row["reason"]] += 1
                last_excluded = row.get("excluded_cumulative", {})
    return dict(rows=report["rows"], coverage=report["coverage"], mixed=report["groups"].get("mixed"),
                target_check=report["target_check"], reasons=dict(reasons), mixed_step_reasons=dict(mixed_reasons),
                excluded_cumulative_through_last_record=last_excluded)


def collect_group(root, group, dirs, trace, profile, target_ms):
    runs = {run["arm"]: run for run in group}
    submitted, expected_digest = load_trace(trace)
    expected_ids = {row["id"] for row in submitted}
    base = load_results(dirs[runs["off"]["name"]] / "requests.jsonl")
    entry = {}
    for arm, run in runs.items():
        folder = dirs[run["name"]]
        manifest = read(folder / "manifest.json")
        summary = read(folder / "summary.json")
        rows = load_results(folder / "requests.jsonl")
        if (manifest["trace_sha256"] != expected_digest or summary["trace_sha256"] != expected_digest
                or rows[0]["trace_sha256"] != expected_digest or summary["backend_labels"] != ["vllm"]):
            raise ValueError(f"unexpected trace or backend in {folder}")
        if {r["id"] for r in rows} != expected_ids:
            raise ValueError(f"request IDs differ from offered trace in {folder}")
        groups = {}
        for kind, g in summary["groups"].items():
            groups[kind] = dict(offered=g["offered"], successful=g["successful"], goodput_rps=g["goodput_rps"],
                                attainment=g["attainment"], ttft_p95_ms=g["ttft_ms"]["p95"],
                                tpot_p95_ms=g["tpot_ms"]["p95"], itl_p95_ms=g["token_itl_ms"]["p95"])
        pair = compare(base, rows)
        entry[arm] = dict(trial=folder.relative_to(root).as_posix(), budget=run["budget"], groups=groups,
                          paired_vs_off=pair)
        if run["mode"] != "off":
            entry[arm]["step_evidence"] = step_coverage(profile, folder / "measured-steps.jsonl", target_ms)
    for value in entry.values():
        value["goodput_delta_rps_vs_off"] = (value["groups"]["all"]["goodput_rps"]
                                              - entry["off"]["groups"]["all"]["goodput_rps"])
    return dict(trace_sha256=expected_digest, arms=entry)


def freeze_profile(root, args, plan, dirs):
    train = [dirs[r["name"]] for r in plan["trials"] if r["stage"] == "fit"]
    hold = [dirs[r["name"]] for r in plan["trials"] if r["stage"] == "holdout"]
    env = read(train[0] / "manifest.json")
    metadata = {k: env[k] for k in ("hardware", "vllm", "torch", "model", "model_config_sha256")}
    metadata.update(capture_mode="eager", calibration_runs=[p.name for p in train],
                    workload_shape=plan["workload_shape"], quick=plan["quick"],
                    implementation=plan.get("implementation"))
    atomic_write(root / "environment.json", metadata)
    profile = root / "profile.json"
    rc = calibration_cli(["fit", "--traces", *[str(p / "measured-steps.jsonl") for p in train],
                          "--metadata", str(root / "environment.json"), "--min-samples", str(plan["min_samples"]),
                          "--out", str(profile)])
    if rc:
        raise ValueError("profile fit failed")
    # Even a zero-coverage case should expose the fallback policy's behavior.
    rc = calibration_cli(["evaluate", "--traces", *[str(p / "measured-steps.jsonl") for p in hold],
                          "--profile", str(profile), "--target-ms", str(args.target_ms),
                          "--out", str(root / "validation.json")])
    if rc:
        print("Holdout has no covered step; continuing with a flagged coverage check.", flush=True)
    return read(profile)


def run_matrix(root, args, plan):
    plan_file = root / "plan.json"
    if plan_file.exists() and (not args.resume or read(plan_file) != plan):
        raise ValueError("output has a different or pre-existing plan; use a fresh --out or --resume with matching options")
    atomic_write(plan_file, plan)
    inputs = trace_inputs(root, plan)
    dirs = {}
    signature = None
    profile = None
    evaluations = defaultdict(list)
    trial_success = {}
    for run in plan["trials"]:
        if run["stage"] == "eval" and profile is None:
            profile = freeze_profile(root, args, plan, dirs)
        folder = run_trial(root, args, run, inputs[run["trace"]], root / "profile.json")
        dirs[run["name"]] = folder
        info = read(folder / "manifest.json")
        current = {k: info[k] for k in ("hardware", "model_config_sha256", "model", "torch", "vllm")}
        if signature is not None and signature != current:
            raise ValueError(f"device or software/model configuration differs: {folder}")
        signature = current
        g = read(folder / "summary.json")["groups"]["all"]
        trial_success[run["name"]] = dict(offered=g["offered"], successful=g["successful"])
        if run["stage"] == "eval":
            evaluations[run["trace"]].append(run)
    validation = read(root / "validation.json")
    mixed = validation["groups"].get("mixed", {})
    checks = dict(all_requests_successful=all(g["offered"] == g["successful"] for g in trial_success.values()),
                  holdout_has_mixed_support=mixed.get("covered_rows", 0) > 0,
                  holdout_meets_min_mixed_coverage=(mixed.get("coverage", 0) >= args.min_mixed_coverage),
                  all_paired_outputs_identical=True, adaptive_used_profile_on_mixed_step=False)
    details = {}
    deltas = defaultdict(list)
    for name, runs in evaluations.items():
        spec = plan["workloads"][name]
        case = collect_group(root, runs, dirs, inputs[name], profile, args.target_ms)
        case["workload"] = spec
        details[name] = case
        for arm, result in case["arms"].items():
            pair = result["paired_vs_off"]
            if pair["different_sequences"] or pair["not_paired_success"]:
                checks["all_paired_outputs_identical"] = False
            deltas[(spec["mix"], spec["rate"], arm)].append(result["goodput_delta_rps_vs_off"])
        evidence = case["arms"]["adaptive"]["step_evidence"]
        if evidence["mixed_step_reasons"].get("profile", 0):
            checks["adaptive_used_profile_on_mixed_step"] = True
    variation = [dict(mix=mix, rate=rate, arm=arm, seeds=len(vals), min_rps=min(vals),
                      median_rps=statistics.median(vals), max_rps=max(vals))
                 for (mix, rate, arm), vals in sorted(deltas.items())]
    report = dict(quick=plan["quick"], checks=checks, holdout=dict(coverage=validation["coverage"], mixed=mixed),
                  trials=trial_success, paired_goodput_deltas=variation, evaluations=details)
    atomic_write(root / "aggregate.json", report)
    print(json.dumps(checks, indent=2), flush=True)
    return 0 if all(checks.values()) else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Pinned local model directory")
    p.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quick", action="store_true", help="Eight small trials for integration diagnostics")
    p.add_argument("--dry-run", action="store_true", help="Print the full plan without writing or starting a server")
    p.add_argument("--resume", action="store_true", help="Reuse independently completed trials with the same plan")
    p.add_argument("--rate-scale", type=float, default=1.0)
    p.add_argument("--target-ms", type=float, default=40.0)
    p.add_argument("--short-ttft-ms", type=float, default=1000.0)
    p.add_argument("--long-ttft-ms", type=float, default=3000.0)
    p.add_argument("--tpot-ms", type=float, default=100.0)
    p.add_argument("--min-mixed-coverage", type=float, default=0.0)
    args = p.parse_args(argv)
    if (args.gpu < 0 or not 0 <= args.min_mixed_coverage <= 1 or
            any(not math.isfinite(v) or v <= 0 for v in
                (args.target_ms, args.short_ttft_ms, args.long_ttft_ms, args.tpot_ms, args.rate_scale))):
        p.error("check GPU, coverage, positive finite timing targets and rate")
    plan = make_plan(args.quick, args.rate_scale)
    args.model = args.model.resolve()
    plan["settings"] = dict(model=str(args.model), gpu=args.gpu, target_ms=args.target_ms,
                            short_ttft_ms=args.short_ttft_ms, long_ttft_ms=args.long_ttft_ms,
                            tpot_ms=args.tpot_ms, min_mixed_coverage=args.min_mixed_coverage)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not sys.platform.startswith("linux"):
        p.error("GPU experiments require Linux; use --dry-run for CPU planning")
    if not (args.model / "config.json").is_file():
        p.error("--model must contain config.json")
    plan["implementation"] = implementation_hashes(args.model)
    import fcntl
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("another matrix is running in --out")
        return run_matrix(root, args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
