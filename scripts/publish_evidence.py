"""Export or verify an anonymous, offline-recomputable experiment.py matrix."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from bench.metrics import compare, summarize
from bench.trace import generate
from calibration.table import Step, evaluate, fit
from scripts.experiment import make_plan


SETTINGS = ("target_ms", "short_ttft_ms", "long_ttft_ms", "tpot_ms", "min_mixed_coverage")
STATUSES = ("ok", "timeout", "client_overload", "http_error", "error")
REASONS = ("native", "profile", "progress", "uncovered", "decode_over_target")
EXCLUDED = ("empty", "preemption", "replay", "decode_mismatch")


def read(path):
    return json.loads(path.read_bytes())


def dump(path, obj):
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def json_line(obj):
    return (json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n").encode()


def stream(path, partial_attempt=None):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as src:
        for line in src:
            if line.strip():
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError):
                    if partial_attempt is not None and not line.endswith(b"\n"):
                        partial_attempt["torn_final_record"] = True
                        return
                    raise
                yield value


def compress(path, rows):
    with path.open("xb") as out, gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as zipped:
        for row in rows:
            zipped.write(json_line(row))


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite nonnegative numeric field")
    return value


def enum(value, choices):
    if value not in choices:
        raise ValueError("unknown enumerated value in evidence")
    return value


def safe_digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid SHA-256")
    return value


def canonical_plan(data):
    # Regenerated plan labels are file names and public run IDs. Never use arbitrary source labels.
    quick = data["quick"]
    if type(quick) is not bool:
        raise ValueError("invalid matrix type")
    plan = make_plan(quick, number(data["rate_scale"]))
    if any(data.get(k) != v for k, v in plan.items()):
        raise ValueError("matrix differs from the repository's deterministic design")
    plan["settings"] = {k: number(data["settings"][k]) for k in SETTINGS}
    return plan


def schedules(plan):
    hashes = {}
    expected = {}
    for key, spec in plan["workloads"].items():
        digest = hashlib.sha256()
        signatures = {}
        for row in generate(**plan["workload_shape"], **spec):
            digest.update(json_line(row))
            signatures[row["id"]] = (row["kind"], row["arrival_s"], len(row["prompt"]), row["output_tokens"])
        hashes[key] = digest.hexdigest()
        expected[key] = signatures
    return hashes, expected


def clean_environment(source):
    raw = source.get("hardware", {})
    # Extract a known model token. Drop UUIDs, hostnames, absolute paths, process IDs and arbitrary text.
    chip = re.search(r"\b(?:RTX\s*\d{4}(?:\s*Ti)?|[AHBL]\d{2,3}|V100|P100|T4)\b",
                     str(raw.get("name", "")).upper(), flags=re.I)
    result = {"gpu_family": chip.group().upper() if chip else "unlisted",
              "memory_mib": number(raw["memory_mib"], True) if "memory_mib" in raw else None}
    for key, value in (("driver",raw.get("driver")), ("vllm",source.get("vllm")),
                       ("torch",source.get("torch")), ("python",source.get("python"))):
        match = re.match(r"^\d+(?:\.\d+){1,3}(?:\+cu\d+)?", str(value))
        result[key] = match.group() if match else None
    for key in ("model_config_sha256", "policy_profile_sha256"):
        if source.get(key):
            result[key] = safe_digest(source[key])
    # Preserve performance-relevant arguments without including strings from positional paths.
    argv = source.get("argv", [])
    for flag in ("--tensor-parallel-size","--max-model-len","--max-num-seqs","--max-num-batched-tokens"):
        if flag in argv:
            result[flag[2:]] = number(int(argv[argv.index(flag) + 1]), True)
    if "--dtype" in argv:
        result["dtype"] = enum(argv[argv.index("--dtype") + 1], ("auto", "half", "bfloat16"))
    if "--gpu-memory-utilization" in argv:
        result["gpu_memory_utilization"] = number(float(argv[argv.index("--gpu-memory-utilization") + 1]))
    for flag in ("--enforce-eager", "--enable-chunked-prefill", "--no-enable-prefix-caching"):
        result[flag[2:]] = flag in argv
    return result


def failure_code(message):
    text = str(message).lower()
    for fragment, code in (("warmup", "warmup_failed"), ("startup timed out", "startup_timeout"),
                           ("server exited", "server_exited"), ("all evaluated requests", "all_requests_failed"),
                           ("no evaluation steps", "no_step_data"), ("usage", "usage_mismatch"),
                           ("before [done]", "truncated_stream"), ("finish reason", "early_finish"),
                           ("deadline", "deadline"), ("token_ids", "token_id_protocol")):
        if fragment in text:
            return code
    return "other"


def request_row(raw, run, trace_hash, expected, model):
    rid = raw["id"]
    state = (raw["kind"], raw["scheduled_s"], raw["input_tokens"], raw["target_tokens"])
    if (raw["schema"] != 1 or raw["trace_sha256"] != trace_hash or expected.get(rid) != state
            or raw["expected_requests"] != len(expected)):
        raise ValueError("raw request conflicts with workload")
    out = dict(schema=1, run_id=run, label=run, model=model, trace_sha256=trace_hash,
               expected_requests=len(expected), id=rid, kind=state[0], scheduled_s=state[1],
               input_tokens=state[2], target_tokens=state[3],
               backend=enum(raw["backend"], ("mock", "vllm")), status=enum(raw["status"], STATUSES),
               sent_s=number(raw["sent_s"]) if raw["sent_s"] is not None else None, end_s=number(raw["end_s"]),
               timeout_s=number(raw["timeout_s"]), max_inflight=number(raw["max_inflight"], True),
               events=[[number(t),number(n, True)] for t,n in raw["events"]],
               output_ids=[number(n, True) for n in raw["output_ids"]],
               http_status=number(raw["http_status"], True) if raw.get("http_status") is not None else None,
               finish_reason=enum(raw.get("finish_reason"), (None,"length","stop","abort","error")))
    if raw.get("usage") is not None:
        out["usage"] = {k: number(raw["usage"][k], True) for k in ("prompt_tokens", "completion_tokens")
                        if k in raw["usage"]}
    if out["status"] != "ok":
        out["error"] = failure_code(raw.get("error"))
    return out


def step_row(row, name):
    step = Step.from_dict(row)
    if step.run_id != name:
        raise ValueError("step run ID differs from matrix")
    output = dict(clock="scheduler_result_wall", run_id=name, decode_count=step.decode_count,
                  context_tokens=step.context_tokens, prefill_tokens=step.prefill_tokens, step_ms=step.step_ms,
                  reason=enum(row["reason"], REASONS), schedule_ms=number(row["schedule_ms"]),
                  cap_tokens=number(row["cap_tokens"], True) if row["cap_tokens"] is not None else None)
    output["excluded_cumulative"] = {k: number(v, True) for k,v in row.get("excluded_cumulative", {}).items()
                                     if k in EXCLUDED}
    return output


def derive(bundle, index):
    plan = canonical_plan(index["plan"])
    hashes, signatures = schedules(plan)
    if hashes != index["workload_sha256"]:
        raise ValueError("input schedule reconstruction failed")
    settings = plan["settings"]
    summaries, pairs, completed, stepfiles = {}, {}, {}, {}
    step_counts = {}
    for trial in plan["trials"]:
        name, key = trial["name"], trial["trace"]
        for item in index["attempts"][name]:
            sequence = number(item["attempt"], True)
            relative = f"trials/{name}/attempt-{sequence}"
            folder = bundle / relative
            status = item["status"]
            raw = folder / "requests.jsonl.gz"
            rows = []
            if raw.exists():
                rows = list(stream(raw))
                ids = {r["id"] for r in rows}
                if (len(ids) != len(rows) or ids - signatures[key].keys()
                        or any((r["trace_sha256"], r["run_id"], r["model"]) !=
                               (hashes[key], name, index["model"]) for r in rows)):
                    raise ValueError("raw outcomes disagree with index")
            counts = dict(Counter(row["status"] for row in rows))
            summaries[relative] = {"outcomes": len(rows), "planned": len(signatures[key]), "statuses": counts}
            if len(rows) == len(signatures[key]):
                summaries[relative]["metrics"] = summarize(rows, settings["short_ttft_ms"],
                                                           settings["long_ttft_ms"], settings["tpot_ms"])
            if status != "completed":
                continue
            if len(rows) != len(signatures[key]) or name in completed:
                raise ValueError("completed attempts must be unique and have all outcomes")
            completed[name] = raw
            stepfile = folder / "steps.jsonl.gz"
            if trial["mode"] != "off":
                if not stepfile.is_file():
                    raise ValueError("completed instrumented trial lacks step data")
                stepfiles[name] = stepfile
                counts = Counter()
                mixed = Counter()
                for row in stream(stepfile):
                    counts[row["reason"]] += 1
                    if row["decode_count"] and row["prefill_tokens"]:
                        mixed[row["reason"]] += 1
                step_counts[name] = {"reasons": dict(counts), "mixed_step_reasons": dict(mixed)}
    for trial in plan["trials"]:
        base = trial["trace"] + "-off"
        if trial["stage"] == "eval" and base in completed and trial["name"] in completed:
            pairs[trial["name"]] = compare(list(stream(completed[base])), list(stream(completed[trial["name"]])))
    report = dict(attempts=summaries, paired_vs_off=pairs, step_reasons=step_counts,
                  completed_trials=len(completed), planned_trials=len(plan["trials"]))
    train = [r["name"] for r in plan["trials"] if r["stage"] == "fit"]
    hold = [r["name"] for r in plan["trials"] if r["stage"] == "holdout"]
    if all(t in stepfiles for t in train):
        def steps(names):
            for name in names:
                for row in stream(stepfiles[name]):
                    yield Step.from_dict(row)
        try:
            profile = fit(steps(train), min_samples=plan["min_samples"])
        except ValueError as exc:
            if "no supported cells" not in str(exc) and "no calibration steps" not in str(exc):
                raise
            report["calibration_status"] = "insufficient_supported_steps"
        else:
            report["profile"] = profile
            if all(t in stepfiles for t in hold):
                report["holdout"] = evaluate(profile, steps(hold), settings["target_ms"])
    return report


def export(matrix, destination, model="local-model"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?", model):
        raise ValueError("--model-name must be a public model label, optionally org/name")
    matrix, destination = Path(matrix).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination == matrix or matrix in destination.parents:
        raise ValueError("place public evidence outside the private run tree")
    source_plan = read(matrix / "plan.json")
    plan = canonical_plan(source_plan)
    hashes, signatures = schedules(plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        public = Path(tmp) / "bundle"
        public.mkdir()
        index = dict(schema=1, plan=plan, workload_sha256=hashes, model=model, attempts={})
        implementation = source_plan.get("implementation", {})
        if implementation:
            index["implementation"] = {
                k: safe_digest(implementation[k]) for k in ("scheduler_sha256", "model_config_sha256")
                if k in implementation
            }
            if "model_snapshot" in implementation:
                index["implementation"]["model_snapshot_sha256"] = safe_digest(
                    implementation["model_snapshot"]["sha256"])
            safe_files = ("scripts/experiment.py", "scripts/trial.py", "scripts/model_identity.py",
                          "bench/__main__.py", "bench/metrics.py", "bench/replay.py", "bench/trace.py",
                          "calibration/__main__.py", "calibration/table.py", "engine/policy.py",
                          "engine/install.py")
            index["implementation"]["sources_sha256"] = {
                k: safe_digest(v) for k, v in implementation.get("sources_sha256", {}).items() if k in safe_files
            }
        for key, expected_hash in hashes.items():
            path = matrix / "traces" / (key + ".jsonl")
            if path.exists() and checksum(path) != expected_hash:
                raise ValueError("private input differs from regenerated workload")
        for trial in plan["trials"]:
            name, key = trial["name"], trial["trace"]
            attempts = index["attempts"][name] = []
            sources = [(int(d.name[8:]), d/name) for d in (matrix/"trials").glob("attempt-*")
                       if re.fullmatch("attempt-[1-9][0-9]*", d.name) and (d/name).is_dir()]
            for sequence, folder in sorted(sources):
                manifest_file = folder / "manifest.json"
                meta = read(manifest_file) if manifest_file.exists() else {}
                status = enum(meta.get("status","unrecorded"), ("completed","starting","failed","unrecorded"))
                if meta and (meta["run_id"] != name or meta["mode"] != trial["mode"]
                             or meta["trace_sha256"] != hashes[key]):
                    raise ValueError("manifest conflicts with planned arm or trace")
                if meta and not (matrix / "traces" / (key + ".jsonl")).is_file():
                    raise ValueError("observed trial lacks its input trace")
                item = dict(attempt=sequence, status=status, environment=clean_environment(meta))
                if status == "failed":
                    item["failure"] = failure_code(meta.get("error"))
                attempts.append(item)
                out = public / "trials" / name / f"attempt-{sequence}"
                out.mkdir(parents=True)
                if (folder / "requests.jsonl").exists():
                    compress(out / "requests.jsonl.gz",
                             (request_row(r, name, hashes[key], signatures[key], model)
                              for r in stream(folder / "requests.jsonl", item if status != "completed" else None)))
                if (folder / "measured-steps.jsonl").exists():
                    compress(out / "steps.jsonl.gz",
                             (step_row(r, name) for r in stream(folder / "measured-steps.jsonl")))
        dump(public / "index.json", index)
        report = derive(public, index)
        for trial in plan["trials"]:
            name = trial["name"]
            for item in index["attempts"][name]:
                attempt = f"attempt-{item['attempt']}"
                summary_file = matrix / "trials" / attempt / name / "summary.json"
                result = report["attempts"][f"trials/{name}/{attempt}"].get("metrics")
                if summary_file.exists():
                    original = read(summary_file)
                    if not result or any(result[k] != original[k]
                                         for k in ("groups", "window_s", "thresholds_ms", "backend_labels")):
                        raise ValueError("public metrics differ from the private trial summary")
        if (matrix / "profile.json").is_file():
            original = read(matrix / "profile.json")
            rebuilt = report.get("profile", {})
            if not rebuilt or any(rebuilt.get(k) != original[k] for k in
                                  ("cells","decode_bins","context_bins","prefill_bins","training_run_ids","min_samples")):
                raise ValueError("public refit differs from the actual matrix profile")
        dump(public / "report.json", report)
        files = {f.relative_to(public).as_posix(): checksum(f) for f in sorted(public.rglob("*")) if f.is_file()}
        dump(public / "sha256.json", files)
        public.rename(destination)
    return report


def verify(bundle):
    bundle = Path(bundle)
    actual = {f.relative_to(bundle).as_posix(): checksum(f) for f in sorted(bundle.rglob("*"))
              if f.is_file() and f != bundle / "sha256.json"}
    if actual != read(bundle / "sha256.json"):
        raise ValueError("public file set or checksum changed")
    report = derive(bundle, read(bundle / "index.json"))
    if report != read(bundle / "report.json"):
        raise ValueError("published summary differs from recomputation")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="mode", required=True)
    ex = cmds.add_parser("export")
    ex.add_argument("--matrix", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--model-name", default="local-model", help="optional public model label (org/name)")
    ve = cmds.add_parser("verify")
    ve.add_argument("--bundle", required=True, type=Path)
    a = parser.parse_args()
    result = export(a.matrix, a.out, a.model_name) if a.mode == "export" else verify(a.bundle)
    print(f"{result['completed_trials']}/{result['planned_trials']} planned trials complete")


if __name__ == "__main__":
    main()
