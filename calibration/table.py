"""A conservative, versioned lookup table for scheduler-result wall time."""

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any

CLOCK = "scheduler_result_wall"
BIN_NAMES = ("decode_bins", "context_bins", "prefill_bins")
CELL_KEYS = ("decode_upper", "context_upper", "prefill_upper")
DEFAULT_BINS = (
    (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    (0, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304),
    (0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
)


def nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def checked_bins(bins):
    if len(bins) != len(BIN_NAMES):
        raise ValueError("expected decode, context and prefill bins")
    result = []
    for name, raw in zip(BIN_NAMES, bins):
        if not isinstance(raw, (tuple, list)) or not raw:
            raise ValueError(f"{name} must be a non-empty sequence")
        seq = tuple(nonnegative_int(item, name) for item in raw)
        if seq[0] != 0 or any(a >= b for a, b in zip(seq, seq[1:])):
            raise ValueError(f"{name} must begin at zero and strictly increase")
        result.append(seq)
    return tuple(result)


def bucket_for(bins, state):
    upper = []
    for seq, value in zip(bins, state):
        idx = bisect_left(seq, value)
        if idx == len(seq):
            return None
        upper.append(seq[idx])
    return tuple(upper)


@dataclass(frozen=True)
class Step:
    run_id: str
    decode_count: int
    context_tokens: int
    prefill_tokens: int
    step_ms: float

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or row.get("clock") != CLOCK:
            raise ValueError(f"expected a step object with clock={CLOCK!r}")
        rid = row.get("run_id")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError("run_id must be a non-empty string")
        d, c, p = (nonnegative_int(row.get(k), k) for k in ("decode_count", "context_tokens", "prefill_tokens"))
        if d == 0 and c != 0:
            raise ValueError("zero decode requests must have zero total decode context")
        return cls(rid, d, c, p, positive_float(row.get("step_ms"), "step_ms"))

    @property
    def state(self):
        return (self.decode_count, self.context_tokens, self.prefill_tokens)


def quantile(values, probability):
    """Empirical nearest rank, including the largest point for p=1."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(probability * len(values)) - 1)]


class Table:
    def __init__(self, profile):
        if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1 or profile.get("clock") != CLOCK:
            raise ValueError("unsupported profile schema or clock")
        self.bins = checked_bins([profile[name] for name in BIN_NAMES])
        self.minimum = nonnegative_int(profile["min_samples"], "min_samples")
        if not self.minimum:
            raise ValueError("min_samples must be positive")
        self.cells = {}
        seen = set()
        for cell in profile["cells"]:
            key = tuple(nonnegative_int(cell[name], name) for name in CELL_KEYS)
            if any(value not in edges for value, edges in zip(key, self.bins)) or key in seen:
                raise ValueError(f"invalid or duplicate cell key {key}")
            seen.add(key)
            n = nonnegative_int(cell["n"], "n")
            if n == 0:
                raise ValueError("cell n must be positive")
            cost = positive_float(cell["p95_ms"], "p95_ms")
            if n >= self.minimum:
                self.cells[key] = cost

    def lookup(self, decode_count, context_tokens, prefill_tokens):
        state = (decode_count, context_tokens, prefill_tokens)
        if any(type(v) is not int or v < 0 for v in state):
            raise ValueError("lookup features must be nonnegative integers")
        return self.cells.get(bucket_for(self.bins, state))


def fit(steps, bins=DEFAULT_BINS, min_samples=20, metadata=None):
    """Estimate empirical per-step q95 in observed cells; omit untrusted cells."""
    bins = checked_bins(bins)
    if nonnegative_int(min_samples, "min_samples") < 1:
        raise ValueError("min_samples must be positive")
    groups = defaultdict(list)
    ranges = {}
    run_ids = set()
    counts = Counter()
    for step in steps:
        counts["rows"] += 1
        run_ids.add(step.run_id)
        key = bucket_for(bins, step.state)
        if key is None:
            counts["out_of_range"] += 1
            continue
        groups[key].append(step.step_ms)
        if key not in ranges:
            ranges[key] = [step.prefill_tokens, step.prefill_tokens]
        else:
            ranges[key][0] = min(ranges[key][0], step.prefill_tokens)
            ranges[key][1] = max(ranges[key][1], step.prefill_tokens)
    if not run_ids:
        raise ValueError("no calibration steps remained after warmup filtering")
    cells = []
    by_previous_budget = defaultdict(float)
    for key, times in sorted(groups.items()):
        if len(times) < min_samples:
            counts["sparse"] += len(times)
            continue
        measured = quantile(times, 0.95)
        # A larger observed budget in the same decode/context box cannot acquire a
        # cheaper bound solely from sampling noise. Keep missing boxes missing.
        bound = max(measured, by_previous_budget[key[:2]])
        by_previous_budget[key[:2]] = bound
        cells.append({
            **dict(zip(CELL_KEYS, key)),
            "n": len(times),
            "p95_ms": bound,
            "raw_p95_ms": measured,
            "observed_prefill_min": ranges[key][0],
            "observed_prefill_max": ranges[key][1],
        })
        counts["retained"] += len(times)
    if not cells:
        raise ValueError("no supported cells; inspect state ranges, bins and sample counts")
    return {
        "schema_version": 1, "clock": CLOCK,
        **{name: list(edges) for name, edges in zip(BIN_NAMES, bins)},
        "min_samples": min_samples, "cells": cells,
        "metadata": metadata or {}, "training_run_ids": sorted(run_ids),
        "stats": {"rows": counts["rows"], "in_supported_cells": counts["retained"],
                  "out_of_range": counts["out_of_range"], "in_sparse_cells": counts["sparse"],
                  "observed_cells": len(groups)},
    }


def evaluate(profile: dict[str, Any], steps, target_ms=None):
    """Score only supported held-out steps; report missing coverage explicitly."""
    table = Table(profile)
    fitted = set(profile["training_run_ids"])
    if target_ms is not None:
        target_ms = positive_float(target_ms, "target_ms")
    by_kind = defaultdict(Counter)
    seen = set()
    errors, bounds, actuals = [], [], []
    for step in steps:
        if step.run_id in fitted:
            raise ValueError(f"train/holdout run_id overlap: {step.run_id!r}")
        seen.add(step.run_id)
        kind = ("mixed" if step.decode_count else "prefill_only") if step.prefill_tokens else (
            "decode_only" if step.decode_count else "other")
        g = by_kind[kind]
        g["total"] += 1
        estimate = table.lookup(*step.state)
        if estimate is None:
            continue
        g["covered"] += 1
        g["above_estimate"] += int(step.step_ms > estimate)
        errors.append(estimate - step.step_ms)
        bounds.append(estimate)
        actuals.append(step.step_ms)

    total = sum(g["total"] for g in by_kind.values())
    if not total:
        raise ValueError("no held-out steps remained after warmup filtering")
    count = len(errors)
    report = {
        "schema_version": 1, "clock": CLOCK, "evaluation_run_ids": sorted(seen),
        "rows": total, "covered_rows": count, "coverage": count / total,
        "groups": {key: {"rows": g["total"], "covered_rows": g["covered"],
                         "coverage": g["covered"] / g["total"],
                         "fraction_above_estimate_on_covered": (g["above_estimate"] / g["covered"]
                                                                if g["covered"] else None)}
                   for key, g in sorted(by_kind.items())},
        "mean_prediction_minus_actual_ms": sum(errors) / count if count else None,
        "mean_absolute_error_ms": sum(abs(e) for e in errors) / count if count else None,
        "p95_absolute_error_ms": quantile([abs(e) for e in errors], .95),
        "fraction_above_estimate_on_covered": sum(e < 0 for e in errors) / count if count else None,
    }
    if target_ms is not None:
        allowed = [actual for bound, actual in zip(bounds, actuals) if bound <= target_ms]
        misses = sum(actual > target_ms for actual in allowed)
        report["target_check"] = {
            "target_ms": target_ms, "predicted_within_target": len(allowed),
            "actual_violations_in_predicted_within": misses,
            "false_safe_fraction": misses / len(allowed) if allowed else None,
        }
    return report
