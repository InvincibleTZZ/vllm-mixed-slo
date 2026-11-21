"""SLO metrics with a common wall-clock window for every workload class."""
from collections import Counter
import json
import math
from pathlib import Path


def load_results(path):
    rows = [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty results file")
    for field in ("run_id", "trace_sha256", "model", "label", "expected_requests", "schema"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"mixed result field: {field}")
    if rows[0]["schema"] != 1 or len(rows) != rows[0]["expected_requests"]:
        raise ValueError("unsupported schema or incomplete run; expected one outcome per arrival")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate request outcome")
    return rows


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower]) * (position-lower)


def stats(values):
    return {"n": len(values), "p50": percentile(values, .5), "p95": percentile(values,.95),
            "p99": percentile(values,.99), "max": max(values,default=None)}


def summarize(rows, short_ttft_ms, long_ttft_ms, tpot_ms, short_tpot_ms=None, long_tpot_ms=None):
    thresholds = {"short": (short_ttft_ms, tpot_ms if short_tpot_ms is None else short_tpot_ms),
                  "long": (long_ttft_ms, tpot_ms if long_tpot_ms is None else long_tpot_ms)}
    if any(not math.isfinite(x) or x <= 0 for pair in thresholds.values() for x in pair):
        raise ValueError("SLO thresholds must be finite and positive")
    duration = max(r["end_s"] for r in rows)
    if duration <= 0:
        raise ValueError("invalid observation window")
    summary = {k: rows[0][k] for k in ("run_id","trace_sha256","model","label")}
    summary.update(window_s=duration, backend_labels=sorted({r["backend"] for r in rows}),
                   thresholds_ms={k: {"ttft":a,"tpot":b} for k,(a,b) in thresholds.items()},
                   groups={})
    for group in ("all","short","long"):
        members = [r for r in rows if group == "all" or r["kind"] == group]
        statuses = Counter(r["status"] for r in members)
        ttfts, tpots, token_gaps, chunk_gaps, end_to_end, dispatch_lags = [], [], [], [], [], []
        passes = tokens = coalesced = ambiguous_first = 0
        for r in members:
            if r["sent_s"] is not None:
                dispatch_lags.append(1000*(r["sent_s"]-r["scheduled_s"]))
            if r["status"] != "ok":
                continue
            events = r["events"]
            ttft = 1000*(events[0][0]-r["scheduled_s"])
            count = len(r["output_ids"])
            tokens += count
            ttfts.append(ttft)
            end_to_end.append(1000*(r["end_s"]-r["scheduled_s"]))
            gaps = [1000*(b[0]-a[0]) for a,b in zip(events,events[1:])]
            chunk_gaps.extend(gaps)
            if all(n == 1 for _,n in events):
                token_gaps.extend(gaps)
            else:
                coalesced += 1
            # A first multi-token chunk does not reveal when the initial single token was ready.
            # Exclude that request from TPOT and count it as an unverified SLO miss.
            unobservable = events[0][1] > 1
            ambiguous_first += int(unobservable)
            tpot = None
            if count > 1 and not unobservable:
                tpot = 1000*(events[-1][0]-events[0][0])/(count-1)
                tpots.append(tpot)
            ttft_limit, tpot_limit = thresholds[r["kind"]]
            if ttft <= ttft_limit and (count == 1 or (tpot is not None and tpot <= tpot_limit)):
                passes += 1
        summary["groups"][group] = dict(
            offered=len(members), statuses=dict(sorted(statuses.items())), successful=statuses["ok"],
            failed=len(members)-statuses["ok"], slo_pass=passes,
            attainment=passes/len(members) if members else None, goodput_rps=passes/duration,
            offered_rps=len(members)/duration, completed_rps=statuses["ok"]/duration,
            successful_output_tokens_per_s=tokens/duration,
            ttft_ms=stats(ttfts), tpot_ms=stats(tpots), e2e_ms=stats(end_to_end),
            dispatch_lag_ms=stats(dispatch_lags), token_itl_ms=stats(token_gaps),
            sse_token_chunk_gap_ms=stats(chunk_gaps), coalesced_requests=coalesced,
            ambiguous_first_chunks=ambiguous_first)
    return summary


def compare(left, right):
    if left[0]["trace_sha256"] != right[0]["trace_sha256"] or left[0]["model"] != right[0]["model"]:
        raise ValueError("comparison requires identical trace digest and model name")
    a = {r["id"]: r for r in left}
    b = {r["id"]: r for r in right}
    if a.keys() != b.keys():
        raise ValueError("unmatched request IDs")
    paired = [rid for rid in sorted(a) if a[rid]["status"] == b[rid]["status"] == "ok"]
    changed = [rid for rid in paired if a[rid]["output_ids"] != b[rid]["output_ids"]]
    return dict(requests=len(a), paired_success=len(paired), not_paired_success=len(a)-len(paired),
                different_sequences=len(changed), same_sequences=len(paired)-len(changed),
                example_differences=changed[:10],
                left_failures=sum(r["status"]!="ok" for r in left),
                right_failures=sum(r["status"]!="ok" for r in right),
                trace_sha256=left[0]["trace_sha256"])
