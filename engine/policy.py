"""Prefill budget controller embedded in the pinned scheduler by engine.install."""

import bisect as _slo_bisect
import json as _slo_json
import math as _slo_math
import os as _slo_os
import time as _slo_time


def _slo_int(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _slo_positive(value, name):
    if type(value) not in (int, float) or not _slo_math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


class MixedSLOProfile:
    def __init__(self, profile):
        if (type(profile["schema_version"]) is not int or profile["schema_version"] != 1
                or profile["clock"] != "scheduler_result_wall"):
            raise ValueError("Mixed-SLO requires a v1 scheduler_result_wall profile")
        axes = []
        for name in ("decode_bins", "context_bins", "prefill_bins"):
            axis = profile[name]
            if not isinstance(axis, list) or not axis:
                raise ValueError(f"empty {name}")
            for edge in axis:
                _slo_int(edge, name, 0)
            if axis[0] != 0 or axis != sorted(set(axis)):
                raise ValueError(f"unordered or duplicate {name}")
            axes.append(tuple(axis))
        self.decode_bins, self.context_bins, self.prefill_bins = axes
        minimum = _slo_int(profile["min_samples"], "min_samples", 1)
        self.cells = {}
        seen = set()
        for cell in profile["cells"]:
            key = tuple(_slo_int(cell[name], name, 0)
                        for name in ("decode_upper", "context_upper", "prefill_upper"))
            if key in seen or any(k not in a for k, a in zip(key, axes)):
                raise ValueError(f"duplicate or out-of-axis cell {key}")
            seen.add(key)
            count = _slo_int(cell["n"], "n", 1)
            latency = _slo_positive(cell["p95_ms"], "p95_ms")
            if count >= minimum:
                self.cells[key] = latency

    @staticmethod
    def upper(axis, value):
        i = _slo_bisect.bisect_left(axis, value)
        return axis[i] if i < len(axis) else None

    def choose(self, decode, context, room, target, fallback):
        d = self.upper(self.decode_bins, decode)
        c = self.upper(self.context_bins, context)
        decode_only = self.cells.get((d, c, 0))
        if decode_only is not None and decode_only > target:
            return 0, "decode_over_target"
        # The last feasible candidate may land inside the next upper-bound bin.
        candidates = sorted({min(edge, room) for edge in self.prefill_bins if edge > 0})
        known = False
        safe = 0
        for candidate in candidates:
            p = self.upper(self.prefill_bins, candidate)
            latency = self.cells.get((d, c, p))
            if latency is None:
                continue
            known = True
            # Guard against a noisy non-monotone row: do not step past an
            # observed over-target prefix, even if a larger cell looks cheaper.
            if latency > target:
                break
            safe = candidate
        if known:
            return safe, "profile"
        return min(room, fallback), "uncovered"


class MixedSLOStep:
    def __init__(self, started, decode, context, cap, reason):
        self.started = started
        self.decode = decode
        self.context = context
        self.cap = cap
        self.reason = reason
        self.prefill = 0
        self.actual_decode = 0
        self.actual_context = 0
        self.replayed = False
        self.denied = False

    def limit(self, prompt, computed, proposed):
        prompt_part = min(proposed, max(0, prompt - computed))
        if self.cap is None or prompt_part == 0:
            return proposed
        remaining = max(0, self.cap - self.prefill)
        if prompt_part > remaining:
            self.denied = True
            return remaining
        return proposed

    def charge(self, prompt, computed, allocated):
        prompt_part = min(allocated, max(0, prompt - computed))
        self.prefill += prompt_part
        post_prompt = allocated - prompt_part
        if post_prompt:
            if prompt_part or post_prompt != 1:
                self.replayed = True
            self.actual_decode += 1
            self.actual_context += computed


class MixedSLO:
    def __init__(self, mode, profile, target, fallback, max_defer, trace, run_id):
        self.mode = mode
        self.profile = profile
        self.target = _slo_positive(target, "MIXED_SLO_TARGET_MS")
        self.fallback = _slo_int(fallback, "MIXED_SLO_FALLBACK_TOKENS", 1)
        self.max_defer = _slo_int(max_defer, "MIXED_SLO_MAX_DEFER_STEPS", 0)
        self.trace = trace
        self.run_id = run_id
        self.stalled = 0
        self.pending = None
        self.skipped = dict(empty=0, preemption=0, replay=0, decode_mismatch=0)

    @classmethod
    def from_env(cls, scheduler):
        env = _slo_os.environ
        mode = env.get("MIXED_SLO_MODE", "off")
        if mode == "off":
            return None
        if mode not in ("observe", "adaptive"):
            raise ValueError("MIXED_SLO_MODE must be off, observe, or adaptive")
        cfg = scheduler.scheduler_config
        par = scheduler.parallel_config
        model = scheduler.vllm_config.model_config
        if (scheduler.use_pp or scheduler.is_encoder_decoder or scheduler.connector
                or scheduler.vllm_config.speculative_config or scheduler.lora_config
                or model.is_multimodal_model or model.runner_type != "generate"
                or getattr(cfg, "async_scheduling", False) or cfg.policy != "fcfs"
                or not cfg.chunked_prefill_enabled or par.tensor_parallel_size != 1
                or par.data_parallel_size != 1 or scheduler.dcp_world_size != 1):
            raise ValueError("Mixed-SLO supports synchronous single-GPU FCFS text generation "
                             "with chunked prefill and local KV cache only")
        profile = None
        if mode == "adaptive":
            with open(env["MIXED_SLO_PROFILE"], encoding="utf-8") as src:
                profile = MixedSLOProfile(_slo_json.load(src))
        ctl = cls(mode, profile, float(env.get("MIXED_SLO_TARGET_MS", "50")),
                  int(env.get("MIXED_SLO_FALLBACK_TOKENS", "64")),
                  int(env.get("MIXED_SLO_MAX_DEFER_STEPS", "8")), None,
                  env.get("MIXED_SLO_RUN_ID", "").strip())
        path = env.get("MIXED_SLO_TRACE")
        if mode == "observe" and not path:
            raise ValueError("observe mode requires MIXED_SLO_TRACE")
        if path:
            if not ctl.run_id:
                raise ValueError("recording traces requires MIXED_SLO_RUN_ID")
            ctl.trace = open(path, "a", buffering=1, encoding="utf-8")
        return ctl

    def begin(self, running, native_budget):
        started = _slo_time.perf_counter()
        if self.pending is not None:
            raise RuntimeError("Mixed-SLO requires one in-flight engine step")
        ready = [r.num_computed_tokens for r in running
                 if r.num_computed_tokens >= r.num_prompt_tokens
                 and r.num_tokens > r.num_computed_tokens]
        decode, context = len(ready), sum(ready)
        cap, reason = None, "native"
        if self.mode == "adaptive" and decode:
            room = max(0, native_budget - decode)
            cap, reason = self.profile.choose(decode, context, room, self.target, self.fallback)
            if cap == 0 and room > 0 and self.stalled >= self.max_defer:
                cap, reason = min(room, self.fallback), "progress"
        return MixedSLOStep(started, decode, context, cap, reason)

    def scheduled(self, step, output, preempted):
        self.stalled = self.stalled + 1 if step.denied and not step.prefill else 0
        if not self.trace:
            return
        # A preemption/resume or saturated native budget can change the ready
        # set. Do not fit those timings against different pre-step features.
        if not output.total_num_scheduled_tokens:
            exclusion = "empty"
        elif preempted:
            exclusion = "preemption"
        elif step.replayed:
            exclusion = "replay"
        elif step.actual_decode != step.decode or step.actual_context != step.context:
            exclusion = "decode_mismatch"
        else:
            exclusion = None
        if exclusion:
            self.skipped[exclusion] += 1
            return
        step.schedule_ms = (_slo_time.perf_counter() - step.started) * 1000
        self.pending = (output, step)

    def result(self, output):
        if self.pending is None:
            return
        scheduled, step = self.pending
        self.pending = None
        if output is not scheduled:
            raise RuntimeError("Mixed-SLO received an out-of-order batch result")
        wall = (_slo_time.perf_counter() - step.started) * 1000
        self.trace.write(_slo_json.dumps(dict(
            clock="scheduler_result_wall", run_id=self.run_id, decode_count=step.decode,
            context_tokens=step.context, prefill_tokens=step.prefill, step_ms=wall,
            schedule_ms=step.schedule_ms, cap_tokens=step.cap, reason=step.reason,
            excluded_cumulative=self.skipped,
        ), separators=(",", ":")) + "\n")

    def close(self):
        if self.trace is not None:
            self.trace.close()
