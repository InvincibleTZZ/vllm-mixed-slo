"""Exercise the actual tagged scheduling loop, with CPU doubles for KV and GPU I/O."""

import ast
import collections
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from engine.install import RELATIVE, inspect_source, operate, render
from engine.policy import MixedSLO, MixedSLOProfile


SOURCE = os.environ.get("MIXED_SLO_UPSTREAM_FIXTURE")


class InstallSafetyTest(unittest.TestCase):
    def test_unrelated_source_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / RELATIVE
            path.parent.mkdir(parents=True)
            path.write_bytes(b"class Scheduler: pass\n")
            for action in ("status", "apply", "revert"):
                with self.assertRaisesRegex(ValueError, "unrecognized"):
                    operate(action, root)
            self.assertEqual(path.read_bytes(), b"class Scheduler: pass\n")


class RequestQueue(collections.deque):
    def peek_request(self):
        return self[0]

    pop_request = collections.deque.popleft
    prepend_request = collections.deque.appendleft


class CPUCache:
    def __init__(self):
        self.blocks = NS(get_block_ids=lambda: ([1],))

    def allocate_slots(self, *args, **kwargs):
        return self.blocks

    def get_computed_blocks(self, req):
        return self.blocks, req.cache_hit

    def get_blocks(self, _):
        return self.blocks

    def get_num_common_prefix_blocks(self, *args):
        return [0]

    def take_events(self):
        return []


def request(name, prompt, total, computed, status, cache_hit=0):
    return NS(request_id=name, num_prompt_tokens=prompt, num_tokens=total,
              num_computed_tokens=computed, num_tokens_with_spec=total,
              num_output_placeholders=0, has_encoder_inputs=False,
              spec_token_ids=[], status=status, cache_hit=cache_hit, num_cached_tokens=-1)


def scheduler_with(schedule_cls, ctl):
    scheduler = schedule_cls()
    scheduler._mixed_slo = ctl
    scheduler.scheduler_config = NS(long_prefill_token_threshold=0, chunked_prefill_enabled=True)
    scheduler.max_num_scheduled_tokens = 16
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_num_running_reqs = 16
    scheduler.max_model_len = 1000
    scheduler.num_lookahead_tokens = 0
    scheduler.is_encoder_decoder = False
    scheduler.log_stats = False
    scheduler.connector = scheduler.lora_config = None
    scheduler.policy = "fcfs"
    scheduler.finished_req_ids = set()
    scheduler.kv_cache_config = NS(kv_cache_groups=[None])
    scheduler.kv_cache_manager = CPUCache()
    scheduler.encoder_cache_manager = NS(get_freed_mm_hashes=lambda: [])
    scheduler._make_cached_request_data = lambda *args: []
    scheduler.get_grammar_bitmask = lambda *args: ({}, None)
    # A 3-token residual prompt precedes a ready decode. The waiting request
    # hits 16 cached tokens and has 14 remaining prompt tokens.
    a = request("partial", 10, 10, 7, "running")
    b = request("decode", 4, 20, 19, "running")
    c = request("cached", 30, 30, 0, "waiting", cache_hit=16)
    scheduler.running = [a, b]
    scheduler.waiting = RequestQueue([c])
    scheduler.requests = {r.request_id: r for r in (a, b, c)}
    return scheduler


def load_real_loop(source):
    # Only the worker, cache and vLLM transport types are stubbed. Execute the
    # original schedule loop and computed-token update from the pinned bytes.
    tree = ast.parse(source)
    scheduler = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    scheduler.bases = []
    scheduler.body = [node for node in scheduler.body if isinstance(node, ast.FunctionDef)
                      and node.name in ("schedule", "_update_after_schedule")]
    tree.body = [scheduler]
    env = dict(time=time, SchedulerOutput=NS, NewRequestData=NS(from_request=lambda *args: None),
               create_request_queue=lambda policy: RequestQueue(), RequestStatus=NS(
                   WAITING="waiting", RUNNING="running", PREEMPTED="preempted",
                   WAITING_FOR_FSM="fsm", WAITING_FOR_REMOTE_KVS="remote"))
    exec("from __future__ import annotations\n" + ast.unparse(tree), env)
    return env["Scheduler"]


@unittest.skipUnless(SOURCE, "set MIXED_SLO_UPSTREAM_FIXTURE to the unmodified v0.10.2 scheduler.py")
class TaggedSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = Path(SOURCE).read_bytes()
        cls.patched = render(cls.clean)

    def test_idempotent_apply_revert_and_detect_drift(self):
        self.assertEqual(inspect_source(self.patched), ("patched", self.clean))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / RELATIVE
            path.parent.mkdir(parents=True)
            path.write_bytes(self.clean)
            for action, expected in (("status", "upstream"), ("apply", "patched"),
                                     ("apply", "patched"), ("status", "patched"),
                                     ("revert", "upstream"), ("revert", "upstream")):
                self.assertEqual(operate(action, root), expected)
            self.assertEqual(path.read_bytes(), self.clean)
            path.write_bytes(self.patched + b"# local edit\n")
            with self.assertRaisesRegex(ValueError, "edited"):
                operate("revert", root)
            self.assertTrue(path.read_bytes().endswith(b"# local edit\n"))

    def test_both_admission_paths_share_one_budget_and_cached_tokens(self):
        native = scheduler_with(load_real_loop(self.clean), None).schedule()
        patched = load_real_loop(self.patched)
        off = scheduler_with(patched, None).schedule()
        self.assertEqual(off.num_scheduled_tokens, native.num_scheduled_tokens)
        observed = MixedSLO("observe", None, 10, 2, 3, io.StringIO(), "native")
        obs = scheduler_with(patched, observed).schedule()
        self.assertEqual(obs.num_scheduled_tokens, native.num_scheduled_tokens)
        observed.result(obs)
        self.assertEqual(json.loads(observed.trace.getvalue())["prefill_tokens"], 15)

        table = MixedSLOProfile(dict(schema_version=1, clock="scheduler_result_wall",
                                     decode_bins=[0, 1], context_bins=[0, 32],
                                     prefill_bins=[0, 6], min_samples=1,
                                     cells=[dict(decode_upper=1, context_upper=32,
                                                 prefill_upper=6, n=4, p95_ms=4)]))
        ctl = MixedSLO("adaptive", table, 10, 2, 3, io.StringIO(), "adaptive")
        limited = scheduler_with(patched, ctl)
        out = limited.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"partial": 3, "decode": 1, "cached": 3})
        self.assertEqual(limited.requests["cached"].num_computed_tokens, 19)
        ctl.result(out)
        row = json.loads(ctl.trace.getvalue())
        self.assertEqual((row["decode_count"], row["context_tokens"], row["prefill_tokens"]),
                         (1, 19, 6))

        ctl.target = 1
        blocked = scheduler_with(patched, ctl)
        out = blocked.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"decode": 1})
        self.assertEqual(blocked.waiting.peek_request().request_id, "cached")
