import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from engine.policy import MixedSLO, MixedSLOProfile, MixedSLOStep


def profile(latencies):
    rows = [dict(decode_upper=2, context_upper=32, prefill_upper=p, p95_ms=ms, n=n)
            for p, ms, n in latencies]
    return MixedSLOProfile(dict(schema_version=1, clock="scheduler_result_wall",
                                decode_bins=[0, 2], context_bins=[0, 32],
                                prefill_bins=[0, 4, 8, 16], min_samples=3, cells=rows))


def req(prompt, computed, total):
    return SimpleNamespace(num_prompt_tokens=prompt, num_computed_tokens=computed,
                           num_tokens=total)


class PolicyTest(unittest.TestCase):
    def test_bins_fallback_and_observed_over_target(self):
        table = profile([(4, 2, 1), (8, 5, 4), (16, 10, 3)])
        self.assertEqual(table.choose(1, 32, 20, 7, 2), (8, "profile"))
        self.assertEqual(table.choose(2, 28, 6, 7, 2), (6, "profile"))
        self.assertEqual(table.choose(2, 33, 5, 7, 2), (2, "uncovered"))
        self.assertEqual(table.choose(2, 25, 3, 7, 2), (2, "uncovered"))
        self.assertEqual(table.choose(2, 25, 16, 3, 2), (0, "profile"))
        self.assertEqual(profile([(0, 9, 4)]).choose(2, 25, 16, 3, 2),
                         (0, "decode_over_target"))

    def test_only_debit_after_commit_and_share_across_requests(self):
        step = MixedSLOStep(0, 1, 24, 8, "profile")
        self.assertEqual(step.limit(20, 0, 12), 8)
        self.assertEqual(step.prefill, 0)  # A failed allocation has no debit.
        self.assertEqual(step.limit(20, 15, 5), 5)
        step.charge(20, 15, 5)
        self.assertEqual(step.limit(40, 4, 20), 3)
        step.charge(40, 4, 3)
        self.assertEqual(step.limit(3, 24, 1), 1)
        step.charge(3, 24, 1)
        self.assertEqual((step.prefill, step.actual_decode, step.actual_context), (8, 1, 24))

    def test_progress_pulse_and_pure_prefill(self):
        ctl = MixedSLO("adaptive", profile([(4, 15, 4)]), 7, 2, 2, None, "")
        running = [req(4, 9, 10), req(80, 2, 80)]
        out = SimpleNamespace(total_num_scheduled_tokens=1)
        for _ in range(2):
            step = ctl.begin(running, 6)
            self.assertEqual(step.cap, 0)
            self.assertEqual(step.limit(80, 2, 4), 0)
            ctl.scheduled(step, out, [])
        step = ctl.begin(running, 6)
        self.assertEqual((step.cap, step.reason), (2, "progress"))
        step.charge(80, 2, 2)
        ctl.scheduled(step, out, [])
        self.assertEqual(ctl.stalled, 0)
        self.assertIsNone(ctl.begin([req(80, 2, 80)], 6).cap)
        self.assertEqual(ctl.begin(running, 1).cap, 0)

    def test_observe_wall_includes_runner_and_trace_agrees_with_fit(self):
        from calibration.table import Step

        trace = io.StringIO()
        ctl = MixedSLO("observe", None, 10, 8, 2, trace, "test")
        out = SimpleNamespace(total_num_scheduled_tokens=6)
        with patch("engine.policy._slo_time.perf_counter", side_effect=[10., 10.001, 10.021]):
            step = ctl.begin([req(4, 9, 10)], 16)
            self.assertEqual(step.limit(7, 2, 5), 5)
            step.charge(7, 2, 5)
            step.charge(4, 9, 1)
            ctl.scheduled(step, out, [])
            ctl.result(out)
        row = json.loads(trace.getvalue())
        self.assertEqual(Step.from_dict(row).state, (1, 9, 5))
        self.assertAlmostEqual(row["step_ms"], 21.)
        self.assertAlmostEqual(row["schedule_ms"], 1.)

    def test_preempted_mismatched_empty_and_replayed_steps_excluded(self):
        ctl = MixedSLO("observe", None, 10, 8, 2, io.StringIO(), "test")
        out = SimpleNamespace(total_num_scheduled_tokens=1)
        for reason in ("preemption", "decode_mismatch", "replay", "empty"):
            step = ctl.begin([req(4, 9, 10)], 16)
            if reason == "replay":
                step.charge(4, 9, 3)
            out.total_num_scheduled_tokens = int(reason != "empty")
            ctl.scheduled(step, out, reason == "preemption")
            ctl.result(out)
        self.assertEqual(set(ctl.skipped.values()), {1})
        self.assertEqual(ctl.trace.getvalue(), "")

    def test_off_does_not_touch_config_or_files(self):
        with patch.dict("os.environ", {"MIXED_SLO_MODE": "off", "MIXED_SLO_PROFILE": ""}):
            self.assertIsNone(MixedSLO.from_env(None))

    def test_runtime_rejects_async_and_non_generation_runners(self):
        cfg = SimpleNamespace(policy="fcfs", chunked_prefill_enabled=True, async_scheduling=True)
        model = SimpleNamespace(is_multimodal_model=False, runner_type="generate")
        scheduler = SimpleNamespace(scheduler_config=cfg,
                                    parallel_config=SimpleNamespace(tensor_parallel_size=1,
                                                                    data_parallel_size=1),
                                    vllm_config=SimpleNamespace(model_config=model,
                                                                speculative_config=None),
                                    use_pp=False, is_encoder_decoder=False, connector=None,
                                    lora_config=None, dcp_world_size=1)
        with patch.dict("os.environ", {"MIXED_SLO_MODE": "observe"}, clear=True):
            with self.assertRaisesRegex(ValueError, "synchronous"):
                MixedSLO.from_env(scheduler)
            cfg.async_scheduling = False
            model.runner_type = "pooling"
            with self.assertRaisesRegex(ValueError, "text generation"):
                MixedSLO.from_env(scheduler)
            model.runner_type = "generate"
            with self.assertRaisesRegex(ValueError, "MIXED_SLO_TRACE"):
                MixedSLO.from_env(scheduler)

    def test_resume_obeys_leading_prompt_dependency(self):
        step = MixedSLOStep(0, 1, 24, 2, "profile")
        self.assertEqual(step.limit(5, 1, 9), 2)
        step.cap = 4
        self.assertEqual(step.limit(5, 1, 9), 9)
        step.charge(5, 1, 9)
        self.assertTrue(step.replayed)

    def test_invalid_profile(self):
        with self.assertRaises(ValueError):
            profile([(4, float("nan"), 4)])
        with self.assertRaises(ValueError):
            profile([(4, 1, 4), (4, 2, 4)])
