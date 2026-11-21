import contextlib
import io
import json
from pathlib import Path
import unittest
from uuid import uuid4

from calibration.__main__ import main
from calibration.table import CLOCK, DEFAULT_BINS, Step, Table, evaluate, fit
from engine.policy import MixedSLOProfile


FIXTURES = Path(__file__).resolve().parents[1] / "calibration" / "fixtures"


def step(d=2, c=512, p=32, ms=4., run="train"):
    return Step.from_dict({"decode_count": d, "context_tokens": c, "prefill_tokens": p,
                           "step_ms": ms, "run_id": run, "clock": CLOCK})


class CalibrationTests(unittest.TestCase):
    def test_inclusive_edges_sparse_unknown_and_monotonicity(self):
        profile = fit([step(ms=5), step(ms=7), step(p=64, ms=2), step(p=64, ms=3),
                       step(p=128, ms=20), step(d=800, c=2000)], min_samples=2)
        table = Table(profile)
        self.assertEqual(table.lookup(2, 512, 32), 7)
        self.assertEqual(table.lookup(2, 512, 33), 7)
        self.assertEqual(profile["cells"][1]["raw_p95_ms"], 3)
        self.assertIsNone(table.lookup(2, 512, 128))
        self.assertIsNone(table.lookup(3, 512, 32))
        self.assertIsNone(table.lookup(2, 2**30, 32))
        self.assertEqual(profile["stats"]["out_of_range"], 1)
        self.assertEqual(profile["stats"]["in_sparse_cells"], 1)

    def test_runtime_consumes_fitted_table_and_applies_fallback(self):
        profile = fit([step(p=0, ms=2), step(p=32, ms=4), step(p=64, ms=8)], min_samples=1)
        runtime = MixedSLOProfile(profile)
        self.assertEqual(runtime.choose(2, 512, 64, 5, 16), (32, "profile"))
        self.assertEqual(runtime.choose(2, 512, 28, 5, 16), (28, "profile"))
        self.assertEqual(runtime.choose(2, 512, 64, 1, 16), (0, "decode_over_target"))
        self.assertEqual(runtime.choose(99, 512, 64, 5, 16), (16, "uncovered"))

    def test_bucket_bound_not_enforced_by_latency_target(self):
        profile = fit([step(ms=4)] * 20)
        report = evaluate(profile, [step(ms=3, run="eval"), step(ms=7, run="eval"),
                                    step(d=8, c=580, run="eval")], target_ms=5)
        self.assertEqual(report["coverage"], 2/3)
        self.assertEqual(report["target_check"]["false_safe_fraction"], .5)
        self.assertEqual(report["p95_absolute_error_ms"], 3)
        self.assertEqual(report["groups"]["mixed"]["rows"], 3)
        self.assertEqual(report["fraction_above_estimate_on_covered"], .5)

    def test_input_schema_validation_and_leakage(self):
        profile = fit([step()] * 20)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate(profile, [step()])
        for kwargs in [dict(d=True), dict(d=0, c=60), dict(ms=float("nan")), dict(ms=0), dict(run="")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                step(**kwargs)
        with self.assertRaises(ValueError):
            Step.from_dict({"clock": "cuda_event"})
        with self.assertRaisesRegex(ValueError, "no supported cells"):
            fit([step()], min_samples=2)
        with self.assertRaises(ValueError):
            fit([step()], bins=([1, 2], *DEFAULT_BINS[1:]), min_samples=1)
        with self.assertRaises(ValueError):
            Table({**profile, "clock": "cuda_event"})
        cell = dict(profile["cells"][0], n=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Table({**profile, "cells": [cell, cell]})

    def test_cli_reproducible_fit_holdout_report_and_invalid_input(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            test_id = uuid4().hex
            output = Path.cwd() / f".calibration-{test_id}-table.json"
            report_path = Path.cwd() / f".calibration-{test_id}-validation.json"
            self.addCleanup(output.unlink, missing_ok=True)
            self.addCleanup(report_path.unlink, missing_ok=True)
            train = FIXTURES / "SYNTHETIC_fit.jsonl"
            test = FIXTURES / "SYNTHETIC_holdout.jsonl"
            params = ["fit", "--traces", str(train), "--out", str(output), "--min-samples", "2",
                      "--metadata", str(FIXTURES / "SYNTHETIC_environment.json")]
            self.assertEqual(main(params), 0)
            first = output.read_bytes()
            self.assertEqual(main(params), 0)
            self.assertEqual(output.read_bytes(), first)
            validate = ["evaluate", "--traces", str(test), "--profile", str(output), "--out",
                        str(report_path), "--target-ms", "11"]
            self.assertEqual(main(validate), 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["coverage"], .6)
            self.assertEqual(report["target_check"]["actual_violations_in_predicted_within"], 1)
            self.assertEqual(main(validate + ["--min-coverage", ".7"]), 2)
            for invalid in [
                    ["evaluate", "--traces", str(train), "--profile", str(output), "--out", str(report_path)],
                    params + ["--skip-first", "100"], params + ["--out", str(train)]]:
                with self.subTest(invalid=invalid), self.assertRaises(SystemExit) as raised:
                    main(invalid)
                self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
