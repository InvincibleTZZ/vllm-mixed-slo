"""Matrix control flow and CPU-only synthetic checks; no GPU measurements."""

from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from bench.metrics import summarize
from bench.trace import load_trace
from scripts.experiment import main, make_plan, run_matrix, trial_command


class ExperimentTests(unittest.TestCase):
    def test_disjoint_seeds_paired_workloads_and_counterbalanced_order(self):
        for quick, total in [(True, 8), (False, 83)]:
            plan = make_plan(quick)
            self.assertEqual(len(plan["trials"]), total)
            self.assertEqual(len(set(r["name"] for r in plan["trials"])), total)
            seeds = [spec["seed"] for spec in plan["workloads"].values()]
            self.assertEqual(len(seeds), len(set(seeds)))
            evals = defaultdict(list)
            for row in plan["trials"]:
                if row["stage"] == "eval":
                    evals[row["trace"]].append(row)
                else:
                    self.assertEqual(row["mode"], "observe")
            for arm_list in evals.values():
                settings = {r["arm"]:r["budget"] for r in arm_list}
                self.assertEqual(set(settings), {"off", "observe", "adaptive", "static-small", "static-mid"})
                self.assertEqual(settings["off"], settings["adaptive"])
                self.assertEqual(settings["off"], settings["observe"])
                self.assertLess(settings["static-small"], settings["static-mid"])
                self.assertLess(settings["static-mid"], settings["off"])
            if not quick:
                self.assertGreater(len({arms[0]["arm"] for arms in evals.values()}), 1)

    def test_dry_run_and_trial_contract(self):
        dest = Path.cwd() / f".dry-experiment-{uuid4().hex}"
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--quick", "--dry-run", "--model", "not-needed", "--gpu", "2",
                                   "--rate-scale", "0.5", "--out", str(dest)]), 0)
        plan = json.loads(buf.getvalue())
        self.assertFalse(dest.exists())
        self.assertEqual(plan["workloads"]["eval-burst-3500"]["rate"], 12)
        args = SimpleNamespace(**plan["settings"])
        for row in plan["trials"]:
            cmd = trial_command(args, row, "trace.jsonl", dest, "profile.json")
            self.assertEqual(cmd[cmd.index("--mode")+1], row["mode"])
            self.assertEqual("--profile" in cmd, row["mode"] == "adaptive")

    def test_synthetic_trials_complete_pipeline_and_flag_changed_greedy_output(self):
        root = Path.cwd() / f".synthetic-experiment-{uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root)
        args = SimpleNamespace(resume=False, target_ms=40., min_mixed_coverage=.5)
        plan = make_plan(True)

        def artificial_trial(output, settings, run, trace, profile):
            path = output / run["name"]
            path.mkdir()
            requests, sha = load_trace(trace)
            manifest = dict(hardware={"name":"SYNTHETIC"}, model="SYNTHETIC", model_config_sha256="fake",
                            torch="SYNTHETIC", vllm="SYNTHETIC", trace_sha256=sha)
            (path / "manifest.json").write_text(json.dumps(manifest))
            rows = []
            for r in requests:
                t = r["arrival_s"]
                token = 2 if r["id"] == requests[0]["id"] and run["arm"] == "adaptive" else 1
                rows.append(dict(schema=1, label=run["name"], model="SYNTHETIC", expected_requests=len(requests),
                                 run_id=run["name"], trace_sha256=sha, id=r["id"], kind=r["kind"],
                                 backend="vllm", status="ok", scheduled_s=t, sent_s=t, end_s=t+.03,
                                 events=[[t+.01,1],[t+.02,1]], output_ids=[token,1]))
            (path / "requests.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            (path / "summary.json").write_text(json.dumps(summarize(rows, 1000, 3000, 100)))
            if run["mode"] != "off":
                row = dict(clock="scheduler_result_wall", run_id=run["name"], decode_count=2, context_tokens=512,
                           prefill_tokens=32, step_ms=5, reason="profile" if run["mode"]=="adaptive" else "native")
                (path / "measured-steps.jsonl").write_text((json.dumps(row)+"\n")*4)
            return path

        with patch("scripts.experiment.run_trial", artificial_trial), redirect_stdout(io.StringIO()):
            self.assertEqual(run_matrix(root, args, plan), 2)
        report = json.loads((root / "aggregate.json").read_text())
        checks = report["checks"]
        self.assertTrue(checks["all_requests_successful"])
        self.assertTrue(checks["holdout_has_mixed_support"])
        self.assertTrue(checks["adaptive_used_profile_on_mixed_step"])
        self.assertFalse(checks["all_paired_outputs_identical"])
        arms = report["evaluations"]["eval-burst-3500"]["arms"]
        self.assertEqual(arms["adaptive"]["paired_vs_off"]["different_sequences"], 1)
        self.assertEqual(arms["static-small"]["paired_vs_off"]["different_sequences"], 0)


if __name__ == "__main__":
    unittest.main()
