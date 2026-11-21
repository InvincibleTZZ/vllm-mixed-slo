from pathlib import Path
import json
import tempfile
import unittest

from scripts.experiment import make_plan, trace_inputs
from scripts.publish_evidence import dump, export, read, stream, verify
from bench.trace import load_trace
from calibration.table import Step, fit


class PublicEvidenceTest(unittest.TestCase):
    def test_rebuild_private_profile_and_holdout_from_public_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = root / "matrix"
            matrix.mkdir()
            plan = make_plan(True)
            plan["settings"] = dict(target_ms=10, short_ttft_ms=500, long_ttft_ms=1000,
                                    tpot_ms=100, min_mixed_coverage=0)
            plan["implementation"] = dict(model_snapshot={"sha256": "a"*64, "files": {"hidden-path": "b"*64}})
            dump(matrix/"plan.json", plan)
            paths = trace_inputs(matrix, plan)
            training = []
            for trial in plan["trials"]:
                if trial["stage"] == "eval":
                    continue
                name = trial["name"]
                folder = matrix / "trials" / "attempt-1" / name
                folder.mkdir(parents=True)
                rows, sig = load_trace(paths[trial["trace"]])
                dump(folder/"manifest.json", dict(status="completed", run_id=name, mode="observe", trace_sha256=sig))
                with (folder/"requests.jsonl").open("w") as f:
                    for r in rows:
                        count = r["output_tokens"]
                        end = r["arrival_s"] + .01
                        payload = dict(schema=1, id=r["id"], kind=r["kind"], status="ok", backend="mock",
                                       scheduled_s=r["arrival_s"], sent_s=r["arrival_s"], end_s=end+.02*count,
                                       trace_sha256=sig, expected_requests=len(rows), events=[[end+j*.02,1]
                                       for j in range(count)], max_inflight=32, timeout_s=2,
                                       output_ids=list(range(count)), input_tokens=len(r["prompt"]), target_tokens=count)
                        f.write(json.dumps(payload)+"\n")
                with (folder/"measured-steps.jsonl").open("w") as f:
                    for ms in (4, 12):
                        step = dict(clock="scheduler_result_wall", run_id=name, decode_count=1,
                                    context_tokens=128, prefill_tokens=32, step_ms=ms, schedule_ms=.2,
                                    reason="native", cap_tokens=None, excluded_cumulative={"hidden-path": 1})
                        f.write(json.dumps(step)+"\n")
                        if trial["stage"] == "fit":
                            training.append(Step.from_dict(step))
            profile = fit(training, min_samples=2, metadata={"path": "hidden-path"})
            dump(matrix/"profile.json", profile)
            report = export(matrix, root/"evidence")
            self.assertEqual(verify(root/"evidence"),report)
            self.assertEqual(report["profile"]["cells"],profile["cells"])
            self.assertEqual(report["holdout"]["coverage"],1.)
            self.assertEqual(report["holdout"]["covered_rows"],2)
            index = read(root/"evidence"/"index.json")
            self.assertEqual(index["implementation"]["model_snapshot_sha256"],"a"*64)
            self.assertNotIn("hidden-path",json.dumps(report))
            profile["cells"][0]["p95_ms"] += 3
            dump(matrix/"profile.json", profile)
            with self.assertRaisesRegex(ValueError, "refit differs"):
                export(matrix,root/"mismatch")
            self.assertFalse((root/"mismatch").exists())

    def test_private_data_exclusion_and_independent_recomputation(self):
        secret = "private-user-XY"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix, out = root/"private", root/"public"
            matrix.mkdir()
            plan = make_plan(True)
            plan["settings"] = dict(model="/home/" + secret, gpu=3, target_ms=40, short_ttft_ms=1000,
                                    long_ttft_ms=3000, tpot_ms=100, min_mixed_coverage=0)
            dump(matrix/"plan.json", plan)
            traces = trace_inputs(matrix, plan)
            runs = [r for r in plan["trials"] if r["stage"]=="eval" and r["arm"] in ("off","static-small")]
            for run in runs:
                folder = matrix / "trials" / "attempt-1" / run["name"]
                folder.mkdir(parents=True)
                offered, digest = load_trace(traces[run["trace"]])
                manifest = dict(run_id=run["name"], mode="off", status="completed", trace_sha256=digest,
                                hardware={"name":"NVIDIA A100-SXM4-80GB", "uuid":"GPU-"+secret},
                                argv=["serve", "/home/"+secret], model=secret, hostname=secret)
                dump(folder/"manifest.json",manifest)
                with (folder/"requests.jsonl").open("w",encoding="utf-8") as f:
                    for i, req in enumerate(offered):
                        n = req["output_tokens"]
                        end = req["arrival_s"]+.1
                        row = dict(id=req["id"], schema=1, model=secret, label=secret, run_id=secret,
                                   backend="mock", kind=req["kind"], scheduled_s=req["arrival_s"],
                                   sent_s=req["arrival_s"], end_s=end, trace_sha256=digest,
                                   expected_requests=len(offered), status="ok" if i else "timeout",
                                   input_tokens=len(req["prompt"]), target_tokens=n, timeout_s=2, max_inflight=64,
                                   events=[[end+j*.001, 1] for j in range(n)] if i else [],
                                   output_ids=list(range(n)) if i else [], error="/home/"+secret, unknown=secret,
                                   finish_reason="length" if i else None, usage={"hostname":secret})
                        row["end_s"] += n*.001
                        f.write(json.dumps(row)+"\n")
                failed = matrix/"trials"/"attempt-2"/run["name"]
                failed.mkdir(parents=True)
                dump(failed/"manifest.json",dict(manifest,status="failed",error="server exited on "+secret))
                (failed/"requests.jsonl").write_bytes(b'{"id":')
            report = export(matrix,out)
            self.assertEqual(report,verify(out))
            self.assertEqual(report["completed_trials"],2)
            self.assertEqual(report["planned_trials"],8)
            hit = runs[0]["name"]
            metrics = report["attempts"][f"trials/{hit}/attempt-1"]["metrics"]["groups"]["all"]
            self.assertEqual((metrics["offered"],metrics["successful"],metrics["slo_pass"]),(16,15,15))
            self.assertEqual(report["paired_vs_off"][hit]["not_paired_success"],1)
            self.assertEqual(read(out/"index.json")["attempts"][hit][1]["failure"],"server_exited")
            self.assertTrue(read(out/"index.json")["attempts"][hit][1]["torn_final_record"])
            for path in out.rglob("*"):
                if path.is_file():
                    payload = json.dumps(list(stream(path))) if path.suffix==".gz" else path.read_text()
                    self.assertNotIn(secret,payload)
                    self.assertNotIn("GPU-",payload)
            self.assertEqual(report["attempts"][f"trials/{hit}/attempt-1"]["metrics"]["backend_labels"],["mock"])
            with (out/"report.json").open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(out)


if __name__ == "__main__":
    unittest.main()
