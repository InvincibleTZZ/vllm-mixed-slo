from pathlib import Path
import json
import tempfile
import unittest

from scripts.experiment import make_plan, trace_inputs
from scripts.publish_evidence import dump, export, read, stream, verify
from bench.trace import load_trace


class PublicEvidenceTest(unittest.TestCase):
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
            report = export(matrix,out)
            self.assertEqual(report,verify(out))
            self.assertEqual(report["completed_trials"],2)
            self.assertEqual(report["planned_trials"],8)
            hit = runs[0]["name"]
            metrics = report["attempts"][f"trials/{hit}/attempt-1"]["metrics"]["groups"]["all"]
            self.assertEqual((metrics["offered"],metrics["successful"],metrics["slo_pass"]),(16,15,15))
            self.assertEqual(report["paired_vs_off"][hit]["not_paired_success"],1)
            self.assertEqual(read(out/"index.json")["attempts"][hit][1]["failure"],"server_exited")
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
