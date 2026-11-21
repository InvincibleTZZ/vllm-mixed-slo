import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from bench.metrics import compare, load_results, summarize
from bench.mock_server import Handler
from bench.replay import run
from bench.trace import generate, load_trace, write_trace


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.worker.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(2)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.trace = self.root/"trace.jsonl"
        write_trace(self.trace,generate(4,"burst",100,42,8,12,3,4,.5))

    def replay(self, name, limit=8, timeout=3):
        out = self.root/(name+".jsonl")
        run(self.trace,self.url,name,out,timeout,limit,"mock","test")
        return load_results(out)

    def test_seeded_trace_and_exclusive_files(self):
        second = self.root/"second.jsonl"
        write_trace(second,generate(4,"burst",100,42,8,12,3,4,.5))
        self.assertEqual(self.trace.read_bytes(),second.read_bytes())
        rows,_ = load_trace(second)
        self.assertEqual([r["arrival_s"] for r in rows],[0.]*4)
        self.assertEqual(sum(r["kind"] == "long" for r in rows),2)
        with self.assertRaises(FileExistsError):
            write_trace(second,[])

    def test_real_http_chunked_sse_and_sequence_comparison(self):
        rows = self.replay("single")
        self.assertEqual({r["status"] for r in rows},{"ok"})
        self.assertEqual({r["backend"] for r in rows},{"mock"})
        summary = summarize(rows,5000,5000,5000)
        all_stats = summary["groups"]["all"]
        self.assertEqual(all_stats["slo_pass"],4)
        self.assertEqual(all_stats["token_itl_ms"]["n"],10)
        self.assertEqual(all_stats["ttft_ms"]["n"],4)
        self.assertAlmostEqual(all_stats["goodput_rps"],
                               sum(summary["groups"][k]["goodput_rps"] for k in ("short","long")))
        changed = [dict(r,output_ids=r["output_ids"].copy()) for r in rows]
        changed[0]["output_ids"][0] += 1
        self.assertEqual(compare(rows,changed)["different_sequences"],1)

    def test_multitoken_chunks_are_not_single_token_itls(self):
        rows = self.replay("batched")
        metrics = summarize(rows,5000,5000,5000)["groups"]["all"]
        self.assertEqual(metrics["coalesced_requests"],4)
        self.assertEqual(metrics["token_itl_ms"]["n"],0)
        self.assertEqual(metrics["tpot_ms"]["n"],0)
        self.assertEqual(metrics["slo_pass"],0)
        self.assertGreater(metrics["sse_token_chunk_gap_ms"]["n"],0)

    def test_scheduled_arrival_drain_and_single_token_semantics(self):
        first = dict(run_id="abc", trace_sha256="xyz", model="m", label="test", backend="mock",
                     kind="short", status="ok", scheduled_s=0, sent_s=1, end_s=4.25,
                     output_ids=[1, 2, 3], events=[[2, 1], [3, 1], [4, 1]])
        second = dict(first, kind="long", scheduled_s=3, sent_s=4, end_s=5.5,
                      output_ids=[4], events=[[5, 1]])
        groups = summarize([first, second], 1999, 2000, 1000)["groups"]
        self.assertEqual(groups["short"]["ttft_ms"]["p99"], 2000)
        self.assertEqual(groups["short"]["tpot_ms"]["p99"], 1000)
        self.assertEqual(groups["short"]["slo_pass"], 0)
        self.assertEqual(groups["long"]["tpot_ms"]["n"], 0)
        self.assertEqual(groups["long"]["slo_pass"], 1)
        self.assertAlmostEqual(groups["all"]["goodput_rps"], 1 / 5.5)

    def test_deadline_and_client_overload_remain_offered(self):
        rows = self.replay("stall",limit=1,timeout=.08)
        metrics = summarize(rows,5000,5000,5000)["groups"]["all"]
        self.assertEqual(metrics["offered"],4)
        self.assertEqual(metrics["statuses"],{"client_overload":3,"timeout":1})
        self.assertEqual(metrics["attainment"],0.)

    def test_rejected_corrupt_and_truncated_streams(self):
        for mode in ("reject","bad_usage","truncated"):
            with self.subTest(mode=mode):
                rows = self.replay(mode)
                metrics = summarize(rows,5000,5000,5000)["groups"]["all"]
                self.assertEqual(metrics["failed"],4)
                self.assertEqual(metrics["goodput_rps"],0)
                if mode == "reject":
                    self.assertEqual({r["http_status"] for r in rows},{429})

    def test_interrupted_result_is_rejected(self):
        rows = self.replay("single")
        partial = self.root/"partial.jsonl"
        with partial.open("w",encoding="utf-8") as out:
            for r in rows[:-1]:
                out.write(json.dumps(r)+"\n")
        with self.assertRaisesRegex(ValueError,"incomplete"):
            load_results(partial)


if __name__ == "__main__":
    unittest.main()
