"""A loopback SSE fixture; its token stream never runs a model or scheduler."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def write_chunk(self, data):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def event(self, payload):
        self.write_chunk(b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n")

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mode = req["model"]
        if mode == "reject":
            self.send_response(429)
            self.send_header("X-Mixed-SLO-Backend", "mock")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("X-Mixed-SLO-Backend", "mock")
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        n = req["max_tokens"]
        try:
            self.event({"choices": [{"index": 0, "token_ids": [], "text": ""}]})
            if mode == "stall":
                time.sleep(0.4)
            for i in range(0, n, 2 if mode == "batched" else 1):
                size = min(2, n-i) if mode == "batched" else 1
                time.sleep(0.001)
                ids = [(req["prompt"][0]+j) % 256 for j in range(i, i+size)]
                self.event({"choices": [{"index": 0, "token_ids": ids, "text": "X" * size,
                                         "finish_reason": "length" if i+size == n else None}]})
            count = n + int(mode == "bad_usage")
            self.event({"choices": [],"usage":{"prompt_tokens":len(req["prompt"]),"completion_tokens":count}})
            if mode != "truncated":
                self.write_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            # The fixture can intentionally exceed the client's deadline.
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as server:
        print(f"mock SSE server listening on loopback:{server.server_port}",flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
