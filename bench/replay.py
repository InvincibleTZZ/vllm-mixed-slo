"""Open-loop HTTP replay against the vLLM v0.10.2 /v1/completions SSE protocol."""
import http.client
import json
import math
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from .trace import load_trace


class ProtocolError(Exception):
    pass


def base(row):
    return dict(id=row["id"], kind=row["kind"], scheduled_s=row["arrival_s"],
                input_tokens=len(row["prompt"]), target_tokens=row["output_tokens"],
                events=[], http_status=None, finish_reason=None, usage=None, output_ids=[])


def request_body(row, model):
    return json.dumps(dict(
        model=model, prompt=row["prompt"], max_tokens=row["output_tokens"],
        n=1, stream=True, temperature=0, ignore_eos=True, add_special_tokens=False,
        return_token_ids=True, stream_options={"include_usage": True},
    ), separators=(",", ":")).encode("ascii")


def consume(response, result, elapsed, deadline):
    """The pinned server sends one JSON data line per SSE event; empty deltas carry no token."""
    for raw in response:
        if time.monotonic() >= deadline:
            raise TimeoutError("total deadline reached")
        data = raw.strip()
        if not data.startswith(b"data:"):
            continue
        payload = data[5:].strip()
        if payload == b"[DONE]":
            break
        stamp = elapsed()
        obj = json.loads(payload)
        if obj.get("error"):
            raise ProtocolError(f"server stream error: {str(obj['error'])[:180]}")
        if obj.get("usage") is not None:
            result["usage"] = obj["usage"]
        for c in obj.get("choices", []):
            if c["index"] != 0:
                raise ProtocolError("unexpected choice index")
            if c.get("finish_reason") is not None:
                result["finish_reason"] = c["finish_reason"]
            ids = c.get("token_ids")
            if ids is None:
                if c.get("text"):
                    raise ProtocolError("stream lacks token_ids; use pinned vLLM with return_token_ids")
                continue
            if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                raise ProtocolError("invalid token_ids")
            if ids:
                result["output_ids"].extend(ids)
                result["events"].append([stamp, len(ids)])
    else:
        raise ProtocolError("EOF before [DONE]")
    usage = result["usage"] or {}
    if usage.get("completion_tokens") != len(result["output_ids"]):
        raise ProtocolError("token_ids count differs from final usage")
    if usage.get("prompt_tokens") != result["input_tokens"]:
        raise ProtocolError("prompt token count differs from trace")
    if len(result["output_ids"]) != result["target_tokens"] or result["finish_reason"] != "length":
        raise ProtocolError("incomplete generation or unexpected finish reason")


def fetch(row, body, endpoint, origin, timeout, backend):
    result = base(row)
    result["backend"] = backend
    elapsed = lambda: time.monotonic() - origin
    result["sent_s"] = elapsed()
    deadline = origin + row["arrival_s"] + timeout
    conn = None
    timer = None
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("deadline before dispatch")
        connection = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
        conn = connection(endpoint.hostname, endpoint.port, timeout=remaining)
        conn.connect()
        sock = conn.sock
        # A per-read timeout alone could be extended forever by slow SSE heartbeats.
        def interrupt_read():
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        timer = threading.Timer(max(0., deadline-time.monotonic()), interrupt_read)
        timer.daemon = True
        timer.start()
        route = endpoint.path.rstrip("/")
        route = route + ("/completions" if route.endswith("/v1") else "/v1/completions")
        conn.request("POST", route, body, {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "close",
        })
        with conn.getresponse() as resp:
            result["http_status"] = resp.status
            if resp.getheader("X-Mixed-SLO-Backend") == "mock":
                result["backend"] = "mock"
            if resp.status != 200:
                result.update(status="http_error", error=f"HTTP {resp.status}")
            else:
                consume(resp, result, elapsed, deadline)
                result["status"] = "ok"
    except (ProtocolError, ValueError, OSError, http.client.HTTPException, KeyError, TypeError) as exc:
        status = "timeout" if isinstance(exc, TimeoutError) or time.monotonic() >= deadline else "error"
        result.update(status=status, error=f"{type(exc).__name__}: {str(exc)[:180]}")
    finally:
        result["end_s"] = elapsed()
        if timer:
            timer.cancel()
        if conn:
            conn.close()
    # Even an otherwise complete request arriving just past the total deadline misses the run.
    if result["end_s"] > row["arrival_s"] + timeout:
        result.update(status="timeout", error="total client deadline reached")
    return result


def run(trace, url, model, out, timeout=120., max_inflight=128, backend="vllm", label="unspecified"):
    if not math.isfinite(timeout) or timeout <= 0 or max_inflight < 1:
        raise ValueError("timeout and max-inflight must be positive")
    endpoint = urlsplit(url)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("provide an HTTP(S) base URL without credentials, query or fragment")
    rows, digest = load_trace(trace)
    # Prepare prompt serialization outside the timed replay.
    bodies = [request_body(row, model) for row in rows]
    common = dict(schema=1, run_id=uuid.uuid4().hex, trace_sha256=digest, expected_requests=len(rows),
                  model=model, label=label, timeout_s=timeout, max_inflight=max_inflight)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = set()
    with path.open("x", encoding="utf-8", newline="\n") as output, ThreadPoolExecutor(max_workers=max_inflight) as pool:
        def record(result):
            output.write(json.dumps({**common, **result}, separators=(",", ":")) + "\n")
            output.flush()
        origin = time.monotonic()
        for row, body in zip(rows,bodies):
            delay = origin + row["arrival_s"] - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            done = {future for future in pending if future.done()}
            for future in done:
                record(future.result())
            pending.difference_update(done)
            if len(pending) >= max_inflight:
                result = base(row)
                result.update(status="client_overload", error="client max-inflight cap",
                              sent_s=None, end_s=time.monotonic()-origin, backend=backend)
                record(result)
                continue
            pending.add(pool.submit(fetch, row, body, endpoint, origin, timeout, backend))
        for future in as_completed(pending):
            record(future.result())
    return path
