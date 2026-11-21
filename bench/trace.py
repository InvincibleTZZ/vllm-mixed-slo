"""Generate deterministic workloads made of explicit integer prompt tokens."""
import hashlib
import json
import math
import random
from pathlib import Path


def generate(n, mix, rate, seed, short_input, long_input, short_output, long_output,
             long_fraction=0.25, token_min=1024, token_max=4095, burst_size=8):
    if n < 1 or min(short_input, long_input, short_output, long_output) < 1:
        raise ValueError("request and token counts must be positive")
    if not math.isfinite(rate) or rate <= 0 or not 0 <= long_fraction <= 1:
        raise ValueError("invalid rate or long fraction")
    if token_min < 0 or token_min > token_max or burst_size < 1:
        raise ValueError("invalid token bounds or burst size")
    if mix not in {"short", "long", "mixed", "burst"}:
        raise ValueError(f"unknown mix {mix}")
    nlong = round(n * long_fraction) if mix in {"mixed","burst"} else (n if mix == "long" else 0)
    classes = ["long"] * nlong + ["short"] * (n - nlong)
    random.Random(seed).shuffle(classes)
    arrivals = random.Random(seed + 1)
    tokens = random.Random(seed + 2)
    offset = 0.0
    for i, cls in enumerate(classes):
        if i and mix != "burst":
            offset += arrivals.expovariate(rate)
        elif mix == "burst":
            offset = (i // burst_size) * burst_size / rate
        size, target = (long_input, long_output) if cls == "long" else (short_input, short_output)
        yield dict(id=f"r{i:06d}", kind=cls, arrival_s=round(offset, 9),
                   prompt=[tokens.randint(token_min, token_max) for _ in range(size)],
                   output_tokens=target)


def write_trace(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_trace(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty trace")
    ids = set()
    last = 0.0
    for row in rows:
        rid = row["id"]
        t = row["arrival_s"]
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError("duplicate or empty request id")
        if not math.isfinite(t) or t < last:
            raise ValueError("arrival offsets must be ordered and nonnegative")
        if row["kind"] not in {"long","short"} or not row["prompt"]:
            raise ValueError(f"invalid request: {rid}")
        if type(row["output_tokens"]) is not int or row["output_tokens"] < 1:
            raise ValueError(f"invalid output length: {rid}")
        if not isinstance(row["prompt"], list) or any(type(v) is not int or v < 0 for v in row["prompt"]):
            raise ValueError(f"invalid token IDs: {rid}")
        ids.add(rid)
        last = t
    return rows, hashlib.sha256(raw).hexdigest()
