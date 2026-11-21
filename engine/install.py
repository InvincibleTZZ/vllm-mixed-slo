"""Apply or reverse the single-file, SHA-256-pinned vLLM scheduler integration."""

import argparse
import hashlib
import importlib.util
import os
import stat
import tempfile
from pathlib import Path


UPSTREAM_SHA = "2bd28fc32cea71370298bb8da0d9a0bbd163ab315b2c6428540131e3999a255f"
RELATIVE = Path("v1/core/sched/scheduler.py")
HEADER = b"# mixed-slo v1 body-sha256 "
START = "# >>> mixed-slo insertion\n"
END = "# <<< mixed-slo insertion\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect_source(raw):
    if digest(raw) == UPSTREAM_SHA:
        return "upstream", raw
    if not raw.startswith(HEADER):
        raise ValueError("unrecognized scheduler.py; expected upstream vLLM 0.10.2 SHA-256 "
                         + UPSTREAM_SHA + ", found " + digest(raw))
    header, _, body = raw.partition(b"\n")
    if header != HEADER + digest(body).encode("ascii"):
        raise ValueError("patched scheduler was edited; refusing to overwrite it")
    original = []
    in_section = False
    for line in body.decode("utf-8").splitlines(keepends=True):
        if line == START and not in_section:
            in_section = True
        elif line == END and in_section:
            in_section = False
        elif not in_section:
            original.append(line)
    clean = "".join(original).encode("utf-8")
    if in_section or digest(clean) != UPSTREAM_SHA:
        raise ValueError("could not recover the pinned upstream bytes")
    return "patched", clean


def render(clean):
    if digest(clean) != UPSTREAM_SHA:
        raise ValueError("source SHA-256 mismatch")
    policy = Path(__file__).with_name("policy.py").read_text(encoding="utf-8")
    after = {
        "        self.use_pp = self.parallel_config.pipeline_parallel_size > 1":
            "        self._mixed_slo = MixedSLO.from_env(self)",
        "        token_budget = self.max_num_scheduled_tokens":
            "        mixed_step = (self._mixed_slo.begin(self.running, token_budget)\n"
            "                      if self._mixed_slo is not None else None)",
        "            num_new_tokens = min(num_new_tokens, token_budget)":
            "            if mixed_step is not None:\n"
            "                num_new_tokens = mixed_step.limit(request.num_prompt_tokens,\n"
            "                                                  request.num_computed_tokens, num_new_tokens)",
        "            token_budget -= num_new_tokens":
            "            if mixed_step is not None:\n"
            "                mixed_step.charge(request.num_prompt_tokens, request.num_computed_tokens,\n"
            "                                  num_new_tokens)",
        "                    num_new_tokens = min(num_new_tokens, token_budget)":
            "                    if mixed_step is not None:\n"
            "                        num_new_tokens = mixed_step.limit(request.num_prompt_tokens,\n"
            "                                                          num_computed_tokens, num_new_tokens)\n"
            "                        if num_new_tokens == 0:\n"
            "                            break",
        "                token_budget -= num_new_tokens":
            "                if mixed_step is not None:\n"
            "                    mixed_step.charge(request.num_prompt_tokens, num_computed_tokens,\n"
            "                                      num_new_tokens)",
        "        self._update_after_schedule(scheduler_output)":
            "        if mixed_step is not None:\n"
            "            self._mixed_slo.scheduled(mixed_step, scheduler_output, preempted_reqs)",
        "    def shutdown(self) -> None:":
            "        if self._mixed_slo is not None:\n"
            "            self._mixed_slo.close()",
    }
    before = {
        "class Scheduler(SchedulerInterface):": policy.strip() + "\n\n",
        "        sampled_token_ids = model_runner_output.sampled_token_ids":
            "        if self._mixed_slo is not None:\n"
            "            self._mixed_slo.result(scheduler_output)",
    }
    result = []
    hits = []
    for line in clean.decode("utf-8").splitlines(keepends=True):
        anchor = line.rstrip("\n")
        if anchor in before:
            hits.append(anchor)
            result.extend((START, before[anchor], "\n", END))
        result.append(line)
        if anchor in after:
            hits.append(anchor)
            result.extend((START, after[anchor], "\n", END))
    if sorted(hits) != sorted([*before, *after]):
        raise ValueError("scheduler anchors are missing or duplicated")
    body = "".join(result).encode("utf-8")
    compile(body, str(RELATIVE), "exec")
    return HEADER + digest(body).encode("ascii") + b"\n" + body


def atomic_write(path, content):
    mode = stat.S_IMODE(path.stat().st_mode)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def operate(action, root):
    path = root / RELATIVE
    original = path.read_bytes()
    status, clean = inspect_source(original)
    if action == "status":
        return status
    if action == "revert":
        desired, label = clean, "upstream"
    elif action == "apply":
        desired, label = render(clean), "patched"
    else:
        raise ValueError("unknown operation")
    if desired != original:
        atomic_write(path, desired)
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "apply", "revert"))
    parser.add_argument("--vllm-root", type=Path, help="vllm package directory containing v1/")
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            parser.error("vllm package not found; specify --vllm-root")
        root = Path(next(iter(spec.submodule_search_locations)))
    try:
        status = operate(args.action, root)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{status} {root / RELATIVE}")


if __name__ == "__main__":
    main()
