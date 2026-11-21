"""Run one owned vLLM server and one immutable replay. Linux, NVIDIA, vLLM 0.10.2."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def smi(gpu, query):
    return subprocess.check_output(["nvidia-smi", "-i", str(gpu), query, "--format=csv,noheader,nounits"], text=True, timeout=12).strip()

def check_gpu(gpu):
    # Preserve an allocation already made by a cluster scheduler.
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and str(gpu) not in inherited.split(","):
        raise RuntimeError(f"physical GPU {gpu} is outside CUDA_VISIBLE_DEVICES={inherited!r}")
    info = smi(gpu, "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu")
    name, uuid, driver, total, used, util = [x.strip() for x in info.split(",")]
    if smi(gpu, "--query-compute-apps=pid") or int(used) > 1024 or int(util) > 10:
        raise RuntimeError(f"GPU {gpu} already in use: {used} MiB, {util}%")
    return dict(name=name, uuid=uuid, driver=driver, memory_mib=int(total))

def port_number():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n")
    tmp.replace(path)

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Pinned local HF model directory")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True, help="Physical GPU index; checks idleness")
    p.add_argument("--mode", choices=["off","observe","adaptive"], required=True)
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--model-len", type=int, default=3072)
    p.add_argument("--seqs", type=int, default=32)
    p.add_argument("--target-ms", type=float, default=40)
    p.add_argument("--profile", type=Path)
    p.add_argument("--dtype", choices=["auto","half","bfloat16"], default="auto", help="Model dtype")
    p.add_argument("--startup-timeout", type=int, default=360)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--short-ttft-ms", default="1000")
    p.add_argument("--long-ttft-ms", default="3000")
    p.add_argument("--tpot-ms", default="100")
    a = p.parse_args()
    if platform.system() != "Linux" or importlib.metadata.version("vllm").split("+")[0] != "0.10.2":
        p.error("requires Linux and vllm==0.10.2")
    model_path = Path(a.model).resolve()
    if not (model_path/"config.json").is_file(): p.error("model must be a local HF snapshot")
    if a.mode=="adaptive" and (not a.profile or not a.profile.is_file()): p.error("provide --profile")
    trace=a.trace.resolve()
    digest=file_sha(trace)
    gpu=check_gpu(a.gpu)
    out=a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    port=port_number()
    url=f"http://127.0.0.1:{port}"
    label=out.name
    env=dict(os.environ)
    for k in list(env):
        if k.startswith("MIXED_SLO_"): del env[k]
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), VLLM_USE_V1="1", VLLM_NO_USAGE_STATS="1",
               HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", MIXED_SLO_MODE=a.mode,
               MIXED_SLO_TRACE=str(out/"steps.jsonl"), MIXED_SLO_RUN_ID=label,
               MIXED_SLO_TARGET_MS=str(a.target_ms))
    if a.profile: env["MIXED_SLO_PROFILE"]=str(a.profile.resolve())
    argv=[sys.executable,"-m","vllm.entrypoints.openai.api_server","--model",str(model_path),
          "--served-model-name","pilot","--host","127.0.0.1","--port",str(port),
          "--tensor-parallel-size","1","--max-model-len",str(a.model_len),
          "--max-num-seqs",str(a.seqs),"--max-num-batched-tokens",str(a.budget),
          "--dtype",a.dtype,"--gpu-memory-utilization","0.78","--enforce-eager",
          "--enable-chunked-prefill","--no-enable-prefix-caching"]
    manifest=dict(mode=a.mode, model=model_path.name, model_config_sha256=file_sha(model_path/"config.json"),
                  trace_sha256=digest, policy_profile_sha256=file_sha(a.profile) if a.profile else None,
                  hardware=gpu, vllm=importlib.metadata.version("vllm"), torch=importlib.metadata.version("torch"),
                  argv=argv, target_ms=a.target_ms, run_id=label, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=platform.python_version(), status="starting", port=port)
    save(out/"manifest.json",manifest)
    def cli(*args):
        subprocess.run([sys.executable,"-m","bench",*map(str,args)],check=True,cwd=ROOT)
    try:
        with (out/"server.log").open("w") as log:
            server=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
            try:
                t=time.monotonic()
                while time.monotonic()-t < a.startup_timeout:
                    if server.poll() is not None: raise RuntimeError(f"server exited {server.returncode}; see server.log")
                    try:
                        with urlopen(url+"/health",timeout=2) as resp:
                            if resp.status==200: break
                    except OSError: time.sleep(1)
                else: raise TimeoutError("server startup timed out")
                # Warm the model using an identical short workload in every arm.
                warm=out/"warmup-trace.jsonl"
                cli("generate","--out",warm,"--mix","mixed","--short-input",64,"--long-input",128,
                    "--short-output",4,"--long-output",4,"--requests",4,"--rate",4,"--seed",701)
                cli("run","--trace",warm,"--url",url,"--model","pilot","--out",out/"warmup.jsonl","--timeout",a.timeout)
                warmup = [json.loads(line) for line in (out/"warmup.jsonl").read_bytes().splitlines()]
                if len(warmup) != 4 or any(row["status"] != "ok" for row in warmup):
                    raise RuntimeError("warmup failed; inspect warmup.jsonl before recording measurements")
                stepfile=out/"steps.jsonl"
                offset=stepfile.stat().st_size if stepfile.exists() else 0
                cli("run","--trace",trace,"--url",url,"--model","pilot","--out",out/"requests.jsonl",
                    "--label",label,"--timeout",a.timeout)
                cli("summarize","--results",out/"requests.jsonl","--out",out/"summary.json",
                    "--short-ttft-ms",a.short_ttft_ms,"--long-ttft-ms",a.long_ttft_ms,"--tpot-ms",a.tpot_ms)
                outcomes = json.loads((out/"summary.json").read_bytes())["groups"]["all"]
                if not outcomes["successful"]:
                    raise RuntimeError("all evaluated requests failed; see requests.jsonl")
                manifest["evaluated_failures"] = outcomes["failed"]
            finally:
                stop(server)
        if a.mode!="off":
            if not stepfile.exists() or stepfile.stat().st_size<=offset:
                raise RuntimeError("instrumentation emitted no evaluation steps; check installation and server.log")
            with stepfile.open("rb") as inp:
                inp.seek(offset)
                (out/"measured-steps.jsonl").write_bytes(inp.read())
        manifest["status"]="completed"
    except BaseException as exc:
        manifest.update(status="failed",error=str(exc))
        raise
    finally:
        save(out/"manifest.json",manifest)
if __name__=="__main__": main()
