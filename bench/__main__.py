import argparse
import json
from pathlib import Path

from .trace import generate, write_trace
from .replay import run
from .metrics import load_results, summarize, compare


def main():
    parser = argparse.ArgumentParser(description="Reproducible vLLM mixed-load replay")
    cmds = parser.add_subparsers(dest="command",required=True)
    gen = cmds.add_parser("generate")
    gen.add_argument("--out",required=True)
    gen.add_argument("--mix",choices=["short","long","mixed","burst"],default="mixed")
    gen.add_argument("--requests",type=int,default=100)
    gen.add_argument("--rate",type=float,default=3.)
    gen.add_argument("--seed",type=int,default=42)
    for key,val in [("short-input",128),("long-input",2048),("short-output",32),("long-output",64),
                    ("token-min",1024),("token-max",4095),("burst-size",8)]:
        gen.add_argument("--"+key,type=int,default=val)
    gen.add_argument("--long-fraction",type=float,default=.25)
    replay = cmds.add_parser("run")
    for opt in ("trace","url","model","out"):
        replay.add_argument("--"+opt,required=True)
    replay.add_argument("--timeout",type=float,default=120.)
    replay.add_argument("--max-inflight",type=int,default=128)
    replay.add_argument("--backend",choices=["vllm","mock"],default="vllm",
                        help="endpoint provenance asserted by operator; mock server also marks replies")
    replay.add_argument("--label",default="unspecified",help="configuration name, e.g. baseline-2048")
    summ = cmds.add_parser("summarize")
    summ.add_argument("--results",required=True)
    summ.add_argument("--out")
    summ.add_argument("--short-ttft-ms",type=float,required=True)
    summ.add_argument("--long-ttft-ms",type=float,required=True)
    summ.add_argument("--tpot-ms",type=float,required=True)
    summ.add_argument("--short-tpot-ms",type=float)
    summ.add_argument("--long-tpot-ms",type=float)
    comp = cmds.add_parser("compare")
    comp.add_argument("--left",required=True)
    comp.add_argument("--right",required=True)
    comp.add_argument("--out")
    args = parser.parse_args()
    if args.command == "generate":
        rows = generate(args.requests,args.mix,args.rate,args.seed,args.short_input,args.long_input,
                        args.short_output,args.long_output,args.long_fraction,args.token_min,
                        args.token_max,args.burst_size)
        write_trace(args.out,rows)
    elif args.command == "run":
        run(args.trace,args.url,args.model,args.out,args.timeout,args.max_inflight,args.backend,args.label)
    else:
        if args.command == "summarize":
            value = summarize(load_results(args.results),args.short_ttft_ms,args.long_ttft_ms,
                              args.tpot_ms,args.short_tpot_ms,args.long_tpot_ms)
        else:
            value = compare(load_results(args.left),load_results(args.right))
        output = json.dumps(value,indent=2,allow_nan=False) + "\n"
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x",encoding="utf-8") as f:
                f.write(output)
        else:
            print(output,end="")


if __name__ == "__main__":
    main()
