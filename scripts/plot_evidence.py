"""Draw paired workload TTFT samples from a verified public evidence bundle.

From the checkout: python scripts/plot_evidence.py --bundle evidence/pilot --out figures/ttft.png
An offline Python environment with Matplotlib is sufficient. PNG, SVG and PDF are supported.
For a full matrix, pass --trace with one evaluation workload key from index.json.
"""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.publish_evidence import read, stream, verify


ARM_NAMES = {
    "off": "Native",
    "observe": "Observe",
    "adaptive": "Adaptive",
    "static-small": "Static low",
    "static-mid": "Static mid",
}
COLORS = {"short": "#246B83", "long": "#8A4372"}
INK = "#233347"
MUTED = "#526477"
SLO = "#AC5219"


def samples(bundle, trace):
    # Check checksums, trace reconstruction, public summaries and same-trace pairing before plotting.
    verify(bundle)
    index = read(bundle / "index.json")
    plan = index["plan"]
    runs = [run for run in plan["trials"] if run["stage"] == "eval"]
    keys = sorted({run["trace"] for run in runs})
    if trace is None:
        if len(keys) != 1:
            raise ValueError("choose one evaluation workload with --trace: " + ", ".join(keys))
        trace = keys[0]
    if trace not in keys:
        raise ValueError("unknown evaluation trace: " + trace)
    by_arm = {r["arm"]: r for r in runs if r["trace"] == trace}
    if set(by_arm) != set(ARM_NAMES):
        raise ValueError("expected the native, observe, adaptive, static-small and static-mid arms")

    groups = {}
    backends = set()
    device = set()
    for arm in ARM_NAMES:
        run = by_arm[arm]
        completed = [a for a in index["attempts"][run["name"]] if a["status"] == "completed"]
        if len(completed) != 1:
            raise ValueError("expected one complete trial per arm; check " + arm)
        item = completed[0]
        device.add(item["environment"]["gpu_family"])
        folder = bundle / "trials" / run["name"] / f"attempt-{item['attempt']}"
        outcome = {"short": {}, "long": {}}
        for row in stream(folder / "requests.jsonl.gz"):
            backends.add(row["backend"])
            value = None
            if row["status"] == "ok":
                value = 1000 * (row["events"][0][0] - row["scheduled_s"])
                if value < 0:
                    raise ValueError("First token precedes scheduled arrival")
            outcome[row["kind"]][row["id"]] = value
        groups[arm] = (run["budget"], outcome)
    for arm, (_, outcome) in groups.items():
        if any(outcome[kind].keys() != groups["off"][1][kind].keys() for kind in ("short", "long")):
            raise ValueError("unpaired workload IDs in " + arm)
    if len(backends) != 1:
        raise ValueError("comparison mixes endpoint types")
    return index, trace, groups, backends.pop(), ", ".join(sorted(device))


def draw(bundle, out, trace=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib import ticker
    except ImportError as exc:
        raise RuntimeError("Drawing requires Matplotlib. Install it with python -m pip install matplotlib") from exc

    out = Path(out)
    if out.suffix.lower() not in (".png", ".svg", ".pdf"):
        raise ValueError("choose .png, .svg or .pdf")
    bundle = Path(bundle).resolve()
    if bundle in out.resolve().parents:
        raise ValueError("save the figure outside the checksummed evidence bundle")
    if out.exists():
        raise FileExistsError("figure already exists: " + str(out))
    index, trace, groups, backend, gpu = samples(bundle, trace)
    plan = index["plan"]
    workload = plan["workloads"][trace]
    counts = {k: len(groups["off"][1][k]) for k in ("short", "long")}
    colors = dict(text=INK, grid="#DBE1E6", axis="#AFBBC4")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.unicode_minus": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8))
        try:
            fig.subplots_adjust(left=.075, right=.98, bottom=.24, top=.74, wspace=.24)
            title = "First-token latency under " + workload["mix"] + " load"
            fig.text(.075, .935, title, ha="left", fontsize=22, fontweight="bold", color=INK)
            provenance = "Mock HTTP fixture" if backend == "mock" else f"vLLM · {gpu}"
            fig.text(.075, .875, f"{provenance}  |  {index['model']}  |  "
                     f"seed {workload['seed']}  |  nominal {workload['rate']:g} req/s  |  "
                     f"{sum(counts.values())} scheduled per arm", color=MUTED, fontsize=10)
            for ax, kind in zip(axes, ("short", "long")):
                slo = plan["settings"][f"{kind}_ttft_ms"]
                ids = sorted(groups["off"][1][kind])
                # Each request keeps its horizontal offset in all five arms so alignment is reproducible.
                offset = {rid: .5 * (i / max(1, len(ids) - 1) - .5) if len(ids) > 1 else 0.
                          for i, rid in enumerate(ids)}
                ticks = []
                all_times = []
                for x, (arm, (budget, outcome)) in enumerate(groups.items()):
                    observed = [(rid, t) for rid, t in outcome[kind].items() if t is not None]
                    values = [t for _, t in observed]
                    all_times.extend(values)
                    ax.scatter([x + offset[rid] for rid, _ in observed], values, s=33, zorder=3,
                               facecolors=COLORS[kind], edgecolors="white", linewidths=.5, alpha=.92)
                    ticks.append(f"{ARM_NAMES[arm]}\n{budget} tok\n{len(values)}/{len(ids)}")
                ax.axhline(slo, linestyle=(0, (4, 3)), linewidth=1.5, color=SLO, zorder=2)
                ax.text(.985, slo, f"SLO {slo:g} ms", ha="right", va="bottom", fontsize=9,
                        color=SLO, transform=ax.get_yaxis_transform(),
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.87, pad=1.5))
                span = all_times + [slo]
                log = min(span) > 0 and max(span) / min(span) > 12
                if log:
                    ax.set_yscale("log")
                    ax.set_ylim(min(span) / 1.4, max(span) * 1.65)
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
                else:
                    ax.set_ylim(0, max(span) * 1.22 or 1)
                ax.set_xlim(-.48, len(groups) - .52)
                ax.set_xticks(range(len(ticks)), labels=ticks)
                ax.set_title(f"{kind.capitalize()} requests  ·  {len(ids)} per arm",
                             loc="left", color=COLORS[kind], pad=12, fontweight="bold", fontsize=14)
                ax.set_ylabel("Client TTFT (ms" + (", log scale)" if log else ")"), color=INK)
                ax.grid(axis="y", color=colors["grid"], linewidth=.8, zorder=0)
                ax.spines[["top", "right"]].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color(colors["axis"])
                ax.tick_params(colors=INK, labelsize=9, length=3)
            fig.text(.075, .094, "One point per successfully completed request. Labels show the native step "
                     "budget and completed/scheduled requests.", color=MUTED, fontsize=9)
            fig.text(.075, .052, f"TTFT starts at the scheduled arrival.  |  Trace SHA-256 "
                     f"{index['workload_sha256'][trace][:16]}…", color=MUTED, fontsize=9)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("xb") as image:
                fig.savefig(image, format=out.suffix[1:].lower(), dpi=180, facecolor="white")
        finally:
            plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="directory created by publish_evidence.py export")
    parser.add_argument("--out", type=Path, required=True, help="new PNG, PDF or SVG path outside the bundle")
    parser.add_argument("--trace", help="evaluation trace to show, required for multi-trace matrices")
    args = parser.parse_args()
    try:
        print(draw(args.bundle, args.out, args.trace))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
