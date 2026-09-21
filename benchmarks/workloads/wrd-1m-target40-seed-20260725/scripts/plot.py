#!/usr/bin/env python3
"""Standalone figures; reads generated records, never synthesizes measurements."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference-generated", type=Path,
                        help="Also show the selected trace from another generated directory")
    args = parser.parse_args()
    out = args.generated.resolve()
    comparison = json.loads((out / "analysis/comparison.json").read_text())
    w = comparison["selected"]["W"]
    data = {label: json.loads((out / f"analysis/{label}-per-request-wrd.json").read_text()) for label in ["w32", f"w{w}"]}
    if args.reference_generated:
        reference = args.reference_generated.resolve()
        rw = json.loads((reference / "analysis/comparison.json").read_text())["selected"]["W"]
        data[f"w{rw}"] = json.loads((reference / f"analysis/w{rw}-per-request-wrd.json").read_text())
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white", "svg.hashsalt": "wrd-fixed-seed-20260725"})

    def save(fig, name):
        for ext in ("png", "svg"):
            metadata = {"Date": None} if ext == "svg" else {"Software": "WRD offline dataset audit"}
            fig.savefig(figures / f"{name}.{ext}", dpi=180, bbox_inches="tight", metadata=metadata)
        plt.close(fig)

    colors = {label: color for label, color in zip(data, ["#53627c", "#087e8b", "#a6621e"])}
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for label, rows in data.items():
        values = np.sort([r["wrd_tokens"] for r in rows if r["wrd_tokens"] is not None])
        xs = np.r_[0., values / 1e6]
        ys = np.r_[0., np.arange(1, len(values) + 1) / len(values)]
        f = np.mean(values <= 1e6)
        ax.step(xs, ys, where="post", linewidth=2.3, color=colors[label], label=f"{label.upper()}  F(1M) = {f:.4%}")
        ax.scatter([1], [f], color=colors[label], s=50, zorder=5)
        ax.annotate(f"{f:.4%}", (1, f), xytext=(12, -15 if label == "w32" else 8), textcoords="offset points", color=colors[label], weight="bold")
    ax.axvline(1, color="#bd5a38", linestyle="--", linewidth=1.3, label="1,000,000-token threshold")
    if comparison.get("target_is_acceptance_requirement", True):
        ax.axhspan(.38, .42, color="#e8bd52", alpha=.16, label="Target: 38%-42%")
    ax.set(xlabel="WRD (million aligned, unsalted tokens)", ylabel="Empirical CDF (equal weight per finite reuse)", xlim=(0, 3.7), ylim=(0, 1.015))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.18)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Full-trace WRD: " + " / ".join(label.upper() for label in data), x=.125, ha="left", fontsize=16, weight="bold")
    fig.text(.125, .015, "3,279 finite reuses per trace; 100 first visits excluded; real zero distances included.\nGlobal submission-order footprint proxy, not an observed cache hit rate.", fontsize=9, color="#555")
    fig.subplots_adjust(bottom=.19, top=.87)
    save(fig, f"wrd-cdf-w32-vs-w{w}")

    rows = data[f"w{w}"]
    selected = comparison["selected"]
    tail = selected["tail_start_ordinal"]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    finite = [r for r in rows if r["wrd_tokens"] is not None]
    ax.scatter([r["ordinal"] for r in finite], [r["wrd_tokens"] / 1e6 for r in finite], s=8, alpha=.5, color=colors[f"w{w}"], linewidths=0, label="Finite WRD")
    firsts = [r["ordinal"] for r in rows if r["first_visit"]]
    ax.scatter(firsts, [-.1] * len(firsts), marker="|", s=55, color="#777", label="First visits (undefined; shown below zero)")
    ax.axhline(1, color="#bd5a38", linestyle="--", linewidth=1.3, label="1,000,000 tokens")
    ax.axvspan(tail - .5, len(rows) - .5, color="#e8bd52", alpha=.25, label=f"Tail: active pool < {w}")
    ax.axvline(tail - .5, color="#9f7e27", linestyle=":")
    ax.set(xlabel="Global submission ordinal (zero-based)", ylabel="WRD (million aligned, unsalted tokens)", xlim=(-5, len(rows) + 5), ylim=(-.17, 2.1))
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"W{w}: per-request WRD and the natural drain tail", x=.125, ha="left", fontsize=16, weight="bold")
    fig.text(.125, .017, f"Full pool: {selected['full_pool_count_le_1000000']:,}/{selected['full_pool_finite_count']:,} = {selected['full_pool_cdf_1000000']:.4%}  |  "
             f"Tail (ordinal {tail} onward): {selected['tail_count_le_1000000']}/{selected['tail_finite_count']} = {selected['tail_cdf_1000000']:.4%}\n"
             f"Full trace: {selected['all_count_le_1000000']:,}/3,279 = {selected['all_cdf_1000000']:.4%}; no tail samples removed.", fontsize=10)
    fig.subplots_adjust(bottom=.20, top=.87)
    save(fig, f"w{w}-wrd-by-order")

    with (out / "analysis/search-all-w.csv").open() as f:
        search = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot([int(r["W"]) for r in search], [float(r["all_cdf_1000000"]) for r in search], marker=".", color="#087e8b", linewidth=1.5)
    if comparison.get("target_is_acceptance_requirement", True):
        ax.axhspan(.38, .42, color="#e8bd52", alpha=.3)
        ax.axhline(.4, color="#bd5a38", linestyle="--", linewidth=1)
    ax.scatter([w], [selected["all_cdf_1000000"]], s=75, color="#bd5a38", zorder=5)
    ax.annotate(f"W{w}: {selected['all_cdf_1000000']:.4%}", (w, selected["all_cdf_1000000"]), xytext=(w + 8, .53), arrowprops={"arrowstyle": "->", "color": "#666"})
    ax.set(xlabel="Active session pool W (client C remains 32)", ylabel="F(1,000,000 tokens)", xlim=(1, 100), ylim=(0, 1.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.18)
    ax.set_title("All 100 integer candidates; no monotonicity assumed", loc="left", pad=15)
    save(fig, "all-w-search")
    print(figures)


if __name__ == "__main__":
    main()
