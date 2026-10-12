#!/usr/bin/env python3
"""Sweep datasets x 5 minimum supports x all algorithm/backend configurations and plot it.

Every (dataset, minsup, config) point is run once as a warm-up (not counted) and then
repeatedly until at least --min-seconds of wall time has accumulated; the reported time is
the mean in-process total (parse + rank + tree/bitmaps + mine + write output) in ms.
Writes docs/sweep.csv and docs/sweep.png.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# (dataset file, 5 minimum supports as fractions of the database, easiest -> hardest)
SWEEP = [
    ("chess.txt", [0.8, 0.75, 0.7, 0.65, 0.6]),
    ("mushrooms.txt", [0.3, 0.2, 0.15, 0.12, 0.1]),
    ("connect.txt", [0.95, 0.9, 0.85, 0.82, 0.8]),
    ("pumsb.txt", [0.8, 0.75, 0.7, 0.67, 0.65]),
    ("accidents.txt", [0.6, 0.5, 0.4, 0.35, 0.3]),
    ("retail.txt", [0.01, 0.004, 0.002, 0.001, 0.0005]),
    ("T10I4D100K.txt", [0.01, 0.005, 0.003, 0.002, 0.001]),
    ("BMS1_itemset_mining.txt", [0.01, 0.005, 0.003, 0.002, 0.0015]),
    ("BMS2_itemset_mining.txt", [0.004, 0.002, 0.0015, 0.001, 0.0008]),
    ("foodmartFIM.txt", [0.005, 0.003, 0.002, 0.0015, 0.001]),
]

# name, label, CLI args, colour (validated categorical slots 1-7, fixed order), linestyle, marker
CONFIGS = [
    ("gpu-fp-device", "GPU FP-Growth (device queue)", ["-m", "gpu", "-a", "fpgrowth", "--sched", "device"], "#2a78d6", "-", "o"),
    ("gpu-fp-host", "GPU FP-Growth (host batches)", ["-m", "gpu", "-a", "fpgrowth", "--sched", "host"], "#eb6834", "-", "s"),
    ("gpu-eclat", "GPU Eclat (bitsets)", ["-m", "gpu", "-a", "eclat"], "#1baf7a", "-", "^"),
    ("cpu-fp", "CPU 12T FP-Growth", ["-m", "cpu", "-a", "fpgrowth"], "#eda100", "--", "D"),
    ("cpu-eclat", "CPU 12T Eclat (bitsets)", ["-m", "cpu", "-a", "eclat"], "#e87ba4", "--", "v"),
    ("cpu1-fp", "CPU 1T FP-Growth", ["-m", "cpu1", "-a", "fpgrowth"], "#008300", ":", "P"),
    ("cpu1-eclat", "CPU 1T Eclat (bitsets)", ["-m", "cpu1", "-a", "eclat"], "#6250d6", ":", "X"),
]

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
RE_TOTAL = re.compile(r"wrote (\d+) patterns to .*total ([\d.]+) ms")
RE_MINE = re.compile(r"mined (\d+) frequent itemsets.* ([\d.]+) ms")


def run_once(binary, args, path, minsup, sep, out, timeout):
    t = time.time()
    p = subprocess.run([binary, *args, str(path), str(out), repr(minsup), sep],
                       capture_output=True, text=True, timeout=timeout, check=True)
    txt = p.stdout + p.stderr
    m = RE_TOTAL.search(txt)
    mi = RE_MINE.search(txt)
    return {
        "wall": time.time() - t,
        "patterns": int(m.group(1)),
        "total_ms": float(m.group(2)),
        "mine_ms": float(mi.group(2)) if mi else float("nan"),
        "fallback": ("falling back" in txt) or ("host-driven batches" in txt),
        "eclat_failed": "falling back to FP-Growth" in txt,
    }


def measure(binary, args, path, minsup, sep, out, min_seconds, timeout, max_runs=5000):
    first = run_once(binary, args, path, minsup, sep, out, timeout)  # warm-up, also the guard
    if first["eclat_failed"]:
        return None, first
    runs, spent = [], 0.0
    while spent < min_seconds and len(runs) < max_runs:
        r = run_once(binary, args, path, minsup, sep, out, timeout)
        runs.append(r)
        spent += r["wall"]
    return runs, first


def sweep(a):
    rows = []
    only = set(a.datasets.split(",")) if a.datasets else None
    out = ROOT / "out" / "sweep_out.txt"
    out.parent.mkdir(exist_ok=True)
    for ds, minsups in SWEEP:
        if only and Path(ds).stem not in only:
            continue
        path = ROOT / "datasets" / ds
        if not path.exists():
            print(f"skip {ds} (missing)", file=sys.stderr)
            continue
        dead = set()  # configs that already exceeded the per-run limit at an easier minsup
        for ms in minsups:
            counts = {}
            for name, label, args, *_ in CONFIGS:
                if name in dead:
                    continue
                try:
                    runs, first = measure(a.binary, args, path, ms, " ", out, a.min_seconds, a.max_run_seconds)
                except subprocess.TimeoutExpired:
                    dead.add(name)
                    print(f"  {ds}@{ms} {name}: timeout", flush=True)
                    continue
                if runs is None:
                    print(f"  {ds}@{ms} {name}: not applicable (bitmaps do not fit)", flush=True)
                    continue
                if first["wall"] > a.max_run_seconds * 0.5:
                    dead.add(name)  # next (harder) minsup would exceed the limit
                tot = np.array([r["total_ms"] for r in runs])
                counts[name] = first["patterns"]
                rows.append({
                    "dataset": Path(ds).stem, "minsup": ms, "config": name, "patterns": first["patterns"],
                    "total_ms_mean": tot.mean(), "total_ms_min": tot.min(),
                    "mine_ms_mean": np.nanmean([r["mine_ms"] for r in runs]),
                    "runs": len(runs), "wall_s": sum(r["wall"] for r in runs), "fallback": first["fallback"],
                })
                print(f"  {ds}@{ms} {name}: {tot.mean():.1f} ms (n={len(runs)}, {first['patterns']} patterns)", flush=True)
            if len(set(counts.values())) > 1:
                raise SystemExit(f"PATTERN COUNT MISMATCH on {ds}@{ms}: {counts}")
    return pd.DataFrame(rows)


def fmt_count(n):
    if n >= 1e6: return f"{n / 1e6:.1f}M"
    if n >= 1e3: return f"{n / 1e3:.0f}k"
    return str(n)


def plot(df, png):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    datasets = [Path(d).stem for d, _ in SWEEP if Path(d).stem in set(df.dataset)]
    ncols = 5
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows + 1.3), facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, ds in enumerate(datasets):
        ax = axes.flat[i]
        ax.set_visible(True)
        ax.set_facecolor(SURFACE)
        d = df[df.dataset == ds]
        mss = sorted(d.minsup.unique(), reverse=True)  # easy -> hard, left -> right
        xs = {m: k for k, m in enumerate(mss)}
        for name, label, _, color, ls, mk in CONFIGS:
            c = d[d.config == name].sort_values("minsup", ascending=False)
            if c.empty:
                continue
            x = [xs[m] for m in c.minsup]
            ax.plot(x, c.total_ms_mean, color=color, ls=ls, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.0,
                    label=label, zorder=3)
            fb = c[c.fallback & (name == "gpu-fp-device")]
            if len(fb):
                ax.plot([xs[m] for m in fb.minsup], fb.total_ms_mean, ls="none", marker=mk, ms=7,
                        mfc=SURFACE, mec=color, mew=1.6, zorder=4)
        ax.set_yscale("log")
        ax.set_title(ds, fontsize=11, loc="left", color=INK, fontweight="bold")
        pc = d.groupby("minsup").patterns.first()
        ax.set_xticks(range(len(mss)))
        ax.set_xticklabels([f"{m:g}\n{fmt_count(int(pc[m]))}" for m in mss], fontsize=8)
        ax.set_xlim(-0.3, len(mss) - 0.7)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.grid(True, which="minor", axis="y", color=GRID, lw=0.4, alpha=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        if i % ncols == 0:
            ax.set_ylabel("total time (ms, log)")
        if i // ncols == nrows - 1 or i + ncols >= len(datasets):
            ax.set_xlabel("min support  /  patterns found", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("G-FPGrowth: time vs minimum support, GPU vs CPU, FP-Growth vs bitset Eclat",
                 fontsize=14, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.945, "lower is better; every point is the mean of repeated runs totalling >= 1 s; "
             "hollow marker = device queue fell back to host batches", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(png, dpi=140, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build/bin/g-fpgrowth"))
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-run-seconds", type=float, default=60.0)
    ap.add_argument("--csv", default=str(ROOT / "docs/sweep.csv"))
    ap.add_argument("--png", default=str(ROOT / "docs/sweep.png"))
    ap.add_argument("--datasets", help="comma separated dataset stems to (re)run; other rows are kept from the CSV")
    ap.add_argument("--replot", action="store_true", help="re-draw from the CSV without running")
    a = ap.parse_args()
    Path(a.csv).parent.mkdir(exist_ok=True)
    if a.replot:
        df = pd.read_csv(a.csv)
    else:
        df = sweep(a)
        if a.datasets and Path(a.csv).exists():
            old = pd.read_csv(a.csv)
            df = pd.concat([old[~old.dataset.isin(set(df.dataset))], df], ignore_index=True)
        df.to_csv(a.csv, index=False)
    plot(df, a.png)
    print(f"wrote {a.csv} and {a.png}")


if __name__ == "__main__":
    main()
