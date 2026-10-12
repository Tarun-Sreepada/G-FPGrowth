#!/usr/bin/env python3
"""Benchmark G-FPGrowth: GPU vs multi-threaded CPU vs single-threaded CPU.

Times come from the binary's own in-process log (parse / tree / mine / total), best of N runs.
Writes out/benchmark.csv and out/benchmark.png.
"""
import argparse
import re
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CASES = [
    ("retail.txt", "0.002"), ("retail.txt", "0.0005"), ("chess.txt", "0.6"), ("mushrooms.txt", "0.1"),
    ("T10I4D100K.txt", "0.002"), ("BMS2_itemset_mining.txt", "0.0008"), ("connect.txt", "0.8"),
    ("pumsb.txt", "0.6"), ("accidents.txt", "0.3"),
]
MODES = ["gpu", "cpu", "cpu1"]
PAT = {
    "parse": re.compile(r"parsed on (?:GPU|CPU).* in ([\d.]+) ms"),
    "tree": re.compile(r"initial FP-tree.* ([\d.]+) ms"),
    "mine": re.compile(r"mined (\d+) frequent itemsets.* ([\d.]+) ms"),
    "total": re.compile(r"wrote (\d+) patterns to .*total ([\d.]+) ms"),
}


def run(binary, mode, path, minsup, out, sep=" "):
    t = time.time()
    p = subprocess.run([binary, "-m", mode, str(path), str(out), minsup, sep], capture_output=True, text=True, check=True)
    wall = (time.time() - t) * 1000
    txt = p.stdout + p.stderr
    r = {"wall_ms": wall}
    r["parse_ms"] = float(PAT["parse"].search(txt).group(1))
    r["tree_ms"] = float(PAT["tree"].search(txt).group(1))
    m = PAT["mine"].search(txt)
    r["patterns"], r["mine_ms"] = int(m.group(1)), float(m.group(2))
    r["total_ms"] = float(PAT["total"].search(txt).group(2))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build/bin/g-fpgrowth"))
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    rows = []
    for ds, ms in CASES:
        path = ROOT / "datasets" / ds
        if not path.exists():
            continue
        counts = set()
        for mode in MODES:
            best = None
            for _ in range(a.repeat):
                r = run(a.binary, mode, path, ms, out / "bench_out.txt")
                if best is None or r["total_ms"] < best["total_ms"]:
                    best = r
            counts.add(best["patterns"])
            rows.append({"dataset": f"{path.stem} @{ms}", "mode": mode, **best})
        assert len(counts) == 1, f"pattern count mismatch on {ds}@{ms}: {counts}"
        print(f"{ds} @{ms}: done ({counts.pop()} patterns)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark.csv", index=False)
    piv = df.pivot(index="dataset", columns="mode", values="total_ms")[MODES]
    piv["cpu1/gpu"] = piv["cpu1"] / piv["gpu"]
    piv["cpu1/cpu"] = piv["cpu1"] / piv["cpu"]
    print(piv.round(2).to_string())
    stage = df.pivot(index="dataset", columns="mode", values=["parse_ms", "tree_ms", "mine_ms"])
    print(stage.round(1).to_string())

    ax = piv[MODES].plot.bar(logy=True, figsize=(11, 5))
    ax.set_ylabel("total time (ms, log scale)")
    ax.set_title("G-FPGrowth: GPU vs CPU (multi-thread) vs CPU (single thread)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "benchmark.png", dpi=130)


if __name__ == "__main__":
    main()
