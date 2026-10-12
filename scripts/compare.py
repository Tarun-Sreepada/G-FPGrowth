#!/usr/bin/env python3
"""Run the PAMI reference FPGrowth and G-FPGrowth on the same input and compare the
mined itemsets (as {frozenset(items): support}). Exit code 0 iff they are identical."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path, sep: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            items, sup = line.rsplit(":", 1)
            key = frozenset(i for i in items.split(sep) if i)
            if key in out:
                raise SystemExit(f"{path}: duplicate pattern {sorted(key)}")
            out[key] = int(sup)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("minsup")
    ap.add_argument("--sep", default=" ", help="literal separator character used in the input file")
    ap.add_argument("--mode", default="gpu", choices=["gpu", "cpu", "cpu1"])
    ap.add_argument("--algo", default="auto")
    ap.add_argument("--binary", default=str(ROOT / "build/bin/g-fpgrowth"))
    ap.add_argument("--workdir", default=str(ROOT / "out"))
    a = ap.parse_args()

    work = Path(a.workdir)
    work.mkdir(exist_ok=True)
    stem = Path(a.input).stem
    ref_out, gpu_out = work / f"{stem}_ref.txt", work / f"{stem}_gpu.txt"

    t = time.time()
    subprocess.run([sys.executable, str(ROOT / "reference/FPGrowth.py"), a.input, str(ref_out), a.minsup, a.sep],
                   check=True, stdout=subprocess.DEVNULL)
    t_ref = time.time() - t
    t = time.time()
    subprocess.run([a.binary, "--mode", a.mode, "--algo", a.algo, a.input, str(gpu_out), a.minsup, a.sep], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_gpu = time.time() - t

    ref, gpu = load(ref_out, "\t"), load(gpu_out, a.sep)
    ok = ref == gpu
    print(f"{stem} minsup={a.minsup}: reference={len(ref)} gpu={len(gpu)} patterns "
          f"[python {t_ref:.2f}s, {a.mode} {t_gpu:.2f}s] -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        only_ref, only_gpu = ref.keys() - gpu.keys(), gpu.keys() - ref.keys()
        diff = [k for k in ref.keys() & gpu.keys() if ref[k] != gpu[k]]
        print(f"  only in reference: {len(only_ref)}, only in gpu: {len(only_gpu)}, support differs: {len(diff)}")
        for k in list(only_ref)[:5]: print("   ref only:", sorted(k), ref[k])
        for k in list(only_gpu)[:5]: print("   gpu only:", sorted(k), gpu[k])
        for k in diff[:5]: print("   differs:", sorted(k), ref[k], gpu[k])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
