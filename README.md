# G-FPGrowth

**This is the FP-Growth algorithm implemented on the GPU.**

G-FPGrowth mines all frequent itemsets of a transaction database. Parsing, support
counting, FP-tree construction and mining all run on the GPU (CUDA).

## Usage

```
g-fpgrowth [-m gpu|cpu|cpu1] [-t threads] [-v] <input> <output> <minsup> [sep]
```

`-m gpu` (default) runs on the GPU, `-m cpu` is the multi-threaded CPU baseline
(`-t` threads, default all cores) and `-m cpu1` the single-threaded one.

| arg      | meaning |
|----------|---------|
| `input`  | transaction file, one transaction per line, items are non-negative integers |
| `output` | frequent itemsets, one `item<sep>item...:support` per line |
| `minsup` | absolute count (`10`) or, if it contains a `.`, a fraction of the database (`0.01`) |
| `sep`    | item separator: `tab` (default), `space`, `comma`, `semicolon`, `pipe` or one character |

```
./build/bin/g-fpgrowth datasets/retail.txt out/retail.txt 0.01 space
```

## Pipeline

1. **Load + parse on the GPU**: the file is copied to the GPU as raw bytes, then
   count `\n` -> allocate an array -> record the newline positions and sort them (line
   begin/end) -> count tokens per line, exclusive scan -> convert ASCII to `int32`,
   one thread per line.
2. **Count + rank**: de-duplicated `(transaction, item)` keys, item supports via
   sort + reduce-by-key, infrequent items dropped, items ranked by support.
3. **Initial FP-tree**: transactions are sorted by rank and inserted concurrently into
   a lock-free hash of `(parent, item) -> node`; nodes are then grouped per item.
4. **Chunked mining**: the size of the initial tree is now known and every conditional
   FP-tree is no larger than it, so the remaining GPU memory is partitioned into
   equal chunks of that size. One thread block builds one conditional tree into one
   chunk (pattern-growth, many trees in flight per batch); chunks are recycled as
   subtrees finish.

## Build

Requires CUDA (nvcc + thrust/cub), CMake >= 3.24, a C++17 compiler.

```
git submodule update --init      # spdlog, CLI11 (extern/)
cmake -S . -B build && cmake --build build -j
```

## Datasets and correctness check

```
scripts/get_datasets.sh                                  # SPMF datasets -> datasets/
uv sync                                                  # python env (pandas, numpy, matplotlib, PAMI, ...)
uv run python scripts/compare.py datasets/retail.txt 0.01
```

Dataset sources (SPMF, with original provenance) are listed in
[`datasets/README.md`](datasets/README.md).

`reference/FPGrowth.py` is the PAMI Python implementation; `scripts/compare.py` runs
both and checks that the sets of itemsets and supports are identical.

## Benchmark

`uv run python scripts/benchmark.py` runs GPU / CPU / CPU-1-thread (best of 3, in-process
timings, includes writing the output) and writes `out/benchmark.csv` and `out/benchmark.png`.
Ryzen 5 5500GT (6C/12T) vs RTX 3070, total ms:

| dataset @minsup | GPU | CPU 12T | CPU 1T |
|---|---|---|---|
| retail @0.0005 | 134 | 681 | 1963 |
| T10I4D100K @0.002 | 61 | 277 | 610 |
| BMS2 @0.0008 | 34 | 43 | 162 |
| accidents @0.3 | 140 | 252 | 420 |
| pumsb @0.6 (19.5M patterns) | 1940 | 1143 | 2801 |
| connect @0.8 | 80 | 46 | 107 |
| chess @0.6 | 39 | 15 | 30 |
| mushrooms @0.1 | 65 | 27 | 66 |

The GPU wins on large sparse data (many transactions, many conditional trees);
small dense inputs are launch/transfer-latency bound and the CPU wins.

## Layout

```
src/        CUDA/C++ sources (cpu_fpgrowth.cpp = CPU baseline)       include/gfpgrowth/   headers
extern/     spdlog, CLI11          datasets/            input data + sources
reference/  Python FPGrowth        scripts/             helpers
```
