# G-FPGrowth

**This is the FP-Growth algorithm implemented on the GPU.**

G-FPGrowth mines all frequent itemsets of a transaction database. Parsing, support
counting, FP-tree construction and mining all run on the GPU (CUDA).

## Usage

```
g-fpgrowth [-m gpu|cpu|cpu1] [-t threads] [-a auto|fpgrowth|eclat] [--sched device|host] [-v] <input> <output> <minsup> [sep]
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

## Algorithms

Two miners, both with the same chunked-memory scheme and both available on CPU and GPU
(`-a auto|fpgrowth|eclat`):

* **FP-Growth** (array FP-tree, conditional trees built by ancestor projection). The CPU
  version additionally short-circuits single-path conditional trees (all subsets are
  enumerated directly) and counts supports with a branch-free reverse sweep.
* **Bitset Eclat** (vertical tid-bitmaps, AND + popcount, AVX2/popcnt on CPU, warp-per-pair
  on GPU). Needs no tree build. `auto` selects it for databases with >= 100k transactions
  over <= 1000 frequent items, FP-Growth otherwise. That rule was measured on the SPMF
  datasets (see below), not derived; override it with `-a`.

### Device-side task queue (GPU FP-Growth)

`--sched device` (default) runs the whole traversal in **one persistent kernel**: no host
round trip per batch. Work lives in *groups* (a published conditional tree whose slots are
the tasks); a task is claimed with a single 64-bit CAS on `depth:16|K:24|next:24`, which also
identifies the group incarnation. Every block statically owns one chunk per recursion level,
so the child of any claimed task always has a chunk (no free list, no deadlock). Idle blocks
steal slots from any live group, shallowest first; an owner reuses a group's chunk only after
all of its slots completed; termination is a global pending-task counter. Patterns go to a
device record buffer (overflow falls back to the host-driven miner).

Because each block needs a full set of chunks, trees with many nodes leave few blocks; if
fewer than half of the resident blocks fit, the host-driven batches are used instead (they
keep many more tasks in flight). Measured mining time, host-driven vs device queue:

| dataset @minsup | host batches | device queue |
|---|---|---|
| pumsb @0.6 | 928 ms | 579 ms |
| mushrooms @0.1 | 27.6 ms | 18.7 ms |
| connect @0.8 | 23.0 ms | 16.5 ms |
| chess @0.6 | 13.3 ms | 9.1 ms |
| retail @0.0005 | 114 ms | 933 ms (not used, too few blocks) |
| T10I4D100K @0.002 | 40 ms | 137 ms (not used, too few blocks) |
| BMS2 @0.0008 | 16 ms | 45 ms (not used, too few blocks) |

The last three rows show what would happen without the guard. Testing: all
scheduler x algorithm combinations match the PAMI reference on the SPMF datasets, and
repeated device-queue runs on dense inputs produce byte-identical (sorted) output.

## Benchmark

`uv run python scripts/benchmark.py` runs GPU / CPU / CPU-1-thread (best of 3, in-process
timings, includes writing the output) and writes `out/benchmark.csv` and `out/benchmark.png`.
Ryzen 5 5500GT (6C/12T) vs RTX 3070, total ms, `-a auto`:

| dataset @minsup | GPU | CPU 12T | CPU 1T |
|---|---|---|---|
| retail @0.0005 | 147 | 793 | 2056 |
| retail @0.002 | 37 | 69 | 207 |
| T10I4D100K @0.002 | 31 | 52 | 170 |
| BMS2 @0.0008 | 36 | 48 | 166 |
| accidents @0.3 | 135 | 129 | 488 |
| connect @0.8 | 82 | 45 | 101 |
| pumsb @0.6 (19.5M patterns) | 2174 | 1099 | 2793 |
| mushrooms @0.1 | 69 | 26 | 57 |
| chess @0.6 | 38 | 15 | 30 |

The GPU wins on large sparse data (many transactions, many conditional trees); on small or
dense inputs it is bound by fixed costs (CUDA context, transfers, per-batch host round
trips) or by output formatting, and the multi-threaded CPU wins.

### Ideas not done yet
* smaller chunks for deep recursion levels so the device queue can use more blocks on large trees
* device-side task queue for the bitset (Eclat) miner
* diffsets (dEclat) for dense data, perfect-extension pruning
* sibling merging after pruning in conditional trees
* GPUDirect Storage for very large inputs

## Layout

```
src/        CUDA/C++ sources (cpu_fpgrowth.cpp = CPU baseline)       include/gfpgrowth/   headers
extern/     spdlog, CLI11          datasets/            input data + sources
reference/  Python FPGrowth        scripts/             helpers
```
