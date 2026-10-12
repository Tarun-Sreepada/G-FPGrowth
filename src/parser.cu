#include <cstdio>
#include <stdexcept>

#include "gfpgrowth/cuda_utils.hpp"
#include "gfpgrowth/log.hpp"
#include "gfpgrowth/parser.cuh"
#include "gfpgrowth/primitives.cuh"

namespace gfp {
namespace {

__device__ __forceinline__ bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r'; }

// Tokenises bytes [s, e) (e is the '\n'): tokens are separated by `sep`, surrounding
// whitespace is trimmed and empty tokens are skipped. Returns the token count; when
// WRITE, token values are stored to out[]. Non-digit / overflowing tokens raise *err.
template <bool WRITE>
__device__ int tokenize(const char* b, int64_t s, int64_t e, char sep, int32_t* out, int* err) {
    int n = 0;
    int64_t i = s;
    while (i <= e) {
        int64_t j = i;
        while (j < e && b[j] != sep) ++j;
        int64_t a = i, z = j;
        while (a < z && is_ws(b[a])) ++a;
        while (z > a && is_ws(b[z - 1])) --z;
        if (z > a) {
            int64_t v = 0;
            bool ok = true;
            for (int64_t p = a; p < z; ++p) {
                int d = b[p] - '0';
                if (d < 0 || d > 9) { ok = false; break; }
                v = v * 10 + d;
                if (v > 2147483647LL) { ok = false; break; }
            }
            if (!ok) { atomic_store(err, 1); v = 0; }
            if (WRITE) out[n] = static_cast<int32_t>(v);
            ++n;
        }
        i = j + 1;
    }
    return n;
}

__global__ void k_count_newlines(const char* b, int64_t n, unsigned long long* total) {
    using Reduce = cub::BlockReduce<unsigned long long, kBlock>;
    __shared__ typename Reduce::TempStorage ts;
    unsigned long long local = 0;
    GRID_STRIDE(i, n) local += (b[i] == '\n');
    unsigned long long sum = Reduce(ts).Sum(local);
    if (threadIdx.x == 0 && sum) atomic_add(total, sum);
}

__global__ void k_mark_newlines(const char* b, int64_t n, int64_t* out, unsigned long long* cursor) {
    GRID_STRIDE(i, n) {
        if (b[i] == '\n') out[atomic_add(cursor, 1ULL)] = i;
    }
}

__global__ void k_count_tokens(const char* b, const int64_t* nl, int64_t nLines, char sep, int64_t* cnt, int* err) {
    GRID_STRIDE(l, nLines) {
        int64_t s = l == 0 ? 0 : nl[l - 1] + 1;
        cnt[l] = tokenize<false>(b, s, nl[l], sep, nullptr, err);
    }
}

__global__ void k_write_tokens(const char* b, const int64_t* nl, int64_t nLines, char sep, const int64_t* off,
                               int32_t* items, int* err) {
    GRID_STRIDE(l, nLines) {
        int64_t s = l == 0 ? 0 : nl[l - 1] + 1;
        tokenize<true>(b, s, nl[l], sep, items + off[l], err);
    }
}

__global__ void k_make_keys(const int32_t* items, const int64_t* off, int64_t nLines, uint64_t* keys) {
    GRID_STRIDE(l, nLines) {
        for (int64_t p = off[l]; p < off[l + 1]; ++p) keys[p] = ((uint64_t)l << 32) | (uint32_t)items[p];
    }
}

}  // namespace

ParsedDB parse_file_on_gpu(const std::string& path, char sep) {
    ParsedDB db;
    StageTimer timer;

    // --- disk -> pinned host -> GPU -------------------------------------------------
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); return db; }
    size_t bytes = static_cast<size_t>(sz);
    db.fileBytes = bytes;

    char* h = nullptr;
    GFP_CUDA(cudaMallocHost(&h, bytes + 1));
    if (std::fread(h, 1, bytes, f) != bytes) { std::fclose(f); cudaFreeHost(h); throw std::runtime_error("short read"); }
    std::fclose(f);
    size_t n = bytes;
    if (h[n - 1] != '\n') h[n++] = '\n';  // virtual terminator for the last line

    DeviceBuffer<char> buf(n);
    GFP_CUDA(cudaMemcpy(buf.data(), h, n, cudaMemcpyHostToDevice));
    cudaFreeHost(h);
    log_debug("file copied to GPU: %zu bytes in %.2f ms", n, timer.elapsed_ms());

    // --- 1. count new lines ----------------------------------------------------------
    DeviceBuffer<unsigned long long> scalar(1);
    scalar.zero();
    k_count_newlines<<<grid_for((int64_t)n), kBlock>>>(buf.data(), (int64_t)n, scalar.data());
    GFP_CUDA(cudaGetLastError());
    const int64_t nLines = (int64_t)scalar.get(0);
    db.numTransactions = (size_t)nLines;

    // --- 2. allocate, record and sort the newline positions --------------------------
    DeviceBuffer<int64_t> nl(nLines);
    scalar.zero();
    k_mark_newlines<<<grid_for((int64_t)n), kBlock>>>(buf.data(), (int64_t)n, nl.data(), scalar.data());
    GFP_CUDA(cudaGetLastError());
    radix_sort(nl, (size_t)nLines, 63);  // atomics scatter positions out of order

    // --- 3. tokens per line, 4. offsets ---------------------------------------------
    DeviceBuffer<int> err(1);
    err.zero();
    DeviceBuffer<int64_t> off(nLines + 1);  // counts, then in-place exclusive scan -> offsets
    off.set(nLines, 0);
    k_count_tokens<<<grid_for(nLines), kBlock>>>(buf.data(), nl.data(), nLines, sep, off.data(), err.data());
    GFP_CUDA(cudaGetLastError());
    exclusive_sum(off.data(), off.data(), (size_t)nLines + 1);
    const int64_t total = off.get(nLines);

    // --- 5. ASCII -> int ------------------------------------------------------------
    DeviceBuffer<int32_t> items(total);
    k_write_tokens<<<grid_for(nLines), kBlock>>>(buf.data(), nl.data(), nLines, sep, off.data(), items.data(),
                                                 err.data());
    GFP_CUDA(cudaGetLastError());
    if (err.get(0)) throw std::runtime_error("input contains a token that is not a non-negative 31-bit integer");
    buf.free();
    nl.free();

    // --- (txn, item) keys, sorted + de-duplicated (itemset semantics) ----------------
    db.keys.resize(total);
    k_make_keys<<<grid_for(nLines), kBlock>>>(items.data(), off.data(), nLines, db.keys.data());
    GFP_CUDA(cudaGetLastError());
    items.free();
    off.free();
    radix_sort(db.keys, (size_t)total);
    db.numKeys = unique_inplace(db.keys, (size_t)total);

    log_info("parsed on GPU: %zu bytes, %lld transactions, %lld item occurrences in %.2f ms", bytes,
             (long long)nLines, (long long)total, timer.elapsed_ms());
    return db;
}

}  // namespace gfp
