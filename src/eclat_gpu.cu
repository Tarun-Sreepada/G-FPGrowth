#include <algorithm>
#include <mma.h>
#include <stdexcept>
#include <string>

#include "gfpgrowth/cuda_utils.hpp"
#include "gfpgrowth/log.hpp"
#include "gfpgrowth/miner.cuh"
#include "gfpgrowth/primitives.cuh"

namespace gfp {
namespace {

constexpr size_t kMaxTasksPerBatch = 4096;
constexpr size_t kMaxChunks = 1u << 15;
constexpr int kMmaThreshold = 64;
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;

// A chunk holds one class: up to F items, each a bitset of W 64-bit words (transactions),
// plus per-item id/support and two scratch int arrays used while the class is built.
struct ELayout {
    int64_t F, W;
    __host__ __device__ size_t stride() const { return (size_t)F * (size_t)W + 2 * (size_t)F; }  // uint64 words
};

struct EView {
    uint64_t* bits;
    int32_t *item, *sup, *s1, *s2;
};

__host__ __device__ inline EView eview(uint64_t* pool, int64_t chunk, const ELayout& L) {
    uint64_t* base = pool + (size_t)chunk * L.stride();
    EView v;
    v.bits = base;
    int32_t* ints = reinterpret_cast<int32_t*>(base + (size_t)L.F * (size_t)L.W);
    v.item = ints;
    v.sup = ints + L.F;
    v.s1 = ints + 2 * L.F;
    v.s2 = ints + 3 * L.F;
    return v;
}

__global__ void k_bitmap(const uint64_t* keys, int64_t M, int F, int64_t W, uint64_t* bits) {
    GRID_STRIDE(i, M) {
        const uint64_t k = keys[i];
        const uint64_t txn = k >> 32;
        const int64_t row = F - 1 - (int64_t)(k & 0xffffffffu);  // least frequent item first
        atomic_or<uint64_t>(&bits[row * W + (txn >> 6)], 1ULL << (txn & 63));
    }
}

__global__ void k_fill_root(uint64_t* pool, ELayout L, const int32_t* support, int F) {
    const EView R = eview(pool, 0, L);
    GRID_STRIDE(p, F) {
        const int rank = F - 1 - (int)p;
        R.item[p] = rank;
        R.sup[p] = support[rank];
    }
}

// 1D triangular block mapping: blockIdx.x -> (tile_i, tile_j) with tile_i <= tile_j
__device__ inline void map_triangular_id(int bid, int num_tiles, int& tile_i, int& tile_j) {
    int i = 0, rem = bid;
    while (rem >= (num_tiles - i)) {
        rem -= (num_tiles - i);
        i++;
    }
    tile_i = i;
    tile_j = i + rem;
}

// Binary Tensor Core (b1 MMA) kernel for upper-triangular Gram matrix: G[i, j] = popcount(T[i] & T[j])
__global__ void k_wmma_triangle(const uint32_t* __restrict__ bits32, int M, int W32, int num_tiles, int* __restrict__ out_G) {
    int tile_i, tile_j;
    map_triangular_id(blockIdx.x, num_tiles, tile_i, tile_j);

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;  // 8 warps in a 256-thread block

    __shared__ uint32_t smemA[TILE_M][4];
    __shared__ uint32_t smemB[TILE_N][4];

    // 4x2 grid of warps covering 64x64 tile; each warp computes 16x32 elements
    const int warp_row = (warp_id >> 1) * 16;
    const int warp_col = (warp_id & 1) * 32;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 128, int> acc[2][4];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            nvcuda::wmma::fill_fragment(acc[r][c], 0);
        }
    }

    const int row_base = tile_i * TILE_M;
    const int col_base = tile_j * TILE_N;

    for (int k_base = 0; k_base < W32; k_base += 4) {
        int r = tid / 4;
        int w = tid % 4;

        int global_r_A = row_base + r;
        smemA[r][w] = (global_r_A < M && (k_base + w) < W32) ? bits32[(size_t)global_r_A * W32 + k_base + w] : 0;

        int global_r_B = col_base + r;
        smemB[r][w] = (global_r_B < M && (k_base + w) < W32) ? bits32[(size_t)global_r_B * W32 + k_base + w] : 0;

        __syncthreads();

        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::load_matrix_sync(a_frag, &smemA[warp_row + r * 8][0], 128);

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::col_major> b_frag;
                nvcuda::wmma::load_matrix_sync(b_frag, &smemB[warp_col + c * 8][0], 128);

                nvcuda::wmma::bmma_sync(acc[r][c], a_frag, b_frag, acc[r][c], nvcuda::wmma::experimental::bmmaBitOpAND);
            }
        }
        __syncthreads();
    }

    __shared__ int smemC[TILE_M][TILE_N];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            nvcuda::wmma::store_matrix_sync(&smemC[warp_row + r * 8][warp_col + c * 8], acc[r][c], TILE_N, nvcuda::wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < TILE_M * TILE_N; idx += blockDim.x) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        int global_i = row_base + r;
        int global_j = col_base + c;
        if (global_i < M && global_j < M && global_j > global_i) {
            out_G[(size_t)global_i * M + global_j] = smemC[r][c];
        }
    }
}

// Task: extend pattern with item i of class T; child class C = { j > i : |T[i] & T[j]| >= minsup }.
// in : [chunk | slot | parentRec | childChunk | m] * n ; out: [parent | item | support | childM] * n
__global__ void k_eclat(uint64_t* pool, ELayout L, const int32_t* in, int n, int minsup, const int32_t* root_G, int32_t* out) {
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    constexpr int kWarps = kBlock / 32;
    const int i = in[n + t];
    const int m = in[4 * n + t];
    const EView T = eview(pool, in[t], L);
    const EView C = eview(pool, in[3 * n + t], L);
    const int cap = m - i - 1;
    const int64_t W = L.W;
    if (tid == 0) {
        out[t] = in[2 * n + t];
        out[n + t] = T.item[i];
        out[2 * n + t] = T.sup[i];
    }
    if (in[t] == 0 && root_G != nullptr) {
        const int32_t* g_row = root_G + (size_t)i * L.F + (i + 1);
        for (int idx = warp; idx < cap; idx += kWarps) {
            int s = g_row[idx];
            if (lane == 0) { C.s1[idx] = s; C.s2[idx] = s >= minsup ? 1 : 0; }
        }
    } else {
        const uint64_t* a = T.bits + (size_t)i * W;
        for (int idx = warp; idx < cap; idx += kWarps) {
            const uint64_t* b = T.bits + (size_t)(i + 1 + idx) * W;
            int s = 0;
            for (int64_t w = lane; w < W; w += 32) s += __popcll(a[w] & b[w]);
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0) { C.s1[idx] = s; C.s2[idx] = s >= minsup ? 1 : 0; }
        }
    }
    __syncthreads();
    const int kept = block_exclusive_scan(C.s2, cap);
    __syncthreads();
    const uint64_t* a = T.bits + (size_t)i * W;
    for (int idx = warp; idx < cap; idx += kWarps) {
        const int s = C.s1[idx];
        if (s < minsup) continue;
        const int pos = C.s2[idx];
        const uint64_t* b = T.bits + (size_t)(i + 1 + idx) * W;
        uint64_t* d = C.bits + (size_t)pos * W;
        if (lane == 0) { C.item[pos] = T.item[i + 1 + idx]; C.sup[pos] = s; }
        for (int64_t w = lane; w < W; w += 32) d[w] = a[w] & b[w];
    }
    if (tid == 0) out[3 * n + t] = kept;
}

struct Group {
    int chunk, m, next, parentRec;
};

}  // namespace

bool mine_eclat(RankedDB& db, int minsup, MiningResult& res) {
    StageTimer timer;
    const int F = db.numFrequent;
    if (F == 0) return true;
    const int64_t W_raw = (int64_t)((db.numTransactions + 63) / 64);
    const int64_t W = (W_raw + 3) / 4 * 4;  // align to multiple of 4 words (256 bits) for Tensor Core MMA
    ELayout L{F, W};
    const size_t chunkWords = L.stride();
    const size_t chunkBytes = chunkWords * 8;
    const size_t reserve = (size_t)std::max(std::min(db.maxPatternLen, F), 1) + 1;

    size_t freeB = 0, totalB = 0;
    GFP_CUDA(cudaMemGetInfo(&freeB, &totalB));
    const size_t C = std::min<size_t>((size_t)(freeB * 0.85) / chunkBytes, kMaxChunks);
    if (C < reserve + 2) {
        log_info("bitset miner needs %zu chunks of %.1f MiB but only %zu fit", reserve + 2, chunkBytes / 1048576.0, C);
        return false;
    }
    uint64_t* pool = nullptr;
    GFP_CUDA(cudaMalloc(&pool, C * chunkBytes));
    log_info("GPU memory partitioned into %zu bitset chunks of %.2f MiB (%d items x %lld words)", C,
             chunkBytes / 1048576.0, F, (long long)W);

    // chunk 0: the root class
    {
        EView R = eview(pool, 0, L);
        GFP_CUDA(cudaMemset(R.bits, 0, (size_t)F * W * 8));
        k_bitmap<<<grid_for((int64_t)db.numKeys), kBlock>>>(db.keys.data(), (int64_t)db.numKeys, F, W, R.bits);
        GFP_CUDA(cudaGetLastError());
        DeviceBuffer<int32_t> sup(F);
        GFP_CUDA(cudaMemcpy(sup.data(), db.support.data(), F * 4, cudaMemcpyHostToDevice));
        k_fill_root<<<grid_for(F), kBlock>>>(pool, L, sup.data(), F);
        GFP_CUDA(cudaGetLastError());
    }
    db.keys.free();

    // Compute upper-triangular Gram matrix G for root class via Binary Tensor Cores (b1 MMA) if F >= 64
    DeviceBuffer<int32_t> root_G;
    if (F >= kMmaThreshold) {
        root_G.resize((size_t)F * F);
        int num_tiles = (F + TILE_M - 1) / TILE_M;
        int total_tiles = num_tiles * (num_tiles + 1) / 2;
        int W32 = (int)W * 2;
        EView R = eview(pool, 0, L);
        k_wmma_triangle<<<total_tiles, kBlock>>>((const uint32_t*)R.bits, F, W32, num_tiles, root_G.data());
        GFP_CUDA(cudaGetLastError());
    }

    const size_t P = std::min(kMaxTasksPerBatch, C);
    DeviceBuffer<int32_t> dIn(5 * P), dOut(4 * P);
    int32_t *hIn = nullptr, *hOut = nullptr;
    GFP_CUDA(cudaMallocHost(&hIn, 5 * P * sizeof(int32_t)));
    GFP_CUDA(cudaMallocHost(&hOut, 4 * P * sizeof(int32_t)));
    std::vector<int32_t> tChunk, tSlot, tPar, cChunk, tM;

    std::vector<int> freeList;
    for (int c = (int)C - 1; c >= 1; --c) freeList.push_back(c);
    std::vector<Group> stack{{0, F, 0, -1}};

    while (true) {
        {
            size_t w = 0;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (stack[i].next >= stack[i].m) {
                    if (stack[i].chunk != 0) freeList.push_back(stack[i].chunk);
                } else {
                    stack[w++] = stack[i];
                }
            }
            stack.resize(w);
        }
        if (stack.empty()) break;
        if (freeList.empty()) {
            cudaFreeHost(hIn); cudaFreeHost(hOut); cudaFree(pool);
            throw std::runtime_error("out of GPU chunks (bitset miner)");
        }
        const size_t avail = freeList.size() > reserve ? freeList.size() - reserve : 1;
        const size_t b = std::min(P, avail);

        tChunk.clear(); tSlot.clear(); tPar.clear(); cChunk.clear(); tM.clear();
        const int recBase = (int)res.item.size();
        for (size_t gi = stack.size(); gi-- > 0 && tSlot.size() < b;) {
            Group& g = stack[gi];
            while (g.next < g.m && tSlot.size() < b) {
                tChunk.push_back(g.chunk);
                tSlot.push_back(g.next++);
                tPar.push_back(g.parentRec);
                cChunk.push_back(freeList.back());
                freeList.pop_back();
                tM.push_back(g.m);
            }
        }
        const size_t n = tSlot.size();
        std::copy(tChunk.begin(), tChunk.end(), hIn);
        std::copy(tSlot.begin(), tSlot.end(), hIn + n);
        std::copy(tPar.begin(), tPar.end(), hIn + 2 * n);
        std::copy(cChunk.begin(), cChunk.end(), hIn + 3 * n);
        std::copy(tM.begin(), tM.end(), hIn + 4 * n);
        GFP_CUDA(cudaMemcpy(dIn.data(), hIn, 5 * n * 4, cudaMemcpyHostToDevice));
        k_eclat<<<(unsigned)n, kBlock>>>(pool, L, dIn.data(), (int)n, minsup, root_G.data(), dOut.data());
        GFP_CUDA(cudaGetLastError());
        GFP_CUDA(cudaMemcpy(hOut, dOut.data(), 4 * n * 4, cudaMemcpyDeviceToHost));
        ++res.batches;

        res.parent.insert(res.parent.end(), hOut, hOut + n);
        res.item.insert(res.item.end(), hOut + n, hOut + 2 * n);
        res.support.insert(res.support.end(), hOut + 2 * n, hOut + 3 * n);
        for (size_t t = 0; t < n; ++t) {
            const int m = hOut[3 * n + t];
            if (m > 0) stack.push_back({cChunk[t], m, 0, recBase + (int)t});
            else freeList.push_back(cChunk[t]);
        }
    }
    cudaFreeHost(hIn);
    cudaFreeHost(hOut);
    cudaFree(pool);
    res.chunkCount = C;
    res.chunkBytes = chunkBytes;
    log_info("mined %zu frequent itemsets in %zu batches, %.2f ms", res.item.size(), res.batches, timer.elapsed_ms());
    return true;
}

}  // namespace gfp
