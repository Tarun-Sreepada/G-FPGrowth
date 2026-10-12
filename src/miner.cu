
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "gfpgrowth/cuda_utils.hpp"
#include "gfpgrowth/log.hpp"
#include "gfpgrowth/miner.cuh"
#include "gfpgrowth/primitives.cuh"

namespace gfp {
namespace {

constexpr int kBS = kBlock;
constexpr size_t kMaxTasksPerBatch = 4096;
constexpr size_t kMaxChunks = 1u << 15;

// ---------------------------------------------------------------------------------------
// Chunk layout. A chunk holds one (conditional) FP-tree plus the scratch its builder
// needs. N = node capacity (= node count of the initial tree), K = slot capacity.
//   item/count/parent : the tree (node 0 = root, nodes grouped by slot)
//   tabItem/tabTot    : per slot, the item rank and its support in this tree
//   tabEnd            : per slot, end of its node range (start = previous end, or 1)
//   c/newid/slotmap   : scratch used while this chunk is being built
// ---------------------------------------------------------------------------------------
struct Layout {
    int64_t N, K;
    __host__ __device__ int64_t stride() const { return 5 * N + 4 * K + 1; }
};

struct View {
    int32_t *item, *count, *parent, *c, *newid, *tabItem, *tabEnd, *tabTot, *slotmap;
};

__host__ __device__ inline View view_of(int32_t* pool, int64_t chunk, const Layout& L) {
    int32_t* b = pool + chunk * L.stride();
    View v;
    v.item = b;
    v.count = b + L.N;
    v.parent = b + 2 * L.N;
    v.c = b + 3 * L.N;
    v.newid = b + 4 * L.N;
    v.tabItem = b + 5 * L.N;
    v.tabEnd = b + 5 * L.N + L.K;           // K+1 entries
    v.tabTot = b + 5 * L.N + 2 * L.K + 1;
    v.slotmap = b + 5 * L.N + 3 * L.K + 1;
    return v;
}

// In-place exclusive scan of a[0,n) by one block; returns the total to every thread.
__device__ int block_exclusive_scan(int32_t* a, int n) {
    using Scan = cub::BlockScan<int, kBS>;
    __shared__ typename Scan::TempStorage ts;
    __shared__ int carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += kBS) {
        int idx = base + threadIdx.x;
        int v = idx < n ? a[idx] : 0, ex, agg;
        Scan(ts).ExclusiveSum(v, ex, agg);
        if (idx < n) a[idx] = ex + carry;
        __syncthreads();
        if (threadIdx.x == 0) carry += agg;
        __syncthreads();
    }
    return carry;
}

// One thread block = one task: "append slot j of tree T to T's pattern and build the
// conditional FP-tree of that pattern into chunk C".
//
// Nodes of slot j are the leaves of the conditional pattern base. Because nodes are
// grouped by ascending slot, every ancestor of a slot-j node lives at an index below
// start(j), and only items of slot < j can occur in the conditional tree.
//   1. c[a]  = sum of counts of the slot-j nodes below a        (walk up, atomics)
//   2. per-slot totals of c; keep slots with total >= minsup
//   3. keep node a iff c[a] > 0 and its slot is kept; counting-sort kept nodes by
//      new slot into C; re-parent to the nearest kept ancestor.
__global__ void k_expand(int32_t* pool, Layout L, const int32_t* tChunk, const int32_t* tSlot, const int32_t* tPar,
                         const int32_t* cChunk, int32_t minsup, int32_t* recPar, int32_t* recItem, int32_t* recSup,
                         int32_t* outK) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const int j = tSlot[t];
    const View T = view_of(pool, tChunk[t], L);
    const View C = view_of(pool, cChunk[t], L);
    const int s = j > 0 ? T.tabEnd[j - 1] : 1;  // first node of slot j == bound for ancestors
    const int e = T.tabEnd[j];

    if (tid == 0) {
        recPar[t] = tPar[t];
        recItem[t] = T.tabItem[j];
        recSup[t] = T.tabTot[j];
    }
    for (int a = tid; a < s; a += kBS) { C.c[a] = 0; C.newid[a] = -1; }
    for (int k = tid; k < j; k += kBS) C.slotmap[k] = 0;
    __syncthreads();

    for (int h = s + tid; h < e; h += kBS) {
        const int v = T.count[h];
        for (int a = T.parent[h]; a > 0; a = T.parent[a]) atomic_add<int32_t>(&C.c[a], v);
    }
    __syncthreads();

    for (int a = 1 + tid; a < s; a += kBS) {
        const int cv = C.c[a];
        if (cv > 0) atomic_add<int32_t>(&C.slotmap[T.item[a]], cv);
    }
    __syncthreads();

    for (int k = tid; k < j; k += kBS) C.tabEnd[k] = C.slotmap[k] >= minsup ? 1 : 0;
    __syncthreads();
    const int KC = block_exclusive_scan(C.tabEnd, j);
    __syncthreads();

    for (int k = tid; k < j; k += kBS) {
        const int tot = C.slotmap[k];
        if (tot >= minsup) {
            const int ns = C.tabEnd[k];
            C.tabItem[ns] = T.tabItem[k];
            C.tabTot[ns] = tot;
            C.slotmap[k] = ns;
        } else {
            C.slotmap[k] = -1;
        }
    }
    __syncthreads();

    for (int k = tid; k <= KC; k += kBS) C.tabEnd[k] = 0;
    __syncthreads();

    for (int a = 1 + tid; a < s; a += kBS) {
        if (C.c[a] > 0) {
            const int ns = C.slotmap[T.item[a]];
            if (ns >= 0) atomic_add<int32_t>(&C.tabEnd[ns], 1);
        }
    }
    __syncthreads();
    const int kept = block_exclusive_scan(C.tabEnd, KC);
    __syncthreads();
    for (int k = tid; k < KC; k += kBS) C.tabEnd[k] += 1;  // node 0 is the root
    __syncthreads();

    for (int a = 1 + tid; a < s; a += kBS) {
        const int cv = C.c[a];
        if (cv > 0) {
            const int ns = C.slotmap[T.item[a]];
            if (ns >= 0) {
                const int d = atomic_add<int32_t>(&C.tabEnd[ns], 1);  // ends up as the group's end
                C.item[d] = ns;
                C.count[d] = cv;
                C.parent[d] = T.parent[a];  // old id, fixed below
                C.newid[a] = d;
            }
        }
    }
    if (tid == 0) {
        C.item[0] = -1;
        C.count[0] = 0;
        C.parent[0] = -1;
        outK[t] = KC;
    }
    __syncthreads();

    for (int d = 1 + tid; d <= kept; d += kBS) {
        int p = C.parent[d];
        while (p > 0 && C.newid[p] < 0) p = T.parent[p];  // skip pruned items
        C.parent[d] = p > 0 ? C.newid[p] : 0;
    }
}

struct Group {
    int chunk;
    int K;
    int next;
    int parentRec;
};

}  // namespace

MiningResult mine_patterns(InitialTree& tree, int minsup) {
    StageTimer timer;
    MiningResult res;
    const int F = tree.numFrequent;
    if (F == 0) return res;

    const int64_t N = tree.numNodes;
    Layout L{N, std::min<int64_t>(F, N)};
    const size_t chunkInts = (size_t)L.stride();
    res.chunkBytes = chunkInts * sizeof(int32_t);

    size_t freeB = 0, totalB = 0;
    GFP_CUDA(cudaMemGetInfo(&freeB, &totalB));
    size_t C = std::min<size_t>((size_t)(freeB * 0.85) / res.chunkBytes, kMaxChunks);
    if (C < 2) {
        throw std::runtime_error("GPU memory too small: the initial FP-tree needs " + std::to_string(res.chunkBytes) +
                                 " bytes per chunk and at least 2 chunks");
    }
    res.chunkCount = C;
    int32_t* pool = nullptr;
    StageTimer allocTimer;
    GFP_CUDA(cudaMalloc(&pool, C * res.chunkBytes));
    log_debug("pool allocation: %.2f ms", allocTimer.elapsed_ms());
    log_info("GPU memory partitioned into %zu chunks of %.2f MiB (initial tree: %lld nodes, %d slots)", C,
             res.chunkBytes / 1048576.0, (long long)N, F);

    // chunk 0 <- initial tree
    {
        View v = view_of(pool, 0, L);
        GFP_CUDA(cudaMemcpy(v.item, tree.item.data(), N * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.count, tree.count.data(), N * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.parent, tree.parent.data(), N * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.tabItem, tree.tabItem.data(), F * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.tabEnd, tree.tabEnd.data(), F * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.tabTot, tree.tabTot.data(), F * 4, cudaMemcpyDeviceToDevice));
    }
    tree.item.free(); tree.count.free(); tree.parent.free();
    tree.tabItem.free(); tree.tabEnd.free(); tree.tabTot.free();

    // Chunks that must stay free so that a plain depth-first descent can always finish:
    // the current stack holds at most (longest pattern) trees.
    const size_t reserve = (size_t)std::max(tree.maxPatternLen, 1) + 1;
    const size_t P = std::min(kMaxTasksPerBatch, C);

    // One packed host<->device transfer per batch: [4 arrays of n ints] in, [4 arrays of n ints] out.
    DeviceBuffer<int32_t> dIn(4 * P), dOut(4 * P);
    int32_t *hIn = nullptr, *hOut = nullptr;
    GFP_CUDA(cudaMallocHost(&hIn, 4 * P * sizeof(int32_t)));
    GFP_CUDA(cudaMallocHost(&hOut, 4 * P * sizeof(int32_t)));
    std::vector<int32_t> hTChunk, hTSlot, hTPar, hCChunk;

    std::vector<int> freeList;
    for (int c = (int)C - 1; c >= 1; --c) freeList.push_back(c);
    std::vector<Group> stack{{0, F, 0, -1}};
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    float kernelMs = 0; size_t totalTasks = 0;
    double tGather = 0, tH2D = 0, tWait = 0, tPost = 0;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto dms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

    while (true) {
        auto q0 = now();
        // retire trees whose slots have all been expanded
        {
            size_t w = 0;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (stack[i].next >= stack[i].K) freeList.push_back(stack[i].chunk);
                else stack[w++] = stack[i];
            }
            stack.resize(w);
        }
        if (stack.empty()) break;
        if (freeList.empty()) {
            throw std::runtime_error("out of GPU chunks; use a larger minimum support or a GPU with more memory");
        }

        // Batch size: use spare chunks for breadth, but never dip into the reserve.
        const size_t avail = freeList.size() > reserve ? freeList.size() - reserve : 1;
        const size_t b = std::min(P, avail);

        hTChunk.clear(); hTSlot.clear(); hTPar.clear(); hCChunk.clear();
        const int recBase = (int)res.item.size();
        for (size_t gi = stack.size(); gi-- > 0 && hTSlot.size() < b;) {
            Group& g = stack[gi];
            while (g.next < g.K && hTSlot.size() < b) {
                hTChunk.push_back(g.chunk);
                hTSlot.push_back(g.next++);
                hTPar.push_back(g.parentRec);
                hCChunk.push_back(freeList.back());
                freeList.pop_back();
            }
        }
        const size_t n = hTSlot.size();
        auto q1 = now();
        std::copy(hTChunk.begin(), hTChunk.end(), hIn);
        std::copy(hTSlot.begin(), hTSlot.end(), hIn + n);
        std::copy(hTPar.begin(), hTPar.end(), hIn + 2 * n);
        std::copy(hCChunk.begin(), hCChunk.end(), hIn + 3 * n);
        GFP_CUDA(cudaMemcpy(dIn.data(), hIn, 4 * n * sizeof(int32_t), cudaMemcpyHostToDevice));
        auto q2 = now();
        cudaEventRecord(ev0);
        k_expand<<<(unsigned)n, kBS>>>(pool, L, dIn.data(), dIn.data() + n, dIn.data() + 2 * n, dIn.data() + 3 * n,
                                       minsup, dOut.data(), dOut.data() + n, dOut.data() + 2 * n,
                                       dOut.data() + 3 * n);
        cudaEventRecord(ev1);
        GFP_CUDA(cudaGetLastError());
        GFP_CUDA(cudaMemcpy(hOut, dOut.data(), 4 * n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        auto q3 = now();
        { float ms; cudaEventElapsedTime(&ms, ev0, ev1); kernelMs += ms; totalTasks += n; }
        ++res.batches;
        const int32_t* hK = hOut + 3 * n;

        res.parent.insert(res.parent.end(), hOut, hOut + n);
        res.item.insert(res.item.end(), hOut + n, hOut + 2 * n);
        res.support.insert(res.support.end(), hOut + 2 * n, hOut + 3 * n);
        for (size_t t = 0; t < n; ++t) {
            if (hK[t] > 0) stack.push_back({hCChunk[t], hK[t], 0, recBase + (int)t});
            else freeList.push_back(hCChunk[t]);
        }
        auto q4 = now();
        tGather += dms(q0, q1); tH2D += dms(q1, q2); tWait += dms(q2, q3); tPost += dms(q3, q4);
    }
    cudaFreeHost(hIn);
    cudaFreeHost(hOut);
    cudaFree(pool);
    log_debug("host: gather %.1f ms, h2d %.1f ms, kernel+d2h %.1f ms, post %.1f ms", tGather, tH2D, tWait, tPost);
    log_debug("expand kernels: %.2f ms for %zu tasks (%.1f us/task, avg batch %.0f)", kernelMs, totalTasks, 1000.0 * kernelMs / (totalTasks ? totalTasks : 1), (double)totalTasks / (res.batches ? res.batches : 1));
    log_info("mined %zu frequent itemsets in %zu batches, %.2f ms", res.item.size(), res.batches, timer.elapsed_ms());
    return res;
}

}  // namespace gfp
