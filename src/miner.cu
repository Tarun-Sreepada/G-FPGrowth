
#include <algorithm>
#include <climits>
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
// Builds the conditional tree of slot j of chunk tChunk into chunk cChunk. Executed by one
// whole thread block; returns the number of slots of the new tree (uniform across threads).
__device__ int expand_task(int32_t* pool, const Layout& L, int tChunk, int j, int cChunk, int32_t minsup) {
    const int tid = threadIdx.x;
    const View T = view_of(pool, tChunk, L);
    const View C = view_of(pool, cChunk, L);
    const int s = j > 0 ? T.tabEnd[j - 1] : 1;  // first node of slot j == bound for ancestors
    const int e = T.tabEnd[j];

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
    }
    __syncthreads();

    for (int d = 1 + tid; d <= kept; d += kBS) {
        int p = C.parent[d];
        while (p > 0 && C.newid[p] < 0) p = T.parent[p];  // skip pruned items
        C.parent[d] = p > 0 ? C.newid[p] : 0;
    }
    __syncthreads();  // the new tree is complete (and the source tree no longer read)
    return KC;
}

// Host-driven batches: one block per task.
__global__ void k_expand(int32_t* pool, Layout L, const int32_t* tChunk, const int32_t* tSlot, const int32_t* tPar,
                         const int32_t* cChunk, int32_t minsup, int32_t* recPar, int32_t* recItem, int32_t* recSup,
                         int32_t* outK) {
    const int t = blockIdx.x;
    const int j = tSlot[t];
    if (threadIdx.x == 0) {
        const View T = view_of(pool, tChunk[t], L);
        recPar[t] = tPar[t];
        recItem[t] = T.tabItem[j];
        recSup[t] = T.tabTot[j];
    }
    const int KC = expand_task(pool, L, tChunk[t], j, cChunk[t], minsup);
    if (threadIdx.x == 0) outK[t] = KC;
}

// ---------------------------------------------------------------------------------------
// Device-side task queue (persistent kernel).
//
// Work lives in "groups": a published tree whose slots are the tasks. Each group has one
// 64-bit state word  depth:16 | K:24 | next:24  so a task is claimed with a single CAS that
// also validates which incarnation of the group it belongs to. Each block statically owns
// LV chunks (one per recursion level), so a block can always build the child of any task it
// claims into its own chunk of level depth+1: no free list, no deadlock. Idle blocks steal
// slots from any live group (shallowest first); the owner retires a group (reusing its
// chunk) only after all of its slots have completed.
// ---------------------------------------------------------------------------------------
struct Sched {
    unsigned long long* state;  // per group (group index == chunk id)
    int32_t* parentRec;
    int32_t* done;
    int32_t* pending;           // tasks created but not yet completed
    int32_t* recCount;
    int32_t* overflow;
    int32_t *recPar, *recItem, *recSup;
    int64_t recCap;
    int nGroups, LV;
};

__device__ __forceinline__ unsigned long long pack_state(int depth, int K, int next) {
    return ((unsigned long long)depth << 48) | ((unsigned long long)K << 24) | (unsigned long long)next;
}
__device__ __forceinline__ int st_depth(unsigned long long s) { return (int)(s >> 48); }
__device__ __forceinline__ int st_K(unsigned long long s) { return (int)((s >> 24) & 0xFFFFFF); }
__device__ __forceinline__ int st_next(unsigned long long s) { return (int)(s & 0xFFFFFF); }

__device__ __forceinline__ unsigned long long ld_acq(unsigned long long* p) {
    return cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void st_rel(unsigned long long* p, unsigned long long v) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*p).store(v, cuda::memory_order_release);
}
__device__ __forceinline__ bool cas_acq(unsigned long long* p, unsigned long long expected, unsigned long long v) {
    return cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*p).compare_exchange_strong(
        expected, v, cuda::memory_order_acq_rel);
}
__device__ __forceinline__ int ld_i32(int32_t* p) {
    return cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}

__global__ void k_persistent(int32_t* pool, Layout L, Sched S, int32_t minsup) {
    extern __shared__ int stk[];  // levels of this block's live groups, increasing
    __shared__ int sp, sFound, sExit, sG, sJ, sDepth, sCand, sRec;
    const int b = blockIdx.x, tid = threadIdx.x;
    if (tid == 0) sp = 0;
    __syncthreads();

    while (true) {
        __syncthreads();  // everyone is done with last iteration's shared flags
        // ---- own top group: claim a slot, or retire it once every slot completed ------
        if (tid == 0) {
            sFound = 0; sExit = 0; sCand = INT_MAX;
            while (sp > 0) {
                const int g = 1 + b * S.LV + (stk[sp - 1] - 1);
                unsigned long long st = ld_acq(&S.state[g]);
                const int K = st_K(st);
                bool got = false;
                while (st_next(st) < K) {
                    if (cas_acq(&S.state[g], st, st + 1)) { got = true; break; }
                    st = ld_acq(&S.state[g]);
                }
                if (got) { sFound = 1; sG = g; sJ = st_next(st); sDepth = st_depth(st); break; }
                if (ld_i32(&S.done[g]) >= K) { st_rel(&S.state[g], 0ULL); --sp; continue; }
                break;  // slots are being finished by other blocks
            }
        }
        __syncthreads();

        // ---- steal: all threads look for the shallowest claimable group ---------------
        if (!sFound) {
            const int top = sp > 0 ? stk[sp - 1] : 0;
            int best = INT_MAX;
            for (int g = tid; g < S.nGroups; g += blockDim.x) {
                const unsigned long long st = ld_acq(&S.state[g]);
                if (st_next(st) < st_K(st) && st_depth(st) + 1 > top) {
                    const int key = (st_depth(st) << 20) | g;
                    if (key < best) best = key;
                }
            }
            if (best != INT_MAX) atomicMin(&sCand, best);
            __syncthreads();
            if (tid == 0) {
                if (sCand != INT_MAX) {
                    const int g = sCand & 0xFFFFF;
                    unsigned long long st = ld_acq(&S.state[g]);
                    while (st_next(st) < st_K(st) && st_depth(st) + 1 > top) {
                        if (cas_acq(&S.state[g], st, st + 1)) {
                            sFound = 1; sG = g; sJ = st_next(st); sDepth = st_depth(st);
                            break;
                        }
                        st = ld_acq(&S.state[g]);
                    }
                } else if (ld_i32(S.pending) == 0) {
                    sExit = 1;
                } else {
                    __nanosleep(256);
                }
            }
            __syncthreads();
        }
        if (sExit) break;
        if (!sFound) continue;

        // ---- run the claimed task ------------------------------------------------------
        const int g = sG, j = sJ, childLevel = sDepth + 1;
        const int cChunk = 1 + b * S.LV + (childLevel - 1);
        if (tid == 0) {
            sRec = atomicAdd(S.recCount, 1);
            if (childLevel > S.LV) atomicExch(S.overflow, 1);
        }
        __syncthreads();
        if (childLevel > S.LV) {  // cannot happen with LV >= longest pattern + 1
            if (tid == 0) { atomicAdd(&S.done[g], 1); atomicAdd(S.pending, -1); }
            __syncthreads();
            continue;
        }
        if (tid == 0) {
            const View T = view_of(pool, g, L);
            if (sRec < S.recCap) {
                S.recPar[sRec] = ld_i32(&S.parentRec[g]);
                S.recItem[sRec] = T.tabItem[j];
                S.recSup[sRec] = T.tabTot[j];
            } else {
                atomicExch(S.overflow, 1);
            }
        }
        const int KC = expand_task(pool, L, g, j, cChunk, minsup);
        if (tid == 0) {
            if (KC > 0) {
                S.parentRec[cChunk] = sRec;
                S.done[cChunk] = 0;
                atomicAdd(S.pending, KC);
                st_rel(&S.state[cChunk], pack_state(childLevel, KC, 0));  // publishes the tree
                stk[sp++] = childLevel;
            }
            __threadfence();
            atomicAdd(&S.done[g], 1);
            atomicAdd(S.pending, -1);
        }
        __syncthreads();
    }
}

struct Group {
    int chunk;
    int K;
    int next;
    int parentRec;
};

}  // namespace

namespace {
MiningResult mine_host_driven(InitialTree& tree, int minsup) {
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

}  // namespace

namespace {

// Persistent-kernel miner with the device-side task queue. Returns false if it cannot run
// (not enough memory for one chunk set per block, or the record buffer overflowed); the
// source tree is left untouched so the host-driven miner can take over.
bool mine_device_queue(InitialTree& tree, int minsup, MiningResult& res) {
    StageTimer timer;
    const int F = tree.numFrequent;
    const int64_t N = tree.numNodes;
    Layout L{N, std::min<int64_t>(F, N)};
    const size_t chunkBytes = (size_t)L.stride() * sizeof(int32_t);
    const int LV = std::max(std::min(tree.maxPatternLen, F), 1) + 2;
    if (LV >= 65535) return false;

    int dev = 0, numSMs = 0, occ = 0;
    GFP_CUDA(cudaGetDevice(&dev));
    GFP_CUDA(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    GFP_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_persistent, kBS, (size_t)LV * 4));
    if (occ < 1) return false;

    size_t freeB = 0, totalB = 0;
    GFP_CUDA(cudaMemGetInfo(&freeB, &totalB));
    const size_t chunksFit = (size_t)(freeB * 0.6) / chunkBytes;
    if (chunksFit < 1 + (size_t)LV) return false;
    const size_t B = std::min<size_t>((size_t)numSMs * occ, (chunksFit - 1) / LV);
    // Each block owns LV full-size chunks, so big initial trees leave too few blocks to hide
    // latency; the host-driven batches (many more concurrent tasks) are faster then.
    if (B * 2 < (size_t)numSMs * occ) {
        log_info("device task queue: only %zu of %d blocks fit in memory, using host-driven batches", B, numSMs * occ);
        return false;
    }
    const size_t nChunks = 1 + B * LV;
    if (nChunks >= (1u << 20)) return false;

    int32_t* pool = nullptr;
    GFP_CUDA(cudaMalloc(&pool, nChunks * chunkBytes));
    size_t freeAfter = 0;
    GFP_CUDA(cudaMemGetInfo(&freeAfter, &totalB));
    const int64_t recCap = (int64_t)std::min<size_t>((size_t)(freeAfter * 0.8) / 12, (size_t)1 << 28);
    if (recCap < 1024) { cudaFree(pool); return false; }

    DeviceBuffer<unsigned long long> state(nChunks);
    DeviceBuffer<int32_t> parentRec(nChunks), done(nChunks), counters(3);  // pending, recCount, overflow
    DeviceBuffer<int32_t> recPar(recCap), recItem(recCap), recSup(recCap);
    state.zero(); done.zero(); parentRec.zero(); counters.zero();
    {
        View v = view_of(pool, 0, L);
        GFP_CUDA(cudaMemcpy(v.item, tree.item.data(), N * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.count, tree.count.data(), N * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.parent, tree.parent.data(), N * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.tabItem, tree.tabItem.data(), F * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.tabEnd, tree.tabEnd.data(), F * 4, cudaMemcpyDeviceToDevice));
        GFP_CUDA(cudaMemcpy(v.tabTot, tree.tabTot.data(), F * 4, cudaMemcpyDeviceToDevice));
        state.set(0, (0ULL << 48) | ((unsigned long long)F << 24));
        parentRec.set(0, -1);
        counters.set(0, F);  // pending root tasks
    }
    Sched S{state.data(), parentRec.data(), done.data(), counters.data(), counters.data() + 1, counters.data() + 2,
            recPar.data(), recItem.data(), recSup.data(), recCap, (int)nChunks, LV};
    log_info("device task queue: %zu blocks x %d levels, %zu chunks of %.2f MiB, record capacity %lld", B, LV, nChunks,
             chunkBytes / 1048576.0, (long long)recCap);

    k_persistent<<<(unsigned)B, kBS, (size_t)LV * 4>>>(pool, L, S, minsup);
    GFP_CUDA(cudaGetLastError());
    GFP_CUDA(cudaDeviceSynchronize());
    const int32_t nRec = counters.get(1), ovf = counters.get(2);
    cudaFree(pool);
    if (ovf || nRec > recCap) {
        log_warn("device task queue overflowed its record buffer (%d records); falling back to host-driven batches", nRec);
        return false;
    }
    res.parent.resize(nRec);
    res.item.resize(nRec);
    res.support.resize(nRec);
    if (nRec) {
        GFP_CUDA(cudaMemcpy(res.parent.data(), recPar.data(), (size_t)nRec * 4, cudaMemcpyDeviceToHost));
        GFP_CUDA(cudaMemcpy(res.item.data(), recItem.data(), (size_t)nRec * 4, cudaMemcpyDeviceToHost));
        GFP_CUDA(cudaMemcpy(res.support.data(), recSup.data(), (size_t)nRec * 4, cudaMemcpyDeviceToHost));
    }
    res.chunkCount = nChunks;
    res.chunkBytes = chunkBytes;
    res.batches = 1;
    log_info("mined %d frequent itemsets with one persistent kernel, %.2f ms", nRec, timer.elapsed_ms());
    return true;
}

}  // namespace

MiningResult mine_patterns(InitialTree& tree, int minsup, bool deviceQueue) {
    if (tree.numFrequent == 0) return MiningResult{};
    if (deviceQueue) {
        MiningResult res;
        if (mine_device_queue(tree, minsup, res)) return res;
        log_info("using host-driven batches");
    }
    return mine_host_driven(tree, minsup);
}

}  // namespace gfp
