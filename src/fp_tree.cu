#include <stdexcept>

#include "gfpgrowth/cuda_utils.hpp"
#include "gfpgrowth/fp_tree.cuh"
#include "gfpgrowth/log.hpp"
#include "gfpgrowth/primitives.cuh"

namespace gfp {
namespace {

constexpr uint64_t kEmpty = ~0ULL;
constexpr uint32_t kRoot = 0xFFFFFFFFu;

__device__ __forceinline__ uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

template <typename T>
__device__ __forceinline__ int64_t lower_idx(const T* a, int64_t n, T v) {
    return cuda::std::lower_bound(a, a + n, v) - a;
}

__global__ void k_item_of_key(const uint64_t* keys, int64_t n, int32_t* item) {
    GRID_STRIDE(i, n) item[i] = (int32_t)(keys[i] & 0xffffffffu);
}

__global__ void k_flag_ge(const int32_t* v, int64_t n, int32_t thr, uint8_t* flag) {
    GRID_STRIDE(i, n) flag[i] = v[i] >= thr;
}

// (support desc, id asc) as one sortable key.
__global__ void k_rank_keys(const int32_t* fid, const int32_t* fcnt, int64_t n, uint64_t* ord) {
    GRID_STRIDE(i, n) ord[i] = ((uint64_t)(0x7fffffff - fcnt[i]) << 32) | (uint32_t)fid[i];
}

__global__ void k_rank_of_id(const uint64_t* ord, const int32_t* fid, int F, int32_t* rankOfId, int32_t* tabTot,
                             int32_t* tabItem) {
    GRID_STRIDE(r, F) {
        int32_t id = (int32_t)(ord[r] & 0xffffffffu);
        rankOfId[lower_idx(fid, F, id)] = (int32_t)r;
        tabTot[r] = 0x7fffffff - (int32_t)(ord[r] >> 32);
        tabItem[r] = (int32_t)r;
    }
}

// (txn, item) -> (txn, rank), or kEmpty (sorts last) when the item is infrequent.
__global__ void k_map_keys(uint64_t* keys, int64_t n, const int32_t* fid, const int32_t* rankOfId, int F) {
    GRID_STRIDE(i, n) {
        uint64_t k = keys[i];
        int32_t item = (int32_t)(k & 0xffffffffu);
        int64_t p = lower_idx(fid, F, item);
        keys[i] = (p < F && fid[p] == item) ? ((k & ~0xffffffffULL) | (uint32_t)rankOfId[p]) : kEmpty;
    }
}

__global__ void k_first_empty(const uint64_t* keys, int64_t n, int64_t* out) {
    const uint64_t empty = kEmpty;
    *out = cuda::std::lower_bound(keys, keys + n, empty) - keys;
}

__global__ void k_txn_of_key(const uint64_t* keys, int64_t n, int32_t* tid) {
    GRID_STRIDE(i, n) tid[i] = (int32_t)(keys[i] >> 32);
}

// One thread per transaction: walk its (rank-sorted) items and find-or-create the
// child node for (parent slot, rank). A node's id is its hash-table slot, so there
// is no separate id allocation and no waiting on other threads.
__global__ void k_insert(const uint64_t* keys, int64_t M, int64_t numTxn, uint64_t* tk, int32_t* tc, uint64_t mask) {
    GRID_STRIDE(t, numTxn) {
        int64_t lo = lower_idx(keys, M, (uint64_t)t << 32);
        int64_t hi = lower_idx(keys, M, (uint64_t)(t + 1) << 32);
        uint32_t par = kRoot;
        for (int64_t p = lo; p < hi; ++p) {
            uint64_t key = ((uint64_t)par << 32) | (keys[p] & 0xffffffffu);
            uint64_t h = mix(key) & mask;
            while (true) {
                uint64_t old = atomic_cas<uint64_t>(&tk[h], kEmpty, key);
                if (old == kEmpty || old == key) break;
                h = (h + 1) & mask;
            }
            atomic_add<int32_t>(&tc[h], 1);
            par = (uint32_t)h;
        }
    }
}

__global__ void k_occupied(const uint64_t* tk, int64_t HS, int32_t* flag) {
    GRID_STRIDE(s, HS) flag[s] = tk[s] != kEmpty;
}

__global__ void k_compact(const uint64_t* tk, const int32_t* tc, int64_t HS, const int32_t* pos, int32_t* item,
                          int32_t* count, int32_t* parent) {
    GRID_STRIDE(s, HS) {
        uint64_t k = tk[s];
        if (k == kEmpty) continue;
        int32_t d = pos[s] + 1;
        uint32_t ps = (uint32_t)(k >> 32);
        item[d] = (int32_t)(k & 0xffffffffu);
        count[d] = tc[s];
        parent[d] = ps == kRoot ? 0 : pos[ps] + 1;
    }
}

__global__ void k_iota(int32_t* a, int64_t n, int32_t start) {
    GRID_STRIDE(i, n) a[i] = start + (int32_t)i;
}

__global__ void k_newid(const int32_t* perm, int32_t n, int32_t* newid) {
    GRID_STRIDE(i, n) newid[perm[i]] = (int32_t)i + 1;
}

__global__ void k_regroup(const int32_t* perm, int32_t n, const int32_t* newid, const int32_t* item,
                          const int32_t* count, const int32_t* parent, int32_t* oItem, int32_t* oCount,
                          int32_t* oParent) {
    GRID_STRIDE(i, n) {
        int32_t o = perm[i];
        oItem[i + 1] = item[o];
        oCount[i + 1] = count[o];
        int32_t p = parent[o];
        oParent[i + 1] = p ? newid[p] : 0;
    }
}

// end of slot k's node range (+1: node 0 is the root)
__global__ void k_tab_end(const int32_t* sortedItems, int32_t n, int32_t F, int32_t* tabEnd) {
    GRID_STRIDE(k, F) tabEnd[k] = (int32_t)(cuda::std::upper_bound(sortedItems, sortedItems + n, (int32_t)k) - sortedItems) + 1;
}

}  // namespace

RankedDB rank_transactions(ParsedDB& db, int minsup) {
    StageTimer timer;
    RankedDB tree;
    const int64_t M0 = (int64_t)db.numKeys;
    if (M0 == 0) return tree;

    // ---- item supports: sort items, run-length encode ----------------------------------
    DeviceBuffer<int32_t> uid(M0), ucnt(M0);
    size_t U;
    {
        DeviceBuffer<int32_t> itm(M0);
        k_item_of_key<<<grid_for(M0), kBlock>>>(db.keys.data(), M0, itm.data());
        GFP_CUDA(cudaGetLastError());
        radix_sort(itm, (size_t)M0, 31);
        U = run_length_encode(itm.data(), (size_t)M0, uid.data(), ucnt.data());
    }
    DeviceBuffer<int32_t> fid(U), fcnt(U);
    {
        DeviceBuffer<uint8_t> flag(U);
        k_flag_ge<<<grid_for(U), kBlock>>>(ucnt.data(), U, minsup, flag.data());
        GFP_CUDA(cudaGetLastError());
        size_t F1 = select_flagged(uid.data(), flag.data(), fid.data(), U);
        size_t F2 = select_flagged(ucnt.data(), flag.data(), fcnt.data(), U);
        if (F1 != F2) throw std::runtime_error("internal error: frequent item selection mismatch");
        tree.numFrequent = (int)F1;
    }
    const int F = tree.numFrequent;
    if (F == 0) return tree;
    uid.free();
    ucnt.free();

    // ---- rank items by (support desc, id asc) ----------------------------------------
    DeviceBuffer<int32_t> rankOfId(F);
    tree.tabItem.resize(F);
    tree.tabTot.resize(F);
    {
        DeviceBuffer<uint64_t> ord(F);
        k_rank_keys<<<grid_for(F), kBlock>>>(fid.data(), fcnt.data(), F, ord.data());
        GFP_CUDA(cudaGetLastError());
        radix_sort(ord, (size_t)F);
        k_rank_of_id<<<grid_for(F), kBlock>>>(ord.data(), fid.data(), F, rankOfId.data(), tree.tabTot.data(),
                                              tree.tabItem.data());
        GFP_CUDA(cudaGetLastError());
        std::vector<uint64_t> h = ord.to_host();
        tree.rankToItem.resize(F);
        tree.support.resize(F);
        for (int r = 0; r < F; ++r) {
            tree.rankToItem[r] = (int32_t)(h[r] & 0xffffffffu);
            tree.support[r] = 0x7fffffff - (int32_t)(h[r] >> 32);
        }
    }
    fcnt.free();

    // ---- rewrite transactions: drop infrequent items, order by rank ------------------
    k_map_keys<<<grid_for(M0), kBlock>>>(db.keys.data(), M0, fid.data(), rankOfId.data(), F);
    GFP_CUDA(cudaGetLastError());
    fid.free();
    rankOfId.free();
    radix_sort(db.keys, (size_t)M0);
    int64_t M;
    {
        DeviceBuffer<int64_t> d(1);
        k_first_empty<<<1, 1>>>(db.keys.data(), M0, d.data());
        GFP_CUDA(cudaGetLastError());
        M = d.get(0);
    }
    if (M == 0) { tree.numFrequent = 0; return tree; }
    tree.numKeys = (size_t)M;
    tree.numTransactions = db.numTransactions;

    // Longest possible pattern: the minsup-th longest filtered transaction.
    {
        DeviceBuffer<int32_t> tid(M), tout(M), len(M);
        k_txn_of_key<<<grid_for(M), kBlock>>>(db.keys.data(), M, tid.data());
        GFP_CUDA(cudaGetLastError());
        size_t nt = run_length_encode(tid.data(), (size_t)M, tout.data(), len.data());
        radix_sort(len, nt, 31, /*descending=*/true);
        tree.maxPatternLen = nt >= (size_t)minsup ? (int)len.get(minsup - 1) : (int)len.get(0);
    }
    tree.keys.swap(db.keys);
    db.numKeys = 0;
    return tree;
}

InitialTree build_tree(RankedDB& r) {
    StageTimer timer;
    InitialTree tree;
    tree.numFrequent = r.numFrequent;
    tree.tabItem.swap(r.tabItem);
    tree.tabTot.swap(r.tabTot);
    tree.rankToItem = r.rankToItem;
    tree.maxPatternLen = r.maxPatternLen;
    tree.filteredItems = r.numKeys;
    const int F = r.numFrequent;
    const int64_t M = (int64_t)r.numKeys;
    if (F == 0 || M == 0) { tree.numFrequent = 0; return tree; }
    // ---- concurrent insertion into a hash-indexed node store -------------------------
    if (M >= (1LL << 30)) throw std::runtime_error("too many item occurrences for the 32-bit node ids");
    uint64_t HS = 1024;
    while (HS < (uint64_t)M * 2) HS <<= 1;
    DeviceBuffer<int32_t> nItem, nCount, nParent;
    int32_t occupied = 0;
    {
        DeviceBuffer<uint64_t> tk(HS);
        DeviceBuffer<int32_t> tc(HS);
        tk.fill_bytes(0xFF);  // kEmpty
        tc.zero();
        k_insert<<<grid_for((int64_t)r.numTransactions), kBlock>>>(r.keys.data(), M, (int64_t)r.numTransactions,
                                                                    tk.data(), tc.data(), HS - 1);
        GFP_CUDA(cudaGetLastError());
        r.keys.free();

        DeviceBuffer<int32_t> flag(HS), pos(HS);
        k_occupied<<<grid_for((int64_t)HS), kBlock>>>(tk.data(), (int64_t)HS, flag.data());
        GFP_CUDA(cudaGetLastError());
        exclusive_sum(flag.data(), pos.data(), HS);
        occupied = pos.get(HS - 1) + flag.get(HS - 1);
        flag.free();
        nItem.resize(occupied + 1);
        nCount.resize(occupied + 1);
        nParent.resize(occupied + 1);
        k_compact<<<grid_for((int64_t)HS), kBlock>>>(tk.data(), tc.data(), (int64_t)HS, pos.data(), nItem.data(),
                                                     nCount.data(), nParent.data());
        GFP_CUDA(cudaGetLastError());
    }
    const int32_t N = occupied + 1;  // + root
    tree.numNodes = N;

    // ---- group nodes by item (slot) so each header list is contiguous ----------------
    tree.item.resize(N);
    tree.count.resize(N);
    tree.parent.resize(N);
    {
        DeviceBuffer<int32_t> permIn(occupied), perm(occupied), keyv(occupied), newid(N);
        k_iota<<<grid_for(occupied), kBlock>>>(permIn.data(), occupied, 1);
        GFP_CUDA(cudaGetLastError());
        int endBit = 1;
        while ((1LL << endBit) < F) ++endBit;
        radix_sort_pairs(nItem.data() + 1, permIn.data(), keyv.data(), perm.data(), (size_t)occupied, endBit);
        newid.zero();
        k_newid<<<grid_for(occupied), kBlock>>>(perm.data(), occupied, newid.data());
        k_regroup<<<grid_for(occupied), kBlock>>>(perm.data(), occupied, newid.data(), nItem.data(), nCount.data(),
                                                  nParent.data(), tree.item.data(), tree.count.data(),
                                                  tree.parent.data());
        GFP_CUDA(cudaGetLastError());
        tree.item.set(0, -1);
        tree.count.set(0, 0);
        tree.parent.set(0, -1);

        tree.tabEnd.resize(F);
        k_tab_end<<<grid_for(F), kBlock>>>(keyv.data(), occupied, F, tree.tabEnd.data());
        GFP_CUDA(cudaGetLastError());
    }
    log_info("initial FP-tree: %d frequent items, %d nodes (%lld filtered item occurrences), max pattern length <= %d, %.2f ms",
             F, N, (long long)M, tree.maxPatternLen, timer.elapsed_ms());
    return tree;
}


InitialTree build_initial_tree(ParsedDB& db, int minsup) {
    RankedDB r = rank_transactions(db, minsup);
    return build_tree(r);
}

}  // namespace gfp
