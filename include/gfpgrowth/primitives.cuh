#pragma once
// Device-side building blocks: libcu++ atomics/algorithms and CUB device primitives.
// (libcu++ provides no parallel sort/scan/select, those come from CUB, its sibling in CCCL.)
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>

#include "gfpgrowth/cuda_utils.hpp"
#include "gfpgrowth/device_buffer.hpp"

namespace gfp {

constexpr int kBlock = 256;

inline int grid_for(int64_t n) {
    int64_t g = (n + kBlock - 1) / kBlock;
    return (int)(g < 1 ? 1 : (g > 65535 ? 65535 : g));
}

#define GRID_STRIDE(i, n) \
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < (n); i += (int64_t)gridDim.x * blockDim.x)

// ---- atomics (cuda::atomic_ref) ---------------------------------------------------------
template <typename T>
__device__ __forceinline__ T atomic_add(T* p, T v) {
    return cuda::atomic_ref<T, cuda::thread_scope_device>(*p).fetch_add(v, cuda::memory_order_relaxed);
}
template <typename T>
__device__ __forceinline__ void atomic_store(T* p, T v) {
    cuda::atomic_ref<T, cuda::thread_scope_device>(*p).store(v, cuda::memory_order_relaxed);
}
// Returns the value that was in *p (== expected on success).
template <typename T>
__device__ __forceinline__ T atomic_cas(T* p, T expected, T desired) {
    cuda::atomic_ref<T, cuda::thread_scope_device>(*p).compare_exchange_strong(expected, desired,
                                                                               cuda::memory_order_relaxed);
    return expected;
}

// ---- CUB two-phase calls ----------------------------------------------------------------
template <typename F>
void cub_run(F&& f) {
    size_t bytes = 0;
    GFP_CUDA(f(nullptr, bytes));
    DeviceBuffer<unsigned char> tmp(bytes ? bytes : 1);
    GFP_CUDA(f(tmp.data(), bytes));
}

// Sorts the first n keys of `keys` (ascending unless `descending`). Radix sort over bits [0, endBit).
template <typename K>
void radix_sort(DeviceBuffer<K>& keys, size_t n, int endBit = sizeof(K) * 8, bool descending = false) {
    if (n < 2) return;
    DeviceBuffer<K> out(n);
    cub_run([&](void* t, size_t& b) {
        return descending ? cub::DeviceRadixSort::SortKeysDescending(t, b, keys.data(), out.data(), (int64_t)n, 0, endBit)
                          : cub::DeviceRadixSort::SortKeys(t, b, keys.data(), out.data(), (int64_t)n, 0, endBit);
    });
    keys.swap(out);
    keys.truncate(n);
}

// Stable sort of (key, value) pairs.
template <typename K, typename V>
void radix_sort_pairs(const K* kIn, const V* vIn, K* kOut, V* vOut, size_t n, int endBit) {
    if (n == 0) return;
    cub_run([&](void* t, size_t& b) {
        return cub::DeviceRadixSort::SortPairs(t, b, kIn, kOut, vIn, vOut, (int64_t)n, 0, endBit);
    });
}

template <typename T>
void exclusive_sum(const T* in, T* out, size_t n) {
    if (n == 0) return;
    cub_run([&](void* t, size_t& b) { return cub::DeviceScan::ExclusiveSum(t, b, in, out, (int64_t)n); });
}

// Compacts `in` by `flags`; returns the number of selected elements.
template <typename T, typename Flag>
size_t select_flagged(const T* in, const Flag* flags, T* out, size_t n) {
    if (n == 0) return 0;
    DeviceBuffer<int64_t> num(1);
    cub_run([&](void* t, size_t& b) {
        return cub::DeviceSelect::Flagged(t, b, in, flags, out, num.data(), (int64_t)n);
    });
    return (size_t)num.get(0);
}

// Removes consecutive duplicates in place; returns the new length.
template <typename T>
size_t unique_inplace(DeviceBuffer<T>& a, size_t n) {
    if (n < 2) return n;
    DeviceBuffer<T> out(n);
    DeviceBuffer<int64_t> num(1);
    cub_run([&](void* t, size_t& b) { return cub::DeviceSelect::Unique(t, b, a.data(), out.data(), num.data(), (int64_t)n); });
    a.swap(out);
    size_t m = (size_t)num.get(0);
    a.truncate(m);
    return m;
}

// Run-length encoding of sorted keys: unique values + run lengths. Returns the run count.
template <typename K, typename C>
size_t run_length_encode(const K* keys, size_t n, K* uniq, C* counts) {
    if (n == 0) return 0;
    DeviceBuffer<int64_t> num(1);
    cub_run([&](void* t, size_t& b) {
        return cub::DeviceRunLengthEncode::Encode(t, b, keys, uniq, counts, num.data(), (int64_t)n);
    });
    return (size_t)num.get(0);
}

}  // namespace gfp
