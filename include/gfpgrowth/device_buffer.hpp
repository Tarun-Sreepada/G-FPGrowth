#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "gfpgrowth/cuda_utils.hpp"

namespace gfp {

// Minimal RAII owner of a device allocation (no thrust).
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { resize(n); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept { swap(o); }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { free(); swap(o); }
        return *this;
    }
    ~DeviceBuffer() { free(); }

    // Discards the contents.
    void resize(size_t n) {
        free();
        if (n) GFP_CUDA(cudaMalloc(&p_, n * sizeof(T)));
        n_ = n;
    }
    void free() {
        if (p_) cudaFree(p_);
        p_ = nullptr;
        n_ = 0;
    }
    // Logically shrink without reallocating.
    void truncate(size_t n) { if (n < n_) n_ = n; }
    void swap(DeviceBuffer& o) noexcept { std::swap(p_, o.p_); std::swap(n_, o.n_); }

    void zero() { if (n_) GFP_CUDA(cudaMemset(p_, 0, n_ * sizeof(T))); }
    void fill_bytes(int byte) { if (n_) GFP_CUDA(cudaMemset(p_, byte, n_ * sizeof(T))); }

    T get(size_t i) const {
        T v;
        GFP_CUDA(cudaMemcpy(&v, p_ + i, sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void set(size_t i, const T& v) { GFP_CUDA(cudaMemcpy(p_ + i, &v, sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> to_host() const {
        std::vector<T> h(n_);
        if (n_) GFP_CUDA(cudaMemcpy(h.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }

    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

}  // namespace gfp
