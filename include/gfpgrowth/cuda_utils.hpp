#pragma once
#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace gfp {

inline void cuda_check(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr + " failed: " +
                                 cudaGetErrorString(e));
    }
}

#define GFP_CUDA(expr) ::gfp::cuda_check((expr), #expr, __FILE__, __LINE__)

// Wall-clock timer that synchronises the device before reading, so stage timings
// include the kernels that were queued.
class StageTimer {
public:
    StageTimer() { reset(); }
    void reset() {
        cudaDeviceSynchronize();
        t0_ = std::chrono::steady_clock::now();
    }
    double elapsed_ms() const {
        cudaDeviceSynchronize();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace gfp
