#pragma once
#include <cstddef>

#include "gfpgrowth/options.hpp"

namespace gfp {

// CPU FP-Growth with the same input/output contract as the GPU version.
// threads == 1 -> single-threaded; threads == 0 -> std::thread::hardware_concurrency().
size_t run_cpu(const Options& opts, int threads);

}  // namespace gfp
