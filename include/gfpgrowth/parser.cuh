#pragma once
#include <cstdint>
#include <string>

#include "gfpgrowth/device_buffer.hpp"

namespace gfp {

// A transaction database living on the GPU as a sorted, de-duplicated list of
// (transactionId << 32 | itemId) keys.
struct ParsedDB {
    DeviceBuffer<uint64_t> keys;
    size_t numKeys = 0;           // valid prefix of keys
    size_t numTransactions = 0;  // number of lines in the file
    size_t fileBytes = 0;
};

// Copies the file to the GPU and parses it there:
//   1. count '\n'            2. record + sort newline positions (line boundaries)
//   3. count tokens / line   4. exclusive scan -> write offsets
//   5. ASCII -> int32 per token, in parallel across lines.
ParsedDB parse_file_on_gpu(const std::string& path, char sep);

}  // namespace gfp
