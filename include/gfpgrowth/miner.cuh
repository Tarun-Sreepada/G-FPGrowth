#pragma once
#include <cstdint>
#include <vector>

#include "gfpgrowth/fp_tree.cuh"

namespace gfp {

// Frequent itemsets as a forest: pattern i = pattern[parent[i]] + {item[i]} with
// support support[i]. parent == -1 marks a singleton. Items are ranks.
struct MiningResult {
    std::vector<int32_t> parent, item, support;
    size_t chunkCount = 0;
    size_t chunkBytes = 0;
    size_t batches = 0;
};

// With deviceQueue the whole traversal runs in one persistent kernel with a device-side task
// queue (work stealing between blocks, no host round trips); otherwise the host launches
// one kernel per batch of tasks.
//
// Pattern-growth over GPU memory that is pre-partitioned into equally sized chunks.
// Every conditional FP-tree has <= the nodes of the initial tree, so one chunk
// (sized for the initial tree) always fits any tree that is ever built.
MiningResult mine_patterns(InitialTree& tree, int minsup, bool deviceQueue = true);

// Dense-data alternative: vertical bitsets + AND/popcount (Eclat). Same chunked-memory
// scheme: every class (set of item bitsets) is no larger than the root class.
// Returns false (and leaves `out` untouched) if too few chunks fit in GPU memory.
bool mine_eclat(RankedDB& db, int minsup, MiningResult& out);

}  // namespace gfp
