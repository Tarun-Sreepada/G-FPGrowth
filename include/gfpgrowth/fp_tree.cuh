#pragma once
#include <cstdint>
#include <vector>

#include "gfpgrowth/device_buffer.hpp"
#include "gfpgrowth/parser.cuh"

namespace gfp {

// The initial FP-tree, stored as flat arrays. Node 0 is the root. Nodes are grouped
// by item slot: the nodes of slot k occupy [k>0 ? tabEnd[k-1] : 1, tabEnd[k]).
// For the initial tree, slot == item rank (0 = most frequent item).
struct InitialTree {
    int numFrequent = 0;  // F: number of frequent items (slots)
    int numNodes = 0;     // including root
    DeviceBuffer<int32_t> item, count, parent;  // [numNodes]
    DeviceBuffer<int32_t> tabItem, tabEnd, tabTot;  // [F]
    std::vector<int32_t> rankToItem;                      // rank -> original item id
    int maxPatternLen = 0;  // upper bound on the length of any frequent itemset
    size_t filteredItems = 0;
};

// Counts items, drops infrequent ones, ranks items by (support desc, id asc),
// re-sorts each transaction by rank and inserts every transaction into the tree
// concurrently (one thread per transaction, lock-free open-addressing hash of
// (parent, item) -> node).
InitialTree build_initial_tree(ParsedDB& db, int minsup);

}  // namespace gfp
