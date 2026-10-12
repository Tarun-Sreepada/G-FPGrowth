#pragma once
#include <string>

namespace gfp {

struct Options {
    std::string input;
    std::string output;
    std::string minsup;  // "10" = absolute count, "0.01" (contains '.') = fraction of #transactions
    char sep = '\t';
    bool verbose = false;
};

// Runs the whole pipeline (parse -> count -> FP-tree -> mine -> write). Returns the
// number of frequent itemsets written.
size_t run(const Options& opts);

}  // namespace gfp
