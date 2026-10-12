#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "gfpgrowth/cuda_utils.hpp"
#include "gfpgrowth/fp_tree.cuh"
#include "gfpgrowth/log.hpp"
#include "gfpgrowth/miner.cuh"
#include "gfpgrowth/options.hpp"
#include "gfpgrowth/parser.cuh"

namespace gfp {
namespace {

// PAMI semantics: a value containing '.' is a fraction of the database size,
// otherwise an absolute count. Support must be >= the (possibly fractional) value.
int resolve_minsup(const std::string& s, size_t numTxn) {
    double v;
    try {
        v = std::stod(s);
    } catch (...) {
        throw std::runtime_error("invalid minimum support: " + s);
    }
    if (s.find('.') != std::string::npos) v *= (double)numTxn;
    if (v < 1) v = 1;
    return (int)std::ceil(v);
}

// Pattern i = pattern[parent[i]] + {item[i]}; formatted in parallel into per-thread buffers.
void write_patterns(const Options& o, const MiningResult& r, const std::vector<int32_t>& rankToItem) {
    FILE* f = std::fopen(o.output.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output file: " + o.output);
    std::vector<std::string> idStr(rankToItem.size());
    for (size_t i = 0; i < rankToItem.size(); ++i) idStr[i] = std::to_string(rankToItem[i]);

    const size_t n = r.item.size();
    const unsigned T = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), 16));
    std::vector<std::string> outs(T);
    std::vector<std::thread> th;
    for (unsigned t = 0; t < T; ++t) {
        th.emplace_back([&, t] {
            const size_t lo = n * t / T, hi = n * (t + 1) / T;
            std::string& out = outs[t];
            out.reserve((hi - lo) * 24);
            int32_t chain[256];
            std::vector<int32_t> big;
            char num[12];
            for (size_t i = lo; i < hi; ++i) {
                size_t len = 0;
                for (int32_t p = (int32_t)i; p >= 0; p = r.parent[p]) {
                    if (len < 256) chain[len] = r.item[p];
                    else { if (big.empty()) big.assign(chain, chain + 256); big.push_back(r.item[p]); }
                    ++len;
                }
                const int32_t* c = len <= 256 ? chain : big.data();
                for (size_t k = 0; k < len; ++k) {
                    if (k) out.push_back(o.sep);
                    out += idStr[c[k]];
                }
                big.clear();
                out.push_back(':');
                uint32_t v = (uint32_t)r.support[i];
                int m = 0;
                do { num[m++] = (char)('0' + v % 10); v /= 10; } while (v);
                while (m) out.push_back(num[--m]);
                out.push_back('\n');
            }
        });
    }
    for (auto& x : th) x.join();
    for (auto& s : outs) std::fwrite(s.data(), 1, s.size(), f);
    std::fclose(f);
}

}  // namespace

size_t run(const Options& o) {
    StageTimer total;
    int devCount = 0;
    GFP_CUDA(cudaGetDeviceCount(&devCount));
    if (devCount == 0) throw std::runtime_error("no CUDA device found");
    cudaDeviceProp prop;
    GFP_CUDA(cudaGetDeviceProperties(&prop, 0));
    log_info("device: %s, %.1f GiB", prop.name, prop.totalGlobalMem / 1073741824.0);

    ParsedDB db = parse_file_on_gpu(o.input, o.sep);
    const int minsup = resolve_minsup(o.minsup, db.numTransactions);
    log_info("minimum support: %d (of %zu transactions)", minsup, db.numTransactions);

    InitialTree tree = build_initial_tree(db, minsup);
    MiningResult res = mine_patterns(tree, minsup);
    write_patterns(o, res, tree.rankToItem);
    log_info("wrote %zu patterns to %s; total %.2f ms", res.item.size(), o.output.c_str(), total.elapsed_ms());
    return res.item.size();
}

}  // namespace gfp
