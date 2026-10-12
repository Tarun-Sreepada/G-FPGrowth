// CPU FP-Growth (single- and multi-threaded) used as the baseline for G-FPGrowth.
//
// Same algorithm family as the GPU path (array FP-tree, nodes grouped by item, conditional
// trees built by projecting ancestors), plus CPU-specific tuning:
//   * mmap + madvise(SEQUENTIAL/WILLNEED) input, AVX2 newline scan (SSE2/memchr fallback)
//   * parallel parse / count / rank-rewrite with thread-local buffers (no atomics)
//   * initial tree built in parallel: one subtree per first-item bucket, own hash table
//   * mining: dynamic scheduling of root items (heaviest first), per-thread level arenas
//     that are reused (no malloc in steady state), per-thread output buffers
//   * conditional-tree support counts via a branch-free reverse sweep (parent index <
//     child index) instead of pointer chasing when that is cheaper
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gfpgrowth/cpu_fpgrowth.hpp"
#include "gfpgrowth/log.hpp"

namespace gfp {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

template <typename F>
void parallel(int T, F&& f) {
    if (T <= 1) { f(0); return; }
    std::vector<std::thread> th;
    std::vector<std::exception_ptr> ex(T);
    for (int t = 0; t < T; ++t) {
        th.emplace_back([&, t] {
            try { f(t); } catch (...) { ex[t] = std::current_exception(); }
        });
    }
    for (auto& x : th) x.join();
    for (auto& e : ex) if (e) std::rethrow_exception(e);
}

inline const char* find_nl(const char* p, const char* end) {
#if defined(__AVX2__)
    const __m256i nl = _mm256_set1_epi8('\n');
    while (p + 32 <= end) {
        unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)p), nl));
        if (m) return p + __builtin_ctz(m);
        p += 32;
    }
#elif defined(__SSE2__)
    const __m128i nl = _mm_set1_epi8('\n');
    while (p + 16 <= end) {
        unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i*)p), nl));
        if (m) return p + __builtin_ctz(m);
        p += 16;
    }
#endif
    const void* r = p < end ? std::memchr(p, '\n', (size_t)(end - p)) : nullptr;
    return r ? (const char*)r : end;
}

inline bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r'; }

struct Mapped {
    const char* data = nullptr;
    size_t size = 0;
    void* base = nullptr;
};

Mapped map_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open input file: " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat failed"); }
    Mapped m;
    m.size = (size_t)st.st_size;
    if (m.size) {
        m.base = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        if (m.base == MAP_FAILED) { ::close(fd); throw std::runtime_error("mmap failed"); }
        madvise(m.base, m.size, MADV_SEQUENTIAL | MADV_WILLNEED);
        m.data = (const char*)m.base;
    }
    ::close(fd);
    return m;
}

struct Local {
    std::vector<int32_t> items;
    std::vector<uint32_t> lens;
    int32_t maxId = -1;
};

void parse_range(const char* p, const char* end, char sep, Local& L) {
    L.items.reserve((size_t)(end - p) / 4 + 16);
    while (p < end) {
        const char* nl = find_nl(p, end);
        uint32_t n = 0;
        const char* i = p;
        while (i <= nl) {
            const char* j = i;
            while (j < nl && *j != sep) ++j;
            const char* a = i;
            const char* z = j;
            while (a < z && is_ws(*a)) ++a;
            while (z > a && is_ws(z[-1])) --z;
            if (z > a) {
                int64_t v = 0;
                for (const char* q = a; q < z; ++q) {
                    unsigned d = (unsigned)(*q - '0');
                    if (d > 9) throw std::runtime_error("input contains a token that is not a non-negative integer");
                    v = v * 10 + d;
                    if (v > 2147483647LL) throw std::runtime_error("item id does not fit 31 bits");
                }
                L.items.push_back((int32_t)v);
                if ((int32_t)v > L.maxId) L.maxId = (int32_t)v;
                ++n;
            }
            i = j + 1;
        }
        L.lens.push_back(n);
        p = nl + 1;
    }
}

struct Tree {
    std::vector<int32_t> item, count, parent, depth;  // nodes, node 0 = root
    std::vector<int32_t> tabItem, tabEnd, tabTot;     // slots
    int n = 0;  // nodes incl. root
    int K = 0;  // slots
    void reserve_nodes(size_t nn) {
        if (item.size() < nn) { item.resize(nn); count.resize(nn); parent.resize(nn); depth.resize(nn); }
    }
    void reserve_slots(size_t k) {
        if (tabItem.size() < k) { tabItem.resize(k); tabEnd.resize(k); tabTot.resize(k); }
    }
};

struct Scratch {
    std::vector<int32_t> c, newid, tot, map, cur;
};

// Builds the conditional tree of slot j of T into C. Returns the number of slots of C.
int build_child(const Tree& T, int j, Tree& C, Scratch& S, int minsup) {
    const int s = j > 0 ? T.tabEnd[j - 1] : 1;
    const int e = T.tabEnd[j];
    int32_t* c = S.c.data();
    const int32_t* parent = T.parent.data();

    uint64_t walk = 0;
    for (int h = s; h < e; ++h) walk += (uint64_t)T.depth[h];
    if (walk <= (uint64_t)s + (uint64_t)(e - s)) {
        std::memset(c, 0, sizeof(int32_t) * (size_t)s);
        for (int h = s; h < e; ++h) {
            const int v = T.count[h];
            for (int a = parent[h]; a > 0; a = parent[a]) c[a] += v;
        }
    } else {  // linear reverse sweep: parents have smaller indices than children
        std::memset(c, 0, sizeof(int32_t) * (size_t)s);
        std::memcpy(c + s, T.count.data() + s, sizeof(int32_t) * (size_t)(e - s));
        for (int a = e - 1; a >= 1; --a) c[parent[a]] += c[a];
    }

    int32_t* tot = S.tot.data();
    std::memset(tot, 0, sizeof(int32_t) * (size_t)j);
    const int32_t* item = T.item.data();
    for (int a = 1; a < s; ++a) tot[item[a]] += c[a];

    int KC = 0;
    int32_t* map = S.map.data();
    C.reserve_slots((size_t)j + 1);
    for (int k = 0; k < j; ++k) {
        if (tot[k] >= minsup) {
            C.tabItem[KC] = T.tabItem[k];
            C.tabTot[KC] = tot[k];
            map[k] = KC++;
        } else {
            map[k] = -1;
        }
    }
    C.K = KC;
    if (KC == 0) return 0;

    int32_t* cur = S.cur.data();
    std::memset(cur, 0, sizeof(int32_t) * (size_t)KC);
    for (int a = 1; a < s; ++a) {
        if (c[a] > 0) {
            const int ns = map[item[a]];
            if (ns >= 0) ++cur[ns];
        }
    }
    int pos = 1;
    for (int k = 0; k < KC; ++k) { const int n = cur[k]; cur[k] = pos; pos += n; }
    C.reserve_nodes((size_t)pos + 1);
    C.n = pos;

    int32_t* newid = S.newid.data();
    for (int a = 1; a < s; ++a) {
        const int cv = c[a];
        const int ns = cv > 0 ? map[item[a]] : -1;
        if (ns < 0) { newid[a] = -1; continue; }
        const int d = cur[ns]++;
        C.item[d] = ns;
        C.count[d] = cv;
        int p = parent[a];
        while (p > 0 && newid[p] < 0) p = parent[p];
        const int np = p > 0 ? newid[p] : 0;
        C.parent[d] = np;
        C.depth[d] = np ? C.depth[np] + 1 : 1;
        newid[a] = d;
    }
    C.item[0] = -1; C.count[0] = 0; C.parent[0] = -1; C.depth[0] = 0;
    for (int k = 0; k < KC; ++k) C.tabEnd[k] = cur[k];
    return KC;
}

struct Worker {
    std::vector<std::unique_ptr<Tree>> levels;
    Scratch S;
    std::vector<int32_t> stack;
    std::string out;
    size_t patterns = 0;
    Tree& level(size_t d) {
        while (levels.size() <= d) levels.emplace_back(new Tree());
        return *levels[d];
    }
};

struct Ctx {
    int minsup;
    char sep;
    const std::vector<std::string>* idStr;  // rank -> decimal id
};

inline void append_uint(std::string& s, uint32_t v) {
    char buf[12];
    int n = 0;
    do { buf[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) s.push_back(buf[--n]);
}

void expand(Worker& W, const Ctx& X, const Tree& T, int j, size_t depth) {
    W.stack.push_back(T.tabItem[j]);
    for (size_t i = 0; i < W.stack.size(); ++i) {
        if (i) W.out.push_back(X.sep);
        W.out += (*X.idStr)[W.stack[i]];
    }
    W.out.push_back(':');
    append_uint(W.out, (uint32_t)T.tabTot[j]);
    W.out.push_back('\n');
    ++W.patterns;

    Tree& C = W.level(depth + 1);
    const int KC = build_child(T, j, C, W.S, X.minsup);
    for (int k = KC - 1; k >= 0; --k) expand(W, X, C, k, depth + 1);
    W.stack.pop_back();
}

int resolve_minsup(const std::string& s, size_t numTxn) {
    double v;
    try { v = std::stod(s); } catch (...) { throw std::runtime_error("invalid minimum support: " + s); }
    if (s.find('.') != std::string::npos) v *= (double)numTxn;
    if (v < 1) v = 1;
    return (int)std::ceil(v);
}

struct TxnRef {
    const int32_t* p;
    uint32_t n;
};

struct Part {
    std::vector<int32_t> item, count, parent;  // local ids are 1-based (index = id-1)
};

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

}  // namespace

size_t run_cpu(const Options& o, int threads) {
    auto tAll = Clock::now();
    int T = threads > 0 ? threads : (int)std::max(1u, std::thread::hardware_concurrency());
    log_info("cpu mode: %d thread(s)", T);

    // ---- 1. mmap + parallel parse ------------------------------------------------------
    auto t0 = Clock::now();
    Mapped mf = map_file(o.input);
    std::vector<size_t> startOf(T + 1, mf.size);
    startOf[0] = 0;
    for (int t = 1; t < T; ++t) {
        size_t nominal = mf.size * (size_t)t / (size_t)T;
        if (nominal >= mf.size) { startOf[t] = mf.size; continue; }
        if (nominal > 0 && mf.data[nominal - 1] != '\n') {
            const char* nl = find_nl(mf.data + nominal, mf.data + mf.size);
            nominal = (size_t)(nl - mf.data) + 1;
            if (nominal > mf.size) nominal = mf.size;
        }
        startOf[t] = std::max(nominal, startOf[t - 1]);
    }
    std::vector<Local> loc(T);
    parallel(T, [&](int t) { parse_range(mf.data + startOf[t], mf.data + startOf[t + 1], o.sep, loc[t]); });
    size_t numTxn = 0, numItems = 0;
    int32_t maxId = -1;
    for (auto& l : loc) { numTxn += l.lens.size(); numItems += l.items.size(); maxId = std::max(maxId, l.maxId); }
    if (mf.base) munmap(mf.base, mf.size);
    log_info("parsed on CPU: %zu bytes, %zu transactions, %zu item occurrences in %.2f ms", mf.size, numTxn, numItems,
             ms_since(t0));

    const int minsup = resolve_minsup(o.minsup, numTxn);
    log_info("minimum support: %d (of %zu transactions)", minsup, numTxn);

    // ---- 2. compress sparse id spaces, count supports (dedupe per transaction) ---------
    auto t1 = Clock::now();
    std::vector<int32_t> origId;  // dense id -> original id (empty = identity)
    if (maxId >= (1 << 22)) {
        std::vector<int32_t> ids;
        ids.reserve(numItems);
        for (auto& l : loc) ids.insert(ids.end(), l.items.begin(), l.items.end());
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        parallel(T, [&](int t) {
            for (auto& v : loc[t].items) v = (int32_t)(std::lower_bound(ids.begin(), ids.end(), v) - ids.begin());
        });
        maxId = (int32_t)ids.size() - 1;
        origId = std::move(ids);
    }
    const size_t D = (size_t)std::max(maxId + 1, 0);
    std::vector<std::vector<uint32_t>> hist(T);
    parallel(T, [&](int t) {
        hist[t].assign(D, 0);
        std::vector<uint32_t> stamp(D, UINT32_MAX);
        uint32_t txn = 0;
        size_t p = 0;
        for (uint32_t len : loc[t].lens) {
            for (uint32_t i = 0; i < len; ++i) {
                const int32_t it = loc[t].items[p + i];
                if (stamp[it] != txn) { stamp[it] = txn; ++hist[t][it]; }
            }
            p += len;
            ++txn;
        }
    });
    std::vector<uint32_t> support(D, 0);
    parallel(T, [&](int t) {
        size_t lo = D * (size_t)t / T, hi = D * (size_t)(t + 1) / T;
        for (size_t i = lo; i < hi; ++i) {
            uint32_t s = 0;
            for (int k = 0; k < T; ++k) s += hist[k][i];
            support[i] = s;
        }
    });
    hist.clear();
    hist.shrink_to_fit();

    std::vector<std::pair<uint32_t, int32_t>> freq;  // (support, id)
    for (size_t i = 0; i < D; ++i) if (support[i] >= (uint32_t)minsup) freq.emplace_back(support[i], (int32_t)i);
    std::sort(freq.begin(), freq.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    const int F = (int)freq.size();
    if (F == 0) {
        FILE* f = std::fopen(o.output.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open output file: " + o.output);
        std::fclose(f);
        log_info("no frequent items; wrote 0 patterns to %s; total %.2f ms", o.output.c_str(), ms_since(tAll));
        return 0;
    }
    std::vector<int32_t> rankOf(D, -1);
    for (int r = 0; r < F; ++r) rankOf[freq[r].second] = r;

    // ---- 3. rewrite transactions as sorted, de-duplicated rank lists (in place) --------
    parallel(T, [&](int t) {
        Local& L = loc[t];
        size_t r = 0, w = 0;
        for (auto& len : L.lens) {
            const size_t start = w;
            for (uint32_t i = 0; i < len; ++i) {
                const int32_t rk = rankOf[L.items[r + i]];
                if (rk >= 0) L.items[w++] = rk;
            }
            r += len;
            int32_t* b = L.items.data() + start;
            const size_t n = w - start;
            if (n <= 24) {
                for (size_t a = 1; a < n; ++a) {
                    int32_t v = b[a];
                    size_t q = a;
                    while (q > 0 && b[q - 1] > v) { b[q] = b[q - 1]; --q; }
                    b[q] = v;
                }
            } else {
                std::sort(b, b + n);
            }
            size_t u = n ? 1 : 0;
            for (size_t a = 1; a < n; ++a) if (b[a] != b[u - 1]) b[u++] = b[a];
            w = start + u;
            len = (uint32_t)u;
        }
        L.items.resize(w);
    });

    // ---- 4. bucket transactions by first (most frequent) item ---------------------------
    std::vector<TxnRef> refs(numTxn);
    {
        std::vector<size_t> base(T + 1, 0);
        for (int t = 0; t < T; ++t) base[t + 1] = base[t] + loc[t].lens.size();
        parallel(T, [&](int t) {
            size_t p = 0, k = base[t];
            for (uint32_t len : loc[t].lens) { refs[k++] = TxnRef{loc[t].items.data() + p, len}; p += len; }
        });
    }
    std::vector<uint32_t> bucketStart(F + 1, 0);
    for (auto& r : refs) if (r.n) ++bucketStart[r.p[0] + 1];
    for (int r = 0; r < F; ++r) bucketStart[r + 1] += bucketStart[r];
    std::vector<uint32_t> order(bucketStart[F]);
    {
        std::vector<uint32_t> cursor(bucketStart.begin(), bucketStart.end() - 1);
        for (size_t i = 0; i < refs.size(); ++i) if (refs[i].n) order[cursor[refs[i].p[0]]++] = (uint32_t)i;
    }

    // ---- 5. build one subtree per first item, in parallel -------------------------------
    std::vector<Part> parts(F);
    {
        std::atomic<int> next{0};
        parallel(T, [&](int) {
            std::vector<uint64_t> keys;
            std::vector<int32_t> vals;
            for (int r = next.fetch_add(1); r < F; r = next.fetch_add(1)) {
                const uint32_t b = bucketStart[r], e = bucketStart[r + 1];
                if (b == e) continue;
                size_t occ = 0;
                for (uint32_t i = b; i < e; ++i) occ += refs[order[i]].n;
                size_t cap = 64;
                while (cap < occ * 2) cap <<= 1;
                keys.assign(cap, 0);
                vals.assign(cap, 0);
                Part& P = parts[r];
                for (uint32_t i = b; i < e; ++i) {
                    const TxnRef& tx = refs[order[i]];
                    int32_t par = 0;
                    for (uint32_t k = 0; k < tx.n; ++k) {
                        const uint64_t key = ((uint64_t)par << 32) | (uint32_t)tx.p[k];
                        size_t h = mix64(key) & (cap - 1);
                        int32_t id;
                        while (true) {
                            if (vals[h] == 0) {
                                P.item.push_back(tx.p[k]);
                                P.count.push_back(0);
                                P.parent.push_back(par);
                                id = (int32_t)P.item.size();
                                keys[h] = key;
                                vals[h] = id;
                                break;
                            }
                            if (keys[h] == key) { id = vals[h]; break; }
                            h = (h + 1) & (cap - 1);
                        }
                        ++P.count[id - 1];
                        par = id;
                    }
                }
            }
        });
    }

    // concatenate parts and group by item (counting sort; parent rank < child rank so
    // parent index < child index afterwards)
    Tree T0;
    {
        std::vector<int32_t> base(F + 1, 1);
        for (int r = 0; r < F; ++r) base[r + 1] = base[r] + (int32_t)parts[r].item.size();
        const int N = base[F];
        std::vector<int32_t> gItem(N), gCount(N), gParent(N);
        parallel(T, [&](int t) {
            for (int r = t; r < F; r += T) {
                const Part& P = parts[r];
                for (size_t i = 0; i < P.item.size(); ++i) {
                    const size_t g = (size_t)base[r] + i;
                    gItem[g] = P.item[i];
                    gCount[g] = P.count[i];
                    gParent[g] = P.parent[i] ? base[r] + P.parent[i] - 1 : 0;
                }
            }
        });
        parts.clear();
        parts.shrink_to_fit();
        std::vector<int32_t> cnt(F + 1, 0);
        for (int i = 1; i < N; ++i) ++cnt[gItem[i] + 1];
        T0.tabEnd.resize(F);
        T0.tabItem.resize(F);
        T0.tabTot.resize(F);
        int pos = 1;
        std::vector<int32_t> cur(F);
        for (int k = 0; k < F; ++k) { cur[k] = pos; pos += cnt[k + 1]; }
        T0.item.resize(N); T0.count.resize(N); T0.parent.resize(N); T0.depth.resize(N);
        std::vector<int32_t> newid(N, 0);
        for (int i = 1; i < N; ++i) newid[i] = cur[gItem[i]]++;
        for (int i = 1; i < N; ++i) {
            const int d = newid[i];
            T0.item[d] = gItem[i];
            T0.count[d] = gCount[i];
            T0.parent[d] = gParent[i] ? newid[gParent[i]] : 0;
        }
        T0.item[0] = -1; T0.count[0] = 0; T0.parent[0] = -1; T0.depth[0] = 0;
        for (int d = 1; d < N; ++d) T0.depth[d] = T0.parent[d] ? T0.depth[T0.parent[d]] + 1 : 1;
        for (int k = 0; k < F; ++k) {
            T0.tabEnd[k] = cur[k];
            T0.tabItem[k] = k;
            T0.tabTot[k] = (int32_t)freq[k].first;
        }
        T0.n = N;
        T0.K = F;
    }
    refs.clear(); refs.shrink_to_fit();
    order.clear(); order.shrink_to_fit();
    loc.clear(); loc.shrink_to_fit();
    log_info("initial FP-tree: %d frequent items, %d nodes in %.2f ms", F, T0.n, ms_since(t1));

    // ---- 6. mine: root items are dynamic tasks, heaviest (highest slot) first -----------
    auto t2 = Clock::now();
    std::vector<std::string> idStr(F);
    for (int r = 0; r < F; ++r) {
        const int32_t id = origId.empty() ? freq[r].second : origId[freq[r].second];
        idStr[r] = std::to_string(id);
    }
    Ctx X{minsup, o.sep, &idStr};
    std::vector<Worker> workers(T);
    for (auto& w : workers) {
        const size_t n = (size_t)T0.n + 1;
        w.S.c.resize(n); w.S.newid.resize(n);
        w.S.tot.resize((size_t)F + 1); w.S.map.resize((size_t)F + 1); w.S.cur.resize((size_t)F + 1);
        w.stack.reserve(64);
        w.out.reserve(1 << 16);
    }
    std::atomic<int> nextSlot{F - 1};
    parallel(T, [&](int t) {
        for (int j = nextSlot.fetch_sub(1); j >= 0; j = nextSlot.fetch_sub(1)) expand(workers[t], X, T0, j, 0);
    });
    size_t total = 0;
    for (auto& w : workers) total += w.patterns;
    log_info("mined %zu frequent itemsets in %.2f ms", total, ms_since(t2));

    FILE* f = std::fopen(o.output.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output file: " + o.output);
    for (auto& w : workers) std::fwrite(w.out.data(), 1, w.out.size(), f);
    std::fclose(f);
    log_info("wrote %zu patterns to %s; total %.2f ms", total, o.output.c_str(), ms_since(tAll));
    return total;
}

}  // namespace gfp
