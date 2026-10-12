#include <CLI/CLI.hpp>
#include <exception>
#include <string>

#include "gfpgrowth/cpu_fpgrowth.hpp"
#include "gfpgrowth/log.hpp"
#include "gfpgrowth/options.hpp"

namespace {
// "tab", "space", "comma", ... or a literal single character / "\t" escape.
char parse_sep(const std::string& s) {
    if (s == "tab" || s == "\\t") return '\t';
    if (s == "space") return ' ';
    if (s == "comma") return ',';
    if (s == "semicolon") return ';';
    if (s == "pipe") return '|';
    if (s.size() == 1) return s[0];
    throw CLI::ValidationError("separator must be a single character or one of: tab, space, comma, semicolon, pipe");
}
}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"G-FPGrowth: the FP-Growth frequent itemset mining algorithm implemented on the GPU"};
    gfp::Options opt;
    std::string sep = "tab";
    app.add_option("input", opt.input, "Input transaction file (one transaction per line)")->required()->check(CLI::ExistingFile);
    app.add_option("output", opt.output, "Output file for the frequent itemsets (items:support)")->required();
    app.add_option("minsup", opt.minsup,
                   "Minimum support: an absolute count (10) or, if it contains '.', a fraction of the database (0.01)")
        ->required();
    app.add_option("sep", sep, "Item separator: tab, space, comma, semicolon, pipe or a single character")
        ->capture_default_str();
    std::string mode = "gpu";
    int threads = 0;
    app.add_option("-m,--mode", mode, "Execution backend: gpu, cpu (multi-threaded) or cpu1 (single-threaded)")
        ->check(CLI::IsMember({"gpu", "cpu", "cpu1"}))
        ->capture_default_str();
    app.add_option("-t,--threads", threads, "CPU threads for --mode cpu (0 = all cores)");
    app.add_option("-a,--algo", opt.algo, "Algorithm: auto (pick by density), fpgrowth or eclat (bitset)")
        ->check(CLI::IsMember({"auto", "fpgrowth", "eclat"}))
        ->capture_default_str();
    app.add_option("--sched", opt.sched, "GPU FP-Growth scheduler: device (persistent kernel + device task queue) or host")
        ->check(CLI::IsMember({"device", "host"}))
        ->capture_default_str();
    app.add_flag("-v,--verbose", opt.verbose, "Verbose (debug) logging");
    CLI11_PARSE(app, argc, argv);

    try {
        opt.sep = parse_sep(sep);
        if (opt.verbose) gfp::set_log_level(gfp::LogLevel::Debug);
        if (mode == "gpu") gfp::run(opt);
        else gfp::run_cpu(opt, mode == "cpu1" ? 1 : threads);
    } catch (const std::exception& e) {
        gfp::log_error("%s", e.what());
        return 1;
    }
    return 0;
}
