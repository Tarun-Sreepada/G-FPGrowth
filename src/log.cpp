#include "gfpgrowth/log.hpp"

#include <spdlog/spdlog.h>

#include <cstdarg>
#include <cstdio>
#include <string>

namespace gfp {
namespace {
std::string vformat(const char* fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    std::string s(n > 0 ? static_cast<size_t>(n) : 0, '\0');
    if (n > 0) std::vsnprintf(s.data(), s.size() + 1, fmt, ap);
    return s;
}
}  // namespace

void set_log_level(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: spdlog::set_level(spdlog::level::debug); break;
        case LogLevel::Info: spdlog::set_level(spdlog::level::info); break;
        case LogLevel::Warn: spdlog::set_level(spdlog::level::warn); break;
        case LogLevel::Error: spdlog::set_level(spdlog::level::err); break;
    }
}

#define GFP_DEFINE_LOG(name, call)             \
    void name(const char* fmt, ...) {          \
        va_list ap;                            \
        va_start(ap, fmt);                     \
        std::string s = vformat(fmt, ap);      \
        va_end(ap);                            \
        call("{}", s);                         \
    }
GFP_DEFINE_LOG(log_debug, spdlog::debug)
GFP_DEFINE_LOG(log_info, spdlog::info)
GFP_DEFINE_LOG(log_warn, spdlog::warn)
GFP_DEFINE_LOG(log_error, spdlog::error)

}  // namespace gfp
