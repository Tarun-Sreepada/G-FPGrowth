#pragma once
// Thin printf-style logging facade. The implementation (src/log.cpp) is backed by
// spdlog; keeping spdlog out of the CUDA translation units avoids nvcc/fmt friction.

namespace gfp {

enum class LogLevel { Debug, Info, Warn, Error };

void set_log_level(LogLevel level);

#if defined(__GNUC__)
#define GFP_PRINTF_FMT(a, b) __attribute__((format(printf, a, b)))
#else
#define GFP_PRINTF_FMT(a, b)
#endif

void log_debug(const char* fmt, ...) GFP_PRINTF_FMT(1, 2);
void log_info(const char* fmt, ...) GFP_PRINTF_FMT(1, 2);
void log_warn(const char* fmt, ...) GFP_PRINTF_FMT(1, 2);
void log_error(const char* fmt, ...) GFP_PRINTF_FMT(1, 2);

}  // namespace gfp
