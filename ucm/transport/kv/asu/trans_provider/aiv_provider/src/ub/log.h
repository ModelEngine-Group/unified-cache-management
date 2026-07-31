/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <thread>

namespace umc::comm::log {

enum class Level : int {
    kError = 0,
    kWarn = 1,
    kInfo = 2,
    kDebug = 3,
};

namespace detail {
inline std::atomic<bool>& DebugEnabledRef()
{
    static std::atomic<bool> v{false};
    return v;
}

inline const char* LevelStr(Level lv)
{
    switch (lv) {
        case Level::kError: return "ERROR";
        case Level::kWarn: return "WARN ";
        case Level::kInfo: return "INFO ";
        case Level::kDebug: return "DEBUG";
    }
    return "?    ";
}

inline void Emit(Level lv, const char* file, int line, const char* fmt, ...)
{
    if (lv == Level::kDebug && !DebugEnabledRef().load(std::memory_order_relaxed)) return;
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() %
        1000000;
    struct tm tm_buf{};
    localtime_r(&t, &tm_buf);
    char ts[32];
    std::snprintf(ts, sizeof(ts), "%02d:%02d:%02d.%06ld", tm_buf.tm_hour, tm_buf.tm_min,
                  tm_buf.tm_sec, (long)us);
    const char* base = file;
    for (const char* p = file; *p; ++p)
        if (*p == '/') base = p + 1;
    std::fprintf(stderr, "[%s][umc][%s][tid=%zu][%s:%d] ", ts, LevelStr(lv),
                 std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFFF, base, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}
}  // namespace detail

inline void SetDebugEnabled(bool on)
{
    detail::DebugEnabledRef().store(on, std::memory_order_relaxed);
}

inline bool IsDebugEnabled() { return detail::DebugEnabledRef().load(std::memory_order_relaxed); }

}  // namespace umc::comm::log

#define UB_LOG_ERROR(fmt, ...)                                                               \
    ::umc::comm::log::detail::Emit(::umc::comm::log::Level::kError, __FILE__, __LINE__, fmt, \
                                   ##__VA_ARGS__)
#define UB_LOG_WARN(fmt, ...)                                                               \
    ::umc::comm::log::detail::Emit(::umc::comm::log::Level::kWarn, __FILE__, __LINE__, fmt, \
                                   ##__VA_ARGS__)
#define UB_LOG_INFO(fmt, ...)                                                               \
    ::umc::comm::log::detail::Emit(::umc::comm::log::Level::kInfo, __FILE__, __LINE__, fmt, \
                                   ##__VA_ARGS__)
#define UB_LOG_DEBUG(fmt, ...)                                                               \
    ::umc::comm::log::detail::Emit(::umc::comm::log::Level::kDebug, __FILE__, __LINE__, fmt, \
                                   ##__VA_ARGS__)

#define RDMA_LOG_ERROR UB_LOG_ERROR
#define RDMA_LOG_WARN UB_LOG_WARN
#define RDMA_LOG_INFO UB_LOG_INFO
#define RDMA_LOG_DEBUG UB_LOG_DEBUG
