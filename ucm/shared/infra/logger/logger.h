/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_INFRA_LOGGER_H
#define UNIFIEDCACHE_INFRA_LOGGER_H

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "cc/spdlog_logger.h"

namespace UC::Logger {

void Log(Level lv, std::string file, std::string func, int line, std::string msg);
void LogRateLimit(Level lv, std::string file, std::string func, int line, std::string msg,
                  std::string_view ori_fmt);

template <typename... Args>
void Log(Level lv, const SourceLocation& loc, fmt::format_string<Args...> fmt, Args&&... args)
{
    std::string msg = fmt::format(fmt, std::forward<Args>(args)...);
    Log(lv, std::string(loc.file), std::string(loc.func), loc.line, std::move(msg));
}

template <typename... Args>
void LogRateLimit(Level lv, const SourceLocation& loc, fmt::format_string<Args...> fmt,
                  Args&&... args)
{
    std::string msg = fmt::format(fmt, std::forward<Args>(args)...);
    fmt::string_view sv = fmt.get();
    LogRateLimit(lv, std::string(loc.file), std::string(loc.func), loc.line, std::move(msg),
                 std::string_view(sv.data(), sv.size()));
}

void Setup(const std::string& path, int max_files, int max_size);
void Flush();
bool isEnabledFor(Level lv);

}  // namespace UC::Logger
#define UC_SOURCE_LOCATION {__FILE__, __FUNCTION__, __LINE__}
#define UC_LOG(lv, fmt, ...) UC::Logger::Log(lv, UC_SOURCE_LOCATION, FMT_STRING(fmt), ##__VA_ARGS__)
#define UC_LOG_LIMIT(lv, fmt, ...) \
    UC::Logger::LogRateLimit(lv, UC_SOURCE_LOCATION, FMT_STRING(fmt), ##__VA_ARGS__)
#define UC_DEBUG(fmt, ...) UC_LOG(UC::Logger::Level::DEBUG, fmt, ##__VA_ARGS__)
#define UC_INFO(fmt, ...) UC_LOG(UC::Logger::Level::INFO, fmt, ##__VA_ARGS__)
#define UC_WARN(fmt, ...) UC_LOG(UC::Logger::Level::WARN, fmt, ##__VA_ARGS__)
#define UC_ERROR(fmt, ...) UC_LOG(UC::Logger::Level::ERROR, fmt, ##__VA_ARGS__)
#define UC_DEBUG_LIMIT(fmt, ...) UC_LOG_LIMIT(UC::Logger::Level::DEBUG, fmt, ##__VA_ARGS__)
#define UC_INFO_LIMIT(fmt, ...) UC_LOG_LIMIT(UC::Logger::Level::INFO, fmt, ##__VA_ARGS__)
#define UC_WARN_LIMIT(fmt, ...) UC_LOG_LIMIT(UC::Logger::Level::WARN, fmt, ##__VA_ARGS__)
#define UC_ERROR_LIMIT(fmt, ...) UC_LOG_LIMIT(UC::Logger::Level::ERROR, fmt, ##__VA_ARGS__)
#endif
