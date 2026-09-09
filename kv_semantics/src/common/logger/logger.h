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
#ifndef KV_SEMANTICS_LOGGER_H
#define KV_SEMANTICS_LOGGER_H

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "spdlog_logger.h"

namespace kv::logger {

void Log(Level lv, std::string file, std::string func, int line, std::string msg);
void LogRateLimit(Level lv, std::string file, std::string func, int line, std::string msg);

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
    LogRateLimit(lv, std::string(loc.file), std::string(loc.func), loc.line, std::move(msg));
}

void Setup(const std::string& path, int max_files, int max_size);
void Flush();
bool isEnabledFor(Level lv);

}  // namespace kv::logger
#define KV_SOURCE_LOCATION {__FILE__, __FUNCTION__, __LINE__}
#define KV_LOG_UNLIMITED(lv, fmt, ...) \
    kv::logger::Log(lv, KV_SOURCE_LOCATION, FMT_STRING(fmt), ##__VA_ARGS__)
#define KV_LOG(lv, fmt, ...) \
    kv::logger::LogRateLimit(lv, KV_SOURCE_LOCATION, FMT_STRING(fmt), ##__VA_ARGS__)
#define KV_DEBUG_UNLIMITED(fmt, ...) KV_LOG_UNLIMITED(kv::logger::Level::DEBUG, fmt, ##__VA_ARGS__)
#define KV_INFO_UNLIMITED(fmt, ...) KV_LOG_UNLIMITED(kv::logger::Level::INFO, fmt, ##__VA_ARGS__)
#define KV_WARN_UNLIMITED(fmt, ...) KV_LOG_UNLIMITED(kv::logger::Level::WARN, fmt, ##__VA_ARGS__)
#define KV_ERROR_UNLIMITED(fmt, ...) KV_LOG_UNLIMITED(kv::logger::Level::ERROR, fmt, ##__VA_ARGS__)
#define KV_DEBUG(fmt, ...) KV_LOG(kv::logger::Level::DEBUG, fmt, ##__VA_ARGS__)
#define KV_INFO(fmt, ...) KV_LOG(kv::logger::Level::INFO, fmt, ##__VA_ARGS__)
#define KV_WARN(fmt, ...) KV_LOG(kv::logger::Level::WARN, fmt, ##__VA_ARGS__)
#define KV_ERROR(fmt, ...) KV_LOG(kv::logger::Level::ERROR, fmt, ##__VA_ARGS__)
#endif
