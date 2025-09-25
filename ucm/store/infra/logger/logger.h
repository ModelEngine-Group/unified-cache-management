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
#ifndef UNIFIEDCACHE_LOGGER_H
#define UNIFIEDCACHE_LOGGER_H

#include <cstddef>
#include <fmt/format.h>
#include <string>

namespace UC::Logger {

enum class Level { DEBUG, INFO, WARN, ERROR };
struct SourceLocation {
    const char* file = "";
    const char* func = "";
    const int32_t line = 0;
};
class ILogger {
public:
    virtual ~ILogger() = default;
    template <typename... Args>
    void Log(Level&& lv, SourceLocation&& loc, fmt::format_string<Args...> fmt, Args&&... args)
    {
        this->Log(std::move(lv), std::move(loc), fmt::format(fmt, std::forward<Args>(args)...));
    }

protected:
    virtual void Log(Level&& lv, SourceLocation&& loc, std::string&& msg) = 0;
};

ILogger* Make();

} // namespace UC::Logger

#define UC_SOURCE_LOCATION {__FILE__, __FUNCTION__, __LINE__}
#define UC_LOG(lv, fmt, ...)                                                                       \
    UC::Logger::Make()->Log(lv, UC_SOURCE_LOCATION, FMT_STRING(fmt), ##__VA_ARGS__)
#define UC_DEBUG(fmt, ...) UC_LOG(UC::Logger::Level::DEBUG, fmt, ##__VA_ARGS__)
#define UC_INFO(fmt, ...) UC_LOG(UC::Logger::Level::INFO, fmt, ##__VA_ARGS__)
#define UC_WARN(fmt, ...) UC_LOG(UC::Logger::Level::WARN, fmt, ##__VA_ARGS__)
#define UC_ERROR(fmt, ...) UC_LOG(UC::Logger::Level::ERROR, fmt, ##__VA_ARGS__)

#endif
