/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef LOGGER_H
#define LOGGER_H

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <memory>

class LoggerImpl;
class Logger {
public:
    enum class Level : uint8_t { DEBUG, INFO, WARN, ERROR, CRITICAL };
    struct SourceLocation {
        const char* file = "";
        const char* func = "";
        const int32_t line = 0;
    };

    static Logger& Instance()
    {
        static Logger instance;
        return instance;
    }
    ~Logger();
    template <typename... Args>
    void Log(Level lv, const SourceLocation& loc, fmt::format_string<Args...> fmt, Args&&... args)
    {
        LogInternal(lv, loc, fmt::format(fmt, std::forward<Args>(args)...));
    }

private:
    Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    void LogInternal(Level lv, const SourceLocation& loc, const std::string& message);

    std::unique_ptr<LoggerImpl> impl_;
};

#define LOG_SOURCE_LOCATION {__FILE__, __FUNCTION__, __LINE__}
#define __LOG(lv, fmt, ...) \
    Logger::Instance().Log(lv, LOG_SOURCE_LOCATION, FMT_STRING(fmt), ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) __LOG(Logger::Level::DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) __LOG(Logger::Level::INFO, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) __LOG(Logger::Level::WARN, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) __LOG(Logger::Level::ERROR, fmt, ##__VA_ARGS__)
#define LOG_CRITICAL(fmt, ...) __LOG(Logger::Level::CRITICAL, fmt, ##__VA_ARGS__)

#endif  // LOGGER_H
