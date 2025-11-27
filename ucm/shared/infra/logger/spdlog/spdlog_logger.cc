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
#include <mutex>
#include <spdlog/cfg/helpers.h>
#include <spdlog/details/os.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include "logger/logger.h"

namespace UC::Logger {

static spdlog::level::level_enum SpdLevels[] = {spdlog::level::debug, spdlog::level::info,
                                                spdlog::level::warn, spdlog::level::err};

class SpdLogger : public ILogger {
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;

public:
    SpdLogger() : logger_{nullptr} {}

protected:
    void Log(Level&& lv, SourceLocation&& loc, std::string&& msg) override
    {
        auto logger = this->Make();
        auto level = SpdLevels[fmt::underlying(lv)];
        logger->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
    }

private:
    std::shared_ptr<spdlog::logger> Make()
    {
        if (this->logger_) { return this->logger_; }
        std::lock_guard<std::mutex> lg(this->mutex_);
        if (this->logger_) { return this->logger_; }
        const std::string name = "UC";
        const std::string envLevel = name + "_LOGGER_LEVEL";
        try {
            this->logger_ = spdlog::stdout_color_mt(name);
            this->logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%n][%^%L%$] %v [%P,%t][%s:%#,%!]");
            auto level = spdlog::details::os::getenv(envLevel.c_str());
            if (!level.empty()) { spdlog::cfg::helpers::load_levels(level); }
            return this->logger_;
        } catch (...) {
            return spdlog::default_logger();
        }
    }
};

ILogger* Make()
{
    static SpdLogger logger;
    return &logger;
}

} // namespace UC::Logger
