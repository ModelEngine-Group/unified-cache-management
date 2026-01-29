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

#include <chrono>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/cfg/helpers.h>
#include <spdlog/details/os.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include "compress_rotate_file_sink.h"
#include "logger.h"

namespace UC::Logger {

static spdlog::level::level_enum SpdLevels[] = {spdlog::level::debug, spdlog::level::info,
                                                spdlog::level::warn, spdlog::level::err};

void Logger::Log(Level&& lv, SourceLocation&& loc, std::string&& msg)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    this->logger_ = this->Make();
    this->logger_->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
}

std::shared_ptr<spdlog::logger> Logger::Make()
{
    if (this->logger_) { return this->logger_; }
    std::lock_guard<std::mutex> lg(this->mutex_);
    if (this->logger_) { return this->logger_; }
    std::string pid = std::to_string(getpid());
    std::string log_path = this->path_ + "/" + pid + "/ucm.log";
    const std::string name = "UC";
    const std::string envLevel = name + "_LOGGER_LEVEL";
    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_path, this->max_size_, this->max_files_);
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(console_sink);
        sinks.push_back(file_sink);
        this->logger_ = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
        this->logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%n][%^%L%$] %v [%P,%t][%s:%#,%!]");
        auto level = spdlog::details::os::getenv(envLevel.c_str());
        if (!level.empty()) { spdlog::cfg::helpers::load_levels(level); }
        spdlog::register_logger(this->logger_);
        return this->logger_;
    } catch (...) {
        return spdlog::default_logger();
    }
}

void Logger::Setup(const std::string& path, int max_files, int max_size)
{
    this->path_ = path;
    this->max_files_ = max_files;
    this->max_size_ = max_size * 1048576;
    this->logger_ = this->Make();
}

void Logger::Flush()
{
    std::lock_guard<std::mutex> lg(this->mutex_);
    if (this->logger_) { this->logger_->flush(); }
}

}  // namespace UC::Logger
