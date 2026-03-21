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
constexpr uint64_t LIMIT_THRESHOLD_MS = 60000;
static spdlog::level::level_enum SpdLevels[] = {spdlog::level::debug, spdlog::level::info,
                                                spdlog::level::warn, spdlog::level::err,
                                                spdlog::level::critical};

void Logger::Log(Level&& lv, SourceLocation&& loc, std::string&& msg)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    this->logger_ = this->Make();
    this->logger_->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
}

inline uint64_t get_current_time_ms()
{
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    return ms.time_since_epoch().count();
}

bool Logger::FilterCallSite(const char* file, int line, std::string_view ori_fmt)
{
    uint64_t now = get_current_time_ms();
    const std::string_view fv(file ? file : "");
    std::hash<std::string_view> h;
    size_t x = h(fv);
    x ^= static_cast<size_t>(line) + 0x9e3779b97f4a7c15ULL + (x << 12) + (x >> 4);
    x ^= h(ori_fmt) + 0x9e3779b97f4a7c15ULL + (x << 12) + (x >> 4);
    const size_t slot_idx = x % HASH_SLOT_NUM;
    std::atomic<uint64_t>& last_time = hash_slots_[slot_idx].last_time;
    uint64_t last = last_time.load(std::memory_order_relaxed);
    for (;;) {
        if (now - last <= LIMIT_THRESHOLD_MS) { return false; }
        if (last_time.compare_exchange_weak(last, now, std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
            return true;
        }
    }
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
        auto level_str = spdlog::details::os::getenv(envLevel.c_str());
        if (!level_str.empty()) {
            auto level = spdlog::level::from_str(level_str);
            if (level != spdlog::level::off || level_str == "off") {
                this->logger_->set_level(level);
            }
        }
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

bool Logger::IsEnabledFor(Level lv)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    if (this->logger_) { return this->logger_->should_log(level); }
    return false;
}

}  // namespace UC::Logger
