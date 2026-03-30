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
constexpr uint32_t RATE_LIMIT_MAX_LOGS_PER_WINDOW = 3;
constexpr uint32_t kRateLimitCountBits = 2;
constexpr uint64_t kRateLimitCountMask = (1u << kRateLimitCountBits) - 1u;
constexpr size_t kHashMixMagic = 0x9e3779b97f4a7c15ULL;
constexpr size_t kHashShiftLeft = 12;
constexpr size_t kHashShiftRight = 4;
static spdlog::level::level_enum SpdLevels[] = {spdlog::level::debug, spdlog::level::info,
                                                spdlog::level::warn, spdlog::level::err,
                                                spdlog::level::critical};

void Logger::Log(Level&& lv, SourceLocation&& loc, std::string&& msg)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    this->logger_ = this->Make();
    this->logger_->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
}

inline uint64_t GetCurrentTimeMs()
{
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    return ms.time_since_epoch().count();
}

bool Logger::FilterCallSite(const char* file, int line)
{
    uint64_t now = GetCurrentTimeMs();
    const std::string_view fv(file);
    std::hash<std::string_view> h;
    size_t x = h(fv);
    x ^= static_cast<size_t>(line) + kHashMixMagic + (x << kHashShiftLeft) + (x >> kHashShiftRight);
    const uint64_t full_hash = static_cast<uint64_t>(x);
    const size_t slot_idx = static_cast<size_t>(full_hash % HASH_SLOT_NUM);
    // key_tag=0 is reserved for empty; so shift by +1.
    const uint64_t key_tag = full_hash + 1u;

    auto& slot = hash_slots_[slot_idx];
    std::atomic<uint64_t>* rate_state = nullptr;

    // 1) Lookup: find an existing chain entry with the same key.
    for (size_t i = 0; i < HASH_CHAIN_LEN; ++i) {
        uint64_t stored = slot.chain_entries[i].key_hash.load(std::memory_order_relaxed);
        if (stored == key_tag) {
            rate_state = &slot.chain_entries[i].rate_limit_state;
            break;
        }
    }

    // 2) Insert: if key not found, try to claim an empty entry.
    if (rate_state == nullptr) {
        for (size_t i = 0; i < HASH_CHAIN_LEN; ++i) {
            uint64_t expected_empty = 0;
            if (slot.chain_entries[i].key_hash.compare_exchange_strong(expected_empty, key_tag,
                                                                       std::memory_order_relaxed,
                                                                       std::memory_order_relaxed)) {
                rate_state = &slot.chain_entries[i].rate_limit_state;
                break;
            }
        }
    }

    // 3) Evict: if the chain is full, overwrite a deterministic entry.
    if (rate_state == nullptr) {
        const size_t evict_idx = static_cast<size_t>(key_tag % HASH_CHAIN_LEN);
        rate_state = &slot.chain_entries[evict_idx].rate_limit_state;
        slot.chain_entries[evict_idx].key_hash.store(key_tag, std::memory_order_relaxed);
        slot.chain_entries[evict_idx].rate_limit_state.store(0, std::memory_order_relaxed);
    }

    uint64_t s = rate_state->load(std::memory_order_relaxed);
    const uint64_t window_start = s >> kRateLimitCountBits;
    const uint32_t count = static_cast<uint32_t>(s & kRateLimitCountMask);

    if (s == 0 || now - window_start > LIMIT_THRESHOLD_MS) {
        const uint64_t desired = (now << kRateLimitCountBits) | 1u;
        if (rate_state->compare_exchange_strong(s, desired, std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
            return true;
        }
        return false;
    }

    if (count >= RATE_LIMIT_MAX_LOGS_PER_WINDOW) { return false; }
    const uint64_t desired =
        (window_start << kRateLimitCountBits) | static_cast<uint64_t>(count + 1u);
    if (rate_state->compare_exchange_strong(s, desired, std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
        return true;
    }
    return false;
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
