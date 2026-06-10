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

#include <algorithm>
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
constexpr uint32_t kRateLimitCountBits = 2;
constexpr uint64_t kRateLimitCountMask = (1u << kRateLimitCountBits) - 1u;
constexpr size_t kHashMixMagic = 0x9e3779b97f4a7c15ULL;
constexpr size_t kHashShiftLeft = 12;
constexpr size_t kHashShiftRight = 4;
static spdlog::level::level_enum SpdLevels[] = {spdlog::level::debug, spdlog::level::info,
                                                spdlog::level::warn, spdlog::level::err,
                                                spdlog::level::critical};

/* The async logger formats messages on a background thread after the caller
 * has returned, but spdlog::source_loc only stores raw char pointers; intern
 * file/function names so those pointers stay valid for the process lifetime. */
const char* InternSourceString(std::string&& s)
{
    static std::mutex mtx;
    static std::unordered_set<std::string> pool;
    std::lock_guard<std::mutex> lg(mtx);
    return pool.insert(std::move(s)).first->c_str();
}

void Logger::Log(Level&& lv, SourceLocation&& loc, std::string&& msg)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    auto logger = this->Make();
    logger->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
}

void Logger::LogFileOnly(Level&& lv, SourceLocation&& loc, std::string&& msg)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    this->Make();
    auto logger = this->file_logger_;
    if (!logger) { return; }
    logger->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
}

inline uint64_t GetCurrentTimeMs()
{
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    return ms.time_since_epoch().count();
}

bool Logger::FilterCallSite(const char* file, int line)
{
    if (!rate_limit_enabled_) { return true; }

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

    if (s == 0 || now - window_start > rate_limit_window_ms_) {
        const uint64_t desired = (now << kRateLimitCountBits) | 1u;
        if (rate_state->compare_exchange_strong(s, desired, std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
            return true;
        }
        return false;
    }

    if (count >= rate_limit_max_logs_) { return false; }
    const uint64_t desired =
        (window_start << kRateLimitCountBits) | static_cast<uint64_t>(count + 1u);
    if (rate_state->compare_exchange_strong(s, desired, std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
        return true;
    }
    return false;
}

static bool EnvFlag(const char* name, bool defaultValue)
{
    auto value = spdlog::details::os::getenv(name);
    if (value.empty()) { return defaultValue; }
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return value != "false" && value != "0" && value != "off";
}

std::shared_ptr<spdlog::logger> Logger::Make()
{
    if (this->logger_) { return this->logger_; }
    std::lock_guard<std::mutex> lg(this->mutex_);
    if (this->logger_) { return this->logger_; }
    std::string pid = std::to_string(getpid());
    std::string log_path = this->path_ + "/" + pid + "/ucm.log";
    const std::string name = "UC";
    try {
        if (!spdlog::thread_pool()) { spdlog::init_thread_pool(8192, 1); }
        auto tp = spdlog::thread_pool();

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(console_sink);

        spdlog::sink_ptr file_sink = nullptr;
        if (EnvFlag("UCM_LOG_TO_FILE", true)) {
            file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_path, this->max_size_, this->max_files_);
            sinks.push_back(file_sink);
        }

        auto logger = std::make_shared<spdlog::async_logger>(
            name, sinks.begin(), sinks.end(), tp, spdlog::async_overflow_policy::overrun_oldest);
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%n][%^%L%$] %v [%P,%t][%s:%#,%!]");

        auto level_str = spdlog::details::os::getenv("UCM_LOG_LEVEL");
        if (level_str.empty()) { level_str = spdlog::details::os::getenv("UC_LOGGER_LEVEL"); }
        if (!level_str.empty()) {
            auto level = spdlog::level::from_str(level_str);
            if (level != spdlog::level::off || level_str == "off") { logger->set_level(level); }
        }
        logger->flush_on(spdlog::level::warn);
        spdlog::register_logger(logger);

        if (file_sink) {
            auto file_logger = std::make_shared<spdlog::async_logger>(
                name + "_FILE", file_sink, tp, spdlog::async_overflow_policy::overrun_oldest);
            file_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%n][%^%L%$] %v [%P,%t][%s:%#,%!]");
            file_logger->set_level(logger->level());
            file_logger->flush_on(spdlog::level::warn);
            spdlog::register_logger(file_logger);
            this->file_logger_ = file_logger;
        }

        spdlog::flush_every(std::chrono::seconds(1));
        this->logger_ = logger;
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
    if (this->file_logger_) { this->file_logger_->flush(); }
}

bool Logger::IsEnabledFor(Level lv)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    if (this->logger_) { return this->logger_->should_log(level); }
    return false;
}

void Logger::LoadRateLimitConfig()
{
    auto enable_str = spdlog::details::os::getenv("UCM_LOG_RATE_LIMIT_ENABLE");
    if (!enable_str.empty()) {
        std::string lower = enable_str;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        rate_limit_enabled_ = (lower != "false" && lower != "0" && lower != "off");
    }

    auto window_str = spdlog::details::os::getenv("UCM_LOG_RATE_LIMIT_WINDOW_MS");
    if (!window_str.empty()) {
        try {
            rate_limit_window_ms_ = std::stoull(window_str);
        } catch (...) {
            rate_limit_window_ms_ = 60000;
        }
    }

    auto max_logs_str = spdlog::details::os::getenv("UCM_LOG_RATE_LIMIT_MAX_LOGS");
    if (!max_logs_str.empty()) {
        try {
            auto val = std::stoul(max_logs_str);
            rate_limit_max_logs_ = static_cast<uint32_t>(
                std::min(val, static_cast<unsigned long>(kRateLimitCountMask)));
        } catch (...) {
            rate_limit_max_logs_ = 3;
        }
    }
}

}  // namespace UC::Logger
