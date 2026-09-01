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
#include <cerrno>
#include <chrono>
#include <memory>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/cfg/helpers.h>
#include <spdlog/details/os.h>
#include <spdlog/details/periodic_worker.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include "compress_rotate_file_sink.h"
#include "logger.h"

namespace UC::Logger {
namespace {
constexpr uint32_t kRateLimitCountBits = 2;
constexpr uint64_t kRateLimitCountMask = (1u << kRateLimitCountBits) - 1u;
constexpr size_t kHashMixMagic = 0x9e3779b97f4a7c15ULL;
constexpr size_t kHashShiftLeft = 12;
constexpr size_t kHashShiftRight = 4;
constexpr size_t kAsyncQueueSize = 8192;
constexpr size_t kAsyncWorkerCount = 1;

spdlog::level::level_enum SpdLevels[] = {spdlog::level::debug, spdlog::level::info,
                                         spdlog::level::warn, spdlog::level::err,
                                         spdlog::level::critical};

class ForkSafeStdoutSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
    ForkSafeStdoutSink() : color_enabled_(isatty(STDOUT_FILENO) != 0) {}

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override
    {
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);

        if (!color_enabled_ || msg.color_range_end <= msg.color_range_start ||
            msg.color_range_end > formatted.size()) {
            WriteAll(formatted.data(), formatted.size());
            return;
        }

        const std::string_view color = ColorFor(msg.level);
        WriteAll(formatted.data(), msg.color_range_start);
        WriteAll(color.data(), color.size());
        WriteAll(formatted.data() + msg.color_range_start,
                 msg.color_range_end - msg.color_range_start);
        constexpr std::string_view reset = "\033[m";
        WriteAll(reset.data(), reset.size());
        WriteAll(formatted.data() + msg.color_range_end, formatted.size() - msg.color_range_end);
    }

    void flush_() override {}

private:
    static void WriteAll(const char* data, size_t size)
    {
        size_t offset = 0;
        while (offset < size) {
            const ssize_t written = write(STDOUT_FILENO, data + offset, size - offset);
            if (written > 0) {
                offset += static_cast<size_t>(written);
            } else if (written < 0 && errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
    }

    static std::string_view ColorFor(spdlog::level::level_enum level)
    {
        switch (level) {
            case spdlog::level::trace: return "\033[37m";
            case spdlog::level::debug: return "\033[36m";
            case spdlog::level::info: return "\033[32m";
            case spdlog::level::warn: return "\033[33m\033[1m";
            case spdlog::level::err: return "\033[31m\033[1m";
            case spdlog::level::critical: return "\033[1m\033[41m";
            default: return "";
        }
    }

    bool color_enabled_;
};

bool EnvFlag(const char* name, bool default_value)
{
    auto value = spdlog::details::os::getenv(name);
    if (value.empty()) { return default_value; }
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return value != "false" && value != "0" && value != "off";
}

void ConfigureLogger(const std::shared_ptr<spdlog::logger>& logger)
{
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%n][%^%L%$] %v [%P,%t][%s:%#,%!]");
    auto level_str = spdlog::details::os::getenv("UCM_LOG_LEVEL");
    if (level_str.empty()) { level_str = spdlog::details::os::getenv("UC_LOGGER_LEVEL"); }
    if (!level_str.empty()) {
        auto level = spdlog::level::from_str(level_str);
        if (level != spdlog::level::off || level_str == "off") { logger->set_level(level); }
    }
    logger->flush_on(spdlog::level::warn);
}

uint64_t GetCurrentTimeMs()
{
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    return ms.time_since_epoch().count();
}
}  // namespace

struct Logger::Backend {
    Backend(int32_t process_id, const std::string& path, int max_files, int max_size,
            bool rate_limit_enabled, uint64_t rate_limit_window_ms, uint32_t rate_limit_max_logs,
            bool forked_child)
        : pid(process_id),
          path_(path),
          max_files_(max_files),
          max_size_(max_size),
          rate_limit_enabled_(rate_limit_enabled),
          rate_limit_window_ms_(rate_limit_window_ms),
          rate_limit_max_logs_(rate_limit_max_logs),
          forked_child_(forked_child)
    {
        MakeMainLogger();
        if (thread_pool_) {
            flusher_ = std::make_unique<spdlog::details::periodic_worker>([this] { Flush(); },
                                                                          std::chrono::seconds(1));
        }
    }

    void RegisterMainLogger()
    {
        try {
            spdlog::register_logger(logger_);
        } catch (...) {
        }
    }

    SourceLocation InternSourceLocation(std::string&& file, std::string&& func, int line)
    {
        std::lock_guard<std::mutex> lock(source_mutex_);
        const char* file_ptr = source_pool_.insert(std::move(file)).first->c_str();
        const char* func_ptr = source_pool_.insert(std::move(func)).first->c_str();
        return SourceLocation{file_ptr, func_ptr, line};
    }

    void Log(Level lv, const SourceLocation& loc, std::string&& msg)
    {
        auto level = SpdLevels[fmt::underlying(lv)];
        logger_->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level, std::move(msg));
    }

    void LogFileOnly(Level lv, const SourceLocation& loc, std::string&& msg)
    {
        auto capture_logger = MakeCaptureLogger();
        if (!capture_logger) { return; }
        auto level = SpdLevels[fmt::underlying(lv)];
        capture_logger->log(spdlog::source_loc{loc.file, loc.line, loc.func}, level,
                            std::move(msg));
    }

    std::shared_ptr<spdlog::logger> MakeCaptureLogger()
    {
        if (!file_enabled_) { return nullptr; }
        std::lock_guard<std::mutex> lock(capture_mutex_);
        if (file_logger_) { return file_logger_; }

        std::string log_path = path_ + "/vllm-" + std::to_string(pid) + ".log";
        try {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_path, max_size_, max_files_);
            if (thread_pool_) {
                file_logger_ = std::make_shared<spdlog::async_logger>(
                    "VLLM", file_sink, thread_pool_, spdlog::async_overflow_policy::overrun_oldest);
            } else {
                file_logger_ = std::make_shared<spdlog::logger>("VLLM", file_sink);
            }
            ConfigureLogger(file_logger_);
            file_logger_->set_level(logger_->level());
            if (!thread_pool_) { file_logger_->flush_on(spdlog::level::trace); }
            if (!forked_child_) {
                try {
                    spdlog::register_logger(file_logger_);
                } catch (...) {
                }
            }
        } catch (...) {
            file_logger_ = nullptr;
        }
        return file_logger_;
    }

    void Flush()
    {
        logger_->flush();
        std::shared_ptr<spdlog::logger> file_logger;
        {
            std::lock_guard<std::mutex> lock(capture_mutex_);
            file_logger = file_logger_;
        }
        if (file_logger) { file_logger->flush(); }
    }

    bool FilterCallSite(const char* file, int line)
    {
        if (!rate_limit_enabled_) { return true; }

        uint64_t now = GetCurrentTimeMs();
        const std::string_view fv(file);
        std::hash<std::string_view> h;
        size_t x = h(fv);
        x ^= static_cast<size_t>(line) + kHashMixMagic + (x << kHashShiftLeft) +
             (x >> kHashShiftRight);
        const uint64_t full_hash = static_cast<uint64_t>(x);
        const uint64_t key_tag = full_hash + 1u;
        auto& slot = hash_slots_[static_cast<size_t>(full_hash % HASH_SLOT_NUM)];
        std::atomic<uint64_t>* rate_state = nullptr;

        for (size_t i = 0; i < HASH_CHAIN_LEN; ++i) {
            uint64_t stored = slot.chain_entries[i].key_hash.load(std::memory_order_relaxed);
            if (stored == key_tag) {
                rate_state = &slot.chain_entries[i].rate_limit_state;
                break;
            }
        }

        if (rate_state == nullptr) {
            for (size_t i = 0; i < HASH_CHAIN_LEN; ++i) {
                uint64_t expected_empty = 0;
                if (slot.chain_entries[i].key_hash.compare_exchange_strong(
                        expected_empty, key_tag, std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    rate_state = &slot.chain_entries[i].rate_limit_state;
                    break;
                }
            }
        }

        if (rate_state == nullptr) {
            const size_t evict_idx = static_cast<size_t>(key_tag % HASH_CHAIN_LEN);
            rate_state = &slot.chain_entries[evict_idx].rate_limit_state;
            slot.chain_entries[evict_idx].key_hash.store(key_tag, std::memory_order_relaxed);
            slot.chain_entries[evict_idx].rate_limit_state.store(0, std::memory_order_relaxed);
        }

        uint64_t state = rate_state->load(std::memory_order_relaxed);
        const uint64_t window_start = state >> kRateLimitCountBits;
        const uint32_t count = static_cast<uint32_t>(state & kRateLimitCountMask);

        if (state == 0 || now - window_start > rate_limit_window_ms_) {
            const uint64_t desired = (now << kRateLimitCountBits) | 1u;
            return rate_state->compare_exchange_strong(state, desired, std::memory_order_relaxed,
                                                       std::memory_order_relaxed);
        }

        if (count >= rate_limit_max_logs_) { return false; }
        const uint64_t desired =
            (window_start << kRateLimitCountBits) | static_cast<uint64_t>(count + 1u);
        return rate_state->compare_exchange_strong(state, desired, std::memory_order_relaxed,
                                                   std::memory_order_relaxed);
    }

    const int32_t pid;
    std::shared_ptr<spdlog::details::thread_pool> thread_pool_;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex source_mutex_;
    std::unordered_set<std::string> source_pool_;
    std::mutex capture_mutex_;
    std::shared_ptr<spdlog::logger> file_logger_;
    std::unique_ptr<spdlog::details::periodic_worker> flusher_;

private:
    void MakeMainLogger()
    {
        std::string log_path = path_ + "/ucm-" + std::to_string(pid) + ".log";
        file_enabled_ = EnvFlag("UCM_LOG_TO_FILE", true);
        try {
            std::vector<spdlog::sink_ptr> sinks;
            sinks.push_back(std::make_shared<ForkSafeStdoutSink>());
            if (file_enabled_) {
                sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    log_path, max_size_, max_files_));
            }
            thread_pool_ =
                std::make_shared<spdlog::details::thread_pool>(kAsyncQueueSize, kAsyncWorkerCount);
            logger_ = std::make_shared<spdlog::async_logger>(
                "UC", sinks.begin(), sinks.end(), thread_pool_,
                spdlog::async_overflow_policy::overrun_oldest);
        } catch (...) {
            thread_pool_ = nullptr;
            logger_ =
                std::make_shared<spdlog::logger>("UC", std::make_shared<ForkSafeStdoutSink>());
        }
        ConfigureLogger(logger_);
        if (!thread_pool_) { logger_->flush_on(spdlog::level::trace); }
    }

    std::string path_;
    int max_files_;
    int max_size_;
    bool file_enabled_{true};
    bool rate_limit_enabled_;
    uint64_t rate_limit_window_ms_;
    uint32_t rate_limit_max_logs_;
    bool forked_child_;
    std::array<SlotData, HASH_SLOT_NUM> hash_slots_;
};

Logger::Logger() : creation_pid_(static_cast<int32_t>(getpid()))
{
    register_at_exit();
    LoadRateLimitConfig();
}

Logger::~Logger()
{
    Backend* backend = backend_.load(std::memory_order_acquire);
    if (backend != nullptr && backend->pid == static_cast<int32_t>(getpid())) { delete backend; }
}

Logger::Backend* Logger::GetBackend()
{
    static_assert(decltype(backend_)::is_always_lock_free);
    const int32_t pid = static_cast<int32_t>(getpid());
    for (;;) {
        Backend* inherited = backend_.load(std::memory_order_acquire);
        if (inherited != nullptr && inherited->pid == pid) { return inherited; }

        const bool forked_child = creation_pid_ != pid;
        std::unique_ptr<Backend> fresh =
            std::make_unique<Backend>(pid, path_, max_files_, max_size_, rate_limit_enabled_,
                                      rate_limit_window_ms_, rate_limit_max_logs_, forked_child);
        Backend* expected = inherited;
        if (backend_.compare_exchange_strong(expected, fresh.get(), std::memory_order_release,
                                             std::memory_order_acquire)) {
            Backend* installed = fresh.release();
            if (inherited == nullptr && !forked_child) { installed->RegisterMainLogger(); }
            // The inherited Backend is intentionally leaked in the child so its copied mutexes
            // and missing worker threads are never destroyed or joined.
            return installed;
        }
    }
}

void Logger::Log(Level lv, const SourceLocation& loc, std::string&& msg)
{
    GetBackend()->Log(lv, loc, std::move(msg));
}

void Logger::LogDynamic(Level lv, std::string&& file, std::string&& func, int line,
                        std::string&& msg)
{
    Backend* backend = GetBackend();
    SourceLocation loc = backend->InternSourceLocation(std::move(file), std::move(func), line);
    backend->Log(lv, loc, std::move(msg));
}

void Logger::LogRateLimit(Level lv, const SourceLocation& loc, std::string&& msg)
{
    Backend* backend = GetBackend();
    if (backend->FilterCallSite(loc.file, loc.line)) { backend->Log(lv, loc, std::move(msg)); }
}

void Logger::LogRateLimitDynamic(Level lv, std::string&& file, std::string&& func, int line,
                                 std::string&& msg)
{
    Backend* backend = GetBackend();
    if (!backend->FilterCallSite(file.c_str(), line)) { return; }
    SourceLocation loc = backend->InternSourceLocation(std::move(file), std::move(func), line);
    backend->Log(lv, loc, std::move(msg));
}

void Logger::LogFileOnlyDynamic(Level lv, std::string&& file, std::string&& func, int line,
                                std::string&& msg)
{
    Backend* backend = GetBackend();
    SourceLocation loc = backend->InternSourceLocation(std::move(file), std::move(func), line);
    backend->LogFileOnly(lv, loc, std::move(msg));
}

void Logger::Setup(const std::string& path, int max_files, int max_size)
{
    path_ = path;
    max_files_ = max_files;
    max_size_ = max_size * 1048576;
    GetBackend();
}

void Logger::Flush()
{
    Backend* backend = backend_.load(std::memory_order_acquire);
    if (backend != nullptr && backend->pid == static_cast<int32_t>(getpid())) { backend->Flush(); }
}

bool Logger::IsEnabledFor(Level lv)
{
    auto level = SpdLevels[fmt::underlying(lv)];
    return GetBackend()->logger_->should_log(level);
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
            rate_limit_window_ms_ = kDefaultRateLimitWindowMs;
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
