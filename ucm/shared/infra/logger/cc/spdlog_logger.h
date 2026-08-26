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
#ifndef UNIFIEDCACHE_INFRA_LOGGER_SPDLOG_LOGGER_H
#define UNIFIEDCACHE_INFRA_LOGGER_SPDLOG_LOGGER_H
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <mutex>
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_set>
namespace UC::Logger {

constexpr size_t HASH_SLOT_NUM = 512;
constexpr size_t HASH_CHAIN_LEN = 4;
enum class Level { DEBUG, INFO, WARN, ERROR, CRITICAL };
struct SourceLocation {
    const char* file = "";
    const char* func = "";
    const int32_t line = 0;
};

const char* InternSourceString(std::string&& s);

class Logger {
    static constexpr uint64_t kDefaultRateLimitWindowMs = 10000;
    std::shared_ptr<spdlog::logger> logger_;
    std::shared_ptr<spdlog::logger> file_logger_;
    std::mutex mutex_;
    bool rate_limit_enabled_{true};
    uint64_t rate_limit_window_ms_{kDefaultRateLimitWindowMs};
    uint32_t rate_limit_max_logs_{3};

public:
    Logger()
    {
        logger_ = nullptr;
        register_at_exit();
        LoadRateLimitConfig();
    }

    void LoadRateLimitConfig();

    void register_at_exit()
    {
        std::signal(SIGSEGV, &_signal_handler);
        std::signal(SIGABRT, &_signal_handler);
        std::signal(SIGFPE, &_signal_handler);
        std::signal(SIGILL, &_signal_handler);
        std::signal(SIGINT, &_signal_handler);
    }

    static void _signal_handler(int signum)
    {
        // 只允许第一次进入时刷日志，防止 Flush 期间又来一个不同的致命信号导致重入
        static std::atomic_flag flushed = ATOMIC_FLAG_INIT;
        if (!flushed.test_and_set()) { Logger::GetInstance().Flush(); }

        // 恢复默认动作后重新发给自己：
        // SIGSEGV/SIGABRT/SIGFPE/SIGILL 的默认动作是 Term+Core，会生成 core dump；
        // SIGINT 的默认动作是 Term，不产生 core dump。
        // 是否 dump 由内核按信号种类决定，不需要代码区分。
        std::signal(signum, SIG_DFL);
        std::raise(signum);
    }

    void Log(Level&& lv, SourceLocation&& loc, std::string&& msg);
    void LogFileOnly(Level&& lv, SourceLocation&& loc, std::string&& msg);
    void Setup(const std::string& path, int max_files, int max_size);
    void Flush();

    static Logger& GetInstance()
    {
        static Logger inst;
        return inst;
    }

    bool IsEnabledFor(Level lv);

    bool FilterCallSite(const char* file, int line);

private:
    struct ChainEntryData {
        std::atomic<uint64_t> key_hash{0};
        std::atomic<uint64_t> rate_limit_state{0};
    };

    struct SlotData {
        std::array<ChainEntryData, HASH_CHAIN_LEN> chain_entries;
    };

    std::array<SlotData, HASH_SLOT_NUM> hash_slots_;

    std::shared_ptr<spdlog::logger> Make();
    std::shared_ptr<spdlog::logger> MakeCapture();
    bool file_enabled_{true};
    std::string path_{"log"};
    int max_files_{3};
    int max_size_{5 * 1048576};  // 5MB
};

}  // namespace UC::Logger

#endif
