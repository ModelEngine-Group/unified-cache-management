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
#include <atomic>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <random>
#include <spdlog/spdlog.h>
#include <string_view>
#include <thread>
#include <vector>
#include "logger/logger.h"

using namespace UC::Logger;

namespace {
void CleanDir(const std::string& path)
{
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        std::cerr << "Failed to remove file: " << path << std::endl;
        std::cerr << "Error: " << ec.message() << std::endl;
        std::exit(1);
    }
}
}  // namespace

class UCLoggerPerfTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        CleanDir(test_log_dir_);
        std::filesystem::create_directories(test_log_dir_);
        std::cout << "test_log_path_: " << test_log_path_ << std::endl;
        logger_ = &Logger::GetInstance();
        logger_->Setup(test_log_dir_, 3, 1);  // 3 files, 1MB max size
    }

    static void TearDownTestSuite()
    {
        CleanDir(test_log_dir_);
        spdlog::drop_all();
    }

    static inline std::string test_log_dir_ = "log_perf_test";
    static inline std::string test_log_path_ = "log_perf_test/test_log.log";
    static inline Logger* logger_ = nullptr;
};
namespace {
static inline void PerfLogInfo() { UC_INFO_UNLIMITED("uc_logger_perf_same_site"); }

static inline void PerfLogInfoLimit() { UC_INFO("uc_logger_perf_same_site"); }

static inline void PerfLogInfoRandom(const std::string& content)
{
    Log(Level::INFO, "logger_perf_random.cc", "PerfLogInfoRandom", 100, std::move(content));
}
static inline void PerfLogInfoLimitRandom(const std::string& content)
{
    LogRateLimit(Level::INFO, "logger_perf_random.cc", "PerfLogInfoLimitRandom", 100,
                 std::string(content));
}

template <typename Fn>
static double BenchmarkMultiThreadNsPerCall(int threads, int iterations, Fn&& fn)
{
    std::vector<std::thread> workers;
    workers.reserve(threads);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) {}
            for (int i = 0; i < iterations; ++i) { fn(tid, i); }
        });
    }

    while (ready.load(std::memory_order_acquire) != threads) {}

    const auto begin = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& t : workers) { t.join(); }
    const auto end = std::chrono::steady_clock::now();

    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    const double total_calls = static_cast<double>(threads) * static_cast<double>(iterations);
    return static_cast<double>(total_ns) / total_calls;
}

}  // namespace

TEST_F(UCLoggerPerfTest, MultiThreadPerfUCInfoVsRateLimit)
{
    auto spdlog_logger = spdlog::get("UC");
    ASSERT_NE(spdlog_logger, nullptr);

    // Keep benchmark focused on `UC_INFO` vs `UC_INFO_LIMIT` overhead.
    // Message formatting still happens in the UC_* macros, but spdlog sinks should do nothing.
    const auto old_level = spdlog_logger->level();
    spdlog_logger->set_level(spdlog::level::off);

    const unsigned hc = std::thread::hardware_concurrency();
    const int threads = static_cast<int>(std::min<unsigned>(8, std::max<unsigned>(2, hc ? hc : 4)));
    const int iterations_per_thread = 2000;

    // Warmup to stabilize instruction-cache and initial rate-limit cache path.
    for (int i = 0; i < 32; ++i) {
        PerfLogInfo();
        PerfLogInfoLimit();
    }

    const double ns_per_call_info_limit = BenchmarkMultiThreadNsPerCall(
        threads, iterations_per_thread, [](int /*tid*/, int /*i*/) { PerfLogInfoLimit(); });
    const double ns_per_call_info = BenchmarkMultiThreadNsPerCall(
        threads, iterations_per_thread, [](int /*tid*/, int /*i*/) { PerfLogInfo(); });

    spdlog_logger->set_level(old_level);

    const double ratio = ns_per_call_info_limit / ns_per_call_info;
    RecordProperty("uc_info_avg_ns_per_call", std::to_string(ns_per_call_info));
    RecordProperty("uc_info_limit_avg_ns_per_call", std::to_string(ns_per_call_info_limit));
    RecordProperty("uc_info_limit_over_uc_info_ratio", std::to_string(ratio));

    std::cout << "[UCLoggerPerf] threads=" << threads
              << " iterations_per_thread=" << iterations_per_thread
              << " uc_info(ns/call)=" << ns_per_call_info
              << " uc_info_limit(ns/call)=" << ns_per_call_info_limit << " ratio=" << ratio
              << std::endl;

    // The test is informational: it reports the ratio so you can confirm
    // whether `UC_INFO_LIMIT` is worse under concurrent call patterns.
    ASSERT_TRUE(ratio <= 1.1);
}

TEST_F(UCLoggerPerfTest, MultiThreadPerfUCInfoVsRateLimitRandomContent)
{
    auto spdlog_logger = spdlog::get("UC");
    ASSERT_NE(spdlog_logger, nullptr);

    const auto old_level = spdlog_logger->level();
    spdlog_logger->set_level(spdlog::level::off);

    const unsigned hc = std::thread::hardware_concurrency();
    const int threads = static_cast<int>(std::min<unsigned>(8, std::max<unsigned>(2, hc ? hc : 4)));
    const int iterations_per_thread = 2000;

    // Pre-generate totally random payloads per thread/call so that the final
    // log message content is different for nearly every invocation.
    std::vector<std::vector<std::string>> payloads(threads,
                                                   std::vector<std::string>(iterations_per_thread));
    std::mt19937_64 rng(123456789ULL);
    std::uniform_int_distribution<int> len_dist(16, 64);
    std::uniform_int_distribution<int> ch_dist(0, 61);  // [0-9A-Za-z]

    auto gen_char = [&](int v) -> char {
        if (v < 10) { return static_cast<char>('0' + v); }
        v -= 10;
        if (v < 26) { return static_cast<char>('A' + v); }
        v -= 26;
        return static_cast<char>('a' + v);
    };

    for (int t = 0; t < threads; ++t) {
        for (int i = 0; i < iterations_per_thread; ++i) {
            const int len = len_dist(rng);
            std::string s;
            s.reserve(static_cast<std::size_t>(len));
            for (int k = 0; k < len; ++k) { s.push_back(gen_char(ch_dist(rng))); }
            payloads[t][i] = std::move(s);
        }
    }

    // Warmup with random content as well, cycling through the 200 templates.
    for (int i = 0; i < 32; ++i) {
        PerfLogInfoRandom(payloads[0][i]);
        PerfLogInfoLimitRandom(payloads[0][i]);
    }

    const double ns_per_call_info_limit = BenchmarkMultiThreadNsPerCall(
        threads, iterations_per_thread,
        [&](int tid, int i) { PerfLogInfoLimitRandom(payloads[tid][i]); });
    const double ns_per_call_info =
        BenchmarkMultiThreadNsPerCall(threads, iterations_per_thread,
                                      [&](int tid, int i) { PerfLogInfoRandom(payloads[tid][i]); });

    spdlog_logger->set_level(old_level);

    const double ratio = ns_per_call_info_limit / ns_per_call_info;
    RecordProperty("uc_info_random_avg_ns_per_call", std::to_string(ns_per_call_info));
    RecordProperty("uc_info_limit_random_avg_ns_per_call", std::to_string(ns_per_call_info_limit));
    RecordProperty("uc_info_limit_random_over_uc_info_random_ratio", std::to_string(ratio));
    std::cout << "[UCLoggerPerf] threads=" << threads
              << " iterations_per_thread=" << iterations_per_thread
              << " uc_info_random_content(ns/call)=" << ns_per_call_info
              << " uc_info_limit_random_content(ns/call)=" << ns_per_call_info_limit
              << " ratio=" << ratio << std::endl;
    ASSERT_TRUE(ratio <= 1.1);
}
