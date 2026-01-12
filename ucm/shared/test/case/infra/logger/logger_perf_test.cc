
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
#include <iostream>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <vector>
#include "logger/spdlog/cc/compress_rotate_file_sink.h"

namespace {

using Clock = std::chrono::steady_clock;

struct PerfResult {
    double total_ms{};
    double per_log_ns{};
};

PerfResult BenchLogger(const std::shared_ptr<spdlog::logger>& logger, std::size_t total_logs,
                       std::size_t thread_count)
{
    std::atomic<std::size_t> counter{0};

    auto worker = [&]() {
        for (;;) {
            auto idx = counter.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_logs) { break; }
            logger->info("perf test log message {}", idx);
        }
    };

    const auto start = Clock::now();

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) { threads.emplace_back(worker); }
    for (auto& t : threads) { t.join(); }

    logger->flush();

    const auto end = Clock::now();
    const auto duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    const double per_log_ns = (duration_ms * 1e6) / static_cast<double>(total_logs);

    return PerfResult{duration_ms, per_log_ns};
}

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

void PerfTest(std::shared_ptr<spdlog::logger> logger, int thread_count)
{
    constexpr std::size_t kTotalLogs = 200'000;
    logger->set_level(spdlog::level::info);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f][%n][%^%L%$] %v [%P,%t][%s:%#,%!]");
    (void)BenchLogger(logger, 10'000, thread_count);
    auto rotating_res = BenchLogger(logger, kTotalLogs, thread_count);
    std::cout << "[LoggerPerfTest] total logs: " << kTotalLogs << ", threads: " << thread_count
              << std::endl;
    std::cout << "  rotating_logger_mt:          " << rotating_res.total_ms << " ms total, "
              << rotating_res.per_log_ns << " ns/log" << std::endl;
}
}  // namespace

class UCLoggerPerfTest : public ::testing::TestWithParam<int> {
protected:
    const std::string LOG_PERF_TEST_DIR = "log_perf_test";
    const std::string log_path_ = this->LOG_PERF_TEST_DIR;
    std::shared_ptr<spdlog::logger> rotating_logger_;
    void SetUp() override
    {
        CleanDir(this->LOG_PERF_TEST_DIR);
        std::filesystem::create_directories(this->log_path_);
        constexpr std::size_t kMaxFiles = 3;
        constexpr std::size_t kMaxFileSize = 5 * 1024 * 1024;  // 5 MB
        this->rotating_logger_ = spdlog::compress_rotating_logger_mt(
            "compress_rotating_logger_mt", this->log_path_ + "/perf.log", kMaxFileSize, kMaxFiles,
            false);
    }
    void TearDown() override
    {
        CleanDir("log_perf_test");
        spdlog::drop_all();
        this->rotating_logger_ = nullptr;
    }
};

TEST_P(UCLoggerPerfTest, CompressRotatingLoggerMt)
{
    PerfTest(rotating_logger_, GetParam());
    SUCCEED();
}

INSTANTIATE_TEST_CASE_P(, UCLoggerPerfTest, ::testing::Values(1, 10, 200));