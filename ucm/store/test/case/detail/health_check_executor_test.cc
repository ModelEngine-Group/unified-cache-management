/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
 */
#include "health_check_executor.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>
#include <vector>

namespace UC::Test {

TEST(UCHealthCheckExecutorTest, ReturnsStoreResult)
{
    Detail::HealthCheckExecutor executor{std::chrono::milliseconds(100)};

    EXPECT_EQ(executor.Run([] { return Status::NotFound(); }), Status::NotFound());
}

TEST(UCHealthCheckExecutorTest, CapsConcurrentTimedOutProbesAt64)
{
    Detail::HealthCheckExecutor executor{std::chrono::milliseconds(100)};

    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    std::atomic<size_t> entered{0};
    std::atomic<size_t> returned{0};
    auto check = [&] {
        entered.fetch_add(1, std::memory_order_relaxed);
        cv.notify_all();
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return release; });
        return Status::OK();
    };

    constexpr size_t maxConcurrent = 64;
    std::vector<Status> results(maxConcurrent, Status::Error());
    std::vector<std::thread> callers;
    callers.reserve(maxConcurrent);
    for (size_t i = 0; i < maxConcurrent; ++i) {
        callers.emplace_back([&, i] {
            results[i] = executor.Run(check);
            returned.fetch_add(1, std::memory_order_relaxed);
            cv.notify_all();
        });
    }

    bool allEntered = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        allEntered = cv.wait_for(lock, std::chrono::seconds(1),
                                 [&] { return entered.load() == maxConcurrent; });
    }
    const auto overflowStart = std::chrono::steady_clock::now();
    const auto overflow = executor.Run(check);
    const auto overflowElapsed = std::chrono::steady_clock::now() - overflowStart;

    bool allReturned = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        allReturned = cv.wait_for(lock, std::chrono::seconds(1),
                                  [&] { return returned.load() == maxConcurrent; });
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
    }
    cv.notify_all();
    for (auto& caller : callers) { caller.join(); }

    EXPECT_TRUE(allEntered);
    EXPECT_TRUE(allReturned);
    EXPECT_EQ(entered, maxConcurrent);
    EXPECT_EQ(overflow, Status::Timeout());
    EXPECT_LT(overflowElapsed, std::chrono::milliseconds(50));
    for (const auto& result : results) { EXPECT_EQ(result, Status::Timeout()); }
}

TEST(UCHealthCheckExecutorTest, ZeroTimeoutReturnsImmediately)
{
    Detail::HealthCheckExecutor executor{std::chrono::milliseconds(0)};

    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(executor.Run([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return Status::OK();
    }),
              Status::Timeout());
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(10));
}

}  // namespace UC::Test
