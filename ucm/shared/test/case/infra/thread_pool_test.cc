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

#include "thread/thread_pool.h"
#include <chrono>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include "thread/latch.h"

class UCThreadPoolTest : public ::testing::Test {};

TEST_F(UCThreadPoolTest, TimeoutDetection)
{
    struct TestTask {
        int taskId;
        std::atomic<bool>* finished;
        std::atomic<bool>* timeout;
    };

    constexpr size_t nWorker = 2;
    constexpr size_t timeoutMs = 20;
    std::atomic<int> timeoutCount{0};
    std::atomic<bool> taskFinished{false};
    std::atomic<bool> taskTimeout{false};

    UC::ThreadPool<TestTask> threadPool;
    threadPool.SetNWorker(nWorker)
        .SetWorkerFn([](TestTask& task, const auto&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            *(task.finished) = true;
        })
        .SetWorkerTimeoutFn(
            [&](TestTask& task, const auto) {
                timeoutCount++;
                task.timeout->store(true);
            },
            timeoutMs, 10)
        .Run();
    std::list<TestTask> tasks{
        {1, &taskFinished, &taskTimeout}
    };
    threadPool.Push(tasks);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_GT(timeoutCount.load(), 0);
    ASSERT_TRUE(taskTimeout.load());
}

TEST_F(UCThreadPoolTest, SimulatedFileSystemHang)
{
    struct TestTask {
        std::atomic<bool>* simulatingHang;
    };

    std::atomic<int> hangDetected{0};
    constexpr size_t hangTimeoutMs = 20;
    std::atomic<bool> taskHang{true};

    UC::ThreadPool<TestTask> threadPool;
    threadPool.SetNWorker(1)
        .SetWorkerFn([](TestTask& task, const auto&) {
            std::mutex fakeMutex;
            std::unique_lock<std::mutex> fakelock(fakeMutex);
            std::condition_variable fakeCond;
            while (*(task.simulatingHang)) {
                fakeCond.wait_for(fakelock, std::chrono::milliseconds(10));  // waiting forever
            }
        })
        .SetWorkerTimeoutFn(
            [&](TestTask& task, const auto) {
                hangDetected++;
                *(task.simulatingHang) = false;  // stop simulating hang
            },
            hangTimeoutMs, 10)
        .Run();
    std::list<TestTask> tasks{{&taskHang}};
    threadPool.Push(tasks);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_GT(hangDetected.load(), 0);
}