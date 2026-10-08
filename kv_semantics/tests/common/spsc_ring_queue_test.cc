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
 * */
#include "spsc_ring_queue.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

namespace kv {
namespace {

TEST(SpscRingQueueTest, ConsumerLoopWakesForTaskAndStop)
{
    SpscRingQueue<size_t> queue;
    queue.Setup(16);
    std::atomic_bool stop{false};
    std::mutex waitMutex;
    std::condition_variable waitCondition;
    std::promise<size_t> processedPromise;
    auto processed = processedPromise.get_future();

    std::thread consumer([&] {
        queue.ConsumerLoop(stop, waitMutex, waitCondition, [&processedPromise](size_t value) {
            processedPromise.set_value(value);
        });
    });
    {
        std::lock_guard<std::mutex> lock(waitMutex);
        EXPECT_TRUE(queue.TryPush(42));
    }
    waitCondition.notify_one();

    const auto waitStatus = processed.wait_for(std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lock(waitMutex);
        stop.store(true, std::memory_order_release);
    }
    waitCondition.notify_one();
    consumer.join();

    ASSERT_EQ(waitStatus, std::future_status::ready);
    EXPECT_EQ(processed.get(), 42);
}

}  // namespace
}  // namespace kv
