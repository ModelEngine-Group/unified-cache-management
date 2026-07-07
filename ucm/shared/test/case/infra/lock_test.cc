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

#include "thread/lock.h"
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

class UCSpinLockTest : public ::testing::Test {};

TEST_F(UCSpinLockTest, TryLockAndUnlock)
{
    UC::SpinLock lock;
    lock.Lock();
    EXPECT_FALSE(lock.TryLock());
    lock.Unlock();
    EXPECT_TRUE(lock.TryLock());
    lock.Unlock();
}

TEST_F(UCSpinLockTest, TryLockFailsWhenHeldByOtherThread)
{
    UC::SpinLock lock;
    std::atomic<bool> holderReady{false};
    std::atomic<bool> stopHold{false};
    std::thread holder([&] {
        lock.Lock();
        holderReady.store(true, std::memory_order_release);
        while (!stopHold.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        lock.Unlock();
    });
    while (!holderReady.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    EXPECT_FALSE(lock.TryLock());
    stopHold.store(true, std::memory_order_release);
    holder.join();
    EXPECT_TRUE(lock.TryLock());
    lock.Unlock();
}

TEST_F(UCSpinLockTest, GuardReleasesOnScopeExit)
{
    UC::SpinLock lock;
    {
        UC::SpinLockGuard guard(lock);
        EXPECT_FALSE(lock.TryLock());
    }
    EXPECT_TRUE(lock.TryLock());
    lock.Unlock();
}

TEST_F(UCSpinLockTest, MutualExclusionUnderContention)
{
    constexpr int nThread = 8;
    constexpr int iterPerThread = 1000;
    UC::SpinLock lock;
    std::atomic<int> value{0};
    std::vector<std::thread> threads;
    threads.reserve(nThread);
    for (int t = 0; t < nThread; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < iterPerThread; ++i) {
                UC::SpinLockGuard guard(lock);
                value.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) { th.join(); }
    EXPECT_EQ(value.load(), nThread * iterPerThread);
}
