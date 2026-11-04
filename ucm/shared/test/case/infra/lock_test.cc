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

#include "thread/lock.h"
#include <atomic>
#include <chrono>
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

class UCRwLockTest : public ::testing::Test {};

TEST_F(UCRwLockTest, ReadOnlyLockUnlock)
{
    UC::RwLock lock;
    lock.LockReadOnly();
    EXPECT_TRUE(lock.TryLockReadOnly());
    EXPECT_FALSE(lock.TryLockReadWrite());
    lock.UnlockReadOnly();
    EXPECT_FALSE(lock.TryLockReadWrite());
    lock.UnlockReadOnly();
    EXPECT_TRUE(lock.TryLockReadWrite());
    lock.UnlockReadWrite();
}

TEST_F(UCRwLockTest, ReadWriteLockUnlock)
{
    UC::RwLock lock;
    lock.LockReadWrite();
    EXPECT_FALSE(lock.TryLockReadOnly());
    EXPECT_FALSE(lock.TryLockReadWrite());
    lock.UnlockReadWrite();
    EXPECT_TRUE(lock.TryLockReadOnly());
    lock.UnlockReadOnly();
}

TEST_F(UCRwLockTest, ReadOnlyGuardReleasesOnScopeExit)
{
    UC::RwLock lock;
    {
        UC::ReadOnlyGuard guard(lock);
        EXPECT_FALSE(lock.TryLockReadWrite());
    }
    EXPECT_TRUE(lock.TryLockReadWrite());
    lock.UnlockReadWrite();
}

TEST_F(UCRwLockTest, ReadWriteGuardReleasesOnScopeExit)
{
    UC::RwLock lock;
    {
        UC::ReadWriteGuard guard(lock);
        EXPECT_FALSE(lock.TryLockReadOnly());
        EXPECT_FALSE(lock.TryLockReadWrite());
    }
    EXPECT_TRUE(lock.TryLockReadOnly());
    lock.UnlockReadOnly();
    EXPECT_TRUE(lock.TryLockReadWrite());
    lock.UnlockReadWrite();
}

TEST_F(UCRwLockTest, WriteBlocksReadersFromOtherThread)
{
    UC::RwLock lock;
    std::atomic<bool> writerReady{false};
    std::atomic<bool> stopHold{false};
    std::thread writer([&] {
        lock.LockReadWrite();
        writerReady.store(true, std::memory_order_release);
        while (!stopHold.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        lock.UnlockReadWrite();
    });
    while (!writerReady.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    EXPECT_FALSE(lock.TryLockReadOnly());
    EXPECT_FALSE(lock.TryLockReadWrite());
    stopHold.store(true, std::memory_order_release);
    writer.join();
    EXPECT_TRUE(lock.TryLockReadOnly());
    lock.UnlockReadOnly();
}

TEST_F(UCRwLockTest, ReadBlocksWriterFromOtherThread)
{
    UC::RwLock lock;
    std::atomic<bool> readerReady{false};
    std::atomic<bool> stopHold{false};
    std::thread reader([&] {
        lock.LockReadOnly();
        readerReady.store(true, std::memory_order_release);
        while (!stopHold.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        lock.UnlockReadOnly();
    });
    while (!readerReady.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    EXPECT_FALSE(lock.TryLockReadWrite());
    EXPECT_TRUE(lock.TryLockReadOnly());
    lock.UnlockReadOnly();
    stopHold.store(true, std::memory_order_release);
    reader.join();
    EXPECT_TRUE(lock.TryLockReadWrite());
    lock.UnlockReadWrite();
}

TEST_F(UCRwLockTest, MultipleReadersShareLock)
{
    constexpr int nReader = 8;
    UC::RwLock lock;
    std::atomic<int> activeReaders{0};
    std::atomic<int> maxReaders{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    threads.reserve(nReader);
    for (int t = 0; t < nReader; ++t) {
        threads.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                UC::ReadOnlyGuard guard(lock);
                int cur = activeReaders.fetch_add(1, std::memory_order_acq_rel) + 1;
                int prev = maxReaders.load(std::memory_order_acquire);
                while (cur > prev &&
                       !maxReaders.compare_exchange_weak(prev, cur, std::memory_order_acq_rel)) {}
                std::this_thread::yield();
                activeReaders.fetch_sub(1, std::memory_order_acq_rel);
            }
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true, std::memory_order_release);
    for (auto& th : threads) { th.join(); }
    EXPECT_GT(maxReaders.load(), 1);
}

TEST_F(UCRwLockTest, WritersAreMutuallyExclusive)
{
    constexpr int nWriter = 8;
    constexpr int iterPerThread = 500;
    UC::RwLock lock;
    std::atomic<int> value{0};
    std::atomic<int> maxConcurrentWriters{0};
    std::atomic<int> activeWriters{0};
    std::vector<std::thread> threads;
    threads.reserve(nWriter);
    for (int t = 0; t < nWriter; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < iterPerThread; ++i) {
                UC::ReadWriteGuard guard(lock);
                int cur = activeWriters.fetch_add(1, std::memory_order_acq_rel) + 1;
                int prev = maxConcurrentWriters.load(std::memory_order_acquire);
                while (cur > prev && !maxConcurrentWriters.compare_exchange_weak(
                                         prev, cur, std::memory_order_acq_rel)) {}
                value.fetch_add(1, std::memory_order_relaxed);
                activeWriters.fetch_sub(1, std::memory_order_acq_rel);
            }
        });
    }
    for (auto& th : threads) { th.join(); }
    EXPECT_EQ(value.load(), nWriter * iterPerThread);
    EXPECT_EQ(maxConcurrentWriters.load(), 1);
}
