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
#ifndef UNIFIEDCACHE_INFRA_LOCK_H
#define UNIFIEDCACHE_INFRA_LOCK_H

#include <atomic>
#include <shared_mutex>

#if defined(__x86_64__)
#include <immintrin.h>
#define CPU_PAUSE() _mm_pause()
#elif defined(__arm__) || defined(__aarch64__)
#define CPU_PAUSE() __asm__ __volatile__("yield")
#else
#define CPU_PAUSE()
#endif

namespace UC {

class SpinLock {
public:
    SpinLock() = default;
    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    void Lock() noexcept
    {
        while (flag_.test_and_set(std::memory_order_acquire)) { CPU_PAUSE(); }
    }

    bool TryLock() noexcept { return !flag_.test_and_set(std::memory_order_acquire); }

    void Unlock() noexcept { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class SpinLockGuard {
public:
    explicit SpinLockGuard(SpinLock& lock) : lock_(lock) { lock_.Lock(); }
    ~SpinLockGuard() { lock_.Unlock(); }
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;

private:
    SpinLock& lock_;
};

class RwLock {
public:
    RwLock() = default;
    RwLock(const RwLock&) = delete;
    RwLock& operator=(const RwLock&) = delete;

    void LockReadOnly() noexcept { mtx_.lock_shared(); }
    void UnlockReadOnly() noexcept { mtx_.unlock_shared(); }
    bool TryLockReadOnly() noexcept { return mtx_.try_lock_shared(); }

    void LockReadWrite() noexcept { mtx_.lock(); }
    void UnlockReadWrite() noexcept { mtx_.unlock(); }
    bool TryLockReadWrite() noexcept { return mtx_.try_lock(); }

private:
    std::shared_mutex mtx_;
};

class ReadOnlyGuard {
public:
    explicit ReadOnlyGuard(RwLock& lock) : lock_(lock) { lock_.LockReadOnly(); }
    ~ReadOnlyGuard() { lock_.UnlockReadOnly(); }
    ReadOnlyGuard(const ReadOnlyGuard&) = delete;
    ReadOnlyGuard& operator=(const ReadOnlyGuard&) = delete;

private:
    RwLock& lock_;
};

class ReadWriteGuard {
public:
    explicit ReadWriteGuard(RwLock& lock) : lock_(lock) { lock_.LockReadWrite(); }
    ~ReadWriteGuard() { lock_.UnlockReadWrite(); }
    ReadWriteGuard(const ReadWriteGuard&) = delete;
    ReadWriteGuard& operator=(const ReadWriteGuard&) = delete;

private:
    RwLock& lock_;
};

}  // namespace UC

#endif  // UNIFIEDCACHE_INFRA_LOCK_H
