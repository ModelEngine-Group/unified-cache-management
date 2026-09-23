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
#include "handle_cache.h"
#include <cerrno>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include "logger/logger.h"
#include "metrics_api.h"

namespace UC::PosixStore {

namespace {
constexpr auto kNewFilePerm = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
}  // namespace

Status LoadHandleCache::Setup(size_t capacity, bool ioDirect)
{
    openFlags_ = O_RDONLY;
    if (ioDirect) { openFlags_ |= O_DIRECT; }
    if (capacity == 0) {
        slots_.clear();
        return Status::OK();
    }
    struct rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
        auto limit = static_cast<size_t>(rl.rlim_cur);
        if (capacity > limit) {
            UC_INFO("Clamp load handle cache size from {} to RLIMIT_NOFILE({}).", capacity, limit);
            capacity = limit;
        }
    }
    try {
        slots_ = std::vector<Slot>(capacity);
    } catch (const std::exception& e) {
        return Status::OutOfMemory();
    }
    hand_.store(0, std::memory_order_relaxed);
    return Status::OK();
}

LoadHandleCache::BorrowedFd LoadHandleCache::GetOrOpen(const Detail::BlockId& id,
                                                       const std::string& path)
{
    auto hit = OptimisticGet(id);
    if (hit.Valid()) {
        RecordHit();
        return hit;
    }
    RecordMiss();
    int32_t fd = -1;
#ifdef UCM_ENABLE_TEST_HOOKS
    auto hook = TestHooks::GetOpenHook();
    fd = hook ? hook(path, openFlags_, kNewFilePerm)
              : ::open(path.c_str(), openFlags_, kNewFilePerm);
#else
    fd = ::open(path.c_str(), openFlags_, kNewFilePerm);
#endif
    if (fd < 0) { return MakeBorrowed(-1, errno, false, kInvalidSlot); }
    if (slots_.empty()) { return MakeBorrowed(fd, 0, false, kInvalidSlot); }
    auto inserted = TryInsert(id, fd);
    if (!inserted.Cached()) { RecordBypass(); }
    return inserted;
}

void LoadHandleCache::Release(BorrowedFd& borrowed)
{
    const auto fd = borrowed.fd_;
    const auto cached = borrowed.cached_;
    const auto slot = borrowed.slot_;
    borrowed.cache_ = nullptr;
    borrowed.fd_ = -1;
    borrowed.cached_ = false;
    borrowed.slot_ = kInvalidSlot;
    if (cached && slot < slots_.size()) {
        UnpinReader(slot);
        return;
    }
    if (fd >= 0) { CloseFd(fd); }
}

void LoadHandleCache::CloseAll()
{
    {
        std::unique_lock<std::shared_mutex> lock(indexMutex_);
        index_.clear();
    }
    for (uint32_t i = 0; i < slots_.size(); ++i) {
        auto& slot = slots_[i];
        if (!TryAcquireWriter(i)) { continue; }
        if (slot.fd >= 0) {
            CloseFd(slot.fd);
            slot.fd = -1;
            slot.id = {};
        }
        ReleaseWriter(i);
    }
}

bool LoadHandleCache::TryPinReader(uint32_t slotIdx, const Detail::BlockId& id)
{
    auto& slot = slots_[slotIdx];
    for (;;) {
        auto cur = slot.pin.load(std::memory_order_acquire);
        if (cur & Slot::kWriterBit) { return false; }
        if (!slot.pin.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
            continue;
        }
        break;
    }
    if (slot.fd < 0 || slot.id != id) {
        UnpinReader(slotIdx);
        return false;
    }
    return true;
}

void LoadHandleCache::UnpinReader(uint32_t slotIdx)
{
    slots_[slotIdx].pin.fetch_sub(1, std::memory_order_acq_rel);
}

bool LoadHandleCache::TryAcquireWriter(uint32_t slotIdx)
{
    uint32_t expected = 0;
    return slots_[slotIdx].pin.compare_exchange_strong(
        expected, Slot::kWriterBit, std::memory_order_acq_rel, std::memory_order_acquire);
}

void LoadHandleCache::ReleaseWriter(uint32_t slotIdx)
{
    slots_[slotIdx].pin.store(0, std::memory_order_release);
}

LoadHandleCache::BorrowedFd LoadHandleCache::OptimisticGet(const Detail::BlockId& id)
{
    uint32_t slot = kInvalidSlot;
    {
        std::shared_lock<std::shared_mutex> lock(indexMutex_);
        auto it = index_.find(id);
        if (it == index_.end()) { return {}; }
        slot = it->second;
    }
    if (slot >= slots_.size()) { return {}; }
    if (!TryPinReader(slot, id)) { return {}; }
    return MakeBorrowed(slots_[slot].fd, 0, true, slot);
}

LoadHandleCache::BorrowedFd LoadHandleCache::TryInsert(const Detail::BlockId& id, int32_t fd)
{
    {
        std::shared_lock<std::shared_mutex> lock(indexMutex_);
        auto it = index_.find(id);
        if (it != index_.end()) {
            lock.unlock();
            auto hit = OptimisticGet(id);
            if (hit.Valid()) {
                CloseFd(fd);
                return hit;
            }
            return MakeBorrowed(fd, 0, false, kInvalidSlot);
        }
    }

    const auto n = slots_.size();
    if (n == 0) { return MakeBorrowed(fd, 0, false, kInvalidSlot); }
    for (size_t i = 0; i < n; ++i) {
        auto cur = static_cast<uint32_t>(hand_.fetch_add(1, std::memory_order_relaxed) % n);
        auto& slot = slots_[cur];
        auto pin = slot.pin.load(std::memory_order_acquire);
        if (pin != 0) { continue; }
        int32_t closedFd = -1;
        if (!TryOccupyVictim(cur, &closedFd)) { continue; }
        CloseFd(closedFd);
        slot.id = id;
        slot.fd = fd;
        {
            std::unique_lock<std::shared_mutex> lock(indexMutex_);
            auto it = index_.find(id);
            if (it != index_.end()) {
                slot.fd = -1;
                slot.id = {};
                ReleaseWriter(cur);
                lock.unlock();
                auto hit = OptimisticGet(id);
                if (hit.Valid()) {
                    CloseFd(fd);
                    return hit;
                }
                return MakeBorrowed(fd, 0, false, kInvalidSlot);
            }
            index_[id] = cur;
        }
        slot.pin.store(1, std::memory_order_release);
        return MakeBorrowed(fd, 0, true, cur);
    }
    return MakeBorrowed(fd, 0, false, kInvalidSlot);
}

bool LoadHandleCache::TryOccupyVictim(uint32_t slotIdx, int32_t* closedFd)
{
    if (!TryAcquireWriter(slotIdx)) { return false; }
    auto& slot = slots_[slotIdx];
    if (slot.fd >= 0) {
        {
            std::unique_lock<std::shared_mutex> lock(indexMutex_);
            auto it = index_.find(slot.id);
            if (it != index_.end() && it->second == slotIdx) { index_.erase(it); }
        }
        *closedFd = slot.fd;
        slot.fd = -1;
        slot.id = {};
        RecordEvict();
    } else {
        *closedFd = -1;
    }
    return true;
}

LoadHandleCache::BorrowedFd LoadHandleCache::MakeBorrowed(int32_t fd, int32_t error, bool cached,
                                                          uint32_t slot)
{
    return BorrowedFd{this, fd, error, cached, slot};
}

void LoadHandleCache::CloseFd(int32_t fd)
{
    if (fd >= 0) { ::close(fd); }
}

void LoadHandleCache::RecordHit()
{
    static UC::Metrics::CachedMetric metric{"posix_handle_cache_hit_total"};
    UC::Metrics::UpdateStats(metric, 1.0);
}

void LoadHandleCache::RecordMiss()
{
    static UC::Metrics::CachedMetric metric{"posix_handle_cache_miss_total"};
    UC::Metrics::UpdateStats(metric, 1.0);
}

void LoadHandleCache::RecordEvict()
{
    static UC::Metrics::CachedMetric metric{"posix_handle_cache_evict_total"};
    UC::Metrics::UpdateStats(metric, 1.0);
}

void LoadHandleCache::RecordBypass()
{
    static UC::Metrics::CachedMetric metric{"posix_handle_cache_bypass_total"};
    UC::Metrics::UpdateStats(metric, 1.0);
}

}  // namespace UC::PosixStore
