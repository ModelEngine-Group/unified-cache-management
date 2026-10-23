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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_HANDLE_CACHE_H
#define UNIFIEDCACHE_POSIX_STORE_CC_HANDLE_CACHE_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include "status/status.h"
#include "type/types.h"

namespace UC::PosixStore {

#ifdef UCM_ENABLE_TEST_HOOKS
namespace TestHooks {
using OpenHook = std::function<int32_t(const std::string&, int32_t, mode_t)>;
inline std::mutex& OpenHookMutex()
{
    static std::mutex mutex;
    return mutex;
}
inline OpenHook& OpenHookSlot()
{
    static OpenHook hook;
    return hook;
}
inline void SetOpenHook(OpenHook hook)
{
    std::lock_guard<std::mutex> lock{OpenHookMutex()};
    OpenHookSlot() = std::move(hook);
}
inline void ClearOpenHook()
{
    std::lock_guard<std::mutex> lock{OpenHookMutex()};
    OpenHookSlot() = nullptr;
}
inline OpenHook GetOpenHook()
{
    std::lock_guard<std::mutex> lock{OpenHookMutex()};
    return OpenHookSlot();
}
}  // namespace TestHooks
#endif

class LoadHandleCache {
public:
    class BorrowedFd {
    public:
        BorrowedFd() = default;
        BorrowedFd(LoadHandleCache* cache, int32_t fd, int32_t error, bool cached, uint32_t slot)
            : cache_{cache}, fd_{fd}, error_{error}, cached_{cached}, slot_{slot}
        {
        }
        BorrowedFd(const BorrowedFd&) = delete;
        BorrowedFd& operator=(const BorrowedFd&) = delete;
        BorrowedFd(BorrowedFd&& other) noexcept { MoveFrom(other); }
        BorrowedFd& operator=(BorrowedFd&& other) noexcept
        {
            if (this != &other) {
                Reset();
                MoveFrom(other);
            }
            return *this;
        }
        ~BorrowedFd() { Reset(); }
        int32_t Fd() const { return fd_; }
        int32_t Error() const { return error_; }
        bool Cached() const { return cached_; }
        uint32_t Slot() const { return slot_; }
        bool Valid() const { return fd_ >= 0; }

    private:
        friend class LoadHandleCache;
        void MoveFrom(BorrowedFd& other) noexcept
        {
            cache_ = other.cache_;
            fd_ = other.fd_;
            error_ = other.error_;
            cached_ = other.cached_;
            slot_ = other.slot_;
            other.cache_ = nullptr;
            other.fd_ = -1;
            other.error_ = 0;
            other.cached_ = false;
            other.slot_ = kInvalidSlot;
        }
        void Reset()
        {
            if (cache_ && fd_ >= 0) { cache_->Release(*this); }
            cache_ = nullptr;
            fd_ = -1;
            error_ = 0;
            cached_ = false;
            slot_ = kInvalidSlot;
        }
        LoadHandleCache* cache_{nullptr};
        int32_t fd_{-1};
        int32_t error_{0};
        bool cached_{false};
        uint32_t slot_{kInvalidSlot};
    };

    static constexpr uint32_t kInvalidSlot = std::numeric_limits<uint32_t>::max();

    LoadHandleCache() = default;
    LoadHandleCache(const LoadHandleCache&) = delete;
    LoadHandleCache& operator=(const LoadHandleCache&) = delete;
    ~LoadHandleCache() { CloseAll(); }

    Status Setup(size_t capacity, bool ioDirect);
    BorrowedFd GetOrOpen(const Detail::BlockId& id, const std::string& path);
    void Release(BorrowedFd& borrowed);
    void CloseAll();

private:
    struct alignas(64) Slot {
        static constexpr uint32_t kWriterBit = 1u << 31;
        std::atomic<uint32_t> pin{0};
        int32_t fd{-1};
        Detail::BlockId id{};
    };

    bool TryPinReader(uint32_t slot, const Detail::BlockId& id);
    void UnpinReader(uint32_t slot);
    bool TryAcquireWriter(uint32_t slot);
    void ReleaseWriter(uint32_t slot);
    BorrowedFd OptimisticGet(const Detail::BlockId& id);
    BorrowedFd TryInsert(const Detail::BlockId& id, int32_t fd);
    bool TryOccupyVictim(uint32_t slot, int32_t* closedFd);
    BorrowedFd MakeBorrowed(int32_t fd, int32_t error, bool cached, uint32_t slot);
    static void CloseFd(int32_t fd);
    static void RecordHit();
    static void RecordMiss();
    static void RecordEvict();
    static void RecordBypass();

    std::vector<Slot> slots_;
    std::atomic<size_t> hand_{0};
    int32_t openFlags_{0};
    mutable std::shared_mutex indexMutex_;
    std::unordered_map<Detail::BlockId, uint32_t, Detail::BlockIdHasher> index_;
};

}  // namespace UC::PosixStore

#endif
