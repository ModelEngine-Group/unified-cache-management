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
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <thread>
#include "mutex/shared_mutex.h"
#include "status/status.h"

namespace UC::Cache2 {

inline constexpr size_t kInvalid{std::numeric_limits<size_t>::max()};
inline constexpr size_t kSlotClaimed{kInvalid};
inline constexpr size_t kMaxRanks{128};
inline constexpr size_t kMaxLockStripes{1ULL << 16};
inline constexpr size_t kMinBuckets{1ULL << 10};
inline constexpr size_t kMaxBuckets{1ULL << 24};
inline constexpr uint32_t kCtrlMagic{0x55433201U};  // "UC2" + layout version 1.

class CtrlLayout {
public:
    using BucketLock = SharedMutex;
    struct SlotMeta {
        enum class State : uint8_t { Loading, Ready, Failed };
        /* Hot line: CAS contention point of the pin protocol. */
        alignas(64) std::atomic<size_t> reference{0};
        /* Cache key snapshot: key[0]/key[1] = BlockId (16B), key[2] = offset. */
        alignas(8) std::atomic<size_t> key[3]{0, 0, 0};
        /* Owning bucket index; kInvalid while unlinked / being reconfigured. */
        std::atomic<size_t> hash{kInvalid};
        std::atomic<size_t> prev{kInvalid};
        std::atomic<size_t> next{kInvalid};
        alignas(64) std::atomic<State> state{State::Loading};
        /* Hot line: CLOCK second-chance bit. */
        alignas(64) std::atomic<uint8_t> accessed{0};

        void Init()
        {
            reference.store(0, std::memory_order_relaxed);
            key[0].store(0, std::memory_order_relaxed);
            key[1].store(0, std::memory_order_relaxed);
            key[2].store(0, std::memory_order_relaxed);
            hash.store(kInvalid, std::memory_order_relaxed);
            prev.store(kInvalid, std::memory_order_relaxed);
            next.store(kInvalid, std::memory_order_relaxed);
            state.store(State::Loading, std::memory_order_relaxed);
            accessed.store(0, std::memory_order_relaxed);
        }
    };
    struct RankDataDesc {
        std::atomic<size_t> handle{kInvalid};

        RankDataDesc() = default;
        RankDataDesc(const RankDataDesc& o) : handle(o.handle.load(std::memory_order_relaxed)) {}
        RankDataDesc& operator=(const RankDataDesc& o)
        {
            handle.store(o.handle.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

private:
    friend class Buffer;
    friend class CtrlStrategy;
    friend struct BufferTestAccess;
    friend struct CtrlLayoutTestAccess;
    friend struct CtrlStrategyTestAccess;

    struct Header {
        std::atomic<uint32_t> magic{0};
        size_t rankCount{0};
        size_t slotsPerRank{0};
        size_t slotSize{0};
        size_t bucketCount{0};
        size_t lockStripeCount{0};
        alignas(64) RankDataDesc rankDescs[kMaxRanks];
        alignas(64) std::atomic<size_t> clockHands[kMaxRanks];
    };

    void* base_{nullptr};
    size_t rankCount_{0};
    size_t slotsPerRank_{0};
    size_t bucketCount_{0};
    size_t lockStripeCount_{0};
    size_t slotCount_{0};

    static size_t AlignUp(size_t value, size_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static size_t LockStripeCount(size_t bucketCount)
    {
        return std::min(bucketCount, kMaxLockStripes);
    }

    static size_t RecommendBucketCount(size_t slotCount)
    {
        auto target = std::max(kMinBuckets, slotCount / 2 + slotCount % 2);
        target = std::min(target, kMaxBuckets);
        size_t result = 1;
        while (result < target) { result <<= 1; }
        return result;
    }

    static size_t BucketsOffset() { return AlignUp(sizeof(Header), 64); }

    static size_t LocksOffset(size_t bucketCount)
    {
        return AlignUp(BucketsOffset() + sizeof(std::atomic<size_t>) * bucketCount,
                       alignof(BucketLock));
    }

    static size_t SlotMetaOffset(size_t bucketCount, size_t lockStripeCount)
    {
        return AlignUp(LocksOffset(bucketCount) + sizeof(BucketLock) * lockStripeCount,
                       alignof(SlotMeta));
    }

    static size_t TotalSize(size_t bucketCount, size_t lockStripeCount, size_t slotCount)
    {
        return SlotMetaOffset(bucketCount, lockStripeCount) + sizeof(SlotMeta) * slotCount;
    }

    void Bind(void* base, size_t rankCount, size_t slotsPerRank, size_t bucketCount,
              size_t lockStripeCount)
    {
        base_ = base;
        rankCount_ = rankCount;
        slotsPerRank_ = slotsPerRank;
        bucketCount_ = bucketCount;
        lockStripeCount_ = lockStripeCount;
        slotCount_ = rankCount * slotsPerRank;
    }

    Header* Hdr() const { return static_cast<Header*>(base_); }

public:
    void InitHeader(size_t slotSize)
    {
        auto* header = ::new (base_) Header();
        header->magic.store(0, std::memory_order_relaxed);
        header->rankCount = rankCount_;
        header->slotsPerRank = slotsPerRank_;
        header->slotSize = slotSize;
        header->bucketCount = bucketCount_;
        header->lockStripeCount = lockStripeCount_;
        for (size_t rank = 0; rank < rankCount_; ++rank) {
            header->rankDescs[rank].handle.store(kInvalid, std::memory_order_relaxed);
            header->clockHands[rank].store(0, std::memory_order_relaxed);
        }
        auto* buckets = Buckets();
        for (size_t i = 0; i < bucketCount_; ++i) {
            ::new (&buckets[i]) std::atomic<size_t>(kInvalid);
        }
        auto* locks = LockArr();
        for (size_t i = 0; i < lockStripeCount_; ++i) {
            ::new (&locks[i]) BucketLock();
            locks[i].Init();
        }
    }

    void InitSlotRange(size_t rank)
    {
        if (rank >= rankCount_) { return; }
        auto begin = rank * slotsPerRank_;
        auto end = begin + slotsPerRank_;
        auto* slots = SlotMetaArr();
        for (size_t i = begin; i < end; ++i) {
            ::new (&slots[i]) SlotMeta();
            slots[i].Init();
        }
    }

    void MarkReady() { Hdr()->magic.store(kCtrlMagic, std::memory_order_release); }

    bool WaitReady(size_t timeoutMs) const
    {
        if (base_ == nullptr) { return false; }
        constexpr auto interval = std::chrono::milliseconds(10);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (Hdr()->magic.load(std::memory_order_acquire) != kCtrlMagic) {
            if (timeoutMs > 0 && std::chrono::steady_clock::now() >= deadline) { return false; }
            std::this_thread::sleep_for(interval);
        }
        return true;
    }

    Status SetRankDesc(size_t rank, const RankDataDesc& desc)
    {
        if (rank >= rankCount_) { return Status::InvalidParam("rank out of range"); }
        auto handle = desc.handle.load(std::memory_order_relaxed);
        Hdr()->rankDescs[rank].handle.store(handle, std::memory_order_release);
        return Status::OK();
    }

    Expected<RankDataDesc> GetRankDesc(size_t rank) const
    {
        if (rank >= rankCount_) { return Status::InvalidParam("rank out of range"); }
        auto handle = Hdr()->rankDescs[rank].handle.load(std::memory_order_acquire);
        if (handle == kInvalid) { return Status::NotFound(); }
        RankDataDesc result;
        result.handle.store(handle, std::memory_order_relaxed);
        return result;
    }

    std::atomic<size_t>* Buckets() const
    {
        return reinterpret_cast<std::atomic<size_t>*>(static_cast<std::byte*>(base_) +
                                                      BucketsOffset());
    }

    BucketLock* LockOf(size_t iBucket) const
    {
        if (iBucket >= bucketCount_ || lockStripeCount_ == 0) { return nullptr; }
        return &LockArr()[iBucket & (lockStripeCount_ - 1)];
    }

    SlotMeta* SlotMetaArr() const
    {
        return reinterpret_cast<SlotMeta*>(static_cast<std::byte*>(base_) +
                                           SlotMetaOffset(bucketCount_, lockStripeCount_));
    }

    size_t BucketCount() const { return bucketCount_; }
    size_t SlotCount() const { return slotCount_; }

private:
    std::atomic<size_t>* ClockHand(size_t rank) const
    {
        return rank < rankCount_ ? &Hdr()->clockHands[rank] : nullptr;
    }

    size_t NextClockSlot(size_t rank, size_t usableSlots) const
    {
        if (rank >= rankCount_ || usableSlots == 0 || usableSlots > slotsPerRank_) {
            return kInvalid;
        }
        auto local = ClockHand(rank)->fetch_add(1, std::memory_order_relaxed) % usableSlots;
        return rank * slotsPerRank_ + local;
    }

    size_t RankCount() const { return rankCount_; }
    size_t SlotsPerRank() const { return slotsPerRank_; }
    size_t SlotSize() const { return Hdr()->slotSize; }
    size_t LockCount() const { return lockStripeCount_; }

    BucketLock* LockArr() const
    {
        return reinterpret_cast<BucketLock*>(static_cast<std::byte*>(base_) +
                                             LocksOffset(bucketCount_));
    }
};

static_assert(std::atomic<uint32_t>::is_always_lock_free, "control magic must be lock-free");
static_assert(std::atomic<uint8_t>::is_always_lock_free, "slot flags must be lock-free");
static_assert(std::atomic<size_t>::is_always_lock_free, "slot indices must be lock-free");

}  // namespace UC::Cache2
