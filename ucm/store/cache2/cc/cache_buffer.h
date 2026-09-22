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

#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include "ctrl_layout.h"
#include "ctrl_strategy.h"
#include "data_strategy.h"
#include "global_config.h"
#include "status/status.h"
#include "type/types.h"

namespace UC::Cache2 {

class Buffer {
    friend struct BufferTestAccess;

    using SlotMeta = CtrlLayout::SlotMeta;
    using State = SlotMeta::State;

    CtrlStrategy ctrl_;
    DataStrategy data_;
    size_t myRank_{kInvalid};
    size_t rankCount_{0};
    size_t slotsPerRank_{0};
    size_t slotSize_{0};
    size_t bucketCount_{0};
    size_t reservedSlots_{0};

    static constexpr size_t kPinSpinFast{64};

public:
    class Handle {
        friend class Buffer;
        Buffer* buf_{nullptr};
        size_t slotIdx_{kInvalid};
        bool owner_{false};

        Handle(Buffer* buf, size_t slotIdx, bool owner)
            : buf_(buf), slotIdx_(slotIdx), owner_(owner)
        {
        }

    public:
        Handle() = default;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& o) noexcept : buf_(o.buf_), slotIdx_(o.slotIdx_), owner_(o.owner_)
        {
            o.buf_ = nullptr;
            o.slotIdx_ = kInvalid;
            o.owner_ = false;
        }
        Handle& operator=(Handle&& o) noexcept
        {
            Handle tmp(std::move(o));
            Swap(tmp);
            return *this;
        }
        ~Handle()
        {
            if (Valid()) {
                if (owner_ && GetState() == State::Loading) { buf_->MarkFailed(slotIdx_); }
                buf_->Release(slotIdx_);
            }
        }
        explicit operator bool() const { return Valid(); }
        bool Owner() const { return owner_; }
        bool HostAccessible() const
        {
            return Valid() && buf_->data_.HostAccessibleOf(slotIdx_);
        }
        size_t SlotIndex() const { return Valid() ? slotIdx_ : kInvalid; }
        void* Data() { return Valid() ? buf_->data_.DataAt(slotIdx_) : nullptr; }
        void* DeviceData() { return Valid() ? buf_->data_.DeviceDataAt(slotIdx_) : nullptr; }
        CtrlLayout::SlotMeta::State GetState() const
        {
            return Valid() ? buf_->GetState(slotIdx_) : State::Failed;
        }
        bool Ready() const { return GetState() == State::Ready; }
        void MarkReady()
        {
            if (Valid() && Owner()) { buf_->MarkReady(slotIdx_); }
        }
        void MarkFailed()
        {
            if (Valid() && Owner()) { buf_->MarkFailed(slotIdx_); }
        }

    private:
        bool Valid() const { return buf_ != nullptr && slotIdx_ != kInvalid; }
        void Swap(Handle& o) noexcept
        {
            std::swap(buf_, o.buf_);
            std::swap(slotIdx_, o.slotIdx_);
            std::swap(owner_, o.owner_);
        }
    };

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Status Setup(const Config& cfg)
    {
        if (cfg.localRankSize == 0 || cfg.localRankSize > kMaxRanks) {
            return Status::InvalidParam("invalid cache2 local rank size({})", cfg.localRankSize);
        }
        if (cfg.deviceId < -1 ||
            (cfg.deviceId >= 0 && static_cast<size_t>(cfg.deviceId) >= cfg.localRankSize)) {
            return Status::InvalidParam("cache2 device/rank({}) must be in [0, {})", cfg.deviceId,
                                        cfg.localRankSize);
        }
        if (cfg.loadExclusiveBufferNumber % cfg.localRankSize != 0) {
            return Status::InvalidParam(
                "cache2 load exclusive buffer number({}) must be divisible by ranks({})",
                cfg.loadExclusiveBufferNumber, cfg.localRankSize);
        }
        auto s = ctrl_.Setup(cfg);
        if (s.Failure()) { return s; }
        auto& layout = ctrl_.Layout();
        rankCount_ = layout.RankCount();
        slotsPerRank_ = layout.SlotsPerRank();
        slotSize_ = layout.SlotSize();
        bucketCount_ = layout.BucketCount();
        reservedSlots_ = cfg.loadExclusiveBufferNumber / rankCount_;

        if (cfg.deviceId < 0) { return Status::OK(); }
        if (reservedSlots_ >= slotsPerRank_) {
            return Status::InvalidParam(
                "cache2 reserved slots per rank({}) must be less than slots per rank({})",
                reservedSlots_, slotsPerRank_);
        }
        /* Cache2 currently identifies the worker-local partition by deviceId. */
        myRank_ = static_cast<size_t>(cfg.deviceId);
        layout.InitSlotRange(myRank_);
        return data_.Setup(layout, cfg.deviceId, slotSize_, slotsPerRank_);
    }

    Handle Get(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        if (myRank_ == kInvalid) { return Handle{}; }
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        if (usable == 0) { return Handle{}; }
        auto attempts = usable > std::numeric_limits<size_t>::max() / 2
                            ? std::numeric_limits<size_t>::max()
                            : 2 * usable;
        for (;;) {
            auto handle = TryGet(blockId, offset, allowReserved, attempts);
            if (handle) { return handle; }
            std::this_thread::yield();
        }
    }

    void Prealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        if (myRank_ == kInvalid) { return; }
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        if (usable == 0) { return; }
        auto attempts = usable > std::numeric_limits<size_t>::max() / 2
                            ? std::numeric_limits<size_t>::max()
                            : 2 * usable;
        TryPrealloc(blockId, offset, allowReserved, attempts);
    }

    bool Exist(const Detail::BlockId& blockId, size_t offset)
    {
        if (bucketCount_ == 0) { return false; }
        auto iBucket = HashKey(blockId);
        auto& layout = ctrl_.Layout();
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner, false)) {
                Release(iNode);
                return true;
            }
        }
        auto* lock = layout.LockOf(iBucket);
        lock->Lock();
        iNode = Lookup(layout, iBucket, blockId, offset);
        auto found = iNode != kInvalid;
        if (found) { layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed); }
        lock->Unlock();
        return found;
    }

    void Touch(const Detail::BlockId* blocks, size_t num)
    {
        if (blocks == nullptr || bucketCount_ == 0) { return; }
        auto& layout = ctrl_.Layout();
        for (size_t i = 0; i < num; ++i) {
            auto iBucket = HashKey(blocks[i]);
            auto iNode = layout.Buckets()[iBucket].load(std::memory_order_acquire);
            size_t walked = 0;
            while (iNode != kInvalid && iNode < layout.SlotCount() &&
                   walked++ < layout.SlotCount()) {
                auto* meta = &layout.SlotMetaArr()[iNode];
                if (MatchBlock(*meta, iBucket, blocks[i])) {
                    meta->accessed.store(1, std::memory_order_relaxed);
                }
                iNode = meta->next.load(std::memory_order_acquire);
            }
        }
    }

    size_t MyRank() const { return myRank_; }
    size_t RankCount() const { return rankCount_; }
    size_t SlotsPerRank() const { return slotsPerRank_; }
    size_t BucketCount() const { return bucketCount_; }

private:
    static void BlockKeyWords(const Detail::BlockId& blockId, size_t (&words)[2])
    {
        static_assert(sizeof(Detail::BlockId) == sizeof(words));
        std::memcpy(words, blockId.data(), sizeof(blockId));
    }

    size_t HashKey(const Detail::BlockId& blockId) const
    {
        static Detail::BlockIdHasher hasher;
        return hasher(blockId) & (bucketCount_ - 1);
    }

    static void StoreKey(SlotMeta& meta, const Detail::BlockId& blockId, size_t offset)
    {
        size_t words[2];
        BlockKeyWords(blockId, words);
        meta.key[0].store(words[0], std::memory_order_relaxed);
        meta.key[1].store(words[1], std::memory_order_relaxed);
        meta.key[2].store(offset, std::memory_order_relaxed);
    }

    static bool MatchBlock(const SlotMeta& meta, size_t iBucket, const Detail::BlockId& blockId)
    {
        if (meta.hash.load(std::memory_order_acquire) != iBucket) { return false; }
        size_t words[2];
        BlockKeyWords(blockId, words);
        return meta.key[0].load(std::memory_order_acquire) == words[0] &&
               meta.key[1].load(std::memory_order_acquire) == words[1];
    }

    static bool MatchKey(const SlotMeta& meta, size_t iBucket, const Detail::BlockId& blockId,
                         size_t offset)
    {
        return MatchBlock(meta, iBucket, blockId) &&
               meta.key[2].load(std::memory_order_acquire) == offset;
    }

    Handle TryGet(const Detail::BlockId& blockId, size_t offset, bool allowReserved,
                  size_t attempts)
    {
        auto iBucket = HashKey(blockId);
        auto& layout = ctrl_.Layout();
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner)) {
                return Handle{this, iNode, owner};
            }
        }

        auto* targetLock = layout.LockOf(iBucket);
        if (!targetLock->TryLock()) { return Handle{}; }
        iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            bool owner = false;
            auto pinned = PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner);
            targetLock->Unlock();
            return pinned ? Handle{this, iNode, owner} : Handle{};
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, 1);
        targetLock->Unlock();
        return iNode == kInvalid ? Handle{} : Handle{this, iNode, true};
    }

    bool TryPrealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved,
                     size_t attempts)
    {
        auto iBucket = HashKey(blockId);
        auto& layout = ctrl_.Layout();
        auto* targetLock = layout.LockOf(iBucket);
        if (!targetLock->TryLock()) { return false; }
        auto iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed);
            targetLock->Unlock();
            return true;
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, 0);
        targetLock->Unlock();
        return iNode != kInvalid;
    }

    bool PinHit(CtrlLayout& layout, size_t iNode, size_t iBucket, const Detail::BlockId& blockId,
                size_t offset, size_t spinBudget, bool& owner, bool takeOwnership = true)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        for (size_t spin = 0; spin < spinBudget;) {
            if (!MatchKey(*meta, iBucket, blockId, offset)) { return false; }
            auto reference = meta->reference.load(std::memory_order_acquire);
            if (reference == kSlotClaimed) {
                ++spin;
                std::this_thread::yield();
                continue;
            }
            if (reference == kSlotClaimed - 1) { return false; }
            auto state = meta->state.load(std::memory_order_acquire);
            if (!takeOwnership && reference == 0 && state != State::Ready) { return false; }
            if (!meta->reference.compare_exchange_weak(reference, reference + 1,
                                                       std::memory_order_acq_rel)) {
                ++spin;
                continue;
            }
            if (!MatchKey(*meta, iBucket, blockId, offset)) {
                meta->reference.fetch_sub(1, std::memory_order_release);
                return false;
            }
            state = meta->state.load(std::memory_order_acquire);
            owner = takeOwnership && reference == 0 && state != State::Ready;
            if (owner && state == State::Failed) {
                meta->state.store(State::Loading, std::memory_order_release);
            }
            meta->accessed.store(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    template <std::memory_order Order>
    size_t LookupT(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId,
                   size_t offset)
    {
        auto iNode = layout.Buckets()[iBucket].load(Order);
        size_t walked = 0;
        while (iNode != kInvalid && iNode < layout.SlotCount() && walked++ < layout.SlotCount()) {
            auto* meta = &layout.SlotMetaArr()[iNode];
            if (MatchKey(*meta, iBucket, blockId, offset)) { return iNode; }
            iNode = meta->next.load(Order);
        }
        return kInvalid;
    }

    size_t Lookup(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId, size_t offset)
    {
        return LookupT<std::memory_order_relaxed>(layout, iBucket, blockId, offset);
    }

    size_t LookupOptimistic(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId,
                            size_t offset)
    {
        return LookupT<std::memory_order_acquire>(layout, iBucket, blockId, offset);
    }

    size_t Alloc(CtrlLayout& layout, const Detail::BlockId& blockId, size_t offset, size_t iBucket,
                 bool allowReserved, size_t attempts, size_t initialReference)
    {
        for (size_t scan = 0; scan < attempts; ++scan) {
            auto iNode = FetchNode(layout, allowReserved);
            if (iNode == kInvalid) { continue; }
            auto* meta = &layout.SlotMetaArr()[iNode];
            size_t expected = 0;
            if (!meta->reference.compare_exchange_strong(expected, kSlotClaimed,
                                                         std::memory_order_acq_rel)) {
                meta->accessed.store(1, std::memory_order_relaxed);
                continue;
            }

            auto oldBucket = meta->hash.load(std::memory_order_acquire);
            if (oldBucket != iBucket) {
                if (oldBucket != kInvalid) {
                    auto* oldLock = layout.LockOf(oldBucket);
                    auto sameStripe = oldLock == layout.LockOf(iBucket);
                    if (!sameStripe && !oldLock->TryLock()) {
                        meta->accessed.store(1, std::memory_order_relaxed);
                        meta->reference.store(0, std::memory_order_release);
                        continue;
                    }
                    Remove(layout, oldBucket, iNode);
                    if (!sameStripe) { oldLock->Unlock(); }
                }
                StoreKey(*meta, blockId, offset);
                meta->state.store(State::Loading, std::memory_order_relaxed);
                MoveTo(layout, iBucket, iNode);
            } else {
                StoreKey(*meta, blockId, offset);
                meta->state.store(State::Loading, std::memory_order_relaxed);
            }
            meta->accessed.store(1, std::memory_order_relaxed);
            meta->reference.store(initialReference, std::memory_order_release);
            return iNode;
        }
        return kInvalid;
    }

    size_t FetchNode(CtrlLayout& layout, bool allowReserved)
    {
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        if (usable == 0 || myRank_ == kInvalid) { return kInvalid; }
        auto iNode = layout.NextClockSlot(myRank_, usable);
        if (layout.SlotMetaArr()[iNode].accessed.exchange(0, std::memory_order_relaxed) != 0) {
            return kInvalid;
        }
        return iNode;
    }

    static void MoveTo(CtrlLayout& layout, size_t iBucket, size_t iNode)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        auto& head = layout.Buckets()[iBucket];
        auto next = head.load(std::memory_order_relaxed);
        meta->next.store(next, std::memory_order_relaxed);
        meta->prev.store(kInvalid, std::memory_order_relaxed);
        if (next != kInvalid) {
            layout.SlotMetaArr()[next].prev.store(iNode, std::memory_order_relaxed);
        }
        meta->hash.store(iBucket, std::memory_order_release);
        head.store(iNode, std::memory_order_release);
    }

    static void Remove(CtrlLayout& layout, size_t iBucket, size_t iNode)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        auto prev = meta->prev.load(std::memory_order_relaxed);
        auto next = meta->next.load(std::memory_order_relaxed);
        if (prev != kInvalid) {
            layout.SlotMetaArr()[prev].next.store(next, std::memory_order_release);
        }
        if (next != kInvalid) {
            layout.SlotMetaArr()[next].prev.store(prev, std::memory_order_relaxed);
        }
        if (layout.Buckets()[iBucket].load(std::memory_order_relaxed) == iNode) {
            layout.Buckets()[iBucket].store(next, std::memory_order_release);
        }
        meta->prev.store(kInvalid, std::memory_order_relaxed);
        meta->next.store(kInvalid, std::memory_order_relaxed);
        meta->hash.store(kInvalid, std::memory_order_release);
    }

    void Release(size_t slotIdx)
    {
        ctrl_.Layout().SlotMetaArr()[slotIdx].reference.fetch_sub(1, std::memory_order_release);
    }

    State GetState(size_t slotIdx) const
    {
        return ctrl_.Layout().SlotMetaArr()[slotIdx].state.load(std::memory_order_acquire);
    }

    void MarkReady(size_t slotIdx)
    {
        ctrl_.Layout().SlotMetaArr()[slotIdx].state.store(State::Ready, std::memory_order_release);
    }

    void MarkFailed(size_t slotIdx)
    {
        ctrl_.Layout().SlotMetaArr()[slotIdx].state.store(State::Failed, std::memory_order_release);
    }
};

}  // namespace UC::Cache2
