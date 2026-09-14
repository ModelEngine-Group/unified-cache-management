#ifndef UCM_CACHE_RESERVED_PREFETCH_H
#define UCM_CACHE_RESERVED_PREFETCH_H

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "buffer_manager.h"

namespace UC::CacheStore {

class ReservedPrefetch {
    struct Reservation {
        std::vector<std::shared_ptr<ReadLease>> leases;
        std::vector<Detail::BlockId> blocks;
        std::unordered_map<Detail::BlockId, size_t, Detail::BlockIdHasher> blockOffsets;
        std::vector<TransBuffer::Handle> pins;
        std::mutex mutex;
        std::atomic<bool> cancelled{false};
        bool started{false};
        bool ready{false};
    };
    StoreV1* backend_;
    TransBuffer* buffer_;
    size_t shardsPerBlock_, shardBytes_, timeoutMs_;
    std::mutex mutex_;
    uint64_t next_{1};
    std::unordered_map<uint64_t, std::shared_ptr<Reservation>> reservations_;

    static Status WaitReady(const TransBuffer::Handle& pin, size_t timeoutMs,
                            const std::atomic<bool>& cancelled)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        for (;;) {
            if (cancelled.load()) { return Status::Error("reserved prefetch cancelled"); }
            const auto state = pin.GetState();
            if (state == TransBuffer::State::READY) { return Status::OK(); }
            if (state == TransBuffer::State::FAILED) { return pin.FailureStatus(); }
            if (std::chrono::steady_clock::now() >= deadline) {
                return Status::Error("shared shard readiness timeout");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

public:
    ReservedPrefetch(BufferManager* manager, StoreV1* backend, size_t blockSize,
                     size_t shardSize, size_t timeoutMs)
        : backend_(backend), buffer_(manager->GetTransBuffer()),
          shardsPerBlock_(blockSize / shardSize), shardBytes_(shardSize),
          timeoutMs_(timeoutMs) {}

    Expected<PrefixReservation> Reserve(const Detail::BlockId* blocks, size_t num)
    {
        if (!buffer_) { return Status::Error("Early requires a worker data buffer"); }
        auto reservation = std::make_shared<Reservation>();
        // Bound speculative pin/lease work after a short hit. Do not stop on an
        // L2 miss alone: a complete, protected L1-only block remains a valid hit.
        constexpr size_t candidateBatchBlocks = 32;
        size_t count = num;
        for (size_t begin = 0; begin < num; begin += candidateBatchBlocks) {
            const auto end = std::min(num, begin + candidateBatchBlocks);
            auto lease = backend_->AcquireReadLease(blocks + begin, end - begin);
            if (!lease) { return lease.Error(); }
            reservation->leases.push_back(std::move(lease.Value()));
            reservation->blocks.insert(reservation->blocks.end(), blocks + begin, blocks + end);
            reservation->pins.resize(end * shardsPerBlock_);
            std::vector<Detail::BlockId> needL2;
            std::vector<size_t> needL2Indices;
            for (size_t b = begin; b < end; ++b) {
                bool complete = true;
                for (size_t shard = 0; shard < shardsPerBlock_; ++shard) {
                    auto& pin = reservation->pins[b * shardsPerBlock_ + shard];
                    bool capacityConflict = false;
                    pin = buffer_->PinReady(blocks[b], shard, &capacityConflict, true);
                    if (capacityConflict) {
                        return Status::Error("prefix occupies native load-exclusive buffer");
                    }
                    if (!pin || !pin.Ready()) { complete = false; }
                }
                if (!complete) {
                    needL2.push_back(blocks[b]);
                    needL2Indices.push_back(b);
                }
            }
            if (!needL2.empty()) {
                auto hit = backend_->LookupOnPrefix(needL2.data(), needL2.size());
                if (!hit) { return hit.Error(); }
                const auto firstMiss = static_cast<size_t>(hit.Value() + 1);
                if (firstMiss < needL2.size()) {
                    count = needL2Indices[firstMiss];
                    break;
                }
            }
        }
        reservation->blocks.resize(count);
        reservation->pins.resize(count * shardsPerBlock_);
        size_t missing = 0;
        for (const auto& pin : reservation->pins) {
            if (!pin || !pin.Ready()) { ++missing; }
        }
        for (size_t i = 0; i < count; ++i) {
            reservation->blockOffsets.emplace(reservation->blocks[i], i);
        }
        reservation->ready = missing == 0;
        std::lock_guard guard(mutex_);
        const auto handle = next_++;
        reservations_.emplace(handle, std::move(reservation));
        return PrefixReservation{handle, count, missing};
    }

    Expected<PrefetchResult> Prefetch(uint64_t handle)
    {
        std::shared_ptr<Reservation> reservation;
        {
            std::lock_guard guard(mutex_);
            auto it = reservations_.find(handle);
            if (it == reservations_.end()) { return Status::Error("unknown reservation"); }
            reservation = it->second;
        }
        auto& r = *reservation;
        {
            std::lock_guard guard(r.mutex);
            if (r.started) { return Status::Error("reservation already prefetched"); }
            r.started = true;
        }
        std::vector<size_t> owners;
        PrefetchResult result;
        auto run = [&]() -> Status {
            Detail::TaskDesc task;
            task.brief = "EarlyL2ToL1";
            for (size_t i = 0; i < r.pins.size(); ++i) {
                if (r.cancelled.load()) { return Status::Error("reserved prefetch cancelled"); }
                std::lock_guard guard(r.mutex);
                auto& pin = r.pins[i];
                if (pin) {
                    if (!pin.Ready()) { ++result.coalescedShards; }
                    continue;  // reuse READY and live in-flight positions
                }
                auto& block = r.blocks[i / shardsPerBlock_];
                auto shard = i % shardsPerBlock_;
                // Bounded capacity search; native load still uses its original policy.
                pin = buffer_->TryGet(block, shard);
                if (!pin) { return Status::Error("no space for reserved prefetch"); }
                if (pin.Ready()) { continue; }
                if (pin.Owner()) {
                    owners.push_back(i);
                    task.push_back({block, shard, {pin.Data()}});
                } else {
                    ++result.coalescedShards;
                }
            }
            if (!task.empty()) {
                result.l2ReadShards = task.size();
                result.l2ReadBytes = task.size() * shardBytes_;
                auto transfer = backend_->Load(std::move(task));
                if (!transfer) { return transfer.Error(); }
                // The native backend owns bounded I/O. Cancellation cannot unpin
                // its destination until Wait has returned.
                auto status = backend_->Wait(transfer.Value());
                if (status.Failure()) { return status; }
                for (auto index : owners) { r.pins[index].MarkReady(); }
            }
            for (auto& pin : r.pins) {
                auto status = WaitReady(pin, timeoutMs_, r.cancelled);
                if (status.Failure()) { return status; }
            }
            return Status::OK();
        };
        auto status = Status::OK();
        try { status = run(); }
        catch (const std::exception& e) { status = Status::Error(fmt::format("prefetch exception: {}", e.what())); }
        if (status.Failure()) {
            for (auto index : owners) {
                if (!r.pins[index].Ready()) { r.pins[index].MarkFailed(status); }
            }
            return status;
        }
        {
            std::lock_guard guard(r.mutex);
            if (r.cancelled.load()) { return Status::Error("reserved prefetch cancelled"); }
            r.ready = true;
        }
        return result;
    }

    std::vector<TransBuffer::Handle> LoadHandles(const Detail::TaskDesc& task)
    {
        std::vector<std::shared_ptr<Reservation>> candidates;
        {
            std::lock_guard guard(mutex_);
            for (auto handle : task.loadReservations) {
                auto it = reservations_.find(handle);
                if (it != reservations_.end()) { candidates.push_back(it->second); }
            }
        }
        std::vector<TransBuffer::Handle> handles;
        for (const auto& reservation : candidates) {
            std::lock_guard guard(reservation->mutex);
            if (!reservation->ready || reservation->cancelled.load()) { continue; }
            for (size_t i = 0; i < task.size(); ++i) {
                if (!handles.empty() && handles[i]) { continue; }
                const auto& shard = task[i];
                if (shard.index >= shardsPerBlock_) { continue; }
                auto found = reservation->blockOffsets.find(shard.owner);
                if (found == reservation->blockOffsets.end()) { continue; }
                const auto& pin = reservation->pins[found->second * shardsPerBlock_ + shard.index];
                if (!pin || !pin.Ready()) { continue; }
                if (handles.empty()) { handles.resize(task.size()); }
                // Handle's native copy acquires a node reference without a hash
                // lookup. It remains valid if Release runs before H2D completes.
                handles[i] = pin;
            }
        }
        return handles;
    }

    void Release(uint64_t handle)
    {
        std::shared_ptr<Reservation> reservation;
        {
            std::lock_guard guard(mutex_);
            auto it = reservations_.find(handle);
            if (it == reservations_.end()) { return; }
            reservation = std::move(it->second);
            reservations_.erase(it);
        }
        reservation->cancelled.store(true);
        std::lock_guard guard(reservation->mutex);
        // A running Prefetch keeps the entire reservation alive until I/O ends.
    }
};
}  // namespace UC::CacheStore
#endif
