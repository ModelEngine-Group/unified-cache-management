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
        std::shared_ptr<ReadLease> lease;
        std::vector<Detail::BlockId> blocks;
        std::vector<TransBuffer::Handle> pins;
        std::mutex mutex;
        std::atomic<bool> cancelled{false};
        bool started{false};
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
        // Protect the candidates before lookup, including the first missing block.
        auto lease = backend_->AcquireReadLease(blocks, num);
        if (!lease) { return lease.Error(); }
        auto reservation = std::make_shared<Reservation>();
        reservation->lease = std::move(lease.Value());
        reservation->blocks.assign(blocks, blocks + num);
        reservation->pins.resize(num * shardsPerBlock_);
        std::vector<Detail::BlockId> needL2;
        std::vector<size_t> needL2Indices;
        for (size_t b = 0; b < num; ++b) {
            bool complete = true;
            for (size_t shard = 0; shard < shardsPerBlock_; ++shard) {
                auto& pin = reservation->pins[b * shardsPerBlock_ + shard];
                bool capacityConflict = false;
                pin = buffer_->PinReady(blocks[b], shard, &capacityConflict);
                if (capacityConflict) {
                    return Status::Error("prefix occupies native load-exclusive buffer");
                }
                if (!pin) { complete = false; }
            }
            if (!complete) {
                needL2.push_back(blocks[b]);
                needL2Indices.push_back(b);
            }
        }
        // Native Cache lookup checks shard-0 existence, including pending dump
        // entries. Only a complete pinned block can bypass the native L2 lookup.
        // Missing KV is a short hit now, never a dependency on a future producer.
        size_t count = num;
        if (!needL2.empty()) {
            auto hit = backend_->LookupOnPrefix(needL2.data(), needL2.size());
            if (!hit) { return hit.Error(); }
            const auto firstMiss = static_cast<size_t>(hit.Value() + 1);
            if (firstMiss < needL2.size()) { count = needL2Indices[firstMiss]; }
        }
        reservation->blocks.resize(count);
        reservation->pins.resize(count * shardsPerBlock_);
        size_t missing = 0;
        for (const auto& pin : reservation->pins) { if (!pin) { ++missing; } }
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
                if (pin) { continue; }  // already pinned READY, never invalidate it
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
        return result;
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
