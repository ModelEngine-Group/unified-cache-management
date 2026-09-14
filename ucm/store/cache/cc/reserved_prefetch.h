#ifndef UCM_CACHE_RESERVED_PREFETCH_H
#define UCM_CACHE_RESERVED_PREFETCH_H

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "buffer_manager.h"
#include "early_stats.h"
#include "early_buffer_pins.h"

namespace UC::CacheStore {

class ReservedPrefetch {
    struct BlockPins {
        std::vector<TransBuffer::Handle> pins;
        bool ready{false};  // immutable once published in readyBlocks_
    };
    struct Reservation {
        std::vector<std::shared_ptr<ReadLease>> leases;
        std::vector<Detail::BlockId> blocks;
        std::unordered_map<Detail::BlockId, size_t, Detail::BlockIdHasher> blockOffsets;
        std::vector<std::shared_ptr<BlockPins>> groups;
        TransBuffer::Handle& Pin(size_t i, size_t shards)
        { return groups[i / shards]->pins[i % shards]; }
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

    std::unordered_map<Detail::BlockId, std::weak_ptr<BlockPins>, Detail::BlockIdHasher> readyBlocks_;

    void PublishReady(const std::shared_ptr<Reservation>& r)
    {
        std::lock_guard guard(mutex_);
        if (r->cancelled.load()) { return; }
        for (size_t i = 0; i < r->blocks.size(); ++i) {
            if (r->groups[i]->ready) { readyBlocks_[r->blocks[i]] = r->groups[i]; }
        }
    }

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
        EarlyStats stats{"reserve"};
        if (!buffer_) { return Status::Error("Early requires a worker data buffer"); }
        auto r = std::make_shared<Reservation>();
        size_t count = num;
        size_t batchBlocks = 32;
        for (size_t begin = 0; begin < num;) {
            const auto end = std::min(num, begin + batchBlocks);
            {
                EarlyStats::Phase phase{stats.leaseMs};
                bool extended = false;
                if (!r->leases.empty()) {
                    auto result = r->leases.back()->Extend(blocks + begin, end - begin);
                    if (!result) { return result.Error(); }
                    extended = result.Value();
                }
                if (!extended) {
                    auto lease = backend_->AcquireReadLease(blocks + begin, end - begin);
                    if (!lease) { return lease.Error(); }
                    r->leases.push_back(std::move(lease.Value()));
                }
            }
            r->blocks.insert(r->blocks.end(), blocks + begin, blocks + end);
            r->groups.resize(end);
            std::vector<Detail::BlockId> needL2;
            std::vector<size_t> needL2Indices;
            {
                EarlyStats::Phase phase{stats.pinMs};
                std::vector<Detail::BlockId> probe;
                std::vector<size_t> offsets;
                {
                    std::lock_guard guard(mutex_);
                    for (size_t b = begin; b < end; ++b) {
                        auto found = readyBlocks_.find(blocks[b]);
                        if (found != readyBlocks_.end()) { r->groups[b] = found->second.lock(); }
                        if (r->groups[b]) {
                            ++stats.reusedBlocks;
                        } else {
                            probe.push_back(blocks[b]);
                            offsets.push_back(b);
                        }
                    }
                }
                auto batch = EarlyBufferPins::PinBlocks(*buffer_, probe.data(), probe.size(), shardsPerBlock_);
                stats.shards += batch.pins.size();
                stats.bucketLocks += batch.bucketLocks;
                if (batch.capacityConflict) {
                    return Status::Error("prefix occupies native load-exclusive buffer");
                }
                for (size_t j = 0; j < offsets.size(); ++j) {
                    auto group = std::make_shared<BlockPins>();
                    group->pins.resize(shardsPerBlock_);
                    group->ready = true;
                    for (size_t shard = 0; shard < shardsPerBlock_; ++shard) {
                        auto& pin = group->pins[shard];
                        pin = std::move(batch.pins[j * shardsPerBlock_ + shard]);
                        if (!pin || !pin.Ready()) { group->ready = false; }
                    }
                    r->groups[offsets[j]] = std::move(group);
                }
                for (size_t b = begin; b < end; ++b) {
                    if (!r->groups[b]->ready) {
                        needL2.push_back(blocks[b]);
                        needL2Indices.push_back(b);
                    }
                }
            }
            if (!needL2.empty()) {
                EarlyStats::Phase phase{stats.lookupMs};
                ++stats.lookups;
                auto hit = backend_->LookupOnPrefix(needL2.data(), needL2.size());
                if (!hit) { return hit.Error(); }
                const auto firstMiss = static_cast<size_t>(hit.Value() + 1);
                if (firstMiss < needL2.size()) {
                    count = needL2Indices[firstMiss];
                    break;
                }
            }
            begin = end;
            batchBlocks = std::min(batchBlocks * 2, size_t{256});
        }
        r->blocks.resize(count);
        r->groups.resize(count);
        size_t missing = 0;
        for (const auto& group : r->groups) {
            if (group->ready) { continue; }
            size_t remaining = 0;
            for (const auto& pin : group->pins) { if (!pin || !pin.Ready()) { ++remaining; } }
            group->ready = remaining == 0;
            missing += remaining;
        }
        for (size_t i = 0; i < count; ++i) { r->blockOffsets.emplace(r->blocks[i], i); }
        r->ready = missing == 0;
        uint64_t handle;
        {
            std::lock_guard guard(mutex_);
            handle = next_++;
            reservations_.emplace(handle, r);
        }
        PublishReady(r);
        stats.handle = handle;
        stats.success = true;
        return PrefixReservation{handle, count, missing};
    }

    Expected<PrefetchResult> Prefetch(uint64_t handle)
    {
        EarlyStats stats{"prefetch"};
        stats.handle = handle;
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
        struct Transfer {
            Detail::TaskHandle handle{0};
            size_t first, count;
        };
        std::deque<Transfer> transfers;
        PrefetchResult result;
        auto complete = [&]() -> Status {
            const auto transfer = transfers.front();
            // Native Wait drains the backend even on failure. All destination
            // pins and read leases remain owned by r until every batch drains.
            EarlyStats::Phase phase{stats.waitMs};
            auto status = backend_->Wait(transfer.handle);
            if (status.Success()) {
                for (size_t i = transfer.first; i < transfer.first + transfer.count; ++i) {
                    r.Pin(owners[i], shardsPerBlock_).MarkReady();
                }
            }
            transfers.pop_front();
            return status;
        };
        auto run = [&]() -> Status {
            constexpr size_t batchShards = 64;
            constexpr size_t inflightBatches = 2;
            Detail::TaskDesc task;
            task.brief = "EarlyL2ToL1";
            size_t firstOwner = 0;
            auto submit = [&]() -> Status {
                if (task.empty()) { return Status::OK(); }
                const auto count = task.size();
                // Allocate bookkeeping before submission so an allocation
                // failure cannot lose the handle of an already accepted I/O.
                transfers.push_back({0, firstOwner, count});
                if (stats.firstIoMs < 0) { stats.firstIoMs = EarlyStats::Milliseconds(stats.elapsed); }
                const StopWatch submitting;
                auto transfer = backend_->Load(std::move(task));
                stats.submitMs += EarlyStats::Milliseconds(submitting);
                if (!transfer) {
                    transfers.pop_back();
                    return transfer.Error();
                }
                transfers.back().handle = transfer.Value();
                result.l2ReadShards += count;
                result.l2ReadBytes += count * shardBytes_;
                firstOwner += count;
                task = Detail::TaskDesc{};
                task.brief = "EarlyL2ToL1";
                if (transfers.size() == inflightBatches) { return complete(); }
                return Status::OK();
            };
            for (size_t i = 0; i < r.blocks.size() * shardsPerBlock_; ++i) {
                if (r.cancelled.load()) { return Status::Error("reserved prefetch cancelled"); }
                // This Prefetch is the sole pin writer. LoadHandles only reads
                // after ready is published; Release merely cancels and drops
                // the map reference. No per-shard reservation mutex is needed.
                auto& pin = r.Pin(i, shardsPerBlock_);
                if (pin) {
                    if (!pin.Ready()) { ++result.coalescedShards; }
                    continue;  // reuse READY and live in-flight positions
                }
                auto& block = r.blocks[i / shardsPerBlock_];
                auto shard = i % shardsPerBlock_;
                // Bounded capacity search; native load still uses its original policy.
                ++stats.shards;
                pin = buffer_->TryGet(block, shard);
                if (!pin) { return Status::Error("no space for reserved prefetch"); }
                if (pin.Ready()) { continue; }
                if (pin.Owner()) {
                    owners.push_back(i);
                    task.push_back({block, shard, {pin.Data()}});
                } else {
                    ++result.coalescedShards;
                }
                if (task.size() == batchShards) {
                    auto status = submit();
                    if (status.Failure()) { return status; }
                }
            }
            auto status = submit();
            if (status.Failure()) { return status; }
            while (!transfers.empty()) {
                status = complete();
                if (status.Failure()) { return status; }
            }
            EarlyStats::Phase phase{stats.waitMs};
            for (size_t i = 0; i < r.blocks.size() * shardsPerBlock_; ++i) {
                auto& pin = r.Pin(i, shardsPerBlock_);
                auto status = WaitReady(pin, timeoutMs_, r.cancelled);
                if (status.Failure()) { return status; }
            }
            return Status::OK();
        };
        auto status = Status::OK();
        try { status = run(); }
        catch (const std::exception& e) { status = Status::Error(fmt::format("prefetch exception: {}", e.what())); }
        // Includes cancellation, a later submit failure, or an allocation
        // failure while preparing the next batch. Never unpin in-flight I/O.
        while (!transfers.empty()) {
            if (transfers.front().handle == 0) {
                transfers.pop_front();
                continue;
            }
            auto drained = complete();
            if (status.Success() && drained.Failure()) { status = drained; }
        }
        stats.pinMs = std::max(0.0, EarlyStats::Milliseconds(stats.elapsed) - stats.waitMs - stats.submitMs);
        if (status.Failure()) {
            // Also cover an allocation exception between TryGet and recording
            // that owner in the batch. Borrowed positions belong to other I/O.
            for (size_t i = 0; i < r.blocks.size() * shardsPerBlock_; ++i) {
                auto& pin = r.Pin(i, shardsPerBlock_);
                if (pin && pin.Owner() && !pin.Ready()) { pin.MarkFailed(status); }
            }
            return status;
        }
        {
            std::lock_guard guard(r.mutex);
            if (r.cancelled.load()) { return Status::Error("reserved prefetch cancelled"); }
            for (const auto& group : r.groups) {
                if (!group->ready) { group->ready = true; }
            }
            r.ready = true;
        }
        PublishReady(reservation);
        stats.success = true;
        return result;
    }

    std::vector<TransBuffer::Handle> LoadHandles(const Detail::TaskDesc& task)
    {
        EarlyStats stats{"load_handles"};
        stats.shards = task.size();
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
                const auto& pin = reservation->groups[found->second]->pins[shard.index];
                if (!pin || !pin.Ready()) { continue; }
                if (handles.empty()) { handles.resize(task.size()); }
                // Handle's native copy acquires a node reference without a hash
                // lookup. It remains valid if Release runs before H2D completes.
                handles[i] = pin;
                ++stats.matched;
            }
        }
        stats.success = true;
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
        {
            std::lock_guard guard(reservation->mutex);
            // A running Prefetch keeps the reservation alive until I/O ends.
        }
        reservation.reset();
        std::lock_guard guard(mutex_);
        for (auto it = readyBlocks_.begin(); it != readyBlocks_.end();) {
            if (it->second.expired()) { it = readyBlocks_.erase(it); }
            else { ++it; }
        }
    }
};
}  // namespace UC::CacheStore
#endif
