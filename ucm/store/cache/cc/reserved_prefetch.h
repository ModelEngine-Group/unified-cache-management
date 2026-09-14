#ifndef UCM_CACHE_RESERVED_PREFETCH_H
#define UCM_CACHE_RESERVED_PREFETCH_H

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <list>
#include <thread>
#include <unordered_map>
#include "buffer_manager.h"
#include "early_stats.h"
#include "early_batch.h"
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
        std::vector<std::shared_ptr<BlockPins>> groups;
        TransBuffer::Handle& Pin(size_t i, size_t shards)
        { return groups[i / shards]->pins[i % shards]; }
        std::mutex mutex;
        std::atomic<bool> cancelled{false};
        std::weak_ptr<Early::Batch> batch;
        bool started{false};
        bool ready{false};
    };
    class Batch;
    std::shared_ptr<Early::Batch> MakeBatch(const std::shared_ptr<Reservation>& r, uint64_t handle);
    struct LoadEntry {
        std::weak_ptr<Reservation> reservation;
        size_t blockOffset;
    };
    struct LoadPlan {
        std::vector<uint64_t> handles;
        std::unordered_map<Detail::BlockId, LoadEntry, Detail::BlockIdHasher> blocks;
    };
    std::shared_ptr<const LoadPlan> loadPlan_;
    StoreV1* backend_;
    TransBuffer* buffer_;
    size_t shardsPerBlock_, shardBytes_, timeoutMs_;
    std::mutex mutex_;
    uint64_t next_{1};
    std::unordered_map<uint64_t, std::shared_ptr<Reservation>> reservations_;

    struct ReadyHint {
        std::weak_ptr<BlockPins> live;
        std::shared_ptr<const std::vector<size_t>> locations;
        std::list<Detail::BlockId>::iterator age;
    };
    // Bounds only disposable location hints, never request admission or KV capacity.
    size_t hintBlockCapacity_;
    std::list<Detail::BlockId> hintAge_;
    std::unordered_map<Detail::BlockId, ReadyHint, Detail::BlockIdHasher> readyBlocks_;

    size_t PublishReady(const std::shared_ptr<Reservation>& r)
    {
        std::lock_guard guard(mutex_);
        if (r->cancelled.load()) { return readyBlocks_.size(); }
        if (r->ready) { loadPlan_.reset(); }
        try {
            for (size_t i = 0; i < r->blocks.size(); ++i) {
                const auto& group = r->groups[i];
                if (!group->ready || hintBlockCapacity_ == 0) { continue; }
                auto found = readyBlocks_.find(r->blocks[i]);
                if (found == readyBlocks_.end()) {
                    auto locations = std::make_shared<const std::vector<size_t>>(
                        EarlyBufferPins::Locations(group->pins));
                    hintAge_.push_front(r->blocks[i]);
                    readyBlocks_.emplace(r->blocks[i], ReadyHint{group, locations, hintAge_.begin()});
                } else {
                    if (found->second.live.lock() != group) {
                        found->second.locations = std::make_shared<const std::vector<size_t>>(
                            EarlyBufferPins::Locations(group->pins));
                    }
                    found->second.live = group;
                    hintAge_.splice(hintAge_.begin(), hintAge_, found->second.age);
                }
                while (readyBlocks_.size() > hintBlockCapacity_) {
                    readyBlocks_.erase(hintAge_.back());
                    hintAge_.pop_back();
                }
            }
        } catch (const std::bad_alloc&) {
            // Optional hints must not turn a successful, already registered
            // reservation into an exception with no handle returned to its owner.
            readyBlocks_.clear();
            hintAge_.clear();
        }
        return readyBlocks_.size();
    }


public:
    ReservedPrefetch(BufferManager* manager, StoreV1* backend, size_t blockSize,
                     size_t shardSize, size_t timeoutMs)
        : backend_(backend), buffer_(manager->GetTransBuffer()),
          shardsPerBlock_(blockSize / shardSize), shardBytes_(shardSize),
          timeoutMs_(timeoutMs),
          hintBlockCapacity_(buffer_ ? EarlyBufferPins::HintBlockCapacity(*buffer_, shardsPerBlock_) : 0) {}

    Expected<PrefixReservation> Reserve(const Detail::BlockId* blocks, size_t num)
    {
        EarlyStats stats{"reserve"};
        if (!buffer_) { return Status::Error("Early requires a worker data buffer"); }
        auto r = std::make_shared<Reservation>();
        {
            std::lock_guard guard(mutex_);
            stats.handle = next_++;
        }
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
                std::vector<size_t> offsets, reused;
                std::vector<std::pair<size_t, std::shared_ptr<const std::vector<size_t>>>> hints;
                {
                    std::lock_guard guard(mutex_);
                    for (size_t b = begin; b < end; ++b) {
                        auto found = readyBlocks_.find(blocks[b]);
                        if (found != readyBlocks_.end()) {
                            r->groups[b] = found->second.live.lock();
                            hintAge_.splice(hintAge_.begin(), hintAge_, found->second.age);
                        }
                        if (r->groups[b]) {
                            ++stats.reusedBlocks;
                            reused.push_back(b);
                        } else if (found != readyBlocks_.end() && found->second.locations->size() == shardsPerBlock_) {
                            hints.emplace_back(b, found->second.locations);
                        } else {
                            probe.push_back(blocks[b]);
                            offsets.push_back(b);
                        }
                    }
                }
                for (const auto& [b, locations] : hints) {
                    auto pins = EarlyBufferPins::Recall(*buffer_, blocks[b], *locations, stats.hintChecks);
                    if (pins.empty()) {
                        probe.push_back(blocks[b]);
                        offsets.push_back(b);
                    } else {
                        auto group = std::make_shared<BlockPins>();
                        group->pins = std::move(pins);
                        group->ready = true;
                        r->groups[b] = std::move(group);
                        ++stats.recalledBlocks;
                    }
                }
                for (auto b : reused) { EarlyBufferPins::Touch(*buffer_, r->groups[b]->pins); }
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
        r->ready = missing == 0;
        const auto handle = stats.handle;
        auto batch = MakeBatch(r, handle);
        r->batch = batch;
        {
            std::lock_guard guard(mutex_);
            reservations_.emplace(handle, r);
        }
        stats.hintEntries = PublishReady(r);
        stats.handle = handle;
        stats.success = true;
        return PrefixReservation{handle, count, missing, std::move(batch)};
    }

    Expected<PrefetchResult> Prefetch(uint64_t handle)
    {
        std::shared_ptr<Reservation> r;
        {
            std::lock_guard guard(mutex_);
            auto it = reservations_.find(handle);
            if (it == reservations_.end()) { return Status::Error("unknown reservation"); }
            r = it->second;
        }
        std::shared_ptr<Early::Batch> batch;
        {
            std::lock_guard guard(r->mutex);
            batch = r->batch.lock();
            if (!batch) { batch = MakeBatch(r, handle); r->batch = batch; }
        }
        return batch->Run();
    }

    std::vector<TransBuffer::Handle> LoadHandles(const Detail::TaskDesc& task)
    {
        EarlyStats stats{"load_handles"};
        stats.shards = task.size();
        std::shared_ptr<const LoadPlan> plan;
        {
            std::lock_guard guard(mutex_);
            if (!loadPlan_ || loadPlan_->handles != task.loadReservations) {
                auto next = std::make_shared<LoadPlan>();
                next->handles = task.loadReservations;
                for (auto handle : task.loadReservations) {
                    auto found = reservations_.find(handle);
                    if (found == reservations_.end()) { continue; }
                    const auto& r = found->second;
                    std::lock_guard lock(r->mutex);
                    if (!r->ready || r->cancelled.load()) { continue; }
                    for (size_t b = 0; b < r->blocks.size(); ++b) {
                        next->blocks.emplace(r->blocks[b], LoadEntry{r, b});
                    }
                }
                loadPlan_ = std::move(next);
                ++stats.indexBuilds;
            }
            plan = loadPlan_;
        }
        // Resolve each requested block once. Plans contain weak references and
        // are invalidated on READY/release; they never keep a consumer alive.
        std::unordered_map<Detail::BlockId, std::shared_ptr<BlockPins>, Detail::BlockIdHasher> selected;
        const Detail::BlockId* previous = nullptr;
        std::shared_ptr<BlockPins> group;
        std::vector<TransBuffer::Handle> handles;
        for (size_t i = 0; i < task.size(); ++i) {
            const auto& shard = task[i];
            if (shard.index >= shardsPerBlock_) { continue; }
            if (!previous || *previous != shard.owner) {
                previous = &shard.owner;
                auto [slot, inserted] = selected.try_emplace(shard.owner);
                if (inserted) {
                    ++stats.blockLookups;
                    auto entry = plan->blocks.find(shard.owner);
                    if (entry != plan->blocks.end()) {
                        if (auto r = entry->second.reservation.lock()) {
                            std::lock_guard lock(r->mutex);
                            if (r->ready && !r->cancelled.load()) {
                                slot->second = r->groups[entry->second.blockOffset];
                            }
                        }
                    }
                }
                group = slot->second;
            }
            if (!group) { continue; }
            const auto& pin = group->pins[shard.index];
            if (!pin || !pin.Ready()) { continue; }
            if (handles.empty()) { handles.resize(task.size()); }
            handles[i] = pin;  // native node reference, held until H2D completes
            ++stats.matched;
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
            loadPlan_.reset();
        }
        reservation->cancelled.store(true);
        {
            std::lock_guard guard(reservation->mutex);
            // A running Prefetch keeps the reservation alive until I/O ends.
        }
        reservation.reset();
        // Only weak ownership and bounded node-number hints remain. A future
        // consumer must atomically validate a hint and acquire new references.
    }
};
}  // namespace UC::CacheStore
#include "reserved_batch_impl.h"
#endif
