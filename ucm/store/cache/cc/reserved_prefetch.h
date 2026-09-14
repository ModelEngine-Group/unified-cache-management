#ifndef UCM_CACHE_RESERVED_PREFETCH_H
#define UCM_CACHE_RESERVED_PREFETCH_H

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <list>
#include <future>
#include <thread>
#include <unordered_map>
#include "buffer_manager.h"
#include "early_stats.h"
#include "early_batch.h"
#include "early_buffer_pins.h"

namespace UC::CacheStore {

class ReservedPrefetch {
    // Per-worker conservative byte claims, including still-missing shards.
    // Actual cross-process exclusion remains the native buffer's CanReserve,
    // node references and bounded TryGet, not this local accounting.
    struct Budget {
        size_t limit;
        std::atomic<size_t> used{0};
        explicit Budget(size_t bytes) : limit(bytes) {}
    };
    struct Charge {
        std::shared_ptr<Budget> budget;
        size_t bytes{0};
        ~Charge() { if (budget) { budget->used.fetch_sub(bytes); } }
    };
    struct BlockPins {
        std::shared_ptr<Charge> charge;  // pins destruct before their claim is returned

        std::vector<TransBuffer::Handle> pins;
        bool ready{false};  // immutable once published in readyBlocks_
    };
    struct Reservation {
        std::shared_ptr<ReadLease> lease;
        std::vector<Detail::BlockId> blocks;
        std::vector<std::shared_ptr<BlockPins>> groups;
        TransBuffer::Handle& Pin(size_t i, size_t shards)
        { return groups[i / shards]->pins[i % shards]; }
        uint64_t handle{0};
        size_t initialMissing{0};
        std::mutex mutex;
        std::atomic<bool> cancelled{false};
        std::weak_ptr<Early::Batch> batch;
        bool started{false};
        bool ready{false};
    };
    struct ProbeOutcome {
        Status status{Status::OK()};
        std::shared_ptr<BlockPins> group;
    };
    struct ProbeFlight {
        std::promise<ProbeOutcome> promise;
        std::shared_future<ProbeOutcome> future{promise.get_future().share()};
    };
    std::unordered_map<Detail::BlockId, std::shared_ptr<ProbeFlight>, Detail::BlockIdHasher> locating_;
    std::shared_ptr<Budget> budget_;
    Expected<std::vector<std::shared_ptr<BlockPins>>> Locate(
        const Detail::BlockId* blocks, size_t count, EarlyStats& stats);
    Expected<std::shared_ptr<BlockPins>> NewGroup();
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

    void InvalidateLoadPlan(uint64_t handle)
    {
        if (loadPlan_ && std::find(loadPlan_->handles.begin(), loadPlan_->handles.end(), handle)
            != loadPlan_->handles.end()) { loadPlan_.reset(); }
    }

    // Caller holds mutex_. Hints never retain idle pins, and are optional.
    void Remember(const Detail::BlockId& block, const std::shared_ptr<BlockPins>& group)
    {
        if (!group->ready || hintBlockCapacity_ == 0) { return; }
        try {
            auto found = readyBlocks_.find(block);
            if (found == readyBlocks_.end()) {
                auto locations = std::make_shared<const std::vector<size_t>>(
                    EarlyBufferPins::Locations(group->pins));
                hintAge_.push_front(block);
                readyBlocks_.emplace(block, ReadyHint{group, locations, hintAge_.begin()});
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
        } catch (const std::bad_alloc&) {
            readyBlocks_.clear();
            hintAge_.clear();
        }
    }

    size_t PublishReady(const std::shared_ptr<Reservation>& r)
    {
        std::lock_guard guard(mutex_);
        if (r->cancelled.load()) { return readyBlocks_.size(); }
        if (r->ready) { InvalidateLoadPlan(r->handle); }
        for (size_t i = 0; i < r->blocks.size(); ++i) { Remember(r->blocks[i], r->groups[i]); }
        return readyBlocks_.size();
    }

public:
    ReservedPrefetch(BufferManager* manager, StoreV1* backend, size_t blockSize,
                     size_t shardSize, size_t timeoutMs)
        : backend_(backend), buffer_(manager->GetTransBuffer()),
          shardsPerBlock_(blockSize / shardSize), shardBytes_(shardSize),
          timeoutMs_(timeoutMs),
          hintBlockCapacity_(buffer_ ? EarlyBufferPins::HintBlockCapacity(*buffer_, shardsPerBlock_) : 0)
    {
        budget_ = std::make_shared<Budget>(buffer_ ? EarlyBufferPins::ReservableSlots(*buffer_) * shardBytes_ : 0);
    }

    Expected<PrefixReservation> Reserve(const Detail::BlockId* blocks, size_t num)
    {
        EarlyStats stats{"reserve"};
        if (!buffer_) { return Status::Error("Early requires a worker data buffer"); }
        auto r = std::make_shared<Reservation>();
        {
            std::lock_guard guard(mutex_);
            stats.handle = r->handle = next_++;
        }
        stats.protectionLimitBytes = budget_->limit;
        size_t count = num;
        if (num) {
            EarlyStats::Phase phase{stats.leaseMs, stats.leaseStartNs, stats.leaseEndNs};
            auto lease = backend_->AcquireReadLease(blocks, num);
            if (!lease) { return lease.Error(); }
            r->lease = std::move(lease.Value());
            r->blocks.assign(blocks, blocks + num);
        }
        std::vector<Detail::BlockId> needL2;
        std::vector<size_t> needL2Indices;
        {
            EarlyStats::Phase phase{stats.pinMs, stats.pinStartNs, stats.pinEndNs};
            auto located = Locate(blocks, num, stats);
            stats.protectedBytes = budget_->used.load();
            if (!located) { return located.Error(); }
            r->groups = std::move(located.Value());
            for (size_t b = 0; b < num; ++b) {
                if (!r->groups[b]->ready) {
                    needL2.push_back(blocks[b]);
                    needL2Indices.push_back(b);
                }
            }
        }
        if (!needL2.empty()) {
            EarlyStats::Phase phase{stats.lookupMs, stats.lookupStartNs, stats.lookupEndNs};
            ++stats.lookups;
            auto hit = backend_->LookupOnPrefix(needL2.data(), needL2.size());
            if (!hit) { return hit.Error(); }
            const auto firstMiss = static_cast<size_t>(hit.Value() + 1);
            if (firstMiss < needL2.size()) { count = needL2Indices[firstMiss]; }
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
        r->initialMissing = missing;
        const auto handle = stats.handle;
        auto batch = MakeBatch(r, handle);
        r->batch = batch;
        {
            std::lock_guard guard(mutex_);
            reservations_.emplace(handle, r);
        }
        stats.hintEntries = PublishReady(r);
        stats.handle = handle;
        stats.protectedBytes = budget_->used.load();
        stats.protectionLimitBytes = budget_->limit;
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
            InvalidateLoadPlan(handle);
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
#include "reserved_localize_impl.h"
#include "reserved_batch_impl.h"
#endif
