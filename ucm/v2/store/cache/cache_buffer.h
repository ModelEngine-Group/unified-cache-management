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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "policy/replacement_policy.h"
#include "status/status.h"
#include "store.h"
#include "type/block.h"

namespace UC::Store {

/**
 * @brief Block-level business layer over a ReplacementPolicy: owns the
 *        block-to-slot mapping, the two-phase admission visibility and
 *        the per-block TTL bookkeeping.
 *
 * The buffer maps the TokenAddressed protocol onto the slot-level
 * engine: Reserve() backs Alloc(), Commit() concludes reservations,
 * Pin() resolves and guards read sources, and Lookup()/Touch() are
 * the presence and use probes. Reservations are invisible to Lookup
 * until their Commit; concurrent admissions of the same block share
 * one reservation, one reference each, and it concludes when the last
 * matching Commit arrives. Re-admissions keep the block visible at
 * its slot and pin it until concluded; a block that has expired by
 * the time of the successful conclusion is revived with the fresh
 * deadline (a failed one leaves it to expire), while a live block's
 * retention is extended only by Touch().
 *
 * TTL: deadlines start at Commit and use a monotonic clock; a block
 * admitted without a TTL fact never expires. Touch() raises a
 * deadline to the larger of the existing one and now + the context's
 * TTL fact, never shortening it. Expiry is lazy -- probed on
 * Lookup/Touch/Pin, and swept off a deadline-ordered index before
 * the eviction discipline is consulted once the pool is full -- and
 * deferred while the block is pinned.
 *
 * @note Not thread-safe: callers must serialize access.
 */
class CacheBuffer {
public:
    /**
     * @brief Sets up the buffer: fixes the slot pool size and takes
     *        ownership of the replacement policy.
     *
     * @param capacity         Slot count of the fixed pool; must be > 0.
     * @param replacementPolicy Registry-created policy instance; must not
     *                         be null.
     *
     * @return Status::Ok() on success; InvalidParam on bad arguments,
     *         otherwise the policy's Setup() result. On error the buffer
     *         keeps its previous state.
     *
     * @note May be called again to reset the buffer.
     */
    Status Setup(std::size_t capacity, std::unique_ptr<ReplacementPolicy> replacementPolicy)
    {
        if (capacity == 0) { return Status::InvalidParam("capacity must be > 0"); }
        if (replacementPolicy == nullptr) {
            return Status::InvalidParam("replacement policy is required");
        }
        Status res = replacementPolicy->Setup(capacity);
        if (res.Failure()) { return res; }
        capacity_ = capacity;
        policy_ = std::move(replacementPolicy);
        committed_.clear();
        pending_.clear();
        occupants_.assign(capacity, BlockId{});
        reservedSlots_ = 0;
        expiryHeap_.clear();
        deferredExpiry_.clear();
        return Status::Ok();
    }
    /**
     * @brief Probes the presence of the given blocks: hits[i] is
     *        non-zero iff blockIds[i] is committed and unexpired.
     *
     * Reservations are not visible until their Commit. Expired entries
     * are dropped in place, unless pinned (reported as a miss, removal
     * deferred). Returns all misses before a successful Setup.
     */
    std::vector<std::uint8_t> Lookup(const std::vector<BlockId>& blockIds)
    {
        std::vector<std::uint8_t> hits(blockIds.size(), 0);
        if (policy_ == nullptr) { return hits; }
        const auto now = Clock::now();
        for (std::size_t i = 0; i < blockIds.size(); i++) {
            auto it = committed_.find(blockIds[i]);
            if (it == committed_.end()) { continue; }
            if (Expired(it->second, now)) {
                DropExpired(it);
                continue;
            }
            hits[i] = 1;
        }
        return hits;
    }
    /**
     * @brief Reserves slots for blocks about to enter the buffer:
     *        backs TokenAddressed::Alloc.
     *
     * Already-committed blocks keep their slot; the re-admission is
     * concluded by Commit, which revives a block expired by then
     * with the fresh deadline on success (a live block keeps its
     * deadline; retention extends via Touch()). The slot is pinned
     * until then, so the in-flight rewrite cannot be displaced.
     * Already-reserved blocks join the outstanding reservation with
     * one reference per call and the latest TTL fact
     * (last-writer-wins). New blocks take free slots, or the
     * slots of evicted victims once the pool is full -- expired blocks
     * are released before the eviction discipline is consulted.
     * Already-known blocks are settled before any newcomer is placed,
     * so batch members can never be displaced by their own batch and
     * the outcome is independent of the order within blockIds;
     * duplicate entries within the batch share one reservation.
     *
     * @return Expected per-block slot vector on success, parallel to
     *         blockIds: the write destinations of the matching Dump
     *         jobs. NoSpace before a successful Setup or when the pool
     *         cannot host the batch -- the batch is then rolled back
     *         completely (reservations cancelled, evicted blocks
     *         restored to their slots); only the release of expired
     *         blocks persists.
     */
    Expected<std::vector<std::size_t>> Reserve(const std::vector<BlockId>& blockIds,
                                               const RequestContext& context)
    {
        const std::size_t newcomers = CountNewcomers(blockIds);
        if (newcomers > capacity_) {
            return Status::NoSpace("admission of {} new blocks exceeds the capacity of {}",
                                   newcomers, capacity_);
        }
        ReserveBatch batch;
        batch.slots.assign(blockIds.size(), 0);
        const auto now = Clock::now();
        SettleKnownBlocks(blockIds, context, batch);
        for (const std::size_t i : batch.newcomerIndices) {
            auto pending = pending_.find(blockIds[i]);  // earlier duplicate in this batch
            if (pending != pending_.end()) {
                batch.slots[i] = pending->second.slot;
                continue;
            }
            auto slot = PlaceNewcomer(blockIds[i], context, batch, now);
            if (!slot) {
                Rollback(batch.journal);
                ReturnSharedRefs(batch.sharedRefs);
                return slot.Error();
            }
            batch.slots[i] = *slot;
        }
        return batch.slots;
    }
    /**
     * @brief Concludes reservations: backs TokenAddressed::Commit.
     *
     * Every Commit occurrence drops one reference of the matching
     * reservation; concurrent admissions of the same block share it,
     * and the reservation concludes when the last reference arrives --
     * committed if any sharer concluded with success, released when
     * none did. On conclusion: new reservations become visible at
     * their slot with the deadline derived from the admission's TTL
     * fact; re-admissions revive a block that has expired in the
     * meantime with the fresh deadline and refresh the slot's
     * hotness. A failed re-admission leaves the block exactly as it
     * was (no revival, no hotness refresh). Blocks without a live
     * reservation are skipped.
     */
    void Commit(const std::vector<BlockId>& blockIds, bool success)
    {
        const auto now = Clock::now();
        for (const auto& blockId : blockIds) {
            auto it = pending_.find(blockId);
            if (it == pending_.end()) { continue; }
            if (success) { it->second.anySuccess = true; }
            if (--it->second.refs > 0) { continue; }  // another admission is in flight
            const PendingEntry entry = it->second;
            pending_.erase(it);
            if (entry.readmitted) {
                auto committed = committed_.find(blockId);
                if (entry.anySuccess && committed != committed_.end()) {
                    if (Expired(committed->second, now)) {
                        const std::optional<TimePoint> deadline = DeadlineFrom(entry.ttl, now);
                        committed->second.deadline = deadline;
                        if (deadline) { PushExpiry({*deadline, blockId}); }
                    }
                    policy_->Touch(entry.slot);
                }
                policy_->Unpin(entry.slot);
                continue;
            }
            reservedSlots_--;
            if (!entry.anySuccess) {
                policy_->AbortSlot(entry.slot);
                continue;
            }
            policy_->CommitSlot(entry.slot);
            const std::optional<TimePoint> deadline = DeadlineFrom(entry.ttl, now);
            committed_[blockId] = CommittedEntry{entry.slot, deadline};
            occupants_[entry.slot] = blockId;
            if (deadline) { PushExpiry({*deadline, blockId}); }
        }
        if (expiryHeap_.size() > 2 * committed_.size() + 16) { CompactExpiry(); }
    }
    /**
     * @brief Pins the given blocks and resolves their slots: backs
     *        TokenAddressed::Pin.
     *
     * @return Expected per-block slot vector on success, parallel to
     *         blockIds: the read sources of the matching Load jobs.
     *         NotFound before a successful Setup or when any block is
     *         missing or expired, with nothing pinned.
     */
    Expected<std::vector<std::size_t>> Pin(const std::vector<BlockId>& blockIds)
    {
        std::vector<std::size_t> slots(blockIds.size(), 0);
        const auto now = Clock::now();
        for (std::size_t i = 0; i < blockIds.size(); i++) {
            auto it = committed_.find(blockIds[i]);
            if (it == committed_.end()) {
                return Status::NotFound("blockIds[{}] is not present", i);
            }
            if (Expired(it->second, now)) {
                DropExpired(it);
                return Status::NotFound("blockIds[{}] is not present", i);
            }
            slots[i] = it->second.slot;
        }
        for (std::size_t i = 0; i < blockIds.size(); i++) {
            auto it = committed_.find(blockIds[i]);
            if (it == committed_.end()) { continue; }  // defensive: validated above
            it->second.externalPins++;
            policy_->Pin(it->second.slot);
        }
        return slots;
    }
    /**
     * @brief Releases pins previously taken via Pin; idempotent --
     *        backs TokenAddressed::Unpin.
     *
     * Unpinning blocks that are not pinned this way -- absent, or only
     * pinned by an in-flight re-admission -- is a no-op.
     */
    void Unpin(const std::vector<BlockId>& blockIds)
    {
        for (const auto& blockId : blockIds) {
            auto it = committed_.find(blockId);
            if (it == committed_.end() || it->second.externalPins == 0) { continue; }
            it->second.externalPins--;
            policy_->Unpin(it->second.slot);
        }
    }
    /**
     * @brief Refreshes the hotness of the given blocks and extends
     *        their retention -- backs Store::Touch.
     *
     * A TTL fact in the context raises each block's deadline to the
     * larger of the existing one and now + ttl; deadlines are never
     * shortened, and blocks without one (never-expiring) stay that
     * way. Absent and expired blocks are ignored.
     */
    void Touch(const std::vector<BlockId>& blockIds, const RequestContext& context)
    {
        if (policy_ == nullptr) { return; }
        const auto now = Clock::now();
        for (const auto& blockId : blockIds) {
            auto it = committed_.find(blockId);
            if (it == committed_.end()) { continue; }
            if (Expired(it->second, now)) {
                DropExpired(it);
                continue;
            }
            policy_->Touch(it->second.slot);
            if (context.ttl && it->second.deadline) {
                const TimePoint extended = now + *context.ttl;
                if (*it->second.deadline < extended) {
                    it->second.deadline = extended;
                    PushExpiry({extended, blockId});
                }
            }
        }
        if (expiryHeap_.size() > 2 * committed_.size() + 16) { CompactExpiry(); }
    }

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    struct CommittedEntry {
        std::size_t slot;
        std::optional<TimePoint> deadline;
        /** Pins taken via Pin(), distinct from internal re-admission pins. */
        std::size_t externalPins{0};
    };
    struct PendingEntry {
        std::size_t slot;
        std::optional<std::chrono::milliseconds> ttl;
        bool readmitted;
        /** Admissions sharing this reservation; it concludes at zero. */
        std::size_t refs{1};
        /** Whether any sharer has concluded successfully so far. */
        bool anySuccess{false};
    };
    struct Reservation {
        BlockId block;
        std::size_t slot;
        bool displaced;
        bool readmitted;
        BlockId victim;
        std::optional<TimePoint> victimDeadline;
    };
    /** A reference this Reserve call added to a pre-existing pending reservation. */
    struct SharedRef {
        BlockId block;
        std::optional<std::chrono::milliseconds> ttl;
    };
    /** Expiry-index entry: one admission-time deadline fact of a committed block. */
    struct DeadlineEntry {
        TimePoint deadline;
        BlockId block;
        bool operator>(const DeadlineEntry& other) const { return deadline > other.deadline; }
    };
    /** Scratch state of one Reserve() call. */
    struct ReserveBatch {
        /** This call's own reservations, for rollback. */
        std::vector<Reservation> journal;
        /** References this call joined into foreign pending reservations. */
        std::vector<SharedRef> sharedRefs;
        /** Per-input slot results, parallel to the Reserve() input. */
        std::vector<std::size_t> slots;
        /** Input positions of the blocks still needing a slot. */
        std::vector<std::size_t> newcomerIndices;
    };

    static std::optional<TimePoint> DeadlineFrom(
        const std::optional<std::chrono::milliseconds>& ttl, TimePoint now)
    {
        if (!ttl) { return std::nullopt; }
        return now + *ttl;
    }

    static bool Expired(const CommittedEntry& entry, TimePoint now)
    {
        return entry.deadline && *entry.deadline <= now;
    }

    void DropExpired(std::unordered_map<BlockId, CommittedEntry>::iterator it)
    {
        if (policy_->IsSlotPinned(it->second.slot)) { return; }
        policy_->ReleaseSlot(it->second.slot);
        committed_.erase(it);
    }

    void PushExpiry(const DeadlineEntry& entry)
    {
        expiryHeap_.push_back(entry);
        std::push_heap(expiryHeap_.begin(), expiryHeap_.end(), std::greater<>{});
    }

    /** Drops entries superseded by a newer deadline or by the block leaving. */
    void CompactExpiry()
    {
        std::vector<DeadlineEntry> valid;
        valid.reserve(committed_.size());
        for (const auto& entry : expiryHeap_) {
            auto it = committed_.find(entry.block);
            if (it != committed_.end() && it->second.deadline == entry.deadline) {
                valid.push_back(entry);
            }
        }
        expiryHeap_ = std::move(valid);
        std::make_heap(expiryHeap_.begin(), expiryHeap_.end(), std::greater<>{});
    }

    void SweepExpired(TimePoint now)
    {
        for (const auto& entry : deferredExpiry_) { PushExpiry(entry); }
        deferredExpiry_.clear();
        while (!expiryHeap_.empty() && expiryHeap_.front().deadline <= now) {
            const DeadlineEntry entry = expiryHeap_.front();
            std::pop_heap(expiryHeap_.begin(), expiryHeap_.end(), std::greater<>{});
            expiryHeap_.pop_back();
            auto it = committed_.find(entry.block);
            if (it == committed_.end() || it->second.deadline != entry.deadline) { continue; }
            if (policy_->IsSlotPinned(it->second.slot)) {
                deferredExpiry_.push_back(entry);
                continue;
            }
            policy_->ReleaseSlot(it->second.slot);
            committed_.erase(it);
        }
    }

    /** Distinct blocks of the batch that are neither committed nor pending. */
    std::size_t CountNewcomers(const std::vector<BlockId>& blockIds) const
    {
        std::unordered_set<BlockId> seen;
        std::size_t newcomers = 0;
        for (const auto& blockId : blockIds) {
            if (!seen.insert(blockId).second) { continue; }  // batch duplicate
            if (committed_.count(blockId) == 0 && pending_.count(blockId) == 0) { newcomers++; }
        }
        return newcomers;
    }

    /**
     * @brief Phase 1 of Reserve(): settles every already-known block --
     *        pending ones join their reservation, committed ones open a
     *        re-admission -- before any newcomer is placed, so batch
     *        members can never be displaced by their own batch.
     */
    void SettleKnownBlocks(const std::vector<BlockId>& blockIds, const RequestContext& context,
                           ReserveBatch& batch)
    {
        for (std::size_t i = 0; i < blockIds.size(); i++) {
            const BlockId& blockId = blockIds[i];
            auto pending = pending_.find(blockId);
            if (pending != pending_.end()) {
                // One reference per listed occurrence: a mirroring Commit
                // decrements once per occurrence as well.
                batch.sharedRefs.push_back(SharedRef{blockId, pending->second.ttl});
                pending->second.ttl = context.ttl;
                pending->second.refs++;
                batch.slots[i] = pending->second.slot;
                continue;
            }
            auto committed = committed_.find(blockId);
            if (committed != committed_.end()) {
                const std::size_t slot = committed->second.slot;
                policy_->Pin(slot);  // guards the in-flight rewrite until Commit
                batch.journal.push_back(Reservation{blockId, slot, false, true, {}, {}});
                pending_.emplace(blockId, PendingEntry{slot, context.ttl, true});
                batch.slots[i] = slot;
                continue;
            }
            batch.newcomerIndices.push_back(i);
        }
    }

    /**
     * @brief Phase 2 of Reserve(): reserves a slot for one new block --
     *        expired blocks are swept before the eviction discipline is
     *        consulted once the pool is full, and a displaced victim is
     *        journaled for rollback.
     */
    Expected<std::size_t> PlaceNewcomer(const BlockId& blockId, const RequestContext& context,
                                        ReserveBatch& batch, TimePoint now)
    {
        if (committed_.size() + reservedSlots_ == capacity_) { SweepExpired(now); }
        auto acquired = policy_->AcquireSlot();
        if (!acquired) { return acquired.Error(); }
        const std::size_t slot = acquired->slot;
        if (acquired->evicted) {
            const BlockId victim = occupants_[slot];
            batch.journal.push_back(
                Reservation{blockId, slot, true, false, victim, committed_.at(victim).deadline});
            committed_.erase(victim);
        } else {
            batch.journal.push_back(Reservation{blockId, slot, false, false, {}, {}});
        }
        pending_.emplace(blockId, PendingEntry{slot, context.ttl, false});
        reservedSlots_++;
        return slot;
    }

    void ReturnSharedRefs(const std::vector<SharedRef>& refs)
    {
        for (auto it = refs.rbegin(); it != refs.rend(); ++it) {
            auto pending = pending_.find(it->block);
            if (pending == pending_.end()) { continue; }  // defensive: cannot happen
            pending->second.refs--;
            pending->second.ttl = it->ttl;
        }
    }

    void Rollback(const std::vector<Reservation>& journal)
    {
        for (auto it = journal.rbegin(); it != journal.rend(); ++it) {
            pending_.erase(it->block);
            if (it->readmitted) {
                policy_->Unpin(it->slot);
                continue;
            }
            reservedSlots_--;
            if (it->displaced) {
                // The failed batch never started writing, so the slot still
                // holds the victim's admitted data: conclude the reservation
                // in favor of the previous occupant, in place. The victim's
                // expiry-index entry (if any) stays valid: its deadline is
                // restored unchanged.
                policy_->CommitSlot(it->slot);
                committed_[it->victim] = CommittedEntry{it->slot, it->victimDeadline};
                occupants_[it->slot] = it->victim;
                continue;
            }
            policy_->AbortSlot(it->slot);
        }
    }

    std::size_t capacity_{0};
    std::unique_ptr<ReplacementPolicy> policy_;
    std::unordered_map<BlockId, CommittedEntry> committed_;
    std::unordered_map<BlockId, PendingEntry> pending_;
    std::vector<BlockId> occupants_;
    /** Slots held by new, not-yet-concluded reservations. */
    std::size_t reservedSlots_{0};
    /**
     * Min-heap of deadline facts for expiry sweeps; superseded or departed
     * entries are validated away on pop, and the heap is compacted once
     * stale entries outnumber the live ones.
     */
    std::vector<DeadlineEntry> expiryHeap_;
    /** Expired-but-pinned entries parked off the heap; requeued on each sweep. */
    std::vector<DeadlineEntry> deferredExpiry_;
};

}  // namespace UC::Store
