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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>
#include "status/status.h"

namespace UC::Store {

/**
 * @brief Result of AcquireSlot(): the fresh reservation plus whether
 *        it was taken by evicting a previous occupant.
 */
struct AcquiredSlot {
    /** Reserved slot handed out; its data is not yet admitted. */
    std::size_t slot;
    /**
     * true when slot was taken by evicting its previous occupant (the
     * pool was exhausted) rather than from the free pool. Callers
     * must invalidate their block-to-slot mapping for slot's previous
     * occupant (and finish any writeback of it) before admitting data
     * into slot.
     */
    bool evicted;
};

/**
 * @brief Slot-level base class of every replacement policy: owns the
 *        fixed-capacity slot pool the eviction engines are built on.
 *
 * The class is deliberately free of business semantics: it deals in slot
 * indices only -- block-to-slot mapping, admission visibility and TTL
 * bookkeeping live in the consuming Store. Derived classes contribute
 * nothing but the eviction discipline: they maintain an order over the
 * committed slots in the On* hooks and rank victims in
 * SelectVictimSlot().
 *
 * Slot lifecycle (three states):
 *   FREE --AcquireSlot()--> RESERVED --CommitSlot()--> COMMITTED
 *   RESERVED --AbortSlot()--> FREE (the slot never joined the
 *                                          eviction discipline)
 *   COMMITTED --ReleaseSlot()--> FREE
 *   COMMITTED --eviction via AcquireSlot()--> RESERVED: the slot is
 *   handed directly to the new admission -- it never re-enters the
 *   free pool, and the displaced occupant is reported via
 *   AcquiredSlot::evicted.
 *
 * Invariants:
 *   - Every slot handed out by AcquireSlot() must be concluded by
 *     exactly one CommitSlot() or AbortSlot().
 *   - A reserved slot is immune to eviction and rejects Touch(),
 *     Pin() and ReleaseSlot() (conclude the reservation instead).
 *   - A committed slot joins the eviction discipline via
 *     OnSlotAdmitted() and may be selected as a victim unless pinned;
 *     pinned slots are never selected as victims.
 *   - Pin()/Unpin() protect committed slots only (read ends); a
 *     reservation is guarded by the reserved state itself, never by
 *     pins.
 *
 * @note Not thread-safe: callers must serialize access.
 */
class ReplacementPolicy {
public:
    virtual ~ReplacementPolicy() = default;
    /**
     * @brief Sets up the engine: fixes the slot pool size.
     *
     * Re-initializes the pool (all slots free, no reservations, no
     * pins) and forwards to OnSetup() for policy-specific setup.
     *
     * @param capacity Slot count of the fixed pool; must be > 0.
     *
     * @return Status::Ok() on success; InvalidParam on capacity == 0;
     *         otherwise the result of OnSetup() (on error the engine
     *         must be considered unusable until the next Setup).
     *
     * @note May be called again to reset the engine.
     */
    Status Setup(std::size_t capacity)
    {
        if (capacity == 0) { return Status::InvalidParam("capacity must be > 0"); }
        capacity_ = capacity;
        freeHead_ = 0;
        nextFree_.resize(capacity);
        for (std::size_t slot = 0; slot < capacity; slot++) { nextFree_[slot] = slot + 1; }
        nextFree_[capacity - 1] = capacity - 1;
        pinCounts_.assign(capacity, 0);
        reserved_.assign(capacity, 0);
        return OnSetup(capacity);
    }
    /**
     * @brief Reserves a slot for a new occupant: prefers free slots
     *        and, when the pool is exhausted, reuses the slot of a
     *        victim selected by SelectVictimSlot().
     *
     * The returned slot starts reserved: it is not yet part of the
     * eviction discipline and must be concluded by CommitSlot() (data
     * admitted) or AbortSlot() (reservation cancelled).
     *
     * @return Expected result on success: result->slot is the
     *         reserved slot; result->evicted is true when the slot
     *         was taken by evicting its previous occupant (the pool
     *         was exhausted) -- callers must invalidate their mapping
     *         for that occupant before reusing the slot. Error status
     *         otherwise: InvalidParam before a successful Setup;
     *         NoSpace when the pool is exhausted and no eligible
     *         victim exists.
     */
    Expected<AcquiredSlot> AcquireSlot()
    {
        if (capacity_ == 0) { return Status::InvalidParam("policy is not set up"); }
        if (freeHead_ != kNone) {
            const std::size_t slot = freeHead_;
            freeHead_ = (nextFree_[slot] == slot) ? kNone : nextFree_[slot];
            nextFree_[slot] = kNone;
            reserved_[slot] = 1;
            return AcquiredSlot{slot, false};
        }
        const auto victim = SelectVictimSlot();
        if (!victim || *victim >= capacity_ || nextFree_[*victim] != kNone ||
            reserved_[*victim] != 0 || pinCounts_[*victim] != 0) {
            return Status::NoSpace("slot pool exhausted and no eligible victim");
        }
        OnSlotReleased(*victim);
        reserved_[*victim] = 1;
        return AcquiredSlot{*victim, true};
    }
    /**
     * @brief Promotes a reserved slot to committed: its data is in
     *        place and the slot joins the eviction discipline.
     *
     * @param slot Slot to commit.
     *
     * @return Status::Ok() on success; InvalidParam when slot is out
     *         of range or not reserved (committed and free slots
     *         cannot commit).
     */
    Status CommitSlot(std::size_t slot)
    {
        if (!IsSlotReserved(slot)) { return Status::InvalidParam("slot {} is not reserved", slot); }
        reserved_[slot] = 0;
        OnSlotAdmitted(slot);
        return Status::Ok();
    }
    /**
     * @brief Cancels a reservation and returns the slot to the free
     *        pool.
     *
     * The slot never joined the eviction discipline, so no policy
     * hook fires.
     *
     * @param slot Slot whose reservation should be cancelled.
     *
     * @return Status::Ok() on success; InvalidParam when slot is out
     *         of range or not reserved.
     */
    Status AbortSlot(std::size_t slot)
    {
        if (!IsSlotReserved(slot)) { return Status::InvalidParam("slot {} is not reserved", slot); }
        nextFree_[slot] = (freeHead_ == kNone) ? slot : freeHead_;
        freeHead_ = slot;
        reserved_[slot] = 0;
        return Status::Ok();
    }
    /**
     * @brief Returns a committed slot to the pool.
     *
     * @param slot Slot to release.
     *
     * @return Status::Ok() on success; InvalidParam when slot is out
     *         of range, not in use, reserved (conclude the
     *         reservation via AbortSlot() instead), or still pinned
     *         (Unpin it first).
     */
    Status ReleaseSlot(std::size_t slot)
    {
        if (IsSlotReserved(slot)) {
            return Status::InvalidParam("slot {} is reserved (AbortSlot it, do not ReleaseSlot it)",
                                        slot);
        }
        if (!IsSlotCommitted(slot)) { return Status::InvalidParam("slot {} is not in use", slot); }
        if (pinCounts_[slot] != 0) { return Status::InvalidParam("slot {} is pinned", slot); }
        nextFree_[slot] = (freeHead_ == kNone) ? slot : freeHead_;
        freeHead_ = slot;
        OnSlotReleased(slot);
        return Status::Ok();
    }
    /**
     * @brief Refreshes the hotness of a committed slot.
     *
     * @param slot Slot whose hotness should be refreshed.
     *
     * @return Status::Ok() on success; InvalidParam when slot is out
     *         of range, not in use, or still reserved (its data is
     *         not admitted yet).
     */
    Status Touch(std::size_t slot)
    {
        if (!IsSlotCommitted(slot)) {
            return Status::InvalidParam("slot {} is not committed", slot);
        }
        OnSlotTouched(slot);
        return Status::Ok();
    }
    /**
     * @brief Pins a committed slot so that it is never selected as a
     *        victim.
     *
     * @param slot Slot to pin.
     *
     * @return Status::Ok() on success; InvalidParam when slot is out
     *         of range, not in use, or still reserved.
     */
    Status Pin(std::size_t slot)
    {
        if (!IsSlotCommitted(slot)) {
            return Status::InvalidParam("slot {} is not committed", slot);
        }
        pinCounts_[slot]++;
        return Status::Ok();
    }
    /**
     * @brief Releases a pin previously taken via Pin.
     *
     * @param slot Slot to unpin.
     *
     * @return Status::Ok() on success; InvalidParam when slot is out
     *         of range, not in use, or not pinned.
     */
    Status Unpin(std::size_t slot)
    {
        if (!IsSlotPinned(slot)) { return Status::InvalidParam("slot {} is not pinned", slot); }
        pinCounts_[slot]--;
        return Status::Ok();
    }
    /**
     * @brief Whether slot is within range, occupied, and pinned.
     *
     * Exposed for the owning buffer, which must defer expiry-driven
     * releases of pinned slots.
     */
    bool IsSlotPinned(std::size_t slot) const { return slot < capacity_ && pinCounts_[slot] != 0; }

protected:
    /**
     * @brief Policy-specific setup, invoked by Setup() once the pool
     *        has been (re-)initialized.
     *
     * @param capacity Slot count of the fixed pool.
     *
     * @return Status::Ok() on success; any error aborts Setup().
     */
    virtual Status OnSetup(std::size_t /*capacity*/) { return Status::Ok(); }
    /**
     * @brief A reserved slot was committed: its data is in place and
     *        the slot joins the eviction discipline. Fired by
     *        CommitSlot(), not by AcquireSlot().
     */
    virtual void OnSlotAdmitted(std::size_t /*slot*/) {}
    /** @brief The hotness of a committed slot was refreshed via Touch(). */
    virtual void OnSlotTouched(std::size_t /*slot*/) {}
    /**
     * @brief A committed slot is leaving the discipline: released
     *        via ReleaseSlot() or displaced as an eviction victim.
     */
    virtual void OnSlotReleased(std::size_t /*slot*/) {}
    /**
     * @brief Ranks the next slot to evict when the pool is exhausted.
     *
     * @return A committed, unpinned slot per the policy's discipline,
     *         or nullopt when no slot is eligible. The base validates
     *         the pick and treats an invalid one (free, reserved, or
     *         pinned) as "no victim".
     */
    virtual std::optional<std::size_t> SelectVictimSlot() = 0;
    /** @brief Whether slot is within range and currently reserved. */
    bool IsSlotReserved(std::size_t slot) const { return slot < capacity_ && reserved_[slot] != 0; }
    /** @brief Whether slot is within range, occupied, and committed. */
    bool IsSlotCommitted(std::size_t slot) const
    {
        return slot < capacity_ && nextFree_[slot] == kNone && reserved_[slot] == 0;
    }

private:
    /** @brief Marks an occupied slot (in nextFree_) and an exhausted pool (freeHead_). */
    static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    std::size_t capacity_{0};
    /** @brief Head of the intrusive free list; kNone when the pool has no free slot. */
    std::size_t freeHead_{kNone};
    /**
     * @brief Dual-purpose state per slot: kNone when occupied (reserved
     *        or committed), a self-link when free at the tail of the
     *        free list, otherwise the next free slot.
     */
    std::vector<std::size_t> nextFree_;
    /**
     * @brief Per-slot reservation flag: 1 between AcquireSlot() and the
     *        concluding CommitSlot()/AbortSlot().
     */
    std::vector<std::uint8_t> reserved_;
    std::vector<std::size_t> pinCounts_;
};

}  // namespace UC::Store
