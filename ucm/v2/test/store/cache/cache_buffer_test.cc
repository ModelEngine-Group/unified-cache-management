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
#include "cache/cache_buffer.h"
#include <chrono>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <thread>
#include <vector>
#include "cache/policy/replacement_policy_registry.h"
#include "status/status.h"
#include "token_addressed.h"
#include "type/block.h"

namespace UC::Store {
namespace {

using namespace std::chrono_literals;

BlockId MakeBlock(std::size_t id)
{
    BlockId blockId;
    blockId.bytes[0] = static_cast<std::byte>(id);
    return blockId;
}

std::unique_ptr<ReplacementPolicy> MakePolicy(const std::string& name = "fifo")
{
    auto created = ReplacementPolicyRegistry::Instance().Create(name);
    return created ? std::unique_ptr<ReplacementPolicy>(*created) : nullptr;
}

RequestContext Ctx(std::optional<std::chrono::milliseconds> ttl)
{
    RequestContext context;
    context.ttl = ttl;
    return context;
}

class CacheBufferTest : public ::testing::Test {
protected:
    CacheBuffer buffer_;
};

TEST_F(CacheBufferTest, SetupRejectsInvalidArguments)
{
    EXPECT_FALSE(buffer_.Setup(0, MakePolicy()).Success());
    EXPECT_FALSE(buffer_.Setup(1, nullptr).Success());
    EXPECT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
}

TEST_F(CacheBufferTest, OperationsBeforeSetupAreInert)
{
    const std::vector<BlockId> blocks{MakeBlock(1)};
    EXPECT_EQ(buffer_.Lookup(blocks), (std::vector<std::uint8_t>{0}));
    auto reserved = buffer_.Reserve(blocks, RequestContext{});
    ASSERT_FALSE(reserved);  // the fast-fail doubles as the unset guard
    EXPECT_EQ(reserved.Error(), Status::NoSpace());
    auto pinned = buffer_.Pin(blocks);
    ASSERT_FALSE(pinned);
    EXPECT_EQ(pinned.Error(), Status::NotFound());
    buffer_.Commit(blocks, true);
    buffer_.Unpin(blocks);
    buffer_.Touch(blocks, RequestContext{});
}

TEST_F(CacheBufferTest, ReservationsAreInvisibleUntilCommit)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const std::vector<BlockId> blocks{MakeBlock(1), MakeBlock(2)};
    auto reserved = buffer_.Reserve(blocks, RequestContext{});
    ASSERT_TRUE(reserved);
    EXPECT_EQ(buffer_.Lookup(blocks), (std::vector<std::uint8_t>{0, 0}));
    auto pinned = buffer_.Pin(blocks);
    ASSERT_FALSE(pinned);
    EXPECT_EQ(pinned.Error(), Status::NotFound());
    buffer_.Commit(blocks, true);
    EXPECT_EQ(buffer_.Lookup(blocks), (std::vector<std::uint8_t>{1, 1}));
}

TEST_F(CacheBufferTest, CommitFalseReleasesReservation)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId first = MakeBlock(1);
    const BlockId second = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({first}, RequestContext{}));
    buffer_.Commit({first}, false);
    ASSERT_TRUE(buffer_.Reserve({second}, RequestContext{}));
    buffer_.Commit({second}, true);
    EXPECT_EQ(buffer_.Lookup({first, second}), (std::vector<std::uint8_t>{0, 1}));
}

TEST_F(CacheBufferTest, ReserveAndPinResolveConsistentSlots)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const std::vector<BlockId> blocks{MakeBlock(1), MakeBlock(2)};
    auto reserved = buffer_.Reserve(blocks, RequestContext{});
    ASSERT_TRUE(reserved);
    EXPECT_EQ((*reserved)[0], 0U);
    EXPECT_EQ((*reserved)[1], 1U);
    buffer_.Commit(blocks, true);
    auto pinned = buffer_.Pin(blocks);
    ASSERT_TRUE(pinned);
    EXPECT_EQ((*pinned)[0], 0U);
    EXPECT_EQ((*pinned)[1], 1U);
    buffer_.Unpin(blocks);
}

TEST_F(CacheBufferTest, EvictionRemovesDisplacedBlock)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId first = MakeBlock(1);
    const BlockId second = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({first}, RequestContext{}));
    buffer_.Commit({first}, true);
    auto reserved = buffer_.Reserve({second}, RequestContext{});
    ASSERT_TRUE(reserved);
    EXPECT_EQ((*reserved)[0], 0U);
    buffer_.Commit({second}, true);
    EXPECT_EQ(buffer_.Lookup({first, second}), (std::vector<std::uint8_t>{0, 1}));
}

TEST_F(CacheBufferTest, ReadmissionKeepsSlotWithoutExtendingDeadline)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const BlockId readmitted = MakeBlock(1);
    const BlockId permanent = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({readmitted}, Ctx(30ms)));
    ASSERT_TRUE(buffer_.Reserve({permanent}, RequestContext{}));
    buffer_.Commit({readmitted, permanent}, true);
    auto reserved = buffer_.Reserve({readmitted}, RequestContext{});
    ASSERT_TRUE(reserved);
    EXPECT_EQ((*reserved)[0], 0U);  // re-admission keeps the slot
    buffer_.Commit({readmitted}, true);
    std::this_thread::sleep_for(60ms);
    // The live block's deadline is not extended by re-admission; only Touch()
    // extends retention.
    EXPECT_EQ(buffer_.Lookup({readmitted, permanent}), (std::vector<std::uint8_t>{0, 1}));
}

TEST_F(CacheBufferTest, ReadmissionPinsSlotUntilConcluded)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId resident = MakeBlock(1);
    const BlockId newcomer = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({resident}, RequestContext{}));
    buffer_.Commit({resident}, true);
    ASSERT_TRUE(buffer_.Reserve({resident}, RequestContext{}));
    auto blocked = buffer_.Reserve({newcomer}, RequestContext{});
    ASSERT_FALSE(blocked);  // the in-flight rewrite pins the only slot
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    buffer_.Commit({resident}, false);
    auto evicting = buffer_.Reserve({newcomer}, RequestContext{});
    ASSERT_TRUE(evicting);
    EXPECT_EQ((*evicting)[0], 0U);
    buffer_.Commit({newcomer}, true);
    EXPECT_EQ(buffer_.Lookup({resident, newcomer}), (std::vector<std::uint8_t>{0, 1}));
}

TEST_F(CacheBufferTest, FailedBatchRollsBackCompletely)
{
    ASSERT_TRUE(buffer_.Setup(3, MakePolicy()).Success());
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    const BlockId c = MakeBlock(3);
    ASSERT_TRUE(buffer_.Reserve({a, b, c}, RequestContext{}));
    buffer_.Commit({a, b, c}, true);
    auto pinned = buffer_.Pin({c});  // the only eligible victim is protected
    ASSERT_TRUE(pinned);
    auto blocked = buffer_.Reserve({MakeBlock(4), MakeBlock(5), MakeBlock(6)}, RequestContext{});
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    // The two blocks evicted before the failure are back at their slots.
    EXPECT_EQ(buffer_.Lookup({a, b, c}), (std::vector<std::uint8_t>{1, 1, 1}));
    EXPECT_EQ(buffer_.Lookup({MakeBlock(4), MakeBlock(5), MakeBlock(6)}),
              (std::vector<std::uint8_t>{0, 0, 0}));
    auto restored = buffer_.Pin({a, b});
    ASSERT_TRUE(restored);
    EXPECT_EQ((*restored)[0], 0U);
    EXPECT_EQ((*restored)[1], 1U);
    buffer_.Unpin({a, b, c});
    // Restored victims rejoin the discipline behind the pinned block, so the
    // next eviction takes c -- the documented order perturbation.
    auto evicting = buffer_.Reserve({MakeBlock(7)}, RequestContext{});
    ASSERT_TRUE(evicting);
    EXPECT_EQ((*evicting)[0], 2U);
    buffer_.Commit({MakeBlock(7)}, true);
    EXPECT_EQ(buffer_.Lookup({a, b, c}), (std::vector<std::uint8_t>{1, 1, 0}));
}

TEST_F(CacheBufferTest, FastFailLeavesEvictionOrderUntouched)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({a, b}, RequestContext{}));
    buffer_.Commit({a, b}, true);
    auto blocked = buffer_.Reserve({MakeBlock(4), MakeBlock(5), MakeBlock(6)}, RequestContext{});
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    EXPECT_EQ(buffer_.Lookup({a, b}), (std::vector<std::uint8_t>{1, 1}));
    auto evicting = buffer_.Reserve({MakeBlock(7)}, RequestContext{});
    ASSERT_TRUE(evicting);
    EXPECT_EQ((*evicting)[0], 0U);  // FIFO still evicts the oldest block
    buffer_.Commit({MakeBlock(7)}, true);
    EXPECT_EQ(buffer_.Lookup({a, b}), (std::vector<std::uint8_t>{0, 1}));
}

TEST_F(CacheBufferTest, PinIsAtomicOnMissingBlocks)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId resident = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({resident}, RequestContext{}));
    buffer_.Commit({resident}, true);
    auto pinned = buffer_.Pin({resident, MakeBlock(9)});
    ASSERT_FALSE(pinned);
    EXPECT_EQ(pinned.Error(), Status::NotFound());
    // The failed Pin left nothing pinned: the resident stays evictable.
    auto evicting = buffer_.Reserve({MakeBlock(2)}, RequestContext{});
    ASSERT_TRUE(evicting);
    buffer_.Commit({MakeBlock(2)}, true);
    EXPECT_EQ(buffer_.Lookup({resident}), (std::vector<std::uint8_t>{0}));
}

TEST_F(CacheBufferTest, ExpiryDeferredWhilePinned)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const std::vector<BlockId> blocks{MakeBlock(1), MakeBlock(2)};
    ASSERT_TRUE(buffer_.Reserve(blocks, Ctx(30ms)));
    buffer_.Commit(blocks, true);
    auto pinned = buffer_.Pin(blocks);
    ASSERT_TRUE(pinned);
    std::this_thread::sleep_for(60ms);
    EXPECT_EQ(buffer_.Lookup(blocks), (std::vector<std::uint8_t>{0, 0}));  // deferred misses
    auto blocked = buffer_.Reserve({MakeBlock(3)}, RequestContext{});
    ASSERT_FALSE(blocked);  // expired but pinned: neither swept nor evicted
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    buffer_.Unpin(blocks);
    auto admitted = buffer_.Reserve({MakeBlock(3)}, RequestContext{});
    ASSERT_TRUE(admitted);  // the sweep releases both expired blocks
    buffer_.Commit({MakeBlock(3)}, true);
    EXPECT_EQ(buffer_.Lookup({blocks[0], blocks[1], MakeBlock(3)}),
              (std::vector<std::uint8_t>{0, 0, 1}));
}

TEST_F(CacheBufferTest, PressureReleasesExpiredBeforeDiscipline)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const BlockId good = MakeBlock(1);      // oldest, never expires
    const BlockId expiring = MakeBlock(2);  // newer, but expires
    ASSERT_TRUE(buffer_.Reserve({good}, RequestContext{}));
    ASSERT_TRUE(buffer_.Reserve({expiring}, Ctx(30ms)));
    buffer_.Commit({good, expiring}, true);
    std::this_thread::sleep_for(60ms);
    auto admitted = buffer_.Reserve({MakeBlock(3)}, RequestContext{});
    ASSERT_TRUE(admitted);
    EXPECT_EQ((*admitted)[0], 1U);  // took the expired block's slot
    buffer_.Commit({MakeBlock(3)}, true);
    auto pinned = buffer_.Pin({good, MakeBlock(3)});
    ASSERT_TRUE(pinned);
    EXPECT_EQ((*pinned)[0], 0U);  // the discipline never saw the good block
    EXPECT_EQ((*pinned)[1], 1U);
    buffer_.Unpin({good, MakeBlock(3)});
    EXPECT_EQ(buffer_.Lookup({good, expiring, MakeBlock(3)}), (std::vector<std::uint8_t>{1, 0, 1}));
}

TEST_F(CacheBufferTest, SetupResetsState)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const std::vector<BlockId> blocks{MakeBlock(1), MakeBlock(2)};
    ASSERT_TRUE(buffer_.Reserve(blocks, RequestContext{}));
    buffer_.Commit(blocks, true);
    auto pinned = buffer_.Pin(blocks);
    ASSERT_TRUE(pinned);
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    EXPECT_EQ(buffer_.Lookup(blocks), (std::vector<std::uint8_t>{0, 0}));
    auto reserved = buffer_.Reserve(blocks, RequestContext{});
    ASSERT_TRUE(reserved);  // no stale reservations or pins
    EXPECT_EQ((*reserved)[0], 0U);
    EXPECT_EQ((*reserved)[1], 1U);
}

TEST_F(CacheBufferTest, TouchExtendsDeadlineWithTheLargerTtl)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const BlockId shortLived = MakeBlock(1);
    const BlockId longLived = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({shortLived}, Ctx(40ms)));
    ASSERT_TRUE(buffer_.Reserve({longLived}, Ctx(200ms)));
    buffer_.Commit({shortLived, longLived}, true);
    buffer_.Touch({shortLived, longLived}, Ctx(120ms));
    std::this_thread::sleep_for(80ms);
    // shortLived was extended past its original 40ms deadline.
    EXPECT_EQ(buffer_.Lookup({shortLived, longLived}), (std::vector<std::uint8_t>{1, 1}));
    std::this_thread::sleep_for(70ms);
    // At ~150ms: shortLived hit its extended deadline; longLived kept its
    // original 200ms instead of being shortened to 120ms.
    EXPECT_EQ(buffer_.Lookup({shortLived, longLived}), (std::vector<std::uint8_t>{0, 1}));
}

TEST_F(CacheBufferTest, TouchWithoutTtlFactKeepsDeadline)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId block = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({block}, Ctx(40ms)));
    buffer_.Commit({block}, true);
    buffer_.Touch({block}, RequestContext{});
    std::this_thread::sleep_for(60ms);
    EXPECT_EQ(buffer_.Lookup({block}), (std::vector<std::uint8_t>{0}));
}

TEST_F(CacheBufferTest, TouchKeepsNeverExpiringDeadline)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId block = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({block}, RequestContext{}));
    buffer_.Commit({block}, true);
    buffer_.Touch({block}, Ctx(30ms));
    std::this_thread::sleep_for(60ms);
    EXPECT_EQ(buffer_.Lookup({block}), (std::vector<std::uint8_t>{1}));
}

TEST_F(CacheBufferTest, ExpiredBlockIsRevivedByReadmission)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const BlockId revived = MakeBlock(1);
    const BlockId expired = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({revived, expired}, Ctx(30ms)));
    buffer_.Commit({revived, expired}, true);
    std::this_thread::sleep_for(60ms);
    auto reserved = buffer_.Reserve({revived}, Ctx(200ms));
    ASSERT_TRUE(reserved);
    EXPECT_EQ((*reserved)[0], 0U);  // revived in place, no data movement
    buffer_.Commit({revived}, true);
    EXPECT_EQ(buffer_.Lookup({revived, expired}), (std::vector<std::uint8_t>{1, 0}));
}

TEST_F(CacheBufferTest, MidFlightExpiryIsRevivedAtCommit)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId x = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(50ms)));
    buffer_.Commit({x}, true);
    std::this_thread::sleep_for(20ms);              // x is still live here
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(300ms)));  // re-admission opens
    std::this_thread::sleep_for(45ms);              // the 50ms deadline passes mid-flight
    buffer_.Commit({x}, true);                      // the conclusion revives with the fresh fact
    EXPECT_EQ(buffer_.Lookup({x}), (std::vector<std::uint8_t>{1}));
    std::this_thread::sleep_for(40ms);  // well past the old deadline
    EXPECT_EQ(buffer_.Lookup({x}), (std::vector<std::uint8_t>{1}));
}

TEST_F(CacheBufferTest, CommitFalseOnReadmissionDoesNotRevive)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId x = MakeBlock(1);
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(30ms)));
    buffer_.Commit({x}, true);
    std::this_thread::sleep_for(60ms);  // x is expired
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(300ms)));
    buffer_.Commit({x}, false);  // the failed rewrite must not revive
    EXPECT_EQ(buffer_.Lookup({x}), (std::vector<std::uint8_t>{0}));
    // A live block's deadline survives a failed re-admission unchanged.
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(200ms)));
    buffer_.Commit({x}, true);
    std::this_thread::sleep_for(30ms);
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(30ms)));  // would shorten
    buffer_.Commit({x}, false);
    std::this_thread::sleep_for(40ms);  // ~130ms into the 200ms deadline
    EXPECT_EQ(buffer_.Lookup({x}), (std::vector<std::uint8_t>{1}));
}

TEST_F(CacheBufferTest, FailedBatchRollsBackUnderClockPolicy)
{
    ASSERT_TRUE(buffer_.Setup(3, MakePolicy("clock")).Success());
    const BlockId a = MakeBlock(1);
    const BlockId b = MakeBlock(2);
    const BlockId c = MakeBlock(3);
    ASSERT_TRUE(buffer_.Reserve({a, b, c}, RequestContext{}));
    buffer_.Commit({a, b, c}, true);
    auto pinned = buffer_.Pin({c});  // the only eligible victim is protected
    ASSERT_TRUE(pinned);
    auto blocked = buffer_.Reserve({MakeBlock(4), MakeBlock(5), MakeBlock(6)}, RequestContext{});
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    // The blocks evicted before the failure are back at their slots.
    EXPECT_EQ(buffer_.Lookup({a, b, c}), (std::vector<std::uint8_t>{1, 1, 1}));
    auto restored = buffer_.Pin({a, b});
    ASSERT_TRUE(restored);
    EXPECT_EQ((*restored)[0], 0U);
    EXPECT_EQ((*restored)[1], 1U);
    buffer_.Unpin({a, b, c});
    // The clock hand rests on c behind the restored victims, so the next
    // eviction takes it.
    auto evicting = buffer_.Reserve({MakeBlock(7)}, RequestContext{});
    ASSERT_TRUE(evicting);
    EXPECT_EQ((*evicting)[0], 2U);
    buffer_.Commit({MakeBlock(7)}, true);
    EXPECT_EQ(buffer_.Lookup({a, b, c}), (std::vector<std::uint8_t>{1, 1, 0}));
}

TEST_F(CacheBufferTest, BatchOrderDoesNotEvictCommittedMembers)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const BlockId a = MakeBlock(1);
    const BlockId c = MakeBlock(2);
    const BlockId b = MakeBlock(3);
    ASSERT_TRUE(buffer_.Reserve({a, c}, RequestContext{}));
    buffer_.Commit({a, c}, true);
    // The newcomer is listed before the re-admitted committed block.
    auto reserved = buffer_.Reserve({b, a}, RequestContext{});
    ASSERT_TRUE(reserved);
    EXPECT_EQ((*reserved)[0], 1U);  // b evicts c, not the batch member a
    EXPECT_EQ((*reserved)[1], 0U);  // a keeps its slot
    buffer_.Commit({b, a}, false);  // the failed admission must not lose a
    EXPECT_EQ(buffer_.Lookup({a, b, c}), (std::vector<std::uint8_t>{1, 0, 0}));
    auto pinned = buffer_.Pin({a});
    ASSERT_TRUE(pinned);
    EXPECT_EQ((*pinned)[0], 0U);
    buffer_.Unpin({a});
    // The mirrored order resolves the same per-block slots.
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    ASSERT_TRUE(buffer_.Reserve({a, c}, RequestContext{}));
    buffer_.Commit({a, c}, true);
    auto mirrored = buffer_.Reserve({a, b}, RequestContext{});
    ASSERT_TRUE(mirrored);
    EXPECT_EQ((*mirrored)[0], 0U);
    EXPECT_EQ((*mirrored)[1], 1U);
    buffer_.Commit({a, b}, false);
    EXPECT_EQ(buffer_.Lookup({a, b, c}), (std::vector<std::uint8_t>{1, 0, 0}));
}

TEST_F(CacheBufferTest, DuplicateBlocksInBatchShareOneSlot)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId b = MakeBlock(1);
    auto reserved = buffer_.Reserve({b, b}, RequestContext{});  // one slot suffices
    ASSERT_TRUE(reserved);
    EXPECT_EQ((*reserved)[0], 0U);
    EXPECT_EQ((*reserved)[1], 0U);
    buffer_.Commit({b, b}, true);
    EXPECT_EQ(buffer_.Lookup({b}), (std::vector<std::uint8_t>{1}));
}

TEST_F(CacheBufferTest, SharedReservationSurvivesOneCallersFailure)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId x = MakeBlock(1);
    const BlockId y = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({x}, RequestContext{}));
    ASSERT_TRUE(buffer_.Reserve({x}, RequestContext{}));  // joins: refs = 2
    buffer_.Commit({x}, false);  // one conclusion: the other's Dump still flies
    auto blocked = buffer_.Reserve({y}, RequestContext{});
    ASSERT_FALSE(blocked);  // the slot was not released under the in-flight Dump
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    buffer_.Commit({x}, true);  // the last conclusion succeeds
    EXPECT_EQ(buffer_.Lookup({x}), (std::vector<std::uint8_t>{1}));
    auto evicting = buffer_.Reserve({y}, RequestContext{});  // now evictable
    ASSERT_TRUE(evicting);
    EXPECT_EQ((*evicting)[0], 0U);
    buffer_.Commit({y}, true);
}

TEST_F(CacheBufferTest, SharedReservationConcludesOnAnySuccess)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId x = MakeBlock(1);
    const BlockId y = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(30ms)));
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(300ms)));  // last writer wins
    buffer_.Commit({x}, true);
    auto blocked = buffer_.Reserve({y}, RequestContext{});
    ASSERT_FALSE(blocked);  // one reference is still outstanding
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    buffer_.Commit({x}, false);  // the last conclusion fails; any success wins
    EXPECT_EQ(buffer_.Lookup({x}), (std::vector<std::uint8_t>{1}));
}

TEST_F(CacheBufferTest, DuplicatePendingBlocksGainOneRefPerOccurrence)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId x = MakeBlock(1);
    const BlockId y = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({x}, RequestContext{}));     // the original reference
    ASSERT_TRUE(buffer_.Reserve({x, x}, RequestContext{}));  // duplicates join in
    buffer_.Commit({x, x}, true);  // mirrors the occurrences: one reference left
    auto blocked = buffer_.Reserve({y}, RequestContext{});
    ASSERT_FALSE(blocked);  // the original reference still holds the slot
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    buffer_.Commit({x}, true);  // the original concludes the reservation
    EXPECT_EQ(buffer_.Lookup({x}), (std::vector<std::uint8_t>{1}));
}

TEST_F(CacheBufferTest, FailedReserveReturnsItsSharedRef)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId x = MakeBlock(1);
    const BlockId y = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({x}, RequestContext{}));
    buffer_.Commit({x}, true);
    ASSERT_TRUE(buffer_.Reserve({x}, RequestContext{}));  // re-admission in flight
    // The joining batch fails wholesale; its shared ref must be returned.
    auto blocked = buffer_.Reserve({x, y}, RequestContext{});
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    buffer_.Commit({x}, false);  // the original admission still concludes
    auto evicting = buffer_.Reserve({y}, RequestContext{});
    ASSERT_TRUE(evicting);  // no leaked pin or reservation
    EXPECT_EQ((*evicting)[0], 0U);
    buffer_.Commit({y}, true);
}

TEST_F(CacheBufferTest, SweepHonorsRefreshedDeadlineOverStaleIndexEntry)
{
    ASSERT_TRUE(buffer_.Setup(2, MakePolicy()).Success());
    const BlockId z = MakeBlock(1);  // oldest, never expires
    const BlockId x = MakeBlock(2);  // expires, then revived
    ASSERT_TRUE(buffer_.Reserve({z}, RequestContext{}));
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(30ms)));
    buffer_.Commit({z, x}, true);
    std::this_thread::sleep_for(60ms);              // x expires
    ASSERT_TRUE(buffer_.Reserve({x}, Ctx(300ms)));  // revival in flight
    buffer_.Commit({x}, true);                      // revived; a stale 30ms index entry lingers
    auto admitted = buffer_.Reserve({MakeBlock(3)}, RequestContext{});  // pressure
    ASSERT_TRUE(admitted);
    EXPECT_EQ((*admitted)[0], 0U);  // the sweep must not release x: FIFO evicts z
    buffer_.Commit({MakeBlock(3)}, true);
    EXPECT_EQ(buffer_.Lookup({z, x}), (std::vector<std::uint8_t>{0, 1}));
}

TEST_F(CacheBufferTest, UnpinWithoutMatchingPinIsNoOp)
{
    ASSERT_TRUE(buffer_.Setup(1, MakePolicy()).Success());
    const BlockId x = MakeBlock(1);
    const BlockId y = MakeBlock(2);
    ASSERT_TRUE(buffer_.Reserve({x}, RequestContext{}));
    buffer_.Commit({x}, true);
    buffer_.Unpin({x});                                   // never pinned via Pin: no-op
    ASSERT_TRUE(buffer_.Reserve({x}, RequestContext{}));  // re-admission in flight
    buffer_.Unpin({x});  // out-of-contract: must not eat the reservation pin
    auto blocked = buffer_.Reserve({y}, RequestContext{});
    ASSERT_FALSE(blocked);  // the in-flight rewrite still pins the slot
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    buffer_.Commit({x}, true);
    auto pinned = buffer_.Pin({x});  // balanced pins still work
    ASSERT_TRUE(pinned);
    buffer_.Unpin({x});
    buffer_.Unpin({x});  // one more than pinned: no-op
    auto evicting = buffer_.Reserve({y}, RequestContext{});
    ASSERT_TRUE(evicting);  // fully unpinned: evictable again
    buffer_.Commit({y}, true);
}

}  // namespace
}  // namespace UC::Store
