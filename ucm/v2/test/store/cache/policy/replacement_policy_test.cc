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
#include "cache/policy/replacement_policy.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include "cache/policy/replacement_policy_registry.h"
#include "status/status.h"

namespace UC::Store {
namespace {

/** @brief Never elects a victim: isolates base-class free-pool behavior. */
class NullPolicy : public ReplacementPolicy {
protected:
    std::optional<std::size_t> SelectVictimSlot() override { return std::nullopt; }
};

/** @brief Elects a scripted victim: exercises the base-class validation of policy picks. */
class StubVictimPolicy : public ReplacementPolicy {
public:
    std::optional<std::size_t> pick;

protected:
    std::optional<std::size_t> SelectVictimSlot() override { return pick; }
};

std::unique_ptr<ReplacementPolicy> CreatePolicy(const std::string& name)
{
    auto created = ReplacementPolicyRegistry::Instance().Create(name);
    if (!created) { return nullptr; }
    return std::unique_ptr<ReplacementPolicy>(*created);
}

class ReplacementPolicyTest : public ::testing::Test {
protected:
    NullPolicy policy_;
};

TEST_F(ReplacementPolicyTest, SetupRejectsZeroCapacity)
{
    EXPECT_FALSE(policy_.Setup(0).Success());
}

TEST_F(ReplacementPolicyTest, AcquireBeforeSetupFails)
{
    auto result = policy_.AcquireSlot();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.Error(), Status::InvalidParam());
}

TEST_F(ReplacementPolicyTest, AcquireReservesEveryFreeSlotThenNoSpace)
{
    ASSERT_TRUE(policy_.Setup(3).Success());
    for (std::size_t expected = 0; expected < 3; expected++) {
        auto result = policy_.AcquireSlot();
        ASSERT_TRUE(result);
        EXPECT_EQ(result->slot, expected);
        EXPECT_FALSE(result->evicted);
    }
    auto exhausted = policy_.AcquireSlot();
    ASSERT_FALSE(exhausted);
    EXPECT_EQ(exhausted.Error(), Status::NoSpace());
}

TEST_F(ReplacementPolicyTest, ReservedSlotRejectsDisciplineEntryPoints)
{
    ASSERT_TRUE(policy_.Setup(1).Success());
    auto result = policy_.AcquireSlot();
    ASSERT_TRUE(result);
    const std::size_t slot = result->slot;
    EXPECT_FALSE(policy_.Touch(slot).Success());
    EXPECT_FALSE(policy_.Pin(slot).Success());
    EXPECT_FALSE(policy_.ReleaseSlot(slot).Success());
    EXPECT_FALSE(policy_.CommitSlot(slot + 1).Success());
    EXPECT_FALSE(policy_.AbortSlot(slot + 1).Success());
}

TEST_F(ReplacementPolicyTest, CommitAndAbortAreTerminal)
{
    ASSERT_TRUE(policy_.Setup(1).Success());
    const std::size_t slot = (*policy_.AcquireSlot()).slot;
    ASSERT_TRUE(policy_.CommitSlot(slot).Success());
    EXPECT_FALSE(policy_.CommitSlot(slot).Success());
    EXPECT_FALSE(policy_.AbortSlot(slot).Success());
}

TEST_F(ReplacementPolicyTest, AbortReturnsSlotToFreePool)
{
    ASSERT_TRUE(policy_.Setup(2).Success());
    auto first = policy_.AcquireSlot();
    auto second = policy_.AcquireSlot();
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(policy_.AbortSlot(first->slot).Success());
    auto reused = policy_.AcquireSlot();
    ASSERT_TRUE(reused);
    EXPECT_EQ(reused->slot, first->slot);
    EXPECT_FALSE(reused->evicted);
}

TEST_F(ReplacementPolicyTest, CommitUnlocksTouchPinRelease)
{
    ASSERT_TRUE(policy_.Setup(1).Success());
    const std::size_t slot = (*policy_.AcquireSlot()).slot;
    EXPECT_FALSE(policy_.Touch(slot).Success());
    ASSERT_TRUE(policy_.CommitSlot(slot).Success());
    EXPECT_TRUE(policy_.Touch(slot).Success());
    EXPECT_TRUE(policy_.Pin(slot).Success());
    EXPECT_FALSE(policy_.ReleaseSlot(slot).Success());
    EXPECT_TRUE(policy_.Unpin(slot).Success());
    EXPECT_FALSE(policy_.Unpin(slot).Success());
    EXPECT_TRUE(policy_.ReleaseSlot(slot).Success());
    EXPECT_FALSE(policy_.ReleaseSlot(slot).Success());
}

TEST_F(ReplacementPolicyTest, SetupResetsReservationsAndPins)
{
    ASSERT_TRUE(policy_.Setup(2).Success());
    auto first = policy_.AcquireSlot();
    auto second = policy_.AcquireSlot();
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(policy_.CommitSlot(first->slot).Success());
    ASSERT_TRUE(policy_.Pin(first->slot).Success());

    ASSERT_TRUE(policy_.Setup(2).Success());
    for (std::size_t i = 0; i < 2; i++) {
        auto result = policy_.AcquireSlot();
        ASSERT_TRUE(result);
        EXPECT_FALSE(result->evicted);
    }
    auto blocked = policy_.AcquireSlot();
    ASSERT_FALSE(blocked);
    // The pre-reset pin is gone: the re-occupied slot releases cleanly.
    ASSERT_TRUE(policy_.CommitSlot(first->slot).Success());
    EXPECT_TRUE(policy_.ReleaseSlot(first->slot).Success());
}

TEST_F(ReplacementPolicyTest, InvalidVictimPickIsTreatedAsNoVictim)
{
    StubVictimPolicy policy;
    ASSERT_TRUE(policy.Setup(2).Success());
    auto committed = policy.AcquireSlot();
    auto reserved = policy.AcquireSlot();
    ASSERT_TRUE(committed);
    ASSERT_TRUE(reserved);
    ASSERT_TRUE(policy.CommitSlot(committed->slot).Success());

    policy.pick = reserved->slot;  // a reserved slot is never a valid victim
    auto blocked = policy.AcquireSlot();
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.Error(), Status::NoSpace());

    policy.pick = 99;  // out of range
    auto outOfRange = policy.AcquireSlot();
    ASSERT_FALSE(outOfRange);
    EXPECT_EQ(outOfRange.Error(), Status::NoSpace());

    policy.pick = committed->slot;  // committed and unpinned: a valid victim
    auto evicting = policy.AcquireSlot();
    ASSERT_TRUE(evicting);
    ASSERT_TRUE(evicting->evicted);
    EXPECT_EQ(evicting->slot, committed->slot);
}

/** @brief Behavior shared by every registered policy. */
class RegisteredPolicyTest : public ::testing::TestWithParam<const char*> {};

TEST_P(RegisteredPolicyTest, ReservedSlotIsImmuneToEviction)
{
    auto policy = CreatePolicy(GetParam());
    ASSERT_NE(policy, nullptr);
    ASSERT_TRUE(policy->Setup(1).Success());
    auto first = policy->AcquireSlot();
    ASSERT_TRUE(first);
    auto blocked = policy->AcquireSlot();
    ASSERT_FALSE(blocked);
    EXPECT_EQ(blocked.Error(), Status::NoSpace());
    ASSERT_TRUE(policy->CommitSlot(first->slot).Success());
    auto evicting = policy->AcquireSlot();
    ASSERT_TRUE(evicting);
    EXPECT_EQ(evicting->slot, first->slot);
    ASSERT_TRUE(evicting->evicted);
}

TEST_P(RegisteredPolicyTest, EvictedSlotIsReservedForNewOccupant)
{
    auto policy = CreatePolicy(GetParam());
    ASSERT_NE(policy, nullptr);
    ASSERT_TRUE(policy->Setup(1).Success());
    auto first = policy->AcquireSlot();
    ASSERT_TRUE(first);
    ASSERT_TRUE(policy->CommitSlot(first->slot).Success());
    auto evicting = policy->AcquireSlot();
    ASSERT_TRUE(evicting);
    EXPECT_FALSE(policy->Touch(evicting->slot).Success());
    EXPECT_FALSE(policy->Pin(evicting->slot).Success());
    EXPECT_FALSE(policy->ReleaseSlot(evicting->slot).Success());
    ASSERT_TRUE(policy->CommitSlot(evicting->slot).Success());
}

TEST_P(RegisteredPolicyTest, PinnedSlotIsImmuneToEviction)
{
    auto policy = CreatePolicy(GetParam());
    ASSERT_NE(policy, nullptr);
    ASSERT_TRUE(policy->Setup(1).Success());
    auto first = policy->AcquireSlot();
    ASSERT_TRUE(first);
    ASSERT_TRUE(policy->CommitSlot(first->slot).Success());
    ASSERT_TRUE(policy->Pin(first->slot).Success());
    auto blocked = policy->AcquireSlot();
    ASSERT_FALSE(blocked);
    ASSERT_TRUE(policy->Unpin(first->slot).Success());
    auto evicting = policy->AcquireSlot();
    ASSERT_TRUE(evicting);
    ASSERT_TRUE(evicting->evicted);
    EXPECT_EQ(evicting->slot, first->slot);
}

TEST_P(RegisteredPolicyTest, EvictsOldestCommittedSlot)
{
    auto policy = CreatePolicy(GetParam());
    ASSERT_NE(policy, nullptr);
    ASSERT_TRUE(policy->Setup(3).Success());
    for (std::size_t slot = 0; slot < 3; slot++) {
        auto result = policy->AcquireSlot();
        ASSERT_TRUE(result);
        ASSERT_TRUE(policy->CommitSlot(slot).Success());
    }
    auto evicting = policy->AcquireSlot();
    ASSERT_TRUE(evicting);
    ASSERT_TRUE(evicting->evicted);
    EXPECT_EQ(evicting->slot, 0U);
}

TEST_P(RegisteredPolicyTest, FullPoolOfReservationsAppliesBackpressure)
{
    auto policy = CreatePolicy(GetParam());
    ASSERT_NE(policy, nullptr);
    ASSERT_TRUE(policy->Setup(3).Success());
    std::optional<std::size_t> slots[3];
    for (std::size_t i = 0; i < 3; i++) {
        auto result = policy->AcquireSlot();
        ASSERT_TRUE(result);
        slots[i] = result->slot;
    }
    auto blocked = policy->AcquireSlot();
    ASSERT_FALSE(blocked);
    ASSERT_TRUE(policy->AbortSlot(*slots[2]).Success());
    auto resumed = policy->AcquireSlot();
    ASSERT_TRUE(resumed);
    EXPECT_EQ(resumed->slot, *slots[2]);
    EXPECT_FALSE(resumed->evicted);
}

INSTANTIATE_TEST_SUITE_P(Store, RegisteredPolicyTest, ::testing::Values("lru", "fifo", "clock"));

/** @brief Discipline-order tests for policies whose victim order follows the admission sequence. */
class DisciplinaryOrderTest : public ::testing::TestWithParam<const char*> {};

TEST_P(DisciplinaryOrderTest, EvictionFollowsCommitOrderNotAcquireOrder)
{
    auto policy = CreatePolicy(GetParam());
    ASSERT_NE(policy, nullptr);
    ASSERT_TRUE(policy->Setup(2).Success());
    auto first = policy->AcquireSlot();
    auto second = policy->AcquireSlot();
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(policy->CommitSlot(second->slot).Success());
    ASSERT_TRUE(policy->CommitSlot(first->slot).Success());
    auto evicting = policy->AcquireSlot();
    ASSERT_TRUE(evicting);
    ASSERT_TRUE(evicting->evicted);
    EXPECT_EQ(evicting->slot, second->slot);
}

INSTANTIATE_TEST_SUITE_P(Store, DisciplinaryOrderTest, ::testing::Values("lru", "fifo"));

TEST(RegisteredLruTest, TouchRefreshesVictimOrder)
{
    auto policy = CreatePolicy("lru");
    ASSERT_NE(policy, nullptr);
    ASSERT_TRUE(policy->Setup(3).Success());
    for (std::size_t slot = 0; slot < 3; slot++) {
        auto result = policy->AcquireSlot();
        ASSERT_TRUE(result);
        ASSERT_TRUE(policy->CommitSlot(slot).Success());
    }
    ASSERT_TRUE(policy->Touch(0).Success());
    auto evicting = policy->AcquireSlot();
    ASSERT_TRUE(evicting);
    ASSERT_TRUE(evicting->evicted);
    EXPECT_EQ(evicting->slot, 1U);
}

TEST(RegisteredPolicyRegistryTest, UnknownNameIsNotFound)
{
    auto created = ReplacementPolicyRegistry::Instance().Create("no-such-policy");
    ASSERT_FALSE(created);
    EXPECT_EQ(created.Error(), Status::NotFound());
}

}  // namespace
}  // namespace UC::Store
