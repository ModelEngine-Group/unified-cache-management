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
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include "dram/cc/drampool/entry.h"
#include "dram/cc/drampool/pos_eviction_policy.h"
#include "dram/dram_test_common.h"

using UC::Status;
using UC::DramPool::PosEvictionPolicy;
using UC::Test::Dram::Clock;
using UC::Test::Dram::EntryStatus;
using UC::Test::Dram::KeyFromHex;
using UC::Test::Dram::MakeEntry;
using UC::Test::Dram::TimePoint;

class UCPosEvictionPolicyTest : public testing::Test {
protected:
    PosEvictionPolicy policy_;
    const TimePoint past_ = Clock::now() - std::chrono::seconds(10);
    const TimePoint future_ = Clock::now() + std::chrono::seconds(10);
};

TEST_F(UCPosEvictionPolicyTest, AddKeyReturnsOk)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
}

TEST_F(UCPosEvictionPolicyTest, AddKeyDuplicateReturnsDuplicateKey)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_EQ(policy_.AddKey(key, MakeEntry(key, 0, past_)), Status::DuplicateKey());
}

TEST_F(UCPosEvictionPolicyTest, AddKeyNullptrReturnsInvalidParam)
{
    auto key = KeyFromHex("a1");
    ASSERT_EQ(policy_.AddKey(key, nullptr), Status::InvalidParam());
}

TEST_F(UCPosEvictionPolicyTest, DeleteKeyReturnsOkAndSecondIsNotFound)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_TRUE(policy_.DeleteKey(key).Success());
    ASSERT_EQ(policy_.DeleteKey(key), Status::NotFound());
}

TEST_F(UCPosEvictionPolicyTest, AccessKeyReturnsOk)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_TRUE(policy_.AccessKey(key).Success());
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsEmptyWhenNoEntries)
{
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsEvictsAllAtFullRatio)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    auto k3 = KeyFromHex("a3");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2, 2, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k3, MakeEntry(k3, 3, past_)).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 3UL);
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsRespectsEvictRatio)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    auto k3 = KeyFromHex("a3");
    auto k4 = KeyFromHex("a4");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2, 2, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k3, MakeEntry(k3, 3, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k4, MakeEntry(k4, 4, past_)).Success());
    auto victims = policy_.GetEvictionResults(0.5);
    ASSERT_EQ(victims.size(), 2UL);
    EXPECT_EQ(victims[0]->key, k4);
    EXPECT_EQ(victims[1]->key, k3);
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsOrdersByPositionDescending)
{
    auto kLow = KeyFromHex("a1");
    auto kHigh = KeyFromHex("a2");
    auto kMid = KeyFromHex("a3");
    ASSERT_TRUE(policy_.AddKey(kLow, MakeEntry(kLow, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(kHigh, MakeEntry(kHigh, 3, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(kMid, MakeEntry(kMid, 2, past_)).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 3UL);
    EXPECT_EQ(victims[0]->key, kHigh);
    EXPECT_EQ(victims[1]->key, kMid);
    EXPECT_EQ(victims[2]->key, kLow);
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsTiebreaksByLifeTimeoutAscending)
{
    auto kLater = KeyFromHex("a1");
    auto kEarlier = KeyFromHex("a2");
    ASSERT_TRUE(policy_.AddKey(kLater, MakeEntry(kLater, 5, future_)).Success());
    ASSERT_TRUE(policy_.AddKey(kEarlier, MakeEntry(kEarlier, 5, past_)).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 2UL);
    EXPECT_EQ(victims[0]->key, kEarlier);
    EXPECT_EQ(victims[1]->key, kLater);
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsSkipsNonReadyAndContinues)
{
    auto kDeleting = KeyFromHex("a1");
    auto kReady = KeyFromHex("a2");
    ASSERT_TRUE(
        policy_.AddKey(kDeleting, MakeEntry(kDeleting, 5, past_, EntryStatus::DELETING)).Success());
    ASSERT_TRUE(policy_.AddKey(kReady, MakeEntry(kReady, 1, past_, EntryStatus::READY)).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0]->key, kReady);
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsSkipsNonZeroRefCnt)
{
    auto k = KeyFromHex("a1");
    ASSERT_TRUE(
        policy_.AddKey(k, MakeEntry(k, 1, past_, EntryStatus::READY, /*refCnt=*/1)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsSkipsLeasedEntry)
{
    auto k = KeyFromHex("a1");
    ASSERT_TRUE(
        policy_.AddKey(k, MakeEntry(k, 1, past_, EntryStatus::READY, 0, future_)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCPosEvictionPolicyTest, GetEvictionResultsMarksVictimsAsDeleting)
{
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 1, past_);
    ASSERT_TRUE(policy_.AddKey(k, entry).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(entry->status, EntryStatus::DELETING);
}
