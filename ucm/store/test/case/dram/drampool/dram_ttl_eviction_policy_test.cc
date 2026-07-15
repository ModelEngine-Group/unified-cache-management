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
#include "dram/cc/drampool/ttl_eviction_policy.h"
#include "dram/dram_test_common.h"

using UC::Status;
using UC::DramPool::TtlEvictionPolicy;
using UC::Test::Dram::Clock;
using UC::Test::Dram::EntryStatus;
using UC::Test::Dram::KeyFromHex;
using UC::Test::Dram::MakeEntry;
using UC::Test::Dram::TimePoint;

class UCTtlEvictionPolicyTest : public testing::Test {
protected:
    TtlEvictionPolicy policy_;
    const TimePoint past_ = Clock::now() - std::chrono::seconds(10);
    const TimePoint future_ = Clock::now() + std::chrono::seconds(10);
};

TEST_F(UCTtlEvictionPolicyTest, AddKeyReturnsOk)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
}

TEST_F(UCTtlEvictionPolicyTest, AddKeyDuplicateReturnsDuplicateKey)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_EQ(policy_.AddKey(key, MakeEntry(key, 0, past_)), Status::DuplicateKey());
}

TEST_F(UCTtlEvictionPolicyTest, AddKeyNullptrReturnsInvalidParam)
{
    auto key = KeyFromHex("a1");
    auto st = policy_.AddKey(key, nullptr);
    ASSERT_EQ(st, Status::InvalidParam());
}

TEST_F(UCTtlEvictionPolicyTest, DeleteKeyReturnsOkAndSecondIsNotFound)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_TRUE(policy_.DeleteKey(key).Success());
    ASSERT_EQ(policy_.DeleteKey(key), Status::NotFound());
}

TEST_F(UCTtlEvictionPolicyTest, AccessKeyReturnsOk)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_TRUE(policy_.AccessKey(key).Success());
}

TEST_F(UCTtlEvictionPolicyTest, GetEvictionResultsEmptyWhenNoEntries)
{
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCTtlEvictionPolicyTest, GetEvictionResultsEmptyWhenAllFuture)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1, 0, future_)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2, 0, future_)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCTtlEvictionPolicyTest, GetEvictionResultsEvictsExpiredReadyEntry)
{
    auto k1 = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1, 0, past_)).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], k1);
}

TEST_F(UCTtlEvictionPolicyTest, GetEvictionResultsSkipsNonReadyAndContinues)
{
    auto kExpiredDeleting = KeyFromHex("a1");
    auto kExpiredReady = KeyFromHex("a2");
    auto kFutureReady = KeyFromHex("a3");
    ASSERT_TRUE(
        policy_
            .AddKey(kExpiredDeleting, MakeEntry(kExpiredDeleting, 0, past_, EntryStatus::DELETING))
            .Success());
    ASSERT_TRUE(
        policy_.AddKey(kExpiredReady, MakeEntry(kExpiredReady, 0, past_, EntryStatus::READY))
            .Success());
    ASSERT_TRUE(
        policy_.AddKey(kFutureReady, MakeEntry(kFutureReady, 0, future_, EntryStatus::READY))
            .Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], kExpiredReady);
}

TEST_F(UCTtlEvictionPolicyTest, GetEvictionResultsSkipsNonZeroRefCnt)
{
    auto k = KeyFromHex("a1");
    ASSERT_TRUE(
        policy_.AddKey(k, MakeEntry(k, 0, past_, EntryStatus::READY, /*refCnt=*/1)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCTtlEvictionPolicyTest, GetEvictionResultsSkipsLeasedEntry)
{
    auto k = KeyFromHex("a1");
    ASSERT_TRUE(
        policy_.AddKey(k, MakeEntry(k, 0, past_, EntryStatus::READY, 0, future_)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCTtlEvictionPolicyTest, GetEvictionResultsMarksVictimsAsDeleting)
{
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_);
    ASSERT_TRUE(policy_.AddKey(k, entry).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(entry->status, EntryStatus::DELETING);
}
