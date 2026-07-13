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
#include <limits>
#include <memory>
#include "detail/types_helper.h"
#include "dram/cc/entry.h"
#include "dram/cc/lru_eviction_policy.h"

namespace {
using Clock = std::chrono::system_clock;
using TimePoint = Clock::time_point;
using UC::DramStore::Entry;
using UC::DramStore::EntryPtr;
using UC::DramStore::EntryStatus;
using UC::DramStore::LruEvictionPolicy;

EntryPtr MakeEntry(UC::Detail::BlockId key, EntryStatus status = EntryStatus::READY,
                   uint32_t refCnt = 0, TimePoint leaseTimeout = TimePoint{})
{
    auto e = std::make_shared<Entry>();
    e->key = key;
    e->status = status;
    e->refCnt = refCnt;
    e->leaseTimeout = leaseTimeout;
    return e;
}

UC::Detail::BlockId KeyFromHex(const char* hex)
{
    return UC::Test::Detail::TypesHelper::MakeBlockId(hex);
}
}  // namespace

class UCLruEvictionPolicyTest : public testing::Test {
protected:
    LruEvictionPolicy policy_;
    const TimePoint future_ = Clock::now() + std::chrono::seconds(10);
};

TEST_F(UCLruEvictionPolicyTest, AddKeyReturnsOk)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key)).Success());
}

TEST_F(UCLruEvictionPolicyTest, AddKeyDuplicateReturnsDuplicateKey)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key)).Success());
    ASSERT_FALSE(policy_.AddKey(key, MakeEntry(key)).Success());
}

TEST_F(UCLruEvictionPolicyTest, AddKeyNullptrReturnsInvalidParam)
{
    auto key = KeyFromHex("a1");
    auto st = policy_.AddKey(key, nullptr);
    ASSERT_FALSE(st.Success());
}

TEST_F(UCLruEvictionPolicyTest, DeleteKeyReturnsOkAndSecondIsNotFound)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key)).Success());
    ASSERT_TRUE(policy_.DeleteKey(key).Success());
    ASSERT_FALSE(policy_.DeleteKey(key).Success());
}

TEST_F(UCLruEvictionPolicyTest, AccessKeyReturnsNotFoundForMissingKey)
{
    auto key = KeyFromHex("a1");
    ASSERT_FALSE(policy_.AccessKey(key).Success());
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsEmptyWhenNoEntries)
{
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsEmptyWhenEvictRatioIsZero)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(0.0).empty());
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsEmptyWhenEvictRatioIsInvalid)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key)).Success());

    EXPECT_TRUE(policy_.GetEvictionResults(-0.1).empty());
    EXPECT_TRUE(policy_.GetEvictionResults(std::numeric_limits<double>::quiet_NaN()).empty());
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsEvictsOldestEntryFirst)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    auto k3 = KeyFromHex("a3");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2)).Success());
    ASSERT_TRUE(policy_.AddKey(k3, MakeEntry(k3)).Success());

    auto victims = policy_.GetEvictionResults(0.34);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], k1);
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsRespectsEvictRatio)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    auto k3 = KeyFromHex("a3");
    auto k4 = KeyFromHex("a4");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2)).Success());
    ASSERT_TRUE(policy_.AddKey(k3, MakeEntry(k3)).Success());
    ASSERT_TRUE(policy_.AddKey(k4, MakeEntry(k4)).Success());

    auto victims = policy_.GetEvictionResults(0.5);
    ASSERT_EQ(victims.size(), 2UL);
    EXPECT_EQ(victims[0], k1);
    EXPECT_EQ(victims[1], k2);
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsClampsRatioAboveOne)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2)).Success());

    auto victims = policy_.GetEvictionResults(1.5);
    ASSERT_EQ(victims.size(), 2UL);
    EXPECT_EQ(victims[0], k1);
    EXPECT_EQ(victims[1], k2);
}

TEST_F(UCLruEvictionPolicyTest, AccessKeyMovesEntryToRecent)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    auto k3 = KeyFromHex("a3");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2)).Success());
    ASSERT_TRUE(policy_.AddKey(k3, MakeEntry(k3)).Success());
    ASSERT_TRUE(policy_.AccessKey(k1).Success());

    auto victims = policy_.GetEvictionResults(0.34);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], k2);
}

TEST_F(UCLruEvictionPolicyTest, DeleteKeyPreservesLruOrder)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    auto k3 = KeyFromHex("a3");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2)).Success());
    ASSERT_TRUE(policy_.AddKey(k3, MakeEntry(k3)).Success());
    ASSERT_TRUE(policy_.DeleteKey(k2).Success());

    auto victims = policy_.GetEvictionResults(0.5);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], k1);
}

TEST_F(UCLruEvictionPolicyTest, DeleteKeyAllowsReinsert)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key)).Success());
    ASSERT_TRUE(policy_.DeleteKey(key).Success());
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key)).Success());

    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], key);
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsSkipsNonReadyAndContinues)
{
    auto kDeleting = KeyFromHex("a1");
    auto kReady = KeyFromHex("a2");
    ASSERT_TRUE(policy_.AddKey(kDeleting, MakeEntry(kDeleting, EntryStatus::DELETING)).Success());
    ASSERT_TRUE(policy_.AddKey(kReady, MakeEntry(kReady)).Success());

    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], kReady);
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsSkipsNonZeroRefCnt)
{
    auto kInUse = KeyFromHex("a1");
    auto kReady = KeyFromHex("a2");
    ASSERT_TRUE(
        policy_.AddKey(kInUse, MakeEntry(kInUse, EntryStatus::READY, /*refCnt=*/1)).Success());
    ASSERT_TRUE(policy_.AddKey(kReady, MakeEntry(kReady)).Success());

    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], kReady);
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsSkipsLeasedEntry)
{
    auto kLeased = KeyFromHex("a1");
    auto kReady = KeyFromHex("a2");
    ASSERT_TRUE(
        policy_.AddKey(kLeased, MakeEntry(kLeased, EntryStatus::READY, 0, future_)).Success());
    ASSERT_TRUE(policy_.AddKey(kReady, MakeEntry(kReady)).Success());

    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], kReady);
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsEmptyWhenAllEntriesIneligible)
{
    auto kDeleting = KeyFromHex("a1");
    auto kInUse = KeyFromHex("a2");
    auto kLeased = KeyFromHex("a3");
    ASSERT_TRUE(policy_.AddKey(kDeleting, MakeEntry(kDeleting, EntryStatus::DELETING)).Success());
    ASSERT_TRUE(
        policy_.AddKey(kInUse, MakeEntry(kInUse, EntryStatus::READY, /*refCnt=*/1)).Success());
    ASSERT_TRUE(
        policy_.AddKey(kLeased, MakeEntry(kLeased, EntryStatus::READY, 0, future_)).Success());

    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCLruEvictionPolicyTest, GetEvictionResultsMarksVictimsAsDeleting)
{
    auto key = KeyFromHex("a1");
    auto entry = MakeEntry(key);
    ASSERT_TRUE(policy_.AddKey(key, entry).Success());

    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], key);
    EXPECT_EQ(entry->status, EntryStatus::DELETING);
}
