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
#include "dram/cc/entry.h"
#include "dram/cc/metadata.h"
#include "dram_test_common.h"

using UC::DramStore::EntryPtr;
using UC::DramStore::EntryStatus;
using UC::DramStore::EvictionPolicyType;
using UC::DramStore::MetadataConfig;
using UC::DramStore::ShardMetadata;
using UC::Test::Dram::Clock;
using UC::Test::Dram::KeyFromHex;
using UC::Test::Dram::MakeEntry;
using UC::Test::Dram::TimePoint;

namespace {
MetadataConfig MakeConfig(EvictionPolicyType periodic = EvictionPolicyType::TTL,
                          EvictionPolicyType deep = EvictionPolicyType::POSITION,
                          std::chrono::milliseconds leaseTime = std::chrono::milliseconds(100),
                          double defaultEvictRatio = 1.0)
{
    return MetadataConfig{periodic, deep, leaseTime, defaultEvictRatio};
}
}  // namespace

class UCShardMetadataTest : public testing::Test {
protected:
    const TimePoint past_ = Clock::now() - std::chrono::seconds(10);
    const TimePoint future_ = Clock::now() + std::chrono::seconds(10);
};

TEST_F(UCShardMetadataTest, AddKeyReturnsOk)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_TRUE(md.AddKey(k, MakeEntry(k, 0, past_)).Success());
}

TEST_F(UCShardMetadataTest, AddKeyDuplicateReturnsOkAndKeepsSingleEntry)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_TRUE(md.AddKey(k, MakeEntry(k, 0, past_)).Success());
    EXPECT_TRUE(md.AddKey(k, MakeEntry(k, 0, past_)).Success());
    EXPECT_EQ(md.GetKeyCnt(), 1UL);
}

TEST_F(UCShardMetadataTest, AddKeyNullptrReturnsInvalidParam)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_FALSE(md.AddKey(k, nullptr).Success());
}

TEST_F(UCShardMetadataTest, DeleteKeyReturnsOkAndSecondIsNotFound)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_TRUE(md.AddKey(k, MakeEntry(k, 0, past_)).Success());
    EXPECT_TRUE(md.DeleteKey(k).Success());
    EXPECT_FALSE(md.DeleteKey(k).Success());
}

TEST_F(UCShardMetadataTest, QueryKeyReturnsTrueAfterAdd)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    md.AddKey(k, MakeEntry(k, 0, past_));
    EXPECT_TRUE(md.QueryKey(k));
}

TEST_F(UCShardMetadataTest, QueryKeyReturnsFalseWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EXPECT_FALSE(md.QueryKey(KeyFromHex("a1")));
}

TEST_F(UCShardMetadataTest, QueryKeyWithEntryOutFillsEntry)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    auto added = MakeEntry(k, 0, past_);
    md.AddKey(k, added);
    EntryPtr out;
    EXPECT_TRUE(md.QueryKey(k, out));
    EXPECT_EQ(out, added);
}

TEST_F(UCShardMetadataTest, QueryKeyWithEntryOutReturnsFalseWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EntryPtr out = std::make_shared<UC::DramStore::Entry>();
    EXPECT_FALSE(md.QueryKey(KeyFromHex("a1"), out));
    EXPECT_EQ(out, nullptr);
}

TEST_F(UCShardMetadataTest, GetKeyCntReflectsAddsAndDeletes)
{
    ShardMetadata md(MakeConfig());
    EXPECT_EQ(md.GetKeyCnt(), 0UL);
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    md.AddKey(k1, MakeEntry(k1, 0, past_));
    md.AddKey(k2, MakeEntry(k2, 0, past_));
    EXPECT_EQ(md.GetKeyCnt(), 2UL);
    md.DeleteKey(k1);
    EXPECT_EQ(md.GetKeyCnt(), 1UL);
}

TEST_F(UCShardMetadataTest, AccessKeyReturnsNotFoundWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EXPECT_FALSE(md.AccessKey(KeyFromHex("a1")).Success());
}

TEST_F(UCShardMetadataTest, AccessKeyUpdatesLeaseTimeout)
{
    ShardMetadata md(MakeConfig(EvictionPolicyType::TTL, EvictionPolicyType::POSITION,
                                std::chrono::milliseconds(100)));
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_);
    md.AddKey(k, entry);
    EXPECT_EQ(entry->leaseTimeout, TimePoint{});
    EXPECT_TRUE(md.AccessKey(k).Success());
    EXPECT_GT(entry->leaseTimeout, Clock::now());
}

TEST_F(UCShardMetadataTest, EvictPeriodicEvictsExpiredOnly)
{
    ShardMetadata md(MakeConfig());
    auto kExpired = KeyFromHex("a1");
    auto kFresh = KeyFromHex("a2");
    md.AddKey(kExpired, MakeEntry(kExpired, 0, past_));
    md.AddKey(kFresh, MakeEntry(kFresh, 0, future_));
    auto victims = md.EvictPeriodic();
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0], kExpired);
}

TEST_F(UCShardMetadataTest, EvictPeriodicEvictsNothingWhenAllFresh)
{
    ShardMetadata md(MakeConfig());
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    md.AddKey(k1, MakeEntry(k1, 0, future_));
    md.AddKey(k2, MakeEntry(k2, 0, future_));
    EXPECT_TRUE(md.EvictPeriodic().empty());
}
