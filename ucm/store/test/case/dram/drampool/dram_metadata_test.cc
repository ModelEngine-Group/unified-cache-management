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
#include <set>
#include <string>
#include <vector>
#include "dram/cc/drampool/entry.h"
#include "dram/cc/drampool/metadata.h"
#include "dram/dram_test_common.h"

using UC::Status;
using UC::Detail::BlockId;
using UC::DramPool::EntryPtr;
using UC::DramPool::EntryStatus;
using UC::DramPool::EvictionPolicyType;
using UC::DramPool::MetadataConfig;
using UC::DramPool::MetadataManager;
using UC::DramPool::ShardMetadata;
using UC::Test::Dram::Clock;
using UC::Test::Dram::KeyFromHex;
using UC::Test::Dram::MakeEntry;
using UC::Test::Dram::TimePoint;

namespace {
MetadataConfig MakeConfig(EvictionPolicyType periodic = EvictionPolicyType::TTL,
                          EvictionPolicyType deep = EvictionPolicyType::POSITION,
                          std::chrono::milliseconds leaseTime = std::chrono::milliseconds(100),
                          double defaultEvictRatio = 1.0,
                          std::chrono::milliseconds evictPeriod = std::chrono::seconds(60))
{
    return MetadataConfig{periodic, deep, leaseTime, defaultEvictRatio, evictPeriod};
}
}  // namespace

class UCShardMetadataTest : public testing::Test {
protected:
    const TimePoint past_ = Clock::now() - std::chrono::seconds(10);
    const TimePoint future_ = Clock::now() + std::chrono::seconds(10);
};

TEST_F(UCShardMetadataTest, StoreBeginReturnsOk)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_TRUE(md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::INITIALIZED)).Success());
}

TEST_F(UCShardMetadataTest, StoreBeginDuplicateReturnsDuplicateKeyAndKeepsSingleEntry)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_TRUE(md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::INITIALIZED)).Success());
    EXPECT_EQ(md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::INITIALIZED)),
              Status::DuplicateKey());
    EXPECT_EQ(md.GetKeyCnt(), 1UL);
}

TEST_F(UCShardMetadataTest, StoreBeginNullptrReturnsInvalidParam)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_EQ(md.StoreBegin(k, nullptr), Status::InvalidParam());
}

TEST_F(UCShardMetadataTest, StoreBeginRejectsReadyEntry)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_EQ(md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::READY)), Status::InvalidParam());
}

TEST_F(UCShardMetadataTest, StoreBeginRejectsNonZeroRefCnt)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_EQ(md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::INITIALIZED, /*refCnt=*/1)),
              Status::InvalidParam());
}

TEST_F(UCShardMetadataTest, StoreEndMarksInitializedAsReady)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    md.StoreBegin(k, entry);
    EXPECT_EQ(entry->status, EntryStatus::INITIALIZED);
    EXPECT_TRUE(md.StoreEnd(k).Success());
    EXPECT_EQ(entry->status, EntryStatus::READY);
}

TEST_F(UCShardMetadataTest, StoreEndReturnsErrorWhenNotInitialized)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    md.StoreBegin(k, entry);
    ASSERT_TRUE(md.StoreEnd(k).Success());
    EXPECT_EQ(md.StoreEnd(k), Status::Error());
}

TEST_F(UCShardMetadataTest, StoreEndReturnsNotFoundWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EXPECT_EQ(md.StoreEnd(KeyFromHex("a1")), Status::NotFound());
}

TEST_F(UCShardMetadataTest, LoadBeginReturnsNotFoundWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EntryPtr out;
    EXPECT_EQ(md.LoadBegin(KeyFromHex("a1"), out), Status::NotFound());
}

TEST_F(UCShardMetadataTest, LoadBeginIncrementsRefCnt)
{
    ShardMetadata md(MakeConfig(EvictionPolicyType::TTL, EvictionPolicyType::POSITION,
                                std::chrono::milliseconds(100)));
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    md.StoreBegin(k, entry);
    md.StoreEnd(k);
    EXPECT_EQ(entry->refCnt, 0U);
    EntryPtr out;
    EXPECT_TRUE(md.LoadBegin(k, out).Success());
    EXPECT_EQ(entry->refCnt, 1U);
    EXPECT_EQ(out.get(), entry.get());
}

TEST_F(UCShardMetadataTest, LoadBeginReturnsErrorWhenNotReady)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    md.StoreBegin(k, entry);
    EntryPtr out;
    EXPECT_EQ(md.LoadBegin(k, out), Status::Error());
    EXPECT_EQ(entry->refCnt, 0U);
}

TEST_F(UCShardMetadataTest, LoadEndReturnsNotFoundWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EXPECT_EQ(md.LoadEnd(KeyFromHex("a1")), Status::NotFound());
}

TEST_F(UCShardMetadataTest, LoadEndDecrementsRefCnt)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    md.StoreBegin(k, entry);
    md.StoreEnd(k);
    EntryPtr out;
    md.LoadBegin(k, out);
    ASSERT_EQ(entry->refCnt, 1U);
    EXPECT_TRUE(md.LoadEnd(k).Success());
    EXPECT_EQ(entry->refCnt, 0U);
}

TEST_F(UCShardMetadataTest, LoadEndReturnsErrorWhenRefCntIsZero)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::INITIALIZED));
    md.StoreEnd(k);
    EXPECT_EQ(md.LoadEnd(k), Status::Error());
}

TEST_F(UCShardMetadataTest, ExistReturnsTrueAndRefreshesLeaseWhenReady)
{
    ShardMetadata md(MakeConfig(EvictionPolicyType::TTL, EvictionPolicyType::POSITION,
                                std::chrono::milliseconds(100)));
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    md.StoreBegin(k, entry);
    md.StoreEnd(k);
    EXPECT_EQ(entry->leaseTimeout, TimePoint{});
    EXPECT_TRUE(md.Exist(k));
    EXPECT_GT(entry->leaseTimeout, Clock::now());
}

TEST_F(UCShardMetadataTest, ExistReturnsFalseWhenNotReady)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    md.StoreBegin(k, entry);
    EXPECT_FALSE(md.Exist(k));
}

TEST_F(UCShardMetadataTest, ExistReturnsFalseWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EXPECT_FALSE(md.Exist(KeyFromHex("a1")));
}

TEST_F(UCShardMetadataTest, QueryReturnsTrueAfterAdd)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::INITIALIZED));
    EXPECT_TRUE(md.Query(k));
}

TEST_F(UCShardMetadataTest, QueryReturnsFalseWhenMissing)
{
    ShardMetadata md(MakeConfig());
    EXPECT_FALSE(md.Query(KeyFromHex("a1")));
}

TEST_F(UCShardMetadataTest, DeleteReturnsOkAndSecondIsNotFound)
{
    ShardMetadata md(MakeConfig());
    auto k = KeyFromHex("a1");
    EXPECT_TRUE(md.StoreBegin(k, MakeEntry(k, 0, past_, EntryStatus::INITIALIZED)).Success());
    EXPECT_TRUE(md.Delete(k).Success());
    EXPECT_EQ(md.Delete(k), Status::NotFound());
}

TEST_F(UCShardMetadataTest, GetKeyCntReflectsAddsAndDeletes)
{
    ShardMetadata md(MakeConfig());
    EXPECT_EQ(md.GetKeyCnt(), 0UL);
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    md.StoreBegin(k1, MakeEntry(k1, 0, past_, EntryStatus::INITIALIZED));
    md.StoreBegin(k2, MakeEntry(k2, 0, past_, EntryStatus::INITIALIZED));
    EXPECT_EQ(md.GetKeyCnt(), 2UL);
    md.Delete(k1);
    EXPECT_EQ(md.GetKeyCnt(), 1UL);
}

TEST_F(UCShardMetadataTest, FullLifecycleStoreStoreEndLoadExist)
{
    ShardMetadata md(MakeConfig(EvictionPolicyType::TTL, EvictionPolicyType::POSITION,
                                std::chrono::milliseconds(100)));
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, past_, EntryStatus::INITIALIZED);
    EXPECT_TRUE(md.StoreBegin(k, entry).Success());
    EXPECT_EQ(entry->status, EntryStatus::INITIALIZED);
    {
        EntryPtr out;
        EXPECT_EQ(md.LoadBegin(k, out), Status::Error());
    }
    EXPECT_TRUE(md.StoreEnd(k).Success());
    EXPECT_EQ(entry->status, EntryStatus::READY);
    {
        EntryPtr out;
        EXPECT_TRUE(md.LoadBegin(k, out).Success());
        EXPECT_EQ(out.get(), entry.get());
    }
    EXPECT_EQ(entry->refCnt, 1U);
    EXPECT_TRUE(md.Exist(k));
    EXPECT_GT(entry->leaseTimeout, Clock::now());
    EXPECT_TRUE(md.LoadEnd(k).Success());
    EXPECT_EQ(entry->refCnt, 0U);
}

class UCMetadataManagerTest : public testing::Test {
protected:
    const TimePoint past_ = Clock::now() - std::chrono::seconds(10);
    const TimePoint future_ = Clock::now() + std::chrono::seconds(10);
};

TEST_F(UCMetadataManagerTest, SameKeyGoesToSameShard)
{
    MetadataManager mgr(MakeConfig());
    auto k = KeyFromHex("a1");
    auto e1 = MakeEntry(k, 0, future_, EntryStatus::INITIALIZED);
    EXPECT_TRUE(mgr.StoreBegin(k, e1).Success());
    EXPECT_TRUE(mgr.StoreEnd(k).Success());
    auto e2 = MakeEntry(k, 0, future_, EntryStatus::INITIALIZED);
    mgr.StoreBegin(k, e2);
    EXPECT_TRUE(mgr.Query(k));
    EXPECT_EQ(e1->shard, e2->shard);
}

TEST_F(UCMetadataManagerTest, DifferentKeysMayHitDifferentShards)
{
    MetadataManager mgr(MakeConfig());
    std::set<uint32_t> shards;
    for (int i = 0; i < 10; ++i) {
        auto k = KeyFromHex(("a" + std::to_string(i)).c_str());
        auto e = MakeEntry(k, 0, future_, EntryStatus::INITIALIZED);
        mgr.StoreBegin(k, e);
        shards.insert(e->shard);
    }
    EXPECT_GE(shards.size(), 2U);
}

TEST_F(UCMetadataManagerTest, PerKeyOpsDispatchAndRoundtrip)
{
    MetadataManager mgr(MakeConfig());
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 0, future_, EntryStatus::INITIALIZED);
    EXPECT_TRUE(mgr.StoreBegin(k, entry).Success());
    EXPECT_EQ(entry->status, EntryStatus::INITIALIZED);
    EXPECT_TRUE(mgr.StoreEnd(k).Success());
    EXPECT_EQ(entry->status, EntryStatus::READY);
    EXPECT_TRUE(mgr.Query(k));
    EXPECT_FALSE(mgr.Query(KeyFromHex("a2")));
    EXPECT_TRUE(mgr.Exist(k));
    EXPECT_GT(entry->leaseTimeout, Clock::now());
    {
        EntryPtr out;
        EXPECT_TRUE(mgr.LoadBegin(k, out).Success());
        EXPECT_EQ(out.get(), entry.get());
    }
    EXPECT_EQ(entry->refCnt, 1U);
    EXPECT_TRUE(mgr.LoadEnd(k).Success());
    EXPECT_EQ(entry->refCnt, 0U);
    EXPECT_TRUE(mgr.Delete(k).Success());
    EXPECT_FALSE(mgr.Query(k));
}

TEST_F(UCMetadataManagerTest, GetKeyCntAggregatesAcrossShards)
{
    MetadataManager mgr(MakeConfig());
    EXPECT_EQ(mgr.GetKeyCnt(), 0U);
    std::vector<BlockId> keys;
    for (int i = 0; i < 10; ++i) {
        auto k = KeyFromHex(("a" + std::to_string(i)).c_str());
        keys.push_back(k);
        EXPECT_TRUE(
            mgr.StoreBegin(k, MakeEntry(k, 0, future_, EntryStatus::INITIALIZED)).Success());
    }
    EXPECT_EQ(mgr.GetKeyCnt(), 10U);
    for (int i = 0; i < 5; ++i) { EXPECT_TRUE(mgr.Delete(keys[i]).Success()); }
    EXPECT_EQ(mgr.GetKeyCnt(), 5U);
}
