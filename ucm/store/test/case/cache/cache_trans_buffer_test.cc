/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <gtest/gtest.h>
#include "cache/cc/trans_buffer.h"
#include "detail/random.h"
#include "detail/types_helper.h"
#include "metrics_api.h"

class UCCacheTransBufferTest : public testing::TestWithParam<bool> {
public:
    UC::Test::Detail::Random rd;
};

INSTANTIATE_TEST_CASE_P(SharedCondition, UCCacheTransBufferTest, ::testing::Values(false, true));

TEST_P(UCCacheTransBufferTest, GetFirstNode)
{
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.bufferCapacity = config.shardSize * 32768;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    auto handle1 = transBuffer.Get(blockId, shardIdx);
    ASSERT_TRUE(handle1);
    ASSERT_TRUE(handle1.Owner());
    ASSERT_FALSE(handle1.Ready());
    auto handle2 = transBuffer.Get(blockId, shardIdx);
    ASSERT_TRUE(handle2);
    ASSERT_FALSE(handle2.Owner());
    ASSERT_FALSE(handle2.Ready());
    ASSERT_EQ(handle1.Data(), handle2.Data());
    handle1.MarkReady();
    ASSERT_TRUE(handle2.Ready());
}

TEST_P(UCCacheTransBufferTest, BackendOnlyLoadReusesIdleCacheEntry)
{
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.bufferCapacity = config.shardSize * 32768;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    config.cacheLoadBackendOnly = true;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    void* cachedAddr = nullptr;
    {
        auto handle1 = transBuffer.Get(blockId, shardIdx);
        ASSERT_TRUE(handle1);
        ASSERT_TRUE(handle1.Owner());
        handle1.MarkReady();
        cachedAddr = handle1.Data();
    }

    auto handle2 = transBuffer.Get(blockId, shardIdx, true, true);
    ASSERT_TRUE(handle2);
    ASSERT_TRUE(handle2.Owner());
    ASSERT_FALSE(handle2.Ready());
    ASSERT_EQ(cachedAddr, handle2.Data());
}

TEST_P(UCCacheTransBufferTest, BackendOnlyReservedGetDoesNotBypassWithoutLoadFlag)
{
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.bufferCapacity = config.shardSize * 32768;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    config.cacheLoadBackendOnly = true;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    {
        auto handle1 = transBuffer.Get(blockId, shardIdx);
        ASSERT_TRUE(handle1);
        ASSERT_TRUE(handle1.Owner());
        handle1.MarkReady();
    }

    auto handle2 = transBuffer.Get(blockId, shardIdx, true);
    ASSERT_TRUE(handle2);
    ASSERT_TRUE(handle2.Owner());
    ASSERT_TRUE(handle2.Ready());
}

TEST_P(UCCacheTransBufferTest, BackendOnlyLoadCoalescesInFlightEntry)
{
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.bufferCapacity = config.shardSize * 32768;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    config.cacheLoadBackendOnly = true;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    auto owner = transBuffer.Get(blockId, shardIdx, true, true);
    ASSERT_TRUE(owner);
    ASSERT_TRUE(owner.Owner());
    auto waiter = transBuffer.Get(blockId, shardIdx, true, true);
    ASSERT_TRUE(waiter);
    ASSERT_FALSE(waiter.Owner());
    ASSERT_EQ(owner.Data(), waiter.Data());
}

TEST_P(UCCacheTransBufferTest, SharesFailureAndRetriesAfterHandlesAreReleased)
{
    using UC::CacheStore::TransBuffer;
    TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.bufferCapacity = config.shardSize * 32768;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    void* failedAddr = nullptr;
    {
        auto owner = transBuffer.Get(blockId, shardIdx, true, true);
        auto waiter = transBuffer.Get(blockId, shardIdx, true, true);
        failedAddr = owner.Data();

        owner.MarkFailed(UC::Status::NotFound());

        ASSERT_EQ(owner.GetState(), TransBuffer::State::FAILED);
        ASSERT_EQ(waiter.GetState(), TransBuffer::State::FAILED);
        ASSERT_EQ(waiter.FailureStatus(), UC::Status::NotFound());
        ASSERT_FALSE(waiter.Ready());
    }

    auto retry = transBuffer.Get(blockId, shardIdx, true, true);
    ASSERT_TRUE(retry.Owner());
    ASSERT_EQ(retry.GetState(), TransBuffer::State::LOADING);
    ASSERT_EQ(retry.Data(), failedAddr);
}

TEST(UCCacheTransBufferSharedTest, SharesFailureAcrossMappings)
{
    using UC::CacheStore::TransBuffer;
    UC::Test::Detail::Random rd;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.bufferCapacity = config.shardSize * 32768;
    config.shareBufferEnable = true;
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    TransBuffer ownerBuffer;
    auto s = ownerBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    TransBuffer waiterBuffer;
    s = waiterBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    auto owner = ownerBuffer.Get(blockId, shardIdx, true, true);
    auto waiter = waiterBuffer.Get(blockId, shardIdx, true, true);

    owner.MarkFailed(UC::Status::NotFound());

    ASSERT_FALSE(waiter.Owner());
    ASSERT_EQ(waiter.GetState(), TransBuffer::State::FAILED);
    ASSERT_EQ(waiter.FailureStatus(), UC::Status::NotFound());
}

TEST_P(UCCacheTransBufferTest, GetReservedNode)
{
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.loadExclusiveBufferNumber = 16;
    config.bufferCapacity = config.shardSize * (config.loadExclusiveBufferNumber + 1);
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId1 = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    auto blockId2 = UC::Test::Detail::TypesHelper::MakeBlockId("a2b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    void* ptr = nullptr;
    {
        auto handle1 = transBuffer.Get(blockId1, shardIdx);
        ASSERT_TRUE(handle1);
        ptr = handle1.Data();
    }
    {
        auto handle2 = transBuffer.Get(blockId2, shardIdx);
        ASSERT_TRUE(handle2);
        ASSERT_EQ(ptr, handle2.Data());
    }
    {
        auto handle1 = transBuffer.Get(blockId1, shardIdx, true);
        ASSERT_TRUE(handle1);
        ptr = handle1.Data();
    }
    {
        auto handle2 = transBuffer.Get(blockId2, shardIdx, true);
        ASSERT_TRUE(handle2);
        ASSERT_NE(ptr, handle2.Data());
    }
}

TEST_P(UCCacheTransBufferTest, InsertDifferentDataRepeatedly)
{
    constexpr size_t nBatch = 2;
    constexpr size_t nBlock = 16;
    constexpr size_t nShard = 64;
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 4096;
    config.bufferCapacity = nBlock * nShard * config.shardSize;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    for (size_t iBatch = 0; iBatch < nBatch; iBatch++) {
        std::vector<UC::Detail::BlockId> blocks(nBlock);
        std::for_each(blocks.begin(), blocks.end(), [&](auto& block) {
            block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        });
        for (size_t iShard = 0; iShard < nShard; iShard++) {
            std::for_each(blocks.begin(), blocks.end(), [&](auto block) {
                ASSERT_FALSE(transBuffer.Exist(block, iShard));
                auto handle = transBuffer.Get(block, iShard);
                ASSERT_TRUE(handle.Owner());
                ASSERT_FALSE(handle.Ready());
                handle.MarkReady();
            });
        }
        for (size_t iShard = 0; iShard < nShard; iShard++) {
            std::for_each(blocks.begin(), blocks.end(), [&](auto block) {
                ASSERT_TRUE(transBuffer.Exist(block, iShard));
                auto handle = transBuffer.Get(block, iShard);
                ASSERT_TRUE(handle.Owner());
                ASSERT_TRUE(handle.Ready());
            });
        }
    }
}

TEST_P(UCCacheTransBufferTest, ClockSparesRecentlyTouchedBlock)
{
    constexpr size_t nNode = 4;
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 32768;
    config.bufferCapacity = config.shardSize * nNode;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
    config.loadExclusiveBufferNumber = 0;
    auto s = transBuffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto b1 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto b2 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto b3 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto b4 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto b5 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto b6 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    constexpr size_t shardIdx = 0;

    {
        auto h = transBuffer.Get(b1, shardIdx);
        h.MarkReady();
    }
    {
        auto h = transBuffer.Get(b2, shardIdx);
        h.MarkReady();
    }
    {
        auto h = transBuffer.Get(b3, shardIdx);
        h.MarkReady();
    }
    {
        auto h = transBuffer.Get(b4, shardIdx);
        h.MarkReady();
    }

    {
        auto h = transBuffer.Get(b5, shardIdx);
        h.MarkReady();
    }
    ASSERT_FALSE(transBuffer.Exist(b1, shardIdx));
    ASSERT_TRUE(transBuffer.Exist(b5, shardIdx));

    {
        auto h = transBuffer.Get(b2, shardIdx);
        ASSERT_TRUE(h.Ready());
    }

    {
        auto h = transBuffer.Get(b6, shardIdx);
        h.MarkReady();
    }
    ASSERT_TRUE(transBuffer.Exist(b2, shardIdx));
    ASSERT_TRUE(transBuffer.Exist(b5, shardIdx));
    ASSERT_FALSE(transBuffer.Exist(b3, shardIdx));
}
