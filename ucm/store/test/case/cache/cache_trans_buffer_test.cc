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
    config.bufferNumber = 32768;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
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

TEST_P(UCCacheTransBufferTest, InsertDifferentDataRepeatedly)
{
    constexpr size_t nBatch = 2;
    constexpr size_t nBlock = 16;
    constexpr size_t nShard = 64;
    UC::CacheStore::TransBuffer transBuffer;
    UC::CacheStore::Config config;
    config.uniqueId = rd.RandomString(10);
    config.shardSize = 4096;
    config.bufferNumber = nBlock * nShard;
    config.shareBufferEnable = GetParam();
    config.deviceId = 0;
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
