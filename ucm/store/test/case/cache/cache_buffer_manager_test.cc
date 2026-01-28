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
#include "cache/cc/buffer_manager.h"
#include "detail/mock_store.h"
#include "detail/random.h"
#include "detail/types_helper.h"

class UCCacheBufferManagerTest : public testing::Test {
protected:
    static UC::Expected<std::vector<uint8_t>> AllMiss(const UC::Detail::BlockId* blocks, size_t num)
    {
        std::vector<uint8_t> founds(num, false);
        return founds;
    }
    static UC::Expected<std::vector<uint8_t>> AllHit(const UC::Detail::BlockId* blocks, size_t num)
    {
        std::vector<uint8_t> founds(num, true);
        return founds;
    }
};

TEST_F(UCCacheBufferManagerTest, Lookup)
{
    UC::Test::Detail::MockStore backend;
    UC::Test::Detail::Random rd;
    UC::CacheStore::BufferManager bufferMgr;
    UC::CacheStore::Config config;
    config.storeBackend = std::shared_ptr<UC::StoreV1>(&backend, [](auto) {});
    config.deviceId = 0;
    config.tensorSize = 4096;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferNumber = 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    ASSERT_TRUE(bufferMgr.Setup(config).Success());
    std::vector<UC::Detail::BlockId> blocks(3);
    std::for_each(blocks.begin(), blocks.end(), [](auto& block) {
        block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    });
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Return(-1));
    EXPECT_CALL(backend, Lookup).WillOnce(testing::Invoke(AllMiss));
    {
        auto foundIdx = bufferMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, -1);
        auto founds = bufferMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::for_each(founds.begin(), founds.end(), [](auto found) { ASSERT_FALSE(found); });
    }
    EXPECT_CALL(backend, LookupOnPrefix).WillOnce(testing::Invoke([](auto, size_t num) {
        return static_cast<ssize_t>(num) - 1;
    }));
    EXPECT_CALL(backend, Lookup).WillOnce(testing::Invoke(AllHit));
    {
        auto foundIdx = bufferMgr.LookupOnPrefix(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(foundIdx, 2);
        auto founds = bufferMgr.Lookup(blocks.data(), blocks.size()).Value();
        ASSERT_EQ(founds.size(), blocks.size());
        std::for_each(founds.begin(), founds.end(), [](auto found) { ASSERT_TRUE(found); });
    }
}
