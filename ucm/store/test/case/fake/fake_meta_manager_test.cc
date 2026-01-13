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
#include "detail/random.h"
#include "detail/types_helper.h"
#include "fake/cc/meta_manager.h"

class UCFakeMetaManagerTest : public testing::Test {
protected:
    UC::Test::Detail::Random rd;
};

TEST_F(UCFakeMetaManagerTest, FirstInFirstEvict)
{
    constexpr auto number = size_t(4);
    UC::FakeStore::Config config;
    config.uniqueId = rd.RandomString(12);
    config.bufferNumber = number;
    config.shareBufferEnable = true;
    UC::FakeStore::MetaManager metaMgr;
    ASSERT_EQ(metaMgr.Setup(config), UC::Status::OK());
    std::vector<UC::Detail::BlockId> blocks(number + 1);
    auto end = blocks.end();
    std::for_each(blocks.begin(), end, [&](auto& block) {
        block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
        ASSERT_FALSE(metaMgr.Exist(block));
    });
    std::for_each(blocks.begin(), end, [&](const auto& block) { metaMgr.Insert(block); });
    auto iter = blocks.begin();
    ASSERT_FALSE(metaMgr.Exist(*iter));
    iter++;
    std::for_each(iter, end, [&](const auto& block) { ASSERT_TRUE(metaMgr.Exist(block)); });
}

TEST_F(UCFakeMetaManagerTest, BlockDeduplication)
{
    constexpr auto number = size_t(2);
    UC::FakeStore::Config config;
    config.uniqueId = rd.RandomString(12);
    config.bufferNumber = number;
    config.shareBufferEnable = true;
    UC::FakeStore::MetaManager metaMgr;
    ASSERT_EQ(metaMgr.Setup(config), UC::Status::OK());
    auto block1 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    auto block2 = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    ASSERT_FALSE(metaMgr.Exist(block1));
    ASSERT_FALSE(metaMgr.Exist(block2));
    metaMgr.Insert(block1);
    metaMgr.Insert(block1);
    metaMgr.Insert(block2);
    metaMgr.Insert(block2);
    ASSERT_TRUE(metaMgr.Exist(block1));
    ASSERT_TRUE(metaMgr.Exist(block2));
}
