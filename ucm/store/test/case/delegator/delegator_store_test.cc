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
 */
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakeDelegatorStore();

namespace UC::Delegator {
namespace {

std::unique_ptr<StoreV1> CreateStore() { return std::unique_ptr<StoreV1>{MakeDelegatorStore()}; }

TEST(UCDelegatorStoreTest, FactoryReturnsStoreV1)
{
    auto store = CreateStore();
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->Readme(), "DelegatorStore");

    Detail::BlockId block{};
    EXPECT_FALSE(store->Lookup(&block, 1));
    EXPECT_FALSE(store->LookupOnPrefix(&block, 1));

    Detail::TaskDesc task;
    task.push_back(Detail::Shard{});
    EXPECT_FALSE(store->Load(task));
    EXPECT_FALSE(store->Dump(task));
    EXPECT_FALSE(store->Check(1));
    EXPECT_EQ(store->Wait(1), Status::Unsupported());
}

TEST(UCDelegatorStoreTest, RejectsInvalidConfigBeforeCreatingBackend)
{
    {
        auto store = CreateStore();
        Detail::Dictionary config;
        config.Set("role", std::string{"scheduler"});
        EXPECT_TRUE(store->Setup(config).Failure());
    }
    {
        auto store = CreateStore();
        Detail::Dictionary config;
        config.Set("delegator_backend", std::string{"UNKNOWN"});
        config.Set("role", std::string{"scheduler"});
        EXPECT_TRUE(store->Setup(config).Failure());
    }
    {
        auto store = CreateStore();
        Detail::Dictionary config;
        config.Set("delegator_backend", std::string{"ASU"});
        config.Set("role", std::string{"invalid"});
        EXPECT_TRUE(store->Setup(config).Failure());
    }
    {
        auto store = CreateStore();
        Detail::Dictionary config;
        config.Set("delegator_backend", std::string{"ASU"});
        config.Set("role", std::string{"worker"});
        EXPECT_TRUE(store->Setup(config).Failure());
    }
}

}  // namespace
}  // namespace UC::Delegator
