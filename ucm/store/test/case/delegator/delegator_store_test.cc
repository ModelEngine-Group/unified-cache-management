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
#include "delegator/cc/delegator_store.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakeDelegatorStore();

namespace UC::Delegator {
namespace {

std::unique_ptr<StoreV1> CreateStore() { return std::unique_ptr<StoreV1>{MakeDelegatorStore()}; }

class FakeBackend final : public StoreV1 {
public:
    Status Setup(const Detail::Dictionary&) override { return Status::OK(); }

    std::string Readme() const override { return "FakeBackend"; }

    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId*, size_t num) override
    {
        return std::vector<uint8_t>(num, true);
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId*, size_t num) override
    {
        return num == 0 ? ssize_t{-1} : static_cast<ssize_t>(num - 1);
    }

    Expected<ssize_t> LookupOnReverse(const Detail::BlockId*, size_t num) override
    {
        return num == 0 ? ssize_t{-1} : 0;
    }

    void Prefetch(const Detail::BlockId*, size_t) override {}

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc) override { return Detail::TaskHandle{1}; }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc) override { return Detail::TaskHandle{2}; }

    Expected<bool> Check(Detail::TaskHandle) override { return true; }

    Status Wait(Detail::TaskHandle) override { return Status::OK(); }
};

StoreV1* MakeFakeBackend() { return new FakeBackend(); }

class BackendFactoryReset final {
public:
    ~BackendFactoryReset() { RegisterBackendFactory(nullptr); }
};

TEST(UCDelegatorStoreTest, FactoryReturnsStoreV1)
{
    auto store = CreateStore();
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->Readme(), "DelegatorStore");

    Detail::BlockId block{};
    EXPECT_FALSE(store->Lookup(&block, 1));
    EXPECT_FALSE(store->LookupOnPrefix(&block, 1));
    EXPECT_FALSE(store->LookupOnReverse(&block, 1));

    Detail::TaskDesc task;
    task.push_back(Detail::Shard{});
    EXPECT_FALSE(store->Load(task));
    EXPECT_FALSE(store->Dump(task));
    EXPECT_FALSE(store->Check(1));
    EXPECT_EQ(store->Wait(1), Status::Unsupported());
}

TEST(UCDelegatorStoreTest, ReportsMissingBackend)
{
    auto store = CreateStore();
    Detail::Dictionary config;
    config.SetNumber("device_id", -1);
    EXPECT_EQ(store->Setup(config), Status::Unsupported());
}

TEST(UCDelegatorStoreTest, InjectsGenericBackendForScheduler)
{
    BackendFactoryReset reset;
    RegisterBackendFactory(&MakeFakeBackend);

    auto store = CreateStore();
    Detail::Dictionary config;
    config.SetNumber("device_id", -1);
    ASSERT_TRUE(store->Setup(config).Success());
    EXPECT_EQ(store->Readme(), "DelegatorStore(FakeBackend)");

    Detail::BlockId block{};
    auto lookup = store->Lookup(&block, 1);
    ASSERT_TRUE(lookup);
    ASSERT_EQ(lookup.Value().size(), std::size_t{1});
    EXPECT_EQ(lookup.Value()[0], 1);
    EXPECT_EQ(store->LookupOnPrefix(&block, 1).Value(), 0);
    EXPECT_EQ(store->LookupOnReverse(&block, 1).Value(), 0);
}

}  // namespace
}  // namespace UC::Delegator
