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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "detail/mock_store.h"
#include "detail/random.h"
#include "type/dictionary.h"
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakeCacheStore();

class UCCacheStoreSetupTest : public testing::Test {
protected:
    UC::Test::Detail::Random rd;

    UC::Detail::Dictionary MakeConfig(UC::Test::Detail::MockStore& backend)
    {
        UC::Detail::Dictionary config;
        config.Set<UC::StoreV1*>("store_backend", &backend);
        config.Set("unique_id", rd.RandomString(10));
        config.Set("share_buffer_enable", false);
        config.SetNumber("device_id", -1);
        return config;
    }
};

TEST_F(UCCacheStoreSetupTest, SetupAcceptsEmptyGpuKvBufferLists)
{
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Readme).WillRepeatedly(testing::Return("MockStore"));
    std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
    auto config = MakeConfig(backend);
    ASSERT_EQ(store->Setup(config), UC::Status::OK());
}

TEST_F(UCCacheStoreSetupTest, SetupAcceptsSchedulerGdrNicList)
{
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Readme).WillRepeatedly(testing::Return("MockStore"));
    std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
    auto config = MakeConfig(backend);
    config.Set("gdr_nic_list", std::vector<std::string>{"mlx5_0", "mlx5_1"});
    ASSERT_EQ(store->Setup(config), UC::Status::OK());
}

TEST_F(UCCacheStoreSetupTest, SetupAcceptsMultipleGpuKvBufferRanges)
{
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Readme).WillRepeatedly(testing::Return("MockStore"));
    std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
    char buffer1[256] {};
    char buffer2[512] {};
    auto config = MakeConfig(backend);
    config.Set("gpu_kv_buffer_addrs",
               std::vector<ssize_t>{static_cast<ssize_t>(reinterpret_cast<uintptr_t>(buffer1)),
                                    static_cast<ssize_t>(reinterpret_cast<uintptr_t>(buffer2))});
    config.Set("gpu_kv_buffer_sizes",
               std::vector<ssize_t>{static_cast<ssize_t>(sizeof(buffer1)),
                                    static_cast<ssize_t>(sizeof(buffer2))});
    ASSERT_EQ(store->Setup(config), UC::Status::OK());
}

TEST_F(UCCacheStoreSetupTest, SetupRejectsInvalidGpuKvBufferRanges)
{
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Readme).Times(testing::AnyNumber());
    char buffer[256] {};
    {
        std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
        auto config = MakeConfig(backend);
        config.Set("gpu_kv_buffer_addrs",
                   std::vector<ssize_t>{static_cast<ssize_t>(reinterpret_cast<uintptr_t>(buffer))});
        ASSERT_EQ(store->Setup(config), UC::Status::InvalidParam());
    }
    {
        std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
        auto config = MakeConfig(backend);
        config.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{0});
        config.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{static_cast<ssize_t>(sizeof(buffer))});
        ASSERT_EQ(store->Setup(config), UC::Status::InvalidParam());
    }
    {
        std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
        auto config = MakeConfig(backend);
        config.Set("gpu_kv_buffer_addrs",
                   std::vector<ssize_t>{static_cast<ssize_t>(reinterpret_cast<uintptr_t>(buffer))});
        config.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{0});
        ASSERT_EQ(store->Setup(config), UC::Status::InvalidParam());
    }
}

TEST_F(UCCacheStoreSetupTest, SetupRejectsInvalidWorkerGdrNicList)
{
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Readme).Times(testing::AnyNumber());
    {
        std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
        auto config = MakeConfig(backend);
        config.SetNumber("device_id", 1);
        config.Set("gdr_nic_list", std::vector<std::string>{"mlx5_0"});
        ASSERT_EQ(store->Setup(config), UC::Status::InvalidParam());
    }
    {
        std::unique_ptr<UC::StoreV1> store{MakeCacheStore()};
        auto config = MakeConfig(backend);
        config.SetNumber("device_id", 0);
        config.Set("gdr_nic_list", std::vector<std::string>{""});
        ASSERT_EQ(store->Setup(config), UC::Status::InvalidParam());
    }
}
