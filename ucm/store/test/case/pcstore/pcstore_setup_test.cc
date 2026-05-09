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
#include <filesystem>
#include <gtest/gtest.h>
#include "pcstore.h"
#include "detail/random.h"
#include "status/status.h"

class UCPcStoreSetupTest : public testing::Test {
protected:
    UC::Test::Detail::Random rd;

    std::filesystem::path MakeStorageRoot()
    {
        auto root = std::filesystem::temp_directory_path() /
                    ("ucm-pcstore-test-" + rd.RandomString(8));
        std::filesystem::create_directories(root);
        return root;
    }
};

TEST_F(UCPcStoreSetupTest, SetupAcceptsEmptyGpuKvBufferLists)
{
    auto root = MakeStorageRoot();
    UC::PcStore store;
    UC::PcStore::Config config({root.string()}, 4096, false);
    ASSERT_EQ(store.Setup(config), UC::Status::OK().Underlying());
    std::filesystem::remove_all(root);
}

TEST_F(UCPcStoreSetupTest, SetupAcceptsSchedulerGdrNicList)
{
    auto root = MakeStorageRoot();
    UC::PcStore store;
    UC::PcStore::Config config({root.string()}, 4096, false);
    config.gdrNicList = {"mlx5_0", "mlx5_1"};
    ASSERT_EQ(store.Setup(config), UC::Status::OK().Underlying());
    std::filesystem::remove_all(root);
}

TEST_F(UCPcStoreSetupTest, SetupAcceptsMultipleGpuKvBufferRanges)
{
    auto root = MakeStorageRoot();
    char buffer1[256] {};
    char buffer2[512] {};
    UC::PcStore store;
    UC::PcStore::Config config({root.string()}, 4096, false);
    config.gpuKvBufferAddrs = {reinterpret_cast<uintptr_t>(buffer1),
                               reinterpret_cast<uintptr_t>(buffer2)};
    config.gpuKvBufferSizes = {sizeof(buffer1), sizeof(buffer2)};
    ASSERT_EQ(store.Setup(config), UC::Status::OK().Underlying());
    std::filesystem::remove_all(root);
}

TEST_F(UCPcStoreSetupTest, SetupRejectsInvalidGpuKvBufferRanges)
{
    auto root = MakeStorageRoot();
    char buffer[256] {};
    {
        UC::PcStore store;
        UC::PcStore::Config config({root.string()}, 4096, false);
        config.gpuKvBufferAddrs = {reinterpret_cast<uintptr_t>(buffer)};
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam().Underlying());
    }
    {
        UC::PcStore store;
        UC::PcStore::Config config({root.string()}, 4096, false);
        config.gpuKvBufferAddrs = {0};
        config.gpuKvBufferSizes = {sizeof(buffer)};
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam().Underlying());
    }
    {
        UC::PcStore store;
        UC::PcStore::Config config({root.string()}, 4096, false);
        config.gpuKvBufferAddrs = {reinterpret_cast<uintptr_t>(buffer)};
        config.gpuKvBufferSizes = {0};
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam().Underlying());
    }
    std::filesystem::remove_all(root);
}

TEST_F(UCPcStoreSetupTest, SetupRejectsInvalidWorkerGdrNicList)
{
    auto root = MakeStorageRoot();
    {
        UC::PcStore store;
        UC::PcStore::Config config({root.string()}, 4096, true);
        config.transferDeviceId = 1;
        config.gdrNicList = {"mlx5_0"};
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam().Underlying());
    }
    {
        UC::PcStore store;
        UC::PcStore::Config config({root.string()}, 4096, true);
        config.transferDeviceId = 0;
        config.gdrNicList = {""};
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam().Underlying());
    }
    std::filesystem::remove_all(root);
}
