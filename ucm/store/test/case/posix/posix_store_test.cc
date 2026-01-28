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
#include "posix/cc/posix_store.h"
#include "detail/data_generator.h"
#include "detail/path_base.h"
#include "detail/types_helper.h"

class UCPosixStoreTest : public UC::Test::Detail::PathBase {};

TEST_F(UCPosixStoreTest, SetupWithInvalidParam)
{
    using namespace UC::PosixStore;
    {
        UC::Detail::Dictionary config;
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam());
    }
    {
        UC::Detail::Dictionary config;
        config.Set("storage_backends", std::vector<std::string>{Path()});
        config.SetNumber("device_id", 0);
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam());
    }
    {
        UC::Detail::Dictionary config;
        config.Set("storage_backends", std::vector<std::string>{Path()});
        config.SetNumber("device_id", 0);
        config.SetNumber("tensor_size", size_t(4096));
        config.SetNumber("shard_size", size_t(4096));
        config.SetNumber("block_size", size_t(4096));
        config.SetNumber("stream_number", size_t(0));
        PosixStore store;
        ASSERT_EQ(store.Setup(config), UC::Status::InvalidParam());
    }
}

TEST_F(UCPosixStoreTest, DumpThenLoad)
{
    using namespace UC::PosixStore;
    UC::Detail::Dictionary config;
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{Path()});
    constexpr size_t dataSize = 32768;
    config.SetNumber("tensor_size", dataSize);
    config.SetNumber("shard_size", dataSize);
    config.SetNumber("block_size", dataSize);
    PosixStore store;
    auto s = store.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t nBlocks = 1;
    auto founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{false});
    UC::Test::Detail::DataGenerator data1{nBlocks, dataSize};
    data1.GenerateRandom();
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    desc1.push_back(UC::Detail::Shard{block, 0, {data1.Buffer()}});
    auto handle1 = store.Dump(desc1);
    ASSERT_TRUE(handle1.HasValue());
    s = store.Wait(handle1.Value());
    ASSERT_EQ(s, UC::Status::OK());
    founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{true});
    UC::Test::Detail::DataGenerator data2{nBlocks, dataSize};
    data2.Generate();
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    desc2.push_back(UC::Detail::Shard{block, 0, {data2.Buffer()}});
    auto handle2 = store.Load(desc2);
    ASSERT_TRUE(handle2.HasValue());
    s = store.Wait(handle2.Value());
    ASSERT_EQ(s, UC::Status::OK());
    ASSERT_EQ(data1.Compare(data2), 0);
}

TEST_F(UCPosixStoreTest, DumpThenLoadWithIoDirect)
{
    using namespace UC::PosixStore;
    UC::Detail::Dictionary config;
    config.SetNumber("device_id", 0);
    config.Set("storage_backends", std::vector<std::string>{Path()});
    constexpr size_t dataSize = 32768;
    config.SetNumber("tensor_size", dataSize);
    config.SetNumber("shard_size", dataSize);
    config.SetNumber("block_size", dataSize);
    config.Set("io_direct", true);
    PosixStore store;
    auto s = store.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t nBlocks = 1;
    auto founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{false});
    void* buffer1 = nullptr;
    auto ret = posix_memalign(&buffer1, 4096, dataSize);
    ASSERT_EQ(ret, 0);
    *(size_t*)buffer1 = 0xfffffffe;
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    desc1.push_back(UC::Detail::Shard{block, 0, {buffer1}});
    auto handle1 = store.Dump(desc1);
    ASSERT_TRUE(handle1.HasValue());
    s = store.Wait(handle1.Value());
    ASSERT_EQ(s, UC::Status::OK());
    founds = store.Lookup(&block, nBlocks);
    ASSERT_TRUE(founds.HasValue());
    ASSERT_EQ(founds.Value(), std::vector<uint8_t>{true});
    void* buffer2 = nullptr;
    ret = posix_memalign(&buffer2, 4096, dataSize);
    ASSERT_EQ(ret, 0);
    *(size_t*)buffer2 = 0x00000001;
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    desc2.push_back(UC::Detail::Shard{block, 0, {buffer2}});
    auto handle2 = store.Load(desc2);
    ASSERT_TRUE(handle2.HasValue());
    s = store.Wait(handle2.Value());
    ASSERT_EQ(s, UC::Status::OK());
    ASSERT_EQ(*(size_t*)buffer1, *(size_t*)buffer2);
    free(buffer1);
    free(buffer2);
}
