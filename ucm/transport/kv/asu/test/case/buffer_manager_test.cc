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
#include "buffer_manager.h"
#include <acl/acl.h>
#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace UC::ASU {
namespace {

class BufferManagerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
    }
    static void TearDownTestSuite()
    {
        aclrtResetDevice(0);
        aclFinalize();
    }
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(BufferManagerTest, InitAndDestroy)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok()) << status.message;
}

TEST_F(BufferManagerTest, InitWithZeroSlotSize)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 0, 100);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(BufferManagerTest, InitWithZeroSlotNum)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 0);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(BufferManagerTest, DoubleInit)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok());
    status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(BufferManagerTest, AllocateWithoutInit)
{
    BufferManager mgr;
    ScatterGatherEntry sge;
    auto status = mgr.Allocate(64, sge);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::NOT_INITIALIZED);
}

TEST_F(BufferManagerTest, AllocateZeroSize)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge;
    status = mgr.Allocate(0, sge);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(BufferManagerTest, AllocateExceedsSlotSize)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge;
    status = mgr.Allocate(2048, sge);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(BufferManagerTest, SingleAllocateAndFree)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge;
    status = mgr.Allocate(64, sge);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(sge.addr, 0);
    ASSERT_EQ(sge.length, 64);
    ASSERT_NE(sge.lkey, 0);
    ASSERT_NE(sge.slot_index, UINT32_MAX);

    auto* ptr = reinterpret_cast<void*>(sge.addr);
    std::memset(ptr, 0xAB, 64);

    status = mgr.Free(sge.slot_index);
    ASSERT_TRUE(status.ok()) << status.message;
}

TEST_F(BufferManagerTest, MultipleAllocatesAndFrees)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok());

    constexpr int kCount = 50;
    std::vector<ScatterGatherEntry> sges(kCount);

    for (int i = 0; i < kCount; ++i) {
        status = mgr.Allocate(128, sges[i]);
        ASSERT_TRUE(status.ok()) << "Failed at i=" << i << ": " << status.message;
        ASSERT_NE(sges[i].addr, 0);
        std::memset(reinterpret_cast<void*>(sges[i].addr), i, 128);
    }

    for (int i = 0; i < kCount; ++i) {
        auto* data = reinterpret_cast<unsigned char*>(sges[i].addr);
        for (int j = 0; j < 128; ++j) { ASSERT_EQ(data[j], static_cast<unsigned char>(i)); }
    }

    for (int i = 0; i < kCount; ++i) {
        status = mgr.Free(sges[i].slot_index);
        ASSERT_TRUE(status.ok()) << status.message;
    }
}

TEST_F(BufferManagerTest, FreeWithoutInit)
{
    BufferManager mgr;
    auto status = mgr.Free(0);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::NOT_INITIALIZED);
}

TEST_F(BufferManagerTest, FreeOutOfRangeIndex)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok());

    status = mgr.Free(200);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(BufferManagerTest, AllocateFullSlotSize)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 10);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge;
    status = mgr.Allocate(1024, sge);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(sge.length, 1024);

    std::memset(reinterpret_cast<void*>(sge.addr), 0xFF, 1024);

    mgr.Free(sge.slot_index);
}

TEST_F(BufferManagerTest, ReuseAfterFree)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 1);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge1;
    status = mgr.Allocate(64, sge1);
    ASSERT_TRUE(status.ok());

    mgr.Free(sge1.slot_index);

    ScatterGatherEntry sge2;
    status = mgr.Allocate(64, sge2);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(sge2.addr, sge1.addr);
    ASSERT_EQ(sge2.slot_index, sge1.slot_index);

    mgr.Free(sge2.slot_index);
}

TEST_F(BufferManagerTest, ConcurrentAllocateAndFree)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1024, 100);
    ASSERT_TRUE(status.ok());

    constexpr int kThreadCount = 4;
    constexpr int kOpsPerThread = 500;

    auto worker = [&mgr](int thread_id) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            ScatterGatherEntry sge;
            auto s = mgr.Allocate(64, sge);
            ASSERT_TRUE(s.ok()) << "Thread " << thread_id << " op " << i << ": " << s.message;

            std::memset(reinterpret_cast<void*>(sge.addr), thread_id, 64);

            s = mgr.Free(sge.slot_index);
            ASSERT_TRUE(s.ok()) << s.message;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreadCount; ++i) { threads.emplace_back(worker, i); }
    for (auto& t : threads) { t.join(); }
}

TEST_F(BufferManagerTest, ConcurrentStressTest)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 256, 16);
    ASSERT_TRUE(status.ok());

    constexpr int kThreadCount = 4;
    constexpr int kOpsPerThread = 1000;

    auto worker = [&mgr](int thread_id) {
        for (int i = 0; i < kOpsPerThread; ++i) {
            ScatterGatherEntry sge;
            auto s = mgr.Allocate(128, sge);
            ASSERT_TRUE(s.ok());

            std::memset(reinterpret_cast<void*>(sge.addr), thread_id, 128);

            for (int j = 0; j < 128; ++j) {
                ASSERT_EQ(reinterpret_cast<unsigned char*>(sge.addr)[j], thread_id);
            }

            s = mgr.Free(sge.slot_index);
            ASSERT_TRUE(s.ok());
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreadCount; ++i) { threads.emplace_back(worker, i); }
    for (auto& t : threads) { t.join(); }
}

TEST_F(BufferManagerTest, FreeZeroesMemory)
{
    BufferManager mgr;
    auto status = mgr.Init("test", MemoryType::HOST, 1024, 1);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge1;
    status = mgr.Allocate(64, sge1);
    ASSERT_TRUE(status.ok());

    auto* ptr = reinterpret_cast<uint8_t*>(sge1.addr);
    std::memset(ptr, 0xAB, 1024);

    status = mgr.Free(sge1.slot_index);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge2;
    status = mgr.Allocate(64, sge2);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(sge2.addr, sge1.addr);
    ASSERT_EQ(sge2.slot_index, sge1.slot_index);

    auto* ptr2 = reinterpret_cast<uint8_t*>(sge2.addr);
    for (size_t i = 0; i < 1024; ++i) {
        ASSERT_EQ(ptr2[i], 0) << "byte " << i << " not zeroed after free";
    }

    mgr.Free(sge2.slot_index);
}

TEST_F(BufferManagerTest, AllocateReturnsBusyWhenFull)
{
    BufferManager mgr;
    auto status = mgr.Init("test", MemoryType::HOST, 1024, 2);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge1, sge2, sge3;
    ASSERT_TRUE(mgr.Allocate(64, sge1).ok());
    ASSERT_TRUE(mgr.Allocate(64, sge2).ok());

    status = mgr.Allocate(64, sge3);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::RESOURCE_BUSY);

    mgr.Free(sge1.slot_index);
    mgr.Free(sge2.slot_index);
}

}  // namespace
}  // namespace UC::ASU
