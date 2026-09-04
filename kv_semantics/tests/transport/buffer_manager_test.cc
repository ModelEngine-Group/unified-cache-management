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
#include "buffer/buffer_manager.h"
#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include "trans_provider.h"

namespace UC::ASU {
namespace {

class BufferManagerTest : public ::testing::Test {
protected:
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

TEST_F(BufferManagerTest, InitHostWithUnalignedSlotCapacity)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 1000, 10);
    ASSERT_TRUE(status.ok()) << status.message;
}

TEST_F(BufferManagerTest, InitDeviceWithUnalignedSlotCapacity)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::DEVICE, 1000, 10);
    ASSERT_TRUE(status.ok()) << status.message;
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
    ASSERT_NE(sge.local_addr, 0);
    ASSERT_EQ(sge.length, 64);
    ASSERT_EQ(sge.tokenId, 0);
    ASSERT_NE(sge.slot_index, UINT32_MAX);
    ASSERT_EQ(sge.memory_type, MemoryType::HOST);

    auto* ptr = reinterpret_cast<void*>(sge.local_addr);
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
        ASSERT_NE(sges[i].local_addr, 0);
        std::memset(reinterpret_cast<void*>(sges[i].local_addr), i, 128);
    }

    for (int i = 0; i < kCount; ++i) {
        auto* data = reinterpret_cast<unsigned char*>(sges[i].local_addr);
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
}

TEST_F(BufferManagerTest, AllocateFull4160ByteSlotCapacity)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 4160, 10);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge;
    status = mgr.Allocate(4160, sge);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(sge.length, 4160);
}

TEST_F(BufferManagerTest, AllocateExceeds4160ByteSlotCapacity)
{
    BufferManager mgr;
    auto status = mgr.Init("test_buffer", MemoryType::HOST, 4160, 10);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge;
    status = mgr.Allocate(4161, sge);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(BufferManagerTest, AllMemoryTypesUseAlignedSlotStride)
{
    for (const auto type : {MemoryType::HOST, MemoryType::HOST_PINNED, MemoryType::DEVICE}) {
        BufferManager mgr;
        auto status = mgr.Init("test_buffer", type, 4160, 2);
        ASSERT_TRUE(status.ok()) << status.message;

        ScatterGatherEntry first;
        ScatterGatherEntry second;
        ASSERT_TRUE(mgr.Allocate(4160, first).ok());
        ASSERT_TRUE(mgr.Allocate(4160, second).ok());
        ASSERT_EQ(second.local_addr - first.local_addr, 4160);
        ASSERT_EQ(second.device_addr - first.device_addr, 4160);
    }
}

TEST_F(BufferManagerTest, FlagBufferCapacity71Uses128ByteStride)
{
    BufferManager mgr;
    auto status = mgr.Init("flag_buffer", MemoryType::HOST_PINNED, 71, 2);
    ASSERT_TRUE(status.ok()) << status.message;

    ScatterGatherEntry first;
    ScatterGatherEntry second;
    ASSERT_TRUE(mgr.Allocate(71, first).ok());
    ASSERT_TRUE(mgr.Allocate(71, second).ok());
    ASSERT_EQ(first.length, 71);
    ASSERT_EQ(second.local_addr - first.local_addr, 128);
    ASSERT_EQ(second.device_addr - first.device_addr, 128);
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
    ASSERT_EQ(sge2.local_addr, sge1.local_addr);
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

            std::memset(reinterpret_cast<void*>(sge.local_addr), thread_id, 64);

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

            std::memset(reinterpret_cast<void*>(sge.local_addr), thread_id, 128);

            for (int j = 0; j < 128; ++j) {
                ASSERT_EQ(reinterpret_cast<unsigned char*>(sge.local_addr)[j], thread_id);
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

    auto* ptr = reinterpret_cast<uint8_t*>(sge1.local_addr);
    std::memset(ptr, 0xAB, 1024);

    status = mgr.Free(sge1.slot_index);
    ASSERT_TRUE(status.ok());

    ScatterGatherEntry sge2;
    status = mgr.Allocate(64, sge2);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(sge2.local_addr, sge1.local_addr);
    ASSERT_EQ(sge2.slot_index, sge1.slot_index);

    auto* ptr2 = reinterpret_cast<uint8_t*>(sge2.local_addr);
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

TEST_F(BufferManagerTest, HostMemoryExposesRegistrationDescription)
{
    BufferManager mgr;
    auto status = mgr.Init("test_host", MemoryType::HOST, 1024, 10);
    ASSERT_TRUE(status.ok()) << status.message;

    TransProvider::RegisterMemoryDesc desc;
    status = mgr.GetRegisterMemoryDesc(desc);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(desc.memoryType, TransProvider::MemType::MEM_HOST);
    ASSERT_NE(desc.addr, 0);
    ASSERT_EQ(desc.localAddr, desc.addr);
    ASSERT_EQ(desc.size, 1024 * 10);
}

TEST_F(BufferManagerTest, HostPinnedMemoryExposesDeviceRegistrationDescription)
{
    BufferManager mgr;
    auto status = mgr.Init("test_pinned", MemoryType::HOST_PINNED, 4096, 1);
    ASSERT_TRUE(status.ok()) << status.message;

    ScatterGatherEntry sge;
    ASSERT_TRUE(mgr.Allocate(64, sge).ok());
    TransProvider::RegisterMemoryDesc desc;
    status = mgr.GetRegisterMemoryDesc(desc);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(desc.memoryType, TransProvider::MemType::MEM_DEVICE);
    ASSERT_EQ(desc.addr, sge.device_addr);
    ASSERT_EQ(desc.localAddr, sge.local_addr);

    // The CPU writes through local_addr while HCOMM and remote RDMA use device_addr.
    // ACL simulators may map both roles to the same numeric address.
    std::memset(reinterpret_cast<void*>(sge.local_addr), 0x5A, sge.length);
    ASSERT_EQ(*reinterpret_cast<unsigned char*>(sge.local_addr), 0x5A);
}

TEST_F(BufferManagerTest, SetTokenIdIsReflectedInAllocatedSge)
{
    BufferManager mgr;
    auto status = mgr.Init("test_token", MemoryType::HOST, 1024, 10);
    ASSERT_TRUE(status.ok()) << status.message;
    mgr.SetTokenId(99);
    ASSERT_EQ(mgr.GetTokenId(), 99);

    ScatterGatherEntry sge;
    status = mgr.Allocate(64, sge);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(sge.tokenId, 99);

    mgr.Free(sge.slot_index);
}

TEST_F(BufferManagerTest, RegistrationDescriptionRequiresInitialization)
{
    BufferManager mgr;
    TransProvider::RegisterMemoryDesc desc;
    const auto status = mgr.GetRegisterMemoryDesc(desc);
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.code, StatusCode::NOT_INITIALIZED);
}

}  // namespace
}  // namespace UC::ASU
