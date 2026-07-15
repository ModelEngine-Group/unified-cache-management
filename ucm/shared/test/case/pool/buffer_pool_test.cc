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
#include "pool/buffer_pool.h"
#include <acl/acl.h>
#include <array>
#include <atomic>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <vector>

namespace UC {
namespace {

using MemoryType = BufferPool::MemoryType;

class BufferPoolTest : public ::testing::Test {
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
};

TEST_F(BufferPoolTest, RejectsInvalidInitAndUseBeforeInit)
{
    BufferPool pool;
    BufferPool::Slot slot;
    EXPECT_FALSE(pool.IsInitialized());

    auto status = pool.Allocate(1, slot);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::Error());

    status = pool.Free(0);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::Error());

    status = pool.Init("zero_capacity", MemoryType::HOST, 0, 1);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Init("zero_slots", MemoryType::HOST, 64, 0);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Init("unsupported", static_cast<MemoryType>(99), 64, 1);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());
    EXPECT_FALSE(pool.IsInitialized());
}

TEST_F(BufferPoolTest, RejectsRepeatedInit)
{
    BufferPool pool;
    ASSERT_TRUE(pool.Init("first", MemoryType::HOST, 64, 1).Success());

    auto status = pool.Init("second", MemoryType::HOST, 64, 1);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());
}

TEST_F(BufferPoolTest, RejectsSlotLayoutOverflow)
{
    BufferPool pool;
    auto status = pool.Init("capacity_overflow", MemoryType::HOST,
                            std::numeric_limits<std::size_t>::max(), 2);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Init("index_overflow", MemoryType::HOST, 64,
                       std::numeric_limits<std::uint32_t>::max());
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());
}

TEST_F(BufferPoolTest, HostPoolUsesAlignedSlotsAndReportsBusyWhenFull)
{
    BufferPool pool;
    auto status = pool.Init("host_pool", MemoryType::HOST, 71, 2, true);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_TRUE(pool.IsInitialized());

    ASSERT_NE(pool.GetLocalAddr(), nullptr);
    EXPECT_EQ(pool.GetLocalAddr(), pool.GetDeviceAddr());
    EXPECT_EQ(pool.GetTotalSize(), 256);
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::HOST);

    const auto* bytes = static_cast<const std::uint8_t*>(pool.GetLocalAddr());
    for (std::size_t i = 0; i < pool.GetTotalSize(); ++i) { EXPECT_EQ(bytes[i], 0); }

    BufferPool::Slot first;
    BufferPool::Slot second;
    BufferPool::Slot third;
    ASSERT_TRUE(pool.Allocate(71, first).Success());
    ASSERT_TRUE(pool.Allocate(71, second).Success());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.local_addr) -
                  reinterpret_cast<std::uintptr_t>(first.local_addr),
              128);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.device_addr) -
                  reinterpret_cast<std::uintptr_t>(first.device_addr),
              128);
    EXPECT_EQ(first.length, 71);
    EXPECT_TRUE(pool.IsValidPointer(first.local_addr));
    EXPECT_FALSE(pool.IsValidPointer(static_cast<char*>(first.local_addr) + 1));
    EXPECT_FALSE(
        pool.IsValidPointer(static_cast<char*>(pool.GetLocalAddr()) + pool.GetTotalSize()));

    status = pool.Allocate(1, third);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::Retry());
}

TEST_F(BufferPoolTest, HostPinnedPoolKeepsLocalAndDeviceAddresses)
{
    BufferPool pool;
    auto status = pool.Init("pinned_pool", MemoryType::HOST_PINNED, 4096, 2);
    ASSERT_TRUE(status.Success()) << status.ToString();

    ASSERT_NE(pool.GetLocalAddr(), nullptr);
    ASSERT_NE(pool.GetDeviceAddr(), nullptr);
    EXPECT_NE(pool.GetLocalAddr(), pool.GetDeviceAddr());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(pool.GetLocalAddr()) % 4096, 0);
    EXPECT_EQ(pool.GetTotalSize(), 8192);
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::HOST_PINNED);

    BufferPool::Slot first;
    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(64, first).Success());
    ASSERT_TRUE(pool.Allocate(64, second).Success());
    EXPECT_EQ(first.local_addr, pool.GetLocalAddr());
    EXPECT_EQ(first.device_addr, pool.GetDeviceAddr());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.local_addr) -
                  reinterpret_cast<std::uintptr_t>(first.local_addr),
              4096);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.device_addr) -
                  reinterpret_cast<std::uintptr_t>(first.device_addr),
              4096);
}

TEST_F(BufferPoolTest, DevicePoolZeroesReleasedSlot)
{
    constexpr std::size_t kSlotCapacity = 71;
    constexpr std::size_t kSlotStride = 128;

    BufferPool pool;
    auto status = pool.Init("device_pool", MemoryType::ASCEND_DEVICE, kSlotCapacity, 1, true);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(pool.GetLocalAddr(), pool.GetDeviceAddr());
    EXPECT_EQ(pool.GetTotalSize(), kSlotStride);
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::ASCEND_DEVICE);

    BufferPool::Slot first;
    ASSERT_TRUE(pool.Allocate(kSlotCapacity, first).Success());
    ASSERT_EQ(aclrtMemset(first.local_addr, kSlotStride, 0xAB, kSlotStride), ACL_SUCCESS);
    ASSERT_TRUE(pool.Free(first.slot_index).Success());

    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(kSlotCapacity, second).Success());
    EXPECT_EQ(second.local_addr, first.local_addr);
    EXPECT_EQ(second.slot_index, first.slot_index);

    std::array<std::uint8_t, kSlotStride> host{};
    ASSERT_EQ(aclrtMemcpy(host.data(), host.size(), second.local_addr, kSlotStride,
                         ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    for (const auto value : host) { EXPECT_EQ(value, 0); }
}

TEST_F(BufferPoolTest, FreeZeroesAndReusesHostSlot)
{
    constexpr std::size_t kSlotStride = 128;

    BufferPool pool;
    auto status = pool.Init("reuse_pool", MemoryType::HOST, 71, 1, true);
    ASSERT_TRUE(status.Success()) << status.ToString();

    BufferPool::Slot first;
    ASSERT_TRUE(pool.Allocate(71, first).Success());
    std::memset(first.local_addr, 0xAB, kSlotStride);
    ASSERT_TRUE(pool.Free(first.slot_index).Success());

    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(71, second).Success());
    EXPECT_EQ(second.local_addr, first.local_addr);
    EXPECT_EQ(second.slot_index, first.slot_index);

    const auto* bytes = static_cast<const std::uint8_t*>(second.local_addr);
    for (std::size_t i = 0; i < kSlotStride; ++i) { EXPECT_EQ(bytes[i], 0); }
}

TEST_F(BufferPoolTest, FreePreservesHostSlotWhenZeroingDisabled)
{
    constexpr std::size_t kSlotStride = 128;

    BufferPool pool;
    auto status = pool.Init("reuse_without_zero", MemoryType::HOST, 71, 1);
    ASSERT_TRUE(status.Success()) << status.ToString();

    BufferPool::Slot first;
    ASSERT_TRUE(pool.Allocate(71, first).Success());
    std::memset(first.local_addr, 0xAB, kSlotStride);
    ASSERT_TRUE(pool.Free(first.slot_index).Success());

    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(71, second).Success());
    EXPECT_EQ(second.local_addr, first.local_addr);
    EXPECT_EQ(second.slot_index, first.slot_index);

    const auto* bytes = static_cast<const std::uint8_t*>(second.local_addr);
    for (std::size_t i = 0; i < kSlotStride; ++i) { EXPECT_EQ(bytes[i], 0xAB); }
}

TEST_F(BufferPoolTest, RejectsInvalidAllocationAndFree)
{
    BufferPool pool;
    ASSERT_TRUE(pool.Init("validation_pool", MemoryType::HOST, 64, 1).Success());

    BufferPool::Slot slot;
    auto status = pool.Allocate(0, slot);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Allocate(65, slot);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Free(1);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());
}

TEST_F(BufferPoolTest, ResetAllowsReinitialization)
{
    BufferPool pool;
    ASSERT_TRUE(pool.Init("first", MemoryType::HOST, 64, 1).Success());
    pool.Reset();

    EXPECT_FALSE(pool.IsInitialized());
    EXPECT_EQ(pool.GetLocalAddr(), nullptr);
    EXPECT_EQ(pool.GetDeviceAddr(), nullptr);
    EXPECT_EQ(pool.GetTotalSize(), 0);
    EXPECT_TRUE(pool.Init("second", MemoryType::HOST, 128, 2).Success());
}

TEST_F(BufferPoolTest, ConcurrentAllocateAndFree)
{
    BufferPool pool;
    ASSERT_TRUE(pool.Init("concurrent_pool", MemoryType::HOST, 64, 32).Success());

    constexpr int kThreadCount = 4;
    constexpr int kOpsPerThread = 500;
    std::atomic<bool> failed{false};

    auto worker = [&pool, &failed]() {
        for (int i = 0; i < kOpsPerThread; ++i) {
            BufferPool::Slot slot;
            auto status = pool.Allocate(64, slot);
            if (status.Failure()) {
                failed = true;
                return;
            }
            std::memset(slot.local_addr, 0xAB, slot.length);
            status = pool.Free(slot.slot_index);
            if (status.Failure()) {
                failed = true;
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreadCount; ++i) { threads.emplace_back(worker); }
    for (auto& thread : threads) { thread.join(); }
    EXPECT_FALSE(failed.load());
}

}  // namespace
}  // namespace UC
