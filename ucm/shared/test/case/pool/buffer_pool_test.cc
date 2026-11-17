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
#include <array>
#include <atomic>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <vector>
#include "pool/pool_test_base.h"
namespace UC {
namespace {

using MemoryType = BufferPool::MemoryType;

class BufferPoolTest : public Test::PoolTestBase {};

TEST_F(BufferPoolTest, RejectsInvalidInitAndUseBeforeInit)
{
    BufferPool pool;
    BufferPool::Slot slot;
    EXPECT_FALSE(pool.IsInitialized());

    auto status = pool.Allocate(slot);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::Error());

    status = pool.Free(0);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::Error());

    status = pool.Init("zero_capacity", MemoryType::Host, 0, 1);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Init("zero_slots", MemoryType::Host, 64, 0);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Init("zero_alignment", MemoryType::Host, 64, 1, false, 0);
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
    ASSERT_TRUE(pool.Init("first", MemoryType::Host, 64, 1).Success());

    auto status = pool.Init("second", MemoryType::Host, 64, 1);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());
}

TEST_F(BufferPoolTest, RejectsSlotLayoutOverflow)
{
    BufferPool pool;
    auto status = pool.Init("capacity_overflow", MemoryType::Host,
                            std::numeric_limits<std::size_t>::max(), 2);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());

    status = pool.Init("index_overflow", MemoryType::Host, 64,
                       std::numeric_limits<std::uint32_t>::max());
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());
}

TEST_F(BufferPoolTest, HostPoolUsesAlignedSlotStrideAndReportsBusyWhenFull)
{
    BufferPool pool;
    auto status = pool.Init("host_pool", MemoryType::Host, 71, 2, true);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_TRUE(pool.IsInitialized());

    ASSERT_NE(pool.GetLocalAddr(), nullptr);
    EXPECT_EQ(pool.GetLocalAddr(), pool.GetDeviceAddr());
    EXPECT_EQ(pool.GetTotalSize(), 256);
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::Host);

    const auto* bytes = static_cast<const std::uint8_t*>(pool.GetLocalAddr());
    for (std::size_t i = 0; i < pool.GetTotalSize(); ++i) { EXPECT_EQ(bytes[i], 0); }

    BufferPool::Slot first;
    BufferPool::Slot second;
    BufferPool::Slot third;
    ASSERT_TRUE(pool.Allocate(first).Success());
    ASSERT_TRUE(pool.Allocate(second).Success());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.localAddr) -
                  reinterpret_cast<std::uintptr_t>(first.localAddr),
              128);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.deviceAddr) -
                  reinterpret_cast<std::uintptr_t>(first.deviceAddr),
              128);
    EXPECT_EQ(first.length, 71);
    EXPECT_TRUE(pool.IsValidPointer(first.localAddr));
    EXPECT_FALSE(pool.IsValidPointer(static_cast<char*>(first.localAddr) + 1));
    EXPECT_FALSE(
        pool.IsValidPointer(static_cast<char*>(pool.GetLocalAddr()) + pool.GetTotalSize()));

    status = pool.Allocate(third);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::NoSpace());
}

TEST_F(BufferPoolTest, SupportsCustomSizeAndOffsetAlignment)
{
    constexpr std::size_t kAlignment = 16 * 1024;
    constexpr std::size_t kCapacity = kAlignment + 1;
    constexpr std::size_t kStride = 2 * kAlignment;

    BufferPool pool;
    auto status = pool.Init("aligned_pool", MemoryType::Host, kCapacity, 2, false, kAlignment);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(pool.GetTotalSize(), 2 * kStride);
    EXPECT_EQ(pool.GetTotalSize() % kAlignment, 0);

    BufferPool::Slot first;
    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(first).Success());
    ASSERT_TRUE(pool.Allocate(second).Success());
    EXPECT_EQ(first.localAddr, pool.GetLocalAddr());
    EXPECT_EQ(first.deviceAddr, pool.GetDeviceAddr());
    EXPECT_EQ(first.length, kCapacity);
    EXPECT_EQ(first.offset, std::size_t{0});
    EXPECT_EQ(second.offset, kStride);
    EXPECT_EQ(static_cast<char*>(pool.GetLocalAddr()) + second.offset, second.localAddr);
    EXPECT_EQ(static_cast<char*>(pool.GetDeviceAddr()) + second.offset, second.deviceAddr);

    const auto localOffset = reinterpret_cast<std::uintptr_t>(second.localAddr) -
                             reinterpret_cast<std::uintptr_t>(first.localAddr);
    const auto deviceOffset = reinterpret_cast<std::uintptr_t>(second.deviceAddr) -
                              reinterpret_cast<std::uintptr_t>(first.deviceAddr);
    EXPECT_EQ(localOffset, kStride);
    EXPECT_EQ(deviceOffset, kStride);
    EXPECT_EQ(localOffset % kAlignment, 0);
    EXPECT_EQ(deviceOffset % kAlignment, 0);
}

TEST_F(BufferPoolTest, HostMappedDevicePoolKeepsLocalAndDeviceAddresses)
{
    if (!SupportsHostMappedDeviceBuffer()) {
        BufferPool unsupportedPool;
        EXPECT_EQ(unsupportedPool.Init("unsupported_host_mapped_device_pool",
                                       MemoryType::HostMappedDevice, 4096, 2),
                  Status::Unsupported());
        EXPECT_FALSE(unsupportedPool.IsInitialized());
        return;
    }

    Trans::Device device;
    auto stream = device.MakeStream();
    ASSERT_NE(stream, nullptr);

    BufferPool pool;
    auto status = pool.Init("host_mapped_device_pool", MemoryType::HostMappedDevice, 4096, 2);
    ASSERT_TRUE(status.Success()) << status.ToString();

    ASSERT_NE(pool.GetLocalAddr(), nullptr);
    ASSERT_NE(pool.GetDeviceAddr(), nullptr);
    EXPECT_EQ(pool.GetTotalSize(), 8192);
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::HostMappedDevice);

    BufferPool::Slot first;
    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(first).Success());
    ASSERT_TRUE(pool.Allocate(second).Success());
    EXPECT_EQ(first.localAddr, pool.GetLocalAddr());
    EXPECT_EQ(first.deviceAddr, pool.GetDeviceAddr());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.localAddr) -
                  reinterpret_cast<std::uintptr_t>(first.localAddr),
              4096);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.deviceAddr) -
                  reinterpret_cast<std::uintptr_t>(first.deviceAddr),
              4096);

    std::memset(first.localAddr, 0xAB, first.length);
    std::array<std::uint8_t, 4096> host{};
    ASSERT_TRUE(stream->DeviceToHost(first.deviceAddr, host.data(), host.size()).Success());
    for (const auto value : host) { EXPECT_EQ(value, 0xAB); }
}

TEST_F(BufferPoolTest, DevicePoolZeroesReleasedSlot)
{
    constexpr std::size_t kSlotCapacity = 71;
    constexpr std::size_t kSlotStride = 128;

    Trans::Device device;
    auto stream = device.MakeStream();
    ASSERT_NE(stream, nullptr);

    BufferPool pool;
    auto status = pool.Init("device_pool", MemoryType::Device, kSlotCapacity, 1, true);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(pool.GetLocalAddr(), pool.GetDeviceAddr());
    EXPECT_EQ(pool.GetTotalSize(), kSlotStride);
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::Device);

    BufferPool::Slot first;
    ASSERT_TRUE(pool.Allocate(first).Success());

    std::array<std::uint8_t, kSlotStride> dirty;
    dirty.fill(0xAB);
    ASSERT_TRUE(stream->HostToDevice(dirty.data(), first.deviceAddr, kSlotStride).Success());

    ASSERT_TRUE(pool.Free(first.slotIndex).Success());

    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(second).Success());
    EXPECT_EQ(second.localAddr, first.localAddr);
    EXPECT_EQ(second.slotIndex, first.slotIndex);

    std::array<std::uint8_t, kSlotStride> host{};
    ASSERT_TRUE(stream->DeviceToHost(second.deviceAddr, host.data(), kSlotStride).Success());

    for (const auto value : host) { EXPECT_EQ(value, 0); }
}

TEST_F(BufferPoolTest, DeviceMappedHostPoolAllocatesAndZeroesReleasedSlot)
{
    if (!SupportsDeviceMappedHostBuffer()) {
        BufferPool unsupportedPool;
        EXPECT_EQ(unsupportedPool.Init("unsupported_device_mapped_host_pool",
                                       MemoryType::DeviceMappedHost, 71, 1, true),
                  Status::Unsupported());
        EXPECT_FALSE(unsupportedPool.IsInitialized());
        return;
    }

    constexpr std::size_t kSlotCapacity = 71;
    constexpr std::size_t kSlotStride = 128;

    Trans::Device device;
    auto stream = device.MakeStream();
    ASSERT_NE(stream, nullptr);

    BufferPool pool;
    auto status =
        pool.Init("device_mapped_host_pool", MemoryType::DeviceMappedHost, kSlotCapacity, 1, true);
    ASSERT_TRUE(status.Success()) << status.ToString();
    EXPECT_EQ(pool.GetLocalAddr(), pool.GetDeviceAddr());
    EXPECT_EQ(pool.GetTotalSize(), kSlotStride);
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::DeviceMappedHost);

    BufferPool::Slot slot;
    ASSERT_TRUE(pool.Allocate(slot).Success());
    std::array<std::uint8_t, kSlotStride> dirty;
    dirty.fill(0xAB);
    ASSERT_TRUE(stream->HostToDevice(dirty.data(), slot.deviceAddr, kSlotStride).Success());
    ASSERT_TRUE(pool.Free(slot.slotIndex).Success());

    ASSERT_TRUE(pool.Allocate(slot).Success());
    std::array<std::uint8_t, kSlotStride> host{};
    ASSERT_TRUE(stream->DeviceToHost(slot.deviceAddr, host.data(), kSlotStride).Success());
    for (const auto value : host) { EXPECT_EQ(value, 0); }
}

TEST_F(BufferPoolTest, FreeZeroesAndReusesHostSlot)
{
    constexpr std::size_t kSlotStride = 128;

    BufferPool pool;
    auto status = pool.Init("reuse_pool", MemoryType::Host, 71, 1, true);
    ASSERT_TRUE(status.Success()) << status.ToString();

    BufferPool::Slot first;
    ASSERT_TRUE(pool.Allocate(first).Success());
    std::memset(first.localAddr, 0xAB, kSlotStride);
    ASSERT_TRUE(pool.Free(first.slotIndex).Success());

    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(second).Success());
    EXPECT_EQ(second.localAddr, first.localAddr);
    EXPECT_EQ(second.slotIndex, first.slotIndex);
    EXPECT_EQ(second.offset, first.offset);

    const auto* bytes = static_cast<const std::uint8_t*>(second.localAddr);
    for (std::size_t i = 0; i < kSlotStride; ++i) { EXPECT_EQ(bytes[i], 0); }
}

TEST_F(BufferPoolTest, FreePreservesHostSlotWhenZeroingDisabled)
{
    constexpr std::size_t kSlotStride = 128;

    BufferPool pool;
    auto status = pool.Init("reuse_without_zero", MemoryType::Host, 71, 1);
    ASSERT_TRUE(status.Success()) << status.ToString();

    BufferPool::Slot first;
    ASSERT_TRUE(pool.Allocate(first).Success());
    std::memset(first.localAddr, 0xAB, kSlotStride);
    ASSERT_TRUE(pool.Free(first.slotIndex).Success());

    BufferPool::Slot second;
    ASSERT_TRUE(pool.Allocate(second).Success());
    EXPECT_EQ(second.localAddr, first.localAddr);
    EXPECT_EQ(second.slotIndex, first.slotIndex);

    const auto* bytes = static_cast<const std::uint8_t*>(second.localAddr);
    for (std::size_t i = 0; i < kSlotStride; ++i) { EXPECT_EQ(bytes[i], 0xAB); }
}

TEST_F(BufferPoolTest, RejectsInvalidFree)
{
    BufferPool pool;
    ASSERT_TRUE(pool.Init("validation_pool", MemoryType::Host, 64, 1).Success());

    auto status = pool.Free(1);
    EXPECT_TRUE(status.Failure());
    EXPECT_EQ(status, Status::InvalidParam());
}

TEST_F(BufferPoolTest, ResetAllowsReinitialization)
{
    BufferPool pool;
    ASSERT_TRUE(pool.Init("first", MemoryType::Host, 64, 1).Success());
    pool.Reset();

    EXPECT_FALSE(pool.IsInitialized());
    EXPECT_EQ(pool.GetLocalAddr(), nullptr);
    EXPECT_EQ(pool.GetDeviceAddr(), nullptr);
    EXPECT_EQ(pool.GetTotalSize(), 0);
    EXPECT_TRUE(pool.Init("second", MemoryType::Host, 128, 2).Success());
}

TEST_F(BufferPoolTest, ConcurrentAllocateAndFree)
{
    BufferPool pool;
    ASSERT_TRUE(pool.Init("concurrent_pool", MemoryType::Host, 64, 32).Success());

    constexpr int kThreadCount = 4;
    constexpr int kOpsPerThread = 500;
    std::atomic<bool> failed{false};

    auto worker = [&pool, &failed]() {
        for (int i = 0; i < kOpsPerThread; ++i) {
            BufferPool::Slot slot;
            auto status = pool.Allocate(slot);
            if (status.Failure()) {
                failed = true;
                return;
            }
            std::memset(slot.localAddr, 0xAB, slot.length);
            status = pool.Free(slot.slotIndex);
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
