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
#include "pool/variable_buffer_pool.h"
#include <acl/acl.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <vector>
#include "pool/pool_test_base.h"

namespace UC {
namespace {

using MemoryType = VariableBufferPool::MemoryType;

class VariableBufferPoolTest : public Test::PoolTestBase {};

TEST_F(VariableBufferPoolTest, RejectsInvalidInitAndUseBeforeInit)
{
    VariableBufferPool pool;
    VariableBufferPool::AllocationHandle handle;

    EXPECT_EQ(pool.Allocate(64, handle), Status::Error());
    EXPECT_EQ(pool.Allocate(0, handle), Status::Error());
    EXPECT_EQ(pool.Free(handle), Status::Error());
    EXPECT_EQ(pool.Init("zero_capacity", MemoryType::HOST, 0, 16), Status::InvalidParam());
    EXPECT_EQ(pool.Init("too_few_allocations", MemoryType::HOST, 64, 2), Status::InvalidParam());
    EXPECT_EQ(pool.Init("capacity_overflow", MemoryType::HOST,
                        std::numeric_limits<std::size_t>::max(), 16),
              Status::InvalidParam());
    EXPECT_EQ(pool.Init("zero_alignment", MemoryType::HOST, 64, 16, false, 0),
              Status::InvalidParam());
    EXPECT_EQ(pool.Init("alignment_overflow", MemoryType::HOST,
                        std::numeric_limits<std::size_t>::max() - 31, 16, false, 64),
              Status::InvalidParam());
    EXPECT_EQ(pool.Init("unsupported", static_cast<MemoryType>(99), 64, 16),
              Status::InvalidParam());
    EXPECT_FALSE(pool.IsInitialized());

    ASSERT_TRUE(pool.Init("initialized", MemoryType::HOST, 64, 8).Success());
    EXPECT_EQ(pool.Allocate(0, handle), Status::InvalidParam());
}

TEST_F(VariableBufferPoolTest, RejectsRepeatedInit)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("first", MemoryType::HOST, 64, 8).Success());

    EXPECT_EQ(pool.Init("second", MemoryType::HOST, 128, 8), Status::InvalidParam());
    EXPECT_EQ(pool.GetName(), "first");
    EXPECT_EQ(pool.GetTotalSize(), std::size_t{64});
}

TEST_F(VariableBufferPoolTest, RejectsAllocatorUnitOverflow)
{
    if constexpr (std::numeric_limits<std::size_t>::max() >
                  std::numeric_limits<std::uint32_t>::max()) {
        constexpr auto unitOverflow =
            (static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1) * 64;
        VariableBufferPool pool;
        EXPECT_EQ(pool.Init("unit_overflow", MemoryType::HOST, unitOverflow, 16),
                  Status::InvalidParam());
        EXPECT_FALSE(pool.IsInitialized());
    }
}

TEST_F(VariableBufferPoolTest, AllocatesDifferentAlignedSizes)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("variable", MemoryType::HOST, 257, 16, true).Success());

    EXPECT_TRUE(pool.IsInitialized());
    EXPECT_EQ(pool.GetName(), "variable");
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::HOST);
    EXPECT_EQ(pool.GetTotalSize(), std::size_t{320});
    ASSERT_NE(pool.GetLocalAddr(), nullptr);
    EXPECT_EQ(pool.GetLocalAddr(), pool.GetDeviceAddr());
    const auto* initialBytes = static_cast<const std::uint8_t*>(pool.GetLocalAddr());
    for (std::size_t index = 0; index < pool.GetTotalSize(); ++index) {
        EXPECT_EQ(initialBytes[index], 0);
    }

    VariableBufferPool::AllocationHandle first;
    VariableBufferPool::AllocationHandle second;
    VariableBufferPool::AllocationHandle third;
    VariableBufferPool::AllocationHandle exhausted;
    ASSERT_TRUE(pool.Allocate(1, first).Success());
    ASSERT_TRUE(pool.Allocate(65, second).Success());
    ASSERT_TRUE(pool.Allocate(128, third).Success());

    EXPECT_EQ(first.requested_size, std::size_t{1});
    EXPECT_EQ(first.allocated_size, std::size_t{64});
    EXPECT_EQ(first.offset, std::size_t{0});
    EXPECT_EQ(second.requested_size, std::size_t{65});
    EXPECT_EQ(second.allocated_size, std::size_t{128});
    EXPECT_EQ(second.offset, std::size_t{64});
    EXPECT_EQ(third.allocated_size, std::size_t{128});
    EXPECT_EQ(third.offset, std::size_t{192});
    EXPECT_EQ(first.local_addr, static_cast<char*>(pool.GetLocalAddr()) + first.offset);
    EXPECT_EQ(first.device_addr, static_cast<char*>(pool.GetDeviceAddr()) + first.offset);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.local_addr) -
                  reinterpret_cast<std::uintptr_t>(first.local_addr),
              std::uintptr_t{64});
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(third.local_addr) -
                  reinterpret_cast<std::uintptr_t>(second.local_addr),
              std::uintptr_t{128});
    EXPECT_EQ(pool.Allocate(1, exhausted), Status::NoSpace());
}

TEST_F(VariableBufferPoolTest, SupportsCustomAllocationAlignment)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("custom_alignment", MemoryType::HOST, 500, 16, false, 256).Success());
    EXPECT_EQ(pool.GetTotalSize(), std::size_t{512});

    VariableBufferPool::AllocationHandle first;
    VariableBufferPool::AllocationHandle second;
    VariableBufferPool::AllocationHandle exhausted;
    ASSERT_TRUE(pool.Allocate(1, first).Success());
    ASSERT_TRUE(pool.Allocate(200, second).Success());

    EXPECT_EQ(first.allocated_size, std::size_t{256});
    EXPECT_EQ(first.offset, std::size_t{0});
    EXPECT_EQ(second.allocated_size, std::size_t{256});
    EXPECT_EQ(second.offset, std::size_t{256});
    EXPECT_EQ(pool.Allocate(1, exhausted), Status::NoSpace());

    ASSERT_TRUE(pool.Free(second).Success());
    ASSERT_TRUE(pool.Free(first).Success());
    VariableBufferPool::AllocationHandle complete;
    ASSERT_TRUE(pool.Allocate(500, complete).Success());
    EXPECT_EQ(complete.allocated_size, std::size_t{512});
    EXPECT_EQ(complete.offset, std::size_t{0});
}

TEST_F(VariableBufferPoolTest, HostPinnedPoolKeepsLocalAndDeviceAddresses)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("pinned", MemoryType::HOST_PINNED, 4096, 16).Success());

    ASSERT_NE(pool.GetLocalAddr(), nullptr);
    ASSERT_NE(pool.GetDeviceAddr(), nullptr);
    EXPECT_NE(pool.GetLocalAddr(), pool.GetDeviceAddr());
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::HOST_PINNED);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(pool.GetLocalAddr()) % 4096, std::uintptr_t{0});

    VariableBufferPool::AllocationHandle first;
    VariableBufferPool::AllocationHandle second;
    ASSERT_TRUE(pool.Allocate(65, first).Success());
    ASSERT_TRUE(pool.Allocate(129, second).Success());
    EXPECT_EQ(first.local_addr, pool.GetLocalAddr());
    EXPECT_EQ(first.device_addr, pool.GetDeviceAddr());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.local_addr) -
                  reinterpret_cast<std::uintptr_t>(first.local_addr),
              std::uintptr_t{128});
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.device_addr) -
                  reinterpret_cast<std::uintptr_t>(first.device_addr),
              std::uintptr_t{128});
}

TEST_F(VariableBufferPoolTest, DevicePoolZeroesReleasedAllocation)
{
    constexpr std::size_t allocationSize = 128;
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("device", MemoryType::ASCEND_DEVICE, allocationSize, 8, true).Success());
    EXPECT_EQ(pool.GetMemoryType(), MemoryType::ASCEND_DEVICE);
    EXPECT_EQ(pool.GetLocalAddr(), pool.GetDeviceAddr());

    VariableBufferPool::AllocationHandle first;
    ASSERT_TRUE(pool.Allocate(65, first).Success());
    ASSERT_EQ(aclrtMemset(first.local_addr, allocationSize, 0xAB, allocationSize), ACL_SUCCESS);
    ASSERT_TRUE(pool.Free(first).Success());

    VariableBufferPool::AllocationHandle second;
    ASSERT_TRUE(pool.Allocate(65, second).Success());
    EXPECT_EQ(second.local_addr, first.local_addr);
    std::array<std::uint8_t, allocationSize> host{};
    ASSERT_EQ(aclrtMemcpy(host.data(), host.size(), second.local_addr, allocationSize,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    for (const auto value : host) { EXPECT_EQ(value, 0); }
}

TEST_F(VariableBufferPoolTest, CoalescesFragmentedAdjacentAllocations)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("coalesce", MemoryType::HOST, 256, 16).Success());

    VariableBufferPool::AllocationHandle first;
    VariableBufferPool::AllocationHandle second;
    VariableBufferPool::AllocationHandle third;
    VariableBufferPool::AllocationHandle fourth;
    ASSERT_TRUE(pool.Allocate(64, first).Success());
    ASSERT_TRUE(pool.Allocate(64, second).Success());
    ASSERT_TRUE(pool.Allocate(64, third).Success());
    ASSERT_TRUE(pool.Allocate(64, fourth).Success());

    ASSERT_TRUE(pool.Free(first).Success());
    ASSERT_TRUE(pool.Free(third).Success());

    VariableBufferPool::AllocationHandle fragmented;
    EXPECT_EQ(pool.Allocate(128, fragmented), Status::NoSpace());

    ASSERT_TRUE(pool.Free(second).Success());
    VariableBufferPool::AllocationHandle merged;
    ASSERT_TRUE(pool.Allocate(128, merged).Success());
    EXPECT_EQ(merged.local_addr, first.local_addr);
}

TEST_F(VariableBufferPoolTest, FreeZeroesAndReusesHostMemory)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("zero", MemoryType::HOST, 128, 8, true).Success());

    VariableBufferPool::AllocationHandle first;
    ASSERT_TRUE(pool.Allocate(65, first).Success());
    std::memset(first.local_addr, 0xAB, first.allocated_size);
    ASSERT_TRUE(pool.Free(first).Success());

    VariableBufferPool::AllocationHandle second;
    ASSERT_TRUE(pool.Allocate(65, second).Success());
    EXPECT_EQ(second.local_addr, first.local_addr);
    const auto* bytes = static_cast<const std::uint8_t*>(second.local_addr);
    for (std::size_t i = 0; i < second.allocated_size; ++i) { EXPECT_EQ(bytes[i], 0); }
}

TEST_F(VariableBufferPoolTest, FreePreservesHostMemoryWhenZeroingDisabled)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("preserve", MemoryType::HOST, 128, 8).Success());

    VariableBufferPool::AllocationHandle first;
    ASSERT_TRUE(pool.Allocate(65, first).Success());
    std::memset(first.local_addr, 0xAB, first.allocated_size);
    ASSERT_TRUE(pool.Free(first).Success());

    VariableBufferPool::AllocationHandle second;
    ASSERT_TRUE(pool.Allocate(65, second).Success());
    EXPECT_EQ(second.local_addr, first.local_addr);
    const auto* bytes = static_cast<const std::uint8_t*>(second.local_addr);
    for (std::size_t i = 0; i < second.allocated_size; ++i) { EXPECT_EQ(bytes[i], 0xAB); }
}

TEST_F(VariableBufferPoolTest, AllowsExactFitWhenMetadataIsExhausted)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("metadata_exact", MemoryType::HOST, 128, 3).Success());

    VariableBufferPool::AllocationHandle first;
    VariableBufferPool::AllocationHandle second;
    ASSERT_TRUE(pool.Allocate(64, first).Success());
    ASSERT_TRUE(pool.Allocate(64, second).Success());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.local_addr) -
                  reinterpret_cast<std::uintptr_t>(first.local_addr),
              std::uintptr_t{64});

    VariableBufferPool::AllocationHandle exhausted;
    EXPECT_EQ(pool.Allocate(1, exhausted), Status::NoSpace());
    EXPECT_TRUE(pool.Free(second).Success());
    EXPECT_TRUE(pool.Free(first).Success());
}

TEST_F(VariableBufferPoolTest, RejectsSplitWhenMetadataIsExhaustedWithoutCorruption)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("metadata_split", MemoryType::HOST, 192, 3).Success());

    VariableBufferPool::AllocationHandle first;
    ASSERT_TRUE(pool.Allocate(64, first).Success());
    VariableBufferPool::AllocationHandle split;
    EXPECT_EQ(pool.Allocate(64, split), Status::NoSpace());

    VariableBufferPool::AllocationHandle exact;
    ASSERT_TRUE(pool.Allocate(128, exact).Success());
    EXPECT_TRUE(pool.Free(exact).Success());
    EXPECT_TRUE(pool.Free(first).Success());

    VariableBufferPool::AllocationHandle complete;
    ASSERT_TRUE(pool.Allocate(192, complete).Success());
    EXPECT_EQ(complete.local_addr, pool.GetLocalAddr());
}

TEST_F(VariableBufferPoolTest, FailedAllocationDoesNotOverwriteHandle)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("preserve_handle", MemoryType::HOST, 64, 8).Success());

    VariableBufferPool::AllocationHandle live;
    ASSERT_TRUE(pool.Allocate(64, live).Success());
    auto output = live;

    EXPECT_EQ(pool.Allocate(1, output), Status::NoSpace());
    EXPECT_EQ(output.allocation.offset, live.allocation.offset);
    EXPECT_EQ(output.allocation.metadata, live.allocation.metadata);
    EXPECT_EQ(output.requested_size, live.requested_size);
    EXPECT_EQ(output.allocated_size, live.allocated_size);
    EXPECT_EQ(output.offset, live.offset);
    EXPECT_EQ(output.local_addr, live.local_addr);
    EXPECT_EQ(output.device_addr, live.device_addr);

    ASSERT_TRUE(pool.Free(live).Success());
    EXPECT_EQ(pool.Allocate(128, output), Status::NoSpace());
    EXPECT_EQ(output.local_addr, live.local_addr);
}

TEST_F(VariableBufferPoolTest, RejectsInvalidAndRepeatedFree)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("validation", MemoryType::HOST, 128, 8).Success());

    VariableBufferPool::AllocationHandle empty;
    EXPECT_EQ(pool.Free(empty), Status::InvalidParam());

    VariableBufferPool::AllocationHandle handle;
    ASSERT_TRUE(pool.Allocate(64, handle).Success());

    auto invalid = handle;
    ++invalid.allocation.offset;
    EXPECT_EQ(pool.Free(invalid), Status::InvalidParam());
    EXPECT_TRUE(pool.Free(handle).Success());
    EXPECT_EQ(pool.Free(handle), Status::InvalidParam());
}

TEST_F(VariableBufferPoolTest, ResetAllowsReinitialization)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("first", MemoryType::HOST, 65, 8, false, 256).Success());
    EXPECT_EQ(pool.GetTotalSize(), std::size_t{256});
    pool.Reset();

    EXPECT_FALSE(pool.IsInitialized());
    EXPECT_EQ(pool.GetLocalAddr(), nullptr);
    EXPECT_EQ(pool.GetDeviceAddr(), nullptr);
    EXPECT_EQ(pool.GetTotalSize(), std::size_t{0});
    VariableBufferPool::AllocationHandle handle;
    EXPECT_EQ(pool.Allocate(64, handle), Status::Error());
    EXPECT_EQ(pool.Free(handle), Status::Error());
    EXPECT_TRUE(pool.Init("second", MemoryType::HOST, 65, 8).Success());
    EXPECT_EQ(pool.GetTotalSize(), std::size_t{128});
}

TEST_F(VariableBufferPoolTest, ConcurrentAllocateAndFree)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("concurrent", MemoryType::HOST, 4096, 128).Success());

    constexpr int kThreadCount = 4;
    constexpr int kOpsPerThread = 500;
    std::atomic<bool> failed{false};

    auto worker = [&pool, &failed]() {
        for (int i = 0; i < kOpsPerThread; ++i) {
            VariableBufferPool::AllocationHandle handle;
            auto status = pool.Allocate(64 + static_cast<std::size_t>(i % 4) * 64, handle);
            if (status.Failure()) {
                failed = true;
                return;
            }
            std::memset(handle.local_addr, 0xAB, handle.allocated_size);
            status = pool.Free(handle);
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

TEST_F(VariableBufferPoolTest, ConcurrentAllocateAndFreeWithZeroing)
{
    VariableBufferPool pool;
    ASSERT_TRUE(pool.Init("concurrent_zero", MemoryType::HOST, 4096, 128, true).Success());

    constexpr int threadCount = 4;
    constexpr int operationsPerThread = 200;
    std::atomic<bool> failed{false};

    auto worker = [&pool, &failed]() {
        for (int operation = 0; operation < operationsPerThread; ++operation) {
            VariableBufferPool::AllocationHandle handle;
            auto status = pool.Allocate(64 + static_cast<std::size_t>(operation % 4) * 64, handle);
            if (status.Failure()) {
                failed = true;
                return;
            }
            std::memset(handle.local_addr, 0xAB, handle.allocated_size);
            status = pool.Free(handle);
            if (status.Failure()) {
                failed = true;
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int thread = 0; thread < threadCount; ++thread) { threads.emplace_back(worker); }
    for (auto& thread : threads) { thread.join(); }
    EXPECT_FALSE(failed.load());
}

}  // namespace
}  // namespace UC
