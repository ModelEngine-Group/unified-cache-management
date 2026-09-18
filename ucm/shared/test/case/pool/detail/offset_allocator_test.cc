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
#include "pool/detail/offset_allocator.h"
#include <array>
#include <atomic>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace OffsetAllocator {
namespace {

static_assert(!std::is_copy_constructible<Allocator>::value);
static_assert(!std::is_copy_assignable<Allocator>::value);
static_assert(!std::is_move_constructible<Allocator>::value);
static_assert(!std::is_move_assignable<Allocator>::value);

TEST(OffsetAllocatorTest, RejectsInvalidConfiguration)
{
    EXPECT_THROW(Allocator(0, 8), std::invalid_argument);
    EXPECT_THROW(Allocator(64, 0), std::invalid_argument);
    EXPECT_THROW(Allocator(64, 1), std::invalid_argument);
    EXPECT_THROW(Allocator(64, 2), std::invalid_argument);
#ifdef USE_16_BIT_NODE_INDICES
    EXPECT_THROW(Allocator(64, static_cast<uint32>(std::numeric_limits<NodeIndex>::max()) + 1U),
                 std::invalid_argument);
#endif
}

TEST(OffsetAllocatorTest, RejectsZeroSizeWithoutConsumingMetadata)
{
    Allocator allocator(64, 8);
    const auto before = allocator.GetStorageReport();

    const auto allocation = allocator.Allocate(0);

    EXPECT_EQ(allocation.offset, NO_SPACE);
    EXPECT_EQ(allocation.nodeIndex, NO_SPACE_NODE_INDEX);
    const auto after = allocator.GetStorageReport();
    EXPECT_EQ(after.totalFreeSpace, before.totalFreeSpace);
    EXPECT_EQ(after.largestFreeRegion, before.largestFreeRegion);
}

TEST(OffsetAllocatorTest, RejectsOversizedAllocationWithoutChangingState)
{
    Allocator allocator(64, 8);
    const auto before = allocator.GetStorageReport();

    const auto allocation = allocator.Allocate(65);

    EXPECT_EQ(allocation.offset, NO_SPACE);
    EXPECT_EQ(allocation.nodeIndex, NO_SPACE_NODE_INDEX);
    const auto after = allocator.GetStorageReport();
    EXPECT_EQ(after.totalFreeSpace, before.totalFreeSpace);
    EXPECT_EQ(after.largestFreeRegion, before.largestFreeRegion);
}

TEST(OffsetAllocatorTest, RejectsInvalidAndMismatchedFree)
{
    Allocator allocator(64, 8);
    EXPECT_FALSE(allocator.Free({NO_SPACE, NO_SPACE_NODE_INDEX}));
    EXPECT_FALSE(allocator.Free({0, static_cast<NodeIndex>(8)}));

    const auto allocation = allocator.Allocate(16);
    ASSERT_NE(allocation.offset, NO_SPACE);

    auto mismatched = allocation;
    ++mismatched.offset;
    EXPECT_FALSE(allocator.Free(mismatched));
    EXPECT_EQ(allocator.GetAllocationSize(mismatched), uint32{0});
    EXPECT_TRUE(allocator.Free(allocation));
}

TEST(OffsetAllocatorTest, ReturnsSizeOnlyForLiveMatchingAllocation)
{
    Allocator allocator(64, 8);
    const auto allocation = allocator.Allocate(16);
    ASSERT_NE(allocation.offset, NO_SPACE);
    EXPECT_EQ(allocator.GetAllocationSize(allocation), uint32{16});

    auto mismatched = allocation;
    ++mismatched.offset;
    EXPECT_EQ(allocator.GetAllocationSize(mismatched), uint32{0});

    ASSERT_TRUE(allocator.Free(allocation));
    EXPECT_EQ(allocator.GetAllocationSize(allocation), uint32{0});
}

TEST(OffsetAllocatorTest, RejectsDuplicateFreeWithoutCorruptingCapacity)
{
    Allocator allocator(64, 8);
    const auto allocation = allocator.Allocate(16);
    ASSERT_NE(allocation.offset, NO_SPACE);

    ASSERT_TRUE(allocator.Free(allocation));
    const auto before = allocator.GetStorageReport();
    EXPECT_FALSE(allocator.Free(allocation));
    const auto after = allocator.GetStorageReport();

    EXPECT_EQ(before.totalFreeSpace, uint32{64});
    EXPECT_EQ(after.totalFreeSpace, before.totalFreeSpace);
    EXPECT_EQ(after.largestFreeRegion, before.largestFreeRegion);
}

TEST(OffsetAllocatorTest, CoalescesAdjacentRegionsAfterFree)
{
    Allocator allocator(64, 16);
    const auto first = allocator.Allocate(16);
    const auto second = allocator.Allocate(16);
    const auto third = allocator.Allocate(16);
    ASSERT_NE(first.offset, NO_SPACE);
    ASSERT_NE(second.offset, NO_SPACE);
    ASSERT_NE(third.offset, NO_SPACE);

    EXPECT_TRUE(allocator.Free(second));
    EXPECT_TRUE(allocator.Free(first));
    EXPECT_TRUE(allocator.Free(third));

    const auto report = allocator.GetStorageReport();
    EXPECT_EQ(report.totalFreeSpace, uint32{64});
    EXPECT_EQ(report.largestFreeRegion, uint32{64});

    const auto complete = allocator.Allocate(64);
    EXPECT_EQ(complete.offset, uint32{0});
    EXPECT_TRUE(allocator.Free(complete));
}

TEST(OffsetAllocatorTest, AllowsExactFitWhenMetadataIsExhausted)
{
    Allocator allocator(2, 3);

    const auto first = allocator.Allocate(1);
    const auto second = allocator.Allocate(1);

    EXPECT_EQ(first.offset, uint32{0});
    EXPECT_EQ(second.offset, uint32{1});
    EXPECT_EQ(allocator.GetStorageReport().totalFreeSpace, uint32{0});
    EXPECT_TRUE(allocator.Free(second));
    EXPECT_TRUE(allocator.Free(first));
}

TEST(OffsetAllocatorTest, RejectsSplitWithoutCorruptingState)
{
    Allocator allocator(3, 3);

    const auto first = allocator.Allocate(1);
    ASSERT_NE(first.offset, NO_SPACE);

    const auto split = allocator.Allocate(1);
    EXPECT_EQ(split.offset, NO_SPACE);
    EXPECT_EQ(allocator.GetStorageReport().totalFreeSpace, uint32{2});
    EXPECT_EQ(allocator.GetAllocationSize(first), uint32{1});

    const auto exact = allocator.Allocate(2);
    EXPECT_EQ(exact.offset, uint32{1});
    EXPECT_TRUE(allocator.Free(exact));
    EXPECT_TRUE(allocator.Free(first));
}

TEST(OffsetAllocatorTest, ReportsFreeSpaceWhenMetadataIsExhausted)
{
    Allocator allocator(3, 3);
    const auto first = allocator.Allocate(1);
    ASSERT_NE(first.offset, NO_SPACE);

    const auto before = allocator.GetStorageReport();
    EXPECT_EQ(before.totalFreeSpace, uint32{2});
    EXPECT_EQ(before.largestFreeRegion, uint32{2});

    EXPECT_EQ(allocator.Allocate(1).offset, NO_SPACE);
    const auto after = allocator.GetStorageReport();
    EXPECT_EQ(after.totalFreeSpace, before.totalFreeSpace);
    EXPECT_EQ(after.largestFreeRegion, before.largestFreeRegion);
    EXPECT_TRUE(allocator.Free(first));
}

TEST(OffsetAllocatorTest, ResetRestoresInitialState)
{
    Allocator allocator(64, 16);
    const auto first = allocator.Allocate(13);
    const auto second = allocator.Allocate(17);
    ASSERT_NE(first.offset, NO_SPACE);
    ASSERT_NE(second.offset, NO_SPACE);

    allocator.Reset();

    EXPECT_EQ(allocator.GetAllocationSize(first), uint32{0});
    const auto report = allocator.GetStorageReport();
    EXPECT_EQ(report.totalFreeSpace, uint32{64});
    EXPECT_EQ(report.largestFreeRegion, uint32{64});
    const auto complete = allocator.Allocate(64);
    EXPECT_EQ(complete.offset, uint32{0});
    EXPECT_TRUE(allocator.Free(complete));
}

TEST(OffsetAllocatorTest, FindsExactFitBehindLargerNodeWithoutMetadata)
{
    Allocator allocator(134, 4);
    const auto larger = allocator.Allocate(69);
    const auto guard = allocator.Allocate(1);
    const auto exact = allocator.Allocate(64);
    ASSERT_NE(larger.offset, NO_SPACE);
    ASSERT_NE(guard.offset, NO_SPACE);
    ASSERT_NE(exact.offset, NO_SPACE);

    ASSERT_TRUE(allocator.Free(exact));
    ASSERT_TRUE(allocator.Free(larger));

    const auto allocation = allocator.Allocate(64);
    EXPECT_EQ(allocation.offset, uint32{70});
}

TEST(OffsetAllocatorTest, FindsLowerBinExactFitWithoutMetadata)
{
    Allocator allocator(135, 4);
    const auto larger = allocator.Allocate(69);
    const auto guard = allocator.Allocate(1);
    const auto exact = allocator.Allocate(65);
    ASSERT_NE(larger.offset, NO_SPACE);
    ASSERT_NE(guard.offset, NO_SPACE);
    ASSERT_NE(exact.offset, NO_SPACE);

    ASSERT_TRUE(allocator.Free(exact));
    ASSERT_TRUE(allocator.Free(larger));

    const auto allocation = allocator.Allocate(65);
    EXPECT_EQ(allocation.offset, uint32{70});
}

TEST(OffsetAllocatorTest, IncludesEighthExactFitCandidate)
{
    constexpr uint32 largerCount = 7;
    constexpr uint32 arenaSize = largerCount * (69 + 1) + 64;
    Allocator allocator(arenaSize, 2 + largerCount * 2);
    std::array<Allocation, largerCount> larger{};
    std::array<Allocation, largerCount> guards{};

    for (uint32 index = 0; index < largerCount; ++index) {
        larger[index] = allocator.Allocate(69);
        guards[index] = allocator.Allocate(1);
        ASSERT_NE(larger[index].offset, NO_SPACE);
        ASSERT_NE(guards[index].offset, NO_SPACE);
    }
    const auto exact = allocator.Allocate(64);
    ASSERT_NE(exact.offset, NO_SPACE);
    ASSERT_TRUE(allocator.Free(exact));
    for (const auto allocation : larger) { ASSERT_TRUE(allocator.Free(allocation)); }

    const auto allocation = allocator.Allocate(64);
    EXPECT_EQ(allocation.offset, arenaSize - 64);
}

TEST(OffsetAllocatorTest, LimitsExactFitSearchToEightCandidates)
{
    constexpr uint32 largerCount = 8;
    constexpr uint32 arenaSize = largerCount * (69 + 1) + 64;
    Allocator allocator(arenaSize, 2 + largerCount * 2);
    std::array<Allocation, largerCount> larger{};
    std::array<Allocation, largerCount> guards{};

    for (uint32 index = 0; index < largerCount; ++index) {
        larger[index] = allocator.Allocate(69);
        guards[index] = allocator.Allocate(1);
        ASSERT_NE(larger[index].offset, NO_SPACE);
        ASSERT_NE(guards[index].offset, NO_SPACE);
    }
    const auto exact = allocator.Allocate(64);
    ASSERT_NE(exact.offset, NO_SPACE);
    ASSERT_TRUE(allocator.Free(exact));
    for (const auto allocation : larger) { ASSERT_TRUE(allocator.Free(allocation)); }

    EXPECT_EQ(allocator.Allocate(64).offset, NO_SPACE);
}

TEST(OffsetAllocatorTest, FallsBackToLowerBinAfterNormalSearchFails)
{
    Allocator allocator(69, 8);

    const auto allocation = allocator.Allocate(67);

    ASSERT_NE(allocation.offset, NO_SPACE);
    EXPECT_EQ(allocation.offset, uint32{0});
    EXPECT_TRUE(allocator.Free(allocation));
    EXPECT_EQ(allocator.GetStorageReport().totalFreeSpace, uint32{69});
}

TEST(OffsetAllocatorTest, FallsBackAcrossTopBinBoundary)
{
    Allocator allocator(127, 8);

    const auto allocation = allocator.Allocate(127);

    ASSERT_NE(allocation.offset, NO_SPACE);
    EXPECT_EQ(allocation.offset, uint32{0});
    EXPECT_TRUE(allocator.Free(allocation));
}

TEST(OffsetAllocatorTest, UnlinksLowerBinFallbackFromMiddleOfList)
{
    Allocator allocator(135, 8);
    const auto larger = allocator.Allocate(69);
    const auto guard = allocator.Allocate(1);
    const auto smaller = allocator.Allocate(65);
    ASSERT_NE(larger.offset, NO_SPACE);
    ASSERT_NE(guard.offset, NO_SPACE);
    ASSERT_NE(smaller.offset, NO_SPACE);

    ASSERT_TRUE(allocator.Free(larger));
    ASSERT_TRUE(allocator.Free(smaller));

    const auto fallback = allocator.Allocate(67);
    ASSERT_NE(fallback.offset, NO_SPACE);
    EXPECT_EQ(fallback.offset, uint32{0});
    EXPECT_TRUE(allocator.Free(fallback));
    EXPECT_TRUE(allocator.Free(guard));
    EXPECT_EQ(allocator.GetStorageReport().totalFreeSpace, uint32{135});
}

TEST(OffsetAllocatorTest, LimitsLowerBinFallbackToEightNodes)
{
    constexpr uint32 smallNodeCount = 8;
    constexpr uint32 arenaSize = 69 + 1 + smallNodeCount * (65 + 1);
    Allocator allocator(arenaSize, 40);

    const auto larger = allocator.Allocate(69);
    const auto firstGuard = allocator.Allocate(1);
    std::array<Allocation, smallNodeCount> smaller{};
    std::array<Allocation, smallNodeCount> guards{};
    ASSERT_NE(larger.offset, NO_SPACE);
    ASSERT_NE(firstGuard.offset, NO_SPACE);
    for (uint32 index = 0; index < smallNodeCount; ++index) {
        smaller[index] = allocator.Allocate(65);
        guards[index] = allocator.Allocate(1);
        ASSERT_NE(smaller[index].offset, NO_SPACE);
        ASSERT_NE(guards[index].offset, NO_SPACE);
    }

    ASSERT_TRUE(allocator.Free(larger));
    for (const auto allocation : smaller) { ASSERT_TRUE(allocator.Free(allocation)); }

    const auto fallback = allocator.Allocate(67);
    EXPECT_EQ(fallback.offset, NO_SPACE);
}

TEST(OffsetAllocatorTest, IncludesEighthNodeInLowerBinFallback)
{
    constexpr uint32 smallNodeCount = 7;
    constexpr uint32 arenaSize = 69 + 1 + smallNodeCount * (65 + 1);
    Allocator allocator(arenaSize, 32);

    const auto larger = allocator.Allocate(69);
    const auto firstGuard = allocator.Allocate(1);
    std::array<Allocation, smallNodeCount> smaller{};
    std::array<Allocation, smallNodeCount> guards{};
    ASSERT_NE(larger.offset, NO_SPACE);
    ASSERT_NE(firstGuard.offset, NO_SPACE);
    for (uint32 index = 0; index < smallNodeCount; ++index) {
        smaller[index] = allocator.Allocate(65);
        guards[index] = allocator.Allocate(1);
        ASSERT_NE(smaller[index].offset, NO_SPACE);
        ASSERT_NE(guards[index].offset, NO_SPACE);
    }

    ASSERT_TRUE(allocator.Free(larger));
    for (const auto allocation : smaller) { ASSERT_TRUE(allocator.Free(allocation)); }

    const auto fallback = allocator.Allocate(67);
    EXPECT_EQ(fallback.offset, uint32{0});
}

TEST(OffsetAllocatorTest, FallsBackAtUint32UpperBoundary)
{
    constexpr uint32 maximum = std::numeric_limits<uint32>::max();
    Allocator allocator(maximum, 3);

    const auto allocation = allocator.Allocate(maximum);

    ASSERT_NE(allocation.offset, NO_SPACE);
    EXPECT_EQ(allocation.offset, uint32{0});
    EXPECT_TRUE(allocator.Free(allocation));
}

TEST(OffsetAllocatorTest, RandomizedOperationsPreserveAllocatorInvariants)
{
    struct LiveAllocation {
        Allocation allocation;
        uint32 size;
    };

    constexpr uint32 arenaSize = 4096;
    Allocator allocator(arenaSize, 512);
    std::mt19937 random(0x5EEDU);
    std::vector<LiveAllocation> live;

    for (uint32 operation = 0; operation < 5000; ++operation) {
        const bool shouldAllocate = live.empty() || random() % 100 < 60;
        if (shouldAllocate) {
            const uint32 size = random() % 256 + 1;
            const auto allocation = allocator.Allocate(size);
            if (allocation.offset == NO_SPACE) { continue; }

            ASSERT_LE(allocation.offset, arenaSize - size);
            for (const auto& current : live) {
                const bool overlaps =
                    allocation.offset < current.allocation.offset + current.size &&
                    current.allocation.offset < allocation.offset + size;
                ASSERT_FALSE(overlaps);
            }
            live.push_back({allocation, size});
            EXPECT_EQ(allocator.GetAllocationSize(allocation), size);
        } else {
            const auto index = static_cast<std::size_t>(random()) % live.size();
            ASSERT_TRUE(allocator.Free(live[index].allocation));
            live[index] = live.back();
            live.pop_back();
        }
    }

    for (const auto& current : live) { ASSERT_TRUE(allocator.Free(current.allocation)); }
    const auto report = allocator.GetStorageReport();
    EXPECT_EQ(report.totalFreeSpace, arenaSize);
    EXPECT_EQ(report.largestFreeRegion, arenaSize);
}

TEST(OffsetAllocatorTest, ConcurrentAllocateAndFree)
{
    constexpr uint32 arenaSize = 4096;
    constexpr int threadCount = 4;
    constexpr int operationsPerThread = 500;
    Allocator allocator(arenaSize, 128);
    std::atomic<bool> failed{false};

    auto worker = [&allocator, &failed]() {
        for (int operation = 0; operation < operationsPerThread; ++operation) {
            const uint32 size = static_cast<uint32>(operation % 4 + 1);
            const auto allocation = allocator.Allocate(size);
            if (allocation.offset == NO_SPACE || allocator.GetAllocationSize(allocation) != size ||
                !allocator.Free(allocation)) {
                failed = true;
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int thread = 0; thread < threadCount; ++thread) { threads.emplace_back(worker); }
    for (auto& thread : threads) { thread.join(); }

    EXPECT_FALSE(failed.load());
    EXPECT_EQ(allocator.GetStorageReport().totalFreeSpace, arenaSize);
}

}  // namespace
}  // namespace OffsetAllocator
