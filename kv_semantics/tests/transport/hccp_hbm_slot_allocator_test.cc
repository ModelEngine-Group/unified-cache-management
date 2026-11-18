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
#ifdef UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER

#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <unordered_set>
#include <vector>
#include "hccp_kv_over_roce/hbm_slot_allocator.h"

namespace kv::test {
namespace {

constexpr std::size_t kTestSlotBytes = 512;
constexpr std::uint32_t kTestCapacity = 4;

std::vector<std::uint8_t> ReadSlot(HbmSlotAllocator& allocator, std::uint32_t slotId,
                                   std::size_t size)
{
    std::vector<std::uint8_t> data(size, 0xAA);
    EXPECT_EQ(aclrtMemcpy(data.data(), data.size(), allocator.SlotAddress(slotId), size,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              ACL_SUCCESS);
    return data;
}

class HccpSlotAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(aclrtGetDevice(&device_), ACL_SUCCESS); }

    std::int32_t device_{0};
};

TEST_F(HccpSlotAllocatorTest, InitRejectsZeroCapacityWithoutDevice)
{
    HbmSlotAllocator allocator(-1, kTestSlotBytes, 0);
    EXPECT_EQ(allocator.Init().code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(HccpSlotAllocatorTest, InitRejectsZeroSlotBytes)
{
    HbmSlotAllocator allocator(-1, 0, kTestCapacity);
    EXPECT_EQ(allocator.Init().code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(HccpSlotAllocatorTest, InitRejectsSizeOverflow)
{
    HbmSlotAllocator allocator(-1, std::numeric_limits<std::size_t>::max(), 2);
    EXPECT_EQ(allocator.Init().code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(HccpSlotAllocatorTest, PreInitOperationsFail)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    EXPECT_EQ(allocator.deviceId(), device_);
    EXPECT_EQ(allocator.slotBytes(), kTestSlotBytes);
    EXPECT_EQ(allocator.capacity(), kTestCapacity);

    std::uint32_t slotId = 0;
    EXPECT_EQ(allocator.Acquire(slotId).code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(allocator.Release(0).code, StatusCode::NOT_INITIALIZED);
    const std::uint8_t payload = 0;
    EXPECT_EQ(allocator.Write(0, &payload, sizeof(payload)).code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(allocator.Clear(0).code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(allocator.SlotAddress(0), nullptr);
}

TEST_F(HccpSlotAllocatorTest, InitIsIdempotent)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());
    ASSERT_TRUE(allocator.Init().ok());

    std::uint32_t slotId = 0;
    ASSERT_TRUE(allocator.Acquire(slotId).ok());
    EXPECT_NE(allocator.SlotAddress(slotId), nullptr);
}

TEST_F(HccpSlotAllocatorTest, AcquireReturnsUniqueIdsUntilExhausted)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());

    std::vector<std::uint32_t> slotIds;
    for (std::uint32_t index = 0; index < kTestCapacity; ++index) {
        std::uint32_t slotId = 0;
        ASSERT_TRUE(allocator.Acquire(slotId).ok());
        slotIds.push_back(slotId);
    }
    std::sort(slotIds.begin(), slotIds.end());
    for (std::uint32_t index = 0; index < kTestCapacity; ++index) {
        EXPECT_EQ(slotIds[index], index);
    }

    std::uint32_t extra = 0;
    EXPECT_EQ(allocator.Acquire(extra).code, StatusCode::RESOURCE_BUSY);
    for (const auto slotId : slotIds) { EXPECT_TRUE(allocator.Release(slotId).ok()); }
}

TEST_F(HccpSlotAllocatorTest, ReleaseRejectsInvalidId)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());
    EXPECT_EQ(allocator.Release(kTestCapacity).code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(HccpSlotAllocatorTest, ReleaseRejectsDoubleRelease)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());

    std::uint32_t slotId = 0;
    ASSERT_TRUE(allocator.Acquire(slotId).ok());
    ASSERT_TRUE(allocator.Release(slotId).ok());
    EXPECT_EQ(allocator.Release(slotId).code, StatusCode::INVALID_ARGUMENT);

    // The rejected release must not corrupt the free list: the whole capacity is
    // still acquirable exactly once.
    std::vector<std::uint32_t> slotIds;
    for (std::uint32_t index = 0; index < kTestCapacity; ++index) {
        std::uint32_t next = 0;
        ASSERT_TRUE(allocator.Acquire(next).ok());
        slotIds.push_back(next);
    }
    const std::unordered_set<std::uint32_t> unique(slotIds.begin(), slotIds.end());
    EXPECT_EQ(unique.size(), kTestCapacity);
}

TEST_F(HccpSlotAllocatorTest, WriteAndClearRequireAcquiredSlot)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());

    const std::vector<std::uint8_t> payload(64, 0x11);
    EXPECT_EQ(allocator.Write(0, payload.data(), payload.size()).code,
              StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(allocator.Clear(0).code, StatusCode::INVALID_ARGUMENT);

    std::uint32_t slotId = 0;
    ASSERT_TRUE(allocator.Acquire(slotId).ok());
    ASSERT_TRUE(allocator.Write(slotId, payload.data(), payload.size()).ok());
    ASSERT_TRUE(allocator.Clear(slotId).ok());
    ASSERT_TRUE(allocator.Release(slotId).ok());

    EXPECT_EQ(allocator.Write(slotId, payload.data(), payload.size()).code,
              StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(allocator.Clear(slotId).code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(HccpSlotAllocatorTest, SlotAddressUsesSlotStride)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());

    auto* first = static_cast<std::uint8_t*>(allocator.SlotAddress(0));
    ASSERT_NE(first, nullptr);
    for (std::uint32_t index = 1; index < kTestCapacity; ++index) {
        auto* address = static_cast<std::uint8_t*>(allocator.SlotAddress(index));
        ASSERT_NE(address, nullptr);
        EXPECT_EQ(address, first + static_cast<std::size_t>(index) * kTestSlotBytes);
    }
    EXPECT_EQ(allocator.SlotAddress(kTestCapacity), nullptr);
    EXPECT_EQ(allocator.SlotAddress(0), first);
}

TEST_F(HccpSlotAllocatorTest, WriteReadBackAndBoundary)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());

    std::uint32_t slotId = 0;
    ASSERT_TRUE(allocator.Acquire(slotId).ok());

    std::vector<std::uint8_t> payload(kTestSlotBytes, 0x5A);
    ASSERT_TRUE(allocator.Write(slotId, payload.data(), payload.size()).ok());
    EXPECT_EQ(ReadSlot(allocator, slotId, kTestSlotBytes), payload);

    std::vector<std::uint8_t> partial(64, 0x11);
    ASSERT_TRUE(allocator.Clear(slotId).ok());
    ASSERT_TRUE(allocator.Write(slotId, partial.data(), partial.size()).ok());
    const auto read = ReadSlot(allocator, slotId, kTestSlotBytes);
    ASSERT_EQ(read.size(), kTestSlotBytes);
    EXPECT_EQ(read[0], 0x11);
    EXPECT_EQ(read[partial.size() - 1], 0x11);
    EXPECT_EQ(read[partial.size()], 0x00);

    EXPECT_EQ(allocator.Write(slotId, nullptr, 4).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(allocator.Write(slotId, partial.data(), 0).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(allocator.Write(slotId, payload.data(), kTestSlotBytes + 1).code,
              StatusCode::INVALID_ARGUMENT);
}

TEST_F(HccpSlotAllocatorTest, ClearZeroesWholeSlot)
{
    HbmSlotAllocator allocator(device_, kTestSlotBytes, kTestCapacity);
    ASSERT_TRUE(allocator.Init().ok());

    std::uint32_t slotId = 0;
    ASSERT_TRUE(allocator.Acquire(slotId).ok());
    const std::vector<std::uint8_t> payload(kTestSlotBytes, 0xFF);
    ASSERT_TRUE(allocator.Write(slotId, payload.data(), payload.size()).ok());
    ASSERT_TRUE(allocator.Clear(slotId).ok());

    const auto read = ReadSlot(allocator, slotId, kTestSlotBytes);
    EXPECT_TRUE(std::all_of(read.begin(), read.end(), [](std::uint8_t byte) { return byte == 0; }));
}

TEST_F(HccpSlotAllocatorTest, ConcurrentAcquireIsUnique)
{
    constexpr std::uint32_t kCapacity = 64;
    constexpr std::uint32_t kThreads = 8;
    constexpr std::uint32_t kPerThread = kCapacity / kThreads;

    HbmSlotAllocator allocator(device_, kTestSlotBytes, kCapacity);
    ASSERT_TRUE(allocator.Init().ok());

    std::vector<std::vector<std::uint32_t>> acquired(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (std::uint32_t thread = 0; thread < kThreads; ++thread) {
        threads.emplace_back([&allocator, &acquired, thread]() {
            for (std::uint32_t index = 0; index < kPerThread; ++index) {
                std::uint32_t slotId = 0;
                if (allocator.Acquire(slotId).ok()) { acquired[thread].push_back(slotId); }
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }

    std::unordered_set<std::uint32_t> unique;
    for (const auto& ids : acquired) {
        for (const auto slotId : ids) { unique.insert(slotId); }
    }
    EXPECT_EQ(unique.size(), kCapacity);

    std::uint32_t extra = 0;
    EXPECT_EQ(allocator.Acquire(extra).code, StatusCode::RESOURCE_BUSY);

    for (const auto slotId : unique) { EXPECT_TRUE(allocator.Release(slotId).ok()); }
    for (std::uint32_t index = 0; index < kCapacity; ++index) {
        std::uint32_t slotId = 0;
        EXPECT_TRUE(allocator.Acquire(slotId).ok());
    }
}

}  // namespace
}  // namespace kv::test

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
