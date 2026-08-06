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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include "reply_service.h"
#include "trans/device.h"

namespace UC::Dram {
namespace {

constexpr std::uint32_t kSlotSize = 64;

class ReplyServiceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        auto status = device_.Init();
        deviceRuntimeOwned_ = status.Success();
        ASSERT_TRUE(deviceRuntimeOwned_ || status == Status::DuplicateKey()) << status.ToString();
        status = device_.Setup(0);
        ASSERT_TRUE(status.Success()) << status.ToString();
    }

    static void TearDownTestSuite()
    {
        if (!deviceRuntimeOwned_) { return; }
        EXPECT_TRUE(device_.Reset(0).Success());
        EXPECT_TRUE(device_.Finalize().Success());
        deviceRuntimeOwned_ = false;
    }

    inline static Trans::Device device_;
    inline static bool deviceRuntimeOwned_{false};
};

RequestToken Token(NodeId node, RequestId request)
{
    return RequestToken{node, kDefaultLaneId, 1, request};
}

Expected<std::unique_ptr<ReplyService>> CreateService(std::size_t slots,
                                                      std::uint32_t slotSize = kSlotSize)
{
    return ReplyService::Create(ReplyService::Options{
        slotSize,
        slots,
        std::chrono::microseconds{100},
        [](NodeId, NodeEvent) {},
    });
}

TEST_F(ReplyServiceTest, SharesSlotsAcrossRequestsAndValidatesOwnership)
{
    auto created = CreateService(3);
    ASSERT_TRUE(created);
    auto service = std::move(created).Value();
    ASSERT_TRUE(service->Start().Success());

    const auto first_token = Token(11, 1);
    const auto second_token = Token(22, 2);
    const auto third_token = Token(11, 3);
    auto first = service->Acquire(first_token, OpType::LOOKUP, 1);
    auto second = service->Acquire(second_token, OpType::DUMP, 1);
    auto third = service->Acquire(third_token, OpType::LOAD, 1);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(third);
    EXPECT_EQ(first.Value().slot_index, std::uint32_t{0});
    EXPECT_EQ(second.Value().slot_index, std::uint32_t{1});
    EXPECT_EQ(third.Value().slot_index, std::uint32_t{2});
    EXPECT_NE(second.Value().local_addr, first.Value().local_addr);
    const auto replyMemory = service->MemoryRegion();
    const auto replyBegin = reinterpret_cast<std::uintptr_t>(replyMemory.address);
    const auto secondAddress = reinterpret_cast<std::uintptr_t>(second.Value().local_addr);
    EXPECT_GE(secondAddress, replyBegin);
    EXPECT_LT(secondAddress, replyBegin + replyMemory.length);
    EXPECT_EQ(second.Value().length, kSlotSize);
    EXPECT_EQ(service->Available(), std::size_t{0});
    auto exhausted = service->Acquire(Token(33, 4), OpType::LOOKUP, 1);
    ASSERT_FALSE(exhausted);
    EXPECT_EQ(exhausted.Error(), Status::Error());

    EXPECT_EQ(service->Release(Token(22, 999), second.Value()), Status::InvalidParam());
    EXPECT_EQ(service->Available(), std::size_t{0});
    ASSERT_TRUE(service->Release(second_token, second.Value()).Success());

    auto reused = service->Acquire(Token(33, 4), OpType::LOOKUP, 1);
    ASSERT_TRUE(reused);
    EXPECT_EQ(reused.Value().slot_index, second.Value().slot_index);
    // No slot generation is required: the old token cannot release a slot after reuse.
    EXPECT_EQ(service->Release(second_token, second.Value()), Status::InvalidParam());
    EXPECT_TRUE(service->Release(Token(33, 4), reused.Value()).Success());
    EXPECT_TRUE(service->Release(first_token, first.Value()).Success());
    EXPECT_TRUE(service->Release(third_token, third.Value()).Success());
    service->Shutdown();
}

TEST_F(ReplyServiceTest, RejectsReplyPayloadLargerThanSlot)
{
    auto created = CreateService(1, 1);
    ASSERT_TRUE(created);
    auto service = std::move(created).Value();
    ASSERT_TRUE(service->Start().Success());
    EXPECT_FALSE(service->Acquire(Token(1, 1), OpType::LOOKUP, 1));
    service->Shutdown();
}

TEST_F(ReplyServiceTest, ShutdownStopsNewLeases)
{
    auto created = CreateService(1);
    ASSERT_TRUE(created);
    auto service = std::move(created).Value();
    ASSERT_TRUE(service->Start().Success());
    service->Shutdown();
    EXPECT_FALSE(service->Acquire(Token(1, 1), OpType::LOOKUP, 1));
}

TEST_F(ReplyServiceTest, SupportsConcurrentLeases)
{
    constexpr std::size_t kSlotCount = 8;
    constexpr std::size_t kOwnerCount = 8;
    constexpr std::size_t kIterations = 1000;
    auto created = CreateService(kSlotCount);
    ASSERT_TRUE(created);
    auto service = std::move(created).Value();
    ASSERT_TRUE(service->Start().Success());

    std::atomic<bool> failed{false};
    std::vector<std::thread> owners;
    owners.reserve(kOwnerCount);
    for (std::size_t owner = 0; owner < kOwnerCount; ++owner) {
        owners.emplace_back([&, owner] {
            for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
                const auto token = Token(owner + 1, owner * kIterations + iteration + 1);
                auto slot = service->Acquire(token, OpType::LOOKUP, 1);
                if (!slot || service->Release(token, slot.Value()).Failure()) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (auto& owner : owners) { owner.join(); }

    EXPECT_FALSE(failed.load(std::memory_order_relaxed));
    EXPECT_EQ(service->Available(), kSlotCount);
    service->Shutdown();
}

}  // namespace
}  // namespace UC::Dram
