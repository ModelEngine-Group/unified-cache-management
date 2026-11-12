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
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <utility>
#include <vector>
#include "memory/dma/dma_engine.h"
#include "memory/dma/dma_engine_registry.h"
#include "status/status.h"

namespace UC::Transfer {
namespace {

constexpr std::size_t kBlockSize = 4096;
constexpr std::size_t kMaxLength = 64 * 1024;
constexpr int32_t kDeviceId = 0;

std::vector<std::uint8_t> MakePattern(std::size_t size, std::uint8_t seed)
{
    std::vector<std::uint8_t> bytes(size);
    for (std::size_t i = 0; i < size; ++i) { bytes[i] = static_cast<std::uint8_t>(seed + i); }
    return bytes;
}

DmaEngine::DmaJob MakeDmaJob(std::vector<const void*> sources, std::vector<void*> destinations,
                             std::vector<std::size_t> sizes)
{
    DmaEngine::DmaJob job;
    job.sources = std::move(sources);
    job.destinations = std::move(destinations);
    job.sizes = std::move(sizes);
    return job;
}

class SimuDmaEngineTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        auto created = DmaEngineRegistry::Instance().Create(DmaEngineKind::Naive);
        ASSERT_TRUE(created.HasValue()) << created.Error().ToString();
        engine_.reset(*created);
        const DmaEngine::InitParam param{kDeviceId, kMaxLength};
        ASSERT_TRUE(engine_->Setup(param).Success());
    }

    void TearDown() override { engine_.reset(); }

    std::vector<std::pair<std::size_t, bool>> DrainFinished(std::size_t expectedCount)
    {
        std::vector<std::pair<std::size_t, bool>> collected;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (collected.size() < expectedCount && std::chrono::steady_clock::now() < deadline) {
            auto finished = engine_->GetFinished();
            if (finished.HasValue()) {
                collected.insert(collected.end(), finished.Value().begin(), finished.Value().end());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return collected;
    }

    std::unique_ptr<DmaEngine> engine_;
};

TEST(SimuDmaEngineSetupTest, SetupRejectsZeroMaxLength)
{
    auto created = DmaEngineRegistry::Instance().Create(DmaEngineKind::Naive);
    ASSERT_TRUE(created.HasValue());
    std::unique_ptr<DmaEngine> engine(*created);
    const DmaEngine::InitParam param{kDeviceId, 0};
    EXPECT_EQ(engine->Setup(param), Status::InvalidParam());
}

TEST(SimuDmaEngineSetupTest, OperationsBeforeSetupAreRejected)
{
    auto created = DmaEngineRegistry::Instance().Create(DmaEngineKind::Naive);
    ASSERT_TRUE(created.HasValue());
    std::unique_ptr<DmaEngine> engine(*created);
    std::vector<std::uint8_t> host(kBlockSize);
    std::vector<std::uint8_t> device(kBlockSize);
    EXPECT_EQ(engine->SubmitH2DBatch(1, MakeDmaJob({host.data()}, {device.data()}, {kBlockSize})),
              Status::Unsupported());
    EXPECT_EQ(engine->SubmitD2HBatch(1, MakeDmaJob({host.data()}, {device.data()}, {kBlockSize})),
              Status::Unsupported());
    auto finished = engine->GetFinished();
    ASSERT_TRUE(!finished.HasValue());
    EXPECT_EQ(finished.Error(), Status::Unsupported());
    EXPECT_EQ(engine->Wait(1), Status::Unsupported());
    EXPECT_EQ(engine->Synchronize(), Status::Unsupported());

    /* the rejections leave the engine pristine: a proper Setup revives it */
    ASSERT_TRUE(engine->Setup(DmaEngine::InitParam{kDeviceId, kMaxLength}).Success());
    EXPECT_TRUE(engine->SubmitH2DBatch(2, MakeDmaJob({host.data()}, {device.data()}, {kBlockSize}))
                    .Success());
    EXPECT_EQ(engine->Wait(2), Status::Ok());
}

TEST(SimuDmaEngineSetupTest, SetupReEntryIsRejectedWithoutDisturbingTheEngine)
{
    auto created = DmaEngineRegistry::Instance().Create(DmaEngineKind::Naive);
    ASSERT_TRUE(created.HasValue());
    std::unique_ptr<DmaEngine> engine(*created);
    ASSERT_TRUE(engine->Setup(DmaEngine::InitParam{kDeviceId, kMaxLength}).Success());
    EXPECT_EQ(engine->Setup(DmaEngine::InitParam{kDeviceId, kBlockSize}), Status::Unsupported());

    /* the rejected re-entry refreshes nothing: the original limit stays in force */
    std::vector<std::uint8_t> host(kMaxLength);
    std::vector<std::uint8_t> device(kMaxLength);
    EXPECT_EQ(
        engine->SubmitH2DBatch(1, MakeDmaJob({host.data()}, {device.data()}, {kMaxLength + 1})),
        Status::InvalidParam());
    const auto pattern = MakePattern(kBlockSize + 1, 0x70);
    EXPECT_TRUE(
        engine->SubmitH2DBatch(1, MakeDmaJob({pattern.data()}, {device.data()}, {kBlockSize + 1}))
            .Success());
    EXPECT_EQ(engine->Wait(1), Status::Ok());
    EXPECT_EQ(std::memcmp(device.data(), pattern.data(), pattern.size()), 0);
}

TEST_F(SimuDmaEngineTest, MalformedJobsAreRejectedWithoutOccupyingTheJobId)
{
    std::vector<std::uint8_t> host(kBlockSize);
    std::vector<std::uint8_t> device(kBlockSize);

    EXPECT_EQ(engine_->SubmitH2DBatch(1, DmaEngine::DmaJob{}), Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitH2DBatch(1, MakeDmaJob({host.data(), host.data()}, {device.data()},
                                                    {kBlockSize, kBlockSize})),
              Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitH2DBatch(1, MakeDmaJob({nullptr}, {device.data()}, {kBlockSize})),
              Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitH2DBatch(1, MakeDmaJob({host.data()}, {nullptr}, {kBlockSize})),
              Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitH2DBatch(1, MakeDmaJob({host.data()}, {device.data()}, {0})),
              Status::InvalidParam());
    EXPECT_EQ(engine_->SubmitD2HBatch(1, MakeDmaJob({host.data()}, {device.data()}, {0})),
              Status::InvalidParam());
    EXPECT_EQ(
        engine_->SubmitH2DBatch(1, MakeDmaJob({host.data()}, {device.data()}, {kMaxLength + 1})),
        Status::InvalidParam());

    EXPECT_TRUE(engine_->SubmitH2DBatch(1, MakeDmaJob({host.data()}, {device.data()}, {kBlockSize}))
                    .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
}

TEST_F(SimuDmaEngineTest, SegmentAtMaxLengthIsAccepted)
{
    const auto pattern = MakePattern(kMaxLength, 0x20);
    std::vector<std::uint8_t> device(kMaxLength);
    EXPECT_TRUE(
        engine_->SubmitH2DBatch(1, MakeDmaJob({pattern.data()}, {device.data()}, {kMaxLength}))
            .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
    EXPECT_EQ(std::memcmp(device.data(), pattern.data(), pattern.size()), 0);
}

TEST_F(SimuDmaEngineTest, DuplicateJobIdIsRejectedWhileInFlightAndReusableAfterConsumption)
{
    const auto pattern = MakePattern(kBlockSize, 0x30);
    std::vector<std::uint8_t> device(kBlockSize);
    const auto job = MakeDmaJob({pattern.data()}, {device.data()}, {kBlockSize});

    EXPECT_TRUE(engine_->SubmitH2DBatch(7, job).Success());
    EXPECT_EQ(engine_->SubmitD2HBatch(7, job), Status::DuplicateKey());

    EXPECT_EQ(engine_->Wait(7), Status::Ok());
    EXPECT_EQ(std::memcmp(device.data(), pattern.data(), pattern.size()), 0);

    EXPECT_TRUE(engine_->SubmitD2HBatch(7, job).Success());
    const auto collected = DrainFinished(1);
    ASSERT_EQ(collected.size(), std::size_t{1});
    EXPECT_EQ(collected[0].first, std::size_t{7});
    EXPECT_TRUE(collected[0].second);

    EXPECT_TRUE(engine_->SubmitH2DBatch(7, job).Success());
    EXPECT_EQ(engine_->Wait(7), Status::Ok());
}

TEST_F(SimuDmaEngineTest, H2DRoundTripAcrossSegments)
{
    const auto first = MakePattern(kBlockSize, 0x00);
    const auto second = MakePattern(kBlockSize / 2, 0x40);
    const auto third = MakePattern(kBlockSize, 0x80);
    std::vector<std::uint8_t> deviceFirst(kBlockSize);
    std::vector<std::uint8_t> deviceSecond(kBlockSize / 2);
    std::vector<std::uint8_t> deviceThird(kBlockSize);
    EXPECT_TRUE(engine_
                    ->SubmitH2DBatch(
                        1, MakeDmaJob({first.data(), second.data(), third.data()},
                                      {deviceFirst.data(), deviceSecond.data(), deviceThird.data()},
                                      {kBlockSize, kBlockSize / 2, kBlockSize}))
                    .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
    EXPECT_EQ(std::memcmp(deviceFirst.data(), first.data(), first.size()), 0);
    EXPECT_EQ(std::memcmp(deviceSecond.data(), second.data(), second.size()), 0);
    EXPECT_EQ(std::memcmp(deviceThird.data(), third.data(), third.size()), 0);
}

TEST_F(SimuDmaEngineTest, D2HRoundTripAcrossSegments)
{
    const auto first = MakePattern(kBlockSize, 0x10);
    const auto second = MakePattern(kBlockSize / 2, 0x50);
    const auto third = MakePattern(kBlockSize, 0x90);
    std::vector<std::uint8_t> hostFirst(kBlockSize);
    std::vector<std::uint8_t> hostSecond(kBlockSize / 2);
    std::vector<std::uint8_t> hostThird(kBlockSize);
    EXPECT_TRUE(
        engine_
            ->SubmitD2HBatch(2, MakeDmaJob({first.data(), second.data(), third.data()},
                                           {hostFirst.data(), hostSecond.data(), hostThird.data()},
                                           {kBlockSize, kBlockSize / 2, kBlockSize}))
            .Success());
    EXPECT_EQ(engine_->Wait(2), Status::Ok());
    EXPECT_EQ(std::memcmp(hostFirst.data(), first.data(), first.size()), 0);
    EXPECT_EQ(std::memcmp(hostSecond.data(), second.data(), second.size()), 0);
    EXPECT_EQ(std::memcmp(hostThird.data(), third.data(), third.size()), 0);
}

TEST_F(SimuDmaEngineTest, PrerequisiteDoesNotBlockSubmission)
{
    const auto pattern = MakePattern(kBlockSize, 0xA0);
    std::vector<std::uint8_t> device(kBlockSize);
    DmaEngine::DmaJob job = MakeDmaJob({pattern.data()}, {device.data()}, {kBlockSize});
    job.prerequisite = 0xDEADBEEF; /* no real stream exists in simu: read and ignored */
    EXPECT_TRUE(engine_->SubmitH2DBatch(1, job).Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
    EXPECT_EQ(std::memcmp(device.data(), pattern.data(), pattern.size()), 0);
}

TEST_F(SimuDmaEngineTest, GetFinishedReportsEachCompletionExactlyOnce)
{
    const auto pattern = MakePattern(kBlockSize, 0x55);
    std::vector<std::uint8_t> device(kBlockSize);
    const auto job = MakeDmaJob({pattern.data()}, {device.data()}, {kBlockSize});

    constexpr std::size_t kJobCount = 8;
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        EXPECT_TRUE(engine_->SubmitH2DBatch(id, job).Success());
    }

    const auto collected = DrainFinished(kJobCount);
    ASSERT_EQ(collected.size(), kJobCount);
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        const auto found = std::find_if(
            collected.begin(), collected.end(),
            [id](const std::pair<std::size_t, bool>& entry) { return entry.first == id; });
        ASSERT_TRUE(found != collected.end());
        EXPECT_TRUE(found->second);
    }

    auto again = engine_->GetFinished();
    ASSERT_TRUE(again.HasValue());
    EXPECT_TRUE(again.Value().empty());

    EXPECT_TRUE(engine_->SubmitH2DBatch(kJobCount + 1, job).Success());
    EXPECT_EQ(engine_->Wait(kJobCount + 1), Status::Ok());
    auto drained = engine_->GetFinished();
    ASSERT_TRUE(drained.HasValue());
    EXPECT_TRUE(drained.Value().empty());
}

TEST_F(SimuDmaEngineTest, WaitOnlyUsageKeepsExactlyOnceAcrossIdReuse)
{
    const auto pattern = MakePattern(kBlockSize, 0x99);
    std::vector<std::uint8_t> device(kBlockSize);
    const auto job = MakeDmaJob({pattern.data()}, {device.data()}, {kBlockSize});

    constexpr std::size_t kJobCount = 8;
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        ASSERT_TRUE(engine_->SubmitH2DBatch(id, job).Success());
        EXPECT_EQ(engine_->Wait(id), Status::Ok());
    }

    auto stale = engine_->GetFinished();
    ASSERT_TRUE(stale.HasValue());
    EXPECT_TRUE(stale.Value().empty());

    for (std::size_t id = 1; id <= kJobCount; ++id) {
        ASSERT_TRUE(engine_->SubmitH2DBatch(id, job).Success());
    }
    const auto collected = DrainFinished(kJobCount);
    ASSERT_EQ(collected.size(), kJobCount);
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        const auto found = std::find_if(
            collected.begin(), collected.end(),
            [id](const std::pair<std::size_t, bool>& entry) { return entry.first == id; });
        ASSERT_TRUE(found != collected.end());
        EXPECT_TRUE(found->second);
    }
    auto again = engine_->GetFinished();
    ASSERT_TRUE(again.HasValue());
    EXPECT_TRUE(again.Value().empty());
}

TEST_F(SimuDmaEngineTest, WaitRejectsUnknownAndConsumedJobIds)
{
    std::vector<std::uint8_t> host(kBlockSize);
    std::vector<std::uint8_t> device(kBlockSize);
    EXPECT_EQ(engine_->Wait(42), Status::InvalidParam());
    EXPECT_TRUE(engine_->SubmitH2DBatch(1, MakeDmaJob({host.data()}, {device.data()}, {kBlockSize}))
                    .Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
    EXPECT_EQ(engine_->Wait(1), Status::InvalidParam());
}

TEST_F(SimuDmaEngineTest, SynchronizeSettlesAllJobsWithoutConsumingTheirReports)
{
    const auto pattern = MakePattern(kBlockSize, 0x66);
    std::vector<std::uint8_t> device(kBlockSize);
    const auto job = MakeDmaJob({pattern.data()}, {device.data()}, {kBlockSize});

    constexpr std::size_t kJobCount = 4;
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        EXPECT_TRUE(engine_->SubmitH2DBatch(id, job).Success());
    }
    EXPECT_EQ(engine_->Synchronize(), Status::Ok());

    auto finished = engine_->GetFinished();
    ASSERT_TRUE(finished.HasValue());
    ASSERT_EQ(finished.Value().size(), kJobCount);
    for (std::size_t id = 1; id <= kJobCount; ++id) {
        const auto found = std::find_if(
            finished.Value().begin(), finished.Value().end(),
            [id](const std::pair<std::size_t, bool>& entry) { return entry.first == id; });
        ASSERT_TRUE(found != finished.Value().end());
        EXPECT_TRUE(found->second);
    }

    EXPECT_TRUE(engine_->SubmitH2DBatch(1, job).Success());
    EXPECT_EQ(engine_->Wait(1), Status::Ok());
    EXPECT_EQ(engine_->Synchronize(), Status::Ok());
}

TEST(SimuDmaEngineInFlightTest, GetFinishedReflectsTheInFlightWindowConsistently)
{
    auto created = DmaEngineRegistry::Instance().Create(DmaEngineKind::Naive);
    ASSERT_TRUE(created.HasValue());
    std::unique_ptr<DmaEngine> engine(*created);
    constexpr std::size_t kBigLength = 64U * 1024 * 1024;
    const DmaEngine::InitParam param{kDeviceId, kBigLength};
    ASSERT_TRUE(engine->Setup(param).Success());

    const std::vector<std::uint8_t> host(kBigLength, 0xA5);
    std::vector<std::uint8_t> device(kBigLength);
    const auto job = MakeDmaJob({host.data()}, {device.data()}, {kBigLength});
    ASSERT_TRUE(engine->SubmitH2DBatch(1, job).Success());

    auto first = engine->GetFinished();
    ASSERT_TRUE(first.HasValue());
    if (first.Value().empty()) {
        /* the copy is still running: the id must remain occupied */
        EXPECT_EQ(engine->SubmitH2DBatch(1, job), Status::DuplicateKey());
    } else {
        /* the copy already landed: the id must already be free */
        ASSERT_EQ(first.Value().size(), std::size_t{1});
        EXPECT_TRUE(engine->SubmitH2DBatch(1, job).Success());
    }
    EXPECT_EQ(engine->Wait(1), Status::Ok());
    EXPECT_EQ(std::memcmp(device.data(), host.data(), kBigLength), 0);
}

TEST(SimuDmaEngineLifetimeTest, DestructionDrainsPendingJobs)
{
    /* buffers declared first: the engine dies (and drains) before they do */
    const auto first = MakePattern(kBlockSize, 0x11);
    const auto second = MakePattern(kBlockSize, 0x22);
    std::vector<std::uint8_t> firstTarget(kBlockSize);
    std::vector<std::uint8_t> secondTarget(kBlockSize);

    auto created = DmaEngineRegistry::Instance().Create(DmaEngineKind::Naive);
    ASSERT_TRUE(created.HasValue());
    std::unique_ptr<DmaEngine> engine(*created);
    ASSERT_TRUE(engine->Setup(DmaEngine::InitParam{kDeviceId, kMaxLength}).Success());

    ASSERT_TRUE(
        engine->SubmitH2DBatch(1, MakeDmaJob({first.data()}, {firstTarget.data()}, {kBlockSize}))
            .Success());
    ASSERT_TRUE(
        engine->SubmitH2DBatch(2, MakeDmaJob({second.data()}, {secondTarget.data()}, {kBlockSize}))
            .Success());
    engine.reset(); /* drain-and-join: queued copies complete before return */

    EXPECT_EQ(std::memcmp(firstTarget.data(), first.data(), kBlockSize), 0);
    EXPECT_EQ(std::memcmp(secondTarget.data(), second.data(), kBlockSize), 0);
}

}  // namespace
}  // namespace UC::Transfer
