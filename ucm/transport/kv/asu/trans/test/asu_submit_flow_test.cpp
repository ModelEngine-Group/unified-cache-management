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
#include "asu_submit_flow.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>
#include "buffer_manager.h"
#include "connection_internal.h"

namespace UC::ASU {
namespace {

std::uint32_t g_kernelCount = 0;
std::uint32_t g_quietCount = 0;
std::vector<Status> g_sendStatuses;

}  // namespace

std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, std::uint32_t kernelCount,
                         std::uint32_t quietCount)
{
    g_kernelCount = kernelCount;
    g_quietCount = quietCount;
    if (!g_sendStatuses.empty()) { return g_sendStatuses; }
    return std::vector<Status>(ioBatches.size(), Status::OK());
}

namespace {

TEST(AsuSubmitFlowTest, SendSubBatchBuffersReadsSendCountsFromAttrs)
{
    g_kernelCount = 0;
    g_quietCount = 0;
    g_sendStatuses.clear();

    std::unordered_map<std::string, std::string> attrs = {
        {"kernel_count", "3"},
        {"quiet_count",  "7"},
    };

    ScatterGatherEntry sge;
    SendIoBatch ioBatch{nullptr, &sge};
    std::vector<SendIoBatch> ioBatches = {ioBatch};
    std::vector<std::size_t> subBatchIndexes = {0};

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());

    ConnectionManager connManager;
    const auto status =
        SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes, attrs, connManager);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(g_kernelCount, std::uint32_t{3});
    EXPECT_EQ(g_quietCount, std::uint32_t{7});
    EXPECT_EQ(subBatchContexts[0].state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContexts[0].status.ok());
}

TEST(AsuSubmitFlowTest, SendSubBatchBuffersReportsSendFailures)
{
    g_sendStatuses = {
        Status::Error(StatusCode::CONNECTION_ERROR, "fake send failure"),
        Status::Error(StatusCode::CONNECTION_ERROR, "fake send failure"),
    };

    std::unordered_map<std::string, std::string> attrs = {
        {"kernel_count", "3"},
        {"quiet_count",  "7"},
    };

    ConnectionManager connManager;
    ASSERT_TRUE(connManager.AddGroup(AsuEndpoint{}, 1).ok());
    auto* channel0 = connManager.SelectConnection();
    auto* channel1 = connManager.SelectConnection();
    ASSERT_NE(channel0, nullptr);
    ASSERT_EQ(channel0, channel1);

    ScatterGatherEntry sge0;
    ScatterGatherEntry sge1;
    std::vector<SendIoBatch> ioBatches = {
        SendIoBatch{channel0->GetNativeQp(), &sge0},
        SendIoBatch{channel1->GetNativeQp(), &sge1},
    };
    std::vector<std::size_t> subBatchIndexes = {0, 1};

    std::vector<TransportSubBatchContext> subBatchContexts(2);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].channel = channel0;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());
    subBatchContexts[1].state = TransportSubBatchState::PENDING;
    subBatchContexts[1].channel = channel1;
    subBatchContexts[1].entryStatus.assign(1, Status::OK());

    const auto status =
        SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes, attrs, connManager);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(channel0->GetState(), ChannelState::DRAINING);
    EXPECT_EQ(subBatchContexts[0].state, TransportSubBatchState::FAILED);
    EXPECT_EQ(subBatchContexts[1].state, TransportSubBatchState::FAILED);
    g_sendStatuses.clear();
}

TEST(AsuSubmitFlowTest, BuildSubBatchSendBuffersReleasesPreFailedSubBatches)
{
    BufferManager sendBufferManager;
    BufferManager flagBufferManager;
    ASSERT_TRUE(sendBufferManager.Init("test send buffer", MemoryType::HOST, 4096, 1).ok());
    ASSERT_TRUE(flagBufferManager.Init("test flag buffer", MemoryType::HOST, 128, 1).ok());

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    auto& subBatchContext = subBatchContexts[0];
    subBatchContext.state = TransportSubBatchState::FAILED;
    subBatchContext.status = Status::Error(StatusCode::INVALID_ARGUMENT, "pre-send failure");
    subBatchContext.entryStatus.assign(1, subBatchContext.status);
    ASSERT_TRUE(sendBufferManager.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(flagBufferManager.Allocate(64, subBatchContext.flagBuffer).ok());

    std::vector<SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    const auto status = BuildSubBatchSendBuffers(subBatchContexts, ioBatches, subBatchIndexes,
                                                 sendBufferManager, flagBufferManager);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_TRUE(ioBatches.empty());
    EXPECT_TRUE(subBatchIndexes.empty());
    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.flagBuffer.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.sendSge.addr, std::uint64_t{0});
    EXPECT_EQ(subBatchContext.flagBuffer.addr, std::uint64_t{0});
}

TEST(AsuSubmitFlowTest, BuildSubBatchSendBuffersMarksMissingFlagBufferFailed)
{
    BufferManager sendBufferManager;
    BufferManager flagBufferManager;
    ASSERT_TRUE(sendBufferManager.Init("test send buffer", MemoryType::HOST, 4096, 1).ok());
    ASSERT_TRUE(flagBufferManager.Init("test flag buffer", MemoryType::HOST, 128, 1).ok());

    ConnectionManager connManager;
    ASSERT_TRUE(connManager.AddGroup(AsuEndpoint{}, 1).ok());
    auto* channel = connManager.SelectConnection();
    ASSERT_NE(channel, nullptr);
    EXPECT_EQ(channel->GetInflightCount(), std::uint32_t{1});

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    auto& subBatchContext = subBatchContexts[0];
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.channel = channel;
    subBatchContext.entryStatus.assign(2, Status::OK());
    ASSERT_TRUE(sendBufferManager.Allocate(64, subBatchContext.sendSge).ok());

    std::vector<SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    const auto status = BuildSubBatchSendBuffers(subBatchContexts, ioBatches, subBatchIndexes,
                                                 sendBufferManager, flagBufferManager);

    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_TRUE(ioBatches.empty());
    EXPECT_TRUE(subBatchIndexes.empty());
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::FAILED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(subBatchContext.channel, nullptr);
    EXPECT_EQ(channel->GetInflightCount(), std::uint32_t{0});
    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    for (const auto& entryStatus : subBatchContext.entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::NOT_INITIALIZED);
    }
}

TEST(AsuSubmitFlowTest, SendSubBatchBuffersFailsAllSentSubBatchesWhenStatusCountMismatches)
{
    g_sendStatuses = {Status::OK()};

    std::unordered_map<std::string, std::string> attrs = {
        {"kernel_count", "3"},
        {"quiet_count",  "7"},
    };

    ScatterGatherEntry sge0;
    ScatterGatherEntry sge1;
    std::vector<SendIoBatch> ioBatches = {
        SendIoBatch{nullptr, &sge0},
        SendIoBatch{nullptr, &sge1},
    };
    std::vector<std::size_t> subBatchIndexes = {0, 1};

    std::vector<TransportSubBatchContext> subBatchContexts(2);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());
    subBatchContexts[1].state = TransportSubBatchState::PENDING;
    subBatchContexts[1].entryStatus.assign(1, Status::OK());

    ConnectionManager connManager;
    const auto status =
        SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes, attrs, connManager);

    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[0].state, TransportSubBatchState::FAILED);
    EXPECT_EQ(subBatchContexts[0].status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[0].entryStatus[0].code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[1].state, TransportSubBatchState::FAILED);
    EXPECT_EQ(subBatchContexts[1].status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[1].entryStatus[0].code, StatusCode::INTERNAL_ERROR);
    g_sendStatuses.clear();
}

}  // namespace
}  // namespace UC::ASU
