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
#include <acl/acl.h>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>
#define private public
#include "asu_transport_impl.h"
#undef private
#include "asu_transport/trans_provider.h"
#include "buffer_manager.h"
#include "connection_internal.h"

namespace UC::ASU {
namespace {

std::uint32_t g_kernelCount = 0;
std::uint32_t g_quietCount = 0;
std::vector<Status> g_sendStatuses;

class StubTransProvider : public TransProvider {
public:
    std::uint32_t registerCount{0};
    std::uint32_t registerCallCount{0};
    std::uint32_t unregisterCount{0};
    std::uint32_t failRegisterAt{0};
    std::uint32_t tokenLookupCount{0};
    std::uint32_t failTokenLookupAt{0};
    std::uint32_t failUnregisterAt{0};
    bool failUnregister{false};

    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                            uint32_t, std::vector<ConnectionHandle>& handles) override
    {
        handles.clear();
        handles.resize(qpNum, nullptr);
        return Status::OK();
    }

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override
    {
        return std::vector<Status>(handles.size(), Status::OK());
    }

    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>& ioBatches,
                             uint32_t kernelCount, uint32_t quietCount) override
    {
        g_kernelCount = kernelCount;
        g_quietCount = quietCount;
        if (!g_sendStatuses.empty()) { return g_sendStatuses; }
        return std::vector<Status>(ioBatches.size(), Status::OK());
    }

    Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MRHandle>& handles) override
    {
        ++registerCallCount;
        handles.clear();
        handles.reserve(memoryDescs.size());
        for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
            ++registerCount;
            if (failRegisterAt != 0 && registerCount == failRegisterAt) {
                return Status::Error(StatusCode::INTERNAL_ERROR, "stub register failed");
            }
            handles.push_back(reinterpret_cast<MRHandle>(static_cast<uintptr_t>(registerCount)));
        }
        return Status::OK();
    }

    std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>& descs) override
    {
        std::vector<Status> statuses;
        statuses.reserve(descs.size());
        for (std::size_t index = 0; index < descs.size(); ++index) {
            ++unregisterCount;
            if (failUnregister || (failUnregisterAt != 0 && unregisterCount == failUnregisterAt)) {
                statuses.emplace_back(
                    Status::Error(StatusCode::INTERNAL_ERROR, "stub unregister failed"));
            } else {
                statuses.emplace_back(Status::OK());
            }
        }
        return statuses;
    }

    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>&) override { return {}; }

    Status GetMemTokenId(MRHandle, uint32_t& tokenId) override
    {
        ++tokenLookupCount;
        if (failTokenLookupAt != 0 && tokenLookupCount == failTokenLookupAt) {
            return Status::Error(StatusCode::INTERNAL_ERROR, "stub token lookup failed");
        }
        tokenId = 1;
        return Status::OK();
    }
};

void CreateTaskExecutor(AsuTransportImpl& transport)
{
    transport.taskExecutor_ = std::make_unique<TransportTaskExecutor>(
        transport.config_, transport.ioScheduler_, transport.transProvider_,
        transport.sendBufferManager_, transport.flagBufferManager_, transport.protocolManager_,
        transport.connManager_, transport.nextRequestCid_);
}

class AsuSubmitFlowBufferTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            FAIL() << "aclInit failed: " << ret;
        }
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    }

    static void TearDownTestSuite() { aclrtResetDevice(0); }

    void SetUp() override
    {
        transport_ = std::make_unique<AsuTransportImpl>();
        transport_->SetTransProvider(std::make_unique<StubTransProvider>());
        CreateTaskExecutor(*transport_);
    }

    std::unique_ptr<AsuTransportImpl> transport_;
};

}  // namespace

namespace {

TEST(AsuSubmitFlowTest, SendSubBatchBuffersReadsSendCountsFromAttrs)
{
    g_kernelCount = 0;
    g_quietCount = 0;
    g_sendStatuses.clear();

    AsuTransportImpl transport;
    transport.SetTransProvider(std::make_unique<StubTransProvider>());
    transport.config_.attrs = {
        {"kernel_count", "3"},
        {"quiet_count",  "7"},
    };
    CreateTaskExecutor(transport);

    TransProvider::SendIoBatch ioBatch{nullptr, nullptr, nullptr, 0};
    std::vector<TransProvider::SendIoBatch> ioBatches = {ioBatch};
    std::vector<std::size_t> subBatchIndexes = {0};

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());

    const auto status =
        transport.taskExecutor_->SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes);

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

    AsuTransportImpl transport;
    transport.SetTransProvider(std::make_unique<StubTransProvider>());
    transport.config_.attrs = {
        {"kernel_count", "3"},
        {"quiet_count",  "7"},
    };
    CreateTaskExecutor(transport);

    transport.connManager_ =
        std::make_unique<ConnectionManager>(*transport.transProvider_, "", 5000);
    ASSERT_TRUE(transport.connManager_->AddGroup(AsuEndpoint{}, 1).ok());
    auto channel0 = transport.connManager_->SelectConnection();
    auto channel1 = transport.connManager_->SelectConnection();
    ASSERT_NE(channel0, nullptr);
    ASSERT_EQ(channel0, channel1);

    std::vector<TransProvider::SendIoBatch> ioBatches = {
        TransProvider::SendIoBatch{channel0->GetConnection(), nullptr, nullptr, 0},
        TransProvider::SendIoBatch{channel1->GetConnection(), nullptr, nullptr, 0},
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
        transport.taskExecutor_->SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(channel0->GetState(), ChannelState::DRAINING);
    EXPECT_EQ(subBatchContexts[0].state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContexts[1].state, TransportSubBatchState::COMPLETED);
    g_sendStatuses.clear();
}

TEST_F(AsuSubmitFlowBufferTest, BuildSubBatchSendBuffersReleasesPreFailedSubBatches)
{
    ASSERT_TRUE(
        transport_->sendBufferManager_.Init("test send buffer", MemoryType::HOST, 4096, 1).ok());
    ASSERT_TRUE(
        transport_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST, 128, 1).ok());

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    auto& subBatchContext = subBatchContexts[0];
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = Status::Error(StatusCode::INVALID_ARGUMENT, "pre-send failure");
    subBatchContext.entryStatus.assign(1, subBatchContext.status);
    ASSERT_TRUE(transport_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(transport_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer).ok());

    std::vector<TransProvider::SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    const auto status = transport_->taskExecutor_->BuildSubBatchSendBuffers(
        subBatchContexts, ioBatches, subBatchIndexes);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_TRUE(ioBatches.empty());
    EXPECT_TRUE(subBatchIndexes.empty());
    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.flagBuffer.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    EXPECT_EQ(subBatchContext.flagBuffer.local_addr, std::uint64_t{0});
}

TEST_F(AsuSubmitFlowBufferTest, BuildSubBatchSendBuffersMarksMissingFlagBufferFailed)
{
    ASSERT_TRUE(
        transport_->sendBufferManager_.Init("test send buffer", MemoryType::HOST, 4096, 1).ok());
    ASSERT_TRUE(
        transport_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST, 128, 1).ok());

    transport_->connManager_ =
        std::make_unique<ConnectionManager>(*transport_->transProvider_, "", 5000);
    ASSERT_TRUE(transport_->connManager_->AddGroup(AsuEndpoint{}, 1).ok());
    auto channel = transport_->connManager_->SelectConnection();
    ASSERT_NE(channel, nullptr);
    EXPECT_EQ(channel->GetInflightCount(), std::uint32_t{1});

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    auto& subBatchContext = subBatchContexts[0];
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.channel = channel;
    subBatchContext.entryStatus.assign(2, Status::OK());
    ASSERT_TRUE(transport_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());

    std::vector<TransProvider::SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    const auto status = transport_->taskExecutor_->BuildSubBatchSendBuffers(
        subBatchContexts, ioBatches, subBatchIndexes);

    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_TRUE(ioBatches.empty());
    EXPECT_TRUE(subBatchIndexes.empty());
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(subBatchContext.channel.get(), nullptr);
    EXPECT_EQ(channel->GetInflightCount(), std::uint32_t{0});
    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    for (const auto& entryStatus : subBatchContext.entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::NOT_INITIALIZED);
    }
}

TEST_F(AsuSubmitFlowBufferTest, BuildSubBatchSendBuffersRejectsZeroSendLength)
{
    ASSERT_TRUE(
        transport_->sendBufferManager_.Init("test send buffer", MemoryType::HOST, 4096, 1).ok());
    ASSERT_TRUE(
        transport_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST, 128, 1).ok());

    transport_->connManager_ =
        std::make_unique<ConnectionManager>(*transport_->transProvider_, "", 5000);
    ASSERT_TRUE(transport_->connManager_->AddGroup(AsuEndpoint{}, 1).ok());

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    auto& subBatchContext = subBatchContexts[0];
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.channel = transport_->connManager_->SelectConnection();
    subBatchContext.entryStatus.assign(1, Status::OK());
    ASSERT_TRUE(transport_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(transport_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer).ok());
    subBatchContext.sendSge.length = 0;

    std::vector<TransProvider::SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    const auto status = transport_->taskExecutor_->BuildSubBatchSendBuffers(
        subBatchContexts, ioBatches, subBatchIndexes);

    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_TRUE(ioBatches.empty());
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
}

TEST_F(AsuSubmitFlowBufferTest, BuildSubBatchSendBuffersRejectsMissingChannel)
{
    ASSERT_TRUE(
        transport_->sendBufferManager_.Init("test send buffer", MemoryType::HOST, 4096, 1).ok());
    ASSERT_TRUE(
        transport_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST, 128, 1).ok());

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    auto& subBatchContext = subBatchContexts[0];
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.entryStatus.assign(1, Status::OK());
    ASSERT_TRUE(transport_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(transport_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer).ok());

    std::vector<TransProvider::SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    const auto status = transport_->taskExecutor_->BuildSubBatchSendBuffers(
        subBatchContexts, ioBatches, subBatchIndexes);

    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_TRUE(ioBatches.empty());
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
}

TEST_F(AsuSubmitFlowBufferTest, BuildSubBatchSendBuffersUsesHostPinnedDeviceAddresses)
{
    ASSERT_TRUE(
        transport_->sendBufferManager_.Init("test send buffer", MemoryType::HOST_PINNED, 4096, 1)
            .ok());
    ASSERT_TRUE(
        transport_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST_PINNED, 128, 1)
            .ok());

    transport_->connManager_ =
        std::make_unique<ConnectionManager>(*transport_->transProvider_, "", 5000);
    ASSERT_TRUE(transport_->connManager_->AddGroup(AsuEndpoint{}, 1).ok());

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    auto& subBatchContext = subBatchContexts[0];
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.channel = transport_->connManager_->SelectConnection();
    subBatchContext.entryStatus.assign(1, Status::OK());
    ASSERT_NE(subBatchContext.channel, nullptr);
    ASSERT_TRUE(transport_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(transport_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer).ok());
    ASSERT_NE(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    ASSERT_NE(subBatchContext.sendSge.device_addr, std::uint64_t{0});
    ASSERT_NE(subBatchContext.flagBuffer.local_addr, std::uint64_t{0});
    ASSERT_NE(subBatchContext.flagBuffer.device_addr, std::uint64_t{0});

    std::vector<TransProvider::SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    const auto status = transport_->taskExecutor_->BuildSubBatchSendBuffers(
        subBatchContexts, ioBatches, subBatchIndexes);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(ioBatches.size(), std::size_t{1});
    EXPECT_EQ(ioBatches[0].sendBuffer,
              reinterpret_cast<void*>(subBatchContext.sendSge.device_addr));
    EXPECT_EQ(ioBatches[0].flagBuffer,
              reinterpret_cast<void*>(subBatchContext.flagBuffer.device_addr));
}

TEST(AsuSubmitFlowTest, SendSubBatchBuffersFailsAllSentSubBatchesWhenStatusCountMismatches)
{
    g_sendStatuses = {Status::OK()};

    AsuTransportImpl transport;
    transport.SetTransProvider(std::make_unique<StubTransProvider>());
    transport.config_.attrs = {
        {"kernel_count", "3"},
        {"quiet_count",  "7"},
    };
    CreateTaskExecutor(transport);

    std::vector<TransProvider::SendIoBatch> ioBatches = {
        TransProvider::SendIoBatch{nullptr, nullptr, nullptr, 0},
        TransProvider::SendIoBatch{nullptr, nullptr, nullptr, 0},
    };
    std::vector<std::size_t> subBatchIndexes = {0, 1};

    std::vector<TransportSubBatchContext> subBatchContexts(2);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());
    subBatchContexts[1].state = TransportSubBatchState::PENDING;
    subBatchContexts[1].entryStatus.assign(1, Status::OK());

    const auto status =
        transport.taskExecutor_->SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes);

    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[0].state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContexts[0].status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[0].entryStatus[0].code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[1].state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContexts[1].status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[1].entryStatus[0].code, StatusCode::INTERNAL_ERROR);
    g_sendStatuses.clear();
}

}  // namespace
}  // namespace UC::ASU
