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
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <unordered_map>
#include <utility>
#include <vector>
#define private public
#include "kv_transport_impl.h"
#undef private
#include "trans_provider.h"
#include "buffer/buffer_manager.h"
#include "conn/connection_internal.h"

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
        transport.config_, transport.transProvider_, transport.connManager_);
}

Status InitTaskExecutorWithoutRecoverLoop(AsuTransportImpl& transport,
                                          std::shared_ptr<TransProvider> provider)
{
    transport.SetTransProvider(std::move(provider));
    CreateTaskExecutor(transport);
    const auto status = transport.taskExecutor_->Init();
    if (!status.ok()) { transport.taskExecutor_.reset(); }
    return status;
}

Status ShutdownTaskExecutorWithoutRecoverLoop(AsuTransportImpl& transport)
{
    if (!transport.taskExecutor_) { return Status::OK(); }
    const auto status = transport.taskExecutor_->Shutdown();
    if (status.ok()) { transport.taskExecutor_.reset(); }
    return status;
}

class AsuSubmitFlowBufferTest : public ::testing::Test {
protected:
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

class AsuTransportBufferRegistrationTest : public ::testing::Test {};

TEST_F(AsuTransportBufferRegistrationTest, InitRegistersAndShutdownUnregistersBothBuffers)
{
    auto provider = std::make_shared<StubTransProvider>();
    AsuTransportImpl transport;

    auto status = InitTaskExecutorWithoutRecoverLoop(transport, provider);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(provider->registerCount, 2);
    EXPECT_EQ(provider->registerCallCount, 2);
    EXPECT_EQ(provider->tokenLookupCount, 2);
    ASSERT_NE(transport.taskExecutor_, nullptr);
    EXPECT_NE(transport.taskExecutor_->sendBufferMrHandle_, kInvalidMRHandle);
    EXPECT_NE(transport.taskExecutor_->flagBufferMrHandle_, kInvalidMRHandle);
    EXPECT_EQ(transport.taskExecutor_->sendBufferManager_.GetTokenId(), 1);
    EXPECT_EQ(transport.taskExecutor_->flagBufferManager_.GetTokenId(), 1);

    status = ShutdownTaskExecutorWithoutRecoverLoop(transport);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(provider->unregisterCount, 2);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(AsuTransportBufferRegistrationTest, DestructorUnregistersBothBuffers)
{
    auto provider = std::make_shared<StubTransProvider>();

    {
        AsuTransportImpl transport;
        const auto status = InitTaskExecutorWithoutRecoverLoop(transport, provider);
        ASSERT_TRUE(status.ok()) << status.message;
        EXPECT_EQ(provider->registerCount, 2);
        EXPECT_EQ(provider->unregisterCount, 0);
    }

    EXPECT_EQ(provider->unregisterCount, 2);
}

TEST_F(AsuTransportBufferRegistrationTest, FirstRegistrationFailureDoesNotUnregister)
{
    auto provider = std::make_shared<StubTransProvider>();
    provider->failRegisterAt = 1;
    AsuTransportImpl transport;

    const auto status = InitTaskExecutorWithoutRecoverLoop(transport, provider);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("send buffer"), std::string::npos);
    EXPECT_EQ(provider->registerCount, 1);
    EXPECT_EQ(provider->registerCallCount, 1);
    EXPECT_EQ(provider->tokenLookupCount, 0);
    EXPECT_EQ(provider->unregisterCount, 0);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(AsuTransportBufferRegistrationTest, FirstTokenLookupFailureCleansUpRegisteredBuffer)
{
    auto provider = std::make_shared<StubTransProvider>();
    provider->failTokenLookupAt = 1;
    AsuTransportImpl transport;

    const auto status = InitTaskExecutorWithoutRecoverLoop(transport, provider);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("send buffer"), std::string::npos);
    EXPECT_EQ(provider->registerCount, 1);
    EXPECT_EQ(provider->registerCallCount, 1);
    EXPECT_EQ(provider->tokenLookupCount, 1);
    EXPECT_EQ(provider->unregisterCount, 1);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(AsuTransportBufferRegistrationTest, TokenLookupRollbackFailureIsRetriedDuringShutdown)
{
    auto provider = std::make_shared<StubTransProvider>();
    provider->failTokenLookupAt = 1;
    provider->failUnregisterAt = 1;
    AsuTransportImpl transport;

    const auto status = InitTaskExecutorWithoutRecoverLoop(transport, provider);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("send buffer"), std::string::npos);
    EXPECT_EQ(provider->registerCount, 1);
    EXPECT_EQ(provider->tokenLookupCount, 1);
    EXPECT_EQ(provider->unregisterCount, 2);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(AsuTransportBufferRegistrationTest, SecondRegistrationFailureCleansUpFirstBuffer)
{
    auto provider = std::make_shared<StubTransProvider>();
    provider->failRegisterAt = 2;
    AsuTransportImpl transport;

    const auto status = InitTaskExecutorWithoutRecoverLoop(transport, provider);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("flag buffer"), std::string::npos);
    EXPECT_EQ(provider->registerCount, 2);
    EXPECT_EQ(provider->unregisterCount, 1);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(AsuTransportBufferRegistrationTest, SecondTokenLookupFailureCleansUpBothBuffers)
{
    auto provider = std::make_shared<StubTransProvider>();
    provider->failTokenLookupAt = 2;
    AsuTransportImpl transport;

    const auto status = InitTaskExecutorWithoutRecoverLoop(transport, provider);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("flag buffer"), std::string::npos);
    EXPECT_EQ(provider->registerCount, 2);
    EXPECT_EQ(provider->tokenLookupCount, 2);
    EXPECT_EQ(provider->unregisterCount, 2);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(AsuTransportBufferRegistrationTest, FlagUnregistrationFailureIncludesCallerContext)
{
    auto provider = std::make_shared<StubTransProvider>();
    AsuTransportImpl transport;

    ASSERT_TRUE(InitTaskExecutorWithoutRecoverLoop(transport, provider).ok());
    provider->failUnregisterAt = 1;

    const auto status = ShutdownTaskExecutorWithoutRecoverLoop(transport);
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("flag buffer"), std::string::npos);
    EXPECT_EQ(provider->unregisterCount, 2);
    ASSERT_NE(transport.taskExecutor_, nullptr);
    EXPECT_NE(transport.taskExecutor_->flagBufferMrHandle_, kInvalidMRHandle);
    EXPECT_EQ(transport.taskExecutor_->sendBufferMrHandle_, kInvalidMRHandle);

    EXPECT_TRUE(ShutdownTaskExecutorWithoutRecoverLoop(transport).ok());
    EXPECT_EQ(provider->unregisterCount, 3);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(AsuTransportBufferRegistrationTest, SendUnregistrationFailureIncludesCallerContext)
{
    auto provider = std::make_shared<StubTransProvider>();
    AsuTransportImpl transport;

    ASSERT_TRUE(InitTaskExecutorWithoutRecoverLoop(transport, provider).ok());
    provider->failUnregisterAt = 2;

    const auto status = ShutdownTaskExecutorWithoutRecoverLoop(transport);
    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("send buffer"), std::string::npos);
    EXPECT_EQ(provider->unregisterCount, 2);
    ASSERT_NE(transport.taskExecutor_, nullptr);
    EXPECT_EQ(transport.taskExecutor_->flagBufferMrHandle_, kInvalidMRHandle);
    EXPECT_NE(transport.taskExecutor_->sendBufferMrHandle_, kInvalidMRHandle);

    EXPECT_TRUE(ShutdownTaskExecutorWithoutRecoverLoop(transport).ok());
    EXPECT_EQ(provider->unregisterCount, 3);
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

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

    std::vector<TransportSubBatchContext> subBatchContexts(1);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());

    transport.taskExecutor_->SendSubBatchBuffers(subBatchContexts, ioBatches);

    EXPECT_EQ(g_kernelCount, std::uint32_t{3});
    EXPECT_EQ(g_quietCount, std::uint32_t{7});
    EXPECT_EQ(subBatchContexts[0].state, TransportSubBatchState::PENDING);
    EXPECT_TRUE(subBatchContexts[0].status.ok());
}

TEST(AsuSubmitFlowTest, AbortBeforeSendPreservesStatusesBeforeFailureAndCancelsFollowing)
{
    AsuTransportImpl transport;
    transport.SetTransProvider(std::make_unique<StubTransProvider>());
    CreateTaskExecutor(transport);

    const auto failedStatus =
        Status::Error(StatusCode::INTERNAL_ERROR, "sub-batch preparation failed");
    std::vector<TransportSubBatchContext> subBatchContexts(3);
    for (auto& subBatchContext : subBatchContexts) { subBatchContext.entryStatus = {Status::OK()}; }
    subBatchContexts[1].status = failedStatus;
    subBatchContexts[1].entryStatus = {failedStatus};

    TransportTask task;
    task.entryStatus.assign(3, Status::OK());
    transport.taskExecutor_->AbortSubBatchesBeforeSend(task, subBatchContexts);

    EXPECT_TRUE(subBatchContexts[0].status.ok());
    EXPECT_TRUE(subBatchContexts[0].entryStatus[0].ok());
    EXPECT_EQ(subBatchContexts[1].status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[1].entryStatus[0].code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(subBatchContexts[2].status.code, StatusCode::CANCELED);
    EXPECT_EQ(subBatchContexts[2].entryStatus[0].code, StatusCode::CANCELED);
    for (const auto& entryStatus : task.entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::CANCELED);
    }
    for (const auto& subBatchContext : subBatchContexts) {
        EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    }
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
    std::vector<TransportSubBatchContext> subBatchContexts(2);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].channel = channel0;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());
    subBatchContexts[1].state = TransportSubBatchState::PENDING;
    subBatchContexts[1].channel = channel1;
    subBatchContexts[1].entryStatus.assign(1, Status::OK());

    transport.taskExecutor_->SendSubBatchBuffers(subBatchContexts, ioBatches);

    EXPECT_EQ(subBatchContexts[0].status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(channel0->GetState(), ChannelState::DRAINING);
    EXPECT_EQ(subBatchContexts[0].state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContexts[1].state, TransportSubBatchState::COMPLETED);
    g_sendStatuses.clear();
}

TEST_F(AsuSubmitFlowBufferTest, BuildSubBatchSendBuffersUsesHostPinnedDeviceAddresses)
{
    ASSERT_TRUE(transport_->taskExecutor_->sendBufferManager_
                    .Init("test send buffer", MemoryType::HOST_PINNED, 4096, 1)
                    .ok());
    ASSERT_TRUE(transport_->taskExecutor_->flagBufferManager_
                    .Init("test flag buffer", MemoryType::HOST_PINNED, 128, 1)
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
    ASSERT_TRUE(
        transport_->taskExecutor_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());
    ASSERT_NE(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    ASSERT_NE(subBatchContext.sendSge.device_addr, std::uint64_t{0});
    ASSERT_NE(subBatchContext.flagBuffer.local_addr, std::uint64_t{0});
    ASSERT_NE(subBatchContext.flagBuffer.device_addr, std::uint64_t{0});

    std::vector<TransProvider::SendIoBatch> ioBatches;
    transport_->taskExecutor_->BuildSubBatchSendBuffers(subBatchContexts, ioBatches);

    ASSERT_EQ(ioBatches.size(), std::size_t{1});
    EXPECT_EQ(ioBatches[0].connectionHandle, subBatchContext.channel->GetConnection());
    EXPECT_EQ(ioBatches[0].sendBuffer,
              reinterpret_cast<void*>(subBatchContext.sendSge.device_addr));
    EXPECT_EQ(ioBatches[0].flagBuffer,
              reinterpret_cast<void*>(subBatchContext.flagBuffer.device_addr));
    EXPECT_EQ(ioBatches[0].len, subBatchContext.sendSge.length);
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
    std::vector<TransportSubBatchContext> subBatchContexts(2);
    subBatchContexts[0].state = TransportSubBatchState::PENDING;
    subBatchContexts[0].entryStatus.assign(1, Status::OK());
    subBatchContexts[1].state = TransportSubBatchState::PENDING;
    subBatchContexts[1].entryStatus.assign(1, Status::OK());

    transport.taskExecutor_->SendSubBatchBuffers(subBatchContexts, ioBatches);

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
