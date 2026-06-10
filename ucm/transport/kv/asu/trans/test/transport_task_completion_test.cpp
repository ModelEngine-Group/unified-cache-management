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
#include <atomic>
#include <cstdint>
#include <gtest/gtest.h>
#define private public
#include "asu_transport_impl.h"
#undef private
#include "buffer_manager.h"
#include "connection_internal.h"
#include "trans_provider.h"

namespace UC::ASU {
namespace {

constexpr std::size_t kTestBufferSlotSize = 512;
constexpr std::size_t kTestBufferSlotNum = 16;

class StubTransProvider : public TransProvider {
public:
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
    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>&, uint32_t,
                             uint32_t) override
    {
        return {};
    }
    Status RegisterMemory(ConnectionHandle, const std::vector<RegisterMemoryDesc>&,
                          std::vector<MemHandle>&) override
    {
        return Status::OK();
    }
    std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>&) override
    {
        return {};
    }
    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }
    std::vector<Status> FreeThread(const std::vector<ThreadHandle>&) override { return {}; }
    Status GetMemTokenId(MemHandle, uint32_t&) override { return Status::OK(); }
};

class TransportTaskCompletionTest : public ::testing::Test {
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

    void SetUp() override
    {
        transport_ = std::make_unique<AsuTransportImpl>();
        transport_->SetTransProvider(std::make_unique<StubTransProvider>());
        auto* provider = transport_->transProvider_.get();
        auto status =
            transport_->sendBufferManager_.Init("test send buffer", MemoryType::HOST,
                                                kTestBufferSlotSize, kTestBufferSlotNum, provider);
        ASSERT_TRUE(status.ok()) << status.message;
        status =
            transport_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST,
                                                kTestBufferSlotSize, kTestBufferSlotNum, provider);
        ASSERT_TRUE(status.ok()) << status.message;
    }

    std::unique_ptr<AsuTransportImpl> transport_;
};

TEST_F(TransportTaskCompletionTest, InitializeCountsAlreadyTerminalSubBatches)
{
    TransportTaskContext ctx;
    ctx.subBatchContexts.resize(3);
    ctx.subBatchContexts[0].state = TransportSubBatchState::PENDING;
    ctx.subBatchContexts[1].state = TransportSubBatchState::COMPLETED;
    ctx.subBatchContexts[2].state = TransportSubBatchState::COMPLETED;
    ctx.subBatchContexts[2].status = Status::Error(StatusCode::IO_ERROR, "fake error");

    ctx.InitializeTerminalSubBatchCount();

    EXPECT_EQ(ctx.completedSubBatchCount, std::uint32_t{2});
}

TEST_F(TransportTaskCompletionTest, CompleteSubBatchOnlyCountsPendingSubBatchOnce)
{
    TransportTaskContext ctx;
    TransportSubBatchContext subBatchContext;
    const auto status = Status::Error(StatusCode::IO_ERROR, "fake error");

    transport_->CompleteSubBatch(ctx, subBatchContext, status);
    transport_->CompleteSubBatch(ctx, subBatchContext, status);

    EXPECT_EQ(ctx.completedSubBatchCount, std::uint32_t{1});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::IO_ERROR);
}

TEST_F(TransportTaskCompletionTest, ReleaseSubBatchResourcesClearsAllocatedSlots)
{
    TransportSubBatchContext subBatchContext;
    ASSERT_TRUE(transport_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(transport_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer).ok());

    transport_->ReleaseSubBatchResources(subBatchContext);

    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.flagBuffer.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.sendSge.addr, std::uint64_t{0});
    EXPECT_EQ(subBatchContext.flagBuffer.addr, std::uint64_t{0});
}

TEST_F(TransportTaskCompletionTest, ReleaseSubBatchResourcesPreservesSubBatchStatus)
{
    TransportSubBatchContext subBatchContext;
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = Status::Error(StatusCode::IO_ERROR, "send failed");
    ASSERT_TRUE(transport_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(transport_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer).ok());

    transport_->ReleaseSubBatchResources(subBatchContext);

    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::IO_ERROR);
}

TEST_F(TransportTaskCompletionTest, ReleaseSubBatchResourcesClearsSlotsAfterFreeFailure)
{
    TransportSubBatchContext subBatchContext;
    subBatchContext.sendSge.slot_index = kTestBufferSlotNum;
    ASSERT_TRUE(transport_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer).ok());

    transport_->ReleaseSubBatchResources(subBatchContext);

    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.flagBuffer.slot_index, UINT32_MAX);
}

TEST_F(TransportTaskCompletionTest, ReleaseSubBatchResourcesReleasesChannelInflight)
{
    StubTransProvider provider;
    ConnectionManager connManager(provider, "", 5000);
    ASSERT_TRUE(connManager.AddGroup(AsuEndpoint{}, 1).ok());
    auto channel = connManager.SelectConnection();
    ASSERT_NE(channel, nullptr);
    ASSERT_EQ(channel->GetInflightCount(), std::uint32_t{1});

    TransportSubBatchContext subBatchContext;
    subBatchContext.channel = channel;

    transport_->ReleaseSubBatchResources(subBatchContext);

    EXPECT_EQ(channel->GetInflightCount(), std::uint32_t{0});
    EXPECT_EQ(subBatchContext.channel.get(), nullptr);
}

TEST_F(TransportTaskCompletionTest, TryFinalizeEmptyTaskUsesExistingFinalStatus)
{
    TransportTaskContext ctx;
    ctx.finalStatus = Status::OK();

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);

    ctx.state.store(TransportTaskState::PENDING, std::memory_order_release);
    ctx.finalStatus = Status::Error(StatusCode::UNSUPPORTED, "unsupported");

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
}

TEST_F(TransportTaskCompletionTest, TryFinalizeWaitsUntilAllSubBatchesFinish)
{
    TransportTaskContext ctx;
    ctx.subBatchContexts.resize(2);
    ctx.completedSubBatchCount = 1;
    ctx.state.store(TransportTaskState::INFLIGHT, std::memory_order_release);

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::INFLIGHT);
}

TEST_F(TransportTaskCompletionTest, TryFinalizeAggregatesSuccessAndFailure)
{
    TransportTaskContext ctx;
    ctx.subBatchContexts.resize(2);
    ctx.subBatchContexts[0].state = TransportSubBatchState::COMPLETED;
    ctx.subBatchContexts[0].status = Status::OK();
    ctx.subBatchContexts[1].state = TransportSubBatchState::COMPLETED;
    ctx.subBatchContexts[1].status = Status::Error(StatusCode::IO_ERROR, "sub-batch failed");
    ctx.completedSubBatchCount = 2;

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
    EXPECT_EQ(ctx.finalStatus.code, StatusCode::PARTIAL_FAILED);

    TransportTaskContext successCtx;
    successCtx.subBatchContexts.resize(1);
    successCtx.subBatchContexts[0].state = TransportSubBatchState::COMPLETED;
    successCtx.subBatchContexts[0].status = Status::OK();
    successCtx.completedSubBatchCount = 1;

    successCtx.TryFinalizeFromSubBatches();

    EXPECT_EQ(successCtx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
    EXPECT_TRUE(successCtx.finalStatus.ok());
}

}  // namespace
}  // namespace UC::ASU
