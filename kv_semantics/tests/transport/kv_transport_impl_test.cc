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
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>
#define private public
#include "kv_transport_impl.h"
#undef private
#include "trans_provider.h"
#include "buffer/buffer_manager.h"
#include "conn/connection_internal.h"

namespace kv::test {
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
    Status RegisterMemory(const std::vector<RegisterMemoryDesc>&,
                          std::vector<MRHandle>& handles) override
    {
        handles.push_back(reinterpret_cast<MRHandle>(static_cast<uintptr_t>(1)));
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
    Status GetMemTokenId(MRHandle, uint32_t& tokenId) override
    {
        tokenId = 1;
        return Status::OK();
    }
};

void CreateTaskExecutor(AsuTransportImpl& transport)
{
    transport.taskExecutor_ = std::make_unique<TransportTaskExecutor>(
        transport.config_, transport.transProvider_, transport.connManager_);
}

class TransportTaskCompletionTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        transport_ = std::make_unique<AsuTransportImpl>();
        transport_->SetTransProvider(std::make_unique<StubTransProvider>());
        CreateTaskExecutor(*transport_);
        auto status = transport_->taskExecutor_->sendBufferManager_.Init(
            "test send buffer", MemoryType::HOST, kTestBufferSlotSize, kTestBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        status = transport_->taskExecutor_->flagBufferManager_.Init(
            "test flag buffer", MemoryType::HOST, kTestBufferSlotSize, kTestBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        status = transport_->taskExecutor_->RegisterBufferMemory(
            transport_->taskExecutor_->sendBufferManager_,
            transport_->taskExecutor_->sendBufferMrHandle_);
        ASSERT_TRUE(status.ok()) << status.message;
        status = transport_->taskExecutor_->RegisterBufferMemory(
            transport_->taskExecutor_->flagBufferManager_,
            transport_->taskExecutor_->flagBufferMrHandle_);
        ASSERT_TRUE(status.ok()) << status.message;
    }

    std::unique_ptr<AsuTransportImpl> transport_;
};

TEST_F(TransportTaskCompletionTest, InitRejectsZeroMaxErrorCount)
{
    TransportConfig config;
    config.maxErrorCount = 0;

    const auto status = transport_->Init(config, transport_->transProvider_);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(TransportTaskCompletionTest, TaskExecutorSupportsExplicitLifecycle)
{
    AsuTransportImpl transport;
    EXPECT_EQ(transport.taskExecutor_, nullptr);
    transport.SetTransProvider(std::make_unique<StubTransProvider>());

    CreateTaskExecutor(transport);
    ASSERT_TRUE(transport.taskExecutor_->Init().ok());
    EXPECT_NE(transport.taskExecutor_, nullptr);

    EXPECT_TRUE(transport.taskExecutor_->Shutdown().ok());
    transport.taskExecutor_.reset();
    EXPECT_EQ(transport.taskExecutor_, nullptr);
}

TEST_F(TransportTaskCompletionTest, InitializeCountsOnlyPendingSubBatches)
{
    TransportTask ctx;
    ctx.subBatchContexts->resize(3);
    (*ctx.subBatchContexts)[0].state = TransportSubBatchState::PENDING;
    (*ctx.subBatchContexts)[1].state = TransportSubBatchState::COMPLETED;
    (*ctx.subBatchContexts)[2].state = TransportSubBatchState::COMPLETED;
    (*ctx.subBatchContexts)[2].status = Status::Error(StatusCode::IO_ERROR, "fake error");

    ctx.InitializeRemainingSubBatchCount();

    EXPECT_EQ(ctx.remainingSubBatchCount, std::uint32_t{1});
}

TEST_F(TransportTaskCompletionTest, CompleteSubBatchCompletesPendingSubBatch)
{
    TransportTask ctx;
    TransportSubBatchContext subBatchContext;
    const auto status = Status::Error(StatusCode::IO_ERROR, "fake error");
    ctx.remainingSubBatchCount = 1;

    transport_->taskExecutor_->CompleteSubBatch(ctx, subBatchContext, status);

    EXPECT_EQ(ctx.remainingSubBatchCount, std::uint32_t{0});
    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::IO_ERROR);
}

TEST_F(TransportTaskCompletionTest, ReleaseSubBatchResourcesClearsAllocatedSlots)
{
    TransportSubBatchContext subBatchContext;
    ASSERT_TRUE(
        transport_->taskExecutor_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());

    transport_->taskExecutor_->ReleaseSubBatchResources(subBatchContext);

    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.flagBuffer.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.sendSge.local_addr, std::uint64_t{0});
    EXPECT_EQ(subBatchContext.flagBuffer.local_addr, std::uint64_t{0});
}

TEST_F(TransportTaskCompletionTest, FailureCqeAccumulatesWithoutSuccessReset)
{
    auto* provider = transport_->transProvider_.get();
    transport_->connManager_ = std::make_unique<ConnectionManager>(*provider, "", 5000, 3);
    ASSERT_TRUE(transport_->connManager_->AddGroup(AsuEndpoint{}, 1).ok());
    auto channel = transport_->connManager_->SelectConnection();
    ASSERT_NE(channel, nullptr);
    transport_->connManager_->ReportFailure(channel);
    ASSERT_EQ(channel->GetErrorCount(), std::uint32_t{1});

    auto ctx = std::make_shared<TransportTask>();
    ctx->state.store(TransportTaskState::INFLIGHT, std::memory_order_release);
    ctx->subBatchContexts->resize(1);

    auto& subBatchContext = (*ctx->subBatchContexts)[0];
    subBatchContext.cid = 123;
    subBatchContext.opType = AsuOpType::BATCH_STORE;
    subBatchContext.entryStatus.assign(1, Status::OK());
    subBatchContext.channel = channel;
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_
            .Allocate((kCqeDwordCount + 1) * sizeof(std::uint32_t), subBatchContext.flagBuffer)
            .ok());

    auto* cqe = reinterpret_cast<std::uint32_t*>(subBatchContext.flagBuffer.local_addr);
    cqe[3] = subBatchContext.cid | (0x716U << 17);

    transport_->taskExecutor_->Poll(ctx);

    EXPECT_EQ(channel->GetErrorCount(), std::uint32_t{2});
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::ASU_CQE_IO_TIMEOUT);
}

TEST_F(TransportTaskCompletionTest, PollTaskCompletionsTimesOutAndReleasesResources)
{
    auto* provider = transport_->transProvider_.get();
    transport_->connManager_ = std::make_unique<ConnectionManager>(*provider, "", 5000, 2);
    ASSERT_TRUE(transport_->connManager_->AddGroup(AsuEndpoint{}, 1).ok());
    auto channel = transport_->connManager_->SelectConnection();
    ASSERT_NE(channel, nullptr);

    auto ctx = std::make_shared<TransportTask>();
    ctx->taskId = 1;
    ctx->state.store(TransportTaskState::INFLIGHT, std::memory_order_release);
    ctx->deadline = std::chrono::steady_clock::now();
    ctx->entryStatus.assign(2, Status::OK());
    ctx->subBatchContexts->resize(1);

    auto& subBatchContext = (*ctx->subBatchContexts)[0];
    subBatchContext.opType = AsuOpType::BATCH_STORE;
    subBatchContext.entryStatus.assign(2, Status::OK());
    subBatchContext.channel = channel;
    ASSERT_TRUE(
        transport_->taskExecutor_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());

    std::size_t callbackCount = 0;
    TaskResult callbackResult;
    ctx->onComplete = [&](TaskResult result) {
        ++callbackCount;
        callbackResult = std::move(result);
    };

    if (transport_->taskExecutor_->Poll(ctx)) { transport_->taskManager_.NotifyCompletion(ctx); }

    EXPECT_EQ(callbackCount, std::size_t{1});
    EXPECT_EQ(ctx->state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
    EXPECT_EQ(callbackResult.status.code, StatusCode::TIMEOUT);
    ASSERT_EQ(callbackResult.entryStatus.size(), std::size_t{2});
    EXPECT_EQ(callbackResult.entryStatus[0].code, StatusCode::TIMEOUT);
    EXPECT_EQ(callbackResult.entryStatus[1].code, StatusCode::TIMEOUT);
    EXPECT_EQ(subBatchContext.sendSge.slot_index, UINT32_MAX);
    EXPECT_EQ(subBatchContext.flagBuffer.slot_index, UINT32_MAX);
    EXPECT_EQ(channel->GetErrorCount(), std::uint32_t{1});
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
}

TEST_F(TransportTaskCompletionTest, ExecutionTimeoutCountsEachSubBatch)
{
    auto* provider = transport_->transProvider_.get();
    transport_->connManager_ = std::make_unique<ConnectionManager>(*provider, "", 5000, 2);
    ASSERT_TRUE(transport_->connManager_->AddGroup(AsuEndpoint{}, 1).ok());
    auto channel = transport_->connManager_->SelectConnection();
    ASSERT_NE(channel, nullptr);
    auto sameChannel = transport_->connManager_->SelectConnection();
    ASSERT_EQ(sameChannel, channel);

    auto ctx = std::make_shared<TransportTask>();
    ctx->state.store(TransportTaskState::INFLIGHT, std::memory_order_release);
    ctx->deadline = std::chrono::steady_clock::now();
    ctx->entryStatus.assign(2, Status::OK());
    ctx->subBatchContexts->resize(2);
    for (auto& subBatchContext : *ctx->subBatchContexts) {
        subBatchContext.channel = channel;
        subBatchContext.entryStatus.assign(1, Status::OK());
    }

    transport_->taskExecutor_->Poll(ctx);

    EXPECT_EQ(channel->GetErrorCount(), std::uint32_t{2});
    EXPECT_EQ(channel->GetState(), ChannelState::DRAINING);
    EXPECT_EQ(channel->GetInflightCount(), std::uint32_t{0});
}

TEST_F(TransportTaskCompletionTest, ReleaseSubBatchResourcesPreservesSubBatchStatus)
{
    TransportSubBatchContext subBatchContext;
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = Status::Error(StatusCode::IO_ERROR, "send failed");
    ASSERT_TRUE(
        transport_->taskExecutor_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());

    transport_->taskExecutor_->ReleaseSubBatchResources(subBatchContext);

    EXPECT_EQ(subBatchContext.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatchContext.status.code, StatusCode::IO_ERROR);
}

TEST_F(TransportTaskCompletionTest, ReleaseSubBatchResourcesClearsSlotsAfterFreeFailure)
{
    TransportSubBatchContext subBatchContext;
    subBatchContext.sendSge.slot_index = kTestBufferSlotNum;
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());

    transport_->taskExecutor_->ReleaseSubBatchResources(subBatchContext);

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

    transport_->taskExecutor_->ReleaseSubBatchResources(subBatchContext);

    EXPECT_EQ(channel->GetInflightCount(), std::uint32_t{0});
    EXPECT_EQ(subBatchContext.channel.get(), nullptr);
}

TEST_F(TransportTaskCompletionTest, TryFinalizeEmptyTaskMarksPartialFailed)
{
    TransportTask ctx;
    ctx.finalStatus = Status::OK();

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
    EXPECT_EQ(ctx.finalStatus.code, StatusCode::PARTIAL_FAILED);

    ctx.state.store(TransportTaskState::PENDING, std::memory_order_release);
    ctx.finalStatus = Status::Error(StatusCode::UNSUPPORTED, "unsupported");

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
    EXPECT_EQ(ctx.finalStatus.code, StatusCode::PARTIAL_FAILED);
}

TEST_F(TransportTaskCompletionTest, TryFinalizeWaitsUntilAllSubBatchesFinish)
{
    TransportTask ctx;
    ctx.subBatchContexts->resize(2);
    ctx.remainingSubBatchCount = 1;
    ctx.state.store(TransportTaskState::INFLIGHT, std::memory_order_release);

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::INFLIGHT);
}

TEST_F(TransportTaskCompletionTest, TryFinalizeAggregatesSuccessAndFailure)
{
    TransportTask ctx;
    ctx.subBatchContexts->resize(2);
    (*ctx.subBatchContexts)[0].state = TransportSubBatchState::COMPLETED;
    (*ctx.subBatchContexts)[0].status = Status::OK();
    (*ctx.subBatchContexts)[1].state = TransportSubBatchState::COMPLETED;
    (*ctx.subBatchContexts)[1].status = Status::Error(StatusCode::IO_ERROR, "sub-batch failed");
    ctx.remainingSubBatchCount = 0;

    ctx.TryFinalizeFromSubBatches();

    EXPECT_EQ(ctx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
    EXPECT_EQ(ctx.finalStatus.code, StatusCode::PARTIAL_FAILED);

    TransportTask successCtx;
    successCtx.subBatchContexts->resize(1);
    (*successCtx.subBatchContexts)[0].state = TransportSubBatchState::COMPLETED;
    (*successCtx.subBatchContexts)[0].status = Status::OK();
    successCtx.remainingSubBatchCount = 0;

    successCtx.TryFinalizeFromSubBatches();

    EXPECT_EQ(successCtx.state.load(std::memory_order_acquire), TransportTaskState::COMPLETED);
    EXPECT_TRUE(successCtx.finalStatus.ok());
}

TEST_F(TransportTaskCompletionTest, NotifyCompletionPassesResultOnce)
{
    TransportTask ctx;
    std::uint32_t callbackCount = 0;
    TaskResult callbackResult;
    ctx.onComplete = [&callbackCount, &callbackResult](TaskResult result) {
        ++callbackCount;
        callbackResult = std::move(result);
    };
    TaskResult result;
    result.status = Status::Error(StatusCode::IO_ERROR, "fake completion error");
    result.entryStatus = {Status::Error(StatusCode::NOT_FOUND, "fake entry error")};

    EXPECT_TRUE(ctx.NotifyCompletion(result));
    EXPECT_FALSE(ctx.NotifyCompletion(std::move(result)));

    EXPECT_EQ(callbackCount, std::uint32_t{1});
    EXPECT_EQ(callbackResult.status.code, StatusCode::IO_ERROR);
    ASSERT_EQ(callbackResult.entryStatus.size(), std::size_t{1});
    EXPECT_EQ(callbackResult.entryStatus[0].code, StatusCode::NOT_FOUND);
    EXPECT_TRUE(ctx.completionNotified.load(std::memory_order_acquire));
}

TEST_F(TransportTaskCompletionTest, TaskManagerNotifyCompletionRemovesTaskAfterCallback)
{
    TaskId taskId = kInvalidTaskId;
    bool callbackInvoked = false;
    bool taskPresentDuringCallback = false;
    auto ctx = std::make_unique<TransportTask>();
    ctx->onComplete = [&](TaskResult) {
        callbackInvoked = true;
        taskPresentDuringCallback = transport_->taskManager_.Get(taskId) != nullptr;
    };
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());
    auto submittedCtx = transport_->taskManager_.Get(taskId);
    ASSERT_NE(submittedCtx, nullptr);

    transport_->taskManager_.NotifyCompletion(submittedCtx);

    EXPECT_TRUE(callbackInvoked);
    EXPECT_TRUE(taskPresentDuringCallback);
    EXPECT_EQ(transport_->taskManager_.Get(taskId), nullptr);
}

TEST_F(TransportTaskCompletionTest, TaskManagerNotifyCompletionRemovesTaskWithoutCallback)
{
    TaskId taskId = kInvalidTaskId;
    auto ctx = std::make_unique<TransportTask>();
    ctx->finalStatus = Status::OK();
    ctx->entryStatus = {Status::OK()};
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());
    auto submittedCtx = transport_->taskManager_.Get(taskId);
    ASSERT_NE(submittedCtx, nullptr);
    submittedCtx->state.store(TransportTaskState::COMPLETED, std::memory_order_release);

    transport_->taskManager_.NotifyCompletion(submittedCtx);

    TaskResult result;
    TransportTaskManager::BuildResult(*submittedCtx, result);
    EXPECT_TRUE(result.status.ok());
    ASSERT_EQ(result.entryStatus.size(), std::size_t{1});
    EXPECT_TRUE(result.entryStatus[0].ok());
    EXPECT_EQ(transport_->taskManager_.Get(taskId), nullptr);
}

TEST_F(TransportTaskCompletionTest, CancelInvokesCallbackOutsideTaskLock)
{
    TaskId taskId = kInvalidTaskId;
    bool callbackInvoked = false;
    bool taskLockAvailable = false;
    bool terminalState = false;
    auto ctx = std::make_unique<TransportTask>();
    auto* rawCtx = ctx.get();
    ctx->onComplete = [&](TaskResult result) {
        callbackInvoked = result.status.code == StatusCode::CANCELED;
        terminalState =
            rawCtx->state.load(std::memory_order_acquire) == TransportTaskState::COMPLETED;
        taskLockAvailable = rawCtx->mutex.try_lock();
        if (taskLockAvailable) { rawCtx->mutex.unlock(); }
    };
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());

    EXPECT_TRUE(transport_->Cancel(taskId).ok());

    EXPECT_TRUE(callbackInvoked);
    EXPECT_TRUE(taskLockAvailable);
    EXPECT_TRUE(terminalState);
    EXPECT_EQ(transport_->taskManager_.Get(taskId), nullptr);
}

TEST_F(TransportTaskCompletionTest, CancelMarksOnlyUnfinishedEntriesCanceled)
{
    TaskId taskId = kInvalidTaskId;
    TaskResult callbackResult;
    bool completedSubBatchPreserved = false;
    bool pendingSubBatchCanceled = false;
    auto ctx = std::make_unique<TransportTask>();
    auto* rawCtx = ctx.get();
    ctx->entryStatus.assign(2, Status::OK());
    ctx->subBatchContexts->resize(2);
    (*ctx->subBatchContexts)[0].state = TransportSubBatchState::COMPLETED;
    (*ctx->subBatchContexts)[0].entryStatus = {Status::OK()};
    (*ctx->subBatchContexts)[1].state = TransportSubBatchState::PENDING;
    (*ctx->subBatchContexts)[1].entryStatus = {Status::OK()};
    ctx->onComplete = [&](TaskResult result) {
        const auto& completedSubBatch = (*rawCtx->subBatchContexts)[0];
        const auto& canceledSubBatch = (*rawCtx->subBatchContexts)[1];
        completedSubBatchPreserved = completedSubBatch.state == TransportSubBatchState::COMPLETED &&
                                     completedSubBatch.status.ok();
        pendingSubBatchCanceled = canceledSubBatch.state == TransportSubBatchState::COMPLETED &&
                                  canceledSubBatch.status.code == StatusCode::CANCELED;
        callbackResult = std::move(result);
    };
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());

    ASSERT_TRUE(transport_->Cancel(taskId).ok());

    EXPECT_EQ(callbackResult.status.code, StatusCode::CANCELED);
    ASSERT_EQ(callbackResult.entryStatus.size(), std::size_t{2});
    EXPECT_TRUE(callbackResult.entryStatus[0].ok());
    EXPECT_EQ(callbackResult.entryStatus[1].code, StatusCode::CANCELED);
    EXPECT_TRUE(completedSubBatchPreserved);
    EXPECT_TRUE(pendingSubBatchCanceled);
}

TEST_F(TransportTaskCompletionTest, FailedSubmissionDoesNotInvokeCallback)
{
    std::vector<KVBuffer> entries(1);
    std::uint32_t callbackCount = 0;

    auto task = std::make_shared<TransportTask>();
    task->taskId = 123;
    task->opType = AsuOpType::BATCH_LOAD;
    task->entries = entries;
    task->onComplete = [&callbackCount](TaskResult) { ++callbackCount; };
    const auto status = transport_->Submit(task);

    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(task->taskId, kInvalidTaskId);
    EXPECT_EQ(callbackCount, std::uint32_t{0});
}

TEST_F(TransportTaskCompletionTest, ShutdownCallbackCannotSubmitNewTask)
{
    std::uint32_t callbackCount = 0;
    std::uint32_t nestedCallbackCount = 0;
    Status nestedSubmitStatus = Status::OK();
    auto ctx = std::make_unique<TransportTask>();
    ctx->onComplete = [&](TaskResult result) {
        ++callbackCount;
        EXPECT_EQ(result.status.code, StatusCode::CANCELED);
        auto nestedTask = std::make_shared<TransportTask>();
        nestedTask->opType = AsuOpType::BATCH_LOAD;
        nestedTask->entries.resize(1);
        nestedTask->onComplete = [&nestedCallbackCount](TaskResult) { ++nestedCallbackCount; };
        nestedSubmitStatus = transport_->Submit(nestedTask);
        EXPECT_EQ(nestedTask->taskId, kInvalidTaskId);
    };

    TaskId taskId = kInvalidTaskId;
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());

    EXPECT_TRUE(transport_->Shutdown().ok());

    EXPECT_EQ(callbackCount, std::uint32_t{1});
    EXPECT_EQ(nestedSubmitStatus.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(nestedCallbackCount, std::uint32_t{0});
}

TEST_F(TransportTaskCompletionTest, ShutdownDrainsInflightTaskBeforeCanceling)
{
    auto ctx = std::make_unique<TransportTask>();
    ctx->entryStatus.assign(1, Status::OK());
    ctx->subBatchContexts->resize(1);
    ctx->remainingSubBatchCount = 1;
    auto& subBatchContext = (*ctx->subBatchContexts)[0];
    subBatchContext.cid = 123;
    subBatchContext.opType = AsuOpType::BATCH_STORE;
    subBatchContext.entryStatus.assign(1, Status::OK());
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());
    auto* cqe = reinterpret_cast<std::uint32_t*>(subBatchContext.flagBuffer.local_addr);
    cqe[3] = subBatchContext.cid;

    std::uint32_t callbackCount = 0;
    Status callbackStatus = Status::Error(StatusCode::INTERNAL_ERROR, "callback not invoked");
    ctx->onComplete = [&](TaskResult result) {
        ++callbackCount;
        callbackStatus = std::move(result.status);
    };

    TaskId taskId = kInvalidTaskId;
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());
    auto submittedCtx = transport_->taskManager_.Get(taskId);
    ASSERT_NE(submittedCtx, nullptr);
    submittedCtx->state.store(TransportTaskState::INFLIGHT, std::memory_order_release);
    transport_->completionWorker_ =
        std::thread(&AsuTransportImpl::CompletionLoop, transport_.get());

    EXPECT_TRUE(transport_->Shutdown().ok());

    EXPECT_EQ(callbackCount, std::uint32_t{1});
    EXPECT_TRUE(callbackStatus.ok()) << callbackStatus.message;
}

TEST_F(TransportTaskCompletionTest, CancelReleasesResourcesBeforeInvokingCallbackOnce)
{
    TaskId taskId = kInvalidTaskId;
    std::uint32_t callbackCount = 0;
    bool resourcesReleased = false;
    auto ctx = std::make_unique<TransportTask>();
    ctx->subBatchContexts->resize(1);
    auto& subBatchContext = (*ctx->subBatchContexts)[0];
    ASSERT_TRUE(
        transport_->taskExecutor_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());
    auto* rawCtx = ctx.get();
    ctx->onComplete = [&](TaskResult result) {
        ++callbackCount;
        resourcesReleased = result.status.code == StatusCode::CANCELED &&
                            (*rawCtx->subBatchContexts)[0].sendSge.slot_index == UINT32_MAX &&
                            (*rawCtx->subBatchContexts)[0].flagBuffer.slot_index == UINT32_MAX;
    };
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());

    EXPECT_TRUE(transport_->Cancel(taskId).ok());
    EXPECT_EQ(transport_->Cancel(taskId).code, StatusCode::TASK_NOT_FOUND);

    EXPECT_EQ(callbackCount, std::uint32_t{1});
    EXPECT_TRUE(resourcesReleased);
    EXPECT_EQ(transport_->taskManager_.Get(taskId), nullptr);
}

TEST_F(TransportTaskCompletionTest, ShutdownReleasesResourcesBeforeInvokingCallbackOnce)
{
    TaskId taskId = kInvalidTaskId;
    std::uint32_t callbackCount = 0;
    bool resourcesReleased = false;
    bool entryCanceled = false;
    auto ctx = std::make_unique<TransportTask>();
    ctx->entryStatus = {Status::OK()};
    ctx->subBatchContexts->resize(1);
    auto& subBatchContext = (*ctx->subBatchContexts)[0];
    subBatchContext.entryStatus = {Status::OK()};
    ASSERT_TRUE(
        transport_->taskExecutor_->sendBufferManager_.Allocate(64, subBatchContext.sendSge).ok());
    ASSERT_TRUE(
        transport_->taskExecutor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer)
            .ok());
    auto* rawCtx = ctx.get();
    ctx->onComplete = [&](TaskResult result) {
        ++callbackCount;
        resourcesReleased = result.status.code == StatusCode::CANCELED &&
                            (*rawCtx->subBatchContexts)[0].sendSge.slot_index == UINT32_MAX &&
                            (*rawCtx->subBatchContexts)[0].flagBuffer.slot_index == UINT32_MAX;
        entryCanceled =
            result.entryStatus.size() == 1 && result.entryStatus[0].code == StatusCode::CANCELED;
    };
    ASSERT_TRUE(transport_->taskManager_.Submit(std::move(ctx), taskId).ok());

    EXPECT_TRUE(transport_->Shutdown().ok());
    EXPECT_TRUE(transport_->Shutdown().ok());

    EXPECT_EQ(callbackCount, std::uint32_t{1});
    EXPECT_TRUE(resourcesReleased);
    EXPECT_TRUE(entryCanceled);
    EXPECT_EQ(transport_->taskManager_.Get(taskId), nullptr);
}

}  // namespace
}  // namespace kv::test
