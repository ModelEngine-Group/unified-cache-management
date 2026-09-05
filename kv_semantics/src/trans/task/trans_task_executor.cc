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
#include "trans_task_executor.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include "utils/trans_task_utils.h"
#include "conn/connection_internal.h"
#include "logger.h"

namespace kv {

TransportTaskExecutor::TransportTaskExecutor(
    const TransportConfig& config, const std::shared_ptr<TransProvider>& transProvider,
    const std::unique_ptr<ConnectionManager>& connectionManager)
    : config_(config),
      ioScheduler_(config),
      transProvider_(transProvider),
      connManager_(connectionManager)
{
}

TransportTaskExecutor::~TransportTaskExecutor()
{
    const auto status = Shutdown();
    if (!status.ok()) {
        KV_WARN("TransportTaskExecutor destructor cleanup failed: code={} message={}",
                static_cast<int>(status.code), status.message);
    }
}

Status TransportTaskExecutor::Init()
{
    auto status = sendBufferManager_.Init("asu send buffer", MemoryType::HOST_PINNED,
                                          config_.sendBufferSlotSize, config_.sendBufferSlotNum);
    if (!status.ok()) {
        (void)Shutdown();
        return status;
    }

    status = flagBufferManager_.Init("asu flag buffer", MemoryType::HOST_PINNED,
                                     config_.flagBufferSlotSize, config_.flagBufferSlotNum);
    if (!status.ok()) {
        (void)Shutdown();
        return status;
    }

    status = RegisterBufferMemory(sendBufferManager_, sendBufferMrHandle_);
    if (!status.ok()) {
        const auto registerStatus =
            Status::Error(status.code, "failed to register send buffer: " + status.message);
        const auto shutdownStatus = Shutdown();
        if (!shutdownStatus.ok()) {
            KV_WARN(
                "TransportTaskExecutor::Init cleanup failed after send buffer registration "
                "failure: code={} message={}",
                static_cast<int>(shutdownStatus.code), shutdownStatus.message);
        }
        return registerStatus;
    }

    status = RegisterBufferMemory(flagBufferManager_, flagBufferMrHandle_);
    if (!status.ok()) {
        const auto registerStatus =
            Status::Error(status.code, "failed to register flag buffer: " + status.message);
        const auto shutdownStatus = Shutdown();
        if (!shutdownStatus.ok()) {
            KV_WARN(
                "TransportTaskExecutor::Init cleanup failed after flag buffer registration "
                "failure: code={} message={}",
                static_cast<int>(shutdownStatus.code), shutdownStatus.message);
        }
        return registerStatus;
    }
    return Status::OK();
}

Status TransportTaskExecutor::Shutdown()
{
    Status finalStatus = Status::OK();
    auto status = UnregisterBufferMemory(flagBufferMrHandle_);
    if (!status.ok()) {
        KV_WARN("Failed to unregister flag buffer: code={} message={}",
                static_cast<int>(status.code), status.message);
        finalStatus =
            Status::Error(status.code, "failed to unregister flag buffer: " + status.message);
    }

    status = UnregisterBufferMemory(sendBufferMrHandle_);
    if (!status.ok()) {
        KV_WARN("Failed to unregister send buffer: code={} message={}",
                static_cast<int>(status.code), status.message);
        if (finalStatus.ok()) {
            finalStatus =
                Status::Error(status.code, "failed to unregister send buffer: " + status.message);
        }
    }

    if (flagBufferMrHandle_ == kInvalidMRHandle) { flagBufferManager_.Shutdown(); }
    if (sendBufferMrHandle_ == kInvalidMRHandle) { sendBufferManager_.Shutdown(); }
    return finalStatus;
}

Status TransportTaskExecutor::RegisterBufferMemory(BufferManager& bufferManager, MRHandle& mrHandle)
{
    if (!transProvider_) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "TransProvider is null");
    }

    TransProvider::RegisterMemoryDesc desc;
    auto status = bufferManager.GetRegisterMemoryDesc(desc);
    if (!status.ok()) { return status; }

    std::vector<MRHandle> mrHandles;
    status = transProvider_->RegisterMemory({desc}, mrHandles);
    if (!status.ok()) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "provider memory registration failed: " + status.message);
    }
    if (mrHandles.empty() || mrHandles[0] == kInvalidMRHandle) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "provider returned no valid MR handle");
    }

    mrHandle = mrHandles[0];
    std::uint32_t tokenId = 0;
    status = transProvider_->GetMemTokenId(mrHandle, tokenId);
    if (!status.ok()) {
        const auto unregisterStatus = UnregisterBufferMemory(mrHandle);
        if (!unregisterStatus.ok()) {
            KV_WARN("Failed to unregister memory after token lookup failure: {}",
                    unregisterStatus.message);
        }
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "memory token lookup failed: " + status.message);
    }

    bufferManager.SetTokenId(tokenId);
    return Status::OK();
}

Status TransportTaskExecutor::UnregisterBufferMemory(MRHandle& mrHandle)
{
    if (mrHandle == kInvalidMRHandle) { return Status::OK(); }
    if (!transProvider_) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "TransProvider is null");
    }

    const auto statuses =
        transProvider_->UnregisterMemory({TransProvider::UnregisterMemoryDesc{mrHandle}});
    for (const auto& status : statuses) {
        if (!status.ok()) {
            return Status::Error(status.code,
                                 "provider memory unregistration failed: " + status.message);
        }
    }
    mrHandle = kInvalidMRHandle;
    return Status::OK();
}

void TransportTaskExecutor::ReleaseSubBatchResources(TransportSubBatchContext& subBatchContext)
{
    if (subBatchContext.sendSge.slot_index != UINT32_MAX) {
        const auto slotIndex = subBatchContext.sendSge.slot_index;
        auto status = sendBufferManager_.Free(slotIndex);
        if (!status.ok()) {
            KV_ERROR("Failed to release sub-batch send buffer slot({}): {}", slotIndex,
                     status.message);
        }
        subBatchContext.sendSge = {};
    }

    if (subBatchContext.flagBuffer.slot_index != UINT32_MAX) {
        const auto slotIndex = subBatchContext.flagBuffer.slot_index;
        auto status = flagBufferManager_.Free(slotIndex);
        if (!status.ok()) {
            KV_ERROR("Failed to release sub-batch flag buffer slot({}): {}", slotIndex,
                     status.message);
        }
        subBatchContext.flagBuffer = {};
    }

    if (subBatchContext.channel != nullptr) {
        subBatchContext.channel->ReleaseInflight();
        subBatchContext.channel = nullptr;
    }
}

void TransportTaskExecutor::ReleaseAllSubBatchResources(
    std::vector<TransportSubBatchContext>& subBatchContexts)
{
    for (auto& subBatchContext : subBatchContexts) { ReleaseSubBatchResources(subBatchContext); }
}

void TransportTaskExecutor::AbortSubBatchesBeforeSend(
    TransportTask& task, std::vector<TransportSubBatchContext>& subBatchContexts)
{
    const auto canceledStatus =
        Status::Error(StatusCode::CANCELED, "sub-batch canceled after pre-send failure");
    std::fill(task.entryStatus.begin(), task.entryStatus.end(), canceledStatus);

    bool hasFailedSubBatch = false;
    for (auto& subBatchContext : subBatchContexts) {
        ReleaseSubBatchResources(subBatchContext);
        if (!subBatchContext.status.ok()) {
            hasFailedSubBatch = true;
        } else if (hasFailedSubBatch) {
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      canceledStatus);
            subBatchContext.status = canceledStatus;
        }
        subBatchContext.state = TransportSubBatchState::COMPLETED;
    }
}

void TransportTaskExecutor::CompleteSubBatch(TransportTask& task,
                                             TransportSubBatchContext& subBatchContext,
                                             const Status& status)
{
    ReleaseSubBatchResources(subBatchContext);
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = status;
    --task.remainingSubBatchCount;
}

bool TransportTaskExecutor::Cancel(const TransportTaskPtr& task)
{
    if (!task) { return false; }

    std::lock_guard<std::mutex> lock(task->mutex);
    if (task->Done()) { return false; }
    const auto canceledStatus = Status::Error(StatusCode::CANCELED, "transport task canceled");
    std::fill(task->entryStatus.begin(), task->entryStatus.end(), canceledStatus);
    for (auto& subBatchContext : *task->subBatchContexts) {
        if (subBatchContext.state == TransportSubBatchState::COMPLETED) { continue; }
        std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                  canceledStatus);
        subBatchContext.status = canceledStatus;
        subBatchContext.state = TransportSubBatchState::COMPLETED;
    }
    task->finalStatus = canceledStatus;
    ReleaseAllSubBatchResources(*task->subBatchContexts);
    task->state.store(TransportTaskState::COMPLETED, std::memory_order_release);
    return true;
}

std::uint16_t TransportTaskExecutor::AllocateRequestCid()
{
    auto requestCid = nextRequestCid_.fetch_add(1, std::memory_order_relaxed);
    if (requestCid == 0) { requestCid = nextRequestCid_.fetch_add(1, std::memory_order_relaxed); }
    return requestCid;
}

Status TransportTaskExecutor::AssignSubBatchConnections(
    std::vector<TransportSubBatchContext>& subBatchContexts)
{
    for (auto& subBatchContext : subBatchContexts) {
        auto channel = connManager_->SelectConnection();
        if (!channel) {
            const auto subBatchStatus =
                Status::Error(StatusCode::CONNECTION_ERROR, "no available connection channel");
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      subBatchStatus);
            subBatchContext.state = TransportSubBatchState::COMPLETED;
            subBatchContext.status = subBatchStatus;
            return subBatchStatus;
        }

        subBatchContext.channel = channel;
    }
    return Status::OK();
}

bool TransportTaskExecutor::Execute(const TransportTaskPtr& task)
{
    TransportTaskState expected = TransportTaskState::PENDING;
    if (!task->state.compare_exchange_strong(expected, TransportTaskState::INFLIGHT,
                                             std::memory_order_acq_rel)) {
        return false;
    }

    std::vector<TransportSubBatchContext> subBatchContexts;
    auto status = PrepareTaskSubBatches(*task, subBatchContexts);

    if (status.ok()) { status = AssignSubBatchConnections(subBatchContexts); }

    std::vector<TransProvider::SendIoBatch> ioBatches;
    if (status.ok()) { BuildSubBatchSendBuffers(subBatchContexts, ioBatches); }

    if (!status.ok()) {
        KV_ERROR("Abort transport task before send task_id={} code={} message={}", task->taskId,
                 static_cast<int>(status.code), status.message);
    } else {
        SendSubBatchBuffers(subBatchContexts, ioBatches);
    }

    bool done = false;
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        if (task->Done()) {
            KV_DEBUG(
                "TransportTaskExecutor::Execute canceled during process task_id={} sub_batches={}",
                task->taskId, subBatchContexts.size());
            ReleaseAllSubBatchResources(subBatchContexts);
            return false;
        }

        if (!status.ok()) { AbortSubBatchesBeforeSend(*task, subBatchContexts); }
        *task->subBatchContexts = std::move(subBatchContexts);
        task->InitializeRemainingSubBatchCount();
        task->TryFinalizeFromSubBatches();
        KV_DEBUG(
            "TransportTaskExecutor::Execute submitted task_id={} op_type={} entries={} keys={} "
            "sub_batches={} done={} code={} message={}",
            task->taskId, static_cast<int>(task->opType), task->entries.size(), task->keys.size(),
            task->subBatchContexts->size(), task->Done(), static_cast<int>(task->finalStatus.code),
            task->finalStatus.message);

        done = task->Done();
    }
    return done;
}

bool TransportTaskExecutor::Poll(const TransportTaskPtr& task)
{
    if (!task) { return false; }

    bool done = false;
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        if (task->state.load(std::memory_order_acquire) != TransportTaskState::INFLIGHT) {
            return false;
        }
        if (task->subBatchContexts->empty()) { return false; }

        if (std::chrono::steady_clock::now() >= task->deadline) {
            const auto timeoutStatus =
                Status::Error(StatusCode::TIMEOUT, "transport task execution timeout");
            std::fill(task->entryStatus.begin(), task->entryStatus.end(), timeoutStatus);
            for (auto& subBatchContext : *task->subBatchContexts) {
                if (subBatchContext.state == TransportSubBatchState::COMPLETED) { continue; }

                std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                          timeoutStatus);
                connManager_->ReportFailure(subBatchContext.channel);
                CompleteSubBatch(*task, subBatchContext, timeoutStatus);
            }
            task->finalStatus = timeoutStatus;
            task->state.store(TransportTaskState::COMPLETED, std::memory_order_release);
        } else {
            for (auto& subBatchContext : *task->subBatchContexts) {
                if (subBatchContext.state == TransportSubBatchState::COMPLETED) { continue; }

                auto completeWithError = [this, &task, &subBatchContext](const Status& status) {
                    std::fill(subBatchContext.entryStatus.begin(),
                              subBatchContext.entryStatus.end(), status);
                    CompleteSubBatch(*task, subBatchContext, status);
                };

                std::uint16_t completedCid = 0;
                const void* responseData =
                    reinterpret_cast<void*>(subBatchContext.flagBuffer.local_addr);

                if (const auto status =
                        protocolManager_.PollResponseCid(responseData, completedCid);
                    !status.ok()) {
                    continue;
                }
                if (completedCid == 0 || completedCid != subBatchContext.cid) { continue; }

                KvResponse response;
                const auto batchNumber =
                    static_cast<std::uint16_t>(subBatchContext.entryStatus.size());
                if (const auto status = protocolManager_.UnpackResponse(
                        responseData, ToKvOpcode(subBatchContext.opType), batchNumber, response);
                    !status.ok()) {
                    completeWithError(status);
                    continue;
                }

                subBatchContext.status = KvResponseStatusToSubBatchStatus(response.status);
                FillEntryStatusFromCqeResult(response, subBatchContext);

                const bool queryResultBufferStatus =
                    subBatchContext.opType == AsuOpType::QUERY &&
                    subBatchContext.status.code == StatusCode::ASU_CQE_CHECK_RESULT_BUFFER;
                const auto status = subBatchContext.status.ok() || queryResultBufferStatus
                                        ? Status::OK()
                                        : subBatchContext.status;
                if (status.code == StatusCode::ASU_CQE_INTERNAL_ERROR ||
                    status.code == StatusCode::ASU_CQE_IO_TIMEOUT) {
                    connManager_->ReportFailure(subBatchContext.channel);
                } else {
                    connManager_->ReportSuccess(subBatchContext.channel);
                }
                CompleteSubBatch(*task, subBatchContext, status);
            }
            task->TryFinalizeFromSubBatches();
        }
        done = task->Done();
    }
    return done;
}

}  // namespace kv
