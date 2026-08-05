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
#include "transport_task_executor.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include "asu_response_status.h"
#include "connection_internal.h"
#include "logger.h"

namespace UC::ASU {

namespace {

constexpr std::size_t kFlagBufferHeaderCopySize = kCqeDwordCount * sizeof(std::uint32_t);

Status CopyDeviceToHost(const ScatterGatherEntry& sge, void* host, std::size_t size)
{
    if (size > sge.length) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "copy size exceeds buffer length");
    }
    const auto ret = aclrtMemcpy(host, size, reinterpret_cast<void*>(sge.device_addr), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "copy device memory to host failed ret=" + std::to_string(ret));
    }
    return Status::OK();
}

}  // namespace

TransportTaskExecutor::TransportTaskExecutor(
    const TransportConfig& config, IoScheduler& ioScheduler,
    const std::unique_ptr<TransProvider>& transProvider, BufferManager& sendBufferManager,
    BufferManager& flagBufferManager, const std::unique_ptr<ProtocolManager>& protocolManager,
    const std::unique_ptr<ConnectionManager>& connectionManager,
    std::atomic<std::uint16_t>& nextRequestCid, std::mutex& registeredRegionsMu,
    const std::unordered_map<MRHandle, RegisteredMemory>& registeredRegions)
    : config_(config),
      ioScheduler_(ioScheduler),
      transProvider_(transProvider),
      sendBufferManager_(sendBufferManager),
      flagBufferManager_(flagBufferManager),
      protocolManager_(protocolManager),
      connManager_(connectionManager),
      nextRequestCid_(nextRequestCid),
      registeredRegionsMu_(registeredRegionsMu),
      registeredRegions_(registeredRegions)
{
}

void TransportTaskExecutor::ReleaseSubBatchResources(TransportSubBatchContext& subBatchContext)
{
    if (subBatchContext.sendSge.slot_index != UINT32_MAX) {
        const auto slotIndex = subBatchContext.sendSge.slot_index;
        auto status = sendBufferManager_.Free(slotIndex);
        if (!status.ok()) {
            UC_ERROR("Failed to release sub-batch send buffer slot({}): {}", slotIndex,
                     status.message);
        }
        subBatchContext.sendSge = {};
    }

    if (subBatchContext.flagBuffer.slot_index != UINT32_MAX) {
        const auto slotIndex = subBatchContext.flagBuffer.slot_index;
        auto status = flagBufferManager_.Free(slotIndex);
        if (!status.ok()) {
            UC_ERROR("Failed to release sub-batch flag buffer slot({}): {}", slotIndex,
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

void TransportTaskExecutor::CompleteSubBatch(TransportTask& task,
                                             TransportSubBatchContext& subBatchContext,
                                             const Status& status)
{
    if (subBatchContext.state != TransportSubBatchState::PENDING) { return; }

    ReleaseSubBatchResources(subBatchContext);
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = status;
    --task.remainingSubBatchCount;
}

bool TransportTaskExecutor::Cancel(const TransportTaskPtr& task, const Status& status)
{
    if (!task) { return false; }

    std::lock_guard<std::mutex> lock(task->mutex);
    if (task->Done()) { return false; }
    std::fill(task->entryStatus.begin(), task->entryStatus.end(), status);
    for (auto& subBatchContext : *task->subBatchContexts) {
        if (subBatchContext.state != TransportSubBatchState::PENDING) { continue; }
        std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(), status);
    }
    task->finalStatus = status;
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
    Status status = Status::OK();
    for (auto& subBatchContext : subBatchContexts) {
        if (!subBatchContext.status.ok()) { continue; }

        auto channel = connManager_->SelectConnection();
        if (!channel) {
            const auto subBatchStatus =
                Status::Error(StatusCode::CONNECTION_ERROR, "no available connection channel");
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      subBatchStatus);
            subBatchContext.state = TransportSubBatchState::COMPLETED;
            subBatchContext.status = subBatchStatus;
            if (status.ok()) { status = subBatchStatus; }
            continue;
        }

        subBatchContext.channel = channel;
    }
    return status;
}

bool TransportTaskExecutor::Execute(const TransportTaskPtr& task)
{
    TransportTaskState expected = TransportTaskState::PENDING;
    if (!task->state.compare_exchange_strong(expected, TransportTaskState::INFLIGHT,
                                             std::memory_order_acq_rel)) {
        return false;
    }

    std::vector<TransportSubBatchContext> subBatchContexts;
    SubmitTaskRequests(*task, subBatchContexts);

    const bool hasSubBatches = !subBatchContexts.empty();
    if (hasSubBatches) {
        AssignSubBatchConnections(subBatchContexts);

        std::vector<TransProvider::SendIoBatch> ioBatches;
        std::vector<std::size_t> subBatchIndexes;
        BuildSubBatchSendBuffers(subBatchContexts, ioBatches, subBatchIndexes);
        SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes);
    }

    bool done = false;
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        if (task->Done()) {
            UC_DEBUG(
                "TransportTaskExecutor::Execute canceled during process task_id={} sub_batches={}",
                task->taskId, subBatchContexts.size());
            ReleaseAllSubBatchResources(subBatchContexts);
            return false;
        }

        if (hasSubBatches) { *task->subBatchContexts = std::move(subBatchContexts); }
        task->InitializeRemainingSubBatchCount();
        task->TryFinalizeFromSubBatches();
        UC_DEBUG(
            "TransportTaskExecutor::Execute submitted task_id={} op_type={} entries={} keys={} "
            "sub_batches={} done={} code={} message={}",
            task->taskId, static_cast<int>(task->opType), task->entries.size(), task->keys.size(),
            task->subBatchContexts->size(), task->Done(), static_cast<int>(task->finalStatus.code),
            task->finalStatus.message);

        for (auto& subBatchContext : *task->subBatchContexts) {
            if (subBatchContext.status.ok()) { continue; }
            ReleaseSubBatchResources(subBatchContext);
        }
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
                if (subBatchContext.state != TransportSubBatchState::PENDING) { continue; }

                std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                          timeoutStatus);
                connManager_->ReportFailure(subBatchContext.channel);
                CompleteSubBatch(*task, subBatchContext, timeoutStatus);
            }
            task->finalStatus = timeoutStatus;
            task->state.store(TransportTaskState::COMPLETED, std::memory_order_release);
        } else {
            for (auto& subBatchContext : *task->subBatchContexts) {
                if (subBatchContext.state != TransportSubBatchState::PENDING) { continue; }

                auto completeWithError = [this, &task, &subBatchContext](const Status& status) {
                    std::fill(subBatchContext.entryStatus.begin(),
                              subBatchContext.entryStatus.end(), status);
                    CompleteSubBatch(*task, subBatchContext, status);
                };

                std::uint16_t completedCid = 0;
                const void* responseData = nullptr;
                std::array<std::uint8_t, kFlagBufferHeaderCopySize> flagHeader{};
                std::vector<std::uint8_t> flagBuffer;
                if (subBatchContext.flagBuffer.memory_type == MemoryType::ASCEND_DEVICE) {
                    auto status = CopyDeviceToHost(subBatchContext.flagBuffer, flagHeader.data(),
                                                   flagHeader.size());
                    if (!status.ok()) {
                        // Without a readable header, this sub-batch cannot be polled or unpacked.
                        UC_ERROR(
                            "Copy flag buffer header from device failed cid={} code={} message={}",
                            subBatchContext.cid, static_cast<int>(status.code), status.message);
                        completeWithError(status);
                        continue;
                    }
                    responseData = flagHeader.data();
                } else {
                    responseData = reinterpret_cast<void*>(subBatchContext.flagBuffer.local_addr);
                }

                if (const auto status =
                        protocolManager_->PollResponseCid(responseData, completedCid);
                    !status.ok()) {
                    continue;
                }
                if (completedCid == 0 || completedCid != subBatchContext.cid) { continue; }

                if (subBatchContext.flagBuffer.memory_type == MemoryType::ASCEND_DEVICE) {
                    // The header matched; copy the full CQE before unpacking entry status.
                    flagBuffer.resize(subBatchContext.flagBuffer.length);
                    auto status = CopyDeviceToHost(subBatchContext.flagBuffer, flagBuffer.data(),
                                                   flagBuffer.size());
                    if (!status.ok()) {
                        // The matched CQE cannot be decoded without the complete flag buffer.
                        UC_ERROR("Copy flag buffer from device failed cid={} code={} message={}",
                                 subBatchContext.cid, static_cast<int>(status.code),
                                 status.message);
                        completeWithError(status);
                        continue;
                    }
                    responseData = flagBuffer.data();
                }

                KvResponse response;
                const auto batchNumber =
                    static_cast<std::uint16_t>(subBatchContext.entryStatus.size());
                if (const auto status = protocolManager_->UnpackResponse(
                        responseData, ToKvOpcode(subBatchContext.opType), batchNumber, response);
                    !status.ok()) {
                    completeWithError(status);
                    continue;
                }

                subBatchContext.status = KvResponseStatusToSubBatchStatus(response.status);
                FillEntryStatusFromCqeResult(response, subBatchContext);

                const bool queryResultBufferStatus =
                    subBatchContext.opType == TransportOpType::QUERY &&
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

}  // namespace UC::ASU
