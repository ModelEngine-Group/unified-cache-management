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
#include <iterator>
#include <string>
#include <utility>
#include "conn/connection_internal.h"
#include "kv_metrics/metrics.h"
#include "logger.h"
#include "utils/trans_task_utils.h"

namespace kv {

TransportTaskExecutor::TransportTaskExecutor(
    const TransportConfig& config, const std::shared_ptr<TransProvider>& transProvider,
    const std::unique_ptr<ConnectionManager>& connectionManager)
    : config_(config),
      ioScheduler_(config),
      transProvider_(transProvider),
      connManager_(connectionManager),
      asuSendMetric_("kv_transport_node_task_send_duration_seconds",
                     {
                         {"node_id", std::to_string(config.nodeId)}
}),
      asuCompletionMetric_("kv_transport_node_task_completion_duration_seconds",
                           {{"node_id", std::to_string(config.nodeId)}})
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
    if (!transProvider_) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "TransProvider is null");
    }

    (void)metrics::RegisterMetricLabels(asuSendMetric_.Name(), asuSendMetric_.Labels());
    (void)metrics::RegisterMetricLabels(asuCompletionMetric_.Name(), asuCompletionMetric_.Labels());
    // The provider decides whether its control buffers need the device address space or plain host
    // memory; the transport must not assume one or the other.
    const MemoryType controlMemory = transProvider_->ControlBufferMemory();
    auto status = sendBufferManager_.Init("asu send buffer", controlMemory,
                                          config_.sendBufferSlotSize, config_.sendBufferSlotNum);
    if (!status.ok()) {
        (void)Shutdown();
        return status;
    }

    status = flagBufferManager_.Init("asu flag buffer", controlMemory, config_.flagBufferSlotSize,
                                     config_.flagBufferSlotNum);
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
    const auto unsentStatus = Status::Error(StatusCode::IN_PROGRESS, "sub-batch unsent");

    bool hasFailedSubBatch = false;
    for (auto& subBatchContext : subBatchContexts) {
        ReleaseSubBatchResources(subBatchContext);
        if (!subBatchContext.status.ok()) {
            if (!hasFailedSubBatch) {
                hasFailedSubBatch = true;
                // 部分中止：首个失败 subBatch 的未发送 Entry 保留 IN_PROGRESS，
                // 其后 subBatch 标为 CANCELED，不整批全弃。
                std::fill(subBatchContext.entryStatus.begin(),
                          subBatchContext.entryStatus.end(), unsentStatus);
            } else {
                std::fill(subBatchContext.entryStatus.begin(),
                          subBatchContext.entryStatus.end(), canceledStatus);
                subBatchContext.status = canceledStatus;
            }
        } else if (hasFailedSubBatch) {
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      canceledStatus);
            subBatchContext.status = canceledStatus;
        }
        subBatchContext.state = TransportSubBatchState::COMPLETED;
    }
    (void)task;
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
            metrics::UpdateStats(KV_METRIC("kv_transport_task_connection_errors_total"), 1.0);
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
    const auto processingStartedAt = std::chrono::steady_clock::now();
    task->asuCompletionMetric = &asuCompletionMetric_;

    std::vector<TransportSubBatchContext> subBatchContexts;
    auto status = PrepareTaskSubBatches(*task, subBatchContexts);

    if (status.ok()) { status = AssignSubBatchConnections(subBatchContexts); }

    std::vector<TransProvider::SendIoBatch> ioBatches;
    if (status.ok()) { BuildSubBatchSendBuffers(subBatchContexts, ioBatches); }

    if (!status.ok()) {
        KV_ERROR("Abort transport task before send task_id={} code={} message={}", task->taskId,
                 static_cast<int>(status.code), status.message);
    } else if (!ioBatches.empty()) {
        const auto preSendAt = std::chrono::steady_clock::now();
        const metrics::MetricUpdate preSendUpdates[] = {
            {KV_METRIC("kv_transport_task_pre_send_duration_seconds"),
             std::chrono::duration<double>(preSendAt - task->submittedAt).count()          },
            {KV_METRIC("kv_transport_task_queue_duration_seconds"),
             std::chrono::duration<double>(processingStartedAt - task->submittedAt).count()},
            {KV_METRIC("kv_transport_task_process_duration_seconds"),
             std::chrono::duration<double>(preSendAt - processingStartedAt).count()        },
        };
        metrics::UpdateStats(preSendUpdates, std::size(preSendUpdates));
        SendSubBatchBuffers(subBatchContexts, ioBatches, *task);
        task->sendCompletedAt = std::chrono::steady_clock::now();
        task->sendReturned.store(true, std::memory_order_release);
        const metrics::MetricUpdate sendUpdates[] = {
            {KV_METRIC("kv_transport_task_send_duration_seconds"),
             std::chrono::duration<double>(task->sendCompletedAt - task->submittedAt).count()},
            {asuSendMetric_,
             std::chrono::duration<double>(task->sendCompletedAt - task->submittedAt).count()},
        };
        metrics::UpdateStats(sendUpdates, std::size(sendUpdates));
        if (task->onSendComplete) { task->onSendComplete(); }
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

        const auto now = std::chrono::steady_clock::now();
        const bool taskExpired = now >= task->deadline;

        for (auto& subBatchContext : *task->subBatchContexts) {
            if (subBatchContext.state == TransportSubBatchState::COMPLETED) { continue; }
            // RETRY_BACKOFF：挂起重试的 subBatch 不在此处轮询响应，
            // 由 HasRetryWork/ExecuteRetry 非阻塞驱动。
            if (subBatchContext.pendingRetryCategory != RetryCategory::NONE) { continue; }
            // READY（未发送）：无在途 attempt 可裁决。
            if (!subBatchContext.InFlight()) { continue; }

            const auto completeWithSynthetic = [this, &task,
                                                &subBatchContext](const Status& synthetic,
                                                                  bool recordFailure) {
                subBatchContext.status = synthetic;
                std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                          synthetic);
                const auto decision =
                    ErrorClassifier::ClassifySubBatch(subBatchContext, synthetic);
                if (connManager_ && recordFailure &&
                    decision.healthAction == ChannelHealthAction::RECORD_FAILURE) {
                    connManager_->ReportFailure(subBatchContext.channel);
                }
                if (!RouteSubBatchFailure(*task, subBatchContext, decision)) {
                    CompleteFailedSubBatch(*task, subBatchContext);
                }
            };

            std::uint16_t completedCid = 0;
            const void* responseData =
                reinterpret_cast<void*>(subBatchContext.flagBuffer.local_addr);

            // 迟到响应防线：Channel 已排空/已替换后到达的响应合成
            // CQE_IO_TIMEOUT（不以旧对象裁决新对象），且不记账迟到响应。
            const bool channelDrained = subBatchContext.channel != nullptr &&
                                        subBatchContext.channel->GetState() !=
                                            ChannelState::ACTIVE;
            bool haveResponse = false;
            if (!channelDrained &&
                protocolManager_.PollResponseCid(responseData, completedCid).ok() &&
                completedCid != 0 && completedCid == subBatchContext.cid) {
                haveResponse = true;
            }

            if (haveResponse) {
                KvResponse response;
                const auto batchNumber =
                    static_cast<std::uint16_t>(subBatchContext.entryStatus.size());
                if (const auto status = protocolManager_.UnpackResponse(
                        responseData, ToKvOpcode(subBatchContext.opType), batchNumber, response);
                    !status.ok()) {
                    // 解码本身失败按兜底换链（未知 raw 不折叠成 IO_ERROR）。
                    completeWithSynthetic(status, true);
                    continue;
                }

                subBatchContext.status = KvResponseStatusToSubBatchStatus(response.status);
                FillEntryStatusFromCqeResult(response, subBatchContext);

                // 正常结果前置排除（含 query miss）：不重试、不记失败。
                if (ErrorClassifier::IsNormalResult(subBatchContext)) {
                    if (connManager_) { connManager_->ReportSuccess(subBatchContext.channel); }
                    if (subBatchContext.status.code == StatusCode::CQE_CHECK_RESULT_BUFFER) {
                        CompleteSubBatch(*task, subBatchContext, Status::OK());
                    } else {
                        CompleteSubBatch(*task, subBatchContext, subBatchContext.status);
                    }
                    continue;
                }

                const auto decision =
                    ErrorClassifier::ClassifySubBatch(subBatchContext, subBatchContext.status);
                KV_DEBUG(
                    "TransportTaskExecutor classified failure task_id={} cid={} op_type={} "
                    "category={} retry_action={} fallback={} multiple_codes={}",
                    task->taskId, subBatchContext.cid, static_cast<int>(subBatchContext.opType),
                    static_cast<int>(decision.category), static_cast<int>(decision.retryAction),
                    decision.fallback, decision.hasMultipleErrorCodes);
                if (connManager_ &&
                    decision.healthAction == ChannelHealthAction::RECORD_FAILURE) {
                    connManager_->ReportFailure(subBatchContext.channel);
                }
                if (!RouteSubBatchFailure(*task, subBatchContext, decision)) {
                    CompleteFailedSubBatch(*task, subBatchContext);
                }
                continue;
            }

            if (channelDrained) {
                metrics::UpdateStats(KV_METRIC("kv_transport_task_io_timeouts_total"), 1.0);
                completeWithSynthetic(Status::Error(StatusCode::CQE_IO_TIMEOUT,
                                                    "late CQE on drained channel"),
                                      false);
                continue;
            }

            // attempt 超时裁决：已实际 Send 后的 attempt 在自身 deadline 内无响应，
            // 按 CQE_IO_TIMEOUT 处理（换链并记链路失败）。
            if (now >= subBatchContext.attemptDeadline) {
                metrics::UpdateStats(KV_METRIC("kv_transport_task_io_timeouts_total"), 1.0);
                completeWithSynthetic(Status::Error(StatusCode::CQE_IO_TIMEOUT,
                                                    "attempt timed out waiting for CQE"),
                                      true);
                continue;
            }
        }

        if (taskExpired) {
            // 总Deadline到期：未完成且不在途的 Entry=TIMEOUT，不再启动新 attempt。
            FinalizeUnsentOnTaskDeadline(*task);
        }
        task->TryFinalizeFromSubBatches();
        done = task->Done();
    }
    return done;
}

std::chrono::steady_clock::time_point TransportTaskExecutor::ComputeAttemptDeadline(
    const TransportTask& task) const
{
    const auto now = std::chrono::steady_clock::now();
    // per-attempt deadline = min(attemptTimeoutMs, 剩余总时间)。
    const auto remainingMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 task.deadline - now)
                                 .count();
    const auto attemptMs = std::min<std::int64_t>(
        static_cast<std::int64_t>(config_.retry.attemptTimeoutMs),
        std::max<std::int64_t>(remainingMs, 0));
    return now + std::chrono::milliseconds(attemptMs);
}

void TransportTaskExecutor::ReleaseAttemptResources(TransportSubBatchContext& subBatchContext)
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

    // 释放旧 attempt 的链路 inflight；channel 字段保留给原链/换链重试。
    if (subBatchContext.channel != nullptr) { subBatchContext.channel->ReleaseInflight(); }
}

void TransportTaskExecutor::CompleteFailedSubBatch(TransportTask& task,
                                                  TransportSubBatchContext& subBatchContext)
{
    // 耗尽保留原始错误：status/entryStatus 保持分类前的原始值。
    CompleteSubBatch(task, subBatchContext, subBatchContext.status);
}

bool TransportTaskExecutor::RouteSubBatchFailure(TransportTask& task,
                                                TransportSubBatchContext& subBatchContext,
                                                const ErrorDecision& decision)
{
    const auto now = std::chrono::steady_clock::now();
    // 总Deadline到期：不再启动新 attempt。
    if (now >= task.deadline) { return false; }

    switch (decision.retryAction) {
        case RetryAction::RETRY_SAME_CHANNEL: {
            // 原链 BUSY 重试 <= M 次；最终 Busy 不换链。
            if (subBatchContext.busyRetries >= config_.retry.busyRetryCount) { return false; }
            subBatchContext.busyRetries += 1;
            subBatchContext.pendingRetryCategory = RetryCategory::BUSY;
            // 增长型退避：min(busyRetryBackoffMaxMs, busyRetryBackoffBaseMs*2^(k-1))。
            const auto exponent =
                std::min<std::uint32_t>(subBatchContext.busyRetries - 1, 20u);
            const auto backoffMs = std::min<std::uint64_t>(
                config_.retry.busyRetryBackoffMaxMs,
                static_cast<std::uint64_t>(config_.retry.busyRetryBackoffBaseMs) << exponent);
            subBatchContext.retryAt = now + std::chrono::milliseconds(backoffMs);
            KV_DEBUG("TransportTaskExecutor BUSY retry scheduled task_id={} cid={} "
                     "busy_retries={} backoff_ms={}",
                     task.taskId, subBatchContext.cid, subBatchContext.busyRetries, backoffMs);
            break;
        }
        case RetryAction::FAILOVER: {
            // RA 与链路故障共用 FAILOVER 力学（无独立计数），排除已试 Channel，
            // 无退避立即换链；耗尽保留原始错误。
            subBatchContext.attemptedChannels.push_back(subBatchContext.channel.get());
            if (subBatchContext.attemptedChannels.size() > config_.retry.failoverRetryCount) {
                return false;
            }
            subBatchContext.pendingRetryCategory = RetryCategory::LINK_FAILURE;
            subBatchContext.retryAt = now;
            KV_DEBUG("TransportTaskExecutor FAILOVER scheduled task_id={} cid={} "
                     "attempted_channels={}",
                     task.taskId, subBatchContext.cid,
                     subBatchContext.attemptedChannels.size());
            break;
        }
        default:
            // RETURN_RESULT（应用类/排除项）：直接返回。
            return false;
    }

    ReleaseAttemptResources(subBatchContext);
    return true;
}

void TransportTaskExecutor::FinalizeUnsentOnTaskDeadline(TransportTask& task)
{
    const auto timeoutStatus =
        Status::Error(StatusCode::TIMEOUT, "transport task execution timeout");
    bool finalizedUnsent = false;
    bool anyNonTerminal = false;

    for (auto& subBatchContext : *task.subBatchContexts) {
        if (subBatchContext.state == TransportSubBatchState::COMPLETED) { continue; }
        if (subBatchContext.InFlight()) {
            // 在途 attempt 由自身 attemptDeadline 裁决（必然早于或等于总 Deadline）。
            anyNonTerminal = true;
            continue;
        }
        // 未发/重试挂起：TIMEOUT 终结，不换链、不记链路失败、不重试。
        std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                  timeoutStatus);
        CompleteSubBatch(task, subBatchContext, timeoutStatus);
        finalizedUnsent = true;
    }

    if (finalizedUnsent && !anyNonTerminal) {
        for (auto& subBatchContext : *task.subBatchContexts) {
            if (subBatchContext.state != TransportSubBatchState::COMPLETED) {
                anyNonTerminal = true;
                break;
            }
        }
    }

    if (finalizedUnsent && !anyNonTerminal) {
        // 未发超时：Executor 直接 TaskResult.status=TIMEOUT 终结。
        metrics::UpdateStats(KV_METRIC("kv_transport_task_completion_timeouts_total"), 1.0);
        task.finalStatus = timeoutStatus;
        task.state.store(TransportTaskState::COMPLETED, std::memory_order_release);
    }
}

Status TransportTaskExecutor::PrepareRetryRequest(TransportSubBatchContext& subBatchContext)
{
    if (!subBatchContext.request) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "sub-batch request was not retained for retry");
    }

    const auto opcode = ToKvOpcode(subBatchContext.opType);
    const auto batchNumber = static_cast<std::size_t>(subBatchContext.entryStatus.size());

    // 新的响应 flag buffer。
    auto status = AllocateSubBatchFlagBuffer(opcode, batchNumber, flagBufferManager_,
                                             subBatchContext);
    if (!status.ok()) { return status; }

    // 新 CID + SQE/响应 Buffer/MR 同步更新（不得只改 context.cid）。
    const auto cid = AllocateRequestCid();
    subBatchContext.cid = cid;
    subBatchContext.request->ApplyRetry(cid, subBatchContext.flagBuffer.device_addr,
                                        subBatchContext.flagBuffer.tokenId);

    // 新 send buffer 并重打包 SQE。
    const auto packedSize = protocolManager_.GetPackedSize(opcode, *subBatchContext.request);
    status = sendBufferManager_.Allocate(packedSize, subBatchContext.sendSge);
    if (!status.ok()) {
        KV_ERROR("PrepareRetryRequest allocate send buffer failed cid={} packed_size={} code={}",
                 cid, packedSize, static_cast<int>(status.code));
        return status;
    }
    status = protocolManager_.PackRequest(
        reinterpret_cast<void*>(subBatchContext.sendSge.local_addr), opcode,
        *subBatchContext.request);
    if (!status.ok()) {
        auto freeStatus = sendBufferManager_.Free(subBatchContext.sendSge.slot_index);
        if (!freeStatus.ok()) {
            KV_ERROR("PrepareRetryRequest free send buffer slot failed: {}", freeStatus.message);
        }
        subBatchContext.sendSge = {};
        return status;
    }
    return Status::OK();
}

bool TransportTaskExecutor::HasRetryWork(const TransportTaskPtr& task)
{
    if (!task || task->Done()) { return false; }
    const auto now = std::chrono::steady_clock::now();
    if (now >= task->deadline) { return false; }  // 总Deadline到期不再启动新 attempt
    std::lock_guard<std::mutex> lock(task->mutex);
    for (auto& subBatchContext : *task->subBatchContexts) {
        if (subBatchContext.state == TransportSubBatchState::COMPLETED) { continue; }
        if (subBatchContext.pendingRetryCategory != RetryCategory::NONE &&
            now >= subBatchContext.retryAt) {
            return true;
        }
    }
    return false;
}

namespace {

std::uint32_t GetRetrySendCountAttr(const std::unordered_map<std::string, std::string>& attrs,
                                    const std::string& name)
{
    auto it = attrs.find(name);
    if (it == attrs.end()) { return 0; }
    return static_cast<std::uint32_t>(std::stoull(it->second, nullptr, 0));
}

}  // namespace

void TransportTaskExecutor::ExecuteRetry(const TransportTaskPtr& task)
{
    if (!task) { return; }

    std::vector<TransportSubBatchContext*> due;
    {
        std::lock_guard<std::mutex> lock(task->mutex);
        if (task->Done()) { return; }
        const auto now = std::chrono::steady_clock::now();
        if (now >= task->deadline) { return; }  // 不再启动新 attempt
        for (auto& subBatchContext : *task->subBatchContexts) {
            if (subBatchContext.state == TransportSubBatchState::COMPLETED) { continue; }
            if (subBatchContext.pendingRetryCategory != RetryCategory::NONE &&
                now >= subBatchContext.retryAt) {
                due.push_back(&subBatchContext);
            }
        }
    }

    for (auto* subBatchContext : due) { ExecuteOneRetry(*task, *subBatchContext); }
}

void TransportTaskExecutor::ExecuteOneRetry(TransportTask& task,
                                            TransportSubBatchContext& subBatchContext)
{
    RetryCategory category = RetryCategory::NONE;
    std::vector<ConnectionChannel*> attempted;
    {
        std::lock_guard<std::mutex> lock(task.mutex);
        if (task.Done()) { return; }
        category = subBatchContext.pendingRetryCategory;
        // 发送前复位为 NONE：新 attempt 的响应到达后按结果重新设置，
        // 不依赖 lastError 反查（退避期间可能被迟到响应覆盖）。
        subBatchContext.pendingRetryCategory = RetryCategory::NONE;
        attempted = subBatchContext.attemptedChannels;
    }

    std::shared_ptr<ConnectionChannel> channel;
    if (!connManager_) {
        // 无连接管理器（仅测试构造场景）：保留原始错误终结。
        std::lock_guard<std::mutex> lock(task.mutex);
        CompleteFailedSubBatch(task, subBatchContext);
        return;
    }
    if (category == RetryCategory::BUSY) {
        // 原链重试：不重置 channel 字段。
        channel = connManager_->AcquireChannel(subBatchContext.channel);
        if (!channel) {
            // 原链不可用（已排空/达上限）：排除已试链路后兜底换链。
            auto excluded = attempted;
            if (subBatchContext.channel) {
                excluded.push_back(subBatchContext.channel.get());
            }
            channel = connManager_->SelectConnection(excluded);
        }
    } else {
        // 换链：两级优先（跨Group优先 -> 同Group兜底），对象身份排除。
        channel = connManager_->SelectConnection(attempted);
    }

    if (!channel) {
        // 满载无候选 fail-fast：保留原始错误。
        std::lock_guard<std::mutex> lock(task.mutex);
        CompleteFailedSubBatch(task, subBatchContext);
        return;
    }

    subBatchContext.channel = channel;

    auto status = PrepareRetryRequest(subBatchContext);
    if (!status.ok()) {
        KV_ERROR("PrepareRetryRequest failed task_id={} cid={} code={}", task.taskId,
                 subBatchContext.cid, static_cast<int>(status.code));
        std::lock_guard<std::mutex> lock(task.mutex);
        subBatchContext.status = status;
        std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(), status);
        CompleteFailedSubBatch(task, subBatchContext);
        return;
    }

    subBatchContext.attemptDeadline = ComputeAttemptDeadline(task);

    // Provider 调用锁外执行（禁止持有 task->mutex 或结构锁）。
    const auto kernelCount = GetRetrySendCountAttr(config_.attrs, "kernel_count");
    const auto quietCount = GetRetrySendCountAttr(config_.attrs, "quiet_count");
    const auto sendStartedAt = std::chrono::steady_clock::now();
    const auto sendStatuses = transProvider_->Send(
        {TransProvider::SendIoBatch{
            channel->GetConnection(),
            reinterpret_cast<void*>(subBatchContext.sendSge.device_addr),
            reinterpret_cast<void*>(subBatchContext.flagBuffer.device_addr),
            subBatchContext.sendSge.length}},
        kernelCount, quietCount);
    metrics::UpdateStats(KV_METRIC("kv_transport_task_retry_send_duration_seconds"),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                       sendStartedAt)
                             .count());

    if (sendStatuses.size() != 1) {
        KV_ERROR("Transport retry send returned unexpected status count expected=1 actual={}",
                 sendStatuses.size());
        const auto failure = Status::Error(StatusCode::INTERNAL_ERROR,
                                           "transport send returned unexpected status count");
        std::lock_guard<std::mutex> lock(task.mutex);
        subBatchContext.status = failure;
        std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                  failure);
        CompleteFailedSubBatch(task, subBatchContext);  // 本地执行错误：直接终结
        return;
    }
    if (sendStatuses[0].ok()) { return; }

    // Send 同步失败：按链路故障分类路由（CONNECTION_ERROR -> 换链）。
    KV_ERROR("Retry send failed task_id={} cid={} code={} message={}", task.taskId,
             subBatchContext.cid, static_cast<int>(sendStatuses[0].code),
             sendStatuses[0].message);
    std::lock_guard<std::mutex> lock(task.mutex);
    subBatchContext.status = sendStatuses[0];
    std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
              sendStatuses[0]);
    const auto decision =
        ErrorClassifier::ClassifySubBatch(subBatchContext, sendStatuses[0]);
    if (decision.healthAction == ChannelHealthAction::RECORD_FAILURE) {
        connManager_->ReportFailure(channel);
    }
    if (!RouteSubBatchFailure(task, subBatchContext, decision)) {
        CompleteFailedSubBatch(task, subBatchContext);
    }
}

}  // namespace kv
