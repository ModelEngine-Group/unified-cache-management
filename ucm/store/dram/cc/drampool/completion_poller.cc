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
#include "completion_poller.h"
#include <algorithm>
#include <thread>
#include <utility>
#include "core/transport_manager.h"
#include "drampool_config.h"
#include "drampool_metrics.h"
#include "logger/logger.h"
#include "metadata.h"

namespace UC::DramPool {
namespace {

void ReleaseResponseBuffer(BufferPool& flagBufferPool, CompletionRecord& record)
{
    const auto releasedSlot = record.local_resp_slot.slotIndex;
    const auto releaseStatus = flagBufferPool.Free(releasedSlot);
    record.local_resp_slot = {};
    if (releaseStatus.Failure()) {
        UC_ERROR(
            "CompletionPoller release response flag buffer failed, request_id={}, slot={}, "
            "error={}",
            record.request_id, releasedSlot, releaseStatus);
    }
}

// Batch outcome counters, reported once at the first SubmitResponse entry
// thanks to the batch_outcome_reported flag, so flag-pool NoSpace retries that
// re-enter SubmitResponse do not double-report. The batch end-to-end duration
// is reported separately on the terminal paths (see ReportBatchEndToEnd).
void ReportBatchMetrics(CompletionRecord& record)
{
    if (record.batch_outcome_reported) { return; }
    record.batch_outcome_reported = true;

    switch (record.opcode) {
        case OpType::DUMP: {
            const auto failedEntries = std::count(record.results.begin(), record.results.end(),
                                                  static_cast<std::uint8_t>(
                                                      DumpLoadResult::Failed));
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kDumpFailedEntriesTotal),
                                     static_cast<double>(failedEntries));
            break;
        }
        case OpType::LOOKUP: {
            // Misses are derived from the response vector (every slot starts as
            // NotFound), keeping the lookup scan and its scan-duration timer free
            // of bookkeeping.
            const auto missEntries = std::count(record.results.begin(), record.results.end(),
                                                static_cast<std::uint8_t>(
                                                    LookupResult::NotFound));
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kLookupMissEntriesTotal),
                                     static_cast<double>(missEntries));
            break;
        }
        default:
            break;
    }
}

// Batch end-to-end duration: the full server-side request lifecycle, from the
// requestQueue push instant (enqueue_us, stamped by the RequestReceiver) to the
// terminal observation — the response transfer reaching a terminal state, or
// the record leaving the Poller on a permanent response failure. Covers queue
// residence, worker processing, transfer wait, response-buffer waits, response
// packing/submission, and response transfer completion. Reported once per
// record via the enqueue_us sentinel (zeroed after reporting).
void ReportBatchEndToEnd(CompletionRecord& record)
{
    if (record.enqueue_us == 0) { return; }
    const auto elapsedMs = (SteadyNowUs() - record.enqueue_us) / 1000.0;
    switch (record.opcode) {
        case OpType::DUMP:
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kDumpBatchTotalDurationMs), elapsedMs);
            break;
        case OpType::LOAD:
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kLoadBatchTotalDurationMs), elapsedMs);
            break;
        case OpType::LOOKUP:
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kLookupBatchTotalDurationMs), elapsedMs);
            break;
        default:
            break;
    }
    record.enqueue_us = 0;
}

// Data-transfer terminal observation: covers the
// GetStatus-failure / Failed / Completed terminal paths. submit_ms still holds the
// data-transfer submission time; SubmitResponse overwrites it afterwards.
void ReportDataTransferMetrics(const CompletionRecord& record,
                               transport::TransferStatus terminalStatus)
{
    if (record.opcode == OpType::DUMP) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kDumpTransferDurationMs),
                                 static_cast<double>(SteadyNowMs() - record.submit_ms));
    } else if (record.opcode == OpType::LOAD) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kLoadTransferDurationMs),
                                 static_cast<double>(SteadyNowMs() - record.submit_ms));
    }
    if (terminalStatus != transport::TransferStatus::Completed) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kTransferFailuresTotal), 1);
    }
}

// Response-buffer release and RTT terminal exit shared by every terminal path:
// observed from the response-transfer submission
// (submit_ms, set in SubmitResponse) to the terminal state. Failed responses only
// bump the failure counter: their latency is not a meaningful RTT observation.
// Reaching this settle point means the response transfer is terminal, so the
// batch end-to-end duration is also reported here.
void SettleResponseTransfer(BufferPool& flagBufferPool, CompletionRecord& record, bool failed)
{
    ReportBatchEndToEnd(record);
    ReleaseResponseBuffer(flagBufferPool, record);
    if (failed) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kResponseFailuresTotal), 1);
        return;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kResponseRttMs),
                             static_cast<double>(SteadyNowMs() - record.submit_ms));
}

}  // namespace

void CompletionPoller::Run(const std::atomic_bool& stop)
{
    while (true) {
        FillPendingWindow();

        const bool stopRequested = stop.load(std::memory_order_acquire);

        if (pending_.empty()) {
            if (stopRequested) { break; }
            std::this_thread::sleep_for(kThreadIdleSleepDuration);
            continue;
        }

        PollPendingCompletions();
    }
}

void CompletionPoller::FillPendingWindow()
{
    while (pending_.size() < g_config.pollerPendingDepth) {
        CompletionRecord record;
        if (!runtime_.completionQueue.TryPop(record)) { break; }
        g_completionQueueLen.fetch_sub(1, std::memory_order_relaxed);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kQueueCompletionSize),
                                 static_cast<double>(g_completionQueueLen.load()));
        pending_.emplace_back(std::move(record));
    }
}

void CompletionPoller::PollPendingCompletions()
{
    const std::size_t scanCount = pending_.size();
    auto iter = pending_.begin();

    // Scan the whole pending snapshot. Completed records free slots that are refilled
    // from completionQueue at the beginning of the next round.
    for (std::size_t scanned = 0; scanned < scanCount; ++scanned) {
        switch (iter->stage) {
            case CompletionStage::PollDataTransfer:
                if (!PollDataTransfer(*iter)) {
                    // The transfer is still in-flight, poll it next round.
                    ++iter;
                    break;
                }
                // Data settlement moves the record to SubmitResponse. Submit its response
                // in this scan instead of deferring it to the next poll round.
                [[fallthrough]];
            case CompletionStage::SubmitResponse:
                if (SubmitResponse(*iter)) {
                    // Returns true if the record is done and should be erased (permanent failure).
                    iter = pending_.erase(iter);
                } else {
                    // Returns false if the record should remain pending (success or temporary
                    // failure).
                    ++iter;
                }
                break;
            case CompletionStage::PollResponseTransfer:
                if (PollResponseTransfer(*iter)) {
                    iter = pending_.erase(iter);
                } else {
                    ++iter;
                }
                break;
            default:
                UC_ERROR("CompletionPoller got invalid completion stage, request_id={}, stage={}",
                         iter->request_id, static_cast<int>(iter->stage));
                iter = pending_.erase(iter);
                break;
        }
    }

    // Round-tail gauges: overwrite-write is safe here
    // because the CompletionPoller thread is the sole writer of both names.
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kQueueCompletionInflight),
                             static_cast<double>(pending_.size()));
    if (g_config.flagBufferSlotCount > 0) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kFlagPoolUsageRatio),
                                 static_cast<double>(runtime_.flagBufferPool.GetUsedCount()) /
                                     g_config.flagBufferSlotCount);
    }
}

bool CompletionPoller::PollDataTransfer(CompletionRecord& record)
{
    transport::TransferStatus transportStatus = transport::TransferStatus::Failed;
    const auto getStatusStartUs = SteadyNowUs();
    const auto queryStatus = runtime_.transport.GetStatus(record.data_handle, transportStatus);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kGetStatusDurationMs),
                             static_cast<double>(SteadyNowUs() - getStatusStartUs) / 1000.0);
    if (queryStatus.Failure()) {
        // GetStatus removes failed handles, so an API failure is also terminal.
        UC_ERROR(
            "CompletionPoller data transfer GetStatus failed, request_id={}, handle={}, error={}",
            record.request_id, record.data_handle, queryStatus);
        SettleDataTransfer(record, transport::TransferStatus::Failed);
        record.data_handle = transport::kInvalidTransferHandle;
        record.stage = CompletionStage::SubmitResponse;
        return true;
    }

    if (transportStatus == transport::TransferStatus::Waiting) {
        // A timeout is diagnostic only. The transfer may still own its handle and buffers, and
        // DramStore is solely responsible for initiating connection teardown.
        if (!record.timeout_reported && OperationTimedOut(record, SteadyNowMs())) {
            record.timeout_reported = true;
            UC_ERROR(
                "CompletionPoller data transfer timed out, request_id={}, peer={}, handle={}, "
                "timeout_ms={}; waiting for Store-initiated disconnect or terminal transport "
                "status",
                record.request_id, record.peer_one_sided_id, record.data_handle,
                g_config.opTimeoutMs);
        }
        return false;
    }

    // A terminal GetStatus releases the data handle before business state is settled.
    UC_DEBUG("CompletionPoller data transfer finished, request_id={}, handle={}, status={}",
             record.request_id, record.data_handle, static_cast<int>(transportStatus));
    SettleDataTransfer(record, transportStatus);
    record.data_handle = transport::kInvalidTransferHandle;
    record.stage = CompletionStage::SubmitResponse;
    UC_DEBUG("CompletionPoller advances to SubmitResponse, request_id={}", record.request_id);
    return true;
}

bool CompletionPoller::SubmitResponse(CompletionRecord& record)
{
    ReportBatchMetrics(record);

    const auto packedSize =
        runtime_.protocol.GetPackedResponseSize(record.opcode, record.results.size());
    auto allocateStatus = runtime_.flagBufferPool.Allocate(record.local_resp_slot);
    if (allocateStatus.Failure()) {
        if (allocateStatus.Underlying() == Status::NoSpace().Underlying()) {
            // Flag pool full: the record stays pending and is retried next round
            // (blocking chain B2), not counted as a response failure.
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kQueueResponseBufferRetryTotal), 1);
            UC_WARN(
                "CompletionPoller flag buffer pool full, request_id={}, opcode={}, error={}, "
                "retrying next round",
                record.request_id, static_cast<int>(record.opcode), allocateStatus);
            return false;
        }

        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kResponseFailuresTotal), 1);
        UC_ERROR(
            "CompletionPoller flag buffer allocation failed, request_id={}, opcode={}, error={}",
            record.request_id, static_cast<int>(record.opcode), allocateStatus);
        // Permanent failure: the record leaves the Poller without a response
        // transfer, so report the batch end-to-end duration here.
        ReportBatchEndToEnd(record);
        return true;
    }
    UC_DEBUG("CompletionPoller allocated response slot, request_id={}, slot={}", record.request_id,
             record.local_resp_slot.slotIndex);

    const auto len = static_cast<std::uint32_t>(packedSize);

    const auto protocolStatus =
        runtime_.protocol.PackResponse(record.local_resp_slot.localAddr, record.opcode,
                                       KvResponse{record.request_id, record.results});
    if (protocolStatus.Failure()) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kResponseFailuresTotal), 1);
        UC_ERROR("CompletionPoller SubmitResponse pack failed, request_id={}, opcode={}, error={}",
                 record.request_id, static_cast<int>(record.opcode), protocolStatus);
        ReportBatchEndToEnd(record);
        return true;
    }
    UC_DEBUG("CompletionPoller packed response, request_id={}, response_len={}", record.request_id,
             len);

    transport::Operation operation;
    operation.opcode = transport::Opcode::Write;
    operation.direct = transport::OperationDirect::RemoteDeviceHost;
    operation.target_manager = record.peer_one_sided_id;
    operation.ops.emplace_back(
        transport::Segment{record.local_resp_slot.localAddr, record.remote_resp_addr, len});

    TransportHandle handle = transport::kInvalidTransferHandle;
    const auto submitStartUs = SteadyNowUs();
    const auto submitStatus = runtime_.transport.ExecuteAsync(operation, handle);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kResponseSubmitDurationMs),
                             static_cast<double>(SteadyNowUs() - submitStartUs) / 1000.0);
    if (submitStatus.Failure() || handle == transport::kInvalidTransferHandle) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kResponseFailuresTotal), 1);
        UC_ERROR(
            "CompletionPoller SubmitResponse ExecuteAsync failed, request_id={}, opcode={}, "
            "handle={}, error={}",
            record.request_id, static_cast<int>(record.opcode), handle, submitStatus);
        ReportBatchEndToEnd(record);
        return true;
    }

    record.response_handle = handle;
    record.submit_ms = SteadyNowMs();
    record.timeout_reported = false;
    record.results.clear();
    record.stage = CompletionStage::PollResponseTransfer;
    UC_DEBUG("CompletionPoller submitted response transfer, request_id={}, handle={}, slot={}",
             record.request_id, handle, record.local_resp_slot.slotIndex);
    return false;
}

bool CompletionPoller::PollResponseTransfer(CompletionRecord& record)
{
    transport::TransferStatus transportStatus = transport::TransferStatus::Failed;
    const auto getStatusStartUs = SteadyNowUs();
    const auto queryStatus = runtime_.transport.GetStatus(record.response_handle, transportStatus);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kGetStatusDurationMs),
                             static_cast<double>(SteadyNowUs() - getStatusStartUs) / 1000.0);
    if (queryStatus.Failure()) {
        // GetStatus removes failed handles, so the response source buffer is no longer in use.
        UC_ERROR("CompletionPoller response GetStatus failed, request_id={}, handle={}, error={}",
                 record.request_id, record.response_handle, queryStatus);
        SettleResponseTransfer(runtime_.flagBufferPool, record, true);
        return true;
    }
    if (transportStatus == transport::TransferStatus::Waiting) {
        // Keep the response source buffer alive until transport reports a terminal state.
        if (!record.timeout_reported && OperationTimedOut(record, SteadyNowMs())) {
            record.timeout_reported = true;
            UC_ERROR(
                "CompletionPoller response transfer timed out, request_id={}, peer={}, handle={}, "
                "timeout_ms={}; waiting for Store-initiated disconnect or terminal transport "
                "status",
                record.request_id, record.peer_one_sided_id, record.response_handle,
                g_config.opTimeoutMs);
        }
        return false;
    }
    if (transportStatus != transport::TransferStatus::Completed) {
        UC_ERROR("CompletionPoller response transfer failed, request_id={}, handle={}, status={}",
                 record.request_id, record.response_handle, static_cast<int>(transportStatus));
        SettleResponseTransfer(runtime_.flagBufferPool, record, true);
        return true;
    }

    SettleResponseTransfer(runtime_.flagBufferPool, record, false);
    UC_DEBUG("CompletionPoller response transfer finished, request_id={}, handle={}, status={}",
             record.request_id, record.response_handle, static_cast<int>(transportStatus));

    return true;
}

// Finalize metadata entry state (StoreEnd/LoadEnd/Delete) and fill record.results.
void CompletionPoller::SettleDataTransfer(CompletionRecord& record,
                                          transport::TransferStatus terminalStatus)
{
    ReportDataTransferMetrics(record, terminalStatus);

    for (const auto& item : record.transfer_items) {
        DumpLoadResult result = DumpLoadResult::Failed;

        // Settle metadata and buffer ownership before completing the request item.
        if (record.opcode == OpType::DUMP) {
            if (terminalStatus == transport::TransferStatus::Completed) {
                const auto status = runtime_.metadata.StoreEnd(item.key);
                if (status.Success()) {
                    result = DumpLoadResult::Ok;
                } else {
                    UC_ERROR("CompletionPoller StoreEnd failed, request_id={}, handle={}, error={}",
                             record.request_id, record.data_handle, status);
                    const auto abortStatus = runtime_.metadata.Delete(item.key);
                    if (abortStatus.Failure()) {
                        UC_ERROR(
                            "CompletionPoller Delete failed after StoreEnd error, request_id={}, "
                            "handle={}, error={}",
                            record.request_id, record.data_handle, abortStatus);
                    }
                }
            } else {
                const auto abortStatus = runtime_.metadata.Delete(item.key);
                if (abortStatus.Failure()) {
                    UC_ERROR(
                        "CompletionPoller Delete reserved DUMP failed, request_id={}, handle={}, "
                        "error={}",
                        record.request_id, record.data_handle, abortStatus);
                }
            }
        } else if (record.opcode == OpType::LOAD) {
            const auto releaseStatus = runtime_.metadata.LoadEnd(item.key);
            if (releaseStatus.Failure()) {
                UC_ERROR("CompletionPoller LoadEnd failed, request_id={}, handle={}, error={}",
                         record.request_id, record.data_handle, releaseStatus);
            } else if (terminalStatus == transport::TransferStatus::Completed) {
                result = DumpLoadResult::Ok;
            }
        }

        record.results[item.index_in_request] = static_cast<std::uint8_t>(result);
    }
    record.transfer_items.clear();
}

bool CompletionPoller::OperationTimedOut(const CompletionRecord& record, std::uint64_t nowMs) const
{
    if (nowMs < record.submit_ms) { return false; }
    return nowMs - record.submit_ms >= g_config.opTimeoutMs;
}

}  // namespace UC::DramPool
