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
#include <cstdio>
#include <thread>
#include <utility>
#include "core/transport_manager.h"
#include "drampool_config.h"
#include "logger/logger.h"
#include "metadata.h"

namespace UC::DramPool {
namespace {

double ReadCpuLoadAvg1m()
{
    FILE* fp = std::fopen("/proc/loadavg", "r");
    if (fp == nullptr) { return 0.0; }
    double load = 0.0;
    std::fscanf(fp, "%lf", &load);
    std::fclose(fp);
    return load;
}

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

void LogRequestDone(CompletionRecord& record, const char* status, const char* failedStage)
{
    record.timing.request_completed_us = SteadyNowUs();
    record.timing.request_completed_ts_us = UnixNowUs();
    const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
        return begin != 0 && end >= begin ? end - begin : 0;
    };
    UC_INFO(
        "[PERF] component=drampool event=request_done request_id={} opcode={} peer={} "
        "batch_size={} data_bytes={} failed_items={} status={} failed_stage={} "
        "received_ts_us={} worker_started_ts_us={} data_transfer_submitted_ts_us={} "
        "data_transfer_completed_ts_us={} response_submitted_ts_us={} completed_ts_us={} "
        "data_tm_execute_async_ts_us={} data_hixl_execute_async_ts_us={} "
        "data_tm_get_status_ts_us={} data_hixl_query_handle_ts_us={} "
        "response_tm_execute_async_ts_us={} response_hixl_execute_async_ts_us={} "
        "response_tm_get_status_ts_us={} response_hixl_query_handle_ts_us={} "
        "request_queue_us={} metadata_prepare_us={} taskworker_prepare_us={} poller_queue_us={} "
        "data_transfer_us={} metadata_settle_us={} response_slot_wait_us={} "
        "response_submit_us={} response_transfer_us={} data_tm_to_hixl_execute_async_us={} "
        "data_tm_to_hixl_query_handle_us={} response_tm_to_hixl_execute_async_us={} "
        "response_tm_to_hixl_query_handle_us={} total_us={} cpu_loadavg_1m={:.2f}",
        record.request_id, static_cast<int>(record.opcode), record.peer_one_sided_id,
        record.batch_size, record.data_bytes, record.failed_items, status, failedStage,
        record.timing.received_ts_us, record.timing.worker_started_ts_us,
        record.timing.data_transfer_submitted_ts_us, record.timing.data_transfer_completed_ts_us,
        record.timing.response_submitted_ts_us, record.timing.request_completed_ts_us,
        record.timing.data_execute_async.manager_entered_ts_us,
        record.timing.data_execute_async.backend_called_ts_us,
        record.timing.data_get_status.manager_entered_ts_us,
        record.timing.data_get_status.backend_called_ts_us,
        record.timing.response_execute_async.manager_entered_ts_us,
        record.timing.response_execute_async.backend_called_ts_us,
        record.timing.response_get_status.manager_entered_ts_us,
        record.timing.response_get_status.backend_called_ts_us,
        elapsed(record.timing.received_us, record.timing.worker_started_us),  // request_queue_us
        elapsed(record.timing.metadata_prepare_started_us,
                record.timing.metadata_prepare_completed_us),
        elapsed(record.timing.worker_started_us, record.timing.completion_queued_us),
        elapsed(record.timing.completion_queued_us, record.timing.poller_admitted_us),
        elapsed(record.timing.data_transfer_submitted_us, record.timing.data_transfer_completed_us),
        elapsed(record.timing.data_transfer_completed_us, record.timing.metadata_settle_completed_us),
        elapsed(record.timing.response_ready_us, record.timing.response_slot_acquired_us),
        elapsed(record.timing.response_slot_acquired_us, record.timing.response_submitted_us),
        elapsed(record.timing.response_submitted_us, record.timing.response_completed_us),
        elapsed(record.timing.data_execute_async.manager_entered_us,
                record.timing.data_execute_async.backend_called_us),
        elapsed(record.timing.data_get_status.manager_entered_us,
                record.timing.data_get_status.backend_called_us),
        elapsed(record.timing.response_execute_async.manager_entered_us,
                record.timing.response_execute_async.backend_called_us),
        elapsed(record.timing.response_get_status.manager_entered_us,
                record.timing.response_get_status.backend_called_us),
        elapsed(record.timing.received_us, record.timing.request_completed_us), ReadCpuLoadAvg1m());
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
        record.timing.poller_admitted_us = SteadyNowUs();
        // Stage PERF log disabled: request_done already contains the corresponding timing.
        // UC_INFO(
        //     "[PERF] component=drampool event=stage request_id={} opcode={} "
        //     "stage=POLLER_ADMITTED ts_us={}",
        //     record.request_id, static_cast<int>(record.opcode), UnixNowUs());
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
                LogRequestDone(*iter, "INVALID_STAGE", "COMPLETION_POLLER");
                iter = pending_.erase(iter);
                break;
        }
    }
}

bool CompletionPoller::PollDataTransfer(CompletionRecord& record)
{
    transport::TransferStatus transportStatus = transport::TransferStatus::Failed;
    const auto queryStatus = runtime_.transport.GetStatus(
        record.data_handle, transportStatus, &record.timing.data_get_status);
    if (queryStatus.Failure()) {
        // GetStatus removes failed handles, so an API failure is also terminal.
        UC_ERROR(
            "CompletionPoller data transfer GetStatus failed, request_id={}, handle={}, error={}",
            record.request_id, record.data_handle, queryStatus);
        record.timing.data_transfer_completed_us = SteadyNowUs();
        record.timing.data_transfer_completed_ts_us = UnixNowUs();
        SettleDataTransfer(record, transport::TransferStatus::Failed);
        record.timing.metadata_settle_completed_us = SteadyNowUs();
        record.timing.response_ready_us = record.timing.metadata_settle_completed_us;
        record.data_handle = transport::kInvalidTransferHandle;
        record.stage = CompletionStage::SubmitResponse;
        // Stage PERF log disabled: request_done already contains the corresponding timing.
        // UC_INFO(
        //     "[PERF] component=drampool event=stage request_id={} opcode={} status={} "
        //     "stage=DATA_TRANSFER_COMPLETED ts_us={}",
        //     record.request_id, static_cast<int>(record.opcode), "GET_STATUS_FAILED",
        //     record.timing.data_transfer_completed_ts_us);
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
    record.timing.data_transfer_completed_us = SteadyNowUs();
    record.timing.data_transfer_completed_ts_us = UnixNowUs();
    record.data_transfer_succeeded = transportStatus == transport::TransferStatus::Completed;
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} handle={} status={} "
    //     "stage=DATA_TRANSFER_COMPLETED ts_us={}",
    //     record.request_id, static_cast<int>(record.opcode), record.data_handle,
    //     static_cast<int>(transportStatus), record.timing.data_transfer_completed_ts_us);
    SettleDataTransfer(record, transportStatus);
    record.timing.metadata_settle_completed_us = SteadyNowUs();
    record.timing.response_ready_us = record.timing.metadata_settle_completed_us;
    record.data_handle = transport::kInvalidTransferHandle;
    record.stage = CompletionStage::SubmitResponse;
    UC_DEBUG("CompletionPoller advances to SubmitResponse, request_id={}", record.request_id);
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} "
    //     "stage=SUBMIT_RESPONSE ts_us={}",
    //     record.request_id, static_cast<int>(record.opcode), UnixNowUs());
    return true;
}

bool CompletionPoller::SubmitResponse(CompletionRecord& record)
{
    if (record.opcode == KvOpcode::Dump || record.opcode == KvOpcode::Load) {
        record.failed_items = static_cast<std::uint16_t>(
            std::count_if(record.results.begin(), record.results.end(), [](std::uint8_t result) {
                return result != static_cast<std::uint8_t>(DumpLoadResult::Ok);
            }));
    }
    const auto packedSize =
        runtime_.protocol.GetPackedResponseSize(record.opcode, record.results.size());
    auto allocateStatus = runtime_.flagBufferPool.Allocate(record.local_resp_slot);
    if (allocateStatus.Failure()) {
        if (allocateStatus.Underlying() == Status::NoSpace().Underlying()) {
            UC_WARN(
                "CompletionPoller flag buffer pool full, request_id={}, opcode={}, error={}, "
                "retrying next round",
                record.request_id, static_cast<int>(record.opcode), allocateStatus);
            return false;
        }

        UC_ERROR(
            "CompletionPoller flag buffer allocation failed, request_id={}, opcode={}, error={}",
            record.request_id, static_cast<int>(record.opcode), allocateStatus);
        LogRequestDone(record, "FLAG_BUFFER_ALLOCATION_FAILED", "SUBMIT_RESPONSE");
        return true;
    }
    record.timing.response_slot_acquired_us = SteadyNowUs();
    UC_DEBUG("CompletionPoller allocated response slot, request_id={}, slot={}", record.request_id,
             record.local_resp_slot.slotIndex);

    const auto len = static_cast<std::uint32_t>(packedSize);

    const auto protocolStatus =
        runtime_.protocol.PackResponse(record.local_resp_slot.localAddr, record.opcode,
                                       KvResponse{record.request_id, record.results});
    if (protocolStatus.Failure()) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        UC_ERROR("CompletionPoller SubmitResponse pack failed, request_id={}, opcode={}, error={}",
                 record.request_id, static_cast<int>(record.opcode), protocolStatus);
        LogRequestDone(record, "RESPONSE_PACK_FAILED", "SUBMIT_RESPONSE");
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
    const auto submitStatus = runtime_.transport.ExecuteAsync(
        operation, handle, &record.timing.response_execute_async);
    if (submitStatus.Failure() || handle == transport::kInvalidTransferHandle) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        UC_ERROR(
            "CompletionPoller SubmitResponse ExecuteAsync failed, request_id={}, opcode={}, "
            "handle={}, error={}",
            record.request_id, static_cast<int>(record.opcode), handle, submitStatus);
        LogRequestDone(record, "RESPONSE_SUBMIT_FAILED", "SUBMIT_RESPONSE");
        return true;
    }

    record.response_handle = handle;
    record.submit_ms = SteadyNowMs();
    record.timing.response_submitted_us = SteadyNowUs();
    record.timing.response_submitted_ts_us = UnixNowUs();
    record.timeout_reported = false;
    record.results.clear();
    record.stage = CompletionStage::PollResponseTransfer;
    UC_DEBUG("CompletionPoller submitted response transfer, request_id={}, handle={}, slot={}",
             record.request_id, handle, record.local_resp_slot.slotIndex);
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} handle={} slot={} "
    //     "stage=RESPONSE_TRANSFER_SUBMITTED ts_us={}",
    //     record.request_id, static_cast<int>(record.opcode), handle,
    //     record.local_resp_slot.slotIndex, record.timing.response_submitted_ts_us);
    return false;
}

bool CompletionPoller::PollResponseTransfer(CompletionRecord& record)
{
    transport::TransferStatus transportStatus = transport::TransferStatus::Failed;
    const auto queryStatus = runtime_.transport.GetStatus(
        record.response_handle, transportStatus, &record.timing.response_get_status);
    if (queryStatus.Failure()) {
        // GetStatus removes failed handles, so the response source buffer is no longer in use.
        UC_ERROR("CompletionPoller response GetStatus failed, request_id={}, handle={}, error={}",
                 record.request_id, record.response_handle, queryStatus);
        record.timing.response_completed_us = SteadyNowUs();
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        LogRequestDone(record, "RESPONSE_STATUS_FAILED", "POLL_RESPONSE_TRANSFER");
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
        record.timing.response_completed_us = SteadyNowUs();
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        LogRequestDone(record, "RESPONSE_TRANSFER_FAILED", "POLL_RESPONSE_TRANSFER");
        return true;
    }

    record.timing.response_completed_us = SteadyNowUs();
    ReleaseResponseBuffer(runtime_.flagBufferPool, record);
    UC_DEBUG("CompletionPoller response transfer finished, request_id={}, handle={}, status={}",
             record.request_id, record.response_handle, static_cast<int>(transportStatus));
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} handle={} status={} "
    //     "stage=RESPONSE_TRANSFER_COMPLETED ts_us={}",
    //     record.request_id, static_cast<int>(record.opcode), record.response_handle,
    //     static_cast<int>(transportStatus), UnixNowUs());
    if (record.data_transfer_required && !record.data_transfer_succeeded) {
        LogRequestDone(
            record, "DATA_TRANSFER_FAILED",
            record.data_transfer_submitted ? "POLL_DATA_TRANSFER" : "SUBMIT_DATA_TRANSFER");
    } else if (record.failed_items != 0) {
        LogRequestDone(record, "ITEM_FAILURE", "METADATA_PROCESSING");
    } else {
        LogRequestDone(record, "SUCCESS", "NONE");
    }

    return true;
}

// Finalize metadata entry state (StoreEnd/LoadEnd/Delete) and fill record.results.
void CompletionPoller::SettleDataTransfer(CompletionRecord& record,
                                          transport::TransferStatus terminalStatus)
{
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
