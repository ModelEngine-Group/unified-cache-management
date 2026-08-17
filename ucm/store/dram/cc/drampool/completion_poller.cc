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
#include <thread>
#include <utility>
#include "core/transport_manager.h"
#include "drampool_config.h"
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
        UC_ERROR("CompletionPoller release response flag buffer failed, slot={}, error={}",
                 releasedSlot, releaseStatus);
    }
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
                UC_ERROR("CompletionPoller got invalid completion stage={}",
                         static_cast<int>(iter->stage));
                iter = pending_.erase(iter);
                break;
        }
    }
}

bool CompletionPoller::PollDataTransfer(CompletionRecord& record)
{
    transport::TransferStatus transportStatus = transport::TransferStatus::Failed;
    const auto queryStatus = runtime_.transport.GetStatus(record.data_handle, transportStatus);
    if (queryStatus.Failure()) {
        // GetStatus removes failed handles, so an API failure is also terminal.
        UC_ERROR("CompletionPoller data GetStatus failed, handle={}", record.data_handle);
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
                "CompletionPoller data transfer timed out, peer={}, handle={}, timeout_ms={}; "
                "waiting for Store-initiated disconnect or terminal transport status",
                record.peer_one_sided_id, record.data_handle, g_config.opTimeoutMs);
        }
        return false;
    }

    // A terminal GetStatus releases the data handle before business state is settled.
    SettleDataTransfer(record, transportStatus);
    record.data_handle = transport::kInvalidTransferHandle;
    record.stage = CompletionStage::SubmitResponse;
    return true;
}

bool CompletionPoller::SubmitResponse(CompletionRecord& record)
{
    const auto packedSize =
        runtime_.protocol.GetPackedResponseSize(record.opcode, record.results.size());
    auto allocateStatus = runtime_.flagBufferPool.Allocate(record.local_resp_slot);
    if (allocateStatus.Failure()) {
        if (allocateStatus.Underlying() == Status::NoSpace().Underlying()) {
            UC_WARN(
                "CompletionPoller flag buffer pool full, opcode={}, error={}, retrying next round",
                static_cast<int>(record.opcode), allocateStatus);
            return false;
        }

        UC_ERROR("CompletionPoller flag buffer allocation failed, opcode={}, error={}",
                 static_cast<int>(record.opcode), allocateStatus);
        return true;
    }

    const auto len = static_cast<std::uint32_t>(packedSize);

    const auto protocolStatus = runtime_.protocol.PackResponse(
        record.local_resp_slot.localAddr, record.opcode, KvResponse{record.results});
    if (protocolStatus.Failure()) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        UC_ERROR("CompletionPoller SubmitResponse pack failed, opcode={}, error={}",
                 static_cast<int>(record.opcode), protocolStatus);
        return true;
    }

    transport::Operation operation;
    operation.opcode = transport::Opcode::Write;
    operation.direct = transport::OperationDirect::RemoteDeviceHost;
    operation.target_manager = record.peer_one_sided_id;
    operation.ops.emplace_back(
        transport::Segment{record.local_resp_slot.localAddr, record.remote_resp_addr, len});

    TransportHandle handle = transport::kInvalidTransferHandle;
    const auto submitStatus = runtime_.transport.ExecuteAsync(operation, handle);
    if (submitStatus.Failure() || handle == transport::kInvalidTransferHandle) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record);
        UC_ERROR("CompletionPoller SubmitResponse ExecuteAsync failed, opcode={}, error={}",
                 static_cast<int>(record.opcode), submitStatus);
        return true;
    }

    record.response_handle = handle;
    record.submit_ms = SteadyNowMs();
    record.timeout_reported = false;
    record.results.clear();
    record.stage = CompletionStage::PollResponseTransfer;
    return false;
}

bool CompletionPoller::PollResponseTransfer(CompletionRecord& record)
{
    transport::TransferStatus transportStatus = transport::TransferStatus::Failed;
    const auto queryStatus = runtime_.transport.GetStatus(record.response_handle, transportStatus);
    if (queryStatus.Failure()) {
        // GetStatus removes failed handles, so the response source buffer is no longer in use.
        UC_ERROR("CompletionPoller response GetStatus failed, handle={}", record.response_handle);
    } else if (transportStatus == transport::TransferStatus::Waiting) {
        // Keep the response source buffer alive until transport reports a terminal state.
        if (!record.timeout_reported && OperationTimedOut(record, SteadyNowMs())) {
            record.timeout_reported = true;
            UC_ERROR(
                "CompletionPoller response transfer timed out, peer={}, handle={}, timeout_ms={}; "
                "waiting for Store-initiated disconnect or terminal transport status",
                record.peer_one_sided_id, record.response_handle, g_config.opTimeoutMs);
        }
        return false;
    } else if (transportStatus != transport::TransferStatus::Completed) {
        UC_ERROR("CompletionPoller response transfer failed, handle={}", record.response_handle);
    }

    ReleaseResponseBuffer(runtime_.flagBufferPool, record);
    return true;
}

// Finalize metadata entry state (StoreEnd/LoadEnd/Delete) and fill record.results.
void CompletionPoller::SettleDataTransfer(CompletionRecord& record,
                                          transport::TransferStatus terminalStatus)
{
    for (const auto& item : record.transfer_items) {
        DumpLoadResult result = DumpLoadResult::Failed;

        // Settle metadata and buffer ownership before completing the request item.
        if (record.opcode == KvOpcode::Dump) {
            if (terminalStatus == transport::TransferStatus::Completed) {
                const auto status = runtime_.metadata.StoreEnd(item.key);
                if (status.Success()) {
                    result = DumpLoadResult::Ok;
                } else {
                    UC_ERROR("CompletionPoller StoreEnd failed, handle={}, error={}",
                             record.data_handle, status);
                    const auto abortStatus = runtime_.metadata.Delete(item.key);
                    if (abortStatus.Failure()) {
                        UC_ERROR(
                            "CompletionPoller Delete failed after StoreEnd error, handle={}, "
                            "error={}",
                            record.data_handle, abortStatus);
                    }
                }
            } else {
                const auto abortStatus = runtime_.metadata.Delete(item.key);
                if (abortStatus.Failure()) {
                    UC_ERROR("CompletionPoller Delete reserved DUMP failed, handle={}, error={}",
                             record.data_handle, abortStatus);
                }
            }
        } else if (record.opcode == KvOpcode::Load) {
            const auto releaseStatus = runtime_.metadata.LoadEnd(item.key);
            if (releaseStatus.Failure()) {
                UC_ERROR("CompletionPoller LoadEnd failed, handle={}, error={}", record.data_handle,
                         releaseStatus);
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
