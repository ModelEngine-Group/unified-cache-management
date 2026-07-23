/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "completion_poller.h"
#include <limits>
#include <thread>
#include <utility>
#include "core/transport_manager.h"
#include "drampool_config.h"
#include "logger/logger.h"
#include "metadata.h"

namespace UC::DramPool {
namespace {

void ReleaseResponseBuffer(BufferPool& flagBufferPool, CompletionRecord& record,
                           const char* context)
{
    const auto releasedSlot = record.resp_buffer.slot_index;
    const auto releaseStatus = flagBufferPool.Free(releasedSlot);
    record.resp_buffer = {};
    if (releaseStatus.Failure()) {
        UC_ERROR(
            "CompletionPoller release response flag buffer after failure failed, context={}, "
            "slot={}, error={}",
            context, releasedSlot, releaseStatus);
    }
}

}  // namespace

void CompletionPoller::Run(const std::atomic_bool& stop)
{
    while (true) {
        const bool stopRequested = stop.load(std::memory_order_acquire);
        if (stopRequested) { SetDisconnectAllTransfers(); }

        const auto newPendingCount = FillPendingWindow();
        PollPendingCompletions();

        if (stopRequested) {
            if (shutdownDrainBlocked_) {
                // Keep unfinished records and their buffers alive. DramPoolServer shuts down
                // transport before destroying the poller and memory pools.
                UC_ERROR_UNLIMITED(
                    "CompletionPoller stopped waiting because an in-flight peer could not be "
                    "disconnected");
                break;
            }
            if (newPendingCount == 0 && pending_.empty()) { break; }
        }
        if (newPendingCount == 0 && pending_.empty()) {
            std::this_thread::sleep_for(kThreadIdleSleepDuration);
        }
    }
}

void CompletionPoller::SetDisconnectAllTransfers() noexcept
{
    disconnectAllTransfers_.store(true, std::memory_order_release);
}

std::size_t CompletionPoller::FillPendingWindow()
{
    std::size_t newPendingCount = 0;
    while (pending_.size() < g_config.pollerPendingDepth) {
        CompletionRecord record;
        if (!runtime_.completionQueue.TryPop(record)) { break; }
        pending_.emplace_back(std::move(record));
        ++newPendingCount;
    }
    return newPendingCount;
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
            case CompletionStage::SubmitResponse: {
                const auto status = SubmitResponse(*iter);
                if (status.Failure()) {
                    UC_ERROR("CompletionPoller SubmitResponse failed, opcode={}, error={}",
                             static_cast<int>(iter->opcode), status);
                    iter = pending_.erase(iter);
                } else {
                    ++iter;
                }
                break;
            }
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
        if (!record.disconnect_attempted &&
            (disconnectAllTransfers_.load(std::memory_order_acquire) ||
             OperationTimedOut(record, SteadyNowMs()))) {
            DisconnectPeer(record, record.data_handle, "data");
        }
        return false;
    }

    // A terminal GetStatus releases the data handle before business state is settled.
    SettleDataTransfer(record, transportStatus);
    record.data_handle = transport::kInvalidTransferHandle;
    record.stage = CompletionStage::SubmitResponse;
    return true;
}

Status CompletionPoller::SubmitResponse(CompletionRecord& record)
{
    if (record.peer_one_sided_id.empty()) {
        return Status::InvalidParam("response peer_one_sided_id is empty");
    }

    const auto packedSize =
        runtime_.protocol.GetPackedResponseSize(record.opcode, record.results.size());
    if (packedSize == 0 || packedSize > std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidParam("invalid packed response size");
    }
    const auto len = static_cast<std::uint32_t>(packedSize);
    if (record.resp_buffer.local_addr != nullptr) {
        return Status::InvalidParam("response flag buffer slot is already allocated");
    }
    auto allocateStatus = runtime_.flagBufferPool.Allocate(record.resp_buffer);
    if (allocateStatus.Failure()) { return allocateStatus; }
    if (packedSize > record.resp_buffer.length) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record, "undersized response");
        return Status::InvalidParam(
            "packed response size exceeds configured flag buffer slot size");
    }

    const auto protocolStatus = runtime_.protocol.PackResponse(
        record.resp_buffer.local_addr, record.opcode, KvResponse{record.results});
    if (protocolStatus.Failure()) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record, "response packing");
        return protocolStatus;
    }

    transport::Operation operation;
    operation.opcode = transport::Opcode::Write;
    operation.direct = transport::OperationDirect::RemoteDeviceHost;
    operation.target_manager = record.peer_one_sided_id;
    operation.ops.emplace_back(
        transport::Segment{record.resp_buffer.local_addr, record.response_addr, len});

    TransportHandle handle = transport::kInvalidTransferHandle;
    const auto submitStatus = runtime_.transport.ExecuteAsync(operation, handle);
    if (submitStatus.Failure() || handle == transport::kInvalidTransferHandle) {
        ReleaseResponseBuffer(runtime_.flagBufferPool, record, "response submission");
        if (submitStatus.Failure()) { return submitStatus; }
        return Status::Error("ExecuteAsync response returned an invalid handle");
    }

    // ExecuteAsync now owns access to the bytes; retain the buffer until handle terminal.
    record.response_handle = handle;
    record.submit_ms = SteadyNowMs();
    record.disconnect_attempted = false;
    record.results.clear();
    record.stage = CompletionStage::PollResponseTransfer;
    return Status::OK();
}

bool CompletionPoller::PollResponseTransfer(CompletionRecord& record)
{
    transport::TransferStatus transportStatus = transport::TransferStatus::Failed;
    const auto queryStatus = runtime_.transport.GetStatus(record.response_handle, transportStatus);
    if (queryStatus.Failure()) {
        // GetStatus removes failed handles, so the response source buffer is no longer in use.
        UC_ERROR("CompletionPoller response GetStatus failed, handle={}", record.response_handle);
    } else if (transportStatus == transport::TransferStatus::Waiting) {
        if (!record.disconnect_attempted &&
            (disconnectAllTransfers_.load(std::memory_order_acquire) ||
             OperationTimedOut(record, SteadyNowMs()))) {
            DisconnectPeer(record, record.response_handle, "response");
        }
        return false;
    } else if (transportStatus != transport::TransferStatus::Completed) {
        UC_ERROR("CompletionPoller response transfer failed, handle={}", record.response_handle);
    }

    const auto releasedSlot = record.resp_buffer.slot_index;
    const auto releaseStatus = runtime_.flagBufferPool.Free(releasedSlot);
    record.resp_buffer = {};
    if (releaseStatus.Failure()) {
        UC_ERROR(
            "CompletionPoller release response flag buffer failed, handle={}, slot={}, "
            "error={}",
            record.response_handle, releasedSlot, releaseStatus);
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

        if (item.index_in_request >= record.results.size()) {
            UC_ERROR("CompletionPoller result index out of range, handle={}, index={}",
                     record.data_handle, item.index_in_request);
            continue;
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

void CompletionPoller::DisconnectPeer(CompletionRecord& record, TransportHandle handle,
                                      const char* transferType)
{
    const auto status =
        runtime_.transport.Disconnect(transport::TransportProtocol::Hixl, record.peer_one_sided_id);
    if (status.Failure()) {
        UC_ERROR(
            "CompletionPoller disconnect peer failed, transfer_type={}, peer={}, handle={}, "
            "error={}",
            transferType, record.peer_one_sided_id, handle, status);
        if (disconnectAllTransfers_.load(std::memory_order_acquire)) {
            shutdownDrainBlocked_ = true;
        }
        return;
    }
    record.disconnect_attempted = true;
}

}  // namespace UC::DramPool
