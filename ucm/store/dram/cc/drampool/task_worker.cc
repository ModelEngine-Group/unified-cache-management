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
 * DEALINGS IN THE SOFTWARE.
 * */
#include "task_worker.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>
#include "core/transport_manager.h"
#include "drampool_config.h"
#include "logger/logger.h"
#include "metadata.h"

namespace UC::DramPool {
namespace {

std::chrono::system_clock::time_point LifeTimeout(std::uint64_t ttlMs)
{
    return ttlMs == 0 ? std::chrono::system_clock::time_point{}
                      : std::chrono::system_clock::now() + std::chrono::milliseconds(ttlMs);
}

}  // namespace

TaskWorker::TaskWorker(DramPoolRuntime& runtime) : runtime_(runtime) {}

void TaskWorker::Run(const std::atomic_bool& stop)
{
    while (true) {
        RequestTaskPtr task;
        if (runtime_.requestQueue.TryPop(task)) {
            const auto processStatus = ProcessOneRequest(std::move(task));
            if (processStatus.Failure()) {
                UC_ERROR("TaskWorker ProcessOneRequest failed: {}", processStatus);
            }
            continue;
        }

        // Stop is requested only after RequestReceiveLoop has exited, so an empty queue is drained.
        if (stop.load(std::memory_order_acquire)) { break; }
        std::this_thread::sleep_for(kThreadIdleSleepDuration);
    }
}

Status TaskWorker::ProcessOneRequest(RequestTaskPtr task)
{
    if (!task || !task->request || task->peer_one_sided_id.empty()) {
        return Status::InvalidParam("TaskWorker got an invalid request task");
    }
    // By the time a request reaches DramPool, the store-initiated Connect control request
    // must already have established the local route.
    const auto& peerOneSidedId = task->peer_one_sided_id;
    const auto& request = task->request;
    UC_DEBUG("TaskWorker processing request, request_id={}, opcode={}, peer={}",
             request->request_id, static_cast<int>(request->opcode), peerOneSidedId);
    task->timing.worker_started_us = SteadyNowUs();
    task->timing.worker_started_ts_us = UnixNowUs();
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} peer={} "
    //     "stage=WORKER_STARTED ts_us={}",
    //     request->request_id, static_cast<int>(request->opcode), peerOneSidedId,
    //     task->timing.worker_started_ts_us);
    switch (request->opcode) {
        case OpType::DUMP: {
            const auto* dump = dynamic_cast<const KvDumpRequest*>(request.get());
            return dump == nullptr ? Status::InvalidParam("DUMP request type does not match opcode")
                                   : ProcessDump(*dump, peerOneSidedId, std::move(task->timing));
        }
        case OpType::LOAD: {
            const auto* load = dynamic_cast<const KvLoadRequest*>(request.get());
            return load == nullptr ? Status::InvalidParam("LOAD request type does not match opcode")
                                   : ProcessLoad(*load, peerOneSidedId, std::move(task->timing));
        }
        case OpType::LOOKUP: {
            const auto* lookup = dynamic_cast<const KvLookupRequest*>(request.get());
            return lookup == nullptr
                       ? Status::InvalidParam("LOOKUP request type does not match opcode")
                       : ProcessLookup(*lookup, peerOneSidedId, std::move(task->timing));
        }
    }
    return Status::InvalidParam("TaskWorker got invalid opcode");
}

Status TaskWorker::ProcessDump(const KvDumpRequest& request,
                               const transport::ManagerID& peerOneSidedId, RequestTiming timing)
{
    if (runtime_.protocol.GetPackedResponseSize(OpType::DUMP, request.batch_size) >
        g_config.flagBufferSlotSizeBytes) {
        return Status::InvalidParam("DUMP response exceeds configured flag buffer slot size");
    }
    const std::uint64_t ttl_ms =
        request.ttl != 0 ? static_cast<std::uint64_t>(request.ttl) : g_config.defaultDumpTtlMs;
    const auto lifeTimeout = LifeTimeout(ttl_ms);
    std::vector<std::uint8_t> results(request.batch_size,
                                      static_cast<std::uint8_t>(DumpLoadResult::Ok));
    const auto mark_remaining_failed = [&results](std::size_t first) {
        std::fill(results.begin() + first, results.end(),
                  static_cast<std::uint8_t>(DumpLoadResult::Failed));
    };
    std::vector<TransferItem> transfer_items;
    transfer_items.reserve(request.entries.size());
    std::uint64_t dataBytes = 0;
    transport::Operation operation;
    operation.opcode = transport::Opcode::Read;
    operation.direct = transport::OperationDirect::RemoteDeviceHost;
    operation.target_manager = peerOneSidedId;
    operation.ops.reserve(request.entries.size());

    timing.metadata_prepare_started_us = SteadyNowUs();
    for (std::uint16_t index = 0; index < request.batch_size; ++index) {
        const auto& entry = request.entries[index];

        auto metadataEntry = std::make_shared<UC::DramPool::Entry>();
        metadataEntry->key = entry.key;
        metadataEntry->size = entry.len;
        metadataEntry->lifeTimeout = lifeTimeout;
        metadataEntry->position = entry.idx;

        const auto storeStatus = runtime_.metadata.StoreBegin(entry.key, metadataEntry);
        if (storeStatus == Status::DuplicateKey()) {
            results[index] = static_cast<std::uint8_t>(DumpLoadResult::Ok);
            continue;
        }
        if (storeStatus.Failure()) {
            UC_ERROR("Dump[{}] StoreBegin failed, request_id={}, error={}", index,
                     request.request_id, storeStatus);
            // StoreBegin failures are typically resource-related after eviction retries.
            // Stop here to avoid costly allocation attempts for the remaining items.
            mark_remaining_failed(index);
            break;
        }

        // INITIALIZED entries are not eviction candidates while the DUMP is in flight.
        transfer_items.emplace_back(TransferItem{index, entry.key});
        dataBytes += entry.len;
        operation.ops.emplace_back(
            transport::Segment{metadataEntry->buffer.addr, entry.addr, entry.len});
    }
    timing.metadata_prepare_completed_us = SteadyNowUs();

    if (transfer_items.empty()) {
        UC_DEBUG("DUMP skips data transfer, request_id={}, batch_size={}", request.request_id,
                 request.batch_size);
        return QueueResponse(OpType::DUMP, request.resp_addr, peerOneSidedId, std::move(results),
                             request.request_id);
    }

    UC_DEBUG("DUMP submits data transfer, request_id={}, items={}, peer={}", request.request_id,
             transfer_items.size(), peerOneSidedId);
    TransportHandle handle = transport::kInvalidTransferHandle;
    transport::TransportCallTiming submitTiming;
    const auto submit_status = runtime_.transport.ExecuteAsync(operation, handle, &submitTiming);
    timing.data_execute_async = submitTiming;
    if (submit_status.Failure() || handle == transport::kInvalidTransferHandle) {
        UC_ERROR("Dump SubmitAsync failed, request_id={}, items={}, error={}", request.request_id,
                 transfer_items.size(), submit_status);
        DeleteItemsMetadata(transfer_items);
        for (const auto& item : transfer_items) {
            results[item.index_in_request] = static_cast<std::uint8_t>(DumpLoadResult::Failed);
        }
        return QueueResponse(OpType::DUMP, request.resp_addr, peerOneSidedId, std::move(results),
                             request.request_id);
    }

    CompletionRecord record;
    record.stage = CompletionStage::PollDataTransfer;
    record.request_id = request.request_id;
    record.opcode = OpType::DUMP;
    record.data_handle = handle;
    record.remote_resp_addr = request.resp_addr;
    record.peer_one_sided_id = peerOneSidedId;
    record.results = std::move(results);
    record.transfer_items = std::move(transfer_items);
    record.submit_ms = SteadyNowMs();
    record.data_transfer_required = true;
    record.data_transfer_submitted = true;
    record.timing.data_transfer_submitted_us = SteadyNowUs();
    record.timing.data_transfer_submitted_ts_us = UnixNowUs();
    UC_DEBUG("DUMP data transfer submitted, request_id={}, handle={}", request.request_id, handle);
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} handle={} "
    //     "stage=DATA_TRANSFER_SUBMITTED ts_us={}",
    //     request.request_id, static_cast<int>(request.opcode), handle,
    //     record.timing.data_transfer_submitted_ts_us);
    return SubmitCompletion(std::move(record));
}

Status TaskWorker::ProcessLoad(const KvLoadRequest& request,
                               const transport::ManagerID& peerOneSidedId, RequestTiming timing)
{
    if (runtime_.protocol.GetPackedResponseSize(OpType::LOAD, request.batch_size) >
        g_config.flagBufferSlotSizeBytes) {
        return Status::InvalidParam("LOAD response exceeds configured flag buffer slot size");
    }
    std::vector<std::uint8_t> results(request.batch_size,
                                      static_cast<std::uint8_t>(DumpLoadResult::Ok));
    std::vector<TransferItem> transfer_items;
    transfer_items.reserve(request.entries.size());
    std::uint64_t dataBytes = 0;
    transport::Operation operation;
    operation.opcode = transport::Opcode::Write;
    operation.direct = transport::OperationDirect::RemoteDeviceHost;
    operation.target_manager = peerOneSidedId;
    operation.ops.reserve(request.entries.size());

    timing.metadata_prepare_started_us = SteadyNowUs();
    for (std::uint16_t index = 0; index < request.batch_size; ++index) {
        const auto& entry = request.entries[index];
        UC::DramPool::EntryPtr metadataEntry;
        const auto loadStatus = runtime_.metadata.LoadBegin(entry.key, metadataEntry);
        if (loadStatus.Failure() || !metadataEntry) {
            results[index] = static_cast<std::uint8_t>(DumpLoadResult::Failed);
            UC_ERROR("Load[{}] LoadBegin failed, request_id={}, error={}", index,
                     request.request_id, loadStatus);
            continue;
        }
        if (entry.len > metadataEntry->size) {
            const auto releaseStatus = runtime_.metadata.LoadEnd(entry.key);
            if (releaseStatus.Failure()) {
                UC_ERROR("Load[{}] LoadEnd after len mismatch failed, request_id={}, error={}",
                         index, request.request_id, releaseStatus);
            }
            UC_ERROR("Load[{}] invalid len, request_id={}, requested={}, stored={}", index,
                     request.request_id, entry.len, metadataEntry->size);
            results[index] = static_cast<std::uint8_t>(DumpLoadResult::Failed);
            continue;
        }

        // The LOAD pin keeps metadata and buffer alive through async transport.
        transfer_items.emplace_back(TransferItem{index, entry.key});
        dataBytes += entry.len;
        operation.ops.emplace_back(
            transport::Segment{metadataEntry->buffer.addr, entry.addr, entry.len});
    }
    timing.metadata_prepare_completed_us = SteadyNowUs();

    if (transfer_items.empty()) {
        UC_DEBUG("LOAD skips data transfer, request_id={}, batch_size={}", request.request_id,
                 request.batch_size);
        return QueueResponse(OpType::LOAD, request.resp_addr, peerOneSidedId, std::move(results),
                             request.request_id);
    }

    UC_DEBUG("LOAD submits data transfer, request_id={}, items={}, peer={}", request.request_id,
             transfer_items.size(), peerOneSidedId);
    TransportHandle handle = transport::kInvalidTransferHandle;
    transport::TransportCallTiming submitTiming;
    const auto submit_status = runtime_.transport.ExecuteAsync(operation, handle, &submitTiming);
    timing.data_execute_async = submitTiming;
    if (submit_status.Failure() || handle == transport::kInvalidTransferHandle) {
        UC_ERROR("Load SubmitAsync failed, request_id={}, items={}, error={}", request.request_id,
                 transfer_items.size(), submit_status);
        LoadEndItems(transfer_items);
        for (const auto& item : transfer_items) {
            results[item.index_in_request] = static_cast<std::uint8_t>(DumpLoadResult::Failed);
        }
        return QueueResponse(OpType::LOAD, request.resp_addr, peerOneSidedId, std::move(results),
                             request.request_id);
    }

    CompletionRecord record;
    record.stage = CompletionStage::PollDataTransfer;
    record.request_id = request.request_id;
    record.opcode = OpType::LOAD;
    record.data_handle = handle;
    record.remote_resp_addr = request.resp_addr;
    record.peer_one_sided_id = peerOneSidedId;
    record.results = std::move(results);
    record.transfer_items = std::move(transfer_items);
    record.submit_ms = SteadyNowMs();
    record.data_transfer_required = true;
    record.data_transfer_submitted = true;
    record.timing.data_transfer_submitted_us = SteadyNowUs();
    record.timing.data_transfer_submitted_ts_us = UnixNowUs();
    UC_DEBUG("LOAD data transfer submitted, request_id={}, handle={}", request.request_id, handle);
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} handle={} "
    //     "stage=DATA_TRANSFER_SUBMITTED ts_us={}",
    //     request.request_id, static_cast<int>(request.opcode), handle,
    //     record.timing.data_transfer_submitted_ts_us);
    return SubmitCompletion(std::move(record));
}

Status TaskWorker::ProcessLookup(const KvLookupRequest& request,
                                 const transport::ManagerID& peerOneSidedId, RequestTiming timing)
{
    if (runtime_.protocol.GetPackedResponseSize(OpType::LOOKUP, request.batch_size) >
        g_config.flagBufferSlotSizeBytes) {
        return Status::InvalidParam("LOOKUP response exceeds configured flag buffer slot size");
    }
    std::vector<std::uint8_t> results(request.batch_size,
                                      static_cast<std::uint8_t>(LookupResult::NotFound));
    timing.metadata_prepare_started_us = SteadyNowUs();
    for (std::uint16_t index = 0; index < request.batch_size; ++index) {
        if (runtime_.metadata.Exist(request.entries[index].key)) {
            results[index] = static_cast<std::uint8_t>(LookupResult::Exists);
        }
    }
    timing.metadata_prepare_completed_us = SteadyNowUs();

    UC_DEBUG("LOOKUP metadata scan completed, request_id={}, batch_size={}", request.request_id,
             request.batch_size);
    return QueueResponse(OpType::LOOKUP, request.resp_addr, peerOneSidedId, std::move(results),
                         request.request_id);
}

void TaskWorker::DeleteItemsMetadata(const std::vector<TransferItem>& items)
{
    for (const auto& item : items) {
        // Remove metadata first so no index can retain a freed buffer address.
        const auto abortStatus = runtime_.metadata.Delete(item.key);
        if (abortStatus.Failure()) {
            UC_ERROR("DeleteItemsMetadata Delete reserved DUMP failed: {}", abortStatus);
        }
    }
}

void TaskWorker::LoadEndItems(const std::vector<TransferItem>& items)
{
    for (const auto& item : items) {
        const auto status = runtime_.metadata.LoadEnd(item.key);
        if (status.Failure()) { UC_ERROR("LoadEndItems LoadEnd failed: {}", status); }
    }
}

Status TaskWorker::QueueResponse(OpType opcode, std::uint64_t responseAddr,
                                 const transport::ManagerID& peerOneSidedId,
                                 std::vector<std::uint8_t>&& results, std::uint64_t requestId,
                                 RequestTiming timing, bool dataTransferRequired,
                                 std::uint64_t dataBytes)
{
    CompletionRecord record;
    record.stage = CompletionStage::SubmitResponse;
    record.request_id = requestId;
    record.batch_size = static_cast<std::uint16_t>(results.size());
    record.data_transfer_required = dataTransferRequired;
    record.data_bytes = dataBytes;
    timing.response_ready_us = SteadyNowUs();
    record.timing = std::move(timing);
    record.opcode = opcode;
    record.remote_resp_addr = responseAddr;
    record.peer_one_sided_id = peerOneSidedId;
    record.results = std::move(results);
    return SubmitCompletion(std::move(record));
}

Status TaskWorker::SubmitCompletion(CompletionRecord&& record)
{
    // TaskWorker is the sole producer; CompletionPoller is the sole consumer.
    UC_DEBUG("TaskWorker queues completion, request_id={}, opcode={}, stage={}, handle={}",
             record.request_id, static_cast<int>(record.opcode), static_cast<int>(record.stage),
             record.data_handle);
    record.timing.completion_queued_us = SteadyNowUs();
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=drampool event=stage request_id={} opcode={} handle={} "
    //     "stage=COMPLETION_QUEUED ts_us={}",
    //     record.request_id, static_cast<int>(record.opcode), record.data_handle, UnixNowUs());
    runtime_.completionQueue.Push(std::move(record));
    return Status::OK();
}

}  // namespace UC::DramPool
