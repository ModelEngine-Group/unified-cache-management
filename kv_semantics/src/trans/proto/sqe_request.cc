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
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "buffer/buffer_manager.h"
#include "conn/connection_manager.h"
#include "kv_protocol.h"
#include "logger.h"
#include "parser_common.h"
#include "task/trans_task_executor.h"

namespace kv {

namespace {

constexpr std::size_t kFlagBufferHeaderSize = 16;

Status ValidateSqeMrKeys(const BatchView<KVBuffer>& entries)
{
    for (std::size_t index = 0; index < entries.size; ++index) {
        if (!entries[index].mrKey.has_value()) {
            return Status::Error(
                StatusCode::BUFFER_NOT_REGISTERED,
                "entry buffer is not registered, entryIndex=" + std::to_string(index));
        }
    }
    return Status::OK();
}

template <typename T>
T GetTransportConfigAttr(const std::unordered_map<std::string, std::string>& attrs,
                         const std::string& name, T fallback = {})
{
    auto iter = attrs.find(name);
    if (iter == attrs.end()) { return fallback; }

    if constexpr (std::is_same_v<T, bool>) {
        const auto value = ToLower(iter->second);
        return value == "1" || value == "true";
    } else {
        return static_cast<T>(std::stoull(iter->second, nullptr, 0));
    }
}

std::size_t GetFlagBufferSize(KvOpcode opcode, std::size_t batchNum)
{
    switch (opcode) {
        case KvOpcode::BatchStore:
        case KvOpcode::BatchRetrieve: return kFlagBufferHeaderSize + (batchNum + 1) / 2;
        case KvOpcode::Delete:
        case KvOpcode::Exist: return kFlagBufferHeaderSize + (batchNum + 7) / 8;
        default: return kFlagBufferHeaderSize + 1;
    }
}

Status AllocateSubBatchFlagBuffer(KvOpcode opcode, std::size_t batchNum,
                                  BufferManager& flagBufferManager,
                                  TransportSubBatchContext& subBatchContext)
{
    const auto flagBufferSize = GetFlagBufferSize(opcode, batchNum);
    auto status = flagBufferManager.Allocate(flagBufferSize, subBatchContext.flagBuffer);
    if (!status.ok()) {
        KV_ERROR(
            "Allocate sub-batch flag buffer failed opcode={} batch_num={} size={} code={} "
            "message={}",
            static_cast<int>(opcode), batchNum, flagBufferSize, static_cast<int>(status.code),
            status.message);
        subBatchContext.flagBuffer = {};
        return status;
    }
    return Status::OK();
}

void SetSubBatchBuildFailed(TransportSubBatchContext& subBatchContext, const Status& status)
{
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = status;
    std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(), status);
}

struct SubBatchRequestSource {
    const BatchView<KVBuffer>* entries{nullptr};
    const BatchView<CacheKey>* keys{nullptr};

    static SubBatchRequestSource FromEntries(const BatchView<KVBuffer>& value)
    {
        return SubBatchRequestSource{&value, nullptr};
    }

    static SubBatchRequestSource FromKeys(const BatchView<CacheKey>& value)
    {
        return SubBatchRequestSource{nullptr, &value};
    }

    static SubBatchRequestSource KeepAlive() { return SubBatchRequestSource{}; }
};

Status PrepareSubBatchRequest(AsuOpType opType, KvOpcode opcode, std::uint16_t cid,
                              std::size_t batchNum, BufferManager& flagBufferManager,
                              TransportSubBatchContext& subBatchContext)
{
    subBatchContext.opType = opType;
    subBatchContext.cid = cid;
    auto status = AllocateSubBatchFlagBuffer(opcode, batchNum, flagBufferManager, subBatchContext);
    if (!status.ok()) {
        SetSubBatchBuildFailed(subBatchContext, status);
        return status;
    }
    return Status::OK();
}

Status PackSubBatchRequest(ProtocolManager& protocolManager, BufferManager& sendBufferManager,
                           KvOpcode opcode, const SqeRequest& request,
                           TransportSubBatchContext& subBatchContext)
{
    auto packedSize = protocolManager.GetPackedSize(opcode, request);
    auto status = sendBufferManager.Allocate(packedSize, subBatchContext.sendSge);
    if (!status.ok()) {
        KV_ERROR(
            "Allocate sub-batch send buffer failed opcode={} cid={} packed_size={} code={} "
            "message={}",
            static_cast<int>(opcode), subBatchContext.cid, packedSize,
            static_cast<int>(status.code), status.message);
        SetSubBatchBuildFailed(subBatchContext, status);
        return status;
    }

    status = protocolManager.PackRequest(
        reinterpret_cast<void*>(subBatchContext.sendSge.local_addr), opcode, request);
    if (!status.ok()) {
        KV_ERROR("Pack sub-batch request failed opcode={} cid={} code={} message={}",
                 static_cast<int>(opcode), subBatchContext.cid, static_cast<int>(status.code),
                 status.message);
        SetSubBatchBuildFailed(subBatchContext, status);
        return status;
    }

    subBatchContext.status = status;
    return status;
}

KvBatchStoreRequest BuildBatchStoreRequest(
    const BatchView<KVBuffer>& entries, const std::unordered_map<std::string, std::string>& attrs,
    std::uint16_t cid, const ScatterGatherEntry& flagBuffer)
{
    KvBatchStoreRequest request;
    request.cid = cid;
    request.kv_ns_id = GetTransportConfigAttr<std::uint32_t>(attrs, "kv_ns_id");
    request.dtype = GetTransportConfigAttr<std::uint8_t>(attrs, "dtype");
    request.dspec = GetTransportConfigAttr<std::uint8_t>(attrs, "dspec");
    request.response_buffer_addr = flagBuffer.device_addr;
    request.response_mr_key = flagBuffer.tokenId;
    request.lr = GetTransportConfigAttr<bool>(attrs, "lr");
    request.rflag = true;
    request.batch_number = static_cast<std::uint16_t>(entries.size);
    request.entries.reserve(entries.size);
    for (std::size_t index = 0; index < entries.size; ++index) {
        KvBatchStoreEntry entry;
        entry.key = entries[index].key;
        entry.offset = entries[index].offset;
        entry.buffer_addr = entries[index].buffer.region.addr;
        entry.mr_key = *entries[index].mrKey;
        entry.length = static_cast<std::uint32_t>(entries[index].buffer.region.size);
        request.entries.emplace_back(std::move(entry));
    }
    return request;
}

KvStoreRequest BuildStoreRequest(const KVBuffer& entry,
                                 const std::unordered_map<std::string, std::string>& attrs,
                                 std::uint16_t cid)
{
    KvStoreRequest request;
    request.cid = cid;
    request.kv_ns_id = GetTransportConfigAttr<std::uint32_t>(attrs, "kv_ns_id");
    request.dtype = GetTransportConfigAttr<std::uint8_t>(attrs, "dtype");
    request.dspec = GetTransportConfigAttr<std::uint8_t>(attrs, "dspec");
    request.buffer_addr = entry.buffer.region.addr;
    request.buffer_length = static_cast<std::uint32_t>(entry.buffer.region.size);
    request.mr_key = *entry.mrKey;
    request.offset = entry.offset;
    request.lr = GetTransportConfigAttr<bool>(attrs, "lr");
    request.length = request.buffer_length;
    request.key = entry.key;
    return request;
}

KvRetrieveRequest BuildRetrieveRequest(const KVBuffer& entry,
                                       const std::unordered_map<std::string, std::string>& attrs,
                                       std::uint16_t cid)
{
    KvRetrieveRequest request;
    request.cid = cid;
    request.kv_ns_id = GetTransportConfigAttr<std::uint32_t>(attrs, "kv_ns_id");
    request.buffer_addr = entry.buffer.region.addr;
    request.buffer_length = static_cast<std::uint32_t>(entry.buffer.region.size);
    request.mr_key = *entry.mrKey;
    request.offset = entry.offset;
    request.lr = GetTransportConfigAttr<bool>(attrs, "lr");
    request.length = request.buffer_length;
    request.key = entry.key;
    return request;
}

KvBatchRetrieveRequest BuildBatchRetrieveRequest(
    const BatchView<KVBuffer>& entries, const std::unordered_map<std::string, std::string>& attrs,
    std::uint16_t cid, const ScatterGatherEntry& flagBuffer)
{
    KvBatchRetrieveRequest request;
    request.cid = cid;
    request.kv_ns_id = GetTransportConfigAttr<std::uint32_t>(attrs, "kv_ns_id");
    request.response_buffer_addr = flagBuffer.device_addr;
    request.response_mr_key = flagBuffer.tokenId;
    request.lr = GetTransportConfigAttr<bool>(attrs, "lr");
    request.rflag = true;
    request.batch_number = static_cast<std::uint16_t>(entries.size);
    request.entries.reserve(entries.size);
    for (std::size_t index = 0; index < entries.size; ++index) {
        KvBatchRetrieveEntry entry;
        entry.key = entries[index].key;
        entry.offset = entries[index].offset;
        entry.buffer_addr = entries[index].buffer.region.addr;
        entry.mr_key = *entries[index].mrKey;
        entry.length = static_cast<std::uint32_t>(entries[index].buffer.region.size);
        request.entries.emplace_back(std::move(entry));
    }
    return request;
}

std::vector<CacheKey> CopyKeys(const BatchView<CacheKey>& keys)
{
    std::vector<CacheKey> requestKeys;
    requestKeys.reserve(keys.size);
    for (std::size_t index = 0; index < keys.size; ++index) {
        requestKeys.emplace_back(keys[index]);
    }
    return requestKeys;
}

KvDeleteRequest BuildDeleteRequest(const BatchView<CacheKey>& keys,
                                   const std::unordered_map<std::string, std::string>& attrs,
                                   std::uint16_t cid, const ScatterGatherEntry& flagBuffer)
{
    KvDeleteRequest request;
    request.cid = cid;
    request.kv_ns_id = GetTransportConfigAttr<std::uint32_t>(attrs, "kv_ns_id");
    request.response_buffer_addr = flagBuffer.device_addr;
    request.response_mr_key = flagBuffer.tokenId;
    request.rflag = true;
    request.keys = CopyKeys(keys);
    request.batch_number = static_cast<std::uint16_t>(request.keys.size());
    return request;
}

KvExistRequest BuildExistRequest(const BatchView<CacheKey>& keys,
                                 const std::unordered_map<std::string, std::string>& attrs,
                                 std::uint16_t cid, const ScatterGatherEntry& flagBuffer)
{
    KvExistRequest request;
    request.cid = cid;
    request.kv_ns_id = GetTransportConfigAttr<std::uint32_t>(attrs, "kv_ns_id");
    request.response_buffer_addr = flagBuffer.device_addr;
    request.response_mr_key = flagBuffer.tokenId;
    request.rflag = true;
    request.sc = GetTransportConfigAttr<bool>(attrs, "sc", true);
    request.keys = CopyKeys(keys);
    request.batch_number = static_cast<std::uint16_t>(request.keys.size());
    return request;
}

KvKeepAliveRequest BuildKeepAliveRequest(std::uint16_t cid, const ScatterGatherEntry& flagBuffer)
{
    KvKeepAliveRequest request;
    request.cid = cid;
    request.response_buffer_addr = flagBuffer.device_addr;
    request.response_mr_key = flagBuffer.tokenId;
    request.rflag = true;
    return request;
}

std::unique_ptr<SqeRequest> BuildSqeRequest(
    KvOpcode opcode, const SubBatchRequestSource& source,
    const std::unordered_map<std::string, std::string>& attrs, std::uint16_t cid,
    const ScatterGatherEntry& flagBuffer, TransportSubBatchContext& subBatchContext)
{
    switch (opcode) {
        case KvOpcode::Store:
            if (source.entries == nullptr || source.entries->size != 1) { return nullptr; }
            return std::make_unique<KvStoreRequest>(
                BuildStoreRequest(source.entries->data[0], attrs, cid));
        case KvOpcode::Retrieve:
            if (source.entries == nullptr || source.entries->size != 1) { return nullptr; }
            return std::make_unique<KvRetrieveRequest>(
                BuildRetrieveRequest(source.entries->data[0], attrs, cid));
        case KvOpcode::BatchRetrieve:
            if (source.entries == nullptr) { return nullptr; }
            return std::make_unique<KvBatchRetrieveRequest>(
                BuildBatchRetrieveRequest(*source.entries, attrs, cid, flagBuffer));
        case KvOpcode::BatchStore:
            if (source.entries == nullptr) { return nullptr; }
            return std::make_unique<KvBatchStoreRequest>(
                BuildBatchStoreRequest(*source.entries, attrs, cid, flagBuffer));
        case KvOpcode::Delete:
            if (source.keys == nullptr) { return nullptr; }
            return std::make_unique<KvDeleteRequest>(
                BuildDeleteRequest(*source.keys, attrs, cid, flagBuffer));
        case KvOpcode::Exist: {
            if (source.keys == nullptr) { return nullptr; }
            auto request = BuildExistRequest(*source.keys, attrs, cid, flagBuffer);
            subBatchContext.useSeekControl = request.sc;
            return std::make_unique<KvExistRequest>(std::move(request));
        }
        case KvOpcode::KeepAlive:
            return std::make_unique<KvKeepAliveRequest>(BuildKeepAliveRequest(cid, flagBuffer));
        default:
            KV_ERROR("Build SQE request failed: unsupported opcode={} cid={}",
                     static_cast<int>(opcode), cid);
            return nullptr;
    }
}

}  // namespace

Status TransportTaskExecutor::BuildEntrySubBatchRequest(
    AsuOpType opType, const IoScheduler::ScheduledIoBatch& subBatch,
    TransportSubBatchContext& subBatchContext)
{
    const auto source = SubBatchRequestSource::FromEntries(subBatch.entries);
    subBatchContext.entryStatus.assign(subBatch.entries.size, Status::OK());
    const auto opcode = ToKvOpcode(opType);

    auto validationStatus = ValidateSqeMrKeys(subBatch.entries);
    if (!validationStatus.ok()) {
        SetSubBatchBuildFailed(subBatchContext, validationStatus);
        return validationStatus;
    }

    const auto cid = AllocateRequestCid();
    auto status = PrepareSubBatchRequest(opType, opcode, cid, subBatch.entries.size,
                                         flagBufferManager_, subBatchContext);
    if (!status.ok()) { return status; }

    auto request = BuildSqeRequest(opcode, source, config_.attrs, subBatchContext.cid,
                                   subBatchContext.flagBuffer, subBatchContext);
    return PackSubBatchRequest(protocolManager_, sendBufferManager_, opcode, *request,
                               subBatchContext);
}

Status TransportTaskExecutor::SubmitKeySubBatchRequest(
    AsuOpType opType, const IoScheduler::ScheduledKeyBatch& subBatch,
    TransportSubBatchContext& subBatchContext)
{
    const auto source = SubBatchRequestSource::FromKeys(subBatch.keys);
    subBatchContext.entryStatus.assign(subBatch.keys.size, Status::OK());
    const auto opcode = ToKvOpcode(opType);

    auto status = PrepareSubBatchRequest(opType, opcode, AllocateRequestCid(), subBatch.keys.size,
                                         flagBufferManager_, subBatchContext);
    if (!status.ok()) { return status; }

    auto request = BuildSqeRequest(opcode, source, config_.attrs, subBatchContext.cid,
                                   subBatchContext.flagBuffer, subBatchContext);
    return PackSubBatchRequest(protocolManager_, sendBufferManager_, opcode, *request,
                               subBatchContext);
}

Status TransportTaskExecutor::SubmitKeepAliveRequest(TransportSubBatchContext& subBatchContext)
{
    const auto opType = AsuOpType::KEEP_ALIVE;
    const auto source = SubBatchRequestSource::KeepAlive();
    subBatchContext.entryStatus.assign(1, Status::OK());
    const auto opcode = ToKvOpcode(opType);

    auto status = PrepareSubBatchRequest(opType, opcode, AllocateRequestCid(), 1,
                                         flagBufferManager_, subBatchContext);
    if (!status.ok()) { return status; }

    auto request = BuildSqeRequest(opcode, source, config_.attrs, subBatchContext.cid,
                                   subBatchContext.flagBuffer, subBatchContext);
    return PackSubBatchRequest(protocolManager_, sendBufferManager_, opcode, *request,
                               subBatchContext);
}

}  // namespace kv
