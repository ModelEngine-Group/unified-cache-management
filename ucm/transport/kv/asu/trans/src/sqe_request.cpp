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
#include "sqe_request.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "buffer_manager.h"
#include "connection_manager.h"
#include "kv_protocol.h"

namespace UC::ASU {

namespace {

constexpr std::size_t kFlagBufferHeaderSize = 16;

std::uint32_t ToSqeMrKey(MRHandle handle)
{
    // TODO: 每个mrhandle对应的mrkey
    return static_cast<std::uint32_t>(handle);
}

std::uint64_t GetResponseBufferAddr(const ScatterGatherEntry& flagBuffer)
{
    return flagBuffer.addr;
}

std::uint32_t GetResponseMrKey(const ScatterGatherEntry& flagBuffer) { return flagBuffer.lkey; }

std::uint16_t NextSqeCid(const SqeCidAllocator& allocateSqeCid) { return allocateSqeCid(); }

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
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

std::size_t GetFlagBufferSize(std::size_t batchNum)
{
    return kFlagBufferHeaderSize + (batchNum + 1) / 2;
}

Status AllocateSubBatchFlagBuffer(std::size_t batchNum, BufferManager& flagBufferManager,
                                  TransportSubBatchContext& subBatchContext)
{
    auto status =
        flagBufferManager.Allocate(GetFlagBufferSize(batchNum), subBatchContext.flagBuffer);
    if (!status.ok()) {
        subBatchContext.flagBuffer = {};
        return status;
    }
    return Status::OK();
}

Status SetSubBatchBuildFailed(TransportSubBatchContext& subBatchContext, const Status& status)
{
    subBatchContext.state = TransportSubBatchState::FAILED;
    subBatchContext.status = status;
    std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(), status);
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
    request.response_buffer_addr = GetResponseBufferAddr(flagBuffer);
    request.response_mr_key = GetResponseMrKey(flagBuffer);
    request.lr = GetTransportConfigAttr<bool>(attrs, "lr");
    request.rflag = true;
    request.batch_number = static_cast<std::uint16_t>(entries.size);
    request.entries.reserve(entries.size);
    for (std::size_t index = 0; index < entries.size; ++index) {
        KvBatchStoreEntry entry;
        entry.key = entries[index].key;
        entry.offset = 0;
        entry.buffer_addr = entries[index].buffer.region.addr;
        entry.mr_key = ToSqeMrKey(entries[index].buffer.handle);
        entry.length = static_cast<std::uint32_t>(entries[index].buffer.region.size);
        request.entries.emplace_back(std::move(entry));
    }
    return request;
}

KvBatchRetrieveRequest BuildBatchRetrieveRequest(
    const BatchView<KVBuffer>& entries, const std::unordered_map<std::string, std::string>& attrs,
    std::uint16_t cid, const ScatterGatherEntry& flagBuffer)
{
    KvBatchRetrieveRequest request;
    request.cid = cid;
    request.kv_ns_id = GetTransportConfigAttr<std::uint32_t>(attrs, "kv_ns_id");
    request.response_buffer_addr = GetResponseBufferAddr(flagBuffer);
    request.response_mr_key = GetResponseMrKey(flagBuffer);
    request.lr = GetTransportConfigAttr<bool>(attrs, "lr");
    request.rflag = true;
    request.batch_number = static_cast<std::uint16_t>(entries.size);
    request.entries.reserve(entries.size);
    for (std::size_t index = 0; index < entries.size; ++index) {
        KvBatchRetrieveEntry entry;
        entry.key = entries[index].key;
        entry.offset = 0;
        entry.buffer_addr = entries[index].buffer.region.addr;
        entry.mr_key = ToSqeMrKey(entries[index].buffer.handle);
        entry.length = static_cast<std::uint32_t>(entries[index].buffer.region.size);
        request.entries.emplace_back(std::move(entry));
    }
    return request;
}

std::vector<std::string> CopyKeys(const BatchView<CacheKey>& keys)
{
    std::vector<std::string> requestKeys;
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
    request.response_buffer_addr = GetResponseBufferAddr(flagBuffer);
    request.response_mr_key = GetResponseMrKey(flagBuffer);
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
    request.response_buffer_addr = GetResponseBufferAddr(flagBuffer);
    request.response_mr_key = GetResponseMrKey(flagBuffer);
    request.rflag = true;
    request.sc = GetTransportConfigAttr<bool>(attrs, "sc");
    request.keys = CopyKeys(keys);
    request.batch_number = static_cast<std::uint16_t>(request.keys.size());
    return request;
}

KvKeepAliveRequest BuildKeepAliveRequest(std::uint16_t cid, const ScatterGatherEntry& flagBuffer)
{
    KvKeepAliveRequest request;
    request.cid = cid;
    request.response_buffer_addr = GetResponseBufferAddr(flagBuffer);
    request.response_mr_key = GetResponseMrKey(flagBuffer);
    request.rflag = true;
    return request;
}

}  // namespace

Status ValidateSqeRequestAttrs(const std::unordered_map<std::string, std::string>& attrs)
{
    const auto validateInteger = [&attrs](const std::string& name, auto maxValue) -> Status {
        auto iter = attrs.find(name);
        if (iter == attrs.end()) { return Status::OK(); }
        try {
            const auto parsed = std::stoull(iter->second, nullptr, 0);
            if (parsed > maxValue) {
                return Status::Error(StatusCode::INVALID_ARGUMENT, name + " exceeds valid range");
            }
        } catch (const std::exception&) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, name + " is not a valid integer");
        }
        return Status::OK();
    };

    const auto validateRequiredPositiveInteger = [&attrs](const std::string& name,
                                                          auto maxValue) -> Status {
        auto iter = attrs.find(name);
        if (iter == attrs.end()) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, name + " is required");
        }
        try {
            const auto parsed = std::stoull(iter->second, nullptr, 0);
            if (parsed > maxValue) {
                return Status::Error(StatusCode::INVALID_ARGUMENT, name + " exceeds valid range");
            }
            if (parsed == 0) {
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     name + " must be greater than zero");
            }
        } catch (const std::exception&) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, name + " is not a valid integer");
        }
        return Status::OK();
    };

    const auto validateBool = [&attrs](const std::string& name) -> Status {
        auto iter = attrs.find(name);
        if (iter == attrs.end()) { return Status::OK(); }
        const auto value = ToLower(iter->second);
        if (value == "1" || value == "0" || value == "true" || value == "false") {
            return Status::OK();
        }
        return Status::Error(StatusCode::INVALID_ARGUMENT, name + " is not a valid bool");
    };

    auto status = validateInteger("kv_ns_id", std::numeric_limits<std::uint32_t>::max());
    if (!status.ok()) { return status; }
    status = validateInteger("dtype", std::numeric_limits<std::uint8_t>::max());
    if (!status.ok()) { return status; }
    status = validateInteger("dspec", std::numeric_limits<std::uint8_t>::max());
    if (!status.ok()) { return status; }
    status = validateBool("sc");
    if (!status.ok()) { return status; }
    status = validateBool("lr");
    if (!status.ok()) { return status; }
    status =
        validateRequiredPositiveInteger("kernel_count", std::numeric_limits<std::uint32_t>::max());
    if (!status.ok()) { return status; }
    return validateRequiredPositiveInteger("quiet_count",
                                           std::numeric_limits<std::uint32_t>::max());
}

Status SubmitEntrySubBatchRequest(TransportOpType opType,
                                  const IoScheduler::ScheduledIoBatch& subBatch,
                                  const std::unordered_map<std::string, std::string>& attrs,
                                  const SqeCidAllocator& allocateSqeCid,
                                  BufferManager& sendBufferManager,
                                  BufferManager& flagBufferManager,
                                  ProtocolManager& protocolManager,
                                  TransportSubBatchContext& subBatchContext)
{
    subBatchContext.entryStatus.assign(subBatch.entries.size, Status::OK());
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.status = Status::OK();

    if (subBatch.entries.empty()) { return Status::OK(); }

    if (opType == TransportOpType::BATCH_LOAD) {
        subBatchContext.opType = opType;
        subBatchContext.cid = NextSqeCid(allocateSqeCid);
        auto status =
            AllocateSubBatchFlagBuffer(subBatch.entries.size, flagBufferManager, subBatchContext);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        auto request = BuildBatchRetrieveRequest(subBatch.entries, attrs, subBatchContext.cid,
                                                 subBatchContext.flagBuffer);
        auto packedSize = protocolManager.GetPackedSize(KvOpcode::BatchRetrieve, request);
        status = sendBufferManager.Allocate(packedSize, subBatchContext.sendSge);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        status = protocolManager.PackRequest(reinterpret_cast<void*>(subBatchContext.sendSge.addr),
                                             KvOpcode::BatchRetrieve, request);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        subBatchContext.status = status;
        return status;
    }

    if (opType == TransportOpType::BATCH_STORE) {
        subBatchContext.opType = opType;
        subBatchContext.cid = NextSqeCid(allocateSqeCid);
        auto status =
            AllocateSubBatchFlagBuffer(subBatch.entries.size, flagBufferManager, subBatchContext);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        auto request = BuildBatchStoreRequest(subBatch.entries, attrs, subBatchContext.cid,
                                              subBatchContext.flagBuffer);
        auto packedSize = protocolManager.GetPackedSize(KvOpcode::BatchStore, request);
        status = sendBufferManager.Allocate(packedSize, subBatchContext.sendSge);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        status = protocolManager.PackRequest(reinterpret_cast<void*>(subBatchContext.sendSge.addr),
                                             KvOpcode::BatchStore, request);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        subBatchContext.status = status;
        return status;
    }

    auto status = Status::Error(StatusCode::UNSUPPORTED,
                                "entry batch submit only supports batch store/retrieve operations");
    return SetSubBatchBuildFailed(subBatchContext, status);
}

Status SubmitKeySubBatchRequest(TransportOpType opType,
                                const IoScheduler::ScheduledKeyBatch& subBatch,
                                const std::unordered_map<std::string, std::string>& attrs,
                                const SqeCidAllocator& allocateSqeCid,
                                BufferManager& sendBufferManager, BufferManager& flagBufferManager,
                                ProtocolManager& protocolManager,
                                TransportSubBatchContext& subBatchContext)
{
    subBatchContext.entryStatus.assign(subBatch.keys.size, Status::OK());
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.status = Status::OK();

    if (subBatch.keys.empty()) { return Status::OK(); }

    if (opType != TransportOpType::QUERY && opType != TransportOpType::DELETE) {
        auto status =
            Status::Error(StatusCode::UNSUPPORTED, "key batch submit only supports query/delete");
        return SetSubBatchBuildFailed(subBatchContext, status);
    }

    subBatchContext.cid = NextSqeCid(allocateSqeCid);
    subBatchContext.opType = opType;
    auto status =
        AllocateSubBatchFlagBuffer(subBatch.keys.size, flagBufferManager, subBatchContext);
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
    if (opType == TransportOpType::DELETE) {
        auto request = BuildDeleteRequest(subBatch.keys, attrs, subBatchContext.cid,
                                          subBatchContext.flagBuffer);
        auto packedSize = protocolManager.GetPackedSize(KvOpcode::Delete, request);
        status = sendBufferManager.Allocate(packedSize, subBatchContext.sendSge);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        status = protocolManager.PackRequest(reinterpret_cast<void*>(subBatchContext.sendSge.addr),
                                             KvOpcode::Delete, request);
    } else if (opType == TransportOpType::QUERY) {
        auto request = BuildExistRequest(subBatch.keys, attrs, subBatchContext.cid,
                                         subBatchContext.flagBuffer);
        subBatchContext.useSeekControl = request.sc;
        auto packedSize = protocolManager.GetPackedSize(KvOpcode::Exist, request);
        status = sendBufferManager.Allocate(packedSize, subBatchContext.sendSge);
        if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
        status = protocolManager.PackRequest(reinterpret_cast<void*>(subBatchContext.sendSge.addr),
                                             KvOpcode::Exist, request);
    }
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }

    subBatchContext.status = Status::OK();
    return Status::OK();
}

Status SubmitKeepAliveRequest(const SqeCidAllocator& allocateSqeCid,
                              BufferManager& sendBufferManager, BufferManager& flagBufferManager,
                              ProtocolManager& protocolManager,
                              TransportSubBatchContext& subBatchContext)
{
    subBatchContext.cid = NextSqeCid(allocateSqeCid);
    subBatchContext.opType = TransportOpType::KEEP_ALIVE;
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.status = Status::OK();
    subBatchContext.entryStatus = {Status::OK()};
    auto status = AllocateSubBatchFlagBuffer(1, flagBufferManager, subBatchContext);
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
    auto request = BuildKeepAliveRequest(subBatchContext.cid, subBatchContext.flagBuffer);
    auto packedSize = protocolManager.GetPackedSize(KvOpcode::KeepAlive, request);
    status = sendBufferManager.Allocate(packedSize, subBatchContext.sendSge);
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
    status = protocolManager.PackRequest(reinterpret_cast<void*>(subBatchContext.sendSge.addr),
                                         KvOpcode::KeepAlive, request);
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
    subBatchContext.status = status;
    return status;
}

}  // namespace UC::ASU
