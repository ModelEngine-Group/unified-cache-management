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
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "asu_transport_impl.h"
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
    subBatchContext.state = TransportSubBatchState::COMPLETED;
    subBatchContext.status = status;
    std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(), status);
    return status;
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

void ResetSubBatchContext(std::size_t batchNum, TransportSubBatchContext& subBatchContext)
{
    subBatchContext.entryStatus.assign(batchNum, Status::OK());
    subBatchContext.state = TransportSubBatchState::PENDING;
    subBatchContext.status = Status::OK();
}

Status PrepareSubBatchRequest(TransportOpType opType, std::uint16_t cid, std::size_t batchNum,
                              BufferManager& flagBufferManager,
                              TransportSubBatchContext& subBatchContext)
{
    subBatchContext.opType = opType;
    subBatchContext.cid = cid;
    auto status = AllocateSubBatchFlagBuffer(batchNum, flagBufferManager, subBatchContext);
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }
    return Status::OK();
}

Status PackSubBatchRequest(ProtocolManager& protocolManager, BufferManager& sendBufferManager,
                           KvOpcode opcode, const SqeRequest& request,
                           TransportSubBatchContext& subBatchContext)
{
    auto packedSize = protocolManager.GetPackedSize(opcode, request);
    auto status = sendBufferManager.Allocate(packedSize, subBatchContext.sendSge);
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }

    status = protocolManager.PackRequest(reinterpret_cast<void*>(subBatchContext.sendSge.addr),
                                         opcode, request);
    if (!status.ok()) { return SetSubBatchBuildFailed(subBatchContext, status); }

    subBatchContext.status = status;
    return status;
}

Status InitializeSubBatchSubmission(TransportOpType opType, std::size_t batchNum,
                                    bool isSupported, const std::string& unsupportedMessage,
                                    TransportSubBatchContext& subBatchContext, KvOpcode& opcode,
                                    bool& shouldSubmit)
{
    ResetSubBatchContext(batchNum, subBatchContext);
    shouldSubmit = false;

    if (batchNum == 0) { return Status::OK(); }

    if (!isSupported) {
        auto status = Status::Error(StatusCode::UNSUPPORTED, unsupportedMessage);
        return SetSubBatchBuildFailed(subBatchContext, status);
    }
    opcode = ToKvOpcode(opType);
    shouldSubmit = true;
    return Status::OK();
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
    request.response_buffer_addr = flagBuffer.addr;
    request.response_mr_key = flagBuffer.lkey;
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
    request.response_buffer_addr = flagBuffer.addr;
    request.response_mr_key = flagBuffer.lkey;
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
    request.response_buffer_addr = flagBuffer.addr;
    request.response_mr_key = flagBuffer.lkey;
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
    request.response_buffer_addr = flagBuffer.addr;
    request.response_mr_key = flagBuffer.lkey;
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
    request.response_buffer_addr = flagBuffer.addr;
    request.response_mr_key = flagBuffer.lkey;
    request.rflag = true;
    return request;
}

std::unique_ptr<SqeRequest> BuildSqeRequest(
    KvOpcode opcode, const SubBatchRequestSource& source,
    const std::unordered_map<std::string, std::string>& attrs, std::uint16_t cid,
    const ScatterGatherEntry& flagBuffer, TransportSubBatchContext& subBatchContext)
{
    switch (opcode) {
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
        default: return nullptr;
    }
}

}  // namespace

Status AsuTransportImpl::ValidateSqeRequestAttrs()
{
    const auto validateInteger = [this](const std::string& name, auto maxValue) -> Status {
        auto iter = config_.attrs.find(name);
        if (iter == config_.attrs.end()) { return Status::OK(); }
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

    const auto validateRequiredPositiveInteger = [this](const std::string& name,
                                                        auto maxValue) -> Status {
        auto iter = config_.attrs.find(name);
        if (iter == config_.attrs.end()) {
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

    const auto validateBool = [this](const std::string& name) -> Status {
        auto iter = config_.attrs.find(name);
        if (iter == config_.attrs.end()) { return Status::OK(); }
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

Status AsuTransportImpl::SubmitEntrySubBatchRequest(TransportOpType opType,
                                                    const IoScheduler::ScheduledIoBatch& subBatch,
                                                    TransportSubBatchContext& subBatchContext)
{
    constexpr auto kUnsupportedMessage =
        "entry batch submit only supports batch store/retrieve operations";
    const auto source = SubBatchRequestSource::FromEntries(subBatch.entries);
    KvOpcode opcode{};
    bool shouldSubmit = false;
    auto status = InitializeSubBatchSubmission(opType, subBatch.entries.size,
                                               IsEntryBatchOp(opType), kUnsupportedMessage,
                                               subBatchContext, opcode, shouldSubmit);
    if (!status.ok() || !shouldSubmit) { return status; }

    status = PrepareSubBatchRequest(opType, AllocateRequestCid(), subBatch.entries.size,
                                    flagBufferManager_, subBatchContext);
    if (!status.ok()) { return status; }

    auto request = BuildSqeRequest(opcode, source, config_.attrs, subBatchContext.cid,
                                   subBatchContext.flagBuffer, subBatchContext);
    return PackSubBatchRequest(*protocolManager_, sendBufferManager_, opcode, *request,
                               subBatchContext);
}

Status AsuTransportImpl::SubmitKeySubBatchRequest(TransportOpType opType,
                                                  const IoScheduler::ScheduledKeyBatch& subBatch,
                                                  TransportSubBatchContext& subBatchContext)
{
    constexpr auto kUnsupportedMessage = "key batch submit only supports query/delete";
    const auto source = SubBatchRequestSource::FromKeys(subBatch.keys);
    KvOpcode opcode{};
    bool shouldSubmit = false;
    auto status = InitializeSubBatchSubmission(opType, subBatch.keys.size, IsKeyBatchOp(opType),
                                               kUnsupportedMessage, subBatchContext, opcode,
                                               shouldSubmit);
    if (!status.ok() || !shouldSubmit) { return status; }

    status = PrepareSubBatchRequest(opType, AllocateRequestCid(), subBatch.keys.size,
                                    flagBufferManager_, subBatchContext);
    if (!status.ok()) { return status; }

    auto request = BuildSqeRequest(opcode, source, config_.attrs, subBatchContext.cid,
                                   subBatchContext.flagBuffer, subBatchContext);
    return PackSubBatchRequest(*protocolManager_, sendBufferManager_, opcode, *request,
                               subBatchContext);
}

Status AsuTransportImpl::SubmitKeepAliveRequest(TransportSubBatchContext& subBatchContext)
{
    constexpr auto kUnsupportedMessage = "keep alive submit only supports keep alive";
    const auto opType = TransportOpType::KEEP_ALIVE;
    const auto source = SubBatchRequestSource::KeepAlive();
    KvOpcode opcode{};
    bool shouldSubmit = false;
    auto status = InitializeSubBatchSubmission(opType, 1, true, kUnsupportedMessage,
                                               subBatchContext, opcode, shouldSubmit);
    if (!status.ok() || !shouldSubmit) { return status; }

    status = PrepareSubBatchRequest(opType, AllocateRequestCid(), 1, flagBufferManager_,
                                    subBatchContext);
    if (!status.ok()) { return status; }

    auto request = BuildSqeRequest(opcode, source, config_.attrs, subBatchContext.cid,
                                   subBatchContext.flagBuffer, subBatchContext);
    return PackSubBatchRequest(*protocolManager_, sendBufferManager_, opcode, *request,
                               subBatchContext);
}

}  // namespace UC::ASU
