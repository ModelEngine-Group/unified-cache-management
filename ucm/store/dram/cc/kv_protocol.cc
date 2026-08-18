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
#include "kv_protocol.h"
#include <algorithm>
#include <cstring>
#include <string>

namespace UC::DramPool {
namespace {

constexpr std::size_t kBitsPerByte = 8U;
constexpr std::size_t kLookupResultBitWidth = 1U;
constexpr std::size_t kDumpLoadResultBitWidth = 4U;

bool IsAllZeroKey(const BlockId& key);

template <typename Entry>
void PackDumpLoadEntry(std::uint8_t* output, const Entry& entry)
{
    std::memcpy(output + kDumpLoadEntryKeyOffset, entry.key.data(), kKvKeySize);
    std::memcpy(output + kDumpLoadEntryAddrOffset, &entry.addr, sizeof(entry.addr));
    std::memcpy(output + kDumpLoadEntryLenOffset, &entry.len, sizeof(entry.len));
    std::memcpy(output + kDumpLoadEntryIdxOffset, &entry.idx, sizeof(entry.idx));
}

template <typename Entry>
Status ValidateDumpLoadEntry(const Entry& entry, const char* protocol, std::size_t index)
{
    if (IsAllZeroKey(entry.key)) {
        return Status::InvalidParam(std::string{protocol} + ": entry[" + std::to_string(index) +
                                    "] key is all zero");
    }
    if (entry.addr == 0) {
        return Status::InvalidParam(std::string{protocol} + ": entry[" + std::to_string(index) +
                                    "] addr is zero");
    }
    if (entry.len == 0) {
        return Status::InvalidParam(std::string{protocol} + ": entry[" + std::to_string(index) +
                                    "] len is zero");
    }
    return Status::OK();
}

template <typename Entry>
Status UnpackDumpLoadEntry(const std::uint8_t* input, Entry& entry, const char* protocol,
                           std::size_t index)
{
    std::memcpy(entry.key.data(), input + kDumpLoadEntryKeyOffset, kKvKeySize);
    std::memcpy(&entry.addr, input + kDumpLoadEntryAddrOffset, sizeof(entry.addr));
    std::memcpy(&entry.len, input + kDumpLoadEntryLenOffset, sizeof(entry.len));
    std::memcpy(&entry.idx, input + kDumpLoadEntryIdxOffset, sizeof(entry.idx));
    return ValidateDumpLoadEntry(entry, protocol, index);
}

std::size_t GetPackedResultSize(std::size_t resultCount, std::size_t resultBitWidth)
{
    const std::size_t resultsPerByte = kBitsPerByte / resultBitWidth;
    return resultCount / resultsPerByte + (resultCount % resultsPerByte != 0U ? 1U : 0U);
}

Status PackResults(void* data, const std::vector<std::uint8_t>& results, std::size_t resultBitWidth,
                   const char* protocol)
{
    const auto maxResult = static_cast<std::uint8_t>((1U << resultBitWidth) - 1U);
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (results[index] > maxResult) {
            return Status::InvalidParam(std::string{protocol} + ": result[" +
                                        std::to_string(index) + "] exceeds " +
                                        std::to_string(resultBitWidth) + " bits");
        }
    }

    auto* packed = static_cast<std::uint8_t*>(data);
    std::fill_n(packed, GetPackedResultSize(results.size(), resultBitWidth), std::uint8_t{0});
    const std::size_t resultsPerByte = kBitsPerByte / resultBitWidth;
    for (std::size_t index = 0; index < results.size(); ++index) {
        const std::size_t bitOffset = (index % resultsPerByte) * resultBitWidth;
        packed[index / resultsPerByte] = static_cast<std::uint8_t>(packed[index / resultsPerByte] |
                                                                   (results[index] << bitOffset));
    }
    return Status::OK();
}

void UnpackResults(const void* data, std::size_t resultCount, std::size_t resultBitWidth,
                   std::vector<std::uint8_t>& results)
{
    const auto* packed = static_cast<const std::uint8_t*>(data);
    const std::size_t resultsPerByte = kBitsPerByte / resultBitWidth;
    const auto resultMask = static_cast<std::uint8_t>((1U << resultBitWidth) - 1U);
    results.resize(resultCount);
    for (std::size_t index = 0; index < resultCount; ++index) {
        const std::size_t bitOffset = (index % resultsPerByte) * resultBitWidth;
        results[index] =
            static_cast<std::uint8_t>((packed[index / resultsPerByte] >> bitOffset) & resultMask);
    }
}

KvOpcode PeekOpcode(const void* data)
{
    return static_cast<KvOpcode>(static_cast<const std::uint8_t*>(data)[kOpcodeOffset]);
}

bool IsAllZeroKey(const BlockId& key)
{
    return std::all_of(key.begin(), key.end(),
                       [](std::byte value) { return value == std::byte{}; });
}

const char* ProtocolName(KvOpcode opcode)
{
    switch (opcode) {
        case KvOpcode::None: return "Unknown";
        case KvOpcode::Dump: return "Dump";
        case KvOpcode::Load: return "Load";
        case KvOpcode::Lookup: return "Lookup";
    }
    return "Unknown";
}

template <typename Request>
Status ValidateRequestHeader(const Request& request)
{
    const std::string protocol = ProtocolName(request.opcode);
    if (request.request_id == 0) { return Status::InvalidParam(protocol + ": request_id is zero"); }
    if (request.resp_addr == 0) { return Status::InvalidParam(protocol + ": resp_addr is zero"); }
    if (request.batch_size == 0 || request.batch_size != request.entries.size()) {
        return Status::InvalidParam(protocol + ": batch_size(" +
                                    std::to_string(request.batch_size) +
                                    ") must be non-zero and equal to entries.size()(" +
                                    std::to_string(request.entries.size()) + ")");
    }
    return Status::OK();
}

void PackHeader(std::uint8_t* out, KvOpcode opcode, std::uint64_t request_id,
                std::uint64_t resp_addr, std::uint16_t batch_size)
{
    out[kOpcodeOffset] = static_cast<std::uint8_t>(opcode);
    std::memcpy(out + kRequestIdOffset, &request_id, sizeof(request_id));
    std::memcpy(out + kRespAddrOffset, &resp_addr, sizeof(resp_addr));
    std::memcpy(out + kLoadLookupBatchSizeOffset, &batch_size, sizeof(batch_size));
}

void PackDumpHeader(std::uint8_t* out, KvOpcode opcode, std::uint64_t request_id,
                    std::uint64_t resp_addr, std::uint32_t ttl, std::uint16_t batch_size)
{
    out[kOpcodeOffset] = static_cast<std::uint8_t>(opcode);
    std::memcpy(out + kRequestIdOffset, &request_id, sizeof(request_id));
    std::memcpy(out + kRespAddrOffset, &resp_addr, sizeof(resp_addr));
    std::memcpy(out + kDumpTtlOffset, &ttl, sizeof(ttl));
    std::memcpy(out + kDumpBatchSizeOffset, &batch_size, sizeof(batch_size));
}

template <typename RequestT>
Status UnpackCommonRequestHeader(const void* data, std::size_t size, std::size_t headerSize,
                                 std::size_t batchSizeOffset, std::size_t entrySize,
                                 KvOpcode expectedOpcode, RequestT& request)
{
    const std::string protocol = ProtocolName(expectedOpcode);
    if (!data || size < headerSize) {
        return Status::InvalidParam(protocol + ": invalid data or size smaller than header");
    }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    request.opcode = PeekOpcode(data);
    if (request.opcode != expectedOpcode) {
        return Status::InvalidParam(protocol + ": opcode mismatch");
    }
    std::memcpy(&request.request_id, bytes + kRequestIdOffset, sizeof(request.request_id));
    if (request.request_id == 0) { return Status::InvalidParam(protocol + ": request_id is zero"); }
    std::memcpy(&request.resp_addr, bytes + kRespAddrOffset, sizeof(request.resp_addr));
    if (request.resp_addr == 0) { return Status::InvalidParam(protocol + ": resp_addr is zero"); }
    std::memcpy(&request.batch_size, bytes + batchSizeOffset, sizeof(request.batch_size));
    if (request.batch_size == 0) { return Status::InvalidParam(protocol + ": batch_size is zero"); }
    const std::size_t expectedSize =
        headerSize + static_cast<std::size_t>(request.batch_size) * entrySize;
    if (size != expectedSize) {
        return Status::InvalidParam(protocol + ": size(" + std::to_string(size) + ") != expected(" +
                                    std::to_string(expectedSize) + ")");
    }
    return Status::OK();
}

}  // namespace

// ===========================================================================
// KvDumpProtocol
// ===========================================================================

// ---- Client side ----

std::size_t KvDumpProtocol::GetPackedRequestSize(const KvRequest& req) const
{
    const auto* request = dynamic_cast<const KvDumpRequest*>(&req);
    if (!request) { return 0; }
    return kKvDumpRequestHeaderSize +
           static_cast<std::size_t>(request->batch_size) * kKvDumpEntrySize;
}

std::size_t KvDumpProtocol::GetPackedResponseSize(std::size_t result_count) const
{
    return kResponseResultsOffset + GetPackedResultSize(result_count, kDumpLoadResultBitWidth);
}

Status KvDumpProtocol::PackRequest(const KvRequest& req, void* target)
{
    const auto* request = dynamic_cast<const KvDumpRequest*>(&req);
    if (!request) { return Status::InvalidParam("Dump: request type mismatch"); }
    const auto& r = *request;
    auto status = ValidateRequest(r);
    if (!status.Success()) { return status; }

    auto* out = static_cast<std::uint8_t*>(target);
    PackDumpHeader(out, r.opcode, r.request_id, r.resp_addr, r.ttl, r.batch_size);

    for (std::size_t i = 0; i < r.entries.size(); ++i) {
        const auto& entry = r.entries[i];
        std::uint8_t* base = out + kKvDumpRequestHeaderSize + i * kKvDumpEntrySize;
        PackDumpLoadEntry(base, entry);
    }
    return Status::OK();
}

Status KvDumpProtocol::UnpackResponse(const void* data, std::uint16_t result_count,
                                      KvResponse& out) const
{
    if (!data) { return Status::InvalidParam("Dump: UnpackResponse data is null"); }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::memcpy(&out.request_id, bytes + kResponseRequestIdOffset, sizeof(out.request_id));
    UnpackResults(bytes + kResponseResultsOffset, result_count, kDumpLoadResultBitWidth,
                  out.results);
    return Status::OK();
}

Status KvDumpProtocol::ValidateRequest(const KvDumpRequest& req) const
{
    if (req.opcode != KvOpcode::Dump) { return Status::InvalidParam("Dump: opcode must be Dump"); }
    auto header_status = ValidateRequestHeader(req);
    if (!header_status.Success()) { return header_status; }

    for (std::size_t i = 0; i < req.entries.size(); ++i) {
        if (auto status = ValidateDumpLoadEntry(req.entries[i], "Dump", i); status.Failure()) {
            return status;
        }
    }
    return Status::OK();
}

// ---- Server side ----

Status KvDumpProtocol::UnpackRequest(const void* data, std::size_t size,
                                     std::unique_ptr<KvRequest>& out) const
{
    auto req = std::make_unique<KvDumpRequest>();
    if (auto status =
            UnpackCommonRequestHeader(data, size, kKvDumpRequestHeaderSize, kDumpBatchSizeOffset,
                                      kKvDumpEntrySize, KvOpcode::Dump, *req);
        status.Failure()) {
        return status;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::memcpy(&req->ttl, bytes + kDumpTtlOffset, sizeof(req->ttl));

    req->entries.resize(req->batch_size);
    for (std::size_t i = 0; i < req->batch_size; ++i) {
        const std::uint8_t* base = bytes + kKvDumpRequestHeaderSize + i * kKvDumpEntrySize;
        if (auto status = UnpackDumpLoadEntry(base, req->entries[i], "Dump", i); status.Failure()) {
            return status;
        }
    }
    out = std::move(req);
    return Status::OK();
}

Status KvDumpProtocol::PackResponse(void* data, const KvResponse& resp) const
{
    if (!data) { return Status::InvalidParam("Dump: PackResponse data is null"); }
    if (resp.request_id == 0) { return Status::InvalidParam("Dump: request_id is zero"); }
    auto* bytes = static_cast<std::uint8_t*>(data);
    std::memcpy(bytes + kResponseRequestIdOffset, &resp.request_id, sizeof(resp.request_id));
    const auto status =
        PackResults(bytes + kResponseResultsOffset, resp.results, kDumpLoadResultBitWidth, "Dump");
    if (status.Failure()) { return status; }
    bytes[kResponseStatusOffset] = static_cast<std::uint8_t>(ResponseStatus::Ready);
    return Status::OK();
}

// ===========================================================================
// KvLoadProtocol
// ===========================================================================

// ---- Client side ----

std::size_t KvLoadProtocol::GetPackedRequestSize(const KvRequest& req) const
{
    const auto* request = dynamic_cast<const KvLoadRequest*>(&req);
    if (!request) { return 0; }
    return kKvLoadRequestHeaderSize +
           static_cast<std::size_t>(request->batch_size) * kKvLoadEntrySize;
}

std::size_t KvLoadProtocol::GetPackedResponseSize(std::size_t result_count) const
{
    return kResponseResultsOffset + GetPackedResultSize(result_count, kDumpLoadResultBitWidth);
}

Status KvLoadProtocol::PackRequest(const KvRequest& req, void* target)
{
    const auto* request = dynamic_cast<const KvLoadRequest*>(&req);
    if (!request) { return Status::InvalidParam("Load: request type mismatch"); }
    const auto& r = *request;
    auto status = ValidateRequest(r);
    if (!status.Success()) { return status; }

    auto* out = static_cast<std::uint8_t*>(target);
    PackHeader(out, r.opcode, r.request_id, r.resp_addr, r.batch_size);

    for (std::size_t i = 0; i < r.entries.size(); ++i) {
        const auto& entry = r.entries[i];
        std::uint8_t* base = out + kKvLoadRequestHeaderSize + i * kKvLoadEntrySize;
        PackDumpLoadEntry(base, entry);
    }
    return Status::OK();
}

Status KvLoadProtocol::UnpackResponse(const void* data, std::uint16_t result_count,
                                      KvResponse& out) const
{
    if (!data) { return Status::InvalidParam("Load: UnpackResponse data is null"); }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::memcpy(&out.request_id, bytes + kResponseRequestIdOffset, sizeof(out.request_id));
    UnpackResults(bytes + kResponseResultsOffset, result_count, kDumpLoadResultBitWidth,
                  out.results);
    return Status::OK();
}

Status KvLoadProtocol::ValidateRequest(const KvLoadRequest& req) const
{
    if (req.opcode != KvOpcode::Load) { return Status::InvalidParam("Load: opcode must be Load"); }
    auto header_status = ValidateRequestHeader(req);
    if (!header_status.Success()) { return header_status; }

    for (std::size_t i = 0; i < req.entries.size(); ++i) {
        if (auto status = ValidateDumpLoadEntry(req.entries[i], "Load", i); status.Failure()) {
            return status;
        }
    }
    return Status::OK();
}

// ---- Server side ----

Status KvLoadProtocol::UnpackRequest(const void* data, std::size_t size,
                                     std::unique_ptr<KvRequest>& out) const
{
    auto req = std::make_unique<KvLoadRequest>();
    if (auto status = UnpackCommonRequestHeader(data, size, kKvLoadRequestHeaderSize,
                                                kLoadLookupBatchSizeOffset, kKvLoadEntrySize,
                                                KvOpcode::Load, *req);
        status.Failure()) {
        return status;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    req->entries.resize(req->batch_size);
    for (std::size_t i = 0; i < req->batch_size; ++i) {
        const std::uint8_t* base = bytes + kKvLoadRequestHeaderSize + i * kKvLoadEntrySize;
        if (auto status = UnpackDumpLoadEntry(base, req->entries[i], "Load", i); status.Failure()) {
            return status;
        }
    }
    out = std::move(req);
    return Status::OK();
}

Status KvLoadProtocol::PackResponse(void* data, const KvResponse& resp) const
{
    if (!data) { return Status::InvalidParam("Load: PackResponse data is null"); }
    if (resp.request_id == 0) { return Status::InvalidParam("Load: request_id is zero"); }
    auto* bytes = static_cast<std::uint8_t*>(data);
    std::memcpy(bytes + kResponseRequestIdOffset, &resp.request_id, sizeof(resp.request_id));
    const auto status =
        PackResults(bytes + kResponseResultsOffset, resp.results, kDumpLoadResultBitWidth, "Load");
    if (status.Failure()) { return status; }
    bytes[kResponseStatusOffset] = static_cast<std::uint8_t>(ResponseStatus::Ready);
    return Status::OK();
}

// ===========================================================================
// KvLookupProtocol
// ===========================================================================

// ---- Client side ----

std::size_t KvLookupProtocol::GetPackedRequestSize(const KvRequest& req) const
{
    const auto* request = dynamic_cast<const KvLookupRequest*>(&req);
    if (!request) { return 0; }
    return kKvLookupRequestHeaderSize +
           static_cast<std::size_t>(request->batch_size) * kKvLookupEntrySize;
}

std::size_t KvLookupProtocol::GetPackedResponseSize(std::size_t result_count) const
{
    return kResponseResultsOffset + GetPackedResultSize(result_count, kLookupResultBitWidth);
}

Status KvLookupProtocol::PackRequest(const KvRequest& req, void* target)
{
    const auto* request = dynamic_cast<const KvLookupRequest*>(&req);
    if (!request) { return Status::InvalidParam("Lookup: request type mismatch"); }
    const auto& r = *request;
    auto status = ValidateRequest(r);
    if (!status.Success()) { return status; }

    auto* out = static_cast<std::uint8_t*>(target);
    PackHeader(out, r.opcode, r.request_id, r.resp_addr, r.batch_size);

    for (std::size_t i = 0; i < r.entries.size(); ++i) {
        const auto& entry = r.entries[i];
        std::uint8_t* base = out + kKvLookupRequestHeaderSize + i * kKvLookupEntrySize;
        std::memcpy(base + kLookupEntryKeyOffset, entry.key.data(), kKvKeySize);
    }
    return Status::OK();
}

Status KvLookupProtocol::UnpackResponse(const void* data, std::uint16_t result_count,
                                        KvResponse& out) const
{
    if (!data) { return Status::InvalidParam("Lookup: UnpackResponse data is null"); }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::memcpy(&out.request_id, bytes + kResponseRequestIdOffset, sizeof(out.request_id));
    UnpackResults(bytes + kResponseResultsOffset, result_count, kLookupResultBitWidth, out.results);
    return Status::OK();
}

Status KvLookupProtocol::ValidateRequest(const KvLookupRequest& req) const
{
    if (req.opcode != KvOpcode::Lookup) {
        return Status::InvalidParam("Lookup: opcode must be Lookup");
    }
    auto header_status = ValidateRequestHeader(req);
    if (!header_status.Success()) { return header_status; }

    for (std::size_t i = 0; i < req.entries.size(); ++i) {
        if (IsAllZeroKey(req.entries[i].key)) {
            return Status::InvalidParam("Lookup: entry[" + std::to_string(i) + "] key is all zero");
        }
    }
    return Status::OK();
}

// ---- Server side ----

Status KvLookupProtocol::UnpackRequest(const void* data, std::size_t size,
                                       std::unique_ptr<KvRequest>& out) const
{
    auto req = std::make_unique<KvLookupRequest>();
    if (auto status = UnpackCommonRequestHeader(data, size, kKvLookupRequestHeaderSize,
                                                kLoadLookupBatchSizeOffset, kKvLookupEntrySize,
                                                KvOpcode::Lookup, *req);
        status.Failure()) {
        return status;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    req->entries.resize(req->batch_size);
    for (std::size_t i = 0; i < req->batch_size; ++i) {
        const std::uint8_t* base = bytes + kKvLookupRequestHeaderSize + i * kKvLookupEntrySize;
        std::memcpy(req->entries[i].key.data(), base + kLookupEntryKeyOffset, kKvKeySize);
        if (IsAllZeroKey(req->entries[i].key)) {
            return Status::InvalidParam("Lookup: entry[" + std::to_string(i) + "] key is all zero");
        }
    }
    out = std::move(req);
    return Status::OK();
}

Status KvLookupProtocol::PackResponse(void* data, const KvResponse& resp) const
{
    if (!data) { return Status::InvalidParam("Lookup: PackResponse data is null"); }
    if (resp.request_id == 0) { return Status::InvalidParam("Lookup: request_id is zero"); }
    if (resp.results.empty()) { return Status::InvalidParam("Lookup: results must not be empty"); }
    auto* bytes = static_cast<std::uint8_t*>(data);
    std::memcpy(bytes + kResponseRequestIdOffset, &resp.request_id, sizeof(resp.request_id));
    const auto status =
        PackResults(bytes + kResponseResultsOffset, resp.results, kLookupResultBitWidth, "Lookup");
    if (status.Failure()) { return status; }
    bytes[kResponseStatusOffset] = static_cast<std::uint8_t>(ResponseStatus::Ready);
    return Status::OK();
}

// ===========================================================================
// ProtocolManager
// ===========================================================================

ProtocolManager::ProtocolManager()
{
    protocols_[KvOpcode::Dump] = std::make_unique<KvDumpProtocol>();
    protocols_[KvOpcode::Load] = std::make_unique<KvLoadProtocol>();
    protocols_[KvOpcode::Lookup] = std::make_unique<KvLookupProtocol>();
}

KvProtocol* ProtocolManager::GetProtocol(KvOpcode opcode) const
{
    auto it = protocols_.find(opcode);
    return it != protocols_.end() ? it->second.get() : nullptr;
}

// ---- Client side ----

std::size_t ProtocolManager::GetPackedRequestSize(KvOpcode opcode, const KvRequest& req) const
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) { return 0; }
    return proto->GetPackedRequestSize(req);
}

std::size_t ProtocolManager::GetPackedResponseSize(KvOpcode opcode, std::size_t result_count) const
{
    auto* proto = GetProtocol(opcode);
    return proto ? proto->GetPackedResponseSize(result_count) : 0;
}

Status ProtocolManager::PackRequest(void* data, KvOpcode opcode, const KvRequest& req)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) { return Status::InvalidParam("unknown opcode in PackRequest"); }
    if (!data) { return Status::InvalidParam("PackRequest data is null"); }
    return proto->PackRequest(req, data);
}

Status ProtocolManager::IsResponseReady(const void* data, std::uint64_t expected_request_id,
                                        bool& ready) const
{
    ready = false;
    if (!data) { return Status::InvalidParam("IsResponseReady data is null"); }
    if (expected_request_id == 0) {
        return Status::InvalidParam("IsResponseReady expected_request_id is zero");
    }

    // The flag can be updated by remote DMA, so each poll must reload the status byte.
    const auto* bytes = static_cast<const volatile std::uint8_t*>(data);
    const auto status = static_cast<ResponseStatus>(bytes[kResponseStatusOffset]);
    switch (status) {
        case ResponseStatus::Pending: return Status::OK();
        case ResponseStatus::Ready: {
            std::uint64_t observedRequestId = 0;
            std::memcpy(&observedRequestId,
                        static_cast<const std::uint8_t*>(data) + kResponseRequestIdOffset,
                        sizeof(observedRequestId));
            if (observedRequestId != expected_request_id) {
                return Status::Error("IsResponseReady request_id mismatch: expected=" +
                                     std::to_string(expected_request_id) +
                                     ", observed=" + std::to_string(observedRequestId));
            }
            ready = true;
            return Status::OK();
        }
        default: return Status::InvalidParam("unknown response status");
    }
}

Status ProtocolManager::UnpackResponse(const void* data, KvOpcode opcode,
                                       std::uint64_t expected_request_id,
                                       std::uint16_t result_count, KvResponse& out)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) { return Status::InvalidParam("unknown opcode in UnpackResponse"); }
    bool ready = false;
    const auto status = IsResponseReady(data, expected_request_id, ready);
    if (status.Failure()) { return status; }
    if (!ready) { return Status::Retry(); }
    return proto->UnpackResponse(data, result_count, out);
}

// ---- Server side ----

Status ProtocolManager::UnpackRequest(const void* data, std::size_t size,
                                      std::unique_ptr<KvRequest>& out)
{
    if (!data || size < sizeof(std::uint8_t)) {
        return Status::InvalidParam("UnpackRequest: invalid data or missing opcode");
    }
    KvOpcode opcode = PeekOpcode(data);
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::InvalidParam("UnpackRequest: unknown opcode " +
                                    std::to_string(static_cast<std::uint8_t>(opcode)));
    }
    return proto->UnpackRequest(data, size, out);
}

Status ProtocolManager::PackResponse(void* data, KvOpcode opcode, const KvResponse& resp)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) { return Status::InvalidParam("unknown opcode in PackResponse"); }
    return proto->PackResponse(data, resp);
}

}  // namespace UC::DramPool
