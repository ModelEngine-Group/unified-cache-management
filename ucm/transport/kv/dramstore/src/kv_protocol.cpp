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
#include <cstring>
#include <string>

namespace UC::DRAMPOOL {

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

KvOpcode PeekOpcode(const void* data)
{
    return static_cast<KvOpcode>(static_cast<const std::uint8_t*>(data)[kOpcodeOffset]);
}

bool IsAllZeroKey(const std::uint8_t* key)
{
    for (std::size_t i = 0; i < kKvKeySize; ++i) {
        if (key[i] != 0) { return false; }
    }
    return true;
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

void PackHeader(std::uint8_t* out, KvOpcode opcode, std::uint64_t resp_addr,
                std::uint16_t batch_size)
{
    out[kOpcodeOffset] = static_cast<std::uint8_t>(opcode);
    std::memcpy(out + kRespAddrOffset, &resp_addr, sizeof(resp_addr));
    std::memcpy(out + kBatchSizeOffset, &batch_size, sizeof(batch_size));
}

// ===========================================================================
// KvDumpLoadProtocol
// ===========================================================================

// ---- Client side ----

std::size_t KvDumpLoadProtocol::PackedSize(const KvRequest& req) const
{
    const auto& r = static_cast<const KvDumpLoadRequest&>(req);
    return kKvHeaderSize + static_cast<std::size_t>(r.batch_size) * kKvDumpLoadEntrySize;
}

Status KvDumpLoadProtocol::PackRequest(const KvRequest& req, void* target)
{
    const auto& r = static_cast<const KvDumpLoadRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    auto* out = static_cast<std::uint8_t*>(target);
    PackHeader(out, r.opcode, r.resp_addr, r.batch_size);

    for (std::size_t i = 0; i < r.entries.size(); ++i) {
        const auto& entry = r.entries[i];
        std::uint8_t* base = out + kKvHeaderSize + i * kKvDumpLoadEntrySize;
        std::memcpy(base + kDumpLoadEntryKeyOffset, entry.key.data(), kKvKeySize);
        std::memcpy(base + kDumpLoadEntryAddrOffset, &entry.addr, sizeof(entry.addr));
        std::memcpy(base + kDumpLoadEntryLenOffset, &entry.len, sizeof(entry.len));
        std::memcpy(base + kDumpLoadEntryIdxOffset, &entry.idx, sizeof(entry.idx));
    }
    return Status::OK();
}

Status KvDumpLoadProtocol::UnpackResponse(const void* data, std::uint16_t result_count,
                                          KvResponse& out) const
{
    if (!data) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "DumpLoad: UnpackResponse data is null");
    }
    out.results.resize(result_count);
    std::memcpy(out.results.data(), data,
                static_cast<std::size_t>(result_count) * sizeof(std::uint32_t));
    return Status::OK();
}

Status KvDumpLoadProtocol::ValidateRequest(const KvDumpLoadRequest& req) const
{
    if (req.opcode != KvOpcode::Dump && req.opcode != KvOpcode::Load) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "DumpLoad: opcode must be Dump or Load");
    }
    auto header_status = ValidateRequestHeader(req);
    if (!header_status.ok()) { return header_status; }

    for (std::size_t i = 0; i < req.entries.size(); ++i) {
        const auto& entry = req.entries[i];
        if (IsAllZeroKey(entry.key.data())) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "DumpLoad: entry[" + std::to_string(i) + "] key is all zero");
        }
        if (entry.addr == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "DumpLoad: entry[" + std::to_string(i) + "] addr is zero");
        }
        if (entry.len == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "DumpLoad: entry[" + std::to_string(i) + "] len is zero");
        }
    }
    return Status::OK();
}

// ---- Server side ----

Status KvDumpLoadProtocol::UnpackRequest(const void* data, std::size_t size,
                                         std::unique_ptr<KvRequest>& out) const
{
    auto* bytes = static_cast<const std::uint8_t*>(data);
    auto req = std::make_unique<KvDumpLoadRequest>();

    req->opcode = PeekOpcode(data);
    if (req->opcode != KvOpcode::Dump && req->opcode != KvOpcode::Load) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "DumpLoad: opcode mismatch");
    }

    std::memcpy(&req->resp_addr, bytes + kRespAddrOffset, sizeof(req->resp_addr));
    if (req->resp_addr == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "DumpLoad: resp_addr is zero");
    }

    std::memcpy(&req->batch_size, bytes + kBatchSizeOffset, sizeof(req->batch_size));
    if (req->batch_size == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "DumpLoad: batch_size is zero");
    }

    std::size_t expected_size =
        kKvHeaderSize + static_cast<std::size_t>(req->batch_size) * kKvDumpLoadEntrySize;
    if (size != expected_size) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "DumpLoad: size(" + std::to_string(size) + ") != expected(" +
                                 std::to_string(expected_size) + ")");
    }

    req->entries.resize(req->batch_size);
    for (std::size_t i = 0; i < req->batch_size; ++i) {
        const std::uint8_t* base = bytes + kKvHeaderSize + i * kKvDumpLoadEntrySize;
        std::memcpy(req->entries[i].key.data(), base + kDumpLoadEntryKeyOffset, kKvKeySize);
        if (IsAllZeroKey(req->entries[i].key.data())) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "DumpLoad: entry[" + std::to_string(i) + "] key is all zero");
        }
        std::memcpy(&req->entries[i].addr, base + kDumpLoadEntryAddrOffset, sizeof(std::uint64_t));
        if (req->entries[i].addr == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "DumpLoad: entry[" + std::to_string(i) + "] addr is zero");
        }
        std::memcpy(&req->entries[i].len, base + kDumpLoadEntryLenOffset, sizeof(std::uint32_t));
        if (req->entries[i].len == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "DumpLoad: entry[" + std::to_string(i) + "] len is zero");
        }
        std::memcpy(&req->entries[i].idx, base + kDumpLoadEntryIdxOffset, sizeof(std::uint32_t));
    }
    out = std::move(req);
    return Status::OK();
}

Status KvDumpLoadProtocol::PackResponse(void* data, const KvResponse& resp) const
{
    if (!data) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "DumpLoad: PackResponse data is null");
    }
    std::memcpy(data, resp.results.data(), resp.results.size() * sizeof(std::uint32_t));
    return Status::OK();
}

// ===========================================================================
// KvLookupProtocol
// ===========================================================================

// ---- Client side ----

std::size_t KvLookupProtocol::PackedSize(const KvRequest& req) const
{
    const auto& r = static_cast<const KvLookupRequest&>(req);
    return kKvHeaderSize + static_cast<std::size_t>(r.batch_size) * kKvLookupEntrySize;
}

Status KvLookupProtocol::PackRequest(const KvRequest& req, void* target)
{
    const auto& r = static_cast<const KvLookupRequest&>(req);
    auto status = ValidateRequest(r);
    if (!status.ok()) { return status; }

    auto* out = static_cast<std::uint8_t*>(target);
    PackHeader(out, r.opcode, r.resp_addr, r.batch_size);

    for (std::size_t i = 0; i < r.entries.size(); ++i) {
        const auto& entry = r.entries[i];
        std::uint8_t* base = out + kKvHeaderSize + i * kKvLookupEntrySize;
        std::memcpy(base + kLookupEntryKeyOffset, entry.key.data(), kKvKeySize);
    }
    return Status::OK();
}

Status KvLookupProtocol::UnpackResponse(const void* data, std::uint16_t result_count,
                                        KvResponse& out) const
{
    if (result_count != 1) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "Lookup: result_count must be 1, got " + std::to_string(result_count));
    }
    if (!data) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Lookup: UnpackResponse data is null");
    }
    out.results.resize(result_count);
    std::memcpy(out.results.data(), data, result_count * sizeof(std::uint32_t));
    return Status::OK();
}

Status KvLookupProtocol::ValidateRequest(const KvLookupRequest& req) const
{
    if (req.opcode != KvOpcode::Lookup) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Lookup: opcode must be Lookup");
    }
    auto header_status = ValidateRequestHeader(req);
    if (!header_status.ok()) { return header_status; }

    for (std::size_t i = 0; i < req.entries.size(); ++i) {
        if (IsAllZeroKey(req.entries[i].key.data())) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "Lookup: entry[" + std::to_string(i) + "] key is all zero");
        }
    }
    return Status::OK();
}

// ---- Server side ----

Status KvLookupProtocol::UnpackRequest(const void* data, std::size_t size,
                                       std::unique_ptr<KvRequest>& out) const
{
    auto* bytes = static_cast<const std::uint8_t*>(data);
    auto req = std::make_unique<KvLookupRequest>();

    req->opcode = PeekOpcode(data);
    if (req->opcode != KvOpcode::Lookup) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Lookup: opcode mismatch");
    }

    std::memcpy(&req->resp_addr, bytes + kRespAddrOffset, sizeof(req->resp_addr));
    if (req->resp_addr == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Lookup: resp_addr is zero");
    }

    std::memcpy(&req->batch_size, bytes + kBatchSizeOffset, sizeof(req->batch_size));
    if (req->batch_size == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Lookup: batch_size is zero");
    }

    std::size_t expected_size =
        kKvHeaderSize + static_cast<std::size_t>(req->batch_size) * kKvLookupEntrySize;
    if (size != expected_size) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Lookup: size(" + std::to_string(size) +
                                                               ") != expected(" +
                                                               std::to_string(expected_size) + ")");
    }

    req->entries.resize(req->batch_size);
    for (std::size_t i = 0; i < req->batch_size; ++i) {
        const std::uint8_t* base = bytes + kKvHeaderSize + i * kKvLookupEntrySize;
        std::memcpy(req->entries[i].key.data(), base + kLookupEntryKeyOffset, kKvKeySize);
        if (IsAllZeroKey(req->entries[i].key.data())) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "Lookup: entry[" + std::to_string(i) + "] key is all zero");
        }
    }
    out = std::move(req);
    return Status::OK();
}

Status KvLookupProtocol::PackResponse(void* data, const KvResponse& resp) const
{
    if (!data) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Lookup: PackResponse data is null");
    }
    if (resp.results.size() != 1) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "Lookup: results.size()(" + std::to_string(resp.results.size()) + ") must be 1");
    }
    std::memcpy(data, resp.results.data(), sizeof(std::uint32_t));
    return Status::OK();
}

// ===========================================================================
// ProtocolManager
// ===========================================================================

ProtocolManager::ProtocolManager()
{
    protocols_[KvOpcode::Dump] = std::make_unique<KvDumpLoadProtocol>();
    protocols_[KvOpcode::Load] = std::make_unique<KvDumpLoadProtocol>();
    protocols_[KvOpcode::Lookup] = std::make_unique<KvLookupProtocol>();
}

KvProtocol* ProtocolManager::GetProtocol(KvOpcode opcode) const
{
    auto it = protocols_.find(opcode);
    return it != protocols_.end() ? it->second.get() : nullptr;
}

// ---- Client side ----

std::size_t ProtocolManager::GetPackedSize(KvOpcode opcode, const KvRequest& req) const
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) { return 0; }
    return proto->PackedSize(req);
}

Status ProtocolManager::PackRequest(void* data, KvOpcode opcode, const KvRequest& req)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "unknown opcode in PackRequest");
    }
    return proto->PackRequest(req, data);
}

Status ProtocolManager::UnpackResponse(const void* data, KvOpcode opcode,
                                       std::uint16_t result_count, KvResponse& out)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "unknown opcode in UnpackResponse");
    }
    return proto->UnpackResponse(data, result_count, out);
}

// ---- Server side ----

Status ProtocolManager::UnpackRequest(const void* data, std::size_t size,
                                      std::unique_ptr<KvRequest>& out)
{
    if (!data || size < kKvHeaderSize) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "UnpackRequest: invalid data or size smaller than header");
    }
    KvOpcode opcode = PeekOpcode(data);
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "UnpackRequest: unknown opcode " + std::to_string(static_cast<std::uint8_t>(opcode)));
    }
    return proto->UnpackRequest(data, size, out);
}

Status ProtocolManager::PackResponse(void* data, KvOpcode opcode, const KvResponse& resp)
{
    KvProtocol* proto = GetProtocol(opcode);
    if (!proto) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "unknown opcode in PackResponse");
    }
    return proto->PackResponse(data, resp);
}

}  // namespace UC::DRAMPOOL
