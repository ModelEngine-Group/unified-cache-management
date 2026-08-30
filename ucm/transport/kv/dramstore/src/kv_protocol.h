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
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../asu/trans/include/asu_transport/types.h"

namespace UC::DRAMPOOL {

using Status = UC::ASU::Status;
using StatusCode = UC::ASU::StatusCode;

enum class KvOpcode : std::uint8_t {
    None = 0x0,
    Dump = 0x1,
    Load = 0x2,
    Lookup = 0x3,
};

constexpr std::size_t kKvKeySize = 16;
constexpr std::size_t kKvHeaderSize =
    sizeof(std::uint8_t) + sizeof(std::uint64_t) + sizeof(std::uint16_t);
constexpr std::size_t kKvDumpLoadEntrySize =
    kKvKeySize + sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(std::uint32_t);
constexpr std::size_t kKvLookupEntrySize = kKvKeySize;
constexpr std::size_t kKvFlagEntrySize = sizeof(std::uint32_t);

// Wire offsets shared by the client (pack) and server (unpack) sides.
constexpr std::size_t kOpcodeOffset = 0;
constexpr std::size_t kRespAddrOffset = 1;
constexpr std::size_t kBatchSizeOffset = 9;

constexpr std::size_t kDumpLoadEntryKeyOffset = 0;
constexpr std::size_t kDumpLoadEntryAddrOffset = 16;
constexpr std::size_t kDumpLoadEntryLenOffset = 24;
constexpr std::size_t kDumpLoadEntryIdxOffset = 28;

constexpr std::size_t kLookupEntryKeyOffset = 0;

class KvDumpLoadEntry {
public:
    std::array<std::uint8_t, kKvKeySize> key{};
    std::uint64_t addr{0};
    std::uint32_t len{0};
    std::uint32_t idx{0};
};

class KvLookupEntry {
public:
    std::array<std::uint8_t, kKvKeySize> key{};
};

class KvRequest {
public:
    virtual ~KvRequest() = default;
};

class KvDumpLoadRequest : public KvRequest {
public:
    KvOpcode opcode{KvOpcode::None};
    std::uint64_t resp_addr{0};
    std::uint16_t batch_size{0};
    std::vector<KvDumpLoadEntry> entries;
};

class KvLookupRequest : public KvRequest {
public:
    KvOpcode opcode{KvOpcode::None};
    std::uint64_t resp_addr{0};
    std::uint16_t batch_size{0};
    std::vector<KvLookupEntry> entries;
};

class KvResponse {
public:
    std::vector<std::uint32_t> results;
};

// ---------------------------------------------------------------------------
// Free helpers (shared by pack / unpack paths)
// ---------------------------------------------------------------------------

// Peek the opcode byte from a packed request buffer (byte 0).
KvOpcode PeekOpcode(const void* data);

// True if every byte of the 16-byte key is zero.
bool IsAllZeroKey(const std::uint8_t* key);

// Human-readable protocol name for log prefixes (Dump/Load/Lookup, "Unknown" otherwise).
const char* ProtocolName(KvOpcode opcode);

// Write the 11-byte header (opcode + resp_addr + batch_size) into out.
void PackHeader(std::uint8_t* out, KvOpcode opcode, std::uint64_t resp_addr,
                std::uint16_t batch_size);

// Validate resp_addr != 0 and batch_size == entries.size() (and non-zero).
template <typename Req>
Status ValidateRequestHeader(const Req& req)
{
    const std::string name = ProtocolName(req.opcode);
    if (req.resp_addr == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name + ": resp_addr is zero");
    }
    if (req.batch_size == 0 || req.batch_size != req.entries.size()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             name + ": batch_size(" + std::to_string(req.batch_size) +
                                 ") must be non-zero and equal to entries.size()(" +
                                 std::to_string(req.entries.size()) + ")");
    }
    return Status::OK();
}

// ---------------------------------------------------------------------------
// Protocol classes
// ---------------------------------------------------------------------------

class KvProtocol {
public:
    virtual ~KvProtocol() = default;

    // Client side: pack request struct -> wire; unpack response wire -> struct.
    virtual std::size_t PackedSize(const KvRequest& req) const = 0;
    virtual Status PackRequest(const KvRequest& req, void* target) = 0;
    virtual Status UnpackResponse(const void* data, std::uint16_t result_count,
                                  KvResponse& out) const = 0;

    // Server side: unpack request wire -> struct; pack response struct -> wire.
    virtual Status UnpackRequest(const void* data, std::size_t size,
                                 std::unique_ptr<KvRequest>& out) const = 0;
    virtual Status PackResponse(void* data, const KvResponse& resp) const = 0;
};

class KvDumpLoadProtocol : public KvProtocol {
public:
    // Client side
    std::size_t PackedSize(const KvRequest& req) const override;
    Status PackRequest(const KvRequest& req, void* target) override;
    Status UnpackResponse(const void* data, std::uint16_t result_count,
                          KvResponse& out) const override;
    // Server side
    Status UnpackRequest(const void* data, std::size_t size,
                         std::unique_ptr<KvRequest>& out) const override;
    Status PackResponse(void* data, const KvResponse& resp) const override;

private:
    Status ValidateRequest(const KvDumpLoadRequest& req) const;
};

class KvLookupProtocol : public KvProtocol {
public:
    // Client side
    std::size_t PackedSize(const KvRequest& req) const override;
    Status PackRequest(const KvRequest& req, void* target) override;
    Status UnpackResponse(const void* data, std::uint16_t result_count,
                          KvResponse& out) const override;
    // Server side
    Status UnpackRequest(const void* data, std::size_t size,
                         std::unique_ptr<KvRequest>& out) const override;
    Status PackResponse(void* data, const KvResponse& resp) const override;

private:
    Status ValidateRequest(const KvLookupRequest& req) const;
};

class ProtocolManager {
public:
    ProtocolManager();
    ~ProtocolManager() = default;

    ProtocolManager(const ProtocolManager&) = delete;
    ProtocolManager& operator=(const ProtocolManager&) = delete;

    // Client side
    std::size_t GetPackedSize(KvOpcode opcode, const KvRequest& req) const;
    Status PackRequest(void* data, KvOpcode opcode, const KvRequest& req);
    Status UnpackResponse(const void* data, KvOpcode opcode, std::uint16_t result_count,
                          KvResponse& out);
    // Server side
    Status UnpackRequest(const void* data, std::size_t size, std::unique_ptr<KvRequest>& out);
    Status PackResponse(void* data, KvOpcode opcode, const KvResponse& resp);

private:
    std::unordered_map<KvOpcode, std::unique_ptr<KvProtocol>> protocols_;

    KvProtocol* GetProtocol(KvOpcode opcode) const;
};

}  // namespace UC::DRAMPOOL
