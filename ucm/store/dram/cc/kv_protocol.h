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
#include <unordered_map>
#include <vector>
#include "status/status.h"
#include "type/types.h"

namespace UC::DramPool {

using BlockId = UC::Detail::BlockId;

enum class KvOpcode : std::uint8_t {
    None = 0x0,
    Dump = 0x1,
    Load = 0x2,
    Lookup = 0x3,
};

enum class ResponseStatus : std::uint8_t {
    Pending = 0,
    Ready = 1,
};

constexpr std::size_t kKvKeySize = sizeof(BlockId);
static_assert(kKvKeySize == 16, "DramPool wire protocol requires a 16-byte BlockId");
constexpr std::size_t kKvLoadRequestHeaderSize =
    sizeof(std::uint8_t) + sizeof(std::uint64_t) + sizeof(std::uint16_t);
constexpr std::size_t kKvLookupRequestHeaderSize =
    sizeof(std::uint8_t) + sizeof(std::uint64_t) + sizeof(std::uint16_t);
constexpr std::size_t kKvDumpRequestHeaderSize =
    sizeof(std::uint8_t) + sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(std::uint16_t);
constexpr std::size_t kKvDumpEntrySize =
    kKvKeySize + sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(std::uint32_t);
constexpr std::size_t kKvLoadEntrySize =
    kKvKeySize + sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(std::uint32_t);
constexpr std::size_t kKvLookupEntrySize = kKvKeySize;
constexpr std::size_t kResponseStatusOffset = 0;
constexpr std::size_t kResponseResultsOffset = sizeof(std::uint8_t);

// Wire offsets shared by the client (pack) and server (unpack) sides.
constexpr std::size_t kOpcodeOffset = 0;
constexpr std::size_t kRespAddrOffset = 1;
constexpr std::size_t kLoadLookupBatchSizeOffset = 9;
constexpr std::size_t kDumpTtlOffset = 9;
constexpr std::size_t kDumpBatchSizeOffset = 13;

constexpr std::size_t kDumpLoadEntryKeyOffset = 0;
constexpr std::size_t kDumpLoadEntryAddrOffset = 16;
constexpr std::size_t kDumpLoadEntryLenOffset = 24;
constexpr std::size_t kDumpLoadEntryIdxOffset = 28;

constexpr std::size_t kLookupEntryKeyOffset = 0;

class KvDumpEntry {
public:
    BlockId key{};
    std::uint64_t addr{0};
    std::uint32_t len{0};
    std::uint32_t idx{0};
};

class KvLoadEntry {
public:
    BlockId key{};
    std::uint64_t addr{0};
    std::uint32_t len{0};
    std::uint32_t idx{0};
};

class KvLookupEntry {
public:
    BlockId key{};
};

class KvRequest {
public:
    virtual ~KvRequest() = default;
    KvOpcode opcode{KvOpcode::None};
};

class KvDumpRequest : public KvRequest {
public:
    std::uint64_t resp_addr{0};
    std::uint32_t ttl{0};
    std::uint16_t batch_size{0};
    std::vector<KvDumpEntry> entries;
};

class KvLoadRequest : public KvRequest {
public:
    std::uint64_t resp_addr{0};
    std::uint16_t batch_size{0};
    std::vector<KvLoadEntry> entries;
};

class KvLookupRequest : public KvRequest {
public:
    std::uint64_t resp_addr{0};
    std::uint16_t batch_size{0};
    std::vector<KvLookupEntry> entries;
};

class KvResponse {
public:
    std::vector<std::uint8_t> results;
};

// ---------------------------------------------------------------------------
// Protocol classes
// ---------------------------------------------------------------------------

class KvProtocol {
public:
    virtual ~KvProtocol() = default;

    // Client side: pack request struct -> wire; unpack response wire -> struct.
    virtual std::size_t GetPackedRequestSize(const KvRequest& req) const = 0;
    virtual std::size_t GetPackedResponseSize(std::size_t result_count) const = 0;
    virtual Status PackRequest(const KvRequest& req, void* target) = 0;
    virtual Status UnpackResponse(const void* data, std::uint16_t result_count,
                                  KvResponse& out) const = 0;

    // Server side: unpack request wire -> struct; pack response struct -> wire.
    virtual Status UnpackRequest(const void* data, std::size_t size,
                                 std::unique_ptr<KvRequest>& out) const = 0;
    virtual Status PackResponse(void* data, const KvResponse& resp) const = 0;
};

class KvDumpProtocol : public KvProtocol {
public:
    // Client side
    std::size_t GetPackedRequestSize(const KvRequest& req) const override;
    std::size_t GetPackedResponseSize(std::size_t result_count) const override;
    Status PackRequest(const KvRequest& req, void* target) override;
    Status UnpackResponse(const void* data, std::uint16_t result_count,
                          KvResponse& out) const override;
    // Server side
    Status UnpackRequest(const void* data, std::size_t size,
                         std::unique_ptr<KvRequest>& out) const override;
    Status PackResponse(void* data, const KvResponse& resp) const override;

private:
    Status ValidateRequest(const KvDumpRequest& req) const;
};

class KvLoadProtocol : public KvProtocol {
public:
    // Client side
    std::size_t GetPackedRequestSize(const KvRequest& req) const override;
    std::size_t GetPackedResponseSize(std::size_t result_count) const override;
    Status PackRequest(const KvRequest& req, void* target) override;
    Status UnpackResponse(const void* data, std::uint16_t result_count,
                          KvResponse& out) const override;
    // Server side
    Status UnpackRequest(const void* data, std::size_t size,
                         std::unique_ptr<KvRequest>& out) const override;
    Status PackResponse(void* data, const KvResponse& resp) const override;

private:
    Status ValidateRequest(const KvLoadRequest& req) const;
};

class KvLookupProtocol : public KvProtocol {
public:
    // Client side
    std::size_t GetPackedRequestSize(const KvRequest& req) const override;
    std::size_t GetPackedResponseSize(std::size_t result_count) const override;
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
    std::size_t GetPackedRequestSize(KvOpcode opcode, const KvRequest& req) const;
    std::size_t GetPackedResponseSize(KvOpcode opcode, std::size_t result_count) const;
    Status PackRequest(void* data, KvOpcode opcode, const KvRequest& req);
    Status IsResponseReady(const void* data, bool& ready) const;
    Status UnpackResponse(const void* data, KvOpcode opcode, std::uint16_t result_count,
                          KvResponse& out);
    // Server side
    Status UnpackRequest(const void* data, std::size_t size, std::unique_ptr<KvRequest>& out);
    Status PackResponse(void* data, KvOpcode opcode, const KvResponse& resp);

private:
    std::unordered_map<KvOpcode, std::unique_ptr<KvProtocol>> protocols_;

    KvProtocol* GetProtocol(KvOpcode opcode) const;
};

}  // namespace UC::DramPool
