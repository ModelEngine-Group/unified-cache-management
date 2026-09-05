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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "types.h"

namespace kv {

enum class KvOpcode : std::uint8_t {
    Store = 0x1,
    Retrieve = 0x2,
    BatchStore = 0x45,
    BatchRetrieve = 0x46,
    Delete = 0x8,
    Exist = 0xC,
    KeepAlive = 0xF4
};

enum class DptrType : std::uint8_t { Standard = 0x40, Batch = 0x1 };

constexpr std::size_t kSqeDwordCount = 16;
constexpr std::uint32_t kFixedBits = 0x3;
constexpr std::uint32_t kAlignmentBytes = kAsuAlignmentBytes;
constexpr std::size_t kBatchEntrySizeBytes = 36;
constexpr std::size_t kBatchEntryDwordCount = 9;
constexpr std::size_t kKeyEntrySizeBytes = 16;
constexpr std::size_t kKeyEntryDwordCount = 4;
constexpr std::size_t kMaxBatchNumber = 110;
constexpr std::size_t kMaxDeleteBatchNumber = 254;
constexpr std::size_t kMaxExistBatchNumber = 256;
constexpr std::size_t kCqeDwordCount = 4;

class SqeRequest {
public:
    virtual ~SqeRequest() = default;
};

class KvStoreRequest : public SqeRequest {
public:
    std::uint16_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint8_t dtype{0};
    std::uint8_t dspec{0};
    std::uint64_t buffer_addr{0};
    std::uint32_t buffer_length{0};
    std::uint32_t mr_key{0};
    std::uint32_t offset{0};
    bool lr{false};
    std::uint32_t length{0};
    CacheKey key{};
};

class KvRetrieveRequest : public SqeRequest {
public:
    std::uint16_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint64_t buffer_addr{0};
    std::uint32_t buffer_length{0};
    std::uint32_t mr_key{0};
    std::uint32_t offset{0};
    bool lr{false};
    std::uint32_t length{0};
    CacheKey key{};
};

class KvBatchStoreEntry {
public:
    std::uint32_t offset{0};
    CacheKey key{};
    std::uint64_t buffer_addr{0};
    std::uint32_t mr_key{0};
    std::uint32_t length{0};
};

class KvBatchStoreRequest : public SqeRequest {
public:
    std::uint16_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint8_t dtype{0};
    std::uint8_t dspec{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool lr{false};
    bool rflag{false};
    std::uint16_t batch_number{0};
    std::vector<KvBatchStoreEntry> entries;
};

class KvBatchRetrieveEntry {
public:
    std::uint32_t offset{0};
    CacheKey key{};
    std::uint64_t buffer_addr{0};
    std::uint32_t mr_key{0};
    std::uint32_t length{0};
};

class KvBatchRetrieveRequest : public SqeRequest {
public:
    std::uint16_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool lr{false};
    bool rflag{false};
    std::uint16_t batch_number{0};
    std::vector<KvBatchRetrieveEntry> entries;
};

class KvDeleteRequest : public SqeRequest {
public:
    std::uint16_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool rflag{false};
    std::uint16_t batch_number{0};
    std::vector<CacheKey> keys;
};

class KvExistRequest : public SqeRequest {
public:
    std::uint16_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool rflag{false};
    bool sc{true};
    std::uint16_t batch_number{0};
    std::vector<CacheKey> keys;
};

class KvKeepAliveRequest : public SqeRequest {
public:
    std::uint16_t cid{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool rflag{false};
};

class KvResponse {
public:
    virtual ~KvResponse() = default;
    std::uint16_t cid{0};
    std::uint16_t status{0};
    std::uint16_t existing_key_number{0};
    std::vector<std::uint8_t> result_buffer;
};

class KvProtocol {
public:
    virtual ~KvProtocol() = default;

    virtual Status PackSqe(const SqeRequest& req, std::uint32_t* target) = 0;
    virtual std::size_t PackedSize(const SqeRequest& req) const = 0;

    virtual Status UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                             KvResponse& out) const
    {
        return Status::Error(StatusCode::UNSUPPORTED, "CQE unpack not supported for this opcode");
    }

    virtual Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const = 0;
};

class KvStoreProtocol : public KvProtocol {
public:
    Status PackSqe(const SqeRequest& req, std::uint32_t* target) override;
    std::size_t PackedSize(const SqeRequest& req) const override
    {
        return kSqeDwordCount * sizeof(std::uint32_t);
    }
    Status UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                     KvResponse& out) const override;
    Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const override;

private:
    Status ValidateRequest(const KvStoreRequest& r) const;
};

class KvRetrieveProtocol : public KvProtocol {
public:
    Status PackSqe(const SqeRequest& req, std::uint32_t* target) override;
    std::size_t PackedSize(const SqeRequest& req) const override
    {
        return kSqeDwordCount * sizeof(std::uint32_t);
    }
    Status UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                     KvResponse& out) const override;
    Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const override;

private:
    Status ValidateRequest(const KvRetrieveRequest& r) const;
};

class KvBatchStoreProtocol : public KvProtocol {
public:
    Status PackSqe(const SqeRequest& req, std::uint32_t* target) override;
    std::size_t PackedSize(const SqeRequest& req) const override;
    Status UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                     KvResponse& out) const override;
    Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const override;

private:
    Status ValidateRequest(const KvBatchStoreRequest& r) const;
    static void PackEntry(const KvBatchStoreEntry& entry, std::uint32_t* base);
};

class KvBatchRetrieveProtocol : public KvProtocol {
public:
    Status PackSqe(const SqeRequest& req, std::uint32_t* target) override;
    std::size_t PackedSize(const SqeRequest& req) const override;
    Status UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                     KvResponse& out) const override;
    Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const override;

private:
    Status ValidateRequest(const KvBatchRetrieveRequest& r) const;
    static void PackEntry(const KvBatchRetrieveEntry& entry, std::uint32_t* base);
};

class KvDeleteProtocol : public KvProtocol {
public:
    Status PackSqe(const SqeRequest& req, std::uint32_t* target) override;
    std::size_t PackedSize(const SqeRequest& req) const override;
    Status UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                     KvResponse& out) const override;
    Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const override;

private:
    Status ValidateRequest(const KvDeleteRequest& r) const;
    static void PackEntry(const CacheKey& key, std::uint32_t* base);
};

class KvExistProtocol : public KvProtocol {
public:
    Status PackSqe(const SqeRequest& req, std::uint32_t* target) override;
    std::size_t PackedSize(const SqeRequest& req) const override;
    Status UnpackCqe(const std::uint32_t* data, std::uint16_t batch_number,
                     KvResponse& out) const override;
    Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const override;

private:
    Status ValidateRequest(const KvExistRequest& r) const;
    static void PackEntry(const CacheKey& key, std::uint32_t* base);
};

class KvKeepAliveProtocol : public KvProtocol {
public:
    Status PackSqe(const SqeRequest& req, std::uint32_t* target) override;
    std::size_t PackedSize(const SqeRequest& req) const override
    {
        return kSqeDwordCount * sizeof(std::uint32_t);
    }
    Status VerifyPackedBuffer(const std::uint32_t* data, std::size_t length) const override;

private:
    Status ValidateRequest(const KvKeepAliveRequest& r) const;
};

class ProtocolManager {
public:
    ProtocolManager();
    ~ProtocolManager() = default;

    ProtocolManager(const ProtocolManager&) = delete;
    ProtocolManager& operator=(const ProtocolManager&) = delete;

    std::size_t GetPackedSize(KvOpcode opcode, const SqeRequest& req) const;
    Status PackRequest(void* data_ptr, KvOpcode opcode, const SqeRequest& req);
    Status UnpackResponse(const void* data_ptr, KvOpcode opcode, std::uint16_t batch_number,
                          KvResponse& out);
    Status PollResponseCid(const void* data_ptr, std::uint16_t& cid) const;
    Status VerifyPackedBuffer(const void* data_ptr, std::size_t length);

private:
    std::unordered_map<KvOpcode, std::unique_ptr<KvProtocol>> protocols_;

    KvProtocol* GetProtocol(KvOpcode opcode) const;
};

}  // namespace kv
