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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "asu_transport/types.h"

namespace UC::ASU {

enum class SqeOpcode : std::uint8_t {
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
constexpr std::uint32_t kAlignmentBytes = 512;
constexpr std::size_t kBatchEntrySizeBytes = 36;
constexpr std::size_t kBatchEntryDwordCount = 9;
constexpr std::size_t kDeleteEntrySizeBytes = 16;
constexpr std::size_t kDeleteEntryDwordCount = 4;
constexpr std::size_t kMaxBatchNumber = 227;

class SqeRequest {
public:
    virtual ~SqeRequest() = default;
};

class KvStoreRequest : public SqeRequest {
public:
    std::uint32_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint8_t dtype{0};
    std::uint8_t dspec{0};
    std::uint64_t buffer_addr{0};
    std::uint32_t buffer_length{0};
    std::uint32_t mr_key{0};
    std::uint32_t offset{0};
    bool lr{false};
    std::uint32_t length{0};
    std::string key;
};

class KvRetrieveRequest : public SqeRequest {
public:
    std::uint32_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint64_t buffer_addr{0};
    std::uint32_t buffer_length{0};
    std::uint32_t mr_key{0};
    std::uint32_t offset{0};
    bool lr{false};
    std::uint32_t length{0};
    std::string key;
};

class KvBatchStoreEntry {
public:
    std::uint32_t offset{0};
    std::string key;
    std::uint64_t buffer_addr{0};
    std::uint32_t mr_key{0};
    std::uint32_t length{0};
};

class KvBatchStoreRequest : public SqeRequest {
public:
    std::uint32_t cid{0};
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
    std::string key;
    std::uint64_t buffer_addr{0};
    std::uint32_t mr_key{0};
    std::uint32_t length{0};
};

class KvBatchRetrieveRequest : public SqeRequest {
public:
    std::uint32_t cid{0};
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
    std::uint32_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool rflag{false};
    std::uint16_t batch_number{0};
    std::vector<std::string> keys;
};

class KvExistRequest : public SqeRequest {
public:
    std::uint32_t cid{0};
    std::uint32_t kv_ns_id{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool rflag{false};
    bool sc{false};
    std::uint16_t batch_number{0};
    std::vector<std::string> keys;
};

class KvKeepAliveRequest : public SqeRequest {
public:
    std::uint32_t cid{0};
    std::uint64_t response_buffer_addr{0};
    std::uint32_t response_mr_key{0};
    bool rflag{false};
};

class Sqe {
public:
    virtual ~Sqe() = default;

    virtual std::uint32_t GetOpcode() const = 0;
    const void* Data() const { return dwords.data(); }
    std::size_t Size() const { return dwords.size() * sizeof(std::uint32_t); }

    virtual Status Pack(const SqeRequest& req) = 0;
    virtual Status Validate() const = 0;

protected:
    // TODO: Optimize memory allocation. Currently using vector for dynamic size,
    // but final implementation should write directly into SendBuffer to avoid copying.
    std::vector<std::uint32_t> dwords;
};

class SqeRegistry {
public:
    using Creator = std::function<std::unique_ptr<Sqe>()>;

    static void Register(std::uint32_t opcode, Creator creator)
    {
        Creators()[opcode] = std::move(creator);
    }

    static std::unique_ptr<Sqe> Create(std::uint32_t opcode)
    {
        auto& creators = Creators();
        auto it = creators.find(opcode);
        return it != creators.end() ? it->second() : nullptr;
    }

private:
    static std::unordered_map<std::uint32_t, Creator>& Creators()
    {
        static std::unordered_map<std::uint32_t, Creator> instance;
        return instance;
    }
};

#define REGISTER_SQE(ClassName, Opcode)                                        \
    inline static bool _reg_##ClassName = []() {                               \
        SqeRegistry::Register(static_cast<std::uint32_t>(Opcode),              \
                              []() { return std::make_unique<ClassName>(); }); \
        return true;                                                           \
    }();

class KvStoreSqe : public Sqe {
public:
    KvStoreSqe() = default;

    std::uint32_t GetOpcode() const override
    {
        return static_cast<std::uint32_t>(SqeOpcode::Store);
    }

    Status Pack(const SqeRequest& req) override;
    Status Validate() const override;
};

class KvRetrieveSqe : public Sqe {
public:
    KvRetrieveSqe() = default;

    std::uint32_t GetOpcode() const override
    {
        return static_cast<std::uint32_t>(SqeOpcode::Retrieve);
    }

    Status Pack(const SqeRequest& req) override;
    Status Validate() const override;
};

class KvBatchStoreSqe : public Sqe {
public:
    KvBatchStoreSqe() = default;

    std::uint32_t GetOpcode() const override
    {
        return static_cast<std::uint32_t>(SqeOpcode::BatchStore);
    }

    Status Pack(const SqeRequest& req) override;
    Status Validate() const override;

private:
    void PackEntry(const KvBatchStoreEntry& entry, std::size_t index);
};

class KvBatchRetrieveSqe : public Sqe {
public:
    KvBatchRetrieveSqe() = default;

    std::uint32_t GetOpcode() const override
    {
        return static_cast<std::uint32_t>(SqeOpcode::BatchRetrieve);
    }

    Status Pack(const SqeRequest& req) override;
    Status Validate() const override;

private:
    void PackEntry(const KvBatchRetrieveEntry& entry, std::size_t index);
};

class KvDeleteSqe : public Sqe {
public:
    KvDeleteSqe() = default;

    std::uint32_t GetOpcode() const override
    {
        return static_cast<std::uint32_t>(SqeOpcode::Delete);
    }

    Status Pack(const SqeRequest& req) override;
    Status Validate() const override;

private:
    void PackEntry(const std::string& key, std::size_t index);
};

class KvExistSqe : public Sqe {
public:
    KvExistSqe() = default;

    std::uint32_t GetOpcode() const override
    {
        return static_cast<std::uint32_t>(SqeOpcode::Exist);
    }

    Status Pack(const SqeRequest& req) override;
    Status Validate() const override;

private:
    void PackEntry(const std::string& key, std::size_t index);
};

class KvKeepAliveSqe : public Sqe {
public:
    KvKeepAliveSqe() = default;

    std::uint32_t GetOpcode() const override
    {
        return static_cast<std::uint32_t>(SqeOpcode::KeepAlive);
    }

    Status Pack(const SqeRequest& req) override;
    Status Validate() const override;
};

REGISTER_SQE(KvStoreSqe, SqeOpcode::Store)
REGISTER_SQE(KvRetrieveSqe, SqeOpcode::Retrieve)
REGISTER_SQE(KvBatchStoreSqe, SqeOpcode::BatchStore)
REGISTER_SQE(KvBatchRetrieveSqe, SqeOpcode::BatchRetrieve)
REGISTER_SQE(KvDeleteSqe, SqeOpcode::Delete)
REGISTER_SQE(KvExistSqe, SqeOpcode::Exist)
REGISTER_SQE(KvKeepAliveSqe, SqeOpcode::KeepAlive)

}  // namespace UC::ASU
