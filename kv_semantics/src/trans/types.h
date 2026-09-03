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
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace UC::ASU {

using TaskId = std::uint64_t;
using MRHandle = std::uint64_t;
constexpr std::size_t kCacheKeySizeBytes = 8;
using CacheKey = std::array<std::byte, kCacheKeySizeBytes>;
using AsuId = std::uint64_t;

inline std::string_view CacheKeyView(const CacheKey& key)
{
    return {reinterpret_cast<const char*>(key.data()), key.size()};
}

inline std::string CacheKeyToHex(const CacheKey& key)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string text;
    text.reserve(key.size() * 2);
    for (auto byte : key) {
        const auto value = static_cast<unsigned char>(std::to_integer<unsigned char>(byte));
        text.push_back(kHex[value >> 4]);
        text.push_back(kHex[value & 0x0F]);
    }
    return text;
}

enum class TransProviderType { AICPU, FAKE, AIV, UNSUPPORTED };

constexpr TaskId kInvalidTaskId = 0;
constexpr MRHandle kInvalidMRHandle = 0;
constexpr std::uint32_t kAsuAlignmentBytes = 512;  // KV protocol requires 512B alignment

enum class StatusCode {
    OK = 0,
    INVALID_ARGUMENT,
    NOT_INITIALIZED,
    TIMEOUT,
    NOT_FOUND,
    PARTIAL_FAILED,
    CONNECTION_ERROR,
    IO_ERROR,
    BUFFER_NOT_REGISTERED,
    BUFFER_NOT_SUPPORTED,
    TASK_NOT_FOUND,
    RESOURCE_BUSY,
    UNSUPPORTED,
    IN_PROGRESS,
    INTERNAL_ERROR,
    CANCELED,

    // ASU entry status codes keep raw entry result values in the low byte.
    ASU_ENTRY_RETRY_ADVISED = 0x0100 | 0x01,
    ASU_ENTRY_NO_RETRY_ADVISED = 0x0100 | 0x02,
    ASU_ENTRY_KEY_NOT_FOUND = 0x0100 | 0x03,
    ASU_ENTRY_DATA_NOT_EXIST = 0x0100 | 0x04,
    ASU_ENTRY_DELETE_FAILED = 0x0200 | 0x01,
    ASU_ENTRY_KEY_NOT_EXIST = 0x0300 | 0x00,
    ASU_ENTRY_KEY_EXIST = 0x0300 | 0x01,

    ASU_CQE_INVALID_COMMAND_OPCODE = 0x10000 | 0x001,
    ASU_CQE_INVALID_FIELD_IN_COMMAND = 0x10000 | 0x002,
    ASU_CQE_INTERNAL_ERROR = 0x10000 | 0x006,
    ASU_CQE_WRITE_FAULT = 0x10000 | 0x280,
    ASU_CQE_UNRECOVERED_READ_ERROR = 0x10000 | 0x281,
    ASU_CQE_KEY_NOT_EXIST = 0x10000 | 0x701,
    ASU_CQE_OUT_OF_CREATE_SIZE = 0x10000 | 0x712,
    ASU_CQE_IO_TIMEOUT = 0x10000 | 0x716,
    ASU_CQE_KEY_ALREADY_EXISTED = 0x10000 | 0x723,
    ASU_CQE_RESOURCE_BUSY = 0x10000 | 0x731,
    ASU_CQE_CHECK_RESULT_BUFFER = 0x10000 | 0x732,
};

struct Status {
    StatusCode code{StatusCode::OK};
    std::string message;

    bool ok() const noexcept { return code == StatusCode::OK; }

    static Status OK() { return {}; }
    static Status Error(StatusCode c, std::string msg) { return Status{c, std::move(msg)}; }
};

enum class Protocol {
    UB = 0,
    ROCE = 1,
    TCP = 2,
};

struct ServerKvCapabilities {
    // A zero limit means that the provider did not advertise that capability.
    std::uint32_t queueNum{0};
    std::uint32_t ioQueueDepth{0};
    std::uint32_t ioQueueKeyConcurrency{0};     // Placeholder
    std::uint32_t connectionKeyConcurrency{0};  // Placeholder
    std::uint64_t singleValueMaxBytes{0};
    std::uint64_t batchValueMaxBytes{0};
    std::uint32_t batchStoreKeys{0};
    std::uint32_t batchLoadKeys{0};
    std::uint32_t deleteKeys{0};
    std::uint32_t queryKeys{0};
    std::uint32_t keyLength{0};
    std::uint32_t kvCapabilities{0};  // Placeholder
};

struct QueryResult {
    std::vector<std::uint8_t> exists;
    std::uint32_t prefixHitKeys{0};
};

enum class MemoryType {
    HOST = 0,
    HOST_PINNED = 1,
    DEVICE = 2,
};

struct MemoryRegion {  // 地址范围抽象
    MemoryType memoryType{MemoryType::HOST};
    std::uint64_t addr{0};
    std::uint64_t size{0};
    std::int32_t deviceId{-1};
    std::int32_t numaNode{-1};
};

struct Buffer {  // 单个IO
    MemoryRegion region;
    MRHandle handle{kInvalidMRHandle};
};

struct KVBuffer {
    CacheKey key;
    Buffer buffer;
    std::uint32_t offset{0};  // target buffer offset
    // Resolved by AsuClient from buffer.handle before the entry reaches the transport.
    std::optional<std::uint32_t> mrKey;
};

// Describes a registered memory region returned by registration or reused for binding.
struct RegisteredMemory {
    MemoryRegion region;
    MRHandle handle{kInvalidMRHandle};
    std::uint32_t tokenId{0};
};

struct TaskResult {
    Status status;
    std::vector<Status> entryStatus;
    std::optional<QueryResult> queryResult;
};

using TaskCompletionCallback = std::function<void(TaskResult)>;

}  // namespace UC::ASU
