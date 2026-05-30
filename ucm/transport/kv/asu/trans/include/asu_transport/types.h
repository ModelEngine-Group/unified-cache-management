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
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace UC::ASU {

using TaskId = std::uint64_t;
using MRHandle = std::uint64_t;
using CacheKey = std::string;
using AsuId = std::uint64_t;

constexpr TaskId kInvalidTaskId = 0;
constexpr MRHandle kInvalidMRHandle = 0;

enum class StatusCode {
    OK = 0,
    INVALID_ARGUMENT,
    NOT_INITIALIZED,
    TIMEOUT,
    SUBMIT_TIMEOUT,
    RESULT_TIMEOUT,
    NOT_FOUND,
    PARTIAL_FAILED,
    CONNECTION_ERROR,
    NO_ACTIVE_CONNECTION,
    IO_ERROR,
    BUFFER_NOT_REGISTERED,
    BUFFER_NOT_SUPPORTED,
    TASK_NOT_FOUND,
    RESOURCE_BUSY,
    UNSUPPORTED,
    IN_PROGRESS,
    INTERNAL_ERROR,
    CANCELED,
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

enum class QueryMode {
    PER_KEY = 0,
    PREFIX = 1,
};

struct QueryOptions {
    QueryMode mode{QueryMode::PER_KEY};
    std::uint64_t timeoutMs{0};
};

struct QueryResult {
    std::vector<std::uint8_t> exists;
    std::uint32_t prefixHitKeys{0};
};

enum class MemoryType {
    HOST = 0,
    HOST_PINNED = 1,
    ASCEND_DEVICE = 2,
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
};

struct RegisterResult {
    Status status;
    MRHandle handle{kInvalidMRHandle};
};

struct RegisteredMemory {  // 用于绑定已注册的内存
    MemoryRegion region;
    MRHandle handle{kInvalidMRHandle};
    std::uint32_t lkey{0};
    std::uint32_t rkey{0};
};

struct TaskResult {
    Status status;
    std::vector<Status> entryStatus;
    std::optional<QueryResult> queryResult;
};

}  // namespace UC::ASU
