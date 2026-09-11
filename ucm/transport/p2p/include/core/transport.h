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
#include <memory>
#include <string>
#include <vector>
#include "status/status.h"

namespace transport {

using Status = UC::Status;
using ManagerID = std::string;
using MemoryHandle = uint64_t;
using TransferHandle = uint64_t;
// Opaque transport-specific bytes exchanged between peers for route, endpoint,
// and registered-memory discovery. The manager and control channel must not
// interpret the contents.
using Metadata = std::vector<uint8_t>;

constexpr MemoryHandle kInvalidMemoryHandle = 0;
constexpr TransferHandle kInvalidTransferHandle = 0;

struct Endpoint {
    std::string host = "127.0.0.1";
    uint16_t port = 0;

    std::string ToString() const { return port == 0 ? host : host + ":" + std::to_string(port); }
};

enum class Opcode {
    Read,
    Write,
};

enum class TransportProtocol : uint32_t {
    Hixl = 0,
};

enum class TransferStatus {
    Waiting,
    Completed,
    Failed,
};

enum class MemoryType {
    Host,
    Device,
};

enum class OperationDirect {
    LocalDeviceDevice,  // Same local device only.
    LocalDeviceHost,
    RemoteDeviceHost,
};

struct MemoryRegion {
    void* addr = nullptr;
    uint64_t length = 0;
    MemoryType type = MemoryType::Host;
    int32_t device_id = -1;
};

struct InitAttrs {
    virtual ~InitAttrs() = default;
};

struct Segment {
    void* local_addr = nullptr;
    uint64_t remote_addr = 0;
    uint64_t length = 0;
};

struct Operation {
    Opcode opcode = Opcode::Read;
    OperationDirect direct = OperationDirect::RemoteDeviceHost;
    ManagerID target_manager;
    std::vector<Segment> ops;
};

class Transport {
public:
    virtual ~Transport() = default;

    virtual TransportProtocol Protocol() const = 0;
    virtual Status Init(const InitAttrs& options) = 0;
    virtual Status Shutdown() = 0;

    virtual Status RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle) = 0;
    virtual Status UnregisterMemory(MemoryHandle handle) = 0;
    virtual Status ExportMetadata(const ManagerID& manager_id, Metadata& out) = 0;
    virtual Status ImportMetadata(const ManagerID& manager_id, const Metadata& metadata) = 0;
    virtual Status Connect(const ManagerID& manager_id) = 0;
    virtual Status Disconnect(const ManagerID& manager_id) = 0;
    virtual Status ExecuteSync(const Operation& request) = 0;
    virtual Status ExecuteAsync(const Operation& request, TransferHandle& handle) = 0;
    virtual Status GetStatus(TransferHandle handle, TransferStatus& status) = 0;
};

using TransportPtr = std::shared_ptr<Transport>;

}  // namespace transport
