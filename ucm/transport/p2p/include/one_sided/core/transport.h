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
