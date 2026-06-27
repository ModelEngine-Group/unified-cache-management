#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace transport {

using ManagerID = std::string;
using MemoryHandle = uint64_t;
using Metadata = std::vector<uint8_t>;

constexpr MemoryHandle kInvalidMemoryHandle = 0;

struct Endpoint {
    std::string host = "127.0.0.1";
    uint16_t port = 0;

    std::string ToString() const { return host + ":" + std::to_string(port); }
};

enum class Opcode {
    Read,
    Write,
};

enum class Status {
    Ok,
    InvalidArgument,
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
    int device_id = -1;
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

    virtual const char* Name() const = 0;
    virtual Status Init(const InitAttrs& options) = 0;
    virtual Status Shutdown() = 0;

    virtual Status RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle) = 0;
    virtual Status UnregisterMemory(MemoryHandle handle) = 0;
    virtual Status ExportMetadata(const ManagerID& manager_id, Metadata& out) = 0;
    virtual Status ImportMetadata(const ManagerID& manager_id, const Metadata& metadata) = 0;
    virtual Status Execute(const Operation& request) = 0;
};

using TransportPtr = std::shared_ptr<Transport>;

}  // namespace transport
