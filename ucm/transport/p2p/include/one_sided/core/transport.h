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

    std::string toString() const { return host + ":" + std::to_string(port); }
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

    virtual const char* name() const = 0;
    virtual Status init(const InitAttrs& options) = 0;
    virtual Status shutdown() = 0;

    virtual Status registerMemory(const MemoryRegion& memory, MemoryHandle& handle) = 0;
    virtual Status unregisterMemory(MemoryHandle handle) = 0;
    virtual Status exportMetadata(const ManagerID& manager_id, Metadata& out) = 0;
    virtual Status importMetadata(const ManagerID& manager_id, const Metadata& metadata) = 0;
    virtual Status connectPeer(const ManagerID& manager_id) = 0;
    virtual Status execute(const Operation& request) = 0;
};

using TransportPtr = std::shared_ptr<Transport>;

}  // namespace transport
