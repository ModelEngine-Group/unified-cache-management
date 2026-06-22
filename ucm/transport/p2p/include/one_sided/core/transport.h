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

    std::string toString() const {
        return host + ":" + std::to_string(port);
    }
};

enum class Opcode {
    Read,
    Write,
};

enum class Status {
    Ok,
    InvalidArgument,
    NotFound,
    AlreadyExists,
    Failed,
};

enum class MemoryType {
    Host,
    Device,
};

struct MemoryRegion {
    void* addr = nullptr;
    uint64_t length = 0;
    MemoryType type = MemoryType::Host;
    int device_id = -1;
};

struct InitAttrs {
    virtual ~InitAttrs();
};

struct TransferOp {
    void* local_addr = nullptr;
    uint64_t remote_addr = 0;
    uint64_t length = 0;
};

struct Transfer {
    Opcode opcode = Opcode::Read;
    ManagerID target_manager;
    std::vector<TransferOp> ops;
};

class Transport {
   public:
    virtual ~Transport();

    virtual const char* name() const = 0;
    virtual Status init(const InitAttrs& options);
    virtual Status shutdown();

    virtual Status registerMemory(const MemoryRegion& memory);
    virtual Status unregisterMemory(const MemoryRegion& memory);
    virtual Status exportMetadata(const ManagerID& manager_id, Metadata& out);
    virtual Status importMetadata(const ManagerID& manager_id, const Metadata& metadata);
    virtual Status connectPeer(const ManagerID& manager_id);
    virtual Status submitTransfer(const Transfer& request);
};

using TransportPtr = std::shared_ptr<Transport>;

}  // namespace transport
