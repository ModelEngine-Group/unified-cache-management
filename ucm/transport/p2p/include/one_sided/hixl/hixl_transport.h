#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include "acl/acl.h"
#include "core/transport.h"
#include "core/transport_init_attrs.h"
#include "hixl/hixl.h"

namespace transport {

struct HixlTransportMetadata {
    std::string local_engine;
};

class HixlTransport final : public Transport {
public:
    HixlTransport();
    ~HixlTransport() override;

    HixlTransport(const HixlTransport&) = delete;
    HixlTransport& operator=(const HixlTransport&) = delete;

    const char* Name() const override;
    Status Init(const InitAttrs& options) override;
    Status Init(const HixlInitAttrs& options);
    Status Shutdown() override;
    Status RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle) override;
    Status UnregisterMemory(MemoryHandle handle) override;
    Status ExportMetadata(const ManagerID& manager_id, Metadata& out) override;
    Status ImportMetadata(const ManagerID& manager_id, const Metadata& metadata) override;
    Status Execute(const Operation& request) override;

private:
    struct Peer {
        std::string remote_engine;
        Metadata metadata;
        bool connected = false;
    };

    struct LocalMemoryRecord {
        MemoryRegion region;
        hixl::MemHandle native_handle = nullptr;
    };

    bool ValidateMemory(uint64_t address, uint64_t length) const;
    Status ConnectPeer(const ManagerID& manager_id);

    std::string local_engine_;
    std::map<std::string, std::string> options_;
    int32_t connect_timeout_ms_ = 1000;
    int32_t transfer_timeout_ms_ = 1000;
    std::unordered_map<ManagerID, Peer> peers_;
    hixl::Hixl hixl_;
    std::unordered_map<uint64_t, LocalMemoryRecord> memories_;
    mutable std::recursive_mutex mutex_;
};

}  // namespace transport
