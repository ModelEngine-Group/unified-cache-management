#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/transport.h"
#include "core/transport_init_attrs.h"
#include "hixl/hixl_types.h"

namespace transport {

class HixlInstance;

struct HixlInstanceInfo {
    Endpoint endpoint;
    int32_t device_id = -1;
};

class HixlTransport final : public Transport {
public:
    HixlTransport();
    ~HixlTransport() override;

    HixlTransport(const HixlTransport&) = delete;
    HixlTransport& operator=(const HixlTransport&) = delete;

    TransportProtocol Protocol() const override;
    Status Init(const InitAttrs& attrs) override;
    Status Init(const HixlInitAttrs& attrs);
    Status Shutdown() override;
    Status RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle) override;
    Status UnregisterMemory(MemoryHandle handle) override;
    Status ExportMetadata(const ManagerID& manager_id, Metadata& out) override;
    Status ImportMetadata(const ManagerID& manager_id, const Metadata& metadata) override;
    Status Connect(const ManagerID& manager_id) override;
    Status Disconnect(const ManagerID& manager_id) override;
    Status ExecuteSync(const Operation& request) override;
    Status ExecuteAsync(const Operation& request, TransferHandle& handle,
                        TransportCallTiming* timing = nullptr) override;
    Status GetStatus(TransferHandle handle, TransferStatus& status,
                     TransportCallTiming* timing = nullptr) override;

private:
    struct Peer {
        std::vector<HixlInstanceInfo> instances;
        HixlRole role = HixlRole::Bidirectional;
        size_t local_index = SIZE_MAX;
        bool connected = false;
    };

    struct LocalMemoryRecord {
        MemoryRegion region;
        std::unordered_map<size_t, hixl::MemHandle> native_handles;
    };

    struct PendingTransfer {
        size_t instance_index = SIZE_MAX;
        hixl::TransferReq request = nullptr;
    };

    Status ValidateTransferLocked(const Operation& batch, size_t instance_index) const;
    Status BuildRouteLocked(const ManagerID& manager_id, Peer& peer);
    Status DisconnectRoute(const Peer& peer, bool ignore_failure);

    int32_t connect_timeout_ms_ = 1000;
    int32_t transfer_timeout_ms_ = 1000;
    HixlRole role_ = HixlRole::Bidirectional;
    std::vector<std::unique_ptr<HixlInstance>> instances_;
    std::unordered_map<ManagerID, Peer> peers_;
    std::unordered_map<MemoryHandle, std::unique_ptr<LocalMemoryRecord>> memories_;
    std::unordered_map<TransferHandle, PendingTransfer> pending_transfers_;
    TransferHandle next_transfer_handle_ = 1;
    mutable std::shared_mutex lifecycle_mutex_;
    mutable std::shared_mutex peers_mutex_;
    mutable std::shared_mutex memories_mutex_;
    mutable std::mutex pending_mutex_;
};

}  // namespace transport
