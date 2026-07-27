#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "control/control_channel.h"
#include "control/control_protocol.h"
#include "core/transport.h"
#include "core/transport_init_attrs.h"

namespace transport {

class TransportManager {
public:
    explicit TransportManager(ManagerID manager_id);
    ~TransportManager();

    TransportManager(const TransportManager&) = delete;
    TransportManager& operator=(const TransportManager&) = delete;

    Status Init();
    Status InstallTransport(TransportProtocol protocol, const InitAttrs& options);

    Status ExchangeMetadata(const ManagerID& manager_id);
    Status Shutdown();

    Status RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle);
    Status UnregisterMemory(MemoryHandle handle);

    Status Connect(TransportProtocol protocol, const ManagerID& manager_id);
    Status Disconnect(TransportProtocol protocol, const ManagerID& manager_id);
    Status ExecuteSync(const Operation& batch);
    Status ExecuteAsync(const Operation& batch, TransferHandle& handle);
    Status GetStatus(TransferHandle handle, TransferStatus& status);

private:
    struct InstalledTransport {
        TransportProtocol protocol;
        TransportPtr transport;
    };

    struct MemoryRecord {
        MemoryRegion region;
        std::unordered_map<TransportProtocol, MemoryHandle> transport_handles;
    };

    struct TransferRecord {
        Transport* transport = nullptr;
        TransferHandle transport_handle = kInvalidTransferHandle;
    };

    TransportPtr CreateTransport(TransportProtocol protocol) const;
    Status FindTransport(Operation& batch, Transport*& transport);
    Status ExportLocalMetadata(const ManagerID& manager_id, Metadata& out);
    Status ImportMetadata(const Metadata& metadata, const ManagerID& manager_id);
    Status HandleMetadataExchange(const ManagerID& manager_id, const Metadata& remote_metadata,
                                  Metadata& local_metadata);
    Status HandleControlRequest(const Metadata& request, Metadata& response);
    Status CoordinateConnection(ControlOperation operation, TransportProtocol protocol,
                                const ManagerID& manager_id);
    Status ApplyConnection(ControlOperation operation, TransportProtocol protocol,
                           const ManagerID& manager_id);
    Endpoint LocalEndpoint() const;
    Status ParseManagerID(const ManagerID& manager_id, Endpoint& endpoint) const;

    ManagerID manager_id_;
    Endpoint local_endpoint_;
    std::shared_ptr<ControlChannel> control_;
    mutable std::recursive_mutex peer_mutex_;
    std::set<std::pair<TransportProtocol, ManagerID>> connections_;
    bool shutting_down_ = false;
    std::unordered_map<TransportProtocol, Transport*> protocol_map_;
    std::vector<InstalledTransport> transports_;
    std::unordered_map<MemoryHandle, std::unique_ptr<MemoryRecord>> memories_;
    std::mutex transfers_mutex_;
    std::unordered_map<TransferHandle, TransferRecord> transfers_;
    TransferHandle next_transfer_handle_ = 1;
};

}  // namespace transport
