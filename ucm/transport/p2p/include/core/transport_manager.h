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
    Status ExecuteAsync(const Operation& batch, TransferHandle& handle,
                        TransportCallTiming* timing = nullptr);
    Status GetStatus(TransferHandle handle, TransferStatus& status,
                     TransportCallTiming* timing = nullptr);

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
        ManagerID target_manager;
        Opcode opcode{Opcode::Read};
        OperationDirect direct{OperationDirect::RemoteDeviceHost};
        std::size_t segment_count{0};
        std::uint64_t bytes{0};
        std::uint64_t submitted_us{0};
        std::uint64_t submitted_ts_us{0};
        std::uint64_t submit_us{0};
    };

    TransportPtr CreateTransport(TransportProtocol protocol) const;
    Status FindTransport(Operation& batch, Transport*& transport);
    Status ExportLocalMetadata(const ManagerID& manager_id, Metadata& out);
    Status ImportMetadata(const Metadata& metadata, const ManagerID& manager_id);
    Status HandleMetadataExchange(const ManagerID& manager_id, const Metadata& remote_metadata,
                                  Metadata& local_metadata);
    Status HandleControlRequest(const Metadata& request, Metadata& response);
    Status CoordinateConnectionWithPeer(ControlOperation operation, TransportProtocol protocol,
                                        const ManagerID& manager_id);
    Status ApplyConnectionLocally(ControlOperation operation, TransportProtocol protocol,
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
