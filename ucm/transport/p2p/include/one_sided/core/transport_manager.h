#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "control/metadata_channel.h"
#include "core/transport.h"

namespace transport {

class TransportManager {
public:
    explicit TransportManager(ManagerID manager_id);
    ~TransportManager();

    TransportManager(const TransportManager&) = delete;
    TransportManager& operator=(const TransportManager&) = delete;

    Status Init();
    Status InstallTransport(const std::string& protocol, const InitAttrs& options);

    Status ExchangeMetadata(const ManagerID& manager_id);
    Status Shutdown();

    Status RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle);
    Status UnregisterMemory(MemoryHandle handle);

    Status Execute(const Operation& batch);

private:
    struct InstalledTransport {
        std::string protocol;
        TransportPtr transport;
    };

    struct MemoryRecord {
        MemoryRegion region;
        std::unordered_map<std::string, MemoryHandle> transport_handles;
    };

    TransportPtr CreateTransport(const std::string& protocol) const;
    Status ExportLocalMetadata(const ManagerID& manager_id, Metadata& out);
    Status ImportMetadata(const Metadata& metadata, const ManagerID& manager_id);
    Status HandleMetadataExchange(const ManagerID& manager_id, const Metadata& remote_metadata,
                                  Metadata& local_metadata);
    Endpoint LocalEndpoint() const;
    Status ParseManagerID(const ManagerID& manager_id, Endpoint& endpoint) const;

    ManagerID manager_id_;
    Endpoint local_endpoint_;
    std::shared_ptr<MetadataChannel> control_;
    mutable std::recursive_mutex peer_mutex_;
    std::unordered_map<std::string, Transport*> protocol_map_;
    std::vector<InstalledTransport> transports_;
    std::unordered_map<MemoryHandle, MemoryRecord> memories_;
    MemoryHandle next_memory_handle_ = 1;
};

}  // namespace transport
