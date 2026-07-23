#include "core/transport_manager.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "common/metadata_codec.h"
#include "control/metadata_channel.h"
#ifdef UCM_P2P_HAS_HIXL
#include "hixl/hixl_transport.h"
#endif
#include "logger/logger.h"

namespace transport {
namespace {

struct TransportMetadataRecord {
    TransportProtocol protocol;
    Metadata metadata;
};

struct PeerAdvertisement {
    Endpoint endpoint;
    std::vector<TransportMetadataRecord> records;
};

Status EncodePeerAdvertisement(const PeerAdvertisement& advertisement, Metadata& out)
{
    if (advertisement.records.size() > UINT32_MAX) { return Status::InvalidParam(); }

    out.clear();
    if (!detail::AppendString(out, advertisement.endpoint.host) ||
        !detail::AppendU32(out, static_cast<uint32_t>(advertisement.endpoint.port)) ||
        !detail::AppendU32(out, static_cast<uint32_t>(advertisement.records.size()))) {
        return Status::InvalidParam();
    }

    for (const auto& record : advertisement.records) {
        if (!detail::AppendU32(out, static_cast<uint32_t>(record.protocol)) ||
            !detail::AppendMetadata(out, record.metadata)) {
            return Status::InvalidParam();
        }
    }
    return Status::OK();
}

Status DecodePeerAdvertisement(const Metadata& in, PeerAdvertisement& advertisement)
{
    size_t offset = 0;
    uint32_t remote_port = 0;
    uint32_t count = 0;
    if (!detail::ReadString(in, offset, advertisement.endpoint.host) ||
        !detail::ReadU32(in, offset, remote_port) || !detail::ReadU32(in, offset, count) ||
        remote_port > UINT16_MAX) {
        return Status::InvalidParam();
    }
    advertisement.endpoint.port = static_cast<uint16_t>(remote_port);

    advertisement.records.clear();
    advertisement.records.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        TransportMetadataRecord record;
        uint32_t protocol = 0;
        if (!detail::ReadU32(in, offset, protocol) ||
            !detail::ReadMetadata(in, offset, record.metadata) ||
            protocol != static_cast<uint32_t>(TransportProtocol::Hixl)) {
            return Status::InvalidParam();
        }
        record.protocol = static_cast<TransportProtocol>(protocol);
        advertisement.records.push_back(std::move(record));
    }

    return offset == in.size() ? Status::OK() : Status::InvalidParam();
}

bool TransportForDirect(OperationDirect direct, TransportProtocol& protocol)
{
    if (direct != OperationDirect::RemoteDeviceHost) { return false; }
    protocol = TransportProtocol::Hixl;
    return true;
}

}  // namespace

TransportManager::TransportManager(ManagerID manager_id) : manager_id_(std::move(manager_id)) {}

TransportManager::~TransportManager()
{
    if (Shutdown() != Status::OK()) {}
}

Status TransportManager::Init()
{
    if (ParseManagerID(manager_id_, local_endpoint_) != Status::OK()) {
        return Status::InvalidParam();
    }
    if (control_) { return Status::OK(); }
    control_ = std::make_shared<MetadataChannel>();
    auto status = control_->Init(
        LocalEndpoint(), [this](const Metadata& remote_metadata, Metadata& local_metadata) {
            return HandleMetadataExchange(ManagerID{}, remote_metadata, local_metadata);
        });
    if (status != Status::OK()) {
        control_.reset();
        return status;
    }
    return Status::OK();
}

Status TransportManager::InstallTransport(TransportProtocol protocol, const InitAttrs& options)
{
    if (protocol_map_.find(protocol) != protocol_map_.end()) { return Status::OK(); }

    auto transport = CreateTransport(protocol);
    if (!transport) { return Status::Unsupported(); }
    const auto status = transport->Init(options);
    if (status != Status::OK()) { return status; }

    protocol_map_[protocol] = transport.get();
    transports_.push_back(InstalledTransport{protocol, std::move(transport)});
    return Status::OK();
}

TransportPtr TransportManager::CreateTransport(TransportProtocol protocol) const
{
#ifdef UCM_P2P_HAS_HIXL
    if (protocol == TransportProtocol::Hixl) { return std::make_shared<HixlTransport>(); }
#else
    (void)protocol;
#endif
    return nullptr;
}

Status TransportManager::Shutdown()
{
    if (control_) { control_->Close(); }

    Status result = Status::OK();
    for (auto& item : transports_) {
        const auto status = item.transport->Shutdown();
        if (status != Status::OK() && result == Status::OK()) { result = status; }
    }
    memories_.clear();
    transfers_.clear();
    next_transfer_handle_ = 1;
    protocol_map_.clear();
    transports_.clear();
    return result;
}

Status TransportManager::ExchangeMetadata(const ManagerID& manager_id)
{
    Endpoint endpoint;
    auto status = ParseManagerID(manager_id, endpoint);
    if (status != Status::OK()) { return status; }

    if (manager_id == LocalEndpoint().ToString()) { return Status::OK(); }

    Metadata local;
    status = ExportLocalMetadata(manager_id, local);
    if (status != Status::OK()) { return status; }
    Metadata remote;
    status = control_->ExchangeMetadata(endpoint, local, remote);
    if (status == Status::OK()) { status = ImportMetadata(remote, manager_id); }
    return status;
}

Status TransportManager::ExportLocalMetadata(const ManagerID& manager_id, Metadata& out)
{
    if (transports_.size() > UINT32_MAX) { return Status::InvalidParam(); }

    PeerAdvertisement advertisement;
    advertisement.endpoint = LocalEndpoint();
    advertisement.records.reserve(transports_.size());
    for (const auto& item : transports_) {
        Metadata metadata;
        const auto status = item.transport->ExportMetadata(manager_id, metadata);
        if (status != Status::OK()) { return status; }
        advertisement.records.push_back(
            TransportMetadataRecord{item.protocol, std::move(metadata)});
    }
    return EncodePeerAdvertisement(advertisement, out);
}

Status TransportManager::ImportMetadata(const Metadata& metadata, const ManagerID& manager_id)
{
    if (metadata.size() < sizeof(uint32_t)) { return Status::InvalidParam(); }

    PeerAdvertisement advertisement;
    const auto decode_status = DecodePeerAdvertisement(metadata, advertisement);
    if (decode_status != Status::OK()) { return decode_status; }

    const auto remote_manager_id = advertisement.endpoint.ToString();
    if (!manager_id.empty() && manager_id != remote_manager_id) { return Status::InvalidParam(); }

    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    for (const auto& record : advertisement.records) {
        const auto it = protocol_map_.find(record.protocol);
        if (it == protocol_map_.end()) { continue; }

        const auto status = it->second->ImportMetadata(remote_manager_id, record.metadata);
        if (status != Status::OK()) { return status; }
    }

    return Status::OK();
}

Status TransportManager::HandleMetadataExchange(const ManagerID& manager_id,
                                                const Metadata& remote_metadata,
                                                Metadata& local_metadata)
{
    PeerAdvertisement advertisement;
    const auto decode_status = DecodePeerAdvertisement(remote_metadata, advertisement);
    if (decode_status != Status::OK()) { return decode_status; }

    const auto remote_manager_id = advertisement.endpoint.ToString();
    const auto& expected_manager_id = manager_id.empty() ? remote_manager_id : manager_id;
    const auto status = ImportMetadata(remote_metadata, expected_manager_id);
    if (status != Status::OK()) { return status; }
    return ExportLocalMetadata(remote_manager_id, local_metadata);
}

Status TransportManager::RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle)
{
    handle = kInvalidMemoryHandle;
    if (memory.addr == nullptr || memory.length == 0) { return Status::InvalidParam(); }
    const auto address = detail::PtrToU64(memory.addr);
    if (memory.length > std::numeric_limits<uint64_t>::max() - address) {
        return Status::InvalidParam();
    }
    if (transports_.empty()) { return Status::Error(); }

    auto record = std::make_unique<MemoryRecord>();
    record->region = memory;
    for (const auto& item : transports_) {
        MemoryHandle transport_handle = kInvalidMemoryHandle;
        auto status = item.transport->RegisterMemory(memory, transport_handle);
        if (status == Status::OK() && transport_handle == kInvalidMemoryHandle) {
            status = Status::Error();
        }
        if (status != Status::OK()) {
            UC_ERROR(
                "transport manager register memory failed protocol={} status={} handle={} "
                "addr=0x{:x} length={}",
                static_cast<int>(item.protocol), status.Underlying(), transport_handle,
                detail::PtrToU64(memory.addr), memory.length);
            continue;
        }
        record->transport_handles.emplace(item.protocol, transport_handle);
    }
    if (record->transport_handles.empty()) {
        UC_ERROR(
            "transport manager register memory failed: no transport accepted addr=0x{:x} "
            "length={}",
            detail::PtrToU64(memory.addr), memory.length);
        return Status::Error();
    }

    handle = reinterpret_cast<MemoryHandle>(record.get());
    memories_.emplace(handle, std::move(record));
    return Status::OK();
}

Status TransportManager::UnregisterMemory(MemoryHandle handle)
{
    if (handle == kInvalidMemoryHandle) { return Status::InvalidParam(); }

    const auto it = memories_.find(handle);
    if (it == memories_.end()) { return Status::Error(); }

    for (const auto& item : it->second->transport_handles) {
        const auto transport_it = protocol_map_.find(item.first);
        if (transport_it == protocol_map_.end()) {
            UC_ERROR("transport manager unregister memory failed protocol={} handle={}",
                     static_cast<int>(item.first), item.second);
            return Status::Error();
        }
        const auto status = transport_it->second->UnregisterMemory(item.second);
        if (status != Status::OK()) {
            UC_ERROR("transport manager unregister memory failed protocol={} status={} handle={}",
                     static_cast<int>(item.first), status.Underlying(), item.second);
            return Status::Error();
        }
    }
    memories_.erase(it);
    return Status::OK();
}

Status TransportManager::FindTransport(Operation& batch, Transport*& transport)
{
    if (batch.target_manager.empty()) { return Status::InvalidParam(); }
    Endpoint endpoint;
    if (ParseManagerID(batch.target_manager, endpoint) != Status::OK()) {
        return Status::InvalidParam();
    }

    TransportProtocol protocol = TransportProtocol::Hixl;
    if (!TransportForDirect(batch.direct, protocol)) { return Status::Error(); }
    const auto transport_it = protocol_map_.find(protocol);
    if (transport_it == protocol_map_.end()) { return Status::Error(); }
    transport = transport_it->second;
    return Status::OK();
}

Status TransportManager::Connect(TransportProtocol protocol, const ManagerID& manager_id)
{
    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK()) { return Status::InvalidParam(); }
    const auto it = protocol_map_.find(protocol);
    if (it == protocol_map_.end()) { return Status::InvalidParam(); }
    return it->second->Connect(manager_id);
}

Status TransportManager::Disconnect(TransportProtocol protocol, const ManagerID& manager_id)
{
    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK()) { return Status::InvalidParam(); }
    const auto it = protocol_map_.find(protocol);
    if (it == protocol_map_.end()) { return Status::InvalidParam(); }
    return it->second->Disconnect(manager_id);
}

Status TransportManager::ExecuteSync(const Operation& batch)
{
    Transport* transport = nullptr;
    auto request = batch;
    auto status = FindTransport(request, transport);
    if (status != Status::OK()) { return status; }
    return transport->ExecuteSync(request);
}

Status TransportManager::ExecuteAsync(const Operation& batch, TransferHandle& handle)
{
    handle = kInvalidTransferHandle;
    Transport* transport = nullptr;
    auto request = batch;
    auto status = FindTransport(request, transport);
    if (status != Status::OK()) { return status; }

    TransferHandle transport_handle = kInvalidTransferHandle;
    status = transport->ExecuteAsync(request, transport_handle);
    if (status != Status::OK() || transport_handle == kInvalidTransferHandle) {
        return status == Status::OK() ? Status::Error() : status;
    }

    handle = next_transfer_handle_++;
    if (handle == kInvalidTransferHandle) { handle = next_transfer_handle_++; }
    transfers_.emplace(handle, TransferRecord{transport, transport_handle});
    return Status::OK();
}

Status TransportManager::GetStatus(TransferHandle handle, TransferStatus& transfer_status)
{
    if (handle == kInvalidTransferHandle) { return Status::InvalidParam(); }
    const auto it = transfers_.find(handle);
    if (it == transfers_.end() || it->second.transport == nullptr) { return Status::Error(); }
    const auto status =
        it->second.transport->GetStatus(it->second.transport_handle, transfer_status);
    if (status != Status::OK() || transfer_status != TransferStatus::Waiting) {
        transfers_.erase(it);
    }
    return status;
}

Endpoint TransportManager::LocalEndpoint() const { return local_endpoint_; }

Status TransportManager::ParseManagerID(const ManagerID& manager_id, Endpoint& endpoint) const
{
    const auto separator = manager_id.rfind(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= manager_id.size()) {
        return Status::InvalidParam();
    }

    const auto host = manager_id.substr(0, separator);
    const auto port_text = manager_id.substr(separator + 1);
    try {
        size_t parsed = 0;
        const auto port = std::stoul(port_text, &parsed, 10);
        if (parsed != port_text.size() || port == 0 ||
            port > std::numeric_limits<uint16_t>::max()) {
            return Status::InvalidParam();
        }
        endpoint = Endpoint{host, static_cast<uint16_t>(port)};
        return Status::OK();
    } catch (const std::exception&) {
        return Status::InvalidParam();
    }
}

}  // namespace transport
