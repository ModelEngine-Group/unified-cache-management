#include "core/transport_manager.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "common/binary_codec.h"
#include "control/control_channel.h"
#include "control/control_protocol.h"
#ifdef UCM_P2P_HAS_HIXL
#include "protocols/hixl/hixl_transport.h"
#endif
#include "logger/logger.h"

namespace transport {
namespace {

struct TransportMetadataRecord {
    TransportProtocol protocol;
    Metadata metadata;
};

struct PeerAdvertisement {
    std::vector<TransportMetadataRecord> records;
};

Status EncodePeerAdvertisement(const PeerAdvertisement& advertisement, Metadata& out)
{
    if (advertisement.records.size() > UINT32_MAX) { return Status::InvalidParam(); }

    out.clear();
    if (!detail::AppendU32(out, static_cast<uint32_t>(advertisement.records.size()))) {
        return Status::InvalidParam();
    }

    for (const auto& record : advertisement.records) {
        if (!detail::AppendU32(out, static_cast<uint32_t>(record.protocol)) ||
            !detail::AppendBytes(out, record.metadata)) {
            return Status::InvalidParam();
        }
    }
    return Status::OK();
}

Status DecodePeerAdvertisement(const Metadata& in, PeerAdvertisement& advertisement)
{
    size_t offset = 0;
    uint32_t count = 0;
    if (!detail::ReadU32(in, offset, count)) { return Status::InvalidParam(); }

    advertisement.records.clear();
    advertisement.records.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        TransportMetadataRecord record;
        uint32_t protocol = 0;
        if (!detail::ReadU32(in, offset, protocol) ||
            !detail::ReadBytes(in, offset, record.metadata)) {
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
    UC_DEBUG("transport manager init begin manager={}", manager_id_);
    if (ParseManagerID(manager_id_, local_endpoint_) != Status::OK()) {
        UC_ERROR("transport manager init failed: invalid manager id={}", manager_id_);
        return Status::InvalidParam();
    }
    if (control_) {
        UC_DEBUG("transport manager init skipped: already initialized manager={}", manager_id_);
        return Status::OK();
    }
    control_ = std::make_shared<ControlChannel>();
    auto status =
        control_->Init(LocalEndpoint(), [this](const Metadata& request, Metadata& response) {
            return HandleControlRequest(request, response);
        });
    if (status != Status::OK()) {
        UC_ERROR("transport manager control init failed manager={} status={}", manager_id_,
                 status.Underlying());
        control_.reset();
        return status;
    }
    UC_DEBUG("transport manager init completed manager={}", manager_id_);
    return Status::OK();
}

Status TransportManager::InstallTransport(TransportProtocol protocol, const InitAttrs& options)
{
    if (protocol_map_.find(protocol) != protocol_map_.end()) {
        UC_DEBUG("transport manager install skipped protocol={}: already installed",
                 static_cast<uint32_t>(protocol));
        return Status::OK();
    }

    auto transport = CreateTransport(protocol);
    if (!transport) {
        UC_ERROR("transport manager install failed: unsupported protocol={}",
                 static_cast<uint32_t>(protocol));
        return Status::Unsupported();
    }
    const auto status = transport->Init(options);
    if (status != Status::OK()) {
        UC_ERROR("transport manager install failed protocol={} status={}",
                 static_cast<uint32_t>(protocol), status.Underlying());
        return status;
    }

    protocol_map_[protocol] = transport.get();
    transports_.push_back(InstalledTransport{protocol, std::move(transport)});
    UC_DEBUG("transport manager installed protocol={}", static_cast<uint32_t>(protocol));
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
    UC_DEBUG("transport manager shutdown begin manager={}", manager_id_);
    Status result = Status::OK();
    std::vector<std::pair<TransportProtocol, ManagerID>> connections;
    {
        std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
        shutting_down_ = true;
        connections.assign(connections_.begin(), connections_.end());
    }
    for (const auto& connection : connections) {
        const auto status = CoordinateConnectionWithPeer(ControlOperation::Disconnect,
                                                         connection.first, connection.second);
        if (status != Status::OK() && result == Status::OK()) { result = status; }
    }

    if (control_) { control_->Close(); }

    for (auto& item : transports_) {
        const auto status = item.transport->Shutdown();
        if (status != Status::OK()) {
            UC_ERROR("transport manager transport shutdown failed protocol={} status={}",
                     static_cast<uint32_t>(item.protocol), status.Underlying());
            if (result == Status::OK()) { result = status; }
        }
    }
    memories_.clear();
    {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        transfers_.clear();
        next_transfer_handle_ = 1;
    }
    {
        std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
        connections_.clear();
    }
    protocol_map_.clear();
    transports_.clear();
    UC_DEBUG("transport manager shutdown completed manager={} status={}", manager_id_,
             result.Underlying());
    return result;
}

Status TransportManager::ExchangeMetadata(const ManagerID& manager_id)
{
    UC_DEBUG("transport manager metadata exchange begin local={} peer={}", manager_id_, manager_id);
    Endpoint endpoint;
    auto status = ParseManagerID(manager_id, endpoint);
    if (status != Status::OK()) {
        UC_ERROR("transport manager metadata exchange invalid peer={} status={}", manager_id,
                 status.Underlying());
        return status;
    }

    if (manager_id == LocalEndpoint().ToString()) {
        UC_DEBUG("transport manager metadata exchange skipped local peer={}", manager_id);
        return Status::OK();
    }

    Metadata local;
    status = ExportLocalMetadata(manager_id, local);
    if (status != Status::OK()) {
        UC_ERROR("transport manager metadata export failed peer={} status={}", manager_id,
                 status.Underlying());
        return status;
    }
    Metadata remote;
    Metadata request;
    status = EncodeControlRequest(ControlRequest{ControlOperation::ExchangeMetadata, std::nullopt,
                                                 manager_id_, std::move(local)},
                                  request);
    if (status != Status::OK()) {
        UC_ERROR("transport manager metadata request encode failed peer={} status={}", manager_id,
                 status.Underlying());
        return status;
    }
    status = control_->Request(endpoint, request, remote);
    if (status != Status::OK()) {
        UC_ERROR("transport manager metadata request failed peer={} status={}", manager_id,
                 status.Underlying());
        return status;
    }
    status = ImportMetadata(remote, manager_id);
    if (status != Status::OK()) {
        UC_ERROR("transport manager metadata import failed peer={} status={}", manager_id,
                 status.Underlying());
        return status;
    }
    UC_DEBUG("transport manager metadata exchange completed local={} peer={}", manager_id_,
             manager_id);
    return status;
}

Status TransportManager::ExportLocalMetadata(const ManagerID& manager_id, Metadata& out)
{
    if (transports_.size() > UINT32_MAX) {
        UC_ERROR("transport manager metadata export failed peer={}: transport count={}", manager_id,
                 transports_.size());
        return Status::InvalidParam();
    }

    PeerAdvertisement advertisement;
    advertisement.records.reserve(transports_.size());
    for (const auto& item : transports_) {
        Metadata metadata;
        const auto status = item.transport->ExportMetadata(manager_id, metadata);
        if (status != Status::OK()) {
            UC_ERROR("transport manager metadata export failed protocol={} peer={} status={}",
                     static_cast<uint32_t>(item.protocol), manager_id, status.Underlying());
            return status;
        }
        advertisement.records.push_back(
            TransportMetadataRecord{item.protocol, std::move(metadata)});
    }
    const auto status = EncodePeerAdvertisement(advertisement, out);
    if (status != Status::OK()) {
        UC_ERROR("transport manager advertisement encode failed peer={} status={}", manager_id,
                 status.Underlying());
    }
    return status;
}

Status TransportManager::ImportMetadata(const Metadata& metadata, const ManagerID& manager_id)
{
    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK() ||
        metadata.size() < sizeof(uint32_t)) {
        UC_ERROR("transport manager metadata import invalid peer={} bytes={}", manager_id,
                 metadata.size());
        return Status::InvalidParam();
    }

    PeerAdvertisement advertisement;
    const auto decode_status = DecodePeerAdvertisement(metadata, advertisement);
    if (decode_status != Status::OK()) {
        UC_ERROR("transport manager advertisement decode failed peer={} bytes={} status={}",
                 manager_id, metadata.size(), decode_status.Underlying());
        return decode_status;
    }

    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    for (const auto& record : advertisement.records) {
        const auto it = protocol_map_.find(record.protocol);
        if (it == protocol_map_.end()) {
            UC_DEBUG("transport manager metadata ignored unsupported protocol={} peer={}",
                     static_cast<uint32_t>(record.protocol), manager_id);
            continue;
        }

        const auto status = it->second->ImportMetadata(manager_id, record.metadata);
        if (status != Status::OK()) {
            UC_ERROR("transport manager metadata import failed protocol={} peer={} status={}",
                     static_cast<uint32_t>(record.protocol), manager_id, status.Underlying());
            return status;
        }
    }

    return Status::OK();
}

Status TransportManager::HandleMetadataExchange(const ManagerID& manager_id,
                                                const Metadata& remote_metadata,
                                                Metadata& local_metadata)
{
    UC_DEBUG("transport manager handling metadata exchange local={} peer={} request_bytes={}",
             manager_id_, manager_id, remote_metadata.size());
    const auto status = ImportMetadata(remote_metadata, manager_id);
    if (status != Status::OK()) {
        UC_ERROR("transport manager handling metadata import failed peer={} status={}", manager_id,
                 status.Underlying());
        return status;
    }
    const auto export_status = ExportLocalMetadata(manager_id, local_metadata);
    if (export_status != Status::OK()) {
        UC_ERROR("transport manager handling metadata export failed peer={} status={}", manager_id,
                 export_status.Underlying());
        return export_status;
    }
    UC_DEBUG("transport manager handled metadata exchange local={} peer={} response_bytes={}",
             manager_id_, manager_id, local_metadata.size());
    return Status::OK();
}

Status TransportManager::HandleControlRequest(const Metadata& request, Metadata& response)
{
    ControlRequest control_request{};
    auto status = DecodeControlRequest(request, control_request);
    if (status != Status::OK()) {
        UC_ERROR("transport manager control request decode failed local={} bytes={} status={}",
                 manager_id_, request.size(), status.Underlying());
        return status;
    }

    UC_DEBUG(
        "transport manager control request decoded local={} operation={} protocol={} peer={}",
        manager_id_, ControlOperationName(control_request.operation),
        control_request.protocol.has_value() ? static_cast<int32_t>(*control_request.protocol) : -1,
        control_request.manager_id);

    if (control_request.operation == ControlOperation::ExchangeMetadata) {
        return HandleMetadataExchange(control_request.manager_id, control_request.payload,
                                      response);
    }
    if (!control_request.protocol.has_value()) {
        UC_ERROR("transport manager control request missing protocol local={} operation={} peer={}",
                 manager_id_, ControlOperationName(control_request.operation),
                 control_request.manager_id);
        return Status::InvalidParam();
    }

    const auto apply_status = ApplyConnectionLocally(
        control_request.operation, *control_request.protocol, control_request.manager_id);
    if (apply_status != Status::OK()) {
        UC_ERROR(
            "transport manager control request apply failed operation={} protocol={} peer={} "
            "status={}",
            ControlOperationName(control_request.operation),
            static_cast<uint32_t>(*control_request.protocol), control_request.manager_id,
            apply_status.Underlying());
        return apply_status;
    }
    UC_DEBUG("transport manager control request applied operation={} protocol={} peer={}",
             ControlOperationName(control_request.operation),
             static_cast<uint32_t>(*control_request.protocol), control_request.manager_id);
    return Status::OK();
}

Status TransportManager::RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle)
{
    handle = kInvalidMemoryHandle;
    if (memory.addr == nullptr || memory.length == 0) {
        UC_ERROR("transport manager register memory invalid addr={} length={}", memory.addr,
                 memory.length);
        return Status::InvalidParam();
    }
    const auto address = detail::PtrToU64(memory.addr);
    if (memory.length > std::numeric_limits<uint64_t>::max() - address) {
        UC_ERROR("transport manager register memory address overflow addr=0x{:x} length={}",
                 address, memory.length);
        return Status::InvalidParam();
    }
    if (transports_.empty()) {
        UC_ERROR("transport manager register memory failed: no installed transports");
        return Status::Error();
    }

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
    UC_DEBUG("transport manager registered memory handle={} addr=0x{:x} length={}", handle, address,
             memory.length);
    return Status::OK();
}

Status TransportManager::UnregisterMemory(MemoryHandle handle)
{
    if (handle == kInvalidMemoryHandle) {
        UC_ERROR("transport manager unregister memory invalid handle={}", handle);
        return Status::InvalidParam();
    }

    const auto it = memories_.find(handle);
    if (it == memories_.end()) {
        UC_ERROR("transport manager unregister memory unknown handle={}", handle);
        return Status::Error();
    }

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
    UC_DEBUG("transport manager unregistered memory handle={}", handle);
    return Status::OK();
}

Status TransportManager::FindTransport(Operation& batch, Transport*& transport)
{
    if (batch.target_manager.empty()) {
        UC_ERROR("transport manager select transport failed: target manager is empty");
        return Status::InvalidParam();
    }
    Endpoint endpoint;
    if (ParseManagerID(batch.target_manager, endpoint) != Status::OK()) {
        UC_ERROR("transport manager select transport failed: invalid peer={}",
                 batch.target_manager);
        return Status::InvalidParam();
    }

    TransportProtocol protocol = TransportProtocol::Hixl;
    if (!TransportForDirect(batch.direct, protocol)) {
        UC_ERROR("transport manager select transport failed: unsupported direction={} peer={}",
                 static_cast<uint32_t>(batch.direct), batch.target_manager);
        return Status::Error();
    }
    const auto transport_it = protocol_map_.find(protocol);
    if (transport_it == protocol_map_.end()) {
        UC_ERROR("transport manager select transport failed: protocol={} not installed peer={}",
                 static_cast<uint32_t>(protocol), batch.target_manager);
        return Status::Error();
    }
    transport = transport_it->second;
    return Status::OK();
}

Status TransportManager::Connect(TransportProtocol protocol, const ManagerID& manager_id)
{
    return CoordinateConnectionWithPeer(ControlOperation::Connect, protocol, manager_id);
}

Status TransportManager::Disconnect(TransportProtocol protocol, const ManagerID& manager_id)
{
    return CoordinateConnectionWithPeer(ControlOperation::Disconnect, protocol, manager_id);
}

Status TransportManager::ApplyConnectionLocally(ControlOperation operation,
                                                TransportProtocol protocol,
                                                const ManagerID& manager_id)
{
    UC_DEBUG("transport manager local {} begin protocol={} peer={}",
             ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id);
    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    if (shutting_down_ && operation == ControlOperation::Connect) { return Status::Error(); }
    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK()) {
        UC_ERROR("transport manager local {} invalid peer={} protocol={}",
                 ControlOperationName(operation), manager_id, static_cast<uint32_t>(protocol));
        return Status::InvalidParam();
    }
    const auto it = protocol_map_.find(protocol);
    if (it == protocol_map_.end()) {
        UC_ERROR("transport manager local {} unavailable protocol={} peer={}",
                 ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id);
        return Status::InvalidParam();
    }
    const auto status = operation == ControlOperation::Connect ? it->second->Connect(manager_id)
                                                               : it->second->Disconnect(manager_id);
    if (status != Status::OK()) {
        UC_ERROR("transport manager local {} failed protocol={} peer={} status={}",
                 ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id,
                 status.Underlying());
        return status;
    }

    const auto connection = std::make_pair(protocol, manager_id);
    if (operation == ControlOperation::Connect) {
        connections_.insert(connection);
    } else {
        connections_.erase(connection);
    }
    UC_DEBUG("transport manager local {} completed protocol={} peer={}",
             ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id);
    return Status::OK();
}

Status TransportManager::CoordinateConnectionWithPeer(ControlOperation operation,
                                                      TransportProtocol protocol,
                                                      const ManagerID& manager_id)
{
    UC_DEBUG("transport manager coordinate {} begin protocol={} peer={}",
             ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id);
    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK()) {
        UC_ERROR("transport manager coordinate {} invalid peer={} protocol={}",
                 ControlOperationName(operation), manager_id, static_cast<uint32_t>(protocol));
        return Status::InvalidParam();
    }
    if (!control_) {
        UC_ERROR(
            "transport manager coordinate {} failed: control channel is unavailable peer={} "
            "protocol={}",
            ControlOperationName(operation), manager_id, static_cast<uint32_t>(protocol));
        return Status::InvalidParam();
    }
    if (protocol_map_.find(protocol) == protocol_map_.end()) {
        UC_ERROR("transport manager coordinate {} unavailable protocol={} peer={}",
                 ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id);
        return Status::InvalidParam();
    }

    Metadata request;
    auto status =
        EncodeControlRequest(ControlRequest{operation, protocol, manager_id_, {}}, request);
    if (status != Status::OK()) {
        UC_ERROR(
            "transport manager coordinate {} request encode failed protocol={} peer={} status={}",
            ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id,
            status.Underlying());
        return status;
    }

    const auto local_status = ApplyConnectionLocally(operation, protocol, manager_id);
    if (operation == ControlOperation::Connect && local_status != Status::OK()) {
        UC_ERROR("transport manager local connect failed protocol={} peer={} status={}",
                 static_cast<uint32_t>(protocol), manager_id, local_status.Underlying());
        return local_status;
    }

    Metadata ack;
    UC_DEBUG(
        "transport manager coordinate {} requesting peer ACK protocol={} peer={} local_status={}",
        ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id,
        local_status.Underlying());
    const auto remote_status = control_->Request(endpoint, request, ack);
    if (local_status != Status::OK() || remote_status != Status::OK()) {
        UC_ERROR("transport manager coordinated {} failed protocol={} peer={} local={} remote={}",
                 operation == ControlOperation::Connect ? "connect" : "disconnect",
                 static_cast<uint32_t>(protocol), manager_id, local_status.Underlying(),
                 remote_status.Underlying());
        if (operation == ControlOperation::Connect && remote_status != Status::OK()) {
            const auto rollback_status =
                ApplyConnectionLocally(ControlOperation::Disconnect, protocol, manager_id);
            UC_WARN("transport manager rolled back local connect protocol={} peer={} status={}",
                    static_cast<uint32_t>(protocol), manager_id, rollback_status.Underlying());
        }
        return local_status != Status::OK() ? local_status : remote_status;
    }
    UC_DEBUG("transport manager coordinated {} success protocol={} peer={} ack_bytes={}",
             ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id,
             ack.size());
    return Status::OK();
}

Status TransportManager::ExecuteSync(const Operation& batch)
{
    UC_DEBUG("transport manager sync transfer begin peer={} segments={}", batch.target_manager,
             batch.ops.size());
    Transport* transport = nullptr;
    auto request = batch;
    auto status = FindTransport(request, transport);
    if (status != Status::OK()) {
        UC_ERROR("transport manager sync transfer selection failed peer={} status={}",
                 batch.target_manager, status.Underlying());
        return status;
    }
    status = transport->ExecuteSync(request);
    if (status != Status::OK()) {
        UC_ERROR("transport manager sync transfer failed peer={} segments={} status={}",
                 batch.target_manager, batch.ops.size(), status.Underlying());
        return status;
    }
    UC_DEBUG("transport manager sync transfer completed peer={} segments={}", batch.target_manager,
             batch.ops.size());
    return Status::OK();
}

Status TransportManager::ExecuteAsync(const Operation& batch, TransferHandle& handle)
{
    handle = kInvalidTransferHandle;
    Transport* transport = nullptr;
    auto request = batch;
    auto status = FindTransport(request, transport);
    if (status != Status::OK()) {
        UC_ERROR("transport manager async transfer selection failed peer={} status={}",
                 batch.target_manager, status.Underlying());
        return status;
    }

    TransferHandle transport_handle = kInvalidTransferHandle;
    status = transport->ExecuteAsync(request, transport_handle);
    if (status != Status::OK() || transport_handle == kInvalidTransferHandle) {
        UC_ERROR(
            "transport manager async transfer submit failed peer={} segments={} status={} "
            "transport_handle={}",
            batch.target_manager, batch.ops.size(), status.Underlying(), transport_handle);
        return status == Status::OK() ? Status::Error() : status;
    }

    {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        handle = next_transfer_handle_++;
        if (handle == kInvalidTransferHandle) { handle = next_transfer_handle_++; }
        transfers_.emplace(handle, TransferRecord{transport, transport_handle});
    }
    UC_DEBUG(
        "transport manager async transfer submitted peer={} segments={} handle={} "
        "transport_handle={}",
        batch.target_manager, batch.ops.size(), handle, transport_handle);
    return Status::OK();
}

Status TransportManager::GetStatus(TransferHandle handle, TransferStatus& transfer_status)
{
    if (handle == kInvalidTransferHandle) {
        UC_ERROR("transport manager transfer status invalid handle={}", handle);
        return Status::InvalidParam();
    }
    TransferRecord record;
    {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        const auto it = transfers_.find(handle);
        if (it == transfers_.end() || it->second.transport == nullptr) {
            UC_ERROR("transport manager transfer status unknown handle={}", handle);
            return Status::Error();
        }
        record = it->second;
    }
    const auto status = record.transport->GetStatus(record.transport_handle, transfer_status);
    if (status != Status::OK()) {
        UC_ERROR(
            "transport manager transfer status query failed handle={} transport_handle={} "
            "status={}",
            handle, record.transport_handle, status.Underlying());
    } else if (transfer_status != TransferStatus::Waiting) {
        UC_DEBUG("transport manager transfer completed handle={} transport_handle={} status={}",
                 handle, record.transport_handle, static_cast<uint32_t>(transfer_status));
    }
    if (status != Status::OK() || transfer_status != TransferStatus::Waiting) {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        transfers_.erase(handle);
    }
    return status;
}

Endpoint TransportManager::LocalEndpoint() const { return local_endpoint_; }

Status TransportManager::ParseManagerID(const ManagerID& manager_id, Endpoint& endpoint) const
{
    const auto separator = manager_id.rfind(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= manager_id.size()) {
        UC_ERROR("transport manager invalid manager id={}", manager_id);
        return Status::InvalidParam();
    }

    const auto host = manager_id.substr(0, separator);
    const auto port_text = manager_id.substr(separator + 1);
    try {
        size_t parsed = 0;
        const auto port = std::stoul(port_text, &parsed, 10);
        if (parsed != port_text.size() || port == 0 ||
            port > std::numeric_limits<uint16_t>::max()) {
            UC_ERROR("transport manager invalid manager port manager={} port={}", manager_id,
                     port_text);
            return Status::InvalidParam();
        }
        endpoint = Endpoint{host, static_cast<uint16_t>(port)};
        return Status::OK();
    } catch (const std::exception& error) {
        UC_ERROR("transport manager failed to parse manager id={} error={}", manager_id,
                 error.what());
        return Status::InvalidParam();
    }
}

}  // namespace transport
