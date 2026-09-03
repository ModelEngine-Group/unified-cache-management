/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "core/transport_manager.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "common/binary_codec.h"
#include "common/status_utils.h"
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

TransportManager::~TransportManager() { (void)Shutdown(); }

Status TransportManager::Init()
{
    UC_DEBUG("transport manager init begin manager={}", manager_id_);
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id_, local_endpoint_),
                        "transport manager init failed manager={}", manager_id_);
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
        control_.reset();
        P2P_RETURN_IF_ERROR(status, "transport manager control init failed manager={}",
                            manager_id_);
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
    P2P_RETURN_IF_TRUE(!transport, Status::Unsupported(),
                       "transport manager install failed: unsupported protocol={}",
                       static_cast<uint32_t>(protocol));
    P2P_RETURN_IF_ERROR(transport->Init(options), "transport manager install failed protocol={}",
                        static_cast<uint32_t>(protocol));

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
        if (status != Status::OK()) {
            P2P_LOG_IF_ERROR(status, "transport manager disconnect failed protocol={} peer={}",
                             static_cast<uint32_t>(connection.first), connection.second);
            if (result == Status::OK()) { result = status; }
        }
    }

    if (control_) { control_->Close(); }

    for (auto& item : transports_) {
        const auto status = item.transport->Shutdown();
        if (status != Status::OK()) {
            P2P_LOG_IF_ERROR(status, "transport manager transport shutdown failed protocol={}",
                             static_cast<uint32_t>(item.protocol));
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
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id, endpoint),
                        "transport manager metadata exchange invalid peer={}", manager_id);

    if (manager_id == LocalEndpoint().ToString()) {
        UC_DEBUG("transport manager metadata exchange skipped local peer={}", manager_id);
        return Status::OK();
    }

    Metadata local;
    P2P_RETURN_IF_ERROR(ExportLocalMetadata(manager_id, local),
                        "transport manager metadata export failed peer={}", manager_id);
    Metadata remote;
    Metadata request;
    P2P_RETURN_IF_ERROR(
        EncodeControlRequest(ControlRequest{ControlOperation::ExchangeMetadata, std::nullopt,
                                            manager_id_, std::move(local)},
                             request),
        "transport manager metadata request encode failed peer={}", manager_id);
    P2P_RETURN_IF_ERROR(control_->Request(endpoint, request, remote),
                        "transport manager metadata request failed peer={}", manager_id);
    P2P_RETURN_IF_ERROR(ImportMetadata(remote, manager_id),
                        "transport manager metadata import failed peer={}", manager_id);
    UC_DEBUG("transport manager metadata exchange completed local={} peer={}", manager_id_,
             manager_id);
    return Status::OK();
}

Status TransportManager::ExportLocalMetadata(const ManagerID& manager_id, Metadata& out)
{
    P2P_RETURN_IF_TRUE(
        transports_.size() > UINT32_MAX, Status::InvalidParam(),
        "transport manager metadata export transport count exceeds limit count={} peer={}",
        transports_.size(), manager_id);

    PeerAdvertisement advertisement;
    advertisement.records.reserve(transports_.size());
    for (const auto& item : transports_) {
        Metadata metadata;
        P2P_RETURN_IF_ERROR(item.transport->ExportMetadata(manager_id, metadata),
                            "protocol={} failed to export metadata for peer={}",
                            static_cast<uint32_t>(item.protocol), manager_id);
        advertisement.records.push_back(
            TransportMetadataRecord{item.protocol, std::move(metadata)});
    }
    P2P_RETURN_IF_ERROR(EncodePeerAdvertisement(advertisement, out),
                        "failed to encode advertisement for peer={}", manager_id);
    return Status::OK();
}

Status TransportManager::ImportMetadata(const Metadata& metadata, const ManagerID& manager_id)
{
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id, endpoint), "invalid metadata peer={}",
                        manager_id);
    P2P_RETURN_IF_TRUE(metadata.size() < sizeof(uint32_t), Status::InvalidParam(),
                       "transport manager metadata import payload is too short peer={} bytes={}",
                       manager_id, metadata.size());

    PeerAdvertisement advertisement;
    P2P_RETURN_IF_ERROR(DecodePeerAdvertisement(metadata, advertisement),
                        "failed to decode advertisement peer={} bytes={}", manager_id,
                        metadata.size());

    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    for (const auto& record : advertisement.records) {
        const auto it = protocol_map_.find(record.protocol);
        if (it == protocol_map_.end()) {
            UC_DEBUG("transport manager metadata ignored unsupported protocol={} peer={}",
                     static_cast<uint32_t>(record.protocol), manager_id);
            continue;
        }

        P2P_RETURN_IF_ERROR(it->second->ImportMetadata(manager_id, record.metadata),
                            "protocol={} failed to import metadata from peer={}",
                            static_cast<uint32_t>(record.protocol), manager_id);
    }

    return Status::OK();
}

Status TransportManager::HandleMetadataExchange(const ManagerID& manager_id,
                                                const Metadata& remote_metadata,
                                                Metadata& local_metadata)
{
    UC_DEBUG("transport manager handling metadata exchange local={} peer={} request_bytes={}",
             manager_id_, manager_id, remote_metadata.size());
    P2P_RETURN_IF_ERROR(ImportMetadata(remote_metadata, manager_id),
                        "failed to import metadata from peer={}", manager_id);
    P2P_RETURN_IF_ERROR(ExportLocalMetadata(manager_id, local_metadata),
                        "failed to export metadata to peer={}", manager_id);
    UC_DEBUG("transport manager handled metadata exchange local={} peer={} response_bytes={}",
             manager_id_, manager_id, local_metadata.size());
    return Status::OK();
}

Status TransportManager::HandleControlRequest(const Metadata& request, Metadata& response)
{
    ControlRequest control_request{};
    P2P_RETURN_IF_ERROR(DecodeControlRequest(request, control_request),
                        "transport manager control request decode failed local={} bytes={}",
                        manager_id_, request.size());

    UC_DEBUG(
        "transport manager control request decoded local={} operation={} protocol={} peer={}",
        manager_id_, ControlOperationName(control_request.operation),
        control_request.protocol.has_value() ? static_cast<int32_t>(*control_request.protocol) : -1,
        control_request.manager_id);

    if (control_request.operation == ControlOperation::ExchangeMetadata) {
        P2P_RETURN_IF_ERROR(
            HandleMetadataExchange(control_request.manager_id, control_request.payload, response),
            "transport manager metadata control request failed local={} peer={}", manager_id_,
            control_request.manager_id);
        return Status::OK();
    }
    P2P_RETURN_IF_TRUE(
        !control_request.protocol.has_value(), Status::InvalidParam(),
        "transport manager control request missing protocol local={} operation={} peer={}",
        manager_id_, ControlOperationName(control_request.operation), control_request.manager_id);

    P2P_RETURN_IF_ERROR(
        ApplyConnectionLocally(control_request.operation, *control_request.protocol,
                               control_request.manager_id),
        "transport manager control request apply failed operation={} protocol={} peer={}",
        ControlOperationName(control_request.operation),
        static_cast<uint32_t>(*control_request.protocol), control_request.manager_id);
    UC_DEBUG("transport manager control request applied operation={} protocol={} peer={}",
             ControlOperationName(control_request.operation),
             static_cast<uint32_t>(*control_request.protocol), control_request.manager_id);
    return Status::OK();
}

Status TransportManager::RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle)
{
    handle = kInvalidMemoryHandle;
    P2P_RETURN_IF_TRUE(memory.addr == nullptr || memory.length == 0, Status::InvalidParam(),
                       "transport manager register memory invalid addr={} length={}", memory.addr,
                       memory.length);
    const auto address = detail::PtrToU64(memory.addr);
    P2P_RETURN_IF_TRUE(memory.length > std::numeric_limits<uint64_t>::max() - address,
                       Status::InvalidParam(),
                       "transport manager register memory address overflow addr=0x{:x} length={}",
                       address, memory.length);
    P2P_RETURN_IF_TRUE(transports_.empty(), Status::Error(),
                       "transport manager register memory failed: no installed transports");

    auto record = std::make_unique<MemoryRecord>();
    record->region = memory;
    std::vector<std::pair<Transport*, MemoryHandle>> registered;
    auto rollback = MakeScopeGuard([&registered] {
        for (auto it = registered.rbegin(); it != registered.rend(); ++it) {
            P2P_LOG_IF_ERROR(it->first->UnregisterMemory(it->second),
                             "transport manager register memory rollback failed handle={}",
                             it->second);
        }
    });
    for (const auto& item : transports_) {
        MemoryHandle transport_handle = kInvalidMemoryHandle;
        P2P_RETURN_IF_ERROR(item.transport->RegisterMemory(memory, transport_handle),
                            "transport manager register memory failed protocol={}",
                            static_cast<int>(item.protocol));
        P2P_RETURN_IF_TRUE(transport_handle == kInvalidMemoryHandle, Status::Error(),
                           "transport manager register memory returned an invalid handle "
                           "protocol={}",
                           static_cast<int>(item.protocol));
        record->transport_handles.emplace(item.protocol, transport_handle);
        registered.emplace_back(item.transport.get(), transport_handle);
    }

    handle = reinterpret_cast<MemoryHandle>(record.get());
    memories_.emplace(handle, std::move(record));
    rollback.Dismiss();
    UC_DEBUG("transport manager registered memory handle={} addr=0x{:x} length={}", handle, address,
             memory.length);
    return Status::OK();
}

Status TransportManager::UnregisterMemory(MemoryHandle handle)
{
    P2P_RETURN_IF_TRUE(handle == kInvalidMemoryHandle, Status::InvalidParam(),
                       "transport manager unregister memory invalid handle={}", handle);

    const auto it = memories_.find(handle);
    P2P_RETURN_IF_TRUE(it == memories_.end(), Status::Error(),
                       "transport manager unregister memory unknown handle={}", handle);

    for (const auto& item : it->second->transport_handles) {
        const auto transport_it = protocol_map_.find(item.first);
        P2P_RETURN_IF_TRUE(transport_it == protocol_map_.end(), Status::Error(),
                           "transport manager unregister memory failed protocol={} handle={}",
                           static_cast<int>(item.first), item.second);
        P2P_RETURN_IF_ERROR(transport_it->second->UnregisterMemory(item.second),
                            "transport manager unregister memory failed protocol={} handle={}",
                            static_cast<int>(item.first), item.second);
    }
    memories_.erase(it);
    UC_DEBUG("transport manager unregistered memory handle={}", handle);
    return Status::OK();
}

Status TransportManager::FindTransport(Operation& batch, Transport*& transport)
{
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ParseManagerID(batch.target_manager, endpoint),
                        "transport manager transfer selection invalid peer={}",
                        batch.target_manager);

    TransportProtocol protocol = TransportProtocol::Hixl;
    P2P_RETURN_IF_TRUE(!TransportForDirect(batch.direct, protocol), Status::Unsupported(),
                       "transport manager transfer selection unsupported direction={} peer={}",
                       static_cast<uint32_t>(batch.direct), batch.target_manager);
    const auto transport_it = protocol_map_.find(protocol);
    P2P_RETURN_IF_TRUE(transport_it == protocol_map_.end(), Status::Unsupported(),
                       "transport manager transfer selection protocol={} is not installed peer={}",
                       static_cast<uint32_t>(protocol), batch.target_manager);
    transport = transport_it->second;
    return Status::OK();
}

Status TransportManager::Connect(TransportProtocol protocol, const ManagerID& manager_id)
{
    P2P_RETURN_IF_ERROR(
        CoordinateConnectionWithPeer(ControlOperation::Connect, protocol, manager_id),
        "transport manager connect failed protocol={} peer={}", static_cast<uint32_t>(protocol),
        manager_id);
    return Status::OK();
}

Status TransportManager::Disconnect(TransportProtocol protocol, const ManagerID& manager_id)
{
    P2P_RETURN_IF_ERROR(
        CoordinateConnectionWithPeer(ControlOperation::Disconnect, protocol, manager_id),
        "transport manager disconnect failed protocol={} peer={}", static_cast<uint32_t>(protocol),
        manager_id);
    return Status::OK();
}

Status TransportManager::ApplyConnectionLocally(ControlOperation operation,
                                                TransportProtocol protocol,
                                                const ManagerID& manager_id)
{
    UC_DEBUG("transport manager local {} begin protocol={} peer={}",
             ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id);
    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    P2P_RETURN_IF_TRUE(
        shutting_down_ && operation == ControlOperation::Connect, Status::Error(),
        "transport manager local connect rejected during shutdown protocol={} peer={}",
        static_cast<uint32_t>(protocol), manager_id);
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id, endpoint),
                        "local {} has invalid peer={} protocol={}", ControlOperationName(operation),
                        manager_id, static_cast<uint32_t>(protocol));
    const auto it = protocol_map_.find(protocol);
    P2P_RETURN_IF_TRUE(it == protocol_map_.end(), Status::Unsupported(),
                       "transport manager local {} has unavailable protocol={} peer={}",
                       ControlOperationName(operation), static_cast<uint32_t>(protocol),
                       manager_id);
    const auto status = operation == ControlOperation::Connect ? it->second->Connect(manager_id)
                                                               : it->second->Disconnect(manager_id);
    P2P_RETURN_IF_ERROR(status, "local {} failed protocol={} peer={}",
                        ControlOperationName(operation), static_cast<uint32_t>(protocol),
                        manager_id);

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
    P2P_RETURN_IF_ERROR(
        ParseManagerID(manager_id, endpoint), "coordinate {} has invalid peer={} protocol={}",
        ControlOperationName(operation), manager_id, static_cast<uint32_t>(protocol));
    P2P_RETURN_IF_TRUE(
        !control_, Status::Error(),
        "transport manager coordinate {} failed: control channel is unavailable peer={} "
        "protocol={}",
        ControlOperationName(operation), manager_id, static_cast<uint32_t>(protocol));
    P2P_RETURN_IF_TRUE(protocol_map_.find(protocol) == protocol_map_.end(), Status::Unsupported(),
                       "transport manager coordinate {} has unavailable protocol={} peer={}",
                       ControlOperationName(operation), static_cast<uint32_t>(protocol),
                       manager_id);

    Metadata request;
    P2P_RETURN_IF_ERROR(
        EncodeControlRequest(ControlRequest{operation, protocol, manager_id_, {}}, request),
        "coordinate {} request encode failed protocol={} peer={}", ControlOperationName(operation),
        static_cast<uint32_t>(protocol), manager_id);

    const auto local_status = ApplyConnectionLocally(operation, protocol, manager_id);
    if (operation == ControlOperation::Connect && local_status != Status::OK()) {
        P2P_RETURN_IF_ERROR(local_status, "local connect failed protocol={} peer={}",
                            static_cast<uint32_t>(protocol), manager_id);
    }

    Metadata ack;
    UC_DEBUG(
        "transport manager coordinate {} requesting peer ACK protocol={} peer={} local_status={}",
        ControlOperationName(operation), static_cast<uint32_t>(protocol), manager_id,
        local_status.Underlying());
    const auto remote_status = control_->Request(endpoint, request, ack);
    if (local_status != Status::OK() || remote_status != Status::OK()) {
        if (operation == ControlOperation::Connect && remote_status != Status::OK()) {
            const auto rollback_status =
                ApplyConnectionLocally(ControlOperation::Disconnect, protocol, manager_id);
            UC_WARN("transport manager rolled back local connect protocol={} peer={} status={}",
                    static_cast<uint32_t>(protocol), manager_id, rollback_status.Underlying());
        }
        const auto status = local_status != Status::OK() ? local_status : remote_status;
        P2P_RETURN_IF_ERROR(status, "coordinated {} failed protocol={} peer={} local={} remote={}",
                            operation == ControlOperation::Connect ? "connect" : "disconnect",
                            static_cast<uint32_t>(protocol), manager_id, local_status.Underlying(),
                            remote_status.Underlying());
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
    P2P_RETURN_IF_ERROR(FindTransport(request, transport),
                        "transport manager sync transfer selection failed peer={}",
                        batch.target_manager);
    P2P_RETURN_IF_ERROR(transport->ExecuteSync(request),
                        "transport manager sync transfer failed peer={} segments={}",
                        batch.target_manager, batch.ops.size());
    UC_DEBUG("transport manager sync transfer completed peer={} segments={}", batch.target_manager,
             batch.ops.size());
    return Status::OK();
}

Status TransportManager::ExecuteAsync(const Operation& batch, TransferHandle& handle)
{
    handle = kInvalidTransferHandle;
    Transport* transport = nullptr;
    auto request = batch;
    P2P_RETURN_IF_ERROR(FindTransport(request, transport),
                        "transport manager async transfer selection failed peer={}",
                        batch.target_manager);

    TransferHandle transport_handle = kInvalidTransferHandle;
    P2P_RETURN_IF_ERROR(transport->ExecuteAsync(request, transport_handle),
                        "transport manager async transfer submit failed peer={} segments={}",
                        batch.target_manager, batch.ops.size());
    P2P_RETURN_IF_TRUE(
        transport_handle == kInvalidTransferHandle, Status::Error(),
        "transport manager async transfer returned an invalid handle peer={} segments={}",
        batch.target_manager, batch.ops.size());

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
    P2P_RETURN_IF_TRUE(handle == kInvalidTransferHandle, Status::InvalidParam(),
                       "transport manager transfer status invalid handle={}", handle);
    TransferRecord record;
    {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        const auto it = transfers_.find(handle);
        P2P_RETURN_IF_TRUE(it == transfers_.end() || it->second.transport == nullptr,
                           Status::Error(), "transport manager transfer status unknown handle={}",
                           handle);
        record = it->second;
    }
    const auto status = record.transport->GetStatus(record.transport_handle, transfer_status);
    if (status != Status::OK()) {
        P2P_LOG_IF_ERROR(status,
                         "transport manager transfer status query failed handle={} "
                         "transport_handle={}",
                         handle, record.transport_handle);
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
        return Status::InvalidParam(fmt::format("invalid manager id={}", manager_id));
    }

    const auto host = manager_id.substr(0, separator);
    const auto port_text = manager_id.substr(separator + 1);
    try {
        size_t parsed = 0;
        const auto port = std::stoul(port_text, &parsed, 10);
        if (parsed != port_text.size() || port == 0 ||
            port > std::numeric_limits<uint16_t>::max()) {
            return Status::InvalidParam(
                fmt::format("invalid manager port manager={} port={}", manager_id, port_text));
        }
        endpoint = Endpoint{host, static_cast<uint16_t>(port)};
        return Status::OK();
    } catch (const std::exception& error) {
        return Status::InvalidParam(
            fmt::format("failed to parse manager id={} error={}", manager_id, error.what()));
    }
}

}  // namespace transport
