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

const char* ConnectionRequestName(ManagerMessageType type)
{
    return type == ManagerMessageType::ConnectRequest ? "connect" : "disconnect";
}

bool TransportForDirect(OperationDirect direct, TransportProtocol& protocol)
{
    if (direct != OperationDirect::RemoteDeviceHost) { return false; }
    protocol = TransportProtocol::Hixl;
    return true;
}

}  // namespace

TransportManager::TransportManager(ManagerID manager_id)
    : manager_id_(std::move(manager_id)),
      memory_region_manager_(std::make_shared<MemoryRegionManager>())
{
}

TransportManager::~TransportManager() { (void)Shutdown(); }

Status TransportManager::Init()
{
    UC_DEBUG("transport manager init begin manager={}", manager_id_);
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id_, local_endpoint_),
                        "transport manager init failed manager={}", manager_id_);
    if (channel_manager_) {
        UC_DEBUG("transport manager init skipped: already initialized manager={}", manager_id_);
        return Status::OK();
    }
    channel_manager_ = std::make_unique<ChannelManager>();
    auto rollback = MakeScopeGuard([this]() { channel_manager_.reset(); });
    P2P_RETURN_IF_ERROR(
        channel_manager_->Init(
            manager_id_, LocalEndpoint(),
            [this](ManagerMessageType type, TransportProtocol protocol, const ManagerID& peer,
                   const Metadata& request, Metadata& response) {
                return HandleControlRequest(type, protocol, peer, request, response);
            }),
        "transport manager control init failed manager={}", manager_id_);
    rollback.Dismiss();
    UC_DEBUG("transport manager init completed manager={}", manager_id_);
    return Status::OK();
}

Status TransportManager::InstallTransport(TransportProtocol protocol, const InitAttrs& options)
{
    std::lock_guard<std::mutex> memory_lock(memory_mutex_);
    if (protocol_map_.find(protocol) != protocol_map_.end()) {
        UC_DEBUG("transport manager install skipped protocol={}: already installed",
                 static_cast<uint32_t>(protocol));
        return Status::OK();
    }

    auto transport = CreateTransport(protocol);
    P2P_RETURN_IF_TRUE(!transport, Status::Unsupported(),
                       "transport manager install failed: unsupported protocol={}",
                       static_cast<uint32_t>(protocol));
    const TransportContext context{memory_region_manager_};
    P2P_RETURN_IF_ERROR(transport->Init(context, options),
                        "transport manager install failed protocol={}",
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
    std::lock_guard<std::mutex> memory_lock(memory_mutex_);
    const auto peers = channel_manager_ ? channel_manager_->Peers() : std::vector<ManagerID>{};
    for (const auto& peer : peers) {
        Endpoint endpoint;
        const auto parse_status = ParseManagerID(peer, endpoint);
        if (parse_status.Failure()) {
            if (result.Success()) { result = parse_status; }
            continue;
        }
        for (const auto protocol : channel_manager_->ConnectedProtocols(endpoint)) {
            const auto status = Disconnect(protocol, peer);
            if (status.Failure() && result.Success()) { result = status; }
        }
        const auto close_status = channel_manager_->Close(endpoint);
        if (close_status.Failure() && result.Success()) { result = close_status; }
    }

    if (channel_manager_) {
        channel_manager_->Shutdown();
        channel_manager_.reset();
    }

    for (auto& item : transports_) {
        const auto status = item.transport->Shutdown();
        if (status != Status::OK()) {
            P2P_LOG_IF_ERROR(status, "transport manager transport shutdown failed protocol={}",
                             static_cast<uint32_t>(item.protocol));
            if (result == Status::OK()) { result = status; }
        }
    }
    {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        transfers_.clear();
        next_transfer_handle_ = 1;
    }
    protocol_map_.clear();
    transports_.clear();
    UC_DEBUG("transport manager shutdown completed manager={} status={}", manager_id_,
             result.Underlying());
    return result;
}

Status TransportManager::ExportLocalMetadata(const ManagerID& manager_id, Metadata& out)
{
    PeerAdvertisement advertisement;
    advertisement.records.reserve(transports_.size());
    for (const auto& item : transports_) {
        Metadata metadata;
        P2P_RETURN_IF_ERROR(item.transport->ExportMetadata(manager_id, metadata),
                            "transport manager metadata export failed protocol={} peer={}",
                            static_cast<uint32_t>(item.protocol), manager_id);
        advertisement.records.push_back(
            TransportMetadataRecord{item.protocol, std::move(metadata)});
    }
    return EncodePeerAdvertisement(advertisement, out);
}

Status TransportManager::ImportMetadata(const Metadata& metadata, const ManagerID& manager_id)
{
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id, endpoint),
                        "transport manager metadata import invalid peer={}", manager_id);
    if (metadata.size() < sizeof(uint32_t)) { return Status::InvalidParam(); }

    PeerAdvertisement advertisement;
    P2P_RETURN_IF_ERROR(DecodePeerAdvertisement(metadata, advertisement),
                        "transport manager metadata decode failed peer={}", manager_id);

    for (const auto& record : advertisement.records) {
        const auto it = protocol_map_.find(record.protocol);
        if (it == protocol_map_.end()) {
            UC_DEBUG("transport manager metadata ignored unsupported protocol={} peer={}",
                     static_cast<uint32_t>(record.protocol), manager_id);
            continue;
        }

        P2P_RETURN_IF_ERROR(it->second->ImportMetadata(manager_id, record.metadata),
                            "transport manager metadata import failed protocol={} peer={}",
                            static_cast<uint32_t>(record.protocol), manager_id);
    }

    return Status::OK();
}

Status TransportManager::HandleControlRequest(ManagerMessageType type, TransportProtocol protocol,
                                              const ManagerID& manager_id, const Metadata& request,
                                              Metadata& response)
{
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id, endpoint),
                        "transport manager control request invalid peer={}", manager_id);
    if (type == ManagerMessageType::ConnectRequest) {
        P2P_RETURN_IF_ERROR(ExportLocalMetadata(manager_id, response),
                            "transport manager local metadata export failed peer={}", manager_id);
    }
    P2P_RETURN_IF_ERROR(ApplyConnectionLocally(type, protocol, manager_id, endpoint, request),
                        "transport manager local {} failed protocol={} peer={}",
                        ConnectionRequestName(type), static_cast<uint32_t>(protocol), manager_id);
    UC_DEBUG("transport manager control request applied operation={} protocol={} peer={}",
             ConnectionRequestName(type), static_cast<uint32_t>(protocol), manager_id);
    return Status::OK();
}

Status TransportManager::RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle)
{
    handle = kInvalidMemoryHandle;
    std::lock_guard<std::mutex> memory_lock(memory_mutex_);
    P2P_RETURN_IF_ERROR(memory_region_manager_->ValidateMemoryRegion(memory),
                        "transport manager register memory validation failed addr={} length={}",
                        memory.addr, memory.length);
    handle = MemoryRegionManager::HashMemoryRegion(memory);
    P2P_RETURN_IF_ERROR(RegisterMemoryWithTransports(memory, handle),
                        "transport manager register memory failed addr={} length={}", memory.addr,
                        memory.length);
    return Status::OK();
}

Status TransportManager::RegisterMemoryWithTransports(const MemoryRegion& memory,
                                                      MemoryHandle handle)
{
    std::vector<Transport*> registered;
    auto rollback = MakeScopeGuard([&registered, handle]() {
        for (auto it = registered.rbegin(); it != registered.rend(); ++it) {
            P2P_LOG_IF_ERROR((*it)->UnregisterMemory(handle),
                             "transport manager memory registration rollback failed handle={}",
                             handle);
        }
    });
    for (const auto& item : transports_) {
        P2P_RETURN_IF_ERROR(item.transport->RegisterMemory(memory, handle),
                            "transport manager register memory failed protocol={} handle={}",
                            static_cast<uint32_t>(item.protocol), handle);
        registered.push_back(item.transport.get());
    }
    rollback.Dismiss();
    UC_DEBUG("transport manager registered memory handle={} addr={} length={}", handle, memory.addr,
             memory.length);
    return Status::OK();
}

Status TransportManager::UnregisterMemory(MemoryHandle handle)
{
    std::lock_guard<std::mutex> memory_lock(memory_mutex_);
    P2P_RETURN_IF_TRUE(handle == kInvalidMemoryHandle, Status::InvalidParam(),
                       "transport manager unregister memory invalid handle={}", handle);
    for (const auto& item : transports_) {
        P2P_RETURN_IF_ERROR(item.transport->UnregisterMemory(handle),
                            "transport manager unregister memory failed protocol={} handle={}",
                            static_cast<int>(item.protocol), handle);
    }
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
    P2P_RETURN_IF_TRUE(
        channel_manager_ == nullptr || !channel_manager_->IsConnected(endpoint, protocol),
        Status::Error(),
        "transport manager transfer selection protocol={} is not connected peer={}",
        static_cast<uint32_t>(protocol), batch.target_manager);
    transport = transport_it->second;
    return Status::OK();
}

Status TransportManager::Connect(TransportProtocol protocol, const ManagerID& manager_id)
{
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ValidateConnection(protocol, manager_id, endpoint),
                        "transport manager connect validation failed protocol={} peer={}",
                        static_cast<uint32_t>(protocol), manager_id);
    if (channel_manager_->IsConnected(endpoint, protocol)) { return Status::OK(); }

    Metadata request;
    P2P_RETURN_IF_ERROR(ExportLocalMetadata(manager_id, request),
                        "transport manager connect metadata export failed protocol={} peer={}",
                        static_cast<uint32_t>(protocol), manager_id);
    Metadata response;
    const auto request_status = channel_manager_->Request(
        endpoint, ManagerMessageType::ConnectRequest, protocol, request, response);
    if (request_status.Failure()) {
        P2P_LOG_IF_ERROR(ApplyConnectionLocally(ManagerMessageType::DisconnectRequest, protocol,
                                                manager_id, endpoint),
                         "transport manager failed peer connect cleanup failed protocol={} peer={}",
                         static_cast<uint32_t>(protocol), manager_id);
        return request_status;
    }
    const auto local_status = ApplyConnectionLocally(ManagerMessageType::ConnectRequest, protocol,
                                                     manager_id, endpoint, response);
    if (local_status.Success()) { return local_status; }

    Metadata ignored;
    P2P_LOG_IF_ERROR(channel_manager_->Request(endpoint, ManagerMessageType::DisconnectRequest,
                                               protocol, {}, ignored),
                     "transport manager connect rollback failed protocol={} peer={}",
                     static_cast<uint32_t>(protocol), manager_id);
    return local_status;
}

Status TransportManager::Disconnect(TransportProtocol protocol, const ManagerID& manager_id)
{
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ValidateConnection(protocol, manager_id, endpoint),
                        "transport manager disconnect validation failed protocol={} peer={}",
                        static_cast<uint32_t>(protocol), manager_id);
    Metadata response;
    const auto request_status = channel_manager_->Request(
        endpoint, ManagerMessageType::DisconnectRequest, protocol, {}, response);
    const auto local_status = ApplyConnectionLocally(ManagerMessageType::DisconnectRequest,
                                                     protocol, manager_id, endpoint);
    return local_status.Failure() ? local_status : request_status;
}

Status TransportManager::ApplyConnectionLocally(ManagerMessageType type, TransportProtocol protocol,
                                                const ManagerID& manager_id,
                                                const Endpoint& endpoint, const Metadata& metadata)
{
    std::lock_guard<std::mutex> lock(ConnectionMutex(manager_id));
    const bool connected = channel_manager_->IsConnected(endpoint, protocol);
    if ((type == ManagerMessageType::ConnectRequest && connected) ||
        (type == ManagerMessageType::DisconnectRequest && !connected)) {
        return Status::OK();
    }
    const auto it = protocol_map_.find(protocol);
    if (it == protocol_map_.end()) { return Status::Unsupported(); }
    if (type == ManagerMessageType::ConnectRequest) {
        P2P_RETURN_IF_ERROR(ImportMetadata(metadata, manager_id),
                            "transport manager peer metadata import failed peer={}", manager_id);
        const auto status = it->second->Connect(manager_id);
        if (status.Failure()) {
            P2P_LOG_IF_ERROR(it->second->Disconnect(manager_id),
                             "transport manager peer metadata cleanup failed protocol={} peer={}",
                             static_cast<uint32_t>(protocol), manager_id);
            return status;
        }
        channel_manager_->SetConnected(endpoint, protocol, true);
    } else {
        const auto status = it->second->Disconnect(manager_id);
        channel_manager_->SetConnected(endpoint, protocol, false);
        if (status.Failure()) { return status; }
    }

    UC_DEBUG("transport manager local {} completed protocol={} peer={}",
             ConnectionRequestName(type), static_cast<uint32_t>(protocol), manager_id);
    return Status::OK();
}

std::mutex& TransportManager::ConnectionMutex(const ManagerID& manager_id)
{
    std::lock_guard<std::mutex> lock(connection_mutexes_mutex_);
    return connection_mutexes_.try_emplace(manager_id).first->second;
}

Status TransportManager::ValidateConnection(TransportProtocol protocol, const ManagerID& manager_id,
                                            Endpoint& endpoint) const
{
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id, endpoint),
                        "transport manager connection invalid peer={}", manager_id);
    if (protocol_map_.find(protocol) == protocol_map_.end()) { return Status::Unsupported(); }
    return Status::OK();
}

Status TransportManager::Send(const ManagerID& manager_id, const Metadata& payload)
{
    Endpoint endpoint;
    P2P_RETURN_IF_ERROR(ParseManagerID(manager_id, endpoint),
                        "transport manager send invalid peer={}", manager_id);
    P2P_RETURN_IF_ERROR(channel_manager_->Send(endpoint, payload),
                        "transport manager send failed peer={} bytes={}", manager_id,
                        payload.size());
    return Status::OK();
}

Status TransportManager::Receive(ManagerID& manager_id, Metadata& payload)
{
    P2P_RETURN_IF_ERROR(channel_manager_->Receive(manager_id, payload),
                        "transport manager receive failed");
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
