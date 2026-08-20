#include "core/transport_manager.h"
#include <chrono>
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

std::uint64_t SteadyNowUs()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::uint64_t UnixNowUs()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
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
    control_ = std::make_shared<ControlChannel>();
    auto status =
        control_->Init(LocalEndpoint(), [this](const Metadata& request, Metadata& response) {
            return HandleControlRequest(request, response);
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
        if (status != Status::OK() && result == Status::OK()) { result = status; }
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
    Metadata request;
    status = EncodeControlRequest(ControlRequest{ControlOperation::ExchangeMetadata, std::nullopt,
                                                 manager_id_, std::move(local)},
                                  request);
    if (status != Status::OK()) { return status; }
    status = control_->Request(endpoint, request, remote);
    if (status == Status::OK()) { status = ImportMetadata(remote, manager_id); }
    return status;
}

Status TransportManager::ExportLocalMetadata(const ManagerID& manager_id, Metadata& out)
{
    if (transports_.size() > UINT32_MAX) { return Status::InvalidParam(); }

    PeerAdvertisement advertisement;
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
    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK() ||
        metadata.size() < sizeof(uint32_t)) {
        return Status::InvalidParam();
    }

    PeerAdvertisement advertisement;
    const auto decode_status = DecodePeerAdvertisement(metadata, advertisement);
    if (decode_status != Status::OK()) { return decode_status; }

    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    for (const auto& record : advertisement.records) {
        const auto it = protocol_map_.find(record.protocol);
        if (it == protocol_map_.end()) { continue; }

        const auto status = it->second->ImportMetadata(manager_id, record.metadata);
        if (status != Status::OK()) { return status; }
    }

    return Status::OK();
}

Status TransportManager::HandleMetadataExchange(const ManagerID& manager_id,
                                                const Metadata& remote_metadata,
                                                Metadata& local_metadata)
{
    const auto status = ImportMetadata(remote_metadata, manager_id);
    if (status != Status::OK()) { return status; }
    return ExportLocalMetadata(manager_id, local_metadata);
}

Status TransportManager::HandleControlRequest(const Metadata& request, Metadata& response)
{
    ControlRequest control_request{};
    auto status = DecodeControlRequest(request, control_request);
    if (status != Status::OK()) { return status; }

    if (control_request.operation == ControlOperation::ExchangeMetadata) {
        return HandleMetadataExchange(control_request.manager_id, control_request.payload,
                                      response);
    }
    if (!control_request.protocol.has_value()) { return Status::InvalidParam(); }

    UC_INFO("transport manager received {} request protocol={} peer={}",
            control_request.operation == ControlOperation::Connect ? "connect" : "disconnect",
            static_cast<uint32_t>(*control_request.protocol), control_request.manager_id);
    return ApplyConnectionLocally(control_request.operation, *control_request.protocol,
                                  control_request.manager_id);
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
    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    if (shutting_down_ && operation == ControlOperation::Connect) { return Status::Error(); }

    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK()) { return Status::InvalidParam(); }
    const auto it = protocol_map_.find(protocol);
    if (it == protocol_map_.end()) { return Status::InvalidParam(); }
    const auto status = operation == ControlOperation::Connect ? it->second->Connect(manager_id)
                                                               : it->second->Disconnect(manager_id);
    if (status != Status::OK()) { return status; }

    const auto connection = std::make_pair(protocol, manager_id);
    if (operation == ControlOperation::Connect) {
        connections_.insert(connection);
    } else {
        connections_.erase(connection);
    }
    return Status::OK();
}

Status TransportManager::CoordinateConnectionWithPeer(ControlOperation operation,
                                                      TransportProtocol protocol,
                                                      const ManagerID& manager_id)
{
    Endpoint endpoint;
    if (ParseManagerID(manager_id, endpoint) != Status::OK() || !control_) {
        return Status::InvalidParam();
    }
    if (protocol_map_.find(protocol) == protocol_map_.end()) { return Status::InvalidParam(); }

    Metadata request;
    auto status =
        EncodeControlRequest(ControlRequest{operation, protocol, manager_id_, {}}, request);
    if (status != Status::OK()) { return status; }

    const auto local_status = ApplyConnectionLocally(operation, protocol, manager_id);
    if (operation == ControlOperation::Connect && local_status != Status::OK()) {
        UC_ERROR("transport manager local connect failed protocol={} peer={} status={}",
                 static_cast<uint32_t>(protocol), manager_id, local_status.Underlying());
        return local_status;
    }

    Metadata ack;
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
    UC_INFO("transport manager coordinated {} success protocol={} peer={}",
            operation == ControlOperation::Connect ? "connect" : "disconnect",
            static_cast<uint32_t>(protocol), manager_id);
    return Status::OK();
}

Status TransportManager::ExecuteSync(const Operation& batch)
{
    Transport* transport = nullptr;
    auto request = batch;
    auto status = FindTransport(request, transport);
    if (status != Status::OK()) { return status; }
    return transport->ExecuteSync(request);
}

Status TransportManager::ExecuteAsync(const Operation& batch, TransferHandle& handle,
                                      TransportCallTiming* timing)
{
    TransportCallTiming localTiming;
    auto& callTiming = timing != nullptr ? *timing : localTiming;
    callTiming = {};
    callTiming.manager_entered_us = SteadyNowUs();
    callTiming.manager_entered_ts_us = UnixNowUs();
    handle = kInvalidTransferHandle;
    Transport* transport = nullptr;
    auto request = batch;
    auto status = FindTransport(request, transport);
    if (status != Status::OK()) { return status; }

    std::uint64_t bytes = 0;
    for (const auto& segment : request.ops) { bytes += segment.length; }
    const auto submitStartedUs = SteadyNowUs();
    TransferHandle transport_handle = kInvalidTransferHandle;
    status = transport->ExecuteAsync(request, transport_handle, &callTiming);
    if (status != Status::OK() || transport_handle == kInvalidTransferHandle) {
        return status == Status::OK() ? Status::Error() : status;
    }

    const auto submittedUs = SteadyNowUs();
    const auto submittedTsUs = UnixNowUs();
    {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        handle = next_transfer_handle_++;
        if (handle == kInvalidTransferHandle) { handle = next_transfer_handle_++; }
        transfers_.emplace(
            handle, TransferRecord{transport, transport_handle, request.target_manager,
                                   request.opcode, request.direct, request.ops.size(), bytes,
                                   submittedUs, submittedTsUs, submittedUs - submitStartedUs});
    }
    // Transport PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=transport event=transfer_submitted manager={} target={} handle={} "
    //     "transport_handle={} opcode={} direct={} segments={} bytes={} submitted_ts_us={} "
    //     "manager_execute_async_ts_us={} backend_execute_async_ts_us={} "
    //     "manager_to_backend_execute_async_us={} submit_us={}",
    //     manager_id_, request.target_manager, handle, transport_handle,
    //     static_cast<unsigned>(request.opcode), static_cast<unsigned>(request.direct),
    //     request.ops.size(), bytes, submittedTsUs, callTiming.manager_entered_ts_us,
    //     callTiming.backend_called_ts_us,
    //     callTiming.backend_called_us >= callTiming.manager_entered_us
    //         ? callTiming.backend_called_us - callTiming.manager_entered_us
    //         : 0,
    //     submittedUs - submitStartedUs);
    return Status::OK();
}

Status TransportManager::GetStatus(TransferHandle handle, TransferStatus& transfer_status,
                                   TransportCallTiming* timing)
{
    TransportCallTiming localTiming;
    auto& callTiming = timing != nullptr ? *timing : localTiming;
    callTiming = {};
    callTiming.manager_entered_us = SteadyNowUs();
    callTiming.manager_entered_ts_us = UnixNowUs();
    if (handle == kInvalidTransferHandle) { return Status::InvalidParam(); }
    TransferRecord record;
    {
        std::lock_guard<std::mutex> lock(transfers_mutex_);
        const auto it = transfers_.find(handle);
        if (it == transfers_.end() || it->second.transport == nullptr) { return Status::Error(); }
        record = it->second;
    }
    const auto status =
        record.transport->GetStatus(record.transport_handle, transfer_status, &callTiming);
    if (status != Status::OK() || transfer_status != TransferStatus::Waiting) {
        const auto completedUs = SteadyNowUs();
        const auto completedTsUs = UnixNowUs();
        // Transport PERF log disabled: request_done already contains the corresponding timing.
        // UC_INFO(
        //     "[PERF] component=transport event=transfer_done manager={} target={} handle={} "
        //     "transport_handle={} opcode={} direct={} segments={} bytes={} status={} "
        //     "api_status={} submitted_ts_us={} completed_ts_us={} manager_get_status_ts_us={} "
        //     "backend_query_ts_us={} submit_us={} transfer_us={} manager_to_backend_query_us={} "
        //     "total_us={}",
        //     manager_id_, record.target_manager, handle, record.transport_handle,
        //     static_cast<unsigned>(record.opcode), static_cast<unsigned>(record.direct),
        //     record.segment_count, record.bytes,
        //     status == Status::OK() ? static_cast<int>(transfer_status) : -1, status.Underlying(),
        //     record.submitted_ts_us, completedTsUs, callTiming.manager_entered_ts_us,
        //     callTiming.backend_called_ts_us, record.submit_us,
        //     completedUs >= record.submitted_us ? completedUs - record.submitted_us : 0,
        //     callTiming.backend_called_us >= callTiming.manager_entered_us
        //         ? callTiming.backend_called_us - callTiming.manager_entered_us
        //         : 0,
        //     record.submit_us +
        //         (completedUs >= record.submitted_us ? completedUs - record.submitted_us : 0));
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
