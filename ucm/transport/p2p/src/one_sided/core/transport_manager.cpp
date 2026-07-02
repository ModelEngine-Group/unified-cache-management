#include "core/transport_manager.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/metadata_codec.h"
#include "control/metadata_channel.h"
#include "hixl/hixl_transport.h"
#include "logger/logger.h"

namespace transport {
namespace {

struct TransportMetadataRecord {
    std::string protocol;
    Metadata metadata;
};

struct PeerAdvertisement {
    Endpoint endpoint;
    std::vector<TransportMetadataRecord> records;
};

Status EncodePeerAdvertisement(const PeerAdvertisement& advertisement, Metadata& out)
{
    if (advertisement.records.size() > UINT32_MAX) { return Status::InvalidArgument; }

    out.clear();
    if (!detail::AppendString(out, advertisement.endpoint.host) ||
        !detail::AppendU32(out, static_cast<uint32_t>(advertisement.endpoint.port)) ||
        !detail::AppendU32(out, static_cast<uint32_t>(advertisement.records.size()))) {
        return Status::InvalidArgument;
    }

    for (const auto& record : advertisement.records) {
        if (!detail::AppendMetadataRecord(out, record.protocol, record.metadata)) {
            return Status::InvalidArgument;
        }
    }
    return Status::Ok;
}

Status DecodePeerAdvertisement(const Metadata& in, PeerAdvertisement& advertisement)
{
    size_t offset = 0;
    uint32_t remote_port = 0;
    uint32_t count = 0;
    if (!detail::ReadString(in, offset, advertisement.endpoint.host) ||
        !detail::ReadU32(in, offset, remote_port) || !detail::ReadU32(in, offset, count) ||
        remote_port > UINT16_MAX) {
        return Status::InvalidArgument;
    }
    advertisement.endpoint.port = static_cast<uint16_t>(remote_port);

    advertisement.records.clear();
    advertisement.records.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        TransportMetadataRecord record;
        if (!detail::ReadMetadataRecord(in, offset, record.protocol, record.metadata)) {
            return Status::InvalidArgument;
        }
        advertisement.records.push_back(std::move(record));
    }

    return offset == in.size() ? Status::Ok : Status::InvalidArgument;
}

bool SameMemoryRegion(const MemoryRegion& left, const MemoryRegion& right)
{
    return left.addr == right.addr && left.length == right.length && left.type == right.type &&
           left.device_id == right.device_id;
}

bool MemoryRange(const MemoryRegion& memory, uint64_t* begin, uint64_t* end)
{
    const auto range_begin = detail::PtrToU64(memory.addr);
    if (memory.length > std::numeric_limits<uint64_t>::max() - range_begin) { return false; }
    if (begin != nullptr) { *begin = range_begin; }
    if (end != nullptr) { *end = range_begin + memory.length; }
    return true;
}

bool MemoryRegionOverlaps(const MemoryRegion& left, const MemoryRegion& right)
{
    if (left.type != right.type || left.device_id != right.device_id) { return false; }

    uint64_t left_begin = 0;
    uint64_t left_end = 0;
    uint64_t right_begin = 0;
    uint64_t right_end = 0;
    if (!MemoryRange(left, &left_begin, &left_end) ||
        !MemoryRange(right, &right_begin, &right_end)) {
        return true;
    }
    return left_begin < right_end && right_begin < left_end;
}

const char* TransportForDirect(OperationDirect direct)
{
    if (direct == OperationDirect::RemoteDeviceHost) { return "hixl"; }
    return nullptr;
}

}  // namespace

TransportManager::TransportManager(ManagerID manager_id) : manager_id_(std::move(manager_id)) {}

TransportManager::~TransportManager()
{
    if (Shutdown() != Status::Ok) {}
}

Status TransportManager::Init()
{
    if (ParseManagerID(manager_id_, local_endpoint_) != Status::Ok) {
        return Status::InvalidArgument;
    }
    if (control_) { return Status::Ok; }
    control_ = std::make_shared<MetadataChannel>();
    auto status = control_->Init(
        LocalEndpoint(), [this](const Metadata& remote_metadata, Metadata& local_metadata) {
            return HandleMetadataExchange(ManagerID{}, remote_metadata, local_metadata);
        });
    if (status != Status::Ok) {
        control_.reset();
        return status;
    }
    return Status::Ok;
}

Status TransportManager::InstallTransport(const std::string& protocol, const InitAttrs& options)
{
    if (protocol_map_.find(protocol) != protocol_map_.end()) { return Status::Ok; }

    auto transport = CreateTransport(protocol);
    if (!transport) { return Status::InvalidArgument; }
    const auto status = transport->Init(options);
    if (status != Status::Ok) { return status; }

    protocol_map_[protocol] = transport.get();
    transports_.push_back(InstalledTransport{protocol, std::move(transport)});
    return Status::Ok;
}

TransportPtr TransportManager::CreateTransport(const std::string& protocol) const
{
    using Factory = std::function<TransportPtr()>;
    static const std::unordered_map<std::string, Factory> factories{
        {"hixl", []() { return std::make_shared<HixlTransport>(); }},
    };

    const auto it = factories.find(protocol);
    return it == factories.end() ? nullptr : it->second();
}

Status TransportManager::Shutdown()
{
    if (control_) { control_->Close(); }

    Status result = Status::Ok;
    for (auto& item : transports_) {
        const auto status = item.transport->Shutdown();
        if (status != Status::Ok && result == Status::Ok) { result = status; }
    }
    memories_.clear();
    next_memory_handle_ = 1;
    protocol_map_.clear();
    transports_.clear();
    return result;
}

Status TransportManager::ExchangeMetadata(const ManagerID& manager_id)
{
    Endpoint endpoint;
    auto status = ParseManagerID(manager_id, endpoint);
    if (status != Status::Ok) { return status; }

    if (manager_id == LocalEndpoint().ToString()) { return Status::Ok; }

    Metadata local;
    status = ExportLocalMetadata(manager_id, local);
    if (status != Status::Ok) { return status; }
    Metadata remote;
    status = control_->ExchangeMetadata(endpoint, local, remote);
    if (status == Status::Ok) { status = ImportMetadata(remote, manager_id); }
    return status;
}

Status TransportManager::ExportLocalMetadata(const ManagerID& manager_id, Metadata& out)
{
    if (transports_.size() > UINT32_MAX) { return Status::InvalidArgument; }

    PeerAdvertisement advertisement;
    advertisement.endpoint = LocalEndpoint();
    advertisement.records.reserve(transports_.size());
    for (const auto& item : transports_) {
        Metadata metadata;
        const auto status = item.transport->ExportMetadata(manager_id, metadata);
        if (status != Status::Ok) { return status; }
        advertisement.records.push_back(
            TransportMetadataRecord{item.protocol, std::move(metadata)});
    }
    return EncodePeerAdvertisement(advertisement, out);
}

Status TransportManager::ImportMetadata(const Metadata& metadata, const ManagerID& manager_id)
{
    if (metadata.size() < sizeof(uint32_t)) { return Status::InvalidArgument; }

    PeerAdvertisement advertisement;
    const auto decode_status = DecodePeerAdvertisement(metadata, advertisement);
    if (decode_status != Status::Ok) { return decode_status; }

    const auto remote_manager_id = advertisement.endpoint.ToString();
    if (!manager_id.empty() && manager_id != remote_manager_id) { return Status::InvalidArgument; }

    std::lock_guard<std::recursive_mutex> lock(peer_mutex_);
    for (const auto& record : advertisement.records) {
        const auto it = protocol_map_.find(record.protocol);
        if (it == protocol_map_.end()) { continue; }

        const auto status = it->second->ImportMetadata(remote_manager_id, record.metadata);
        if (status != Status::Ok) { return status; }
    }

    return Status::Ok;
}

Status TransportManager::HandleMetadataExchange(const ManagerID& manager_id,
                                                const Metadata& remote_metadata,
                                                Metadata& local_metadata)
{
    PeerAdvertisement advertisement;
    const auto decode_status = DecodePeerAdvertisement(remote_metadata, advertisement);
    if (decode_status != Status::Ok) { return decode_status; }

    const auto remote_manager_id = advertisement.endpoint.ToString();
    const auto& expected_manager_id = manager_id.empty() ? remote_manager_id : manager_id;
    const auto status = ImportMetadata(remote_metadata, expected_manager_id);
    if (status != Status::Ok) { return status; }
    return ExportLocalMetadata(remote_manager_id, local_metadata);
}

Status TransportManager::RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle)
{
    handle = kInvalidMemoryHandle;
    if (memory.addr == nullptr || memory.length == 0) { return Status::InvalidArgument; }
    if (!MemoryRange(memory, nullptr, nullptr)) { return Status::InvalidArgument; }
    if (transports_.empty()) { return Status::Failed; }

    for (const auto& item : memories_) {
        if (SameMemoryRegion(item.second.region, memory)) {
            handle = item.first;
            return Status::Ok;
        }
        if (MemoryRegionOverlaps(item.second.region, memory)) { return Status::InvalidArgument; }
    }

    MemoryRecord record;
    record.region = memory;
    for (const auto& item : transports_) {
        MemoryHandle transport_handle = kInvalidMemoryHandle;
        auto status = item.transport->RegisterMemory(memory, transport_handle);
        if (status == Status::Ok && transport_handle == kInvalidMemoryHandle) {
            status = Status::Failed;
        }
        if (status != Status::Ok) {
            UC_ERROR(
                "transport manager register memory failed protocol={} status={} handle={} "
                "addr=0x{:x} length={}",
                item.protocol, static_cast<int>(status), transport_handle,
                detail::PtrToU64(memory.addr), memory.length);
            continue;
        }
        record.transport_handles.emplace(item.protocol, transport_handle);
    }
    if (record.transport_handles.empty()) {
        UC_ERROR(
            "transport manager register memory failed: no transport accepted addr=0x{:x} "
            "length={}",
            detail::PtrToU64(memory.addr), memory.length);
        return Status::Failed;
    }

    handle = next_memory_handle_++;
    if (handle == kInvalidMemoryHandle) { handle = next_memory_handle_++; }
    memories_.emplace(handle, std::move(record));
    return Status::Ok;
}

Status TransportManager::UnregisterMemory(MemoryHandle handle)
{
    if (handle == kInvalidMemoryHandle) { return Status::InvalidArgument; }

    const auto it = memories_.find(handle);
    if (it == memories_.end()) { return Status::Failed; }

    for (const auto& item : it->second.transport_handles) {
        const auto transport_it = protocol_map_.find(item.first);
        if (transport_it == protocol_map_.end()) {
            UC_ERROR("transport manager unregister memory failed protocol={} handle={}", item.first,
                     item.second);
            return Status::Failed;
        }
        const auto status = transport_it->second->UnregisterMemory(item.second);
        if (status != Status::Ok) {
            UC_ERROR("transport manager unregister memory failed protocol={} status={} handle={}",
                     item.first, static_cast<int>(status), item.second);
            return Status::Failed;
        }
    }
    memories_.erase(it);
    return Status::Ok;
}

Status TransportManager::Execute(const Operation& batch)
{
    if (batch.target_manager.empty()) { return Status::InvalidArgument; }
    Endpoint endpoint;
    if (ParseManagerID(batch.target_manager, endpoint) != Status::Ok) {
        return Status::InvalidArgument;
    }
    const auto manager_id = endpoint.ToString();

    const auto* transport_name = TransportForDirect(batch.direct);
    if (transport_name == nullptr) { return Status::Failed; }
    const auto transport_it = protocol_map_.find(transport_name);
    if (transport_it == protocol_map_.end()) { return Status::Failed; }
    Operation normalized = batch;
    normalized.target_manager = manager_id;
    return transport_it->second->Execute(normalized);
}

Endpoint TransportManager::LocalEndpoint() const { return local_endpoint_; }

Status TransportManager::ParseManagerID(const ManagerID& manager_id, Endpoint& endpoint) const
{
    const auto separator = manager_id.rfind(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 >= manager_id.size()) {
        return Status::InvalidArgument;
    }

    const auto host = manager_id.substr(0, separator);
    const auto port_text = manager_id.substr(separator + 1);
    try {
        size_t parsed = 0;
        const auto port = std::stoul(port_text, &parsed, 10);
        if (parsed != port_text.size() || port == 0 ||
            port > std::numeric_limits<uint16_t>::max()) {
            return Status::InvalidArgument;
        }
        endpoint = Endpoint{host, static_cast<uint16_t>(port)};
        return Status::Ok;
    } catch (const std::exception&) {
        return Status::InvalidArgument;
    }
}

}  // namespace transport
