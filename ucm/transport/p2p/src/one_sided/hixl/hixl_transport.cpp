#include "hixl/hixl_transport.h"
#include <arpa/inet.h>
#include <limits>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "common/metadata_codec.h"
#include "hixl/hixl_instance.h"
#include "logger/logger.h"

namespace transport {
namespace {

Status PickAvailablePort(const std::string& host, uint16_t& port)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    if (getaddrinfo(host.c_str(), "0", &hints, &results) != 0) { return Status::Failed; }

    Status status = Status::Failed;
    for (auto* item = results; item != nullptr; item = item->ai_next) {
        const int candidate = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (candidate < 0) { continue; }

        if (bind(candidate, item->ai_addr, item->ai_addrlen) == 0) {
            sockaddr_storage address{};
            socklen_t address_length = sizeof(address);
            if (getsockname(candidate, reinterpret_cast<sockaddr*>(&address), &address_length) ==
                0) {
                if (address.ss_family == AF_INET) {
                    port = ntohs(reinterpret_cast<sockaddr_in*>(&address)->sin_port);
                    status = port == 0 ? Status::Failed : Status::Ok;
                } else if (address.ss_family == AF_INET6) {
                    port = ntohs(reinterpret_cast<sockaddr_in6*>(&address)->sin6_port);
                    status = port == 0 ? Status::Failed : Status::Ok;
                }
            }
        }

        close(candidate);
        if (status == Status::Ok) { break; }
    }

    freeaddrinfo(results);
    return status;
}

Status EncodeMetadata(const std::vector<HixlInstanceInfo>& instances, Metadata& out)
{
    if (instances.empty() || instances.size() > std::numeric_limits<uint32_t>::max()) {
        return Status::InvalidArgument;
    }

    out.clear();
    if (!detail::AppendU32(out, static_cast<uint32_t>(instances.size()))) {
        return Status::InvalidArgument;
    }
    for (const auto& instance : instances) {
        if (instance.device_id < 0 || !detail::AppendString(out, instance.endpoint.host) ||
            !detail::AppendU16(out, instance.endpoint.port) ||
            !detail::AppendU32(out, static_cast<uint32_t>(instance.device_id))) {
            return Status::InvalidArgument;
        }
    }
    return Status::Ok;
}

Status DecodeMetadata(const Metadata& in, std::vector<HixlInstanceInfo>& instances)
{
    size_t offset = 0;
    uint32_t count = 0;
    if (!detail::ReadU32(in, offset, count) || count == 0) { return Status::InvalidArgument; }

    instances.clear();
    instances.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        HixlInstanceInfo instance;
        uint32_t device_id = 0;
        if (!detail::ReadString(in, offset, instance.endpoint.host) ||
            !detail::ReadU16(in, offset, instance.endpoint.port) ||
            !detail::ReadU32(in, offset, device_id) ||
            device_id > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            return Status::InvalidArgument;
        }
        instance.device_id = static_cast<int32_t>(device_id);
        instances.push_back(std::move(instance));
    }
    return offset == in.size() ? Status::Ok : Status::InvalidArgument;
}

}  // namespace

HixlTransport::HixlTransport() = default;

HixlTransport::~HixlTransport()
{
    if (Shutdown() != Status::Ok) {}
}

TransportProtocol HixlTransport::Protocol() const { return TransportProtocol::Hixl; }

Status HixlTransport::Init(const InitAttrs& attrs)
{
    const auto* hixl_attrs = dynamic_cast<const HixlInitAttrs*>(&attrs);
    return hixl_attrs == nullptr ? Status::InvalidArgument : Init(*hixl_attrs);
}

Status HixlTransport::Init(const HixlInitAttrs& attrs)
{
    if (!instances_.empty()) { return Status::Ok; }
    if (attrs.instances.empty()) { return Status::InvalidArgument; }

    for (size_t i = 0; i < attrs.instances.size(); ++i) {
        const auto& instance_attrs = attrs.instances[i];
        Endpoint local_endpoint;
        local_endpoint.host = attrs.ip;
        if (instance_attrs.port < 0) {
            const auto status = PickAvailablePort(local_endpoint.host, local_endpoint.port);
            if (status != Status::Ok) {
                UC_ERROR("[Transport][HIXL] pick available port failed: host={}",
                         local_endpoint.host);
                return status;
            }
        } else if (instance_attrs.port <=
                   static_cast<int32_t>(std::numeric_limits<uint16_t>::max())) {
            local_endpoint.port = static_cast<uint16_t>(instance_attrs.port);
        } else {
            UC_ERROR("[Transport][HIXL] invalid port: port={}", instance_attrs.port);
            return Status::InvalidArgument;
        }
        UC_DEBUG("[Transport][HIXL] init instance={} endpoint={} device={} options={}", i,
                 local_endpoint.ToString(), instance_attrs.device_id,
                 instance_attrs.options.size());

        instances_.push_back(
            std::make_unique<HixlInstance>(std::move(local_endpoint), instance_attrs.device_id));
    }

    connect_timeout_ms_ = attrs.connect_timeout_ms;
    transfer_timeout_ms_ = attrs.transfer_timeout_ms;

    for (size_t i = 0; i < instances_.size(); ++i) {
        const auto status = instances_[i]->Initialize(attrs.instances[i].options);
        if (status != Status::Ok) {
            for (auto& instance : instances_) { instance->Finalize(); }
            instances_.clear();
            return status;
        }
    }
    UC_INFO("[Transport][HIXL] init success instances={}", instances_.size());
    return Status::Ok;
}

Status HixlTransport::Shutdown()
{
    std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
    Status result = Status::Ok;
    for (auto& item : peers_) {
        auto& peer = item.second;
        if (peer.local_index >= instances_.size() || !peer.connected) { continue; }
        const auto status = DisconnectRoute(peer, true);
        if (status != Status::Ok && result == Status::Ok) { result = status; }
        peer.connected = false;
    }

    for (const auto& memory : memories_) {
        for (const auto& handle : memory.second->native_handles) {
            if (handle.first >= instances_.size() || handle.second == nullptr) { continue; }
            const auto status = instances_[handle.first]->UnregisterMemory(handle.second);
            if (status != Status::Ok && result == Status::Ok) { result = status; }
        }
    }

    for (auto& instance : instances_) { instance->Finalize(); }
    instances_.clear();
    peers_.clear();
    memories_.clear();
    pending_transfers_.clear();
    next_transfer_handle_ = 1;
    return result;
}

Status HixlTransport::RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    handle = kInvalidMemoryHandle;
    if (instances_.empty()) { return Status::Failed; }

    std::unique_lock<std::shared_mutex> memory_lock(memories_mutex_);

    auto record = std::make_unique<LocalMemoryRecord>();
    record->region = memory;

    for (size_t i = 0; i < instances_.size(); ++i) {
        if (memory.type == MemoryType::Device && instances_[i]->DeviceId() != memory.device_id) {
            continue;
        }

        hixl::MemHandle native_handle = nullptr;
        const auto status = instances_[i]->RegisterMemory(memory, native_handle);
        if (status != Status::Ok || native_handle == nullptr) {
            for (const auto& item : record->native_handles) {
                if (instances_[item.first]->UnregisterMemory(item.second) != Status::Ok) {
                    UC_ERROR(
                        "[Transport][HIXL] rollback memory registration failed: instance={} "
                        "handle={}",
                        item.first, item.second);
                }
            }
            return status == Status::Ok ? Status::Failed : status;
        }
        record->native_handles.emplace(i, native_handle);
    }

    if (record->native_handles.empty()) { return Status::InvalidArgument; }
    handle = reinterpret_cast<MemoryHandle>(record.get());
    memories_.emplace(handle, std::move(record));
    return Status::Ok;
}

Status HixlTransport::UnregisterMemory(MemoryHandle handle)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (handle == kInvalidMemoryHandle) { return Status::InvalidArgument; }

    std::unique_lock<std::shared_mutex> memory_lock(memories_mutex_);
    const auto record_it = memories_.find(handle);
    if (record_it == memories_.end()) { return Status::Failed; }
    auto& record = *record_it->second;
    while (!record.native_handles.empty()) {
        const auto item = *record.native_handles.begin();
        if (item.first >= instances_.size() || item.second == nullptr) { return Status::Failed; }
        const auto status = instances_[item.first]->UnregisterMemory(item.second);
        if (status != Status::Ok) { return status; }
        record.native_handles.erase(item.first);
    }
    memories_.erase(record_it);
    return Status::Ok;
}

Status HixlTransport::ExportMetadata(const ManagerID&, Metadata& out)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    std::vector<HixlInstanceInfo> metadata;
    metadata.reserve(instances_.size());
    for (const auto& instance : instances_) {
        metadata.push_back(HixlInstanceInfo{instance->LocalEndpoint(), instance->DeviceId()});
    }
    return EncodeMetadata(metadata, out);
}

Status HixlTransport::ImportMetadata(const ManagerID& manager_id, const Metadata& metadata)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    std::vector<HixlInstanceInfo> remote_instances;
    const auto status = DecodeMetadata(metadata, remote_instances);
    if (status != Status::Ok) { return status; }

    {
        std::unique_lock<std::shared_mutex> peer_lock(peers_mutex_);
        const auto peer_it = peers_.find(manager_id);
        if (peer_it != peers_.end()) {
            // A second metadata exchange means the previous remote instance has stopped, even
            // when the new instance uses the same endpoint and device identifiers.
            if (peer_it->second.connected && DisconnectRoute(peer_it->second, true) != Status::Ok) {
                UC_ERROR("[Transport][HIXL] cleanup stale route failed: peer={}", manager_id);
            }
            peers_.erase(peer_it);
        }

        Peer peer_state;
        peer_state.instances = std::move(remote_instances);
        if (peer_state.instances.size() > 1) {
            UC_DEBUG(
                "[Transport][HIXL] import peer metadata with multiple remote instances: peer={} "
                "remote_instances={}, use first instance for transfer route",
                manager_id, peer_state.instances.size());
        }
        const auto route_status = BuildRouteLocked(manager_id, peer_state);
        if (route_status != Status::Ok) { return route_status; }

        peers_[manager_id] = std::move(peer_state);
    }
    return Status::Ok;
}

Status HixlTransport::BuildRouteLocked(const ManagerID& manager_id, Peer& peer)
{
    peer.local_index = SIZE_MAX;
    peer.connected = false;
    if (instances_.empty() || peer.instances.empty()) {
        UC_ERROR("[Transport][HIXL] build route failed: peer={} remote_instances={}", manager_id,
                 peer.instances.size());
        return Status::InvalidArgument;
    }

    const auto local_count = instances_.size();
    std::vector<size_t> load(local_count, 0);
    for (const auto& item : peers_) {
        if (item.first == manager_id) { continue; }
        if (item.second.local_index < load.size()) { ++load[item.second.local_index]; }
    }

    const auto& remote = peer.instances.front();
    std::vector<size_t> candidates;
    size_t min_load = std::numeric_limits<size_t>::max();
    for (size_t local_index = 0; local_index < local_count; ++local_index) {
        if (instances_[local_index]->LocalEndpoint().host == remote.endpoint.host &&
            instances_[local_index]->DeviceId() == remote.device_id) {
            continue;
        }
        if (load[local_index] < min_load) {
            candidates.clear();
            min_load = load[local_index];
        }
        if (load[local_index] == min_load) { candidates.push_back(local_index); }
    }
    if (candidates.empty()) {
        UC_ERROR(
            "[Transport][HIXL] build route failed: no valid local instance for endpoint={} "
            "device={}",
            remote.endpoint.ToString(), remote.device_id);
        return Status::Failed;
    }

    const auto local_index = candidates.front();
    peer.local_index = local_index;
    UC_DEBUG(
        "[Transport][HIXL] build route peer={} local_instance={} local_engine={} "
        "local_device={} remote_engine={} remote_device={}",
        manager_id, local_index, instances_[local_index]->LocalEndpoint().ToString(),
        instances_[local_index]->DeviceId(), remote.endpoint.ToString(), remote.device_id);
    return Status::Ok;
}

Status HixlTransport::DisconnectRoute(const Peer& peer, bool ignore_failure)
{
    if (peer.local_index >= instances_.size() || peer.instances.empty()) { return Status::Failed; }

    const auto remote_engine = peer.instances.front().endpoint.ToString();
    const auto status =
        instances_[peer.local_index]->Disconnect(remote_engine, connect_timeout_ms_);
    return ignore_failure ? Status::Ok : status;
}

Status HixlTransport::Connect(const ManagerID& manager_id)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    std::unique_lock<std::shared_mutex> peer_lock(peers_mutex_);
    const auto peer_it = peers_.find(manager_id);
    if (peer_it == peers_.end()) { return Status::Failed; }
    auto& peer = peer_it->second;
    if (peer.local_index == SIZE_MAX || peer.instances.empty()) { return Status::Failed; }
    if (peer.connected) { return Status::Ok; }
    if (peer.local_index >= instances_.size()) { return Status::Failed; }

    const auto remote_engine = peer.instances.front().endpoint.ToString();
    const auto status = instances_[peer.local_index]->Connect(remote_engine, connect_timeout_ms_);
    if (status != Status::Ok) { return status; }

    peer.connected = true;
    return Status::Ok;
}

Status HixlTransport::Disconnect(const ManagerID& manager_id)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    std::unique_lock<std::shared_mutex> peer_lock(peers_mutex_);
    const auto peer_it = peers_.find(manager_id);
    if (peer_it == peers_.end()) { return Status::Failed; }
    auto& peer = peer_it->second;
    if (!peer.connected) { return Status::Ok; }
    if (peer.local_index >= instances_.size() || peer.instances.empty()) { return Status::Failed; }

    const auto status = DisconnectRoute(peer, false);
    if (status != Status::Ok) { return status; }

    peer.connected = false;
    return Status::Ok;
}

Status HixlTransport::ValidateTransferLocked(const Operation& batch, size_t instance_index) const
{
    if (batch.target_manager.empty() || batch.ops.empty() || instance_index >= instances_.size()) {
        return Status::InvalidArgument;
    }
    for (const auto& item : batch.ops) {
        if (item.local_addr == nullptr || item.length == 0 || item.remote_addr == 0) {
            return Status::InvalidArgument;
        }

        const auto local_address = detail::PtrToU64(item.local_addr);
        bool registered = false;
        for (const auto& memory : memories_) {
            const auto begin = detail::PtrToU64(memory.second->region.addr);
            if (local_address < begin) { continue; }

            const auto offset = local_address - begin;
            if (offset <= memory.second->region.length &&
                item.length <= memory.second->region.length - offset &&
                memory.second->native_handles.find(instance_index) !=
                    memory.second->native_handles.end()) {
                registered = true;
                break;
            }
        }
        if (!registered) { return Status::InvalidArgument; }
    }
    return Status::Ok;
}

Status HixlTransport::ExecuteSync(const Operation& batch)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    size_t local_index = SIZE_MAX;
    std::string remote_engine;
    {
        std::shared_lock<std::shared_mutex> peer_lock(peers_mutex_);
        const auto peer_it = peers_.find(batch.target_manager);
        if (peer_it == peers_.end()) { return Status::Failed; }
        const auto& peer_state = peer_it->second;
        if (peer_state.local_index >= instances_.size() || peer_state.instances.empty() ||
            !peer_state.connected) {
            return Status::Failed;
        }
        local_index = peer_state.local_index;
        remote_engine = peer_state.instances.front().endpoint.ToString();
    }

    {
        std::shared_lock<std::shared_mutex> memory_lock(memories_mutex_);
        const auto transfer_status = ValidateTransferLocked(batch, local_index);
        if (transfer_status != Status::Ok) { return transfer_status; }
    }

    return instances_[local_index]->TransferSync(remote_engine, batch.opcode, batch.ops,
                                                 transfer_timeout_ms_);
}

Status HixlTransport::ExecuteAsync(const Operation& batch, TransferHandle& handle)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    handle = kInvalidTransferHandle;
    size_t local_index = SIZE_MAX;
    std::string remote_engine;
    {
        std::shared_lock<std::shared_mutex> peer_lock(peers_mutex_);
        const auto peer_it = peers_.find(batch.target_manager);
        if (peer_it == peers_.end()) { return Status::Failed; }
        const auto& peer_state = peer_it->second;
        if (peer_state.local_index >= instances_.size() || peer_state.instances.empty() ||
            !peer_state.connected) {
            return Status::Failed;
        }
        local_index = peer_state.local_index;
        remote_engine = peer_state.instances.front().endpoint.ToString();
    }

    {
        std::shared_lock<std::shared_mutex> memory_lock(memories_mutex_);
        const auto transfer_status = ValidateTransferLocked(batch, local_index);
        if (transfer_status != Status::Ok) { return transfer_status; }
    }

    hixl::TransferReq request = nullptr;
    const auto status =
        instances_[local_index]->TransferAsync(remote_engine, batch.opcode, batch.ops, request);
    if (status != Status::Ok) { return status; }

    {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        handle = next_transfer_handle_++;
        if (handle == kInvalidTransferHandle) { handle = next_transfer_handle_++; }
        pending_transfers_.emplace(handle, PendingTransfer{local_index, request});
    }
    return Status::Ok;
}

Status HixlTransport::GetStatus(TransferHandle handle, TransferStatus& status)
{
    status = TransferStatus::Failed;
    if (handle == kInvalidTransferHandle) { return Status::InvalidArgument; }
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    PendingTransfer pending;
    {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        const auto it = pending_transfers_.find(handle);
        if (it == pending_transfers_.end() || it->second.instance_index >= instances_.size()) {
            return Status::Failed;
        }
        pending = it->second;
    }

    TransferStatus transfer_status = TransferStatus::Waiting;
    const auto query_status =
        instances_[pending.instance_index]->GetTransferStatus(pending.request, transfer_status);
    if (query_status != Status::Ok) {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        pending_transfers_.erase(handle);
        return query_status;
    }
    status = transfer_status;
    if (status != TransferStatus::Waiting) {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        pending_transfers_.erase(handle);
    }
    return Status::Ok;
}

}  // namespace transport
