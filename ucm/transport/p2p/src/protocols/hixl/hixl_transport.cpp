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
#include "protocols/hixl/hixl_transport.h"
#include <arpa/inet.h>
#include <limits>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "common/binary_codec.h"
#include "logger/logger.h"
#include "protocols/hixl/hixl_instance.h"

namespace transport {
namespace {

Status PickAvailablePort(const std::string& host, uint16_t& port)
{
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    if (getaddrinfo(host.c_str(), "0", &hints, &results) != 0) {
        return Status::Error(fmt::format("resolve host for available port failed: host={}", host));
    }

    Status status = Status::Error();
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
                    status = port == 0 ? Status::Error() : Status::OK();
                } else if (address.ss_family == AF_INET6) {
                    port = ntohs(reinterpret_cast<sockaddr_in6*>(&address)->sin6_port);
                    status = port == 0 ? Status::Error() : Status::OK();
                }
            }
        }

        close(candidate);
        if (status == Status::OK()) { break; }
    }

    freeaddrinfo(results);
    return status.Success() ? status
                            : Status::Error(fmt::format("no available port found: host={}", host));
}

Status EncodeMetadata(HixlRole role, const std::vector<HixlInstanceInfo>& instances, Metadata& out)
{
    if (instances.empty() || instances.size() > std::numeric_limits<uint32_t>::max()) {
        return Status::InvalidParam(
            fmt::format("encode metadata failed: instances={}", instances.size()));
    }

    out.clear();
    if (!detail::AppendU8(out, static_cast<uint8_t>(role)) ||
        !detail::AppendU32(out, static_cast<uint32_t>(instances.size()))) {
        return Status::InvalidParam(
            fmt::format("encode metadata header failed: role={} instances={}",
                        static_cast<uint32_t>(role), instances.size()));
    }
    for (const auto& instance : instances) {
        if (instance.physical_device_id < 0 || !detail::AppendString(out, instance.endpoint.host) ||
            !detail::AppendU16(out, instance.endpoint.port) ||
            !detail::AppendU32(out, static_cast<uint32_t>(instance.physical_device_id))) {
            return Status::InvalidParam(
                fmt::format("encode instance metadata failed: engine={} physical_device={}",
                            instance.endpoint.ToString(), instance.physical_device_id));
        }
    }
    return Status::OK();
}

Status DecodeMetadata(const Metadata& in, HixlRole& role, std::vector<HixlInstanceInfo>& instances)
{
    size_t offset = 0;
    uint8_t raw_role = 0;
    uint32_t count = 0;
    if (!detail::ReadU8(in, offset, raw_role) ||
        raw_role > static_cast<uint8_t>(HixlRole::Bidirectional) ||
        !detail::ReadU32(in, offset, count) || count == 0) {
        return Status::InvalidParam(
            fmt::format("decode metadata header failed: bytes={}", in.size()));
    }
    role = static_cast<HixlRole>(raw_role);

    instances.clear();
    instances.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        HixlInstanceInfo instance;
        uint32_t physical_device_id = 0;
        if (!detail::ReadString(in, offset, instance.endpoint.host) ||
            !detail::ReadU16(in, offset, instance.endpoint.port) ||
            !detail::ReadU32(in, offset, physical_device_id) ||
            physical_device_id > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            return Status::InvalidParam(
                fmt::format("decode instance metadata failed: index={} bytes={}", i, in.size()));
        }
        instance.physical_device_id = static_cast<int32_t>(physical_device_id);
        instances.push_back(std::move(instance));
    }
    if (offset != in.size()) {
        return Status::InvalidParam(fmt::format(
            "decode metadata has trailing bytes: consumed={} total={}", offset, in.size()));
    }
    return Status::OK();
}

}  // namespace

HixlTransport::HixlTransport() = default;

HixlTransport::~HixlTransport() { (void)Shutdown(); }

TransportProtocol HixlTransport::Protocol() const { return TransportProtocol::Hixl; }

Status HixlTransport::Init(const TransportContext& context, const InitAttrs& attrs)
{
    const auto* hixl_attrs = dynamic_cast<const HixlInitAttrs*>(&attrs);
    if (hixl_attrs == nullptr) {
        return Status::InvalidParam("invalid HIXL initialization attribute type");
    }
    return Init(context, *hixl_attrs);
}

Status HixlTransport::Init(const TransportContext& context, const HixlInitAttrs& attrs)
{
    if (!instances_.empty()) {
        UC_DEBUG("[Transport][HIXL] transport already initialized: instances={}",
                 instances_.size());
        return Status::OK();
    }
    if (attrs.instances.empty()) { return Status::InvalidParam("no HIXL instances configured"); }
    if (!context.memory_region_manager) {
        return Status::InvalidParam("memory region manager is not configured");
    }
    if (attrs.role != HixlRole::Client && attrs.instances.size() > 1) {
        return Status::InvalidParam(
            fmt::format("only Client role supports multiple HIXL instances: role={} instances={}",
                        static_cast<uint32_t>(attrs.role), attrs.instances.size()));
    }

    for (size_t i = 0; i < attrs.instances.size(); ++i) {
        const auto& instance_attrs = attrs.instances[i];
        Endpoint local_endpoint;
        local_endpoint.host = attrs.ip;
        if (attrs.role == HixlRole::Client) {
            local_endpoint.port = 0;
        } else if (instance_attrs.port < 0) {
            const auto status = PickAvailablePort(local_endpoint.host, local_endpoint.port);
            if (status != Status::OK()) {
                return {status.Underlying(),
                        fmt::format("pick available HIXL port failed host={}: {}",
                                    local_endpoint.host, status.ToString())};
            }
        } else if (instance_attrs.port > 0 &&
                   instance_attrs.port <=
                       static_cast<int32_t>(std::numeric_limits<uint16_t>::max())) {
            local_endpoint.port = static_cast<uint16_t>(instance_attrs.port);
        } else {
            return Status::InvalidParam(fmt::format("invalid HIXL port={}", instance_attrs.port));
        }
        UC_DEBUG("[Transport][HIXL] init instance={} role={} engine={} device={} options={}", i,
                 static_cast<uint32_t>(attrs.role), local_endpoint.ToString(),
                 instance_attrs.device_id, instance_attrs.options.size());

        instances_.push_back(
            std::make_unique<HixlInstance>(std::move(local_endpoint), instance_attrs.device_id));
    }

    connect_timeout_ms_ = attrs.connect_timeout_ms;
    transfer_timeout_ms_ = attrs.transfer_timeout_ms;
    role_ = attrs.role;
    memory_region_manager_ = context.memory_region_manager;

    for (size_t i = 0; i < instances_.size(); ++i) {
        const auto status = instances_[i]->Initialize(attrs.instances[i].options);
        if (status != Status::OK()) {
            for (auto& instance : instances_) { instance->Finalize(); }
            instances_.clear();
            return {status.Underlying(),
                    fmt::format("HIXL instance initialization failed instance={} device={}: {}", i,
                                attrs.instances[i].device_id, status.ToString())};
        }
    }
    UC_DEBUG("[Transport][HIXL] init success role={} instances={}", static_cast<uint32_t>(role_),
             instances_.size());
    return Status::OK();
}

Status HixlTransport::Shutdown()
{
    Status result = Status::OK();
    for (auto& item : peers_) {
        auto& peer = item.second;
        if (peer.local_index >= instances_.size() || !peer.connected) { continue; }
        const auto status = DisconnectRoute(peer, true);
        if (status != Status::OK() && result == Status::OK()) { result = status; }
        peer.connected = false;
    }

    if (memory_region_manager_) {
        std::vector<MemoryRegionManager::Region> memories;
        const auto query_status = memory_region_manager_->GetMemoryRegions(Protocol(), memories);
        if (query_status.Failure()) {
            result = query_status;
        } else {
            for (const auto& memory : memories) {
                const auto& native_handles =
                    memory.registrations[static_cast<size_t>(Protocol())]->native_handles;
                for (size_t i = 0; i < native_handles.size(); ++i) {
                    const auto native_handle = reinterpret_cast<hixl::MemHandle>(native_handles[i]);
                    const auto status = instances_[i]->UnregisterMemory(native_handle);
                    if (status != Status::OK() && result == Status::OK()) { result = status; }
                }
                (void)memory_region_manager_->RemoveMemoryRegion(Protocol(), memory.handle);
            }
        }
    }

    for (auto& instance : instances_) { instance->Finalize(); }
    instances_.clear();
    peers_.clear();
    pending_transfers_.clear();
    next_transfer_handle_ = 1;
    return result;
}

Status HixlTransport::RegisterMemory(const MemoryRegion& memory, MemoryHandle handle)
{
    MemoryRegionManager::Region existing;
    const auto query_status = memory_region_manager_->FindMemoryRegion(Protocol(), memory.device_id,
                                                                       memory.addr, existing);
    if (query_status.Success()) { return Status::DuplicateKey(); }

    MemoryRegionManager::NativeHandles native_handles;
    native_handles.reserve(instances_.size());
    auto rollback = MakeScopeGuard([this, &native_handles]() {
        for (size_t i = 0; i < native_handles.size(); ++i) {
            const auto registered = reinterpret_cast<hixl::MemHandle>(native_handles[i]);
            P2P_LOG_IF_ERROR(instances_[i]->UnregisterMemory(registered),
                             "HIXL rollback memory registration failed instance={} handle={}", i,
                             native_handles[i]);
        }
    });

    for (size_t i = 0; i < instances_.size(); ++i) {
        hixl::MemHandle native_handle = nullptr;
        P2P_RETURN_IF_ERROR(instances_[i]->RegisterMemory(memory, native_handle),
                            "HIXL register memory failed instance={}", i);
        P2P_RETURN_IF_TRUE(native_handle == nullptr, Status::Error(),
                           "HIXL register memory returned invalid handle instance={}", i);
        native_handles.push_back(
            static_cast<MemoryHandle>(reinterpret_cast<std::uintptr_t>(native_handle)));
    }

    rollback.Dismiss();
    P2P_RETURN_IF_ERROR(memory_region_manager_->AddMemoryRegion(Protocol(), memory,
                                                                std::move(native_handles), handle),
                        "HIXL memory region registration failed handle={}", handle);
    UC_DEBUG("[Transport][HIXL] memory registration completed: handle={} addr={} length={}", handle,
             memory.addr, memory.length);
    return Status::OK();
}

Status HixlTransport::UnregisterMemory(MemoryHandle handle)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    P2P_RETURN_IF_TRUE(handle == kInvalidMemoryHandle, Status::InvalidParam(),
                       "HIXL unregister memory received invalid handle");

    if (!memory_region_manager_) {
        return Status::Error("memory region manager is not configured");
    }
    MemoryRegionManager::Region record;
    P2P_RETURN_IF_ERROR(memory_region_manager_->FindMemoryRegion(Protocol(), handle, record),
                        "HIXL unregister memory received unknown handle={}", handle);
    const auto& native_handles =
        record.registrations[static_cast<size_t>(Protocol())]->native_handles;
    for (size_t i = 0; i < native_handles.size(); ++i) {
        const auto native_handle = reinterpret_cast<hixl::MemHandle>(native_handles[i]);
        P2P_RETURN_IF_ERROR(instances_[i]->UnregisterMemory(native_handle),
                            "HIXL unregister memory failed instance={} handle={}", i,
                            native_handle);
    }
    P2P_RETURN_IF_ERROR(memory_region_manager_->RemoveMemoryRegion(Protocol(), handle),
                        "HIXL remove memory region failed handle={}", handle);
    UC_DEBUG("[Transport][HIXL] memory unregistration completed: handle={}", handle);
    return Status::OK();
}

Status HixlTransport::ExportMetadata(const ManagerID&, Metadata& out)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    std::vector<HixlInstanceInfo> metadata;
    metadata.reserve(instances_.size());
    for (const auto& instance : instances_) {
        metadata.push_back(
            HixlInstanceInfo{instance->LocalEndpoint(), instance->PhysicalDeviceId()});
    }
    return EncodeMetadata(role_, metadata, out);
}

Status HixlTransport::ImportMetadata(const ManagerID& manager_id, const Metadata& metadata)
{
    std::vector<HixlInstanceInfo> remote_instances;
    HixlRole remote_role = HixlRole::Bidirectional;
    const auto status = DecodeMetadata(metadata, remote_role, remote_instances);
    if (status != Status::OK()) {
        return {status.Underlying(), fmt::format("HIXL metadata import failed peer={}: {}",
                                                 manager_id, status.ToString())};
    }

    {
        std::unique_lock<std::shared_mutex> peer_lock(peers_mutex_);
        const auto peer_it = peers_.find(manager_id);
        if (peer_it != peers_.end()) {
            // A second metadata exchange means the previous remote instance has stopped, even
            // when the new instance uses the same endpoint and device identifiers.
            if (peer_it->second.connected &&
                DisconnectRoute(peer_it->second, true) != Status::OK()) {
                UC_ERROR("[Transport][HIXL] cleanup stale route failed: peer={}", manager_id);
            }
            peers_.erase(peer_it);
        }

        Peer peer_state;
        peer_state.role = remote_role;
        peer_state.instances = std::move(remote_instances);
        if (peer_state.instances.size() > 1) {
            UC_DEBUG(
                "[Transport][HIXL] import peer metadata with multiple remote instances: peer={} "
                "remote_instances={}, use first instance for transfer route",
                manager_id, peer_state.instances.size());
        }
        const auto route_status = BuildRouteLocked(manager_id, peer_state);
        if (route_status.Failure()) {
            return {route_status.Underlying(),
                    fmt::format("HIXL metadata import failed to build route peer={}: {}",
                                manager_id, route_status.ToString())};
        }

        peers_[manager_id] = std::move(peer_state);
    }
    return Status::OK();
}

Status HixlTransport::BuildRouteLocked(const ManagerID& manager_id, Peer& peer)
{
    peer.local_index = SIZE_MAX;
    peer.connected = false;
    if (instances_.empty() || peer.instances.empty()) {
        return Status::InvalidParam(
            fmt::format("cannot build HIXL route peer={} local_instances={} remote_instances={}",
                        manager_id, instances_.size(), peer.instances.size()));
    }

    const auto& remote = peer.instances.front();
    const bool initiates_connection = role_ != HixlRole::Server && peer.role != HixlRole::Client;
    const auto local_count = instances_.size();
    if (local_count == 1) {
        if (initiates_connection && peer.instances.size() == 1 &&
            instances_.front()->LocalEndpoint().host == remote.endpoint.host &&
            instances_.front()->PhysicalDeviceId() == remote.physical_device_id) {
            return Status::Error(fmt::format(
                "local and remote single HIXL instances use the same device endpoint={} device={}",
                remote.endpoint.ToString(), remote.physical_device_id));
        }
        peer.local_index = 0;
        UC_DEBUG(
            "[Transport][HIXL] build route peer={} local_instance=0 local_engine={} "
            "local_device={} remote_engine={} remote_device={}",
            manager_id, instances_.front()->LocalEndpoint().ToString(),
            instances_.front()->PhysicalDeviceId(), remote.endpoint.ToString(),
            remote.physical_device_id);
        return Status::OK();
    }

    std::vector<size_t> load(local_count, 0);
    for (const auto& item : peers_) {
        if (item.first == manager_id) { continue; }
        if (item.second.local_index < load.size()) { ++load[item.second.local_index]; }
    }

    std::vector<size_t> candidates;
    size_t min_load = std::numeric_limits<size_t>::max();
    for (size_t local_index = 0; local_index < local_count; ++local_index) {
        if (initiates_connection &&
            instances_[local_index]->LocalEndpoint().host == remote.endpoint.host &&
            instances_[local_index]->PhysicalDeviceId() == remote.physical_device_id) {
            continue;
        }
        if (load[local_index] < min_load) {
            candidates.clear();
            min_load = load[local_index];
        }
        if (load[local_index] == min_load) { candidates.push_back(local_index); }
    }
    if (candidates.empty()) {
        return Status::Error(fmt::format("no valid local HIXL instance for endpoint={} device={}",
                                         remote.endpoint.ToString(), remote.physical_device_id));
    }

    const auto local_index = candidates.front();
    peer.local_index = local_index;
    UC_DEBUG(
        "[Transport][HIXL] build route peer={} local_instance={} local_engine={} "
        "local_device={} remote_engine={} remote_device={}",
        manager_id, local_index, instances_[local_index]->LocalEndpoint().ToString(),
        instances_[local_index]->PhysicalDeviceId(), remote.endpoint.ToString(),
        remote.physical_device_id);
    return Status::OK();
}

Status HixlTransport::DisconnectRoute(const Peer& peer, bool ignore_failure)
{
    if (peer.local_index >= instances_.size() || peer.instances.empty()) {
        return Status::Error(fmt::format(
            "invalid HIXL disconnect route local_instance={} local_count={} remote_count={}",
            peer.local_index, instances_.size(), peer.instances.size()));
    }
    if (role_ == HixlRole::Server || peer.role == HixlRole::Client) { return Status::OK(); }

    const auto remote_engine = peer.instances.front().endpoint.ToString();
    const auto status =
        instances_[peer.local_index]->Disconnect(remote_engine, connect_timeout_ms_);
    return ignore_failure ? Status::OK() : status;
}

Status HixlTransport::Connect(const ManagerID& manager_id)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    std::unique_lock<std::shared_mutex> peer_lock(peers_mutex_);
    const auto peer_it = peers_.find(manager_id);
    if (peer_it == peers_.end()) {
        return Status::Error(fmt::format("unknown HIXL peer={}", manager_id));
    }
    auto& peer = peer_it->second;
    if (peer.local_index == SIZE_MAX || peer.instances.empty()) {
        return Status::Error(fmt::format("HIXL peer={} has no route", manager_id));
    }
    if (peer.connected) {
        UC_DEBUG("[Transport][HIXL] connect skipped: peer={} already connected", manager_id);
        return Status::OK();
    }
    if (peer.local_index >= instances_.size()) {
        return Status::Error(
            fmt::format("invalid HIXL route peer={} local_instance={} local_count={}", manager_id,
                        peer.local_index, instances_.size()));
    }

    if (role_ == peer.role && role_ != HixlRole::Bidirectional) {
        return Status::InvalidParam(fmt::format(
            "incompatible HIXL roles local={} remote={} peer={}", static_cast<uint32_t>(role_),
            static_cast<uint32_t>(peer.role), manager_id));
    }

    const auto remote_engine = peer.instances.front().endpoint.ToString();
    if (role_ != HixlRole::Server && peer.role != HixlRole::Client) {
        const auto status =
            instances_[peer.local_index]->Connect(remote_engine, connect_timeout_ms_);
        if (status.Failure()) {
            return {status.Underlying(),
                    fmt::format("HIXL connection failed peer={} local_instance={}: {}", manager_id,
                                peer.local_index, status.ToString())};
        }
    }

    peer.connected = true;
    UC_DEBUG("[Transport][HIXL] connect completed: peer={} local_instance={} remote_engine={}",
             manager_id, peer.local_index, remote_engine);
    return Status::OK();
}

Status HixlTransport::Disconnect(const ManagerID& manager_id)
{
    std::unique_lock<std::shared_mutex> peer_lock(peers_mutex_);
    const auto peer_it = peers_.find(manager_id);
    if (peer_it == peers_.end()) {
        return Status::Error(fmt::format("unknown HIXL peer={}", manager_id));
    }
    auto& peer = peer_it->second;
    if (!peer.connected) {
        UC_DEBUG("[Transport][HIXL] disconnect skipped: peer={} is not connected", manager_id);
        return Status::OK();
    }
    if (peer.local_index >= instances_.size() || peer.instances.empty()) {
        return Status::Error(fmt::format("HIXL peer={} has an invalid route", manager_id));
    }

    const auto status = DisconnectRoute(peer, false);
    peer.connected = false;
    if (status == Status::OK()) {
        UC_DEBUG("[Transport][HIXL] disconnect completed: peer={}", manager_id);
        return status;
    }
    return {status.Underlying(),
            fmt::format("HIXL disconnection failed peer={}: {}", manager_id, status.ToString())};
}

Status HixlTransport::ValidateTransferLocked(const Operation& batch, size_t instance_index) const
{
    if (batch.target_manager.empty() || batch.ops.empty() || instance_index >= instances_.size()) {
        return Status::InvalidParam(
            fmt::format("invalid HIXL transfer peer={} segments={} instance={} instances={}",
                        batch.target_manager, batch.ops.size(), instance_index, instances_.size()));
    }
    for (const auto& item : batch.ops) {
        if (item.local_addr == nullptr || item.length == 0 || item.remote_addr == 0) {
            return Status::InvalidParam(
                fmt::format("invalid HIXL transfer segment local_addr={} remote_addr={} length={}",
                            item.local_addr, item.remote_addr, item.length));
        }

        if (!memory_region_manager_) {
            return Status::Error("memory region manager is not configured");
        }
        MemoryRegionManager::Region memory;
        const auto query_status = memory_region_manager_->FindContainingMemoryRegion(
            Protocol(), -1, item.local_addr, item.length, memory);
        const bool registered =
            query_status.Success() &&
            instance_index <
                memory.registrations[static_cast<size_t>(Protocol())]->native_handles.size();
        if (!registered) {
            return Status::InvalidParam(fmt::format(
                "HIXL transfer memory is not registered local_addr={} length={} instance={}",
                item.local_addr, item.length, instance_index));
        }
    }
    return Status::OK();
}

Status HixlTransport::ExecuteSync(const Operation& batch)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    if (role_ == HixlRole::Server) {
        return Status(Status::Unsupported().Underlying(),
                      "HIXL Server role cannot initiate synchronous transfer");
    }
    size_t local_index = SIZE_MAX;
    std::string remote_engine;
    {
        std::shared_lock<std::shared_mutex> peer_lock(peers_mutex_);
        const auto peer_it = peers_.find(batch.target_manager);
        if (peer_it == peers_.end()) {
            return Status::Error(
                fmt::format("synchronous HIXL transfer has unknown peer={}", batch.target_manager));
        }
        const auto& peer_state = peer_it->second;
        if (peer_state.local_index >= instances_.size() || peer_state.instances.empty() ||
            !peer_state.connected) {
            return Status::Error(
                fmt::format("synchronous HIXL transfer has invalid route peer={} connected={} "
                            "local_instance={} local_count={} remote_count={}",
                            batch.target_manager, peer_state.connected, peer_state.local_index,
                            instances_.size(), peer_state.instances.size()));
        }
        local_index = peer_state.local_index;
        remote_engine = peer_state.instances.front().endpoint.ToString();
    }

    const auto transfer_status = ValidateTransferLocked(batch, local_index);
    if (transfer_status != Status::OK()) {
        return {transfer_status.Underlying(),
                fmt::format("synchronous HIXL transfer validation failed peer={}: {}",
                            batch.target_manager, transfer_status.ToString())};
    }

    UC_DEBUG("[Transport][HIXL] synchronous transfer started: peer={} opcode={} segments={}",
             batch.target_manager, static_cast<int>(batch.opcode), batch.ops.size());
    const auto status = instances_[local_index]->TransferSync(remote_engine, batch.opcode,
                                                              batch.ops, transfer_timeout_ms_);
    return status.Success() ? status
                            : Status(status.Underlying(),
                                     fmt::format("synchronous HIXL transfer failed peer={}: {}",
                                                 batch.target_manager, status.ToString()));
}

Status HixlTransport::ExecuteAsync(const Operation& batch, TransferHandle& handle)
{
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    handle = kInvalidTransferHandle;
    if (role_ == HixlRole::Server) {
        return Status(Status::Unsupported().Underlying(),
                      "HIXL Server role cannot initiate asynchronous transfer");
    }
    size_t local_index = SIZE_MAX;
    std::string remote_engine;
    {
        std::shared_lock<std::shared_mutex> peer_lock(peers_mutex_);
        const auto peer_it = peers_.find(batch.target_manager);
        if (peer_it == peers_.end()) {
            return Status::Error(fmt::format("asynchronous HIXL transfer has unknown peer={}",
                                             batch.target_manager));
        }
        const auto& peer_state = peer_it->second;
        if (peer_state.local_index >= instances_.size() || peer_state.instances.empty() ||
            !peer_state.connected) {
            return Status::Error(
                fmt::format("asynchronous HIXL transfer has invalid route peer={} connected={} "
                            "local_instance={} local_count={} remote_count={}",
                            batch.target_manager, peer_state.connected, peer_state.local_index,
                            instances_.size(), peer_state.instances.size()));
        }
        local_index = peer_state.local_index;
        remote_engine = peer_state.instances.front().endpoint.ToString();
    }

    const auto transfer_status = ValidateTransferLocked(batch, local_index);
    if (transfer_status != Status::OK()) {
        return {transfer_status.Underlying(),
                fmt::format("asynchronous HIXL transfer validation failed peer={}: {}",
                            batch.target_manager, transfer_status.ToString())};
    }

    hixl::TransferReq request = nullptr;
    const auto status =
        instances_[local_index]->TransferAsync(remote_engine, batch.opcode, batch.ops, request);
    if (status != Status::OK()) {
        return {status.Underlying(),
                fmt::format("asynchronous HIXL transfer submission failed peer={}: {}",
                            batch.target_manager, status.ToString())};
    }

    {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        handle = next_transfer_handle_++;
        if (handle == kInvalidTransferHandle) { handle = next_transfer_handle_++; }
        pending_transfers_.emplace(handle, PendingTransfer{local_index, request});
    }
    UC_DEBUG(
        "[Transport][HIXL] asynchronous transfer tracked: peer={} opcode={} segments={} "
        "instance={} handle={} request={}",
        batch.target_manager, static_cast<int>(batch.opcode), batch.ops.size(), local_index, handle,
        request);
    return Status::OK();
}

Status HixlTransport::GetStatus(TransferHandle handle, TransferStatus& status)
{
    status = TransferStatus::Failed;
    if (handle == kInvalidTransferHandle) {
        return Status::InvalidParam("invalid HIXL transfer handle");
    }
    std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
    PendingTransfer pending;
    {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        const auto it = pending_transfers_.find(handle);
        if (it == pending_transfers_.end() || it->second.instance_index >= instances_.size()) {
            return Status::Error(fmt::format("unknown HIXL transfer handle={} instances={}", handle,
                                             instances_.size()));
        }
        pending = it->second;
    }

    TransferStatus transfer_status = TransferStatus::Waiting;
    const auto query_status =
        instances_[pending.instance_index]->GetTransferStatus(pending.request, transfer_status);
    if (query_status != Status::OK()) {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        pending_transfers_.erase(handle);
        return {query_status.Underlying(),
                fmt::format("HIXL transfer status query failed handle={} request={}: {}", handle,
                            pending.request, query_status.ToString())};
    }
    status = transfer_status;
    if (status != TransferStatus::Waiting) {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        pending_transfers_.erase(handle);
        UC_DEBUG("[Transport][HIXL] asynchronous transfer completed: handle={} status={}", handle,
                 static_cast<int>(status));
    }
    return Status::OK();
}

}  // namespace transport
