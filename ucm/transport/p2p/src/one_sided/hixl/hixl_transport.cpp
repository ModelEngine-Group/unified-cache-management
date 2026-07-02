#include "hixl/hixl_transport.h"
#include <string>
#include <vector>
#include "common/acl_runtime_context.h"
#include "common/metadata_codec.h"
#include "logger/logger.h"

namespace transport {
namespace {

Status EncodeMetadata(const HixlTransportMetadata& metadata, Metadata& out)
{
    if (metadata.local_engine.empty()) { return Status::InvalidArgument; }
    out.clear();
    return detail::AppendString(out, metadata.local_engine) ? Status::Ok : Status::InvalidArgument;
}

Status DecodeMetadata(const Metadata& in, HixlTransportMetadata& metadata)
{
    size_t offset = 0;
    if (!detail::ReadString(in, offset, metadata.local_engine) || offset != in.size() ||
        metadata.local_engine.empty()) {
        return Status::InvalidArgument;
    }
    return Status::Ok;
}

TransferStatus ConvertTransferStatus(hixl::TransferStatus status)
{
    switch (status) {
        case hixl::TransferStatus::WAITING: return TransferStatus::Waiting;
        case hixl::TransferStatus::COMPLETED: return TransferStatus::Completed;
        case hixl::TransferStatus::FAILED:
        case hixl::TransferStatus::TIMEOUT: return TransferStatus::Failed;
    }
    return TransferStatus::Failed;
}

}  // namespace

HixlTransport::HixlTransport() = default;

HixlTransport::~HixlTransport()
{
    if (Shutdown() != Status::Ok) {}
}

TransportProtocol HixlTransport::Protocol() const { return TransportProtocol::Hixl; }

Status HixlTransport::Init(const InitAttrs& options)
{
    const auto* attrs = dynamic_cast<const HixlInitAttrs*>(&options);
    return attrs == nullptr ? Status::InvalidArgument : Init(*attrs);
}

Status HixlTransport::Init(const HixlInitAttrs& options)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (options.local_engine.empty() || options.device_id < 0) { return Status::InvalidArgument; }

    aclrtContext previous_context = nullptr;
    (void)aclrtGetCurrentContext(&previous_context);
    const auto set_device_status = aclrtSetDevice(options.device_id);
    if (set_device_status != ACL_ERROR_NONE) {
        UC_ERROR("transport hixl set device failed: aclrtSetDevice({}) returned {}",
                 options.device_id, static_cast<int>(set_device_status));
        return Status::Failed;
    }
    aclrtContext context = nullptr;
    const auto get_context_status = aclrtGetCurrentContext(&context);
    if (get_context_status != ACL_ERROR_NONE || context == nullptr) {
        UC_ERROR("transport hixl get context failed: aclrtGetCurrentContext returned {}",
                 static_cast<int>(get_context_status));
        if (previous_context != nullptr) { (void)aclrtSetCurrentContext(previous_context); }
        return Status::Failed;
    }

    std::map<hixl::AscendString, hixl::AscendString> hixl_options;
    for (const auto& item : options.options) {
        hixl_options.emplace(item.first.c_str(), item.second.c_str());
    }
    const auto init_status = hixl_.Initialize(options.local_engine.c_str(), hixl_options);
    if (init_status != hixl::SUCCESS) {
        UC_ERROR("transport hixl init failed: Initialize(\"{}\") returned {}", options.local_engine,
                 static_cast<int>(init_status));
        if (previous_context != nullptr) { (void)aclrtSetCurrentContext(previous_context); }
        return Status::Failed;
    }
    if (previous_context != nullptr && previous_context != context) {
        const auto restore_status = aclrtSetCurrentContext(previous_context);
        if (restore_status != ACL_ERROR_NONE) {
            UC_ERROR("transport hixl restore context failed: aclrtSetCurrentContext returned {}",
                     static_cast<int>(restore_status));
            hixl_.Finalize();
            return Status::Failed;
        }
    }
    local_engine_ = options.local_engine;
    options_ = options.options;
    context_ = context;
    device_id_ = options.device_id;
    connect_timeout_ms_ = options.connect_timeout_ms;
    transfer_timeout_ms_ = options.transfer_timeout_ms;
    return Status::Ok;
}

bool HixlTransport::ValidateMemory(uint64_t address, uint64_t length) const
{
    if (length == 0) { return false; }
    for (const auto& item : memories_) {
        const auto begin = detail::PtrToU64(item.second.region.addr);
        if (address < begin) { continue; }
        const auto offset = address - begin;
        if (offset <= item.second.region.length && length <= item.second.region.length - offset) {
            return true;
        }
    }
    return false;
}

Status HixlTransport::Shutdown()
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    Status result = Status::Ok;
    if (!local_engine_.empty()) {
        WithAclRuntimeContext context_guard(context_);
        if (context_guard.status() != Status::Ok) { return context_guard.status(); }
        for (const auto& item : peers_) {
            const auto status =
                hixl_.Disconnect(item.second.remote_engine.c_str(), connect_timeout_ms_);
            if (status != hixl::SUCCESS) {
                UC_WARN(
                    "transport hixl disconnect during shutdown returned: Disconnect(\"{}\") "
                    "returned {}",
                    item.second.remote_engine, static_cast<int>(status));
            }
        }
        for (const auto& item : memories_) {
            if (item.second.native_handle != nullptr) {
                const auto status = hixl_.DeregisterMem(item.second.native_handle);
                if (status != hixl::SUCCESS) {
                    UC_ERROR("transport hixl deregister memory failed: DeregisterMem returned {}",
                             static_cast<int>(status));
                    result = Status::Failed;
                }
            }
        }
        hixl_.Finalize();
        context_ = nullptr;
        device_id_ = -1;
        local_engine_.clear();
    }
    peers_.clear();
    memories_.clear();
    pending_transfers_.clear();
    next_transfer_handle_ = 1;
    return result;
}

Status HixlTransport::RegisterMemory(const MemoryRegion& memory, MemoryHandle& handle)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }
    handle = kInvalidMemoryHandle;
    const auto address = detail::PtrToU64(memory.addr);

    LocalMemoryRecord record;
    record.region = memory;
    hixl::MemDesc desc{};
    desc.addr = static_cast<uintptr_t>(address);
    desc.len = static_cast<size_t>(memory.length);
    hixl::MemHandle native_handle = nullptr;
    const auto type = memory.type == MemoryType::Device ? hixl::MEM_DEVICE : hixl::MEM_HOST;
    const auto status = hixl_.RegisterMem(desc, type, native_handle);
    if (status != hixl::SUCCESS) {
        UC_ERROR(
            "transport hixl register memory failed: RegisterMem(addr=0x{:x}, length={}) returned "
            "{}",
            address, memory.length, static_cast<int>(status));
        return Status::Failed;
    }
    record.native_handle = native_handle;
    memories_.emplace(address, record);
    handle = address;
    return Status::Ok;
}

Status HixlTransport::UnregisterMemory(MemoryHandle handle)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }
    if (handle == kInvalidMemoryHandle) { return Status::InvalidArgument; }
    const auto it = memories_.find(handle);
    if (it == memories_.end()) { return Status::Failed; }
    if (it->second.native_handle != nullptr) {
        if (hixl_.DeregisterMem(it->second.native_handle) != hixl::SUCCESS) {
            return Status::Failed;
        }
        it->second.native_handle = nullptr;
    }
    memories_.erase(it);
    return Status::Ok;
}

Status HixlTransport::ExportMetadata(const ManagerID&, Metadata& out)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return EncodeMetadata(HixlTransportMetadata{local_engine_}, out);
}

Status HixlTransport::ImportMetadata(const ManagerID& manager_id, const Metadata& metadata)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }
    if (manager_id.empty()) { return Status::InvalidArgument; }
    HixlTransportMetadata remote_meta;
    const auto status = DecodeMetadata(metadata, remote_meta);
    if (status != Status::Ok) { return status; }

    const auto peer_it = peers_.find(manager_id);
    if (peer_it != peers_.end()) {
        auto& peer = peer_it->second;
        if (peer.metadata == metadata) { return Status::Ok; }
        if (peer.remote_engine == remote_meta.local_engine) {
            peer.metadata = metadata;
            return Status::Ok;
        }
        if (peer.connected) {
            const auto disconnect_status =
                hixl_.Disconnect(peer.remote_engine.c_str(), connect_timeout_ms_);
            if (disconnect_status != hixl::SUCCESS) {
                UC_ERROR("transport hixl disconnect failed: Disconnect(\"{}\") returned {}",
                         peer.remote_engine, static_cast<int>(disconnect_status));
                return Status::Failed;
            }
        }
    }

    Peer peer_state;
    peer_state.remote_engine = remote_meta.local_engine;
    peer_state.metadata = metadata;
    peers_[manager_id] = std::move(peer_state);
    return Status::Ok;
}

Status HixlTransport::Connect(const ManagerID& peer)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }
    const auto peer_it = peers_.find(peer);
    if (peer_it == peers_.end()) { return Status::Failed; }
    auto& peer_state = peer_it->second;
    if (peer_state.connected) { return Status::Ok; }
    const auto connect_status =
        hixl_.Connect(peer_state.remote_engine.c_str(), connect_timeout_ms_);
    if (connect_status != hixl::SUCCESS) {
        UC_ERROR("transport hixl connect failed: Connect(\"{}\") returned {}",
                 peer_state.remote_engine, static_cast<int>(connect_status));
        return Status::Failed;
    }
    peer_state.connected = true;
    return Status::Ok;
}

Status HixlTransport::Disconnect(const ManagerID& peer)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }
    const auto peer_it = peers_.find(peer);
    if (peer_it == peers_.end()) { return Status::Failed; }
    auto& peer_state = peer_it->second;
    if (!peer_state.connected) { return Status::Ok; }

    peer_state.connected = false;
    const auto status = hixl_.Disconnect(peer_state.remote_engine.c_str(), connect_timeout_ms_);
    if (status != hixl::SUCCESS) {
        UC_WARN("transport hixl disconnect returned: Disconnect(\"{}\") returned {}",
                peer_state.remote_engine, static_cast<int>(status));
        return Status::Failed;
    }
    return Status::Ok;
}

Status HixlTransport::BuildTransfer(const Operation& batch,
                                    std::vector<hixl::TransferOpDesc>& descs)
{
    if (batch.target_manager.empty() || batch.ops.empty()) { return Status::InvalidArgument; }
    descs.clear();
    descs.reserve(batch.ops.size());
    for (const auto& item : batch.ops) {
        const auto local_address = detail::PtrToU64(item.local_addr);
        if (!ValidateMemory(local_address, item.length) || item.remote_addr == 0) {
            return Status::InvalidArgument;
        }
        descs.push_back(hixl::TransferOpDesc{
            static_cast<uintptr_t>(local_address),
            static_cast<uintptr_t>(item.remote_addr),
            static_cast<size_t>(item.length),
        });
    }
    return Status::Ok;
}

Status HixlTransport::ExecuteSync(const Operation& batch)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }

    const auto peer_it = peers_.find(batch.target_manager);
    if (peer_it == peers_.end()) { return Status::Failed; }
    if (!peer_it->second.connected) { return Status::Failed; }
    auto& peer_state = peer_it->second;

    std::vector<hixl::TransferOpDesc> descs;
    const auto build_status = BuildTransfer(batch, descs);
    if (build_status != Status::Ok) { return build_status; }

    const auto op = batch.opcode == Opcode::Read ? hixl::READ : hixl::WRITE;
    const auto transfer_status =
        hixl_.TransferSync(peer_state.remote_engine.c_str(), op, descs, transfer_timeout_ms_);
    if (transfer_status != hixl::SUCCESS) {
        UC_ERROR(
            "transport hixl operation failed: TransferSync(\"{}\", ops={}, timeout_ms={}) returned "
            "{}",
            peer_state.remote_engine, descs.size(), transfer_timeout_ms_,
            static_cast<int>(transfer_status));
        peer_state.connected = false;
        const auto disconnect_status =
            hixl_.Disconnect(peer_state.remote_engine.c_str(), connect_timeout_ms_);
        if (disconnect_status != hixl::SUCCESS) {
            UC_ERROR("transport hixl disconnect failed: Disconnect(\"{}\") returned {}",
                     peer_state.remote_engine, static_cast<int>(disconnect_status));
        }
        return Status::Failed;
    }
    return Status::Ok;
}

Status HixlTransport::ExecuteAsync(const Operation& batch, TransferHandle& handle)
{
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }
    handle = kInvalidTransferHandle;

    const auto peer_it = peers_.find(batch.target_manager);
    if (peer_it == peers_.end()) { return Status::Failed; }
    if (!peer_it->second.connected) { return Status::Failed; }
    auto& peer_state = peer_it->second;

    std::vector<hixl::TransferOpDesc> descs;
    const auto build_status = BuildTransfer(batch, descs);
    if (build_status != Status::Ok) { return build_status; }

    hixl::TransferReq request = nullptr;
    hixl::TransferArgs args;
    const auto op = batch.opcode == Opcode::Read ? hixl::READ : hixl::WRITE;
    const auto transfer_status =
        hixl_.TransferAsync(peer_state.remote_engine.c_str(), op, descs, args, request);
    if (transfer_status != hixl::SUCCESS || request == nullptr) {
        UC_ERROR(
            "transport hixl async operation failed: TransferAsync(\"{}\", ops={}) returned "
            "{} request={}",
            peer_state.remote_engine, descs.size(), static_cast<int>(transfer_status), request);
        peer_state.connected = false;
        const auto disconnect_status =
            hixl_.Disconnect(peer_state.remote_engine.c_str(), connect_timeout_ms_);
        if (disconnect_status != hixl::SUCCESS) {
            UC_ERROR("transport hixl disconnect failed: Disconnect(\"{}\") returned {}",
                     peer_state.remote_engine, static_cast<int>(disconnect_status));
        }
        return Status::Failed;
    }

    handle = next_transfer_handle_++;
    if (handle == kInvalidTransferHandle) { handle = next_transfer_handle_++; }
    pending_transfers_.emplace(handle, request);
    return Status::Ok;
}

Status HixlTransport::GetStatus(TransferHandle handle, TransferStatus& status)
{
    status = TransferStatus::Failed;
    if (handle == kInvalidTransferHandle) { return Status::InvalidArgument; }
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    WithAclRuntimeContext context_guard(context_);
    if (context_guard.status() != Status::Ok) { return context_guard.status(); }
    const auto it = pending_transfers_.find(handle);
    if (it == pending_transfers_.end()) { return Status::Failed; }

    hixl::TransferStatus transfer_status = hixl::TransferStatus::WAITING;
    const auto query_status = hixl_.GetTransferStatus(it->second, transfer_status);
    if (query_status != hixl::SUCCESS) {
        UC_ERROR("transport hixl get transfer status failed: req={} returned {}", it->second,
                 static_cast<int>(query_status));
        status = TransferStatus::Failed;
        pending_transfers_.erase(it);
        return Status::Failed;
    }
    status = ConvertTransferStatus(transfer_status);
    if (status != TransferStatus::Waiting) { pending_transfers_.erase(it); }
    return Status::Ok;
}

}  // namespace transport
