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
#include "hccp_kv_over_roce_trans_provider.h"

#ifdef UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER

#include <dlfcn.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "hbm_slot_allocator.h"
#include "hccp_acl_context.h"
#include "hccp_kv_over_roce_abi.h"
#include "logger.h"
#include "parser_common.h"

namespace kv {
namespace {

constexpr const char* kProviderSignature = "UCM_ASU_HCCP_KV_OVER_ROCE_PROVIDER_ABI_V1";
constexpr const char* kDefaultLibraryPath = "libhccp_kv_over_roce.so";
constexpr const char* kLibraryPathEnv = "UCM_HCCP_KV_ROCE_LIB_PATH";

// Every connection owns one HBM slot. The slot receives the operator QPC and is
// later handed to the operator IO path through Send.
static_assert(sizeof(HccpKvRoceQpc) <= kHccpKvRoceConnSlotBytes,
              "one connection QPC must fit into one HBM slot");

using CreateConnectionFn = decltype(&::CreateConnection);
using DeleteConnectionsFn = decltype(&::DeleteConnections);
using RegisterMemoryFn = decltype(&::RegisterMemory);
using GetAbiVersionFn = decltype(&::GetAbiVersion);
using UnregisterMemoryFn = decltype(&::UnregisterMemory);
using SendFn = decltype(&::Send);
using AllocThreadFn = decltype(&::AllocThread);
using FreeThreadFn = decltype(&::FreeThread);
using GetServerCapabilitiesFn = decltype(&::GetServerCapabilities);

StatusCode MapAbiStatusCode(std::int32_t code)
{
    switch (code) {
        case HCCP_KV_ROCE_OK:
            return StatusCode::OK;
        case HCCP_KV_ROCE_INVALID_ARGUMENT:
            return StatusCode::INVALID_ARGUMENT;
        case HCCP_KV_ROCE_UNSUPPORTED:
            return StatusCode::UNSUPPORTED;
        case HCCP_KV_ROCE_RESOURCE_BUSY:
            return StatusCode::RESOURCE_BUSY;
        case HCCP_KV_ROCE_CONNECTION_ERROR:
            return StatusCode::CONNECTION_ERROR;
        case HCCP_KV_ROCE_IO_ERROR:
            return StatusCode::IO_ERROR;
        case HCCP_KV_ROCE_BUFFER_NOT_REGISTERED:
            return StatusCode::BUFFER_NOT_REGISTERED;
        case HCCP_KV_ROCE_INTERNAL_ERROR:
            return StatusCode::INTERNAL_ERROR;
        default:
            return StatusCode::INTERNAL_ERROR;
    }
}

Status AbiError(const char* operation, std::int32_t code, StatusCode fallback)
{
    auto mapped = MapAbiStatusCode(code);
    if (mapped == StatusCode::OK) { mapped = fallback; }
    std::string message = std::string("HccpKvOverRoceTransProvider: ") + operation +
                          " failed code=" + std::to_string(code);
    KV_ERROR("{}", message);
    return Status::Error(mapped, std::move(message));
}

std::string ResolveLibraryPath(const TransportConfig& config)
{
    auto configured = GetConfigAttr(config, {"hccp_kv_roce_lib_path", "hccp_lib_path"});
    if (!configured.empty()) { return configured; }

    if (const char* fromEnv = std::getenv(kLibraryPathEnv);
        fromEnv != nullptr && fromEnv[0] != '\0') {
        return fromEnv;
    }
    return kDefaultLibraryPath;
}

std::uint32_t ResolveSlotCapacity(const TransportConfig& config)
{
    auto configured =
        GetConfigAttr(config, {"hccp_kv_roce_max_connections", "hccp_max_connections"});
    if (configured.empty()) { return kHccpKvRoceDefaultSlotCapacity; }
    return ParseConfigUint32(configured, kHccpKvRoceDefaultSlotCapacity);
}

int32_t ToAbiMemoryType(TransProvider::MemType memoryType)
{
    return memoryType == TransProvider::MemType::MEM_HOST ? HCCP_KV_ROCE_MEM_HOST
                                                          : HCCP_KV_ROCE_MEM_DEVICE;
}

// Resolves and owns the external library. All function pointers are resolved
// once; optional entries stay null when the library does not export them.
struct AbiApi {
    void* handle{nullptr};
    CreateConnectionFn createConnection{nullptr};
    DeleteConnectionsFn deleteConnections{nullptr};
    RegisterMemoryFn registerMemory{nullptr};
    UnregisterMemoryFn unregisterMemory{nullptr};
    SendFn send{nullptr};
    AllocThreadFn allocThread{nullptr};
    FreeThreadFn freeThread{nullptr};
    GetServerCapabilitiesFn getServerCapabilities{nullptr};

    AbiApi() = default;
    AbiApi(const AbiApi&) = delete;
    AbiApi& operator=(const AbiApi&) = delete;
    AbiApi(AbiApi&&) = delete;
    AbiApi& operator=(AbiApi&&) = delete;

    ~AbiApi()
    {
        if (handle != nullptr) {
            (void)dlclose(handle);
            handle = nullptr;
        }
    }

    Status Load(const std::string& libraryPath)
    {
        if (handle != nullptr) { return Status::OK(); }

        void* library = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            const char* error = dlerror();
            return Status::Error(StatusCode::NOT_FOUND,
                                 "HccpKvOverRoceTransProvider: dlopen " + libraryPath +
                                     " failed: " + (error == nullptr ? "unknown" : error));
        }

        auto resolve = [&](const char* symbol) -> void* {
            (void)dlerror();
            return dlsym(library, symbol);
        };

        createConnection =
            reinterpret_cast<CreateConnectionFn>(resolve("CreateConnection"));
        deleteConnections =
            reinterpret_cast<DeleteConnectionsFn>(resolve("DeleteConnections"));
        registerMemory = reinterpret_cast<RegisterMemoryFn>(resolve("RegisterMemory"));
        if (createConnection == nullptr || deleteConnections == nullptr ||
            registerMemory == nullptr) {
            const char* missing = createConnection == nullptr
                                      ? "CreateConnection"
                                      : (deleteConnections == nullptr ? "DeleteConnections"
                                                                      : "RegisterMemory");
            (void)dlclose(library);
            return Status::Error(StatusCode::NOT_FOUND,
                                 "HccpKvOverRoceTransProvider: " + libraryPath +
                                     " does not export required symbol " + missing);
        }

        // Optional symbols: a missing entry keeps the provider fallback.
        unregisterMemory = reinterpret_cast<UnregisterMemoryFn>(resolve("UnregisterMemory"));
        send = reinterpret_cast<SendFn>(resolve("Send"));
        allocThread = reinterpret_cast<AllocThreadFn>(resolve("AllocThread"));
        freeThread = reinterpret_cast<FreeThreadFn>(resolve("FreeThread"));
        getServerCapabilities =
            reinterpret_cast<GetServerCapabilitiesFn>(resolve("GetServerCapabilities"));

        const auto getAbiVersion = reinterpret_cast<GetAbiVersionFn>(resolve("GetAbiVersion"));
        if (getAbiVersion != nullptr) {
            const auto version = getAbiVersion();
            if (version != HCCP_KV_ROCE_ABI_VERSION) {
                (void)dlclose(library);
                return Status::Error(
                    StatusCode::INTERNAL_ERROR,
                    "HccpKvOverRoceTransProvider: " + libraryPath + " ABI version " +
                        std::to_string(version) + " does not match expected version " +
                        std::to_string(HCCP_KV_ROCE_ABI_VERSION));
            }
        }

        handle = library;
        KV_INFO(
            "HccpKvOverRoceTransProvider: external library loaded path={} optional_symbols="
            "unregister={} send={} alloc_thread={} free_thread={} capabilities={}",
            libraryPath, unregisterMemory != nullptr ? 1 : 0, send != nullptr ? 1 : 0,
            allocThread != nullptr ? 1 : 0, freeThread != nullptr ? 1 : 0,
            getServerCapabilities != nullptr ? 1 : 0);
        return Status::OK();
    }
};

void ReleaseSlots(HbmSlotAllocator& allocator, const std::vector<std::uint32_t>& slots)
{
    for (const auto slotId : slots) {
        const auto status = allocator.Release(slotId);
        if (!status.ok()) {
            KV_WARN("HccpKvOverRoceTransProvider: failed to release HBM slot_id={} message={}",
                    slotId, status.message);
        }
    }
}

}  // namespace

struct HccpKvOverRoceTransProvider::Impl {
    struct ConnectionRecord {
        void* externalHandle{nullptr};
        std::uint32_t slotId{0};
        // Provider-owned QPC storage. Its address is handed to the library as
        // the connection handle and the payload is copied into the HBM slot.
        HccpKvRoceQpc qpc{};
    };

    using ConnectionMap = std::unordered_map<ConnectionHandle, std::shared_ptr<ConnectionRecord>>;

    // Provider-owned MR bookkeeping. The external library only reports the
    // registered memory info; the handle that crosses the TransProvider API is
    // allocated here and never leaves the provider.
    struct MemoryRecord {
        MRHandle handle{kInvalidMRHandle};
        HccpKvRoceMemoryInfo info{};
        TransProvider::MemType memoryType{TransProvider::MemType::MEM_DEVICE};
        std::uintptr_t localAddr{0};
        // True for records created by RegisterMemory, false for BindMemory
        // records that reuse an MR registered by another provider.
        bool registeredWithOperator{false};
    };

    using MemoryMap = std::unordered_map<MRHandle, std::shared_ptr<MemoryRecord>>;

    explicit Impl(const TransportConfig& config)
        : libraryPath(ResolveLibraryPath(config)), slotCapacity(ResolveSlotCapacity(config))
    {
        if (slotCapacity == 0) {
            throw std::runtime_error(
                "HccpKvOverRoceTransProvider: hccp_kv_roce_max_connections must be greater than "
                "zero");
        }

        const auto loadStatus = api.Load(libraryPath);
        if (!loadStatus.ok()) { throw std::runtime_error(loadStatus.message); }

        const auto bindingStatus = ResolveHccpAclBinding(config, binding);
        if (!bindingStatus.ok()) { throw std::runtime_error(bindingStatus.message); }

        ScopedHccpAclContext aclScope("Init", binding);
        if (!aclScope.status().ok()) { throw std::runtime_error(aclScope.status().message); }

        allocator = std::make_unique<HbmSlotAllocator>(
            binding.deviceId, kHccpKvRoceConnSlotBytes, slotCapacity);
        const auto initStatus = allocator->Init();
        if (!initStatus.ok()) { throw std::runtime_error(initStatus.message); }

        KV_INFO(
            "HCCP_KV_ROCE_TRANSPORT_PROVIDER_SIGNATURE={} pid={} node_id={} device_id={} "
            "device_source={} provider_context={} library={} slot_bytes={} slot_capacity={}",
            kProviderSignature, static_cast<long>(::getpid()), config.nodeId, binding.deviceId,
            binding.source, static_cast<const void*>(binding.context), libraryPath,
            allocator->slotBytes(), allocator->capacity());
    }

    ~Impl()
    {
        if (allocator == nullptr) { return; }

        // Free the HBM region on the provider's own device/context.
        ScopedHccpAclContext aclScope("Cleanup", binding);
        if (!aclScope.status().ok()) {
            KV_WARN(
                "HccpKvOverRoceTransProvider: freeing HBM without device binding: {}",
                aclScope.status().message);
        }
        allocator.reset();
    }

    AbiApi api;
    std::string libraryPath;
    std::uint32_t slotCapacity{kHccpKvRoceDefaultSlotCapacity};
    HccpAclBinding binding;
    std::unique_ptr<HbmSlotAllocator> allocator;

    mutable std::shared_mutex connectionMu;
    ConnectionMap connections;

    mutable std::shared_mutex memoryMu;
    MemoryMap memories;
    std::uint64_t nextMemoryHandle{1};

    // Caller must hold memoryMu exclusively. Handles start at one so that zero
    // always means "invalid".
    MRHandle NextMemoryHandleLocked()
    {
        MRHandle handle = nextMemoryHandle;
        if (handle == kInvalidMRHandle) { handle = 1; }
        nextMemoryHandle = handle + 1;
        return handle;
    }
};

HccpKvOverRoceTransProvider::HccpKvOverRoceTransProvider(const TransportConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

HccpKvOverRoceTransProvider::~HccpKvOverRoceTransProvider()
{
    if (!impl_) { return; }

    ScopedHccpAclContext aclScope("Destroy", impl_->binding);
    if (!aclScope.status().ok()) {
        KV_WARN("HccpKvOverRoceTransProvider: cleanup continues without ACL binding: {}",
                aclScope.status().message);
    }

    std::vector<ConnectionHandle> connectionHandles;
    {
        std::shared_lock<std::shared_mutex> lock(impl_->connectionMu);
        connectionHandles.reserve(impl_->connections.size());
        for (const auto& item : impl_->connections) { connectionHandles.push_back(item.first); }
    }
    if (!connectionHandles.empty()) {
        const auto statuses = DeleteConnections(connectionHandles);
        for (std::size_t index = 0; index < statuses.size(); ++index) {
            if (!statuses[index].ok()) {
                KV_WARN(
                    "HccpKvOverRoceTransProvider: connection cleanup failed index={} message={}",
                    index, statuses[index].message);
            }
        }
    }

    std::vector<HccpKvRoceMemoryInfo> memoryInfos;
    {
        std::shared_lock<std::shared_mutex> lock(impl_->memoryMu);
        memoryInfos.reserve(impl_->memories.size());
        for (const auto& item : impl_->memories) {
            const auto& record = item.second;
            if (record == nullptr || !record->registeredWithOperator) { continue; }
            memoryInfos.push_back(record->info);
        }
    }
    if (!memoryInfos.empty() && impl_->api.unregisterMemory != nullptr) {
        const auto ret = impl_->api.unregisterMemory(
            memoryInfos.data(), static_cast<std::uint32_t>(memoryInfos.size()));
        if (ret != 0) {
            KV_WARN("HccpKvOverRoceTransProvider: memory cleanup failed count={} code={}",
                    memoryInfos.size(), ret);
        }
    }
}

Status HccpKvOverRoceTransProvider::CreateConnection(
    const std::string& localIp, const std::string& remoteIp, std::uint32_t port,
    std::uint32_t qpNum, std::uint32_t timeout,
    std::vector<ConnectionHandle>& connectionHandles)
{
    connectionHandles.clear();
    if (!impl_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider is not initialized");
    }
    ScopedHccpAclContext aclScope("CreateConnection", impl_->binding);
    if (!aclScope.status().ok()) { return aclScope.status(); }

    if (qpNum == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HccpKvOverRoceTransProvider: qp_num must be greater than zero");
    }
    if (remoteIp.empty()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HccpKvOverRoceTransProvider: remote IP is required");
    }
    if (qpNum > impl_->slotCapacity) {
        return Status::Error(StatusCode::RESOURCE_BUSY,
                             "HccpKvOverRoceTransProvider: requested qp_num " +
                                 std::to_string(qpNum) + " exceeds slot capacity " +
                                 std::to_string(impl_->slotCapacity));
    }

    std::vector<std::uint32_t> slots;
    slots.reserve(qpNum);
    for (std::uint32_t index = 0; index < qpNum; ++index) {
        std::uint32_t slotId = 0;
        const auto status = impl_->allocator->Acquire(slotId);
        if (!status.ok()) {
            ReleaseSlots(*impl_->allocator, slots);
            return Status::Error(status.code,
                                 "HccpKvOverRoceTransProvider: failed to allocate HBM "
                                 "connection slot: " +
                                     status.message);
        }
        slots.push_back(slotId);
    }

    // The library fills one provider-owned QPC per connection. The addresses are
    // stable for the connection lifetime and are also the handles passed to
    // DeleteConnections.
    std::vector<std::shared_ptr<Impl::ConnectionRecord>> records;
    records.reserve(qpNum);
    std::vector<void*> abiHandles(qpNum, nullptr);
    for (std::uint32_t index = 0; index < qpNum; ++index) {
        auto record = std::make_shared<Impl::ConnectionRecord>();
        record->slotId = slots[index];
        abiHandles[index] = &record->qpc;
        records.push_back(std::move(record));
    }

    auto deleteAbiHandles = [&](std::uint32_t handleNum) {
        const auto limit = std::min<std::size_t>(handleNum, abiHandles.size());
        std::vector<void*> created;
        created.reserve(limit);
        for (std::size_t index = 0; index < limit; ++index) {
            if (abiHandles[index] != nullptr) { created.push_back(abiHandles[index]); }
        }
        if (created.empty()) { return; }
        (void)impl_->api.deleteConnections(created.data(),
                                           static_cast<std::uint32_t>(created.size()));
    };

    KV_INFO(
        "HccpKvOverRoceTransProvider: CreateConnection start local_addr={} remote_addr={} "
        "remote_port={} qp_num={} timeout_ms={}",
        localIp, remoteIp, port, qpNum, timeout);

    std::uint32_t createdNum = 0;
    const auto ret = impl_->api.createConnection(localIp.c_str(), remoteIp.c_str(), port, qpNum,
                                                 timeout, abiHandles.data(), &createdNum);
    if (ret != 0) {
        deleteAbiHandles(createdNum);
        ReleaseSlots(*impl_->allocator, slots);
        return AbiError("CreateConnection", ret, StatusCode::CONNECTION_ERROR);
    }
    if (createdNum > qpNum) {
        // Never trust the array beyond qpNum even if the library reports more.
        deleteAbiHandles(qpNum);
        ReleaseSlots(*impl_->allocator, slots);
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: CreateConnection reported " +
                                 std::to_string(createdNum) + " handles for qp_num " +
                                 std::to_string(qpNum));
    }
    if (createdNum != qpNum) {
        deleteAbiHandles(createdNum);
        ReleaseSlots(*impl_->allocator, slots);
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: CreateConnection created " +
                                 std::to_string(createdNum) + " of " + std::to_string(qpNum) +
                                 " connections");
    }

    for (std::uint32_t index = 0; index < qpNum; ++index) {
        if (abiHandles[index] == nullptr) {
            deleteAbiHandles(qpNum);
            ReleaseSlots(*impl_->allocator, slots);
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "HccpKvOverRoceTransProvider: CreateConnection returned a null "
                                 "connection handle");
        }
    }

    for (std::uint32_t index = 0; index < qpNum; ++index) {
        const auto status = impl_->allocator->Write(records[index]->slotId, abiHandles[index],
                                                    sizeof(HccpKvRoceQpc));
        if (!status.ok()) {
            deleteAbiHandles(qpNum);
            for (std::uint32_t written = 0; written <= index; ++written) {
                (void)impl_->allocator->Clear(records[written]->slotId);
            }
            ReleaseSlots(*impl_->allocator, slots);
            return status;
        }
        records[index]->externalHandle = abiHandles[index];
    }

    {
        std::unique_lock<std::shared_mutex> lock(impl_->connectionMu);
        for (const auto& record : records) {
            ConnectionHandle handle = record.get();
            impl_->connections.emplace(handle, record);
            connectionHandles.push_back(handle);
        }
    }

    KV_INFO(
        "HccpKvOverRoceTransProvider: CreateConnection complete remote_addr={} remote_port={} "
        "handles={}",
        remoteIp, port, connectionHandles.size());
    return Status::OK();
}

std::vector<Status> HccpKvOverRoceTransProvider::DeleteConnections(
    const std::vector<ConnectionHandle>& connectionHandles)
{
    std::vector<Status> results(connectionHandles.size(), Status::OK());
    if (!impl_) {
        std::fill(results.begin(), results.end(),
                  Status::Error(StatusCode::NOT_INITIALIZED,
                                "HccpKvOverRoceTransProvider is not initialized"));
        return results;
    }
    if (connectionHandles.empty()) { return results; }

    ScopedHccpAclContext aclScope("DeleteConnections", impl_->binding);
    if (!aclScope.status().ok()) {
        std::fill(results.begin(), results.end(), aclScope.status());
        return results;
    }

    std::vector<std::shared_ptr<Impl::ConnectionRecord>> removed(connectionHandles.size());
    std::vector<void*> externalHandles;
    std::vector<std::size_t> externalIndex;
    externalHandles.reserve(connectionHandles.size());
    externalIndex.reserve(connectionHandles.size());

    for (std::size_t index = 0; index < connectionHandles.size(); ++index) {
        auto* record = static_cast<Impl::ConnectionRecord*>(connectionHandles[index]);
        if (record == nullptr) {
            results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                           "HccpKvOverRoceTransProvider: invalid connection "
                                           "handle");
            continue;
        }

        std::shared_ptr<Impl::ConnectionRecord> owner;
        {
            std::unique_lock<std::shared_mutex> lock(impl_->connectionMu);
            const auto iter = impl_->connections.find(record);
            if (iter == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: invalid connection "
                                               "handle");
                continue;
            }
            owner = std::move(iter->second);
            impl_->connections.erase(iter);
        }
        externalHandles.push_back(owner->externalHandle);
        externalIndex.push_back(index);
        removed[index] = std::move(owner);
    }

    if (externalHandles.empty()) { return results; }

    const auto ret = impl_->api.deleteConnections(
        externalHandles.data(), static_cast<std::uint32_t>(externalHandles.size()));
    if (ret != 0) {
        const auto error = AbiError("DeleteConnections", ret, StatusCode::CONNECTION_ERROR);
        std::unique_lock<std::shared_mutex> lock(impl_->connectionMu);
        for (const auto index : externalIndex) {
            // Read the key before moving the owner: argument evaluation order is
            // unspecified, and a moved-from shared_ptr is null.
            auto record = std::move(removed[index]);
            auto* handle = record.get();
            impl_->connections.emplace(handle, std::move(record));
        }
        for (const auto index : externalIndex) { results[index] = error; }
        return results;
    }

    for (const auto index : externalIndex) {
        const auto& record = removed[index];
        const auto clearStatus = impl_->allocator->Clear(record->slotId);
        if (!clearStatus.ok()) {
            KV_WARN(
                "HccpKvOverRoceTransProvider: failed to clear HBM slot_id={} message={}",
                record->slotId, clearStatus.message);
        }
        const auto releaseStatus = impl_->allocator->Release(record->slotId);
        if (!releaseStatus.ok()) {
            KV_WARN(
                "HccpKvOverRoceTransProvider: failed to release HBM slot_id={} message={}",
                record->slotId, releaseStatus.message);
        }
        results[index] = Status::OK();
    }

    KV_INFO("HccpKvOverRoceTransProvider: DeleteConnections complete requested={} deleted={}",
            connectionHandles.size(), externalHandles.size());
    return results;
}

Status HccpKvOverRoceTransProvider::GetServerCapabilities(
    ConnectionHandle connectionHandle, ServerKvCapabilities& capabilities)
{
    capabilities = {};
    if (!impl_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider is not initialized");
    }
    if (impl_->api.getServerCapabilities == nullptr) {
        return Status::Error(StatusCode::UNSUPPORTED,
                             "HccpKvOverRoceTransProvider: GetServerCapabilities is not "
                             "exported by the external library");
    }

    ScopedHccpAclContext aclScope("GetServerCapabilities", impl_->binding);
    if (!aclScope.status().ok()) { return aclScope.status(); }

    HccpKvRoceServerCapabilities abiCapabilities{};
    std::int32_t ret = 0;
    {
        // Hold the connection map lock across the ABI call so a concurrent
        // DeleteConnections cannot recycle the external handle mid-query.
        std::shared_lock<std::shared_mutex> lock(impl_->connectionMu);
        if (connectionHandle == nullptr) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: invalid connection handle");
        }
        const auto iter = impl_->connections.find(connectionHandle);
        if (iter == impl_->connections.end()) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: invalid connection handle");
        }
        ret = impl_->api.getServerCapabilities(iter->second->externalHandle, &abiCapabilities);
    }
    if (ret != 0) { return AbiError("GetServerCapabilities", ret, StatusCode::INTERNAL_ERROR); }

    capabilities.queueNum = abiCapabilities.queueNum;
    capabilities.ioQueueDepth = abiCapabilities.ioQueueDepth;
    capabilities.ioQueueKeyConcurrency = abiCapabilities.ioQueueKeyConcurrency;
    capabilities.connectionKeyConcurrency = abiCapabilities.connectionKeyConcurrency;
    capabilities.singleValueMaxBytes = abiCapabilities.singleValueMaxBytes;
    capabilities.batchValueMaxBytes = abiCapabilities.batchValueMaxBytes;
    capabilities.batchStoreKeys = abiCapabilities.batchStoreKeys;
    capabilities.batchLoadKeys = abiCapabilities.batchLoadKeys;
    capabilities.deleteKeys = abiCapabilities.deleteKeys;
    capabilities.queryKeys = abiCapabilities.queryKeys;
    capabilities.keyLength = abiCapabilities.keyLength;
    capabilities.kvCapabilities = abiCapabilities.kvCapabilities;
    return Status::OK();
}

std::vector<Status> HccpKvOverRoceTransProvider::Send(
    const std::vector<SendIoBatch>& ioBatches, std::uint32_t kernelCount, std::uint32_t quietCount)
{
    std::vector<Status> results(ioBatches.size(), Status::OK());
    if (!impl_) {
        std::fill(results.begin(), results.end(),
                  Status::Error(StatusCode::NOT_INITIALIZED,
                                "HccpKvOverRoceTransProvider is not initialized"));
        return results;
    }
    if (ioBatches.empty()) { return results; }
    if (impl_->api.send == nullptr) {
        std::fill(results.begin(), results.end(),
                  Status::Error(StatusCode::UNSUPPORTED,
                                "HccpKvOverRoceTransProvider: Send is not exported by the "
                                "external library"));
        return results;
    }

    std::vector<HccpKvRoceSendIoBatch> abiBatches;
    std::vector<std::size_t> abiIndex;
    std::vector<std::int32_t> statuses;
    abiBatches.reserve(ioBatches.size());
    abiIndex.reserve(ioBatches.size());
    std::int32_t ret = 0;

    ScopedHccpAclContext aclScope("Send", impl_->binding);
    if (!aclScope.status().ok()) {
        std::fill(results.begin(), results.end(), aclScope.status());
        return results;
    }

    {
        // Keep the connection map locked for the whole ABI call so a concurrent
        // DeleteConnections cannot recycle a connection slot while Send uses it.
        std::shared_lock<std::shared_mutex> lock(impl_->connectionMu);
        for (std::size_t index = 0; index < ioBatches.size(); ++index) {
            auto* record =
                static_cast<Impl::ConnectionRecord*>(ioBatches[index].connectionHandle);
            if (record == nullptr) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: invalid connection "
                                               "handle");
                continue;
            }
            const auto iter = impl_->connections.find(record);
            if (iter == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: invalid connection "
                                               "handle");
                continue;
            }
            // The operator reads the QPC out of the HBM slot, so the batch carries
            // the slot address instead of a host-side handle.
            auto* slotAddress = impl_->allocator->SlotAddress(iter->second->slotId);
            if (slotAddress == nullptr) {
                results[index] = Status::Error(StatusCode::INTERNAL_ERROR,
                                               "HccpKvOverRoceTransProvider: connection slot is "
                                               "not initialized");
                continue;
            }
            abiBatches.push_back({slotAddress, ioBatches[index].sendBuffer,
                                  ioBatches[index].flagBuffer, ioBatches[index].len});
            abiIndex.push_back(index);
        }
        if (abiBatches.empty()) { return results; }

        statuses.assign(abiBatches.size(), 0);
        ret = impl_->api.send(abiBatches.data(), static_cast<std::uint32_t>(abiBatches.size()),
                              kernelCount, quietCount, statuses.data());
    }
    if (ret != 0) {
        const auto error = AbiError("Send", ret, StatusCode::IO_ERROR);
        for (const auto index : abiIndex) { results[index] = error; }
        return results;
    }
    for (std::size_t index = 0; index < statuses.size(); ++index) {
        if (statuses[index] == 0) { continue; }
        results[abiIndex[index]] = AbiError("Send", statuses[index], StatusCode::IO_ERROR);
    }
    return results;
}

Status HccpKvOverRoceTransProvider::RegisterMemory(
    const std::vector<RegisterMemoryDesc>& memoryDescs, std::vector<MRHandle>& mrHandles)
{
    mrHandles.clear();
    if (!impl_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider is not initialized");
    }
    if (memoryDescs.empty()) { return Status::OK(); }

    ScopedHccpAclContext aclScope("RegisterMemory", impl_->binding);
    if (!aclScope.status().ok()) { return aclScope.status(); }

    std::vector<HccpKvRoceRegisterMemoryDesc> abiDescs;
    abiDescs.reserve(memoryDescs.size());
    for (const auto& desc : memoryDescs) {
        // The HCCP data path only handles device memory: host buffers cannot be
        // addressed by the operator, so reject them instead of registering MRs
        // that would never be usable.
        if (desc.memoryType != TransProvider::MemType::MEM_DEVICE) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: RegisterMemory only accepts "
                                 "device memory");
        }
        if (desc.addr == 0 || desc.size == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: RegisterMemory requires a "
                                 "non-zero address and size");
        }
        abiDescs.push_back({ToAbiMemoryType(desc.memoryType), static_cast<std::uint64_t>(desc.addr),
                            static_cast<std::uint64_t>(desc.size),
                            static_cast<std::uint64_t>(desc.localAddr)});
    }

    // The operator only registers the MRs and reports addr/size/access/rkey. The
    // provider owns the MR handle space and keeps the handle-to-info mapping.
    std::vector<HccpKvRoceMemoryInfo> memoryInfos(memoryDescs.size());
    const auto ret = impl_->api.registerMemory(
        abiDescs.data(), static_cast<std::uint32_t>(abiDescs.size()), memoryInfos.data());

    auto rollback = [&](const char* reason) {
        std::vector<HccpKvRoceMemoryInfo> registered;
        registered.reserve(memoryInfos.size());
        for (const auto& info : memoryInfos) {
            if (info.rkey != 0) { registered.push_back(info); }
        }
        if (registered.empty()) { return; }
        if (impl_->api.unregisterMemory == nullptr) {
            KV_WARN(
                "HccpKvOverRoceTransProvider: RegisterMemory rollback skipped ({}); external "
                "library does not export UnregisterMemory",
                reason);
            return;
        }
        const auto rollbackRet = impl_->api.unregisterMemory(
            registered.data(), static_cast<std::uint32_t>(registered.size()));
        if (rollbackRet != 0) {
            KV_WARN("HccpKvOverRoceTransProvider: RegisterMemory rollback failed count={} code={}",
                    registered.size(), rollbackRet);
        }
    };

    if (ret != 0) {
        rollback("operator registration failed");
        return AbiError("RegisterMemory", ret, StatusCode::BUFFER_NOT_SUPPORTED);
    }

    for (std::size_t index = 0; index < memoryInfos.size(); ++index) {
        if (memoryInfos[index].rkey != 0) { continue; }
        rollback("zero rkey");
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: RegisterMemory returned a zero "
                             "token id at index " +
                                 std::to_string(index));
    }

    {
        std::unique_lock<std::shared_mutex> lock(impl_->memoryMu);
        std::vector<std::shared_ptr<Impl::MemoryRecord>> records;
        records.reserve(memoryDescs.size());
        for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
            auto record = std::make_shared<Impl::MemoryRecord>();
            record->handle = impl_->NextMemoryHandleLocked();
            record->info = memoryInfos[index];
            record->memoryType = memoryDescs[index].memoryType;
            record->localAddr = memoryDescs[index].localAddr;
            record->registeredWithOperator = true;
            records.push_back(std::move(record));
        }
        for (const auto& record : records) {
            impl_->memories.emplace(record->handle, record);
            mrHandles.push_back(record->handle);
        }
    }

    KV_INFO(
        "HccpKvOverRoceTransProvider: RegisterMemory complete count={} first_handle={} "
        "last_handle={} first_token_id={}",
        mrHandles.size(), mrHandles.front(), mrHandles.back(), memoryInfos.front().rkey);
    return Status::OK();
}

Status HccpKvOverRoceTransProvider::BindMemory(
    const std::vector<BindMemoryDesc>& memoryDescs, std::vector<MRHandle>& mrHandles)
{
    mrHandles.clear();
    if (!impl_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider is not initialized");
    }
    if (memoryDescs.empty()) { return Status::OK(); }

    ScopedHccpAclContext aclScope("BindMemory", impl_->binding);
    if (!aclScope.status().ok()) { return aclScope.status(); }

    for (const auto& desc : memoryDescs) {
        if (desc.memoryType != TransProvider::MemType::MEM_DEVICE) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: BindMemory only accepts device "
                                 "memory");
        }
        if (desc.addr == 0 || desc.size == 0 || desc.tokenId == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: BindMemory requires a non-zero "
                                 "address, size and token id");
        }
    }

    // The MR was already registered by another provider, so binding must not call
    // the operator again. Only a provider-local record is created; UnregisterMemory
    // drops it without forwarding anything.
    {
        std::unique_lock<std::shared_mutex> lock(impl_->memoryMu);
        std::vector<std::shared_ptr<Impl::MemoryRecord>> records;
        records.reserve(memoryDescs.size());
        for (const auto& desc : memoryDescs) {
            auto record = std::make_shared<Impl::MemoryRecord>();
            record->handle = impl_->NextMemoryHandleLocked();
            record->info.addr = static_cast<std::uint64_t>(desc.addr);
            record->info.size = static_cast<std::uint64_t>(desc.size);
            record->info.access = 0;
            record->info.rkey = desc.tokenId;
            record->memoryType = desc.memoryType;
            record->localAddr = desc.addr;
            record->registeredWithOperator = false;
            records.push_back(std::move(record));
        }
        for (const auto& record : records) {
            impl_->memories.emplace(record->handle, record);
            mrHandles.push_back(record->handle);
        }
    }

    KV_INFO(
        "HccpKvOverRoceTransProvider: BindMemory complete count={} first_handle={} "
        "last_handle={} first_token_id={}",
        mrHandles.size(), mrHandles.front(), mrHandles.back(), memoryDescs.front().tokenId);
    return Status::OK();
}

std::vector<Status> HccpKvOverRoceTransProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& memoryDescs)
{
    std::vector<Status> results(memoryDescs.size(), Status::OK());
    if (!impl_) {
        std::fill(results.begin(), results.end(),
                  Status::Error(StatusCode::NOT_INITIALIZED,
                                "HccpKvOverRoceTransProvider is not initialized"));
        return results;
    }
    if (memoryDescs.empty()) { return results; }

    ScopedHccpAclContext aclScope("UnregisterMemory", impl_->binding);
    if (!aclScope.status().ok()) {
        std::fill(results.begin(), results.end(), aclScope.status());
        return results;
    }

    std::vector<std::shared_ptr<Impl::MemoryRecord>> removed(memoryDescs.size());
    std::vector<HccpKvRoceMemoryInfo> operatorInfos;
    std::vector<std::size_t> operatorIndices;
    std::unordered_set<MRHandle> seen;

    std::int32_t ret = 0;
    {
        std::unique_lock<std::shared_mutex> lock(impl_->memoryMu);
        for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
            const auto handle = memoryDescs[index].mrHandle;
            if (!seen.insert(handle).second) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: duplicate memory "
                                               "handle");
                continue;
            }
            const auto iter = impl_->memories.find(handle);
            if (iter == impl_->memories.end()) {
                results[index] = Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                                               "HccpKvOverRoceTransProvider: invalid memory "
                                               "handle");
                continue;
            }
            removed[index] = std::move(iter->second);
            impl_->memories.erase(iter);
            if (removed[index]->registeredWithOperator) {
                operatorInfos.push_back(removed[index]->info);
                operatorIndices.push_back(index);
            }
        }

        if (!operatorInfos.empty()) {
            if (impl_->api.unregisterMemory != nullptr) {
                ret = impl_->api.unregisterMemory(
                    operatorInfos.data(), static_cast<std::uint32_t>(operatorInfos.size()));
            } else {
                // The external library only exports the three required methods. Drop
                // the local records so the client can still release its buffers.
                KV_WARN(
                    "HccpKvOverRoceTransProvider: external library does not export "
                    "UnregisterMemory; dropping local records for {} memories",
                    operatorInfos.size());
            }
            if (ret != 0) {
                // Keep operator-owned records so a later call can retry; bind-only
                // records are local and stay released.
                for (const auto index : operatorIndices) {
                    auto record = std::move(removed[index]);
                    const auto handle = record->handle;
                    impl_->memories.emplace(handle, std::move(record));
                }
            }
        }
    }

    if (ret != 0) {
        const auto error = AbiError("UnregisterMemory", ret, StatusCode::INTERNAL_ERROR);
        for (const auto index : operatorIndices) { results[index] = error; }
    }
    return results;
}

Status HccpKvOverRoceTransProvider::AllocThread(
    std::uint32_t threadNum, const std::vector<std::uint32_t>& notifyNumPerThread,
    std::vector<ThreadHandle>& threads)
{
    if (!impl_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider is not initialized");
    }
    threads.clear();
    if (impl_->api.allocThread == nullptr) {
        // Matches the AICPU provider: the transport owns no per-thread resources.
        return Status::OK();
    }
    if (threadNum == 0) { return Status::OK(); }
    if (notifyNumPerThread.size() != threadNum) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HccpKvOverRoceTransProvider: notify count does not match thread "
                             "count");
    }

    ScopedHccpAclContext aclScope("AllocThread", impl_->binding);
    if (!aclScope.status().ok()) { return aclScope.status(); }

    threads.assign(threadNum, nullptr);
    const auto ret =
        impl_->api.allocThread(threadNum, notifyNumPerThread.data(), threads.data());
    if (ret != 0) {
        threads.clear();
        return AbiError("AllocThread", ret, StatusCode::INTERNAL_ERROR);
    }
    return Status::OK();
}

std::vector<Status> HccpKvOverRoceTransProvider::FreeThread(
    const std::vector<ThreadHandle>& threads)
{
    std::vector<Status> results(threads.size(), Status::OK());
    if (!impl_) {
        std::fill(results.begin(), results.end(),
                  Status::Error(StatusCode::NOT_INITIALIZED,
                                "HccpKvOverRoceTransProvider is not initialized"));
        return results;
    }
    if (threads.empty() || impl_->api.freeThread == nullptr) { return results; }

    ScopedHccpAclContext aclScope("FreeThread", impl_->binding);
    if (!aclScope.status().ok()) {
        std::fill(results.begin(), results.end(), aclScope.status());
        return results;
    }

    std::vector<std::int32_t> statuses(threads.size(), 0);
    const auto ret = impl_->api.freeThread(
        threads.data(), static_cast<std::uint32_t>(threads.size()), statuses.data());
    if (ret != 0) {
        const auto error = AbiError("FreeThread", ret, StatusCode::INTERNAL_ERROR);
        for (auto& result : results) { result = error; }
        return results;
    }
    for (std::size_t index = 0; index < statuses.size(); ++index) {
        if (statuses[index] == 0) { continue; }
        results[index] = AbiError("FreeThread", statuses[index], StatusCode::INTERNAL_ERROR);
    }
    return results;
}

Status HccpKvOverRoceTransProvider::GetMemTokenId(MRHandle mrHandle, std::uint32_t& tokenId)
{
    tokenId = 0;
    if (!impl_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider is not initialized");
    }

    // The token id was captured when the MR was registered or bound; no operator
    // call is needed here.
    std::shared_lock<std::shared_mutex> lock(impl_->memoryMu);
    const auto iter = impl_->memories.find(mrHandle);
    if (iter == impl_->memories.end()) {
        return Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                             "HccpKvOverRoceTransProvider: invalid memory handle");
    }
    tokenId = iter->second->info.rkey;
    return Status::OK();
}

}  // namespace kv

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
