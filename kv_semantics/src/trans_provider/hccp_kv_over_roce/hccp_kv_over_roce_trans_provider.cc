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
#include <cstring>
#include <limits>
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
#include "hccp_kv_over_roce_abi.h"
#include "logger.h"
#include "parser_common.h"

namespace kv {
namespace {

constexpr const char* kProviderSignature = "UCM_ASU_HCCP_KV_OVER_ROCE_PROVIDER_ABI_V1";
constexpr const char* kDefaultLibraryPath = "libhccp_kv_over_roce.so";
constexpr const char* kLibraryPathEnv = "UCM_HCCP_KV_ROCE_LIB_PATH";
constexpr std::uint64_t kConnInfoMagic = 0x484343504B565243ULL;  // "HCCPKVRC"
constexpr std::uint32_t kConnInfoVersion = 1U;
constexpr std::size_t kIpFieldBytes = 64;

// Provider-owned description of one live connection. The slot is fixed-size
// HBM; the concrete layout is intentionally local to this provider.
struct ConnInfo {
    std::uint64_t magic{0};
    std::uint32_t version{0};
    std::uint32_t slotId{0};
    std::uint32_t qpNum{0};
    std::uint32_t qpIndex{0};
    std::uint32_t remotePort{0};
    std::uint32_t reserved{0};
    std::uint64_t externalHandle{0};
    char localIp[kIpFieldBytes]{};
    char remoteIp[kIpFieldBytes]{};
};

static_assert(sizeof(ConnInfo) <= kHccpKvRoceConnInfoSlotBytes,
              "one connection info record must fit into one HBM slot");

using CreateConnectionFn = decltype(&::CreateConnection);
using DeleteConnectionsFn = decltype(&::DeleteConnections);
using RegisterMemoryFn = decltype(&::RegisterMemory);
using GetAbiVersionFn = decltype(&::GetAbiVersion);
using BindMemoryFn = decltype(&::BindMemory);
using UnregisterMemoryFn = decltype(&::UnregisterMemory);
using SendFn = decltype(&::Send);
using AllocThreadFn = decltype(&::AllocThread);
using FreeThreadFn = decltype(&::FreeThread);
using GetMemTokenIdFn = decltype(&::GetMemTokenId);
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

std::int32_t ResolveDeviceId(const TransportConfig& config)
{
    if (config.deviceId >= 0) { return config.deviceId; }

    auto configured = GetConfigAttr(config, {"device_id", "deviceId", "logical_device_id"});
    if (configured.empty()) { return -1; }
    const auto parsed = ParseConfigUint32(configured, 0);
    if (parsed > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        return -1;
    }
    return static_cast<std::int32_t>(parsed);
}

void CopyIp(char (&destination)[kIpFieldBytes], const std::string& ip)
{
    // Copy explicitly instead of snprintf so -Wformat-truncation never fires on
    // the fixed-size slot layout under -Werror.
    const auto bytes = std::min(ip.size(), kIpFieldBytes - 1);
    std::memcpy(destination, ip.data(), bytes);
    destination[bytes] = '\0';
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
    BindMemoryFn bindMemory{nullptr};
    UnregisterMemoryFn unregisterMemory{nullptr};
    SendFn send{nullptr};
    AllocThreadFn allocThread{nullptr};
    FreeThreadFn freeThread{nullptr};
    GetMemTokenIdFn getMemTokenId{nullptr};
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
        bindMemory = reinterpret_cast<BindMemoryFn>(resolve("BindMemory"));
        unregisterMemory = reinterpret_cast<UnregisterMemoryFn>(resolve("UnregisterMemory"));
        send = reinterpret_cast<SendFn>(resolve("Send"));
        allocThread = reinterpret_cast<AllocThreadFn>(resolve("AllocThread"));
        freeThread = reinterpret_cast<FreeThreadFn>(resolve("FreeThread"));
        getMemTokenId = reinterpret_cast<GetMemTokenIdFn>(resolve("GetMemTokenId"));
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
            "bind={} unregister={} send={} alloc_thread={} free_thread={} mem_token={} "
            "capabilities={}",
            libraryPath, bindMemory != nullptr ? 1 : 0, unregisterMemory != nullptr ? 1 : 0,
            send != nullptr ? 1 : 0, allocThread != nullptr ? 1 : 0, freeThread != nullptr ? 1 : 0,
            getMemTokenId != nullptr ? 1 : 0, getServerCapabilities != nullptr ? 1 : 0);
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
        std::uint32_t qpIndex{0};
        std::uint32_t qpNum{0};
        std::uint32_t remotePort{0};
        std::string localIp;
        std::string remoteIp;
    };

    using ConnectionMap = std::unordered_map<ConnectionHandle, std::shared_ptr<ConnectionRecord>>;

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

        allocator = std::make_unique<HbmSlotAllocator>(
            ResolveDeviceId(config), kHccpKvRoceConnInfoSlotBytes, slotCapacity);
        const auto initStatus = allocator->Init();
        if (!initStatus.ok()) { throw std::runtime_error(initStatus.message); }

        KV_INFO(
            "HCCP_KV_ROCE_TRANSPORT_PROVIDER_SIGNATURE={} pid={} node_id={} device_id={} "
            "library={} slot_bytes={} slot_capacity={}",
            kProviderSignature, static_cast<long>(::getpid()), config.nodeId,
            allocator->deviceId(), libraryPath, allocator->slotBytes(), allocator->capacity());
    }

    Status WriteConnInfo(const ConnectionRecord& record)
    {
        ConnInfo info{};
        info.magic = kConnInfoMagic;
        info.version = kConnInfoVersion;
        info.slotId = record.slotId;
        info.qpNum = record.qpNum;
        info.qpIndex = record.qpIndex;
        info.remotePort = record.remotePort;
        info.externalHandle = reinterpret_cast<std::uintptr_t>(record.externalHandle);
        CopyIp(info.localIp, record.localIp);
        CopyIp(info.remoteIp, record.remoteIp);
        return allocator->Write(record.slotId, &info, sizeof(info));
    }

    AbiApi api;
    std::string libraryPath;
    std::uint32_t slotCapacity{kHccpKvRoceDefaultSlotCapacity};
    std::unique_ptr<HbmSlotAllocator> allocator;

    mutable std::shared_mutex connectionMu;
    ConnectionMap connections;

    mutable std::shared_mutex memoryMu;
    std::unordered_set<MRHandle> memories;
};

HccpKvOverRoceTransProvider::HccpKvOverRoceTransProvider(const TransportConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

HccpKvOverRoceTransProvider::~HccpKvOverRoceTransProvider()
{
    if (!impl_) { return; }

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

    std::vector<MRHandle> memoryHandles;
    {
        std::shared_lock<std::shared_mutex> lock(impl_->memoryMu);
        memoryHandles.assign(impl_->memories.begin(), impl_->memories.end());
    }
    if (!memoryHandles.empty() && impl_->api.unregisterMemory != nullptr) {
        const auto ret = impl_->api.unregisterMemory(
            memoryHandles.data(), static_cast<std::uint32_t>(memoryHandles.size()));
        if (ret != 0) {
            KV_WARN("HccpKvOverRoceTransProvider: memory cleanup failed count={} code={}",
                    memoryHandles.size(), ret);
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
    if (qpNum == 0) { return Status::OK(); }
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

    KV_INFO(
        "HccpKvOverRoceTransProvider: CreateConnection start local_addr={} remote_addr={} "
        "remote_port={} qp_num={} timeout_ms={}",
        localIp, remoteIp, port, qpNum, timeout);

    std::vector<void*> externalHandles(qpNum, nullptr);
    const auto ret = impl_->api.createConnection(localIp.c_str(), remoteIp.c_str(), port, qpNum,
                                                 timeout, externalHandles.data());
    if (ret != 0) {
        // Best effort: the library may have created some handles before it
        // reported failure, so release anything non-null before freeing slots.
        std::vector<void*> createdHandles;
        createdHandles.reserve(externalHandles.size());
        for (const auto& externalHandle : externalHandles) {
            if (externalHandle != nullptr) { createdHandles.push_back(externalHandle); }
        }
        if (!createdHandles.empty()) {
            (void)impl_->api.deleteConnections(
                createdHandles.data(), static_cast<std::uint32_t>(createdHandles.size()));
        }
        ReleaseSlots(*impl_->allocator, slots);
        return AbiError("CreateConnection", ret, StatusCode::CONNECTION_ERROR);
    }

    bool hasNullHandle = false;
    for (const auto& externalHandle : externalHandles) {
        if (externalHandle == nullptr) {
            hasNullHandle = true;
            break;
        }
    }
    if (hasNullHandle) {
        std::vector<void*> validExternalHandles;
        validExternalHandles.reserve(externalHandles.size());
        for (const auto& externalHandle : externalHandles) {
            if (externalHandle != nullptr) { validExternalHandles.push_back(externalHandle); }
        }
        if (!validExternalHandles.empty()) {
            (void)impl_->api.deleteConnections(
                validExternalHandles.data(),
                static_cast<std::uint32_t>(validExternalHandles.size()));
        }
        ReleaseSlots(*impl_->allocator, slots);
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: CreateConnection returned a null "
                             "connection handle");
    }

    std::vector<std::shared_ptr<Impl::ConnectionRecord>> records;
    records.reserve(qpNum);
    for (std::uint32_t index = 0; index < qpNum; ++index) {
        auto record = std::make_shared<Impl::ConnectionRecord>();
        record->externalHandle = externalHandles[index];
        record->slotId = slots[index];
        record->qpIndex = index;
        record->qpNum = qpNum;
        record->remotePort = port;
        record->localIp = localIp;
        record->remoteIp = remoteIp;

        const auto status = impl_->WriteConnInfo(*record);
        if (!status.ok()) {
            (void)impl_->api.deleteConnections(externalHandles.data(), qpNum);
            for (std::uint32_t written = 0; written <= index; ++written) {
                (void)impl_->allocator->Clear(slots[written]);
            }
            ReleaseSlots(*impl_->allocator, slots);
            return status;
        }
        records.push_back(std::move(record));
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
            impl_->connections.emplace(removed[index].get(), std::move(removed[index]));
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

    {
        // Keep the connection map locked for the whole ABI call so a concurrent
        // DeleteConnections cannot recycle an external handle while Send uses it.
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
            abiBatches.push_back({iter->second->externalHandle, ioBatches[index].sendBuffer,
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

    std::vector<HccpKvRoceRegisterMemoryDesc> abiDescs;
    abiDescs.reserve(memoryDescs.size());
    for (const auto& desc : memoryDescs) {
        if (desc.addr == 0 || desc.size == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: RegisterMemory requires a "
                                 "non-zero address and size");
        }
        abiDescs.push_back({ToAbiMemoryType(desc.memoryType), static_cast<std::uint64_t>(desc.addr),
                            static_cast<std::uint64_t>(desc.size),
                            static_cast<std::uint64_t>(desc.localAddr)});
    }

    std::vector<std::uint64_t> handles(memoryDescs.size(), 0);
    const auto ret = impl_->api.registerMemory(abiDescs.data(),
                                               static_cast<std::uint32_t>(abiDescs.size()),
                                               handles.data());
    if (ret != 0) { return AbiError("RegisterMemory", ret, StatusCode::BUFFER_NOT_SUPPORTED); }

    for (const auto handle : handles) {
        if (handle != 0) { continue; }
        std::vector<std::uint64_t> validHandles;
        validHandles.reserve(handles.size());
        for (const auto candidate : handles) {
            if (candidate != 0) { validHandles.push_back(candidate); }
        }
        if (!validHandles.empty() && impl_->api.unregisterMemory != nullptr) {
            (void)impl_->api.unregisterMemory(validHandles.data(),
                                              static_cast<std::uint32_t>(validHandles.size()));
        }
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: RegisterMemory returned an invalid "
                             "handle");
    }

    {
        std::unique_lock<std::shared_mutex> lock(impl_->memoryMu);
        for (const auto handle : handles) { impl_->memories.insert(handle); }
    }
    mrHandles.assign(handles.begin(), handles.end());
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
    if (impl_->api.bindMemory == nullptr) {
        return Status::Error(StatusCode::UNSUPPORTED,
                             "HccpKvOverRoceTransProvider: BindMemory is not exported by the "
                             "external library");
    }
    if (memoryDescs.empty()) { return Status::OK(); }

    std::vector<HccpKvRoceBindMemoryDesc> abiDescs;
    abiDescs.reserve(memoryDescs.size());
    for (const auto& desc : memoryDescs) {
        if (desc.addr == 0 || desc.size == 0) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "HccpKvOverRoceTransProvider: BindMemory requires a non-zero "
                                 "address and size");
        }
        abiDescs.push_back({ToAbiMemoryType(desc.memoryType), static_cast<std::uint64_t>(desc.addr),
                            static_cast<std::uint64_t>(desc.size), desc.tokenId});
    }

    std::vector<std::uint64_t> handles(memoryDescs.size(), 0);
    const auto ret = impl_->api.bindMemory(abiDescs.data(),
                                           static_cast<std::uint32_t>(abiDescs.size()),
                                           handles.data());
    if (ret != 0) { return AbiError("BindMemory", ret, StatusCode::BUFFER_NOT_SUPPORTED); }

    for (const auto handle : handles) {
        if (handle == 0) {
            std::vector<std::uint64_t> validHandles;
            validHandles.reserve(handles.size());
            for (const auto candidate : handles) {
                if (candidate != 0) { validHandles.push_back(candidate); }
            }
            if (!validHandles.empty() && impl_->api.unregisterMemory != nullptr) {
                (void)impl_->api.unregisterMemory(
                    validHandles.data(), static_cast<std::uint32_t>(validHandles.size()));
            }
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "HccpKvOverRoceTransProvider: BindMemory returned an invalid "
                                 "handle");
        }
    }

    {
        std::unique_lock<std::shared_mutex> lock(impl_->memoryMu);
        for (const auto handle : handles) { impl_->memories.insert(handle); }
    }
    mrHandles.assign(handles.begin(), handles.end());
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

    std::vector<std::uint64_t> handles;
    std::vector<std::size_t> indices;
    handles.reserve(memoryDescs.size());
    indices.reserve(memoryDescs.size());

    std::int32_t ret = 0;
    {
        // Hold the memory lock across the ABI call so GetMemTokenId cannot query a
        // handle that is being unregistered concurrently.
        std::unique_lock<std::shared_mutex> lock(impl_->memoryMu);
        for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
            const auto handle = memoryDescs[index].mrHandle;
            if (impl_->memories.count(handle) == 0) {
                results[index] = Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                                               "HccpKvOverRoceTransProvider: invalid memory "
                                               "handle");
                continue;
            }
            handles.push_back(handle);
            indices.push_back(index);
        }
        if (handles.empty()) { return results; }

        if (impl_->api.unregisterMemory != nullptr) {
            ret = impl_->api.unregisterMemory(handles.data(),
                                              static_cast<std::uint32_t>(handles.size()));
        } else {
            // The external library only exports the three required methods. Drop
            // the local records so the client can still release its buffers.
            KV_WARN(
                "HccpKvOverRoceTransProvider: external library does not export "
                "UnregisterMemory; dropping local records for {} memories",
                handles.size());
        }
        if (ret == 0) {
            for (const auto handle : handles) { impl_->memories.erase(handle); }
        }
    }

    if (ret != 0) {
        const auto error = AbiError("UnregisterMemory", ret, StatusCode::INTERNAL_ERROR);
        for (const auto index : indices) { results[index] = error; }
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
    if (!impl_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HccpKvOverRoceTransProvider is not initialized");
    }
    if (impl_->api.getMemTokenId == nullptr) {
        return Status::Error(StatusCode::UNSUPPORTED,
                             "HccpKvOverRoceTransProvider: GetMemTokenId is not exported by the "
                             "external library");
    }

    std::uint32_t token = 0;
    std::int32_t ret = 0;
    {
        std::shared_lock<std::shared_mutex> lock(impl_->memoryMu);
        if (impl_->memories.count(mrHandle) == 0) {
            return Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                                 "HccpKvOverRoceTransProvider: invalid memory handle");
        }
        ret = impl_->api.getMemTokenId(mrHandle, &token);
    }
    if (ret != 0) { return AbiError("GetMemTokenId", ret, StatusCode::INTERNAL_ERROR); }
    tokenId = token;
    return Status::OK();
}

}  // namespace kv

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
