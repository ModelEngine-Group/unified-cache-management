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
#include <cctype>
#include <cstdint>
#include <cstdlib>
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
#include "hccp_acl_context.h"
#include "hccp_kv_over_roce_abi.h"
#include "logger.h"
#include "parser_common.h"

namespace kv {
namespace {

constexpr const char* kProviderSignature = "UCM_ASU_HCCP_KV_OVER_ROCE_PROVIDER_ABI_V1";
constexpr const char* kDefaultLibraryPath = "libhccp_kv_over_roce.so";
constexpr const char* kLibraryPathEnv = "UCM_HCCP_KV_ROCE_LIB_PATH";
constexpr const char* kInitAttrPrefix = "hccp_kv_roce_";

using InitFn = decltype(&::Init);

// InitConfig defaults mirror the comments next to each field in
// hccp_kv_over_roce_abi.h. Every field listed here is always assigned, so the
// header comments and these tables must stay in sync.
struct InitUint64Field {
    const char* name;
    std::uint64_t HccpKvRoceInitConfig::*member;
    std::uint64_t defaultValue;
};

constexpr InitUint64Field kInitUint64Fields[] = {
    {"mem_size", &HccpKvRoceInitConfig::memSize, 671088640ULL},
    {"send_mr_size", &HccpKvRoceInitConfig::sendMrSize, 0ULL},
    {"flag_mr_size", &HccpKvRoceInitConfig::flagMrSize, 0ULL},
};

struct InitUint32Field {
    const char* name;
    std::uint32_t HccpKvRoceInitConfig::*member;
    std::uint32_t defaultValue;
};

constexpr InitUint32Field kInitUint32Fields[] = {
    {"sdk_ioq_depth", &HccpKvRoceInitConfig::sdkIoqDepth, 128},
    {"timeout_ms", &HccpKvRoceInitConfig::timeoutMs, 60000},
    {"kvnsid", &HccpKvRoceInitConfig::kvnsid, 0},
    {"server_batch_limit", &HccpKvRoceInitConfig::serverBatchLimit, 32},
    {"server_delete_limit", &HccpKvRoceInitConfig::serverDeleteLimit, 254},
    {"server_exist_limit", &HccpKvRoceInitConfig::serverExistLimit, 256},
    {"max_inflight_kv_per_ip", &HccpKvRoceInitConfig::maxInflightKvPerIp, 2048},
    {"max_inflight_kv_per_qp", &HccpKvRoceInitConfig::maxInflightKvPerQp, 256},
    {"send_aiv_count", &HccpKvRoceInitConfig::sendAivCount, 1},
    {"wait_aiv_count", &HccpKvRoceInitConfig::waitAivCount, 1},
    {"cache_count", &HccpKvRoceInitConfig::cacheCount, 4},
    {"stripe_count", &HccpKvRoceInitConfig::stripeCount, 64},
    {"max_flags_per_stripe", &HccpKvRoceInitConfig::maxFlagsPerStripe, 20480},
    {"npu_cacheline_size", &HccpKvRoceInitConfig::npuCachelineSize, 256},
    {"quiet_interval", &HccpKvRoceInitConfig::quietInterval, 8},
    {"send_cq_wait", &HccpKvRoceInitConfig::sendCqWait, 1},
    {"virtual_nodes", &HccpKvRoceInitConfig::virtualNodes, 5},
    {"post_connect_delay_ms", &HccpKvRoceInitConfig::postConnectDelayMs, 0},
    {"wait_poll_yield_us", &HccpKvRoceInitConfig::waitPollYieldUs, 0},
    {"wait_task_timeout_ms", &HccpKvRoceInitConfig::waitTaskTimeoutMs, 2000},
    {"max_aiv_retry", &HccpKvRoceInitConfig::maxAivRetry, 3},
    {"iterations", &HccpKvRoceInitConfig::iterations, 1},
    {"total_kv_count", &HccpKvRoceInitConfig::totalKvCount, 4096},
    {"block_size", &HccpKvRoceInitConfig::blockSize, 131072},
    {"op_type", &HccpKvRoceInitConfig::opType, 0},
    {"pipeline_num_layers", &HccpKvRoceInitConfig::pipelineNumLayers, 64},
    {"comp_delay_us", &HccpKvRoceInitConfig::compDelayUs, 400000},
    {"mt_num_threads", &HccpKvRoceInitConfig::mtNumThreads, 4},
    {"mt_ops_per_thread", &HccpKvRoceInitConfig::mtOpsPerThread, 10},
    {"debug", &HccpKvRoceInitConfig::debug, 0},
    {"neg_start_psn", &HccpKvRoceInitConfig::negStartPsn, 0},
};

struct InitUint8Field {
    const char* name;
    std::uint8_t HccpKvRoceInitConfig::*member;
    std::uint8_t defaultValue;
};

constexpr InitUint8Field kInitUint8Fields[] = {
    {"neg_mtu", &HccpKvRoceInitConfig::negMtu, 5},
    {"neg_sl", &HccpKvRoceInitConfig::negSl, 0},
    {"neg_traffic_class", &HccpKvRoceInitConfig::negTrafficClass, 3},
    {"neg_rnr_timer", &HccpKvRoceInitConfig::negRnrTimer, 12},
    {"neg_rnr_retry_cnt", &HccpKvRoceInitConfig::negRnrRetryCnt, 7},
    {"neg_timeout", &HccpKvRoceInitConfig::negTimeout, 14},
    {"neg_retry_cnt", &HccpKvRoceInitConfig::negRetryCnt, 7},
    {"neg_qp_rd_atom", &HccpKvRoceInitConfig::negQpRdAtom, 16},
};

std::string InitAttrName(const char* field)
{
    return std::string(kInitAttrPrefix) + field;
}

bool LookupAttrValue(const TransportConfig& config, const std::string& key, std::string& value)
{
    const auto iter = config.attrs.find(key);
    if (iter == config.attrs.end()) { return false; }
    value = iter->second;
    return true;
}

std::string ToLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

void WarnInvalidInitValue(const std::string& attr, const std::string& value,
                          const std::string& expected)
{
    KV_WARN("HccpKvOverRoceTransProvider: ignoring invalid {} value=\"{}\"; using the default {}",
            attr, value, expected);
}

HccpKvRoceInitConfig ResolveInitConfig(const TransportConfig& config,
                                       std::uint32_t& overrideCount)
{
    HccpKvRoceInitConfig initConfig{};
    initConfig.structSize = static_cast<std::uint32_t>(sizeof(initConfig));
    initConfig.version = HCCP_KV_ROCE_INIT_CONFIG_VERSION;
    constexpr auto kInvalidValue = std::numeric_limits<std::uint64_t>::max();

    for (const auto& field : kInitUint64Fields) {
        initConfig.*(field.member) = field.defaultValue;
        const auto attr = InitAttrName(field.name);
        std::string value;
        if (!LookupAttrValue(config, attr, value)) { continue; }
        const auto parsed = ParseConfigUint64(value, kInvalidValue);
        if (parsed == kInvalidValue) {
            WarnInvalidInitValue(attr, value, std::to_string(field.defaultValue));
            continue;
        }
        initConfig.*(field.member) = parsed;
        ++overrideCount;
    }

    for (const auto& field : kInitUint32Fields) {
        initConfig.*(field.member) = field.defaultValue;
        const auto attr = InitAttrName(field.name);
        std::string value;
        if (!LookupAttrValue(config, attr, value)) { continue; }
        const auto parsed = ParseConfigUint64(value, kInvalidValue);
        if (parsed == kInvalidValue || parsed > std::numeric_limits<std::uint32_t>::max()) {
            WarnInvalidInitValue(attr, value, std::to_string(field.defaultValue));
            continue;
        }
        initConfig.*(field.member) = static_cast<std::uint32_t>(parsed);
        ++overrideCount;
    }

    for (const auto& field : kInitUint8Fields) {
        initConfig.*(field.member) = field.defaultValue;
        const auto attr = InitAttrName(field.name);
        std::string value;
        if (!LookupAttrValue(config, attr, value)) { continue; }
        const auto parsed = ParseConfigUint64(value, kInvalidValue);
        if (parsed == kInvalidValue || parsed > std::numeric_limits<std::uint8_t>::max()) {
            WarnInvalidInitValue(attr, value, std::to_string(field.defaultValue));
            continue;
        }
        initConfig.*(field.member) = static_cast<std::uint8_t>(parsed);
        ++overrideCount;
    }

    const std::string oobAttr = InitAttrName("oob_transport");
    initConfig.oobTransport = HCCP_KV_ROCE_OOB_TCP;
    std::string oobValue;
    if (LookupAttrValue(config, oobAttr, oobValue)) {
        const auto lowered = ToLowerAscii(oobValue);
        if (lowered == "tcp" || lowered == "0") {
            initConfig.oobTransport = HCCP_KV_ROCE_OOB_TCP;
            ++overrideCount;
        } else if (lowered == "rasocket" || lowered == "1") {
            initConfig.oobTransport = HCCP_KV_ROCE_OOB_RASOCKET;
            ++overrideCount;
        } else {
            WarnInvalidInitValue(oobAttr, oobValue, "TCP (0)");
        }
    }
    return initConfig;
}

// Every connection owns one HBM slot. The slot receives the operator QPC and is
// later handed to the operator IO path through Send.
static_assert(sizeof(HccpKvRoceQpc) == kHccpKvRoceConnSlotBytes,
              "the HBM connection slot must be exactly one QPC");

using CreateConnectionFn = decltype(&::CreateConnection);
using DeleteConnectionsFn = decltype(&::DeleteConnections);
using RegisterMemoryFn = decltype(&::RegisterMemory);
using GetAbiVersionFn = decltype(&::GetAbiVersion);
using UnregisterMemoryFn = decltype(&::UnregisterMemory);
using SendFn = decltype(&::Send);
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

// Init installs a process-wide configuration on its first successful call.
// Later calls return OK without changing it, so the provider calls Init on
// every construction and serializes concurrent constructions to keep the first
// call deterministic.
Status InstallInitConfig(InitFn init, const HccpKvRoceInitConfig& config)
{
    static std::mutex initMutex;
    std::lock_guard<std::mutex> lock(initMutex);
    const auto ret = init(&config);
    if (ret != HCCP_KV_ROCE_OK) { return AbiError("Init", ret, StatusCode::INTERNAL_ERROR); }
    return Status::OK();
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

    constexpr std::uint64_t kInvalidConfigValue = std::numeric_limits<std::uint64_t>::max();
    const auto parsed = ParseConfigUint64(configured, kInvalidConfigValue);
    if (parsed == kInvalidConfigValue || parsed > std::numeric_limits<std::uint32_t>::max()) {
        KV_WARN(
            "HccpKvOverRoceTransProvider: ignoring invalid hccp_kv_roce_max_connections "
            "value=\"{}\"; using the default capacity {}",
            configured, kHccpKvRoceDefaultSlotCapacity);
        return kHccpKvRoceDefaultSlotCapacity;
    }
    return static_cast<std::uint32_t>(parsed);
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
    GetServerCapabilitiesFn getServerCapabilities{nullptr};
    InitFn init{nullptr};

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
        init = reinterpret_cast<InitFn>(resolve("Init"));
        if (init == nullptr || createConnection == nullptr || deleteConnections == nullptr ||
            registerMemory == nullptr) {
            const char* missing =
                init == nullptr
                    ? "Init"
                    : (createConnection == nullptr
                           ? "CreateConnection"
                           : (deleteConnections == nullptr ? "DeleteConnections"
                                                           : "RegisterMemory"));
            (void)dlclose(library);
            return Status::Error(StatusCode::NOT_FOUND,
                                 "HccpKvOverRoceTransProvider: " + libraryPath +
                                     " does not export required symbol " + missing);
        }

        // Optional symbols: a missing entry keeps the provider fallback.
        unregisterMemory = reinterpret_cast<UnregisterMemoryFn>(resolve("UnregisterMemory"));
        send = reinterpret_cast<SendFn>(resolve("Send"));
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
            "unregister={} send={} capabilities={}",
            libraryPath, unregisterMemory != nullptr ? 1 : 0, send != nullptr ? 1 : 0,
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
    // One QP of an IP-pair connection. Its address is the ConnectionHandle the
    // provider returns to the transport, and its QPC is copied into the HBM slot
    // used by Send.
    struct QpRecord {
        std::uint32_t slotId{0};
        // Position inside ConnectionRecord::qps. DeleteConnections uses it to
        // prove that the caller passed the complete QP set of an IP pair.
        std::uint32_t index{0};
        HccpKvRoceQpc qpc{};
    };

    // One CreateConnection call. The external library returns a single manager
    // that owns every QP created for this IP pair, so the provider keeps one
    // record per manager instead of one record per QP.
    struct ConnectionRecord {
        void* manager{nullptr};
        std::vector<std::unique_ptr<QpRecord>> qps;
    };

    using ConnectionMap = std::unordered_map<ConnectionHandle, std::shared_ptr<ConnectionRecord>>;

    // A record whose rollback DeleteConnections failed. The ABI keeps a manager
    // valid until a delete succeeds, so the provider retains the whole record
    // (manager and QPC storage) and retries it on later cleanup passes. The HBM
    // slots of the failed call are released by the rollback itself; the retry
    // only calls DeleteConnections.
    struct OrphanedConnection {
        std::shared_ptr<ConnectionRecord> record;
        std::string remoteIp;
        std::uint32_t port{0};
    };

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

        std::uint32_t initOverrides = 0;
        const auto initConfig = ResolveInitConfig(config, initOverrides);
        const auto abiInitStatus = InstallInitConfig(api.init, initConfig);
        if (!abiInitStatus.ok()) { throw std::runtime_error(abiInitStatus.message); }

        allocator = std::make_unique<HbmSlotAllocator>(
            binding.deviceId, kHccpKvRoceConnSlotBytes, slotCapacity);
        const auto initStatus = allocator->Init();
        if (!initStatus.ok()) { throw std::runtime_error(initStatus.message); }

        KV_INFO(
            "HCCP_KV_ROCE_TRANSPORT_PROVIDER_SIGNATURE={} pid={} node_id={} device_id={} "
            "device_source={} provider_context={} library={} slot_bytes={} slot_capacity={} "
            "mem_size={} oob_transport={} timeout_ms={} sdk_ioq_depth={} init_overrides={}",
            kProviderSignature, static_cast<long>(::getpid()), config.nodeId, binding.deviceId,
            binding.source, static_cast<const void*>(binding.context), libraryPath,
            allocator->slotBytes(), allocator->capacity(), initConfig.memSize,
            initConfig.oobTransport, initConfig.timeoutMs, initConfig.sdkIoqDepth, initOverrides);
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
    std::vector<OrphanedConnection> orphanedConnections;

    // Caller must not hold connectionMu.
    void KeepOrphanedConnection(std::shared_ptr<ConnectionRecord> record,
                                const std::string& remoteIp, std::uint32_t port)
    {
        std::unique_lock<std::shared_mutex> lock(connectionMu);
        orphanedConnections.push_back(OrphanedConnection{std::move(record), remoteIp, port});
    }

    // Retries every retained record. A record is dropped only after the library
    // reports success; failures stay queued for the next cleanup pass. The
    // caller must hold the provider ACL binding.
    void RetryOrphanedConnections(const char* stage)
    {
        std::vector<OrphanedConnection> pending;
        {
            std::unique_lock<std::shared_mutex> lock(connectionMu);
            pending.swap(orphanedConnections);
        }
        if (pending.empty()) { return; }

        std::vector<OrphanedConnection> remaining;
        remaining.reserve(pending.size());
        for (auto& orphan : pending) {
            void* manager = orphan.record->manager;
            const auto ret = api.deleteConnections(&manager, 1);
            if (ret == 0) {
                orphan.record->manager = nullptr;
                KV_INFO(
                    "HccpKvOverRoceTransProvider: released retained connection stage={} "
                    "remote_addr={} remote_port={} qp_num={}",
                    stage, orphan.remoteIp, orphan.port, orphan.record->qps.size());
                continue;
            }
            KV_WARN(
                "HccpKvOverRoceTransProvider: retained connection is still busy stage={} "
                "remote_addr={} remote_port={} qp_num={} code={}; keeping it for the next "
                "cleanup",
                stage, orphan.remoteIp, orphan.port, orphan.record->qps.size(), ret);
            remaining.push_back(std::move(orphan));
        }

        if (!remaining.empty()) {
            std::unique_lock<std::shared_mutex> lock(connectionMu);
            for (auto& orphan : remaining) { orphanedConnections.push_back(std::move(orphan)); }
        }
    }

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

    // Last chance to release managers retained after a failed rollback. When
    // live connections were present, DeleteConnections above already attempted
    // this once.
    impl_->RetryOrphanedConnections("Destroy");

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

    // Retry managers retained after an earlier failed rollback before opening a
    // new IP pair; this also frees operator-side resources of the same peer.
    impl_->RetryOrphanedConnections("CreateConnection");

    // One record per IP pair: it keeps the library manager plus every QP created
    // by this call. The QP addresses are stable for the record lifetime and are
    // the handles returned to the transport.
    auto record = std::make_shared<Impl::ConnectionRecord>();
    record->qps.reserve(qpNum);
    std::vector<std::uint32_t> slots;
    slots.reserve(qpNum);
    std::vector<void*> abiHandles(qpNum, nullptr);
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

        auto qp = std::make_unique<Impl::QpRecord>();
        qp->slotId = slotId;
        qp->index = index;
        abiHandles[index] = &qp->qpc;
        record->qps.push_back(std::move(qp));
    }

    std::uint32_t createdNum = 0;

    // The operator releases a whole IP pair through its manager, so every error
    // path after creation goes through this helper.
    auto deleteCreatedManager = [&](const char* stage) {
        if (record->manager == nullptr) {
            if (createdNum != 0) {
                KV_WARN(
                    "HccpKvOverRoceTransProvider: {} cannot release operator connections "
                    "created={} because the library returned no manager",
                    stage, createdNum);
            }
            return;
        }
        void* manager = record->manager;
        const auto deleteRet = impl_->api.deleteConnections(&manager, 1);
        if (deleteRet != 0) {
            // The ABI keeps the manager valid after a failed delete, and it is
            // the only handle that can release the operator-side QPs. Retain the
            // record and retry it on a later cleanup instead of dropping it.
            KV_WARN(
                "HccpKvOverRoceTransProvider: {} failed to release operator connections "
                "created={} code={}; keeping the manager for a later cleanup",
                stage, createdNum, deleteRet);
            impl_->KeepOrphanedConnection(record, remoteIp, port);
            return;
        }
        record->manager = nullptr;
    };

    KV_INFO(
        "HccpKvOverRoceTransProvider: CreateConnection start local_addr={} remote_addr={} "
        "remote_port={} qp_num={} timeout_ms={}",
        localIp, remoteIp, port, qpNum, timeout);

    const auto ret = impl_->api.createConnection(localIp.c_str(), remoteIp.c_str(), port, qpNum,
                                                 timeout, abiHandles.data(), &createdNum,
                                                 &record->manager);
    if (ret != 0) {
        deleteCreatedManager("CreateConnection");
        ReleaseSlots(*impl_->allocator, slots);
        return AbiError("CreateConnection", ret, StatusCode::CONNECTION_ERROR);
    }
    if (createdNum > qpNum) {
        // Never trust the array beyond qpNum even if the library reports more.
        deleteCreatedManager("CreateConnection");
        ReleaseSlots(*impl_->allocator, slots);
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: CreateConnection reported " +
                                 std::to_string(createdNum) + " handles for qp_num " +
                                 std::to_string(qpNum));
    }
    if (createdNum != qpNum) {
        deleteCreatedManager("CreateConnection");
        ReleaseSlots(*impl_->allocator, slots);
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: CreateConnection created " +
                                 std::to_string(createdNum) + " of " + std::to_string(qpNum) +
                                 " connections");
    }

    for (std::uint32_t index = 0; index < qpNum; ++index) {
        if (abiHandles[index] == nullptr) {
            deleteCreatedManager("CreateConnection");
            ReleaseSlots(*impl_->allocator, slots);
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "HccpKvOverRoceTransProvider: CreateConnection returned a null "
                                 "connection handle");
        }
    }

    // The manager is the only way to release this IP pair later, so a success
    // without one is treated as a failed creation.
    if (record->manager == nullptr) {
        deleteCreatedManager("CreateConnection");
        ReleaseSlots(*impl_->allocator, slots);
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "HccpKvOverRoceTransProvider: CreateConnection returned no "
                             "connection manager");
    }

    for (std::uint32_t index = 0; index < qpNum; ++index) {
        const auto status = impl_->allocator->Write(record->qps[index]->slotId, abiHandles[index],
                                                    sizeof(HccpKvRoceQpc));
        if (!status.ok()) {
            deleteCreatedManager("CreateConnection");
            for (std::uint32_t written = 0; written <= index; ++written) {
                (void)impl_->allocator->Clear(record->qps[written]->slotId);
            }
            ReleaseSlots(*impl_->allocator, slots);
            return status;
        }
    }

    {
        std::unique_lock<std::shared_mutex> lock(impl_->connectionMu);
        for (const auto& qp : record->qps) {
            ConnectionHandle handle = qp.get();
            impl_->connections.emplace(handle, record);
            connectionHandles.push_back(handle);
        }
    }

    KV_INFO(
        "HccpKvOverRoceTransProvider: CreateConnection complete remote_addr={} remote_port={} "
        "qp_num={} handles={}",
        remoteIp, port, qpNum, connectionHandles.size());
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
    ScopedHccpAclContext aclScope("DeleteConnections", impl_->binding);
    if (!aclScope.status().ok()) {
        std::fill(results.begin(), results.end(), aclScope.status());
        return results;
    }

    // DeleteConnections is a cleanup entry point, so retry managers retained
    // after a failed rollback even when the caller passes no handles.
    impl_->RetryOrphanedConnections("DeleteConnections");
    if (connectionHandles.empty()) { return results; }

    // Group the requested QP handles by their IP-pair record. A record may only
    // be deleted when the caller passes every QP of that record, because the
    // operator releases the whole set through one manager.
    struct RecordRequest {
        std::shared_ptr<Impl::ConnectionRecord> record;
        std::vector<std::size_t> indices;
        std::vector<bool> seen;
        bool invalid{false};
        bool complete{false};
    };

    std::unordered_map<Impl::ConnectionRecord*, RecordRequest> requests;
    std::unordered_set<ConnectionHandle> seenHandles;
    std::vector<void*> managers;

    {
        std::unique_lock<std::shared_mutex> lock(impl_->connectionMu);
        for (std::size_t index = 0; index < connectionHandles.size(); ++index) {
            auto* qp = static_cast<Impl::QpRecord*>(connectionHandles[index]);
            if (qp == nullptr) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: invalid connection "
                                               "handle");
                continue;
            }
            const auto iter = impl_->connections.find(qp);
            if (iter == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: invalid connection "
                                               "handle");
                continue;
            }

            auto& request = requests[iter->second.get()];
            request.record = iter->second;
            if (request.seen.empty()) {
                request.seen.assign(iter->second->qps.size(), false);
            }
            if (!seenHandles.insert(qp).second) {
                request.invalid = true;
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: duplicate connection "
                                               "handle");
                continue;
            }
            request.indices.push_back(index);
            request.seen[qp->index] = true;
        }

        for (auto& item : requests) {
            auto& request = item.second;
            request.complete =
                !request.invalid &&
                std::all_of(request.seen.begin(), request.seen.end(),
                            [](bool value) { return value; });
            if (!request.complete) {
                const auto error = Status::Error(
                    StatusCode::INVALID_ARGUMENT,
                    "HccpKvOverRoceTransProvider: incomplete connection set; every QP of an IP "
                    "pair must be deleted together");
                for (const auto index : request.indices) { results[index] = error; }
                continue;
            }

            for (const auto& qp : request.record->qps) {
                impl_->connections.erase(qp.get());
            }
            managers.push_back(request.record->manager);
        }
    }

    if (managers.empty()) {
        KV_INFO(
            "HccpKvOverRoceTransProvider: DeleteConnections complete requested={} deleted=0",
            connectionHandles.size());
        return results;
    }

    const auto ret = impl_->api.deleteConnections(
        managers.data(), static_cast<std::uint32_t>(managers.size()));
    if (ret != 0) {
        const auto error = AbiError("DeleteConnections", ret, StatusCode::CONNECTION_ERROR);
        std::unique_lock<std::shared_mutex> lock(impl_->connectionMu);
        for (auto& item : requests) {
            auto& request = item.second;
            if (!request.complete) { continue; }
            for (const auto& qp : request.record->qps) {
                impl_->connections.emplace(qp.get(), request.record);
            }
        }
        for (auto& item : requests) {
            auto& request = item.second;
            if (!request.complete) { continue; }
            for (const auto index : request.indices) { results[index] = error; }
        }
        return results;
    }

    for (auto& item : requests) {
        auto& request = item.second;
        if (!request.complete) { continue; }
        for (const auto& qp : request.record->qps) {
            const auto clearStatus = impl_->allocator->Clear(qp->slotId);
            if (!clearStatus.ok()) {
                KV_WARN(
                    "HccpKvOverRoceTransProvider: failed to clear HBM slot_id={} message={}",
                    qp->slotId, clearStatus.message);
            }
            const auto releaseStatus = impl_->allocator->Release(qp->slotId);
            if (!releaseStatus.ok()) {
                KV_WARN(
                    "HccpKvOverRoceTransProvider: failed to release HBM slot_id={} message={}",
                    qp->slotId, releaseStatus.message);
            }
        }
    }

    KV_INFO(
        "HccpKvOverRoceTransProvider: DeleteConnections complete requested={} ip_pairs={}",
        connectionHandles.size(), managers.size());
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
        ret = impl_->api.getServerCapabilities(iter->second->manager, &abiCapabilities);
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
            auto* qp = static_cast<Impl::QpRecord*>(ioBatches[index].connectionHandle);
            if (qp == nullptr) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: invalid connection "
                                               "handle");
                continue;
            }
            const auto iter = impl_->connections.find(qp);
            if (iter == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "HccpKvOverRoceTransProvider: invalid connection "
                                               "handle");
                continue;
            }
            // The operator reads the QPC out of the HBM slot, so the batch carries
            // this QP's slot address instead of a host-side handle.
            auto* slotAddress = impl_->allocator->SlotAddress(qp->slotId);
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
    }

    // Call the library without memoryMu: the records were detached above, so a
    // slow UnregisterMemory does not block concurrent RegisterMemory, BindMemory
    // or GetMemTokenId.
    std::int32_t ret = 0;
    if (!operatorInfos.empty()) {
        if (impl_->api.unregisterMemory != nullptr) {
            ret = impl_->api.unregisterMemory(operatorInfos.data(),
                                              static_cast<std::uint32_t>(operatorInfos.size()));
        } else {
            // The external library only exports the three required methods. Drop
            // the local records so the client can still release its buffers.
            KV_WARN(
                "HccpKvOverRoceTransProvider: external library does not export "
                "UnregisterMemory; dropping local records for {} memories",
                operatorInfos.size());
        }
    }

    if (ret != 0) {
        // Keep operator-owned records so a later call can retry; bind-only
        // records are local and stay released. Handles are never reused, so the
        // reinsertion cannot race with a concurrently registered record.
        std::unique_lock<std::shared_mutex> lock(impl_->memoryMu);
        for (const auto index : operatorIndices) {
            auto record = std::move(removed[index]);
            const auto handle = record->handle;
            impl_->memories.emplace(handle, std::move(record));
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
    // The external operator owns no per-thread resources, so the provider never
    // forwards this call and the transport can proceed without thread handles.
    (void)threadNum;
    (void)notifyNumPerThread;
    threads.clear();
    return Status::OK();
}

std::vector<Status> HccpKvOverRoceTransProvider::FreeThread(
    const std::vector<ThreadHandle>& threads)
{
    // Matching no-op for AllocThread: nothing was allocated, so every requested
    // thread is released successfully.
    return std::vector<Status>(threads.size(), Status::OK());
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
