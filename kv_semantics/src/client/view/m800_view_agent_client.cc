/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: dlopen-backed M800 view agent client.
 */

#include "view/m800_view_agent_client.h"

#include <dlfcn.h>

#include <limits>
#include <atomic>
#include <mutex>
#include <new>
#include <string>
#include <utility>

#include "KVClientAgentABI.h"
#include "logger.h"

namespace kv {
namespace {

struct AgentApi {
    decltype(&kv_client_agent_get_abi_version) getAbiVersion{nullptr};
    decltype(&kv_client_agent_create) create{nullptr};
    decltype(&kv_client_agent_init) init{nullptr};
    decltype(&kv_client_agent_start) start{nullptr};
    decltype(&kv_client_agent_stop) stop{nullptr};
    decltype(&kv_client_agent_get_epoch) getEpoch{nullptr};
    decltype(&kv_client_agent_get_all_ns_name) getAllNsName{nullptr};
    decltype(&kv_client_agent_get_view_by_ns) getViewByNs{nullptr};
    decltype(&kv_client_agent_get_in_by_ns) getInByNs{nullptr};
    decltype(&kv_client_agent_free_ns_names) freeNsNames{nullptr};
    decltype(&kv_client_agent_free_node_views) freeNodeViews{nullptr};
    decltype(&kv_client_agent_destroy) destroy{nullptr};
};

template <typename T>
bool LoadSymbol(void* library, const char* name, T& function, std::string& error)
{
    dlerror();
    void* symbol = dlsym(library, name);
    const char* symbolError = dlerror();
    if (symbolError != nullptr || symbol == nullptr) {
        error = "missing symbol " + std::string{name};
        if (symbolError != nullptr) { error += ": " + std::string{symbolError}; }
        function = nullptr;
        return false;
    }
    function = reinterpret_cast<T>(symbol);
    return true;
}

bool HasEmbeddedNul(const std::string& value)
{
    return value.find('\0') != std::string::npos;
}

bool LoadApi(void* library, AgentApi& api, std::string& error)
{
    return LoadSymbol(library, "kv_client_agent_get_abi_version", api.getAbiVersion, error) &&
           LoadSymbol(library, "kv_client_agent_create", api.create, error) &&
           LoadSymbol(library, "kv_client_agent_init", api.init, error) &&
           LoadSymbol(library, "kv_client_agent_start", api.start, error) &&
           LoadSymbol(library, "kv_client_agent_stop", api.stop, error) &&
           LoadSymbol(library, "kv_client_agent_get_epoch", api.getEpoch, error) &&
           LoadSymbol(library, "kv_client_agent_get_all_ns_name", api.getAllNsName, error) &&
           LoadSymbol(library, "kv_client_agent_get_view_by_ns", api.getViewByNs, error) &&
           LoadSymbol(library, "kv_client_agent_get_in_by_ns", api.getInByNs, error) &&
           LoadSymbol(library, "kv_client_agent_free_ns_names", api.freeNsNames, error) &&
           LoadSymbol(library, "kv_client_agent_free_node_views", api.freeNodeViews, error) &&
           LoadSymbol(library, "kv_client_agent_destroy", api.destroy, error);
}

Status FromAbiCode(int32_t code, StatusCode fallback, const std::string& context)
{
    if (code == KV_CLIENT_AGENT_OK) { return Status::OK(); }
    StatusCode target = fallback;
    switch (code) {
        case KV_CLIENT_AGENT_INVALID_ARGUMENT: target = StatusCode::INVALID_ARGUMENT; break;
        case KV_CLIENT_AGENT_ABI_VERSION_MISMATCH: target = StatusCode::UNSUPPORTED; break;
        case KV_CLIENT_AGENT_NOT_INITIALIZED: target = StatusCode::NOT_INITIALIZED; break;
        case KV_CLIENT_AGENT_EMPTY_VIEW: target = StatusCode::NOT_FOUND; break;
        case KV_CLIENT_AGENT_TIMEOUT: target = StatusCode::TIMEOUT; break;
        case KV_CLIENT_AGENT_INTERNAL_ERROR: target = StatusCode::INTERNAL_ERROR; break;
        default: break;
    }
    return Status::Error(target, context + " failed with agent ABI code " + std::to_string(code));
}

Status MillisecondsToSeconds(std::uint64_t milliseconds, int32_t& seconds,
                             const std::string& fieldName)
{
    seconds = 0;
    if (milliseconds == 0) { return Status::OK(); }
    const auto rounded = milliseconds / 1000 + (milliseconds % 1000 == 0 ? 0 : 1);
    if (rounded > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max())) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, fieldName + " exceeds agent ABI range");
    }
    seconds = static_cast<int32_t>(rounded);
    return Status::OK();
}

class DlopenAgentClient final : public M800ViewAgentClient {
public:
    DlopenAgentClient(void* library, kv_client_agent_handle* handle, AgentApi api,
                      const M800ViewServerConfig& config)
        : library_(library), handle_(handle), api_(std::move(api)), config_(config)
    {
    }

    ~DlopenAgentClient() override
    {
        try {
            std::lock_guard<std::mutex> lock{mutex_};
            DestroyLocked();
        } catch (...) {
            // The C ABI entry points are non-throwing; destructors must not throw.
        }
    }

    Status Initialize() override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (handle_ == nullptr || library_ == nullptr) { return NotInitialized("initialize"); }
        if (stopRequested_.load()) {
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "cannot initialize a stopped M800 agent client");
        }
        if (initialized_.load()) { return Status::OK(); }
        if (HasEmbeddedNul(config_.monHost) || HasEmbeddedNul(config_.clientName) ||
            HasEmbeddedNul(config_.clientKey)) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "agent connection configuration contains embedded NUL");
        }

        int32_t refreshSeconds = 0;
        auto status = MillisecondsToSeconds(config_.refreshIntervalMs, refreshSeconds,
                                            "view.refresh_interval_ms");
        if (!status.ok()) { return status; }
        int32_t fetchSeconds = 0;
        status = MillisecondsToSeconds(config_.agentFetchTimeoutMs, fetchSeconds,
                                       "view.agent_fetch_timeout_ms");
        if (!status.ok()) { return status; }

        kv_client_agent_config config{};
        config.mon_host = config_.monHost.c_str();
        config.client_name = config_.clientName.c_str();
        config.client_key = config_.clientKey.c_str();
        config.log_dir = nullptr;
        config.view_refresh_interval = refreshSeconds;
        config.view_fetch_timeout = fetchSeconds;
        const auto code = api_.init(handle_, &config, KV_CLIENT_AGENT_ABI_VERSION, sizeof(config));
        auto initStatus = FromAbiCode(code, StatusCode::CONNECTION_ERROR, "agent init");
        if (code == KV_CLIENT_AGENT_OK) {
            initialized_ = true;
            KV_INFO("KVClientAgent init success, mon_host={}", config_.monHost);
        } else {
            KV_WARN("KVClientAgent init failed, mon_host={}, err={}", config_.monHost,
                    initStatus.message);
        }
        return initStatus;
    }

    Status Start() override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (handle_ == nullptr || !initialized_) { return NotInitialized("start"); }
        if (started_) { return Status::OK(); }
        if (stopRequested_) {
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "cannot restart a stopped M800 agent client");
        }
        const auto code = api_.start(handle_);
        if (code == KV_CLIENT_AGENT_OK) { started_ = true; }
        return FromAbiCode(code, StatusCode::CONNECTION_ERROR, "agent start");
    }

    void Stop() override
    {
        stopRequested_.store(true);
        if (initialized_.load() && started_.load()) {
            // Invoke the agent cancellation outside the query mutex so a
            // blocked namespace read can be interrupted by stop_view_refresh.
            api_.stop(handle_);
        }
        std::lock_guard<std::mutex> lock{mutex_};
        started_.store(false);
    }

    Status GetEpoch(std::uint64_t& epoch) const override
    {
        epoch = 0;
        std::lock_guard<std::mutex> lock{mutex_};
        const auto ready = CheckReadyLocked("get epoch");
        if (!ready.ok()) { return ready; }
        const auto code = api_.getEpoch(handle_, &epoch);
        if (code != KV_CLIENT_AGENT_OK) { epoch = 0; }
        return FromAbiCode(code, StatusCode::INTERNAL_ERROR, "agent get epoch");
    }

    Status GetAllNsNames(std::vector<std::string>& nsNames) const override
    {
        nsNames.clear();
        std::lock_guard<std::mutex> lock{mutex_};
        const auto ready = CheckReadyLocked("get namespaces");
        if (!ready.ok()) { return ready; }

        const char** rawNames = nullptr;
        size_t rawCount = 0;
        const auto code = api_.getAllNsName(handle_, &rawNames, &rawCount);
        NsNamesGuard guard{api_, handle_, rawNames, rawCount};
        if (code != KV_CLIENT_AGENT_OK) {
            return FromAbiCode(code, StatusCode::CONNECTION_ERROR, "agent get namespaces");
        }
        if (rawCount != 0 && rawNames == nullptr) {
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "agent returned a null namespace array with nonzero count");
        }
        try {
            nsNames.reserve(rawCount);
            for (size_t i = 0; i < rawCount; ++i) {
                if (rawNames[i] == nullptr) {
                    nsNames.clear();
                    return Status::Error(StatusCode::INTERNAL_ERROR,
                                         "agent returned a null namespace name");
                }
                nsNames.emplace_back(rawNames[i]);
            }
        } catch (...) {
            nsNames.clear();
            return Status::Error(StatusCode::INTERNAL_ERROR, "failed to copy namespace names");
        }
        KV_DEBUG("get_all_ns_name returned {} ns", nsNames.size());
        return Status::OK();
    }

    Status GetViewByNs(const std::string& nsName, bool up, bool in,
                       std::vector<M800AgentNodeView>& views) const override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return GetNodeViewsLocked(nsName, up, in, false, views);
    }

    Status GetInByNs(const std::string& nsName,
                     std::vector<M800AgentNodeView>& views) const override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto status = GetNodeViewsLocked(nsName, false, true, true, views);
        if (!status.ok()) {
            KV_WARN("get_in_by_ns failed ns={}, err={}", nsName, status.message);
        } else {
            KV_DEBUG("get_in_by_ns ns={} osd_count={}", nsName, views.size());
        }
        return status;
    }

private:
    struct NsNamesGuard {
        const AgentApi& api;
        kv_client_agent_handle* handle;
        const char** names;
        size_t count;

        ~NsNamesGuard()
        {
            if (api.freeNsNames != nullptr) { api.freeNsNames(handle, names, count); }
        }
    };

    struct NodeViewsGuard {
        const AgentApi& api;
        kv_client_agent_handle* handle;
        kv_client_agent_node_view* views;
        size_t count;

        ~NodeViewsGuard()
        {
            if (api.freeNodeViews != nullptr) { api.freeNodeViews(handle, views, count); }
        }
    };

    Status NotInitialized(const std::string& operation) const
    {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "M800 agent client is not ready for " + operation);
    }

    Status CheckReadyLocked(const std::string& operation) const
    {
        if (handle_ == nullptr || library_ == nullptr || !initialized_ || !started_ ||
            stopRequested_) {
            return NotInitialized(operation);
        }
        return Status::OK();
    }

    Status GetNodeViewsLocked(const std::string& nsName, bool up, bool in, bool onlyIn,
                              std::vector<M800AgentNodeView>& views) const
    {
        views.clear();
        if (nsName.empty()) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "namespace name must not be empty");
        }
        const auto ready = CheckReadyLocked("get namespace view");
        if (!ready.ok()) { return ready; }

        kv_client_agent_node_view* rawViews = nullptr;
        size_t rawCount = 0;
        const auto code = onlyIn ? api_.getInByNs(handle_, nsName.c_str(), &rawViews, &rawCount)
                                 : api_.getViewByNs(handle_, nsName.c_str(), up ? 1 : 0, in ? 1 : 0,
                                                    &rawViews, &rawCount);
        NodeViewsGuard guard{api_, handle_, rawViews, rawCount};
        const auto operation = onlyIn ? "agent get in view" : "agent get view";
        if (code != KV_CLIENT_AGENT_OK) {
            return FromAbiCode(code, StatusCode::NOT_FOUND, operation);
        }
        if (rawCount != 0 && rawViews == nullptr) {
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "agent returned a null node array with nonzero count");
        }
        try {
            views.reserve(rawCount);
            for (size_t i = 0; i < rawCount; ++i) {
                const auto& source = rawViews[i];
                if (source.addr_count != 0 && source.addrs == nullptr) {
                    views.clear();
                    return Status::Error(StatusCode::INTERNAL_ERROR,
                                         "agent returned a null address array with nonzero count");
                }
                M800AgentNodeView target;
                target.osdId = source.osd_id;
                target.up = source.up != 0;
                target.in = source.in != 0;
                target.addrs.reserve(source.addr_count);
                for (size_t j = 0; j < source.addr_count; ++j) {
                    if (source.addrs[j] == nullptr) {
                        views.clear();
                        return Status::Error(StatusCode::INTERNAL_ERROR,
                                             "agent returned a null node address");
                    }
                    target.addrs.emplace_back(source.addrs[j]);
                }
                views.emplace_back(std::move(target));
            }
        } catch (...) {
            views.clear();
            return Status::Error(StatusCode::INTERNAL_ERROR, "failed to copy node views");
        }
        return Status::OK();
    }

    void StopLocked()
    {
        if (handle_ == nullptr || !initialized_) { return; }
        const bool wasStarted = started_;
        const auto code = api_.stop(handle_);
        (void)code;  // Stop has no status channel; destroy remains the final cleanup barrier.
        started_ = false;
        if (wasStarted) { stopRequested_ = true; }
    }

    void DestroyLocked()
    {
        StopLocked();
        if (handle_ != nullptr) {
            api_.destroy(handle_);
            handle_ = nullptr;
        }
        initialized_ = false;
        started_ = false;
        if (library_ != nullptr) {
            dlclose(library_);
            library_ = nullptr;
        }
    }

    void* library_{nullptr};
    kv_client_agent_handle* handle_{nullptr};
    AgentApi api_;
    M800ViewServerConfig config_;
    mutable std::mutex mutex_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> started_{false};
    std::atomic<bool> stopRequested_{false};
};

class DlopenAgentClientFactory final : public M800ViewAgentClientFactory {
public:
    Status Create(const M800ViewServerConfig& config,
                  std::unique_ptr<M800ViewAgentClient>& client) const override
    {
        client.reset();
        if (config.agentLibPath.empty()) {
            return Status::Error(StatusCode::NOT_FOUND, "agent library path must not be empty");
        }
        if (HasEmbeddedNul(config.agentLibPath)) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "agent library path contains embedded NUL");
        }

        dlerror();
        void* library = dlopen(config.agentLibPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            const char* openError = dlerror();
            std::string message = "failed to load agent library " + config.agentLibPath;
            if (openError != nullptr) { message += ": " + std::string{openError}; }
            KV_ERROR("dlopen failed, lib={}, err={}", config.agentLibPath, message);
            return Status::Error(StatusCode::NOT_FOUND, message);
        }

        AgentApi api;
        std::string loadError;
        if (!LoadApi(library, api, loadError)) {
            dlclose(library);
            return Status::Error(StatusCode::NOT_FOUND,
                                 "incompatible agent library " + config.agentLibPath + ": " +
                                     loadError);
        }
        const auto version = api.getAbiVersion();
        if (version != KV_CLIENT_AGENT_ABI_VERSION) {
            dlclose(library);
            return Status::Error(StatusCode::UNSUPPORTED,
                                 "agent ABI version mismatch: expected " +
                                     std::to_string(KV_CLIENT_AGENT_ABI_VERSION) + ", got " +
                                     std::to_string(version));
        }

        kv_client_agent_handle* handle = api.create();
        if (handle == nullptr) {
            dlclose(library);
            return Status::Error(StatusCode::INTERNAL_ERROR, "agent create returned null");
        }
        try {
            std::unique_ptr<M800ViewAgentClient> created{
                new DlopenAgentClient(library, handle, std::move(api), config)};
            client = std::move(created);
            KV_INFO("KVClientAgent loaded from {}", config.agentLibPath);
            return Status::OK();
        } catch (...) {
            api.destroy(handle);
            dlclose(library);
            return Status::Error(StatusCode::INTERNAL_ERROR, "failed to create agent client");
        }
    }
};

}  // namespace

std::unique_ptr<M800ViewAgentClientFactory> CreateM800ViewAgentClientFactory()
{
    return std::unique_ptr<M800ViewAgentClientFactory>{new (std::nothrow) DlopenAgentClientFactory};
}

}  // namespace kv
