/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Stable C ABI for KVClientAgent.
 */

#include "KVClientAgentABI.h"

#include <cerrno>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "KVClientAgent.h"

struct kv_client_agent_handle {
    agent::KVClientAgent impl;
    bool initialized{false};
    bool started{false};
    // The current C++ agent cannot restart its refresher after Stop. Keep this
    // terminal state explicit rather than reporting a stale restart as success.
    bool stopped{false};
};

namespace {

int32_t FromAgentResult(int result)
{
    if (result == 0) { return KV_CLIENT_AGENT_OK; }
    switch (result) {
        case -EINVAL: return KV_CLIENT_AGENT_INVALID_ARGUMENT;
        case -ENOTCONN: return KV_CLIENT_AGENT_NOT_INITIALIZED;
        case -ETIMEDOUT: return KV_CLIENT_AGENT_TIMEOUT;
        default: return KV_CLIENT_AGENT_INTERNAL_ERROR;
    }
}

int32_t CheckReady(const kv_client_agent_handle* handle)
{
    if (handle == nullptr) { return KV_CLIENT_AGENT_INVALID_ARGUMENT; }
    if (!handle->initialized || !handle->started) { return KV_CLIENT_AGENT_NOT_INITIALIZED; }
    return KV_CLIENT_AGENT_OK;
}

void ReleaseNsNames(const char** nsNames, size_t nsCount) noexcept
{
    if (nsNames == nullptr) { return; }
    for (size_t i = 0; i < nsCount; ++i) { delete[] nsNames[i]; }
    delete[] nsNames;
}

void ReleaseNodeViews(kv_client_agent_node_view* views, size_t viewCount) noexcept
{
    if (views == nullptr) { return; }
    for (size_t i = 0; i < viewCount; ++i) {
        if (views[i].addrs != nullptr) {
            for (size_t j = 0; j < views[i].addr_count; ++j) { delete[] views[i].addrs[j]; }
            delete[] views[i].addrs;
            views[i].addrs = nullptr;
        }
        views[i].addr_count = 0;
    }
    delete[] views;
}

const char* CopyCString(const std::string& value)
{
    char* copy = new char[value.size() + 1];
    if (!value.empty()) { std::memcpy(copy, value.data(), value.size()); }
    copy[value.size()] = '\0';
    return copy;
}

void CopyNsNames(const std::vector<std::string>& source, const char*** nsNames, size_t* nsCount)
{
    if (source.empty()) { return; }
    const char** output = new const char*[source.size()]();
    size_t copied = 0;
    try {
        for (; copied < source.size(); ++copied) { output[copied] = CopyCString(source[copied]); }
    } catch (...) {
        ReleaseNsNames(output, source.size());
        throw;
    }
    *nsNames = output;
    *nsCount = copied;
}

void CopyNodeViews(const std::vector<agent::KVClientNodeView>& source,
                   kv_client_agent_node_view** views, size_t* viewCount)
{
    if (source.empty()) { return; }
    kv_client_agent_node_view* output = new kv_client_agent_node_view[source.size()]();
    try {
        for (size_t i = 0; i < source.size(); ++i) {
            output[i].osd_id = source[i].osd_id;
            output[i].up = source[i].up ? 1 : 0;
            output[i].in = source[i].in ? 1 : 0;
            output[i].addr_count = 0;
            if (source[i].addrs.empty()) { continue; }
            output[i].addrs = new const char*[source[i].addrs.size()]();
            output[i].addr_count = source[i].addrs.size();
            for (size_t j = 0; j < source[i].addrs.size(); ++j) {
                output[i].addrs[j] = CopyCString(source[i].addrs[j]);
            }
        }
    } catch (...) {
        ReleaseNodeViews(output, source.size());
        throw;
    }
    *views = output;
    *viewCount = source.size();
}

int32_t GetNodeViews(kv_client_agent_handle* handle, const char* nsName, int32_t up, int32_t in,
                     bool onlyIn, kv_client_agent_node_view** views, size_t* viewCount)
{
    if (views != nullptr) { *views = nullptr; }
    if (viewCount != nullptr) { *viewCount = 0; }
    if (views == nullptr || viewCount == nullptr || nsName == nullptr || nsName[0] == '\0') {
        return KV_CLIENT_AGENT_INVALID_ARGUMENT;
    }
    const auto ready = CheckReady(handle);
    if (ready != KV_CLIENT_AGENT_OK) { return ready; }
    try {
        const auto source = onlyIn ? handle->impl.get_in_by_ns(nsName)
                                   : handle->impl.get_view_by_ns(nsName, up != 0, in != 0);
        CopyNodeViews(source, views, viewCount);
        return KV_CLIENT_AGENT_OK;
    } catch (...) {
        ReleaseNodeViews(*views, *viewCount);
        *views = nullptr;
        *viewCount = 0;
        return KV_CLIENT_AGENT_INTERNAL_ERROR;
    }
}

}  // namespace

extern "C" {

KV_CLIENT_AGENT_EXPORT uint32_t kv_client_agent_get_abi_version(void)
{
    return KV_CLIENT_AGENT_ABI_VERSION;
}

KV_CLIENT_AGENT_EXPORT kv_client_agent_handle* kv_client_agent_create(void)
{
    try {
        return new (std::nothrow) kv_client_agent_handle;
    } catch (...) {
        return nullptr;
    }
}

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_init(kv_client_agent_handle* handle,
                                                    const kv_client_agent_config* config,
                                                    uint32_t abiVersion, size_t configSize)
{
    if (handle == nullptr || config == nullptr) { return KV_CLIENT_AGENT_INVALID_ARGUMENT; }
    if (abiVersion != KV_CLIENT_AGENT_ABI_VERSION) {
        return KV_CLIENT_AGENT_ABI_VERSION_MISMATCH;
    }
    if (configSize != sizeof(kv_client_agent_config)) { return KV_CLIENT_AGENT_INVALID_ARGUMENT; }
    if (handle->initialized) { return KV_CLIENT_AGENT_INTERNAL_ERROR; }
    if (config->mon_host == nullptr || config->mon_host[0] == '\0' ||
        config->client_name == nullptr || config->client_name[0] == '\0' ||
        config->client_key == nullptr || config->client_key[0] == '\0' ||
        config->view_refresh_interval < 0 || config->view_fetch_timeout < 0) {
        return KV_CLIENT_AGENT_INVALID_ARGUMENT;
    }

    try {
        agent::KVClientAgentConfig agentConfig;
        agentConfig.mon_host = config->mon_host;
        agentConfig.client_name = config->client_name;
        agentConfig.client_key = config->client_key;
        agentConfig.view_refresh_interval = config->view_refresh_interval == 0
                                                ? KV_VIEW_REFRESH_INTERVAL_DEFAULT
                                                : config->view_refresh_interval;
        agentConfig.view_fetch_timeout = config->view_fetch_timeout == 0
                                             ? KV_VIEW_FETCH_TIMEOUT_DEFAULT
                                             : config->view_fetch_timeout;
        const auto result = FromAgentResult(handle->impl.init(agentConfig, config->log_dir));
        if (result == KV_CLIENT_AGENT_OK) {
            handle->initialized = true;
            handle->started = false;
            handle->stopped = false;
        }
        return result;
    } catch (...) {
        return KV_CLIENT_AGENT_INTERNAL_ERROR;
    }
}

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_start(kv_client_agent_handle* handle)
{
    if (handle == nullptr) { return KV_CLIENT_AGENT_INVALID_ARGUMENT; }
    if (!handle->initialized) { return KV_CLIENT_AGENT_NOT_INITIALIZED; }
    if (handle->started) { return KV_CLIENT_AGENT_OK; }
    if (handle->stopped) { return KV_CLIENT_AGENT_INTERNAL_ERROR; }
    try {
        const auto result = FromAgentResult(handle->impl.start());
        if (result == KV_CLIENT_AGENT_OK) { handle->started = true; }
        return result;
    } catch (...) {
        return KV_CLIENT_AGENT_INTERNAL_ERROR;
    }
}

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_stop(kv_client_agent_handle* handle)
{
    if (handle == nullptr) { return KV_CLIENT_AGENT_INVALID_ARGUMENT; }
    if (!handle->initialized) { return KV_CLIENT_AGENT_NOT_INITIALIZED; }
    if (!handle->started) { return KV_CLIENT_AGENT_OK; }
    try {
        const auto result = FromAgentResult(handle->impl.stop_view_refresh());
        if (result == KV_CLIENT_AGENT_OK) {
            handle->started = false;
            handle->stopped = true;
        }
        return result;
    } catch (...) {
        return KV_CLIENT_AGENT_INTERNAL_ERROR;
    }
}

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_get_epoch(kv_client_agent_handle* handle,
                                                         uint64_t* epoch)
{
    if (epoch != nullptr) { *epoch = 0; }
    if (epoch == nullptr) { return KV_CLIENT_AGENT_INVALID_ARGUMENT; }
    const auto ready = CheckReady(handle);
    if (ready != KV_CLIENT_AGENT_OK) { return ready; }
    try {
        *epoch = handle->impl.get_epoch();
        return KV_CLIENT_AGENT_OK;
    } catch (...) {
        *epoch = 0;
        return KV_CLIENT_AGENT_INTERNAL_ERROR;
    }
}

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_get_all_ns_name(kv_client_agent_handle* handle,
                                                               const char*** nsNames,
                                                               size_t* nsCount)
{
    if (nsNames != nullptr) { *nsNames = nullptr; }
    if (nsCount != nullptr) { *nsCount = 0; }
    if (nsNames == nullptr || nsCount == nullptr) { return KV_CLIENT_AGENT_INVALID_ARGUMENT; }
    const auto ready = CheckReady(handle);
    if (ready != KV_CLIENT_AGENT_OK) { return ready; }
    try {
        const auto source = handle->impl.get_all_ns_name();
        CopyNsNames(source, nsNames, nsCount);
        return KV_CLIENT_AGENT_OK;
    } catch (...) {
        ReleaseNsNames(*nsNames, *nsCount);
        *nsNames = nullptr;
        *nsCount = 0;
        return KV_CLIENT_AGENT_INTERNAL_ERROR;
    }
}

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_get_view_by_ns(kv_client_agent_handle* handle,
                                                              const char* nsName, int32_t up,
                                                              int32_t in,
                                                              kv_client_agent_node_view** views,
                                                              size_t* viewCount)
{
    return GetNodeViews(handle, nsName, up, in, false, views, viewCount);
}

KV_CLIENT_AGENT_EXPORT int32_t kv_client_agent_get_in_by_ns(kv_client_agent_handle* handle,
                                                            const char* nsName,
                                                            kv_client_agent_node_view** views,
                                                            size_t* viewCount)
{
    return GetNodeViews(handle, nsName, 0, 1, true, views, viewCount);
}

KV_CLIENT_AGENT_EXPORT void kv_client_agent_free_ns_names(kv_client_agent_handle* handle,
                                                          const char** nsNames, size_t nsCount)
{
    (void)handle;
    ReleaseNsNames(nsNames, nsCount);
}

KV_CLIENT_AGENT_EXPORT void kv_client_agent_free_node_views(kv_client_agent_handle* handle,
                                                            kv_client_agent_node_view* views,
                                                            size_t viewCount)
{
    (void)handle;
    ReleaseNodeViews(views, viewCount);
}

KV_CLIENT_AGENT_EXPORT void kv_client_agent_destroy(kv_client_agent_handle* handle)
{
    if (handle == nullptr) { return; }
    try {
        handle->impl.stop_view_refresh();
    } catch (...) {
        // Destruction must continue so the library can be unloaded safely.
    }
    delete handle;
}

}  // extern "C"
