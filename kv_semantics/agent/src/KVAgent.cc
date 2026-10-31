/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: KV Agent 实现
*/
// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include <cerrno>
#include <memory>
#include <new>

#include "KVAgent.h"
#include "KVAgentImpl.h"

struct kv_agent_handle {
    agent::AgentLogger logger;
    std::unique_ptr<agent::KVAgentImpl> impl;
};

extern "C" KV_AGENT_EXPORT uint32_t kv_agent_get_abi_version(void)
{
    return KV_OSD_AGENT_ABI_VERSION;
}

extern "C" KV_AGENT_EXPORT kv_agent_handle *kv_agent_create(void)
{
    auto handle = std::unique_ptr<kv_agent_handle>(
        new (std::nothrow) kv_agent_handle
    );
    if (!handle) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "failed to allocate agent handle: " + std::to_string(-ENOMEM));
        return nullptr;
    }

    handle->impl = agent::create_kv_osd_agent();
    if (!handle->impl) {
        AGENT_LOG(&handle->logger, ::agent::LogLevel::ERROR, "failed to create OSD agent: " + std::to_string(-ENOMEM));
        return nullptr;
    }

    return handle.release();
}

extern "C" KV_AGENT_EXPORT int32_t kv_agent_init(
    kv_agent_handle *agent,
    const kv_osd_agent_config *config,
    uint32_t abi_version,
    size_t config_size,
    const char *log_dir)
{
    if (!agent || !agent->impl) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle: " + std::to_string(-EINVAL));
        return INVALID_AGENT_HANDLE;
    }

    int r = agent->logger.init("/home/ibc_os_hs/agent_log", "kv_agent");
    if (r < 0) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "failed to initialize file logging: " + std::to_string(r));
        return r;
    }

    if (!config) {
        AGENT_LOG(&agent->logger, ::agent::LogLevel::ERROR, "invalid configuration pointer: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    if (abi_version != KV_OSD_AGENT_ABI_VERSION) {
        AGENT_LOG(&agent->logger, ::agent::LogLevel::ERROR, "unsupported ABI version: " + std::to_string(-EPROTONOSUPPORT));
        return -EPROTONOSUPPORT;
    }

    if (config_size < sizeof(kv_osd_agent_config)) {
        AGENT_LOG(&agent->logger, ::agent::LogLevel::ERROR, "configuration is too small: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    if (config->osd_id < 0 ||
        !config->mon_host || config->mon_host[0] == '\0' ||
        !config->osd_keyring || config->osd_keyring[0] == '\0' ||
        !config->osd_uuid || config->osd_uuid[0] == '\0' ||
        config->osd_mon_heartbeat_interval == 0 ||
        !config->addrs || config->addrs[0] == '\0') {
        AGENT_LOG(&agent->logger, ::agent::LogLevel::ERROR, "invalid configuration: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    agent::KVAgentConfig impl_config{
        .dpu_id = config->dpu_id,
        .osd_id = config->osd_id,
        .mon_host = config->mon_host,
        .osd_key = config->osd_keyring,
        .osd_uuid = config->osd_uuid,
        .osd_mon_heartbeat_interval = config->osd_mon_heartbeat_interval,
        .boot_beacon_delay = config->boot_beacon_delay,
        .addrs = config->addrs,
    };

    return agent->impl->init(impl_config, &agent->logger);
}

extern "C" KV_AGENT_EXPORT int32_t kv_agent_is_initialized(
    kv_agent_handle *agent)
{
    if (!agent || !agent->impl) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle: " + std::to_string(INVALID_AGENT_HANDLE));
        return INVALID_AGENT_HANDLE;
    }

    return agent->impl->is_initialized() ? 1 : 0;
}

extern "C" KV_AGENT_EXPORT int32_t kv_agent_start_heartbeat(
    kv_agent_handle *agent)
{
    if (!agent || !agent->impl) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    return agent->impl->start_heartbeat();
}

extern "C" KV_AGENT_EXPORT int32_t kv_agent_stop_heartbeat(
    kv_agent_handle *agent)
{
    if (!agent || !agent->impl) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    return agent->impl->stop_heartbeat();
}

extern "C" KV_AGENT_EXPORT void kv_agent_destroy(kv_agent_handle *agent)
{
    delete agent;
}

// ============== 异步调用 API ==============
// 回调签名: void cb(int32_t result, void *user_data)
// result 为操作返回值; user_data 由调用方透传.
// agent 内部线程池执行 sync 实现, 完成后在 worker 线程内调用 cb.
extern "C" KV_AGENT_EXPORT int32_t kv_agent_create_crush_root_async(
    kv_agent_handle *agent, char* kvPoolName, uint32_t nameLen, int32_t* osdIdList, uint32_t listSize, kv_agent_async_cb cb, void *user_data
)
{
    if (!agent || !agent->impl || !cb) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle or callback: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    return agent->impl->kv_agent_create_crush_root_async(kvPoolName, nameLen, osdIdList, listSize, [cb](int32_t result, void *ud) { cb(result, ud); }, user_data);
}

extern "C" KV_AGENT_EXPORT int32_t kv_agent_delete_crush_root_async(
    kv_agent_handle *agent, char* kvPoolName, uint32_t nameLen, int32_t* osdIdList, uint32_t listSize, kv_agent_async_cb cb, void *user_data
)
{
    if (!agent || !agent->impl || !cb) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle or callback: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    return agent->impl->kv_agent_delete_crush_root_async(kvPoolName, nameLen, osdIdList, listSize, [cb](int32_t result, void *ud) { cb(result, ud); }, user_data);
}

extern "C" KV_AGENT_EXPORT int32_t kv_agent_create_ceph_pool_async(
    kv_agent_handle *agent, char* kvPoolName, uint32_t kvPoolLen, char* kvnsName, uint32_t kvnsLen, kv_agent_async_cb cb, void *user_data
)
{
    if (!agent || !agent->impl || !cb) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle or callback: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    return agent->impl->kv_agent_create_ceph_pool_async(kvPoolName, kvPoolLen, kvnsName, kvnsLen, [cb](int32_t result, void *ud) { cb(result, ud); }, user_data);
}

extern "C" KV_AGENT_EXPORT int32_t kv_agent_delete_ceph_pool_async(
    kv_agent_handle *agent, char* kvnsName, uint32_t nameLen, kv_agent_async_cb cb, void *user_data
)
{
    if (!agent || !agent->impl || !cb) {
        AGENT_LOG(nullptr, ::agent::LogLevel::ERROR, "invalid agent handle or callback: " + std::to_string(-EINVAL));
        return -EINVAL;
    }

    return agent->impl->kv_agent_delete_ceph_pool_async(kvnsName, nameLen, [cb](int32_t result, void *ud) { cb(result, ud); }, user_data);
}