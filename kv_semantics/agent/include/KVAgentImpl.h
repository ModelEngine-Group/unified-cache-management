// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "AgentLogger.h"

#define KV_AGENT_MON_AUTH_TIMEOUT_SECONDS 20U

namespace agent {

using KVAgentAsyncCallback = std::function<void(int32_t result, void *user_data)>;

struct KVAgentConfig {
    uint32_t dpu_id = 0;
    int32_t osd_id = -1;
    std::string mon_host;
    std::string osd_key;
    std::string osd_uuid;
    uint32_t osd_mon_heartbeat_interval = 0;
    uint32_t boot_beacon_delay = 0;
    std::string addrs;
};

class KVAgentImpl {
public:
    virtual ~KVAgentImpl() = default;
    virtual int init(const KVAgentConfig& config, AgentLogger *logger) = 0;
    virtual bool is_initialized() const = 0;
    virtual int start_heartbeat() = 0;
    // 异步 API: 内部线程池执行 mon 命令, 完成后回调 cb. 返回 0 表示已投递.
    // 主动停止发心跳 (轻量软停): 仅停 MOSDBeacon 发送并将状态回置 DOWN,
    // 保留通信层与句柄, 可再次 start_heartbeat() 恢复: MON 连接监控/IP 切换
    // 自动恢复能力保留. 停止期间 kv_agent_is_online() 返回 false.
    virtual int stop_heartbeat() = 0;
    virtual int32_t kv_agent_create_crush_root_async(char* kvPoolName, uint32_t nameLen,
        int32_t* osdIdList, uint32_t listSize,
        KVAgentAsyncCallback cb, void* user_data) = 0;
    virtual int32_t kv_agent_delete_crush_root_async(char* kvPoolName, uint32_t nameLen,
        int32_t* osdIdList, uint32_t listSize,
        KVAgentAsyncCallback cb, void* user_data) = 0;
    virtual int32_t kv_agent_create_ceph_pool_async(char* kvPoolName, uint32_t kvPoolLen,
        char* kvnsName, uint32_t kvnsLen,
        KVAgentAsyncCallback cb, void* user_data) = 0;
    virtual int32_t kv_agent_delete_ceph_pool_async(char* kvnsName, uint32_t nameLen,
        KVAgentAsyncCallback cb, void* user_data) = 0;
};

std::unique_ptr<KVAgentImpl> create_kv_osd_agent();

} // namespace agent
