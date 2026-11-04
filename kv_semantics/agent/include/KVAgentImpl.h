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
 */
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

using KVAgentAsyncCallback = std::function<void(int32_t result, void* user_data)>;

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
    virtual int init(const KVAgentConfig& config, AgentLogger* logger) = 0;
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

}  // namespace agent
