/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: KV Client Agent 抽象实现头文件
*/

// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#pragma once

#include <cstdint>
#include <memory>

#include "AgentLogger.h"
#include "KVClientAgent.h"

namespace agent {

class KVClientAgentImpl {
public:
    virtual ~KVClientAgentImpl() = default;
    virtual int init(const KVClientAgentConfig& config,
                     AgentLogger *logger) = 0;
    virtual int start() = 0;
    virtual uint64_t get_epoch() const = 0;
    virtual std::vector<KVClientNodeView> get_view() const = 0;
    virtual std::vector<KVClientNodeView> get_view_by_ns(
        const std::string& ns, bool up, bool in) const = 0;
    virtual std::vector<KVClientNodeView> get_up_in_by_ns(
        const std::string& ns) const = 0;
    virtual std::vector<KVClientNodeView> get_down_in_by_ns(
        const std::string& ns) const = 0;
    virtual std::vector<KVClientNodeView> get_up_out_by_ns(
        const std::string& ns) const = 0;
    virtual std::vector<KVClientNodeView> get_down_out_by_ns(
        const std::string& ns) const = 0;
    virtual std::vector<KVClientNodeView> get_in() const = 0;
    virtual std::vector<KVClientNodeView> get_in_by_ns(
        const std::string& ns) const = 0;
    virtual std::vector<std::string> get_all_ns_names() const = 0;
    virtual int start_view_refresh() = 0;
    virtual int stop_view_refresh() = 0;
};

std::unique_ptr<KVClientAgentImpl> create_kv_client_agent();

} // namespace agent