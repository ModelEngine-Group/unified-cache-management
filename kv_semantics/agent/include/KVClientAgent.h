/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: KV Client Agent 头文件
*/

// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace agent {

#define KV_VIEW_REFRESH_INTERVAL_DEFAULT 30 /* 视图刷新周期默认值(秒) */
#define KV_VIEW_FETCH_TIMEOUT_DEFAULT    10 /* 单次拉取超时默认值(秒) */

struct KVClientAgentConfig {
    std::string mon_host;
    std::string client_name;
    std::string client_key;
    // 视图刷新配置, 可选参数, 0表示使用默认值
    int view_refresh_interval = KV_VIEW_REFRESH_INTERVAL_DEFAULT;
    int view_fetch_timeout = KV_VIEW_FETCH_TIMEOUT_DEFAULT;
};

struct KVClientNodeView {
    int32_t osd_id = -1;
    bool up = false;
    bool in = false;
    std::vector<std::string> addrs;
};

class KVClientAgent {
public:
    KVClientAgent();
    ~KVClientAgent();

    KVClientAgent(const KVClientAgent&) = delete;
    KVClientAgent& operator=(const KVClientAgent&) = delete;
    KVClientAgent(KVClientAgent&&) noexcept;
    KVClientAgent& operator=(KVClientAgent&&) noexcept;

    int init(const KVClientAgentConfig& config, const char *log_dir = nullptr);
    int start();
    uint64_t get_epoch() const;
    std::vector<KVClientNodeView> get_view() const;

    int start_view_refresh();
    int stop_view_refresh();

    // 按照ns (Ceph pool name) 查询指定状态的OSD视图
    std::vector<KVClientNodeView> get_view_by_ns(
        const std::string& ns, bool up, bool in) const;

    // 按 ns（Ceph pool name）查询对应状态的OSD视图
    std::vector<KVClientNodeView> get_up_in_by_ns(const std::string& ns) const;
    std::vector<KVClientNodeView> get_down_in_by_ns(const std::string& ns) const;
    std::vector<KVClientNodeView> get_up_out_by_ns(const std::string& ns) const;
    std::vector<KVClientNodeView> get_down_out_by_ns(const std::string& ns) const;

    // 查询所有in状态的OSD视图
    std::vector<KVClientNodeView> get_in() const;
    // 按 ns（Ceph pool name）查询所有in状态的OSD视图
    std::vector<KVClientNodeView> get_in_by_ns(const std::string& ns) const;

    // 获取所有 ns（Ceph pool name）的name列表
    std::vector<std::string> get_all_ns_name() const;

private:
    class Private;
    std::unique_ptr<Private> private_impl;
};

} // namespace agent
