// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace agent {

#define KV_VIEW_REFRESH_INTERVAL_DEFAULT 30 /* 视图刷新周期默认值(秒) */
#define KV_VIEW_FETCH_TIMEOUT_DEFAULT 10    /* 单次拉取超时默认值(秒) */

struct KVClientAgentConfig {
    std::string mon_host;
    std::string client_name;
    std::string client_key;
    // Shard8: 视图刷新配置 (可选, 0 表示使用默认值)
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

    int init(const KVClientAgentConfig& config, const char* log_dir = nullptr);
    int start();
    uint64_t get_epoch() const;
    std::vector<KVClientNodeView> get_view() const;

    // Shard8: 视图刷新控制
    // start() 已自动启动视图刷新, 此函数供独立控制场景使用
    int start_view_refresh();
    int stop_view_refresh();

    // 按 ns (Ceph pool name) 查询指定 up/in 状态的 OSD 视图。
    std::vector<KVClientNodeView> get_view_by_ns(const std::string& ns, bool up, bool in) const;

    // 按 ns 查询 up+in 状态的 OSD 视图。
    std::vector<KVClientNodeView> get_up_in_by_ns(const std::string& ns) const;
    // 按 ns 查询 down+in 状态的 OSD 视图。
    std::vector<KVClientNodeView> get_down_in_by_ns(const std::string& ns) const;
    // 按 ns 查询 up+out 状态的 OSD 视图。
    std::vector<KVClientNodeView> get_up_out_by_ns(const std::string& ns) const;
    // 按 ns 查询 down+out 状态的 OSD 视图。
    std::vector<KVClientNodeView> get_down_out_by_ns(const std::string& ns) const;

    // 查询所有 in 状态 (含 up+in 和 down+in) 的 OSD 视图。
    std::vector<KVClientNodeView> get_in() const;

    // 按 ns 查询该 ns 下所有 in 状态的 OSD 视图。
    std::vector<KVClientNodeView> get_in_by_ns(const std::string& ns) const;

    // 获取所有 ns (即 Ceph pool) 的 name 列表。
    std::vector<std::string> get_all_ns_name() const;

private:
    class Private;
    std::unique_ptr<Private> private_impl;
};

}  // namespace agent
