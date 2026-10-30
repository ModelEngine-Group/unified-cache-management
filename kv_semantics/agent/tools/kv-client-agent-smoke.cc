// -*- mode:C++; tab-width:8; c-basic-offset:2; indent-tabs-mode:t -*-
// vim: ts=8 sw=2 smarttab

#include "KVClientAgent.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

const char* get_arg(int argc, const char** argv, const std::string& name)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return nullptr;
}

bool has_flag(int argc, const char** argv, const std::string& name)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

void print_nodes(const char *label,
                 const std::vector<agent::KVClientNodeView>& nodes)
{
    for (const auto& node : nodes) {
        std::cout << label << " osd." << node.osd_id;
        for (size_t i = 0; i < node.addrs.size(); ++i) {
            std::cout << " addr[" << i << "]=" << node.addrs[i];
        }
        std::cout << std::endl;
    }
}

} // namespace

int main(int argc, const char** argv)
{
    agent::KVClientAgentConfig config{
        .mon_host = get_arg(argc, argv, "--mon-host"),
        .client_name = get_arg(argc, argv, "--client-name"),
        .client_key = get_arg(argc, argv, "--client-key"),
        .view_refresh_interval = 0, // 使用默认值 KV_VIEW_REFRESH_INTERVAL_DEFAULT
        .view_fetch_timeout = 0     // 使用默认值 KV_VIEW_FETCH_TIMEOUT_DEFAULT
    };
    const char *log_dir = get_arg(argc, argv, "--log-dir");
    const char *wait_arg = get_arg(argc, argv, "--wait");
    const char *ns_name = get_arg(argc, argv, "--ns");
    bool flag_get_in = has_flag(argc, argv, "--get-in");
    const char *get_in_ns = get_arg(argc, argv, "--get-in-by-ns");
    bool flag_get_all_ns = has_flag(argc, argv, "--get-all-ns");
    const char *up_in_ns = get_arg(argc, argv, "--get-up-in-by-ns");
    const char *down_in_ns = get_arg(argc, argv, "--get-down-in-by-ns");
    const char *up_out_ns = get_arg(argc, argv, "--get-up-out-by-ns");
    const char *down_out_ns = get_arg(argc, argv, "--get-down-out-by-ns");
    unsigned wait_seconds = wait_arg ? std::strtoul(wait_arg, nullptr, 10) : 2;
    if (config.mon_host.empty() || config.client_name.empty() ||
        config.client_key.empty()) {
        std::cerr << "usage: kv-client-agent-smoke --mon-host <hosts> "
                     "--client-name <client.name> --client-key <cephx-key> "
                     "[--log-dir <directory>] [--wait <seconds>] "
                     "[--ns <namespace>] [--get-in] "
                     "[--get-in-by-ns <namespace>] [--get-all-ns] "
                     "[--get-up-in-by-ns <ns>] [--get-down-in-by-ns <ns>] "
                     "[--get-up-out-by-ns <ns>] [--get-down-out-by-ns <ns>]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    agent::KVClientAgent client;
    int32_t result = client.init(config, log_dir);
    if (result == 0) {
        result = client.start();
    }
    if (result == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(wait_seconds));
        const auto nodes = client.get_view();
        for (const auto& node : nodes) {
            std::cout << "osd." << node.osd_id;
            for (size_t i = 0; i < node.addrs.size(); ++i) {
                std::cout << " addr[" << i << "]=" << node.addrs[i];
            }
            std::cout << std::endl;
        }

        // ns 视图 (4 种状态过滤)
        if (ns_name) {
            std::cout << "--- ns view: " << ns_name << " ---" << std::endl;
            struct NsQuery {
                const char *label;
                bool up;
                bool in;
            };
            const NsQuery queries[] = {
                {"up_in",       true, true},
                {"down_in",     false, true},
                {"up_out",      true, false},
                {"down_out",    false, false},
            };
            for (const auto& q : queries) {
                auto ns_nodes = client.get_view_by_ns(ns_name, q.up, q.in);
                if (ns_nodes.empty()) {
                    std::cout << q.label << ": (none)" << std::endl;
                    continue;
                }
                print_nodes(q.label, ns_nodes);
            }
        }

        // get_in: 全局 in 状态 OSD
        if (flag_get_in) {
            std::cout << "--- get_in ---" << std::endl;
            auto in_nodes = client.get_in();
            if (in_nodes.empty()) {
                std::cout << "get_in: (none)" << std::endl;
            } else {
                print_nodes("in", in_nodes);
            }
        }

        // get_in_by_ns: 按 ns 查询 in 状态 OSD
        if (get_in_ns) {
            std::cout << "--- get_in_by_ns: " << get_in_ns << " ---" << std::endl;
            auto in_ns_nodes = client.get_in_by_ns(get_in_ns);
            if (in_ns_nodes.empty()) {
                std::cout << "get_in_by_ns: (none)" << std::endl;
            } else {
                print_nodes("in_ns", in_ns_nodes);
            }
        }

        // get_all_ns_name: 获取所有 ns 名称
        if (flag_get_all_ns) {
            std::cout << "--- get_all_ns_name ---" << std::endl;
            auto ns_names = client.get_all_ns_name();
            if (ns_names.empty()) {
                std::cout << "get_all_ns_name: (none)" << std::endl;
            } else {
                for (size_t i = 0; i < ns_names.size(); ++i) {
                    std::cout << "ns[" << i << "]=" << ns_names[i] << std::endl;
                }
            }
        }

        // 四个具名 ns 接口
        struct NamedNsQuery {
            const char *label;
            const char *ns_arg;
            std::vector<agent::KVClientNodeView> (agent::KVClientAgent::*fn)(const std::string&) const;
        };
        const NamedNsQuery named_queries[] = {
            {"get_up_in_by_ns",     up_in_ns,       &agent::KVClientAgent::get_up_in_by_ns},
            {"get_down_in_by_ns",   down_in_ns,     &agent::KVClientAgent::get_down_in_by_ns},
            {"get_up_out_by_ns",    up_out_ns,      &agent::KVClientAgent::get_up_out_by_ns},
            {"get_down_out_by_ns",  down_out_ns,    &agent::KVClientAgent::get_down_out_by_ns},
        };
        for (const auto& q : named_queries) {
            if (!q.ns_arg) {
                continue;
            }
            std::cout << "--- " << q.label << ": " << q.ns_arg << " ---" << std::endl;
            auto nodes = (client.*q.fn)(q.ns_arg);
            if (nodes.empty()) {
                std::cout << q.label << ": (none)" << std::endl;
            } else {
                print_nodes(q.label, nodes);
            }
        }
    } else {
        std::cerr << "kv-client agent failed: " << result << std::endl;
    }
    return result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}