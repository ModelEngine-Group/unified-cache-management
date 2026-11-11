// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "agent_adapter.h"
#include <cstring>
#include <iostream>
#include <unistd.h>
#include "KVClientAgent.h"
#include "log.h"
#include "xxhash.h"
#define MAX_LINE_LEN 256

// 后台线程：周期性轮询Agent，检测OSD上下线事件并触发hash环更新
pthread_t update_hash_ring_tid;
bool update_thread_running = false;

NDS_ERROR_T (*client_update_cb)() = NULL;
uint64_t old_view_version = 0;
uint32_t view_query_interval = 30;  // 默认间隔30秒查询视图

agent::KVClientAgent g_agent;

// 初始化和启动agent
NDS_ERROR_T AgentInit(const char* path_config)
{
    agent::KVClientAgentConfig config;
    char log_dir[512] = {0};
    FILE* fp = fopen(path_config, "r");
    if (fp != NULL) {
        char line[MAX_LINE_LEN];
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (strstr(line, "mon_host")) {
                char val[256] = {0};
                if (sscanf(line, "mon_host = %255s", val) == 1) { config.mon_host = val; }
            } else if (strstr(line, "client_name")) {
                char val[256] = {0};
                if (sscanf(line, "client_name = %255s", val) == 1) { config.client_name = val; }
            } else if (strstr(line, "client_key")) {
                char val[256] = {0};
                if (sscanf(line, "client_key = %255s", val) == 1) { config.client_key = val; }
            } else if (strstr(line, "view_query_interval")) {
                uint32_t temp_view_query_interval;
                if (sscanf(line, "view_query_interval = %u", &temp_view_query_interval) == 1) {
                    view_query_interval = temp_view_query_interval;
                } else {
                    LOG_E("Error: Failed to parse view_query_interval value\n");
                }
            } else if (strstr(line, "log_dir")) {
                char val[512] = {0};
                if (sscanf(line, "log_dir = %511[^\n]", val) == 1) {
                    strncpy(log_dir, val, sizeof(log_dir) - 1);
                }
            }
        }
        fclose(fp);
    }

    int ret = g_agent.init(config, log_dir[0] ? log_dir : nullptr);
    if (ret != 0) {
        LOG_E("KVClientAgent init failed: %d", ret);
        return NDS_FAILED;
    }

    ret = g_agent.start();
    if (ret != 0) {
        LOG_E("KVClientAgent start failed: %d", ret);
        return NDS_FAILED;
    }

    LOG_I("KVClientAgent init and start success");
    return NDS_SUCCESS;
}

NDS_ERROR_T AgentDestroy()
{
    update_thread_running = false;
    pthread_join(update_hash_ring_tid, NULL);
    client_update_cb = NULL;
    old_view_version = 0;
    return NDS_SUCCESS;
}

uint64_t GetViewVersion() { return g_agent.get_epoch(); }

NDS_ERROR_T AgentSetUpdateCb(NDS_ERROR_T (*update_cb)(), uint64_t view_version)
{
    if (update_cb == NULL) {
        LOG_E("update_cb is NULL");
        return NDS_FAILED;
    }
    client_update_cb = update_cb;
    old_view_version = view_version;

    if (update_thread_running) { return NDS_SUCCESS; }
    update_thread_running = true;
    if (pthread_create(&update_hash_ring_tid, NULL, UpdateHashRing, NULL)) {
        update_thread_running = false;
        perror("Create UpdateHashRing thread failed");
        return NDS_FAILED;
    }
    return NDS_SUCCESS;
}

void* UpdateHashRing(void* arg)
{
    (void)arg;
    while (update_thread_running) {
        uint64_t new_view_version = GetViewVersion();
        LOG_I("new_view_version: %ld", new_view_version);
        if (new_view_version != old_view_version && client_update_cb &&
            client_update_cb() == NDS_SUCCESS) {
            LOG_W(
                "[UpdateHashRing] old_view_version:%lu -> new_view_version:%lu UpdateHashRing "
                "finished.",
                old_view_version, new_view_version);
            old_view_version = new_view_version;
        }
        sleep(view_query_interval);
    }
    return NULL;
}

// 获取所有in状态DPU的IP地址列表
std::vector<std::string> GetAllInDpuipList()
{
    std::vector<std::string> dpu_ip_list;
    auto nodes = g_agent.get_in();
    for (const auto& node : nodes) {
        for (const auto& addr : node.addrs) { dpu_ip_list.push_back(addr); }
    }
    return dpu_ip_list;
}

// 调用Agent的get_in()接口，记录Dpuid(osd_id) -> Dpuip 的映射
std::unordered_map<int32_t, std::vector<std::string>> GetAllInDpuidDpuipMap()
{
    std::unordered_map<int32_t, std::vector<std::string>> dpuid_dpuip_map;
    auto nodes = g_agent.get_in();
    for (const auto& node : nodes) {
        for (const auto& addr : node.addrs) { dpuid_dpuip_map[node.osd_id].push_back(addr); }
    }
    return dpuid_dpuip_map;
}

// 调用Agent的get_in()接口，获取所有in状态OSD的IP地址列表, 记录Dpuip -> Dpuid(osd_id) 的映射
std::unordered_map<std::string, int32_t> GetAllInDpuipDpuidMap()
{
    std::unordered_map<std::string, int32_t> dpuip_dpuid_map;
    auto nodes = g_agent.get_in();
    for (const auto& node : nodes) {
        for (const auto& addr : node.addrs) { dpuip_dpuid_map[addr] = node.osd_id; }
    }
    return dpuip_dpuid_map;
}

std::vector<std::string> GetAllNsName() { return g_agent.get_all_ns_name(); }

// 获取所有nsid下包含该namespace所对应的所有in状态DPU的IP地址
std::unordered_map<uint32_t, std::vector<std::string>> GetNsidDpuipMap(
    const std::unordered_map<std::string, uint32_t>& name_nsid_map)
{
    std::unordered_map<uint32_t, std::vector<std::string>> nsid_dpuip_map;
    for (const auto& [name, nsid] : name_nsid_map) {
        auto nodes = g_agent.get_in_by_ns(name);
        std::vector<std::string> ips;
        for (const auto& node : nodes) {
            for (const auto& addr : node.addrs) { ips.push_back(addr); }
        }
        nsid_dpuip_map[nsid] = std::move(ips);
    }
    return nsid_dpuip_map;
}

// 构建 dpu_ip -> nsid 映射表
std::unordered_map<std::string, std::unordered_set<uint32_t>> BuildDpuipNsidMap()
{
    auto name_nsid_map = GetNameNsidMap();
    if (name_nsid_map.size() == 0) {
        LOG_W("[BuildDpuipNsidMap] The size of obtained name_nsid_map is 0.");
    }
    auto nsid_dpuip_map = GetNsidDpuipMap(name_nsid_map);
    if (nsid_dpuip_map.size() == 0) {
        LOG_W("[BuildDpuipNsidMap] The size of obtained nsid_dpuip_map is 0.");
    }

    std::unordered_map<std::string, std::unordered_set<uint32_t>> dpuip_to_nsid;
    for (const auto& [nsid, ips] : nsid_dpuip_map) {
        for (const auto& ip : ips) {
            dpuip_to_nsid[ip].insert(nsid);
            LOG_W("dpuip: %s, nsid: %u", ip.c_str(), nsid);
        }
    }
    return dpuip_to_nsid;
}

// 保存namespace_name 与 nsid 的映射关系
std::unordered_map<std::string, uint32_t> GetNameNsidMap()
{
    std::unordered_map<std::string, uint32_t> name_nsid_map;
    auto ns_names = GetAllNsName();
    if (ns_names.size() == 0) { LOG_W("[GetAllNsidList] The number of obtained NS names is 0."); }
    for (const auto& name : ns_names) {
        uint32_t hash =
            XXH32(name.c_str(), name.size(), HASH_SEED_VALUE);  // hash算法和盘框测保持一致
        name_nsid_map[name] = hash;
    }
    return name_nsid_map;
}

std::vector<uint32_t> GetAllNsidList()
{
    std::vector<uint32_t> nsid_list;
    auto ns_names = GetAllNsName();
    if (ns_names.size() == 0) { LOG_W("[GetAllNsidList] The number of obtained NS names is 0."); }
    for (const auto& name : ns_names) {
        uint32_t hash =
            XXH32(name.c_str(), name.size(), HASH_SEED_VALUE);  // hash算法和盘框测保持一致
        nsid_list.push_back(hash);
    }
    return nsid_list;
}
