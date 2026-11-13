// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "cluster.h"
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "agent_adapter.h"
#include "epoll.h"
#include "utils.h"

// 集群模式下 nsid -> NsidConn 全局映射表，按namespace隔离管理连接与hash环
std::unordered_map<uint32_t, std::shared_ptr<NsidConn>> nsid_conn_map;

// nsid_conn_map 的读写锁，仅保护map结构（条目的增删查），不保护ring内容（ring有自己的锁）
// 锁序约定：map锁在外、ring锁在内，禁止持ring锁时再取map锁
static pthread_rwlock_t nsid_conn_map_lock;

// C++静态初始化对象，进程启动时自动初始化nsid_conn_map_lock
static struct NsidConnMapLockInit {
    NsidConnMapLockInit() { pthread_rwlock_init(&nsid_conn_map_lock, NULL); }
} g_nsid_conn_map_lock_init;

local_ip_param g_local_ip_param;

// 默认对hash环加读锁
hash_ring_t* NsidConn::current_ring(uint8_t* idx_ptr, bool with_ring_lock)
{
    pthread_rwlock_rdlock(&this->ring_idx_lock);
    uint8_t idx = this->ring_idx;
    pthread_rwlock_unlock(&this->ring_idx_lock);

    if (with_ring_lock) { pthread_rwlock_rdlock(&this->ring_lock[idx]); }
    if (idx_ptr) { *idx_ptr = idx; }
    return &this->ring[idx];
}

// 仅单个 rebuild 线程调用
// 默认对hash环加写锁
// 在switch前调用，获取的是新环，准备开始构建
// 在switch后调用，获取的是旧环，准备开始销毁
hash_ring_t* NsidConn::next_ring(uint8_t* idx_ptr, bool with_ring_lock)
{
    pthread_rwlock_rdlock(&this->ring_idx_lock);
    uint8_t idx = this->ring_idx ^ 1;
    pthread_rwlock_unlock(&this->ring_idx_lock);

    if (with_ring_lock) { pthread_rwlock_wrlock(&this->ring_lock[idx]); }
    if (idx_ptr) { *idx_ptr = idx; }
    return &this->ring[idx];
}

// 释放与获取必须配对；支持释放读锁或写锁
void NsidConn::release_ring_lock(uint8_t idx)
{
    if (idx == 0 || idx == 1) {
        pthread_rwlock_unlock(&this->ring_lock[idx]);
    } else {
        LOG_E("release ring lock failed: invalid idx %d", idx);
    }
}

// rebuild 完成前不得 switch
void NsidConn::switch_ring_cycle()
{
    pthread_rwlock_wrlock(&this->ring_idx_lock);
    this->ring_idx ^= 1;
    pthread_rwlock_unlock(&this->ring_idx_lock);
}

std::shared_ptr<NsidConn> FindNsidConnInMap(uint32_t nsid)
{
    pthread_rwlock_rdlock(&nsid_conn_map_lock);
    auto it = nsid_conn_map.find(nsid);
    std::shared_ptr<NsidConn> p = (it != nsid_conn_map.end()) ? it->second : nullptr;
    pthread_rwlock_unlock(&nsid_conn_map_lock);
    return p;
}

void NsidConnMapSwitchRing()
{
    // 对于nsid_conn_map而言，只是遍历，没有增删，因此只需加读锁
    pthread_rwlock_rdlock(&nsid_conn_map_lock);
    for (const auto& [nsid, nsidConn] : nsid_conn_map) {
        if (nsidConn) { nsidConn->switch_ring_cycle(); }
    }
    pthread_rwlock_unlock(&nsid_conn_map_lock);
}

void NsidConnMapReleaseOldRing()
{
    // 对于nsid_conn_map而言，只是遍历，没有增删，因此只需加读锁
    pthread_rwlock_rdlock(&nsid_conn_map_lock);
    for (const auto& [nsid, nsidConn] : nsid_conn_map) {
        if (nsidConn) {
            uint8_t ring_idx = UINT8_MAX;
            // 在switch到新环后，再调用next_ring获取到的就是之前的老环
            // 获取到next_ring时会加写锁，获取到就表示老环没有使用者了
            hash_ring_t* ring = nsidConn->next_ring(&ring_idx);
            memset(ring->nodes, 0, sizeof(hash_node_t) * ring->total_nodes);
            ring->total_nodes = 0;
            nsidConn->dpu_count[ring_idx] = 0;
            nsidConn->release_ring_lock(ring_idx);
        }
    }
    pthread_rwlock_unlock(&nsid_conn_map_lock);
}

// 判断是否启用集群模式。解析sceneMode：1=集群模式，0=单机模式，默认0
int SceneMode(const char* path_nds_config)
{
    FILE* fp = fopen(path_nds_config, "r");
    if (fp == NULL) {
        LOG_W("Unable to open config %s, sceneMode defaults to 0", path_nds_config);
        return false;
    }
    int scene_mode = 0;
    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (strstr(line, "sceneMode")) {
            if (sscanf(line, "sceneMode = %d", &scene_mode) != 1) { scene_mode = 0; }
        }
    }
    fclose(fp);
    return scene_mode;
}

// 从nds_config中解析本端所有RDMA网卡的IP、GID index、端口号及对端监听端口
local_ip_param ParseLocalipList(const char* path_nds_config)
{
    local_ip_param local_list;
    memset(&local_list, 0, sizeof(local_ip_param));
    local_list.local_ip_count = 0;
    FILE* fp = fopen(path_nds_config, "r");
    if (fp == NULL) {
        LOG_W("Unable to open config %s, using empty local ip list", path_nds_config);
        return local_list;
    }
    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (strstr(line, "local_ip_config")) {
            if (local_list.local_ip_count >= IP_NUM_T) {
                LOG_W("local_ip_config count reached max (%d), skipping", IP_NUM_T);
                continue;
            }
            char local_ip[32] = {0};
            uint64_t server_port = 0, roce_idx = 0, roce_port = 0;
            int ret =
                sscanf(line, "local_ip_config = %31[^,], %" SCNu64 ", %" SCNu64 ", %" SCNu64 "",
                       local_ip, &server_port, &roce_idx, &roce_port);
            if (ret == 4) {
                int idx = local_list.local_ip_count;
                strncpy(local_list.ip[idx].local_ip, local_ip,
                        sizeof(local_list.ip[idx].local_ip) - 1);
                local_list.ip[idx].server_port = (int)server_port;
                local_list.ip[idx].roce_idx = (int)roce_idx;
                local_list.ip[idx].roce_port = (int)roce_port;
                local_list.local_ip_count++;
            } else {
                LOG_W("local_ip_config line malformed (read %d fields): %s", ret, line);
            }
        }
    }
    fclose(fp);
    return local_list;
}

// 新建NsidConn：初始化nsid/ring_idx/is_update、两个hash环及对应的读写锁
static std::shared_ptr<NsidConn> CreateNsidConn(uint32_t nsid)
{
    auto p = std::make_shared<NsidConn>();
    p->nsid = nsid;
    p->ring_idx = 0;
    p->is_update = false;
    p->dpu_count[0] = 0;
    p->dpu_count[1] = 0;
    hash_ring_init(&p->ring[0]);
    hash_ring_init(&p->ring[1]);
    pthread_rwlock_init(&p->ring_lock[0], NULL);
    pthread_rwlock_init(&p->ring_lock[1], NULL);
    pthread_rwlock_init(&p->ring_idx_lock, NULL);
    return p;
}

// 从nsid_conn_map中查找nsid，不存在则创建并加入map
static std::shared_ptr<NsidConn> GetOrCreateNsidConn(uint32_t nsid)
{
    pthread_rwlock_wrlock(&nsid_conn_map_lock);
    auto [map_it, inserted] = nsid_conn_map.try_emplace(nsid, std::shared_ptr<NsidConn>());
    if (inserted) { map_it->second = CreateNsidConn(nsid); }
    std::shared_ptr<NsidConn> nsidConn = map_it->second;
    pthread_rwlock_unlock(&nsid_conn_map_lock);
    return nsidConn;
}

// 在NsidConn的dpus[arr_idx]中查找dpuid对应的dpu_group_t，不存在则新建并加入hash环
static dpu_group_t* FindOrCreateDpuGroup(NsidConn* nsidConn, uint8_t arr_idx, int32_t dpuid,
                                         hash_ring_t* ring)
{
    for (int j = 0; j < nsidConn->dpu_count[arr_idx]; j++) {
        if (nsidConn->dpus[arr_idx][j].dpu_id == dpuid) { return &nsidConn->dpus[arr_idx][j]; }
    }
    if (nsidConn->dpu_count[arr_idx] >= MAX_DPU_NUM) {
        LOG_E("NsidConn nsid=%u dpu_count exceeded MAX_DPU_NUM=%d", nsidConn->nsid, MAX_DPU_NUM);
        return NULL;
    }
    dpu_group_t* dpu = &nsidConn->dpus[arr_idx][nsidConn->dpu_count[arr_idx]++];
    dpu->dpu_id = dpuid;
    dpu->conn_count = 0;
    dpu->next_conn_idx = 0;
    hash_ring_add_dpu(ring, dpu);
    return dpu;
}

// 建立单条RDMA连接：注册mr -> TCP建链 -> RDMA建链，成功置online并加入epoll
static bool EstablishSingleConnection(struct rdma_connection* c, int conn_idx,
                                      const local_ip_pair_t& local_cfg, const std::string& dpu_ip)
{
    if (init_rdma_device_and_mr(c, local_cfg.local_ip, g_kv_cache, g_mr_num) != NDS_SUCCESS) {
        LOG_E("init_rdma_device_and_mr failed: local=%s dpu=%s idx=%d", local_cfg.local_ip,
              dpu_ip.c_str(), conn_idx);
        return false;
    }
    if (setup_tcp_connection(c, local_cfg.local_ip, dpu_ip.c_str(), local_cfg.server_port) !=
        NDS_SUCCESS) {
        LOG_E("TCP connection failed: local=%s dpu=%s port=%d idx=%d", local_cfg.local_ip,
              dpu_ip.c_str(), local_cfg.server_port, conn_idx);
        return false;
    }
    if (setup_rdma_connection(c, local_cfg.roce_idx, local_cfg.roce_port) != NDS_SUCCESS) {
        LOG_E("RDMA connection failed: local=%s dpu=%s idx=%d", local_cfg.local_ip, dpu_ip.c_str(),
              conn_idx);
        return false;
    }

    c->online = true;
    LOG_I("Cluster connection [%d] established: local=%s -> dpu=%s", conn_idx, local_cfg.local_ip,
          dpu_ip.c_str());
    if (nds_epoll_add_conn(c) != NDS_SUCCESS) { LOG_E("epoll add conn failed: idx=%d", conn_idx); }
    return true;
}

// 将连接加入对应nsid的hash环，若nsid不存在则新建NsidConn
void AddConnToNsidRing(
    struct rdma_connection* c, const std::string& dpu_ip,
    const std::unordered_map<std::string, std::unordered_set<uint32_t>>& dpuip_to_nsid,
    const std::unordered_map<std::string, int32_t>& dpuip_to_dpuid, bool current)
{
    auto it = dpuip_to_nsid.find(dpu_ip);
    if (it == dpuip_to_nsid.end()) {
        LOG_W("dpu_ip %s not found in any nsid, skipping hash ring add", dpu_ip.c_str());
        return;
    }

    auto dpuid_it = dpuip_to_dpuid.find(dpu_ip);
    if (dpuid_it == dpuip_to_dpuid.end()) {
        LOG_E("dpu_ip %s not found in any dpuid, skipping hash ring add", dpu_ip.c_str());
        return;
    }
    int32_t dpuid = dpuid_it->second;
    c->ip.dpu_id = dpuid;

    for (const auto& nsid : it->second) {
        std::shared_ptr<NsidConn> nsidConn = GetOrCreateNsidConn(nsid);

        uint8_t arr_idx;
        hash_ring_t* ring;
        if (current) {
            ring = nsidConn->current_ring(&arr_idx, false);
        } else {
            ring = nsidConn->next_ring(&arr_idx, false);
        }

        dpu_group_t* target_dpu = FindOrCreateDpuGroup(nsidConn.get(), arr_idx, dpuid, ring);
        if (target_dpu == NULL) { continue; }
        if (target_dpu->conn_count >= MAX_CONNS_PER_DPU) {
            LOG_E("dpu_group_t dpu_id=%d conn_count exceeded MAX_CONNS_PER_DPU=%d", dpuid,
                  MAX_CONNS_PER_DPU);
            continue;
        }
        target_dpu->conns[target_dpu->conn_count++] = c;
    }
}

NDS_ERROR_T SetupClusterConnection(
    const local_ip_param& local_list, const std::vector<std::string>& dpu_ip_list,
    const std::unordered_map<std::string, std::unordered_set<uint32_t>>& dpuip_to_nsid,
    const std::unordered_map<std::string, int32_t>& dpuip_to_dpuid, int& conn_count)
{
    if (local_list.local_ip_count <= 0 || dpu_ip_list.empty()) {
        LOG_E("SetupClusterConnection: no local ip or dpu ip provided");
        return NDS_FAILED;
    }
    int failed_conn_count = 0;
    int conn_idx = 0;
    for (int li = 0; li < local_list.local_ip_count; li++) {
        const auto& local_cfg = local_list.ip[li];
        for (size_t di = 0; di < dpu_ip_list.size(); di++) {
            const std::string& dpu_ip = dpu_ip_list[di];
            if (conn_idx >= IP_NUM_T) {
                LOG_W("SetupClusterConnection: conn_idx=%d reached IP_NUM_T=%d, skipping remaining",
                      conn_idx, IP_NUM_T);
                break;
            }
            struct rdma_connection* c = &conn[conn_idx];
            c->conn_idx = conn_idx;
            // 保存链接信息，重链时使用
            save_cluster_ip_information(c, local_cfg, dpu_ip);
            // 若建链失败, 销毁资源
            c->is_used = true;
            if (!EstablishSingleConnection(c, conn_idx, local_cfg, dpu_ip)) {
                // 链接建链失败短暂性的置为online，不置为online，进不去try_destroy_connection函数
                // 目前处于建链阶段，还没下req，置为online也不会有req
                // try_destroy_connection函数在最开始的时候再置为offline，抑制其他线程对此链接断链
                c->online = true;
                if (!try_destroy_connection(c)) { LOG_W("try_destroy_connection failed!"); }
                failed_conn_count++;
            }
            c->is_destroyed = false;
            conn_idx++;
            conn_count++;
            // 建链成功或失败都要加入hash环，失败的链接发心跳包发不过去会触发重链，这里不对失败的链接做重链
            AddConnToNsidRing(c, dpu_ip, dpuip_to_nsid, dpuip_to_dpuid, true);
        }
    }
    return failed_conn_count == conn_count ? NDS_FAILED : NDS_SUCCESS;
}

// 解析出新增的dpuiplist
std::vector<std::string> ParseNewAddDpuipList(std::vector<std::string> old_dpuips,
                                              std::vector<std::string> new_dpuips)
{
    std::unordered_set<std::string> old_set(old_dpuips.begin(), old_dpuips.end());
    std::vector<std::string> added;
    for (const auto& ip : new_dpuips) {
        if (old_set.find(ip) == old_set.end()) { added.push_back(ip); }
    }
    return added;
}

// 解析出减少的dpuiplist
std::vector<std::string> ParseDeleteDpuipList(std::vector<std::string> old_dpuips,
                                              std::vector<std::string> new_dpuips)
{
    std::unordered_set<std::string> new_set(new_dpuips.begin(), new_dpuips.end());
    std::vector<std::string> removed;
    for (const auto& ip : old_dpuips) {
        if (new_set.find(ip) == new_set.end()) { removed.push_back(ip); }
    }
    return removed;
}

// 解析出新增的nsid
std::vector<uint32_t> ParseNewAddNsids(std::vector<uint32_t> old_nsids,
                                       std::vector<uint32_t> new_nsids)
{
    std::unordered_set<uint32_t> old_set(old_nsids.begin(), old_nsids.end());
    std::vector<uint32_t> added;
    for (const auto& nsid : new_nsids) {
        if (old_set.find(nsid) == old_set.end()) { added.push_back(nsid); }
    }
    return added;
}

// 解析出减少的nsid
std::vector<uint32_t> ParseDeleteNsids(std::vector<uint32_t> old_nsids,
                                       std::vector<uint32_t> new_nsids)
{
    std::unordered_set<uint32_t> new_set(new_nsids.begin(), new_nsids.end());
    std::vector<uint32_t> removed;
    for (const auto& nsid : old_nsids) {
        if (new_set.find(nsid) == new_set.end()) { removed.push_back(nsid); }
    }
    return removed;
}

// 获取 ns 下新增的dpu ip
std::unordered_map<uint32_t, std::vector<std::string>> ParseNewAddNsDpuip(
    std::unordered_map<uint32_t, std::vector<std::string>> old_nsid_to_dpuip,
    std::unordered_map<uint32_t, std::vector<std::string>> new_nsid_to_dpuip)
{
    std::unordered_map<uint32_t, std::vector<std::string>> result;
    for (const auto& [nsid, new_ips] : new_nsid_to_dpuip) {
        std::unordered_set<std::string> old_set;
        auto old_it = old_nsid_to_dpuip.find(nsid);
        if (old_it != old_nsid_to_dpuip.end()) {
            old_set.insert(old_it->second.begin(), old_it->second.end());
        }
        for (const auto& ip : new_ips) {
            if (old_set.find(ip) == old_set.end()) { result[nsid].push_back(ip); }
        }
    }
    return result;
}

// 获取 ns 下减少的dpu ip
std::unordered_map<uint32_t, std::vector<std::string>> ParseDeleteNsDpuip(
    std::unordered_map<uint32_t, std::vector<std::string>> old_nsid_to_dpuip,
    std::unordered_map<uint32_t, std::vector<std::string>> new_nsid_to_dpuip)
{
    std::unordered_map<uint32_t, std::vector<std::string>> result;
    for (const auto& [nsid, old_ips] : old_nsid_to_dpuip) {
        std::unordered_set<std::string> new_set;
        auto new_it = new_nsid_to_dpuip.find(nsid);
        if (new_it != new_nsid_to_dpuip.end()) {
            new_set.insert(new_it->second.begin(), new_it->second.end());
        }
        for (const auto& ip : old_ips) {
            if (new_set.find(ip) == new_set.end()) { result[nsid].push_back(ip); }
        }
    }
    return result;
}

void CreateAllAddNsidConn(std::vector<uint32_t> addNsids)
{
    for (const auto& nsid : addNsids) { GetOrCreateNsidConn(nsid); }
}

void SetUpAllAddDpuipConn(std::vector<std::string> addDpuips,
                          const std::unordered_map<std::string, int32_t>& dpuip_to_dpuid)
{
    for (const auto& dpu_ip : addDpuips) {
        for (int li = 0; li < g_local_ip_param.local_ip_count; li++) {
            const auto& local_cfg = g_local_ip_param.ip[li];
            int conn_idx = FindAvailableConn();
            if (conn_idx < 0) {
                LOG_W("No available conn slot for new dpu_ip=%s, skipping", dpu_ip.c_str());
                continue;
            }
            struct rdma_connection* c = &conn[conn_idx];
            c->conn_idx = conn_idx;
            save_cluster_ip_information(c, local_cfg, dpu_ip);
            c->is_used = true;
            auto dpuid_it = dpuip_to_dpuid.find(dpu_ip);
            if (dpuid_it != dpuip_to_dpuid.end()) { c->ip.dpu_id = dpuid_it->second; }
            if (!EstablishSingleConnection(c, conn_idx, local_cfg, dpu_ip)) {
                c->online = true;
                try_destroy_connection(c);
            }
            c->is_destroyed = false;
            LOG_I("rebuild: new dpu_ip=%s added, conn_idx=%d", dpu_ip.c_str(), conn_idx);
        }
    }
}

void ModifyRemovedDpuipConnFlag(std::vector<std::string> deleteDpuip)
{
    std::unordered_set<std::string> removedSet(deleteDpuip.begin(), deleteDpuip.end());
    if (removedSet.size() <= 0) { return; }
    for (int i = 0; i < IP_NUM_T; i++) {
        if (conn[i].is_used && removedSet.count(conn[i].ip.server_ip)) {
            conn[i].is_used = false;
            LOG_I("rebuild: dpu_ip=%s is_used=false, conn_idx=%d", conn[i].ip.server_ip, i);
        }
    }
}

void DestroyRemovedOldNsidConn(std::vector<uint32_t> deleteNsids)
{
    for (const auto& nsid : deleteNsids) {
        auto it = nsid_conn_map.find(nsid);
        if (it != nsid_conn_map.end()) { nsid_conn_map.erase(it); }
    }
}

// 遍历所有链接，对is_used=true的链接加到新环里面去
void AddConnToNewRing(
    std::unordered_map<std::string, std::unordered_set<uint32_t>> new_dpuip_to_nsid,
    const std::unordered_map<std::string, int32_t>& new_dpuip_to_dpuid)
{
    for (int ip_index = 0; ip_index < IP_NUM_T; ip_index++) {
        struct rdma_connection* c = &conn[ip_index];
        if (!c->is_used) { continue; }
        string dpu_ip(c->ip.server_ip);
        AddConnToNsidRing(c, dpu_ip, new_dpuip_to_nsid, new_dpuip_to_dpuid, false);
    }
}

void DestroyRemovedConn()
{
    for (int i = 0; i < IP_NUM_T; i++) {
        if (!conn[i].is_used && !conn[i].is_reconnecting && !conn[i].is_destroyed) {
            bool destroyed = try_destroy_connection(&conn[i]);
            if (destroyed) { conn[i].is_destroyed = true; }
        }
    }
}
