// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __NDS_CLUSTER_H__
#define __NDS_CLUSTER_H__

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "NDS_client.h"

// 存放nds_config中解析的local ip、RDMA 设备信息及盘框端口
typedef struct {
    char local_ip[32];  // 本端RDMA网卡IP
    int server_port;    // 对端RDMA监听端口
    int roce_idx;       // RoCE GID index（标识RDMA网卡设备）
    int roce_port;      // RoCE端口号
} local_ip_pair_t;

typedef struct {
    local_ip_pair_t ip[IP_NUM_T];  // 本端所有RDMA网卡配置
    int local_ip_count;            // 本端RDMA网卡数量
} local_ip_param;

// 单个nsid下的连接集合，用于集群模式下按namespace管理RDMA连接
typedef struct {
    uint32_t nsid;                   // namespace id（由namespace name xxhash得到）
    hash_ring_t ring[2];             // 该nsid下的所有RDMA连接组成的hash环
    pthread_rwlock_t ring_lock[2];   // ring的读写锁
    uint8_t ring_idx;                // 有效值为0和1，表示当前使用ring/ring_lock数组的元素
    pthread_rwlock_t ring_idx_lock;  // ring_idx的读写锁
    bool is_update;

    dpu_group_t dpus[2][MAX_DPU_NUM];
    int dpu_count[2];  // 标记连接集合是否需要更新

    hash_ring_t* current_ring(uint8_t* idx_ptr, bool with_ring_lock = true);
    // 建议调用顺序：next_ring -> rebuild hash_ring -> release_ring_lock(如果有加锁的话) ->
    // switch_ring_cycle
    hash_ring_t* next_ring(uint8_t* idx_ptr, bool with_ring_lock = true);
    void release_ring_lock(uint8_t idx);
    void switch_ring_cycle();
} NsidConn;

extern local_ip_param g_local_ip_param;

// 在nsid_conn_map中通过nsid查找对应的NsidConn节点，
// 找到返回该节点的shared_ptr（持有期间NsidConn保证存活），未找到返回空shared_ptr
std::shared_ptr<NsidConn> FindNsidConnInMap(uint32_t nsid);

// 将连接加入对应nsid的hash环，若nsid不存在则新建NsidConn
// 默认添加到hash环的current_ring
void AddConnToNsidRing(
    struct rdma_connection* c, const std::string& dpu_ip,
    const std::unordered_map<std::string, std::unordered_set<uint32_t>>& dpuip_to_nsid,
    const std::unordered_map<std::string, int32_t>& dpuip_to_dpuid, bool current = true);

// 将nsid_conn_map中的所有hash环都切到新环
void NsidConnMapSwitchRing();

// 释放nsid_conn_map中的所有hash老环
void NsidConnMapReleaseOldRing();

// 判断是否启用集群模式。解析sceneMode：1=集群模式，0=单机模式，默认0
int SceneMode(const char* path_nds_config);

// 从nds_config中解析本端所有RDMA网卡的IP、GID index、端口号及对端监听端口
local_ip_param ParseLocalipList(const char* path_nds_config);

// 用本端每个RDMA网卡IP与对端每个DPU IP逐一建立RDMA连接，
// 建好的连接按nsid归入对应的NsidConn中
NDS_ERROR_T SetupClusterConnection(
    const local_ip_param& local_list, const std::vector<std::string>& dpu_ip_list,
    const std::unordered_map<std::string, std::unordered_set<uint32_t>>& dpuip_to_nsid,
    const std::unordered_map<std::string, int32_t>& dpuip_to_dpuid, int& conn_count);

std::vector<std::string> ParseNewAddDpuipList(std::vector<std::string> old_dpuips,
                                              std::vector<std::string> new_dpuips);

std::vector<std::string> ParseDeleteDpuipList(std::vector<std::string> old_dpuips,
                                              std::vector<std::string> new_dpuips);

std::vector<uint32_t> ParseNewAddNsids(std::vector<uint32_t> old_nsids,
                                       std::vector<uint32_t> new_nsids);

std::vector<uint32_t> ParseDeleteNsids(std::vector<uint32_t> old_nsids,
                                       std::vector<uint32_t> new_nsids);

std::unordered_map<uint32_t, std::vector<std::string>> ParseNewAddNsDpuip(
    std::unordered_map<uint32_t, std::vector<std::string>> old_nsid_to_dpuip,
    std::unordered_map<uint32_t, std::vector<std::string>> new_nsid_to_dpuip);

std::unordered_map<uint32_t, std::vector<std::string>> ParseDeleteNsDpuip(
    std::unordered_map<uint32_t, std::vector<std::string>> old_nsid_to_dpuip,
    std::unordered_map<uint32_t, std::vector<std::string>> new_nsid_to_dpuip);

void CreateAllAddNsidConn(std::vector<uint32_t> addNsids);
void SetUpAllAddDpuipConn(std::vector<std::string> addDpuips,
                          const std::unordered_map<std::string, int32_t>& dpuip_to_dpuid);
void ModifyRemovedDpuipConnFlag(std::vector<std::string> deleteDpuip);
void DestroyRemovedOldNsidConn(std::vector<uint32_t> deleteNsids);
void DestroyRemovedConn();
void AddConnToNewRing(
    std::unordered_map<std::string, std::unordered_set<uint32_t>> new_dpuip_to_nsid,
    const std::unordered_map<std::string, int32_t>& new_dpuip_to_dpuid);

#endif  // END __NDS_CLUSTER_H__
