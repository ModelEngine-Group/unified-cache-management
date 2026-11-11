// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __AGENT_ADAPTER_H__
#define __AGENT_ADAPTER_H__

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common.h"

#define HASH_SEED_VALUE (5381)  // hash初始种子值，与盘框保持一致

NDS_ERROR_T AgentInit(const char* path_config);

NDS_ERROR_T AgentDestroy();

// 获取视图版本号
uint64_t GetViewVersion();

// 设置视图更新回调函数，并初始化adapter内部的视图监控相关逻辑
NDS_ERROR_T AgentSetUpdateCb(NDS_ERROR_T (*update_cb)(), uint64_t view_version);

// 调用Agent的get_in()接口，获取所有in状态OSD的IP地址列表
std::vector<std::string> GetAllInDpuipList();

// 调用Agent的get_in()接口，记录Dpuid(osd_id) -> Dpuip 的映射
std::unordered_map<int32_t, std::vector<std::string>> GetAllInDpuidDpuipMap();

// 调用Agent的get_in()接口，记录Dpuip -> Dpuid(osd_id) 的映射
std::unordered_map<std::string, int32_t> GetAllInDpuipDpuidMap();

// 返回所有的namespace_name
std::vector<std::string> GetAllNsName();

// 将namespace name -> nsid映射转换为 {nsid -> [dpu_ip]} 映射，
// 每个nsid下包含该namespace所对应的所有in状态DPU的IP地址
std::unordered_map<uint32_t, std::vector<std::string>> GetNsidDpuipMap(
    const std::unordered_map<std::string, uint32_t>& name_nsid_map);

// 构建 dpu_ip -> nsid 映射表
std::unordered_map<std::string, std::unordered_set<uint32_t>> BuildDpuipNsidMap();

// 保存namespace_name 与 nsid 的映射关系
std::unordered_map<std::string, uint32_t> GetNameNsidMap();

std::vector<uint32_t> GetAllNsidList();

// 后台线程入口：周期性调用Agent轮询OSD状态变化，
// 检测到上下线事件时调用client注册的update函数重建hash环
void* UpdateHashRing(void* arg);

#endif  // END __AGENT_ADAPTER_H__
