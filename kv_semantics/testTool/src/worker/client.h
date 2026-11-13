/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_client.h"

/**
 * kv client 封装：配置加载、客户端生命周期（初始化建链/关闭）、内存区域注册、
 * 异步 IO 提交（写/读/查询/删除）与任务等待，worker 模块统一通过本类访问 kv client。
 */
class Client {
public:
    /** 从 kv client 配置文件初始化（加载配置、创建客户端、建链）；失败填充 err */
    bool Init(const std::string &confPath, std::string &err);
    /** 关闭并释放 kv client；未初始化或已关闭时安全跳过 */
    void Shutdown();

    /** 注册值 buffer 内存区域（读写前必须注册）；成功后 registered 与 regions 逐项对应 */
    bool RegisterRegions(const std::vector<kv::MemoryRegion> &regions,
                         std::vector<kv::RegisteredMemory> &registered, std::string &err);
    /** 注销内存区域；被在途任务引用的区域须等任务完成后再注销 */
    bool UnregisterRegions(const std::vector<kv::MRHandle> &handles, std::string &err);

    /** 异步写入：把 entries 中各 buffer 数据写入对应 key；提交成功后 taskId 用于等待结果 */
    bool StoreAsync(const std::vector<kv::KVBuffer> &entries, kv::TaskId &taskId, std::string &err);
    /** 异步读取：把各 key 的数据读回 entries 中对应 buffer；提交成功后 taskId 用于等待结果 */
    bool LoadAsync(const std::vector<kv::KVBuffer> &entries, kv::TaskId &taskId, std::string &err);
    /** 异步存在性查询；提交成功后 taskId 用于等待结果 */
    bool QueryAsync(const std::vector<kv::CacheKey> &keys, kv::TaskId &taskId, std::string &err);
    /** 异步删除；提交成功后 taskId 用于等待结果 */
    bool DeleteAsync(const std::vector<kv::CacheKey> &keys, kv::TaskId &taskId, std::string &err);

    /**
     * 阻塞等待任务完成（全链路唯一阻塞点），超时也会消费任务
     * @param timeoutMs 等待超时毫秒数，0 表示用配置的默认超时
     * @param result 任务级与逐 entry 结果（随任务带出）
     * @return false 表示等待本身失败（超时/任务不存在等）；任务级结果仍随 result 带出由调用方检查
     */
    bool Wait(kv::TaskId taskId, std::uint64_t timeoutMs, kv::TaskResult &result, std::string &err);

    /** 底层 kv client 访问入口；未初始化时为 nullptr */
    kv::KvClient *Get() const
    {
        return client_.get();
    }

private:
    std::unique_ptr<kv::KvClient> client_;
};
