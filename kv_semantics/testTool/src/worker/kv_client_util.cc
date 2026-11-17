/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "worker/kv_client_util.h"

#include "log/logger_manager.h"

bool KvClientUtil::Init(const std::string &confPath, std::string &err)
{
    kv::KvClientConfig config;
    kv::Status status = kv::KvClientConfig::LoadFromFile(confPath, config);
    if (!status.ok()) {
        err = "kv client config load failed: " + status.message;
        return false;
    }

    client_ = kv::CreateKvClient();
    status = client_->Init(config);
    if (!status.ok()) {
        err = "kv client init failed: " + status.message;
        client_.reset();
        return false;
    }
    initialized_ = true;
    return true;
}

void KvClientUtil::Shutdown()
{
    if (client_ == nullptr) {
        return;
    }
    kv::Status status = client_->Shutdown();
    if (!status.ok()) {
        LOG_WARN("kv client shutdown error: {}", status.message);
    }
    client_.reset();
    initialized_ = false;
}

bool KvClientUtil::RegisterRegions(const std::vector<kv::MemoryRegion> &regions,
                                   std::vector<kv::RegisteredMemory> &registered,
                                   std::string &err)
{
    if (!initialized_) {
        err = "kv client not initialized";
        return false;
    }
    if (regions.empty()) {
        return true;
    }
    kv::Status status = client_->RegisterRegions(regions, registered);
    if (!status.ok()) {
        err = "register memory regions failed: " + status.message;
        return false;
    }
    return true;
}

bool KvClientUtil::UnregisterRegions(const std::vector<kv::MRHandle> &handles, std::string &err)
{
    if (handles.empty()) {
        return true;
    }
    if (!initialized_) {
        err = "kv client not initialized";
        return false;
    }
    kv::Status status = client_->UnregisterRegions(handles);
    if (!status.ok()) {
        err = "unregister memory regions failed: " + status.message;
        return false;
    }
    return true;
}

bool KvClientUtil::StoreAsync(const std::vector<kv::KVBuffer> &entries, kv::TaskId &taskId,
                              std::string &err)
{
    if (!initialized_) {
        err = "kv client not initialized";
        return false;
    }
    kv::Status status = client_->StoreAsync(entries, taskId);
    if (!status.ok()) {
        err = "store task submit failed: " + status.message;
        return false;
    }
    return true;
}

bool KvClientUtil::LoadAsync(const std::vector<kv::KVBuffer> &entries, kv::TaskId &taskId,
                             std::string &err)
{
    if (!initialized_) {
        err = "kv client not initialized";
        return false;
    }
    kv::Status status = client_->LoadAsync(entries, taskId);
    if (!status.ok()) {
        err = "load task submit failed: " + status.message;
        return false;
    }
    return true;
}

bool KvClientUtil::QueryAsync(const std::vector<kv::CacheKey> &keys, kv::TaskId &taskId,
                              std::string &err)
{
    if (!initialized_) {
        err = "kv client not initialized";
        return false;
    }
    kv::Status status = client_->QueryAsync(keys, taskId);
    if (!status.ok()) {
        err = "query task submit failed: " + status.message;
        return false;
    }
    return true;
}

bool KvClientUtil::DeleteAsync(const std::vector<kv::CacheKey> &keys, kv::TaskId &taskId,
                               std::string &err)
{
    if (!initialized_) {
        err = "kv client not initialized";
        return false;
    }
    kv::Status status = client_->DeleteAsync(keys, taskId);
    if (!status.ok()) {
        err = "delete task submit failed: " + status.message;
        return false;
    }
    return true;
}

bool KvClientUtil::Wait(kv::TaskId taskId, std::uint64_t timeoutMs, kv::TaskResult &result,
                        std::string &err)
{
    if (!initialized_) {
        err = "kv client not initialized";
        return false;
    }
    kv::Status status = client_->Wait(taskId, timeoutMs, result);
    // 底层 Wait 的返回值即任务级结果状态：等待本身失败仅限超时与任务不存在，
    // 其余非 OK（如 PARTIAL_FAILED）是任务数据结果，交由调用方在 result 中分类观察
    if (status.code == kv::StatusCode::TIMEOUT || status.code == kv::StatusCode::TASK_NOT_FOUND) {
        err = "wait task completion failed: " + status.message;
        return false;
    }
    return true;
}
