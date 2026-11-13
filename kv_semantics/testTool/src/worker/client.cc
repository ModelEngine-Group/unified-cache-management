/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "worker/client.h"

#include "log/logger_manager.h"

namespace {

/** 校验 kv client 已初始化；未初始化时填 err 并返回 false */
bool CheckReady(const std::unique_ptr<kv::KvClient> &client, std::string &err)
{
    if (client == nullptr) {
        err = "kv client 未初始化";
        return false;
    }
    return true;
}

} // namespace

bool Client::Init(const std::string &confPath, std::string &err)
{
    kv::KvClientConfig config;
    kv::Status status = kv::KvClientConfig::LoadFromFile(confPath, config);
    if (!status.ok()) {
        err = "kv client 配置加载失败: " + status.message;
        return false;
    }

    client_ = kv::CreateKvClient();
    status = client_->Init(config);
    if (!status.ok()) {
        err = "kv client 初始化失败: " + status.message;
        client_.reset();
        return false;
    }
    return true;
}

void Client::Shutdown()
{
    if (client_ == nullptr) {
        return;
    }
    kv::Status status = client_->Shutdown();
    if (!status.ok()) {
        LOG_WARN("kv client 关闭异常: {}", status.message);
    }
    client_.reset();
}

bool Client::RegisterRegions(const std::vector<kv::MemoryRegion> &regions,
                            std::vector<kv::RegisteredMemory> &registered, std::string &err)
{
    if (!CheckReady(client_, err)) {
        return false;
    }
    kv::Status status = client_->RegisterRegions(regions, registered);
    if (!status.ok()) {
        err = "内存区域注册失败: " + status.message;
        return false;
    }
    return true;
}

bool Client::UnregisterRegions(const std::vector<kv::MRHandle> &handles, std::string &err)
{
    if (!CheckReady(client_, err)) {
        return false;
    }
    kv::Status status = client_->UnregisterRegions(handles);
    if (!status.ok()) {
        err = "内存区域注销失败: " + status.message;
        return false;
    }
    return true;
}

bool Client::StoreAsync(const std::vector<kv::KVBuffer> &entries, kv::TaskId &taskId,
                        std::string &err)
{
    if (!CheckReady(client_, err)) {
        return false;
    }
    kv::Status status = client_->StoreAsync(entries, taskId);
    if (!status.ok()) {
        err = "写入任务提交失败: " + status.message;
        return false;
    }
    return true;
}

bool Client::LoadAsync(const std::vector<kv::KVBuffer> &entries, kv::TaskId &taskId,
                       std::string &err)
{
    if (!CheckReady(client_, err)) {
        return false;
    }
    kv::Status status = client_->LoadAsync(entries, taskId);
    if (!status.ok()) {
        err = "读取任务提交失败: " + status.message;
        return false;
    }
    return true;
}

bool Client::QueryAsync(const std::vector<kv::CacheKey> &keys, kv::TaskId &taskId, std::string &err)
{
    if (!CheckReady(client_, err)) {
        return false;
    }
    kv::Status status = client_->QueryAsync(keys, taskId);
    if (!status.ok()) {
        err = "查询任务提交失败: " + status.message;
        return false;
    }
    return true;
}

bool Client::DeleteAsync(const std::vector<kv::CacheKey> &keys, kv::TaskId &taskId, std::string &err)
{
    if (!CheckReady(client_, err)) {
        return false;
    }
    kv::Status status = client_->DeleteAsync(keys, taskId);
    if (!status.ok()) {
        err = "删除任务提交失败: " + status.message;
        return false;
    }
    return true;
}

bool Client::Wait(kv::TaskId taskId, std::uint64_t timeoutMs, kv::TaskResult &result,
                 std::string &err)
{
    if (!CheckReady(client_, err)) {
        return false;
    }
    kv::Status status = client_->Wait(taskId, timeoutMs, result);
    if (!status.ok()) {
        err = "等待任务完成失败: " + status.message;
        return false;
    }
    return true;
}
