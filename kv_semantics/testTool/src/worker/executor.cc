/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "worker/executor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "log/logger_manager.h"
#include "worker/client.h"
#include "worker/request.h"

namespace {

/** 基础验证的 entry 数与单条 value 字节数 */
constexpr std::size_t kEntryCount = 2;
constexpr std::size_t kValueSize = 1024;
/** 读取前回读 buffer 的填充值：读回后若仍是该值说明数据未真正回填 */
constexpr std::uint8_t kLoadFillByte = 0xA5;

/** 生成测试 key：毫秒时间戳低 32 位 + entry 序号，保证每次运行都是全新 key（查询断言不受历史数据干扰） */
kv::CacheKey MakeTestKey(std::uint32_t stamp, std::uint32_t index)
{
    kv::CacheKey key{};
    std::memcpy(key.data(), &stamp, sizeof(stamp));
    std::memcpy(key.data() + sizeof(stamp), &index, sizeof(index));
    return key;
}

/** 校验任务结果：任务级与逐 entry 状态全 OK 才通过，异常项逐条打日志 */
bool CheckTaskResult(const kv::TaskResult &result, const char *opName)
{
    bool allOk = result.status.ok();
    if (!allOk) {
        LOG_WARN("Executor：{} 任务级状态异常: code={} message={}", opName,
                 static_cast<int>(result.status.code), result.status.message);
    }
    for (std::size_t i = 0; i < result.entryStatus.size(); ++i) {
        if (!result.entryStatus[i].ok()) {
            allOk = false;
            LOG_WARN("Executor：{} entry {} 状态异常: code={} message={}", opName, i,
                     static_cast<int>(result.entryStatus[i].code), result.entryStatus[i].message);
        }
    }
    return allOk;
}

/** 提交查询并等待，校验所有 key 的存在性与 expect 一致 */
bool CheckExists(Client &client, const std::vector<kv::CacheKey> &keys, bool expect,
                 const char *stage)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    if (!client.QueryAsync(keys, taskId, err)) {
        LOG_ERROR("Executor：{}查询提交失败: {}", stage, err);
        return false;
    }
    kv::TaskResult result;
    if (!client.Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：{}查询等待失败: {}", stage, err);
        return false;
    }
    // entryStatus 中的 KEY_NOT_EXIST/KEY_EXIST 是存在性回答的编码（不算异常），只校验任务级状态
    if (!result.status.ok()) {
        LOG_ERROR("Executor：{}查询任务级状态异常: code={} message={}", stage,
                  static_cast<int>(result.status.code), result.status.message);
        return false;
    }
    if (!result.queryResult.has_value() || result.queryResult->exists.size() != keys.size()) {
        LOG_ERROR("Executor：{}查询结果数量异常", stage);
        return false;
    }
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (result.queryResult->exists[i] != static_cast<std::uint8_t>(expect ? 1 : 0)) {
            LOG_ERROR("Executor：{}查询 entry {} 存在性异常: exists={} 期望={}", stage, i,
                      static_cast<int>(result.queryResult->exists[i]),
                      static_cast<int>(expect ? 1 : 0));
            return false;
        }
    }
    LOG_INFO("Executor：{}查询通过，{} 个 key 存在性均为 {}", stage, keys.size(),
             expect ? "存在" : "不存在");
    return true;
}

/** 提交写入任务并等待完成，校验任务级与 entry 级状态 */
bool StoreAndWait(Client &client, const std::vector<kv::KVBuffer> &entries)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    if (!client.StoreAsync(entries, taskId, err)) {
        LOG_ERROR("Executor：写入任务提交失败: {}", err);
        return false;
    }
    kv::TaskResult result;
    if (!client.Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：写入任务等待失败: {}", err);
        return false;
    }
    return CheckTaskResult(result, "写入");
}

/** 提交读取任务并等待完成，校验任务级与 entry 级状态 */
bool LoadAndWait(Client &client, const std::vector<kv::KVBuffer> &entries)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    if (!client.LoadAsync(entries, taskId, err)) {
        LOG_ERROR("Executor：读取任务提交失败: {}", err);
        return false;
    }
    kv::TaskResult result;
    if (!client.Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：读取任务等待失败: {}", err);
        return false;
    }
    return CheckTaskResult(result, "读取");
}

/** 提交删除任务并等待完成，校验任务级与 entry 级状态 */
bool DeleteAndWait(Client &client, const std::vector<kv::CacheKey> &keys)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    if (!client.DeleteAsync(keys, taskId, err)) {
        LOG_ERROR("Executor：删除任务提交失败: {}", err);
        return false;
    }
    kv::TaskResult result;
    if (!client.Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：删除任务等待失败: {}", err);
        return false;
    }
    return CheckTaskResult(result, "删除");
}

} // namespace

void Executor::SetClient(Client *client)
{
    client_ = client;
}

bool Executor::Run(std::string &err)
{
    if (client_ == nullptr) {
        err = "Executor 未绑定 kv client";
        return false;
    }
    Client &client = *client_;

    // key 用毫秒时间戳 + 序号派生，保证每次运行都是全新 key
    const auto stamp = static_cast<std::uint32_t>(
        std::chrono::system_clock::now().time_since_epoch().count() / 1000000);

    // 组装请求数据包：全新 key + 写入 pattern；期望副本须在注册与读写前留存
    Request request;
    for (std::uint32_t i = 0; i < kEntryCount; ++i) {
        std::vector<std::uint8_t> value(kValueSize);
        for (std::size_t j = 0; j < value.size(); ++j) {
            value[j] = static_cast<std::uint8_t>(i * 131 + j);
        }
        request.Add(MakeTestKey(stamp, i), value);
    }
    const std::vector<std::vector<std::uint8_t>> expected = request.valueBuffers;

    // 注册值 buffer 并组装请求项
    if (!request.Register(client, err)) {
        LOG_ERROR("Executor：内存区域注册失败: {}", err);
        return false;
    }
    LOG_INFO("Executor：{} 块值 buffer 内存区域注册成功", request.handles.size());

    bool allPassed = true;

    // 1. 写入前查询：新 key 应全部不存在
    if (!CheckExists(client, request.keys, false, "写入前")) {
        err = "写入前查询验证失败";
        allPassed = false;
    }

    // 2. 异步写入并等待完成
    if (allPassed) {
        if (StoreAndWait(client, request.entries)) {
            LOG_INFO("Executor：写入 {} 条数据完成", kEntryCount);
        } else {
            err = "写入验证失败";
            allPassed = false;
        }
    }

    // 3. 写入后查询：应全部存在
    if (allPassed) {
        if (!CheckExists(client, request.keys, true, "写入后")) {
            err = "写入后查询验证失败";
            allPassed = false;
        }
    }

    // 4. 回读比对：先填 0xA5 再读回，逐字节与写入 pattern 比对
    if (allPassed) {
        for (auto &buffer : request.valueBuffers) {
            std::fill(buffer.begin(), buffer.end(), kLoadFillByte);
        }
        bool loadPassed = LoadAndWait(client, request.entries);
        for (std::size_t i = 0; i < kEntryCount; ++i) {
            if (std::memcmp(request.valueBuffers[i].data(), expected[i].data(), kValueSize) != 0) {
                LOG_ERROR("Executor：entry {} 回读数据与写入不一致", i);
                loadPassed = false;
            }
        }
        if (loadPassed) {
            LOG_INFO("Executor：回读比对通过，{} 条数据逐字节一致", kEntryCount);
        } else {
            err = "回读比对验证失败";
        }
        allPassed = loadPassed && allPassed;
    }

    // 5. 异步删除并等待完成
    if (allPassed) {
        if (DeleteAndWait(client, request.keys)) {
            LOG_INFO("Executor：删除 {} 条数据完成", kEntryCount);
        } else {
            err = "删除验证失败";
            allPassed = false;
        }
    }

    // 6. 删除后查询：应全部不存在
    if (allPassed) {
        if (!CheckExists(client, request.keys, false, "删除后")) {
            err = "删除后查询验证失败";
            allPassed = false;
        }
    }

    // 注销内存区域（全部任务已等待完成）
    if (!request.Unregister(client, err)) {
        LOG_ERROR("Executor：内存区域注销失败: {}", err);
        allPassed = false;
    }

    LOG_INFO("Executor：注册/写入/回读比对/删除全链路{}", allPassed ? "验证通过" : "验证失败");
    return allPassed;
}
