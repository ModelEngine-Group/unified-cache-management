/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "worker/request.h"

#include "worker/client.h"

void Request::Add(const kv::CacheKey &key, const std::vector<std::uint8_t> &value)
{
    keys.emplace_back(key);
    valueBuffers.emplace_back(value);
}

bool Request::Register(Client &client, std::string &err)
{
    if (!entries.empty() || !handles.empty()) {
        err = "Request 已注册，不可重复注册";
        return false;
    }

    // 值 buffer 注册为 HOST 内存区域（FAKE/simu 下普通堆内存即可）
    std::vector<kv::MemoryRegion> regions;
    regions.reserve(valueBuffers.size());
    for (const auto &buffer : valueBuffers) {
        kv::MemoryRegion region;
        region.memoryType = kv::MemoryType::HOST;
        region.addr = reinterpret_cast<std::uint64_t>(buffer.data());
        region.size = buffer.size();
        regions.emplace_back(region);
    }

    std::vector<kv::RegisteredMemory> registered;
    if (!client.RegisterRegions(regions, registered, err)) {
        return false;
    }

    // 组装请求项：key + 注册返回的 region 与句柄（写入与读取共用同一批 buffer）
    entries.reserve(keys.size());
    handles.reserve(registered.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        kv::KVBuffer entry;
        entry.key = keys[i];
        entry.buffer.region = registered[i].region;
        entry.buffer.handle = registered[i].handle;
        entries.emplace_back(entry);
    }
    for (const auto &item : registered) {
        handles.emplace_back(item.handle);
    }
    return true;
}

bool Request::Unregister(Client &client, std::string &err)
{
    if (handles.empty()) {
        return true;
    }
    if (!client.UnregisterRegions(handles, err)) {
        return false;
    }
    // 句柄已失效，请求项一并清空防止误用
    handles.clear();
    entries.clear();
    return true;
}
