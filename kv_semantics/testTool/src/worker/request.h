/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_client.h"

class Client;

/**
 * kv client 请求数据包：集中保存一次读写请求的全部数据与生命周期
 * （key、值 buffer、注册句柄、组装好的请求项）。
 * 用法：Add 逐条填入 key 与值 -> Register 注册并组装 entries -> IO 读写 -> Unregister 注销。
 * 当前由 Scheduler 负责装配与注册/注销，将来 requestManager 从 input.txt 产出请求数据后交其统一管理。
 */
struct Request {
    /** key 列表（与 valueBuffers 逐项对应，Add 时成对追加） */
    std::vector<kv::CacheKey> keys;
    /** 值 buffer（IO 内存，注册后地址不得再变；生命周期随本结构体） */
    std::vector<std::vector<std::uint8_t>> valueBuffers;
    /** 组装好的 kv client 请求项（key + 已注册 buffer 句柄），Register 成功后有效 */
    std::vector<kv::KVBuffer> entries;
    /** 注册句柄（Register 成功后有效，Unregister 后清空） */
    std::vector<kv::MRHandle> handles;

    /** 追加一条 key 与值（value 拷贝保存）；须在 Register 前调用 */
    void Add(const kv::CacheKey &key, const std::vector<std::uint8_t> &value);

    /** 注册全部值 buffer 并组装 entries（HOST 内存；真实设备 DEVICE 路径需 512 字节对齐，后续扩展） */
    bool Register(Client &client, std::string &err);

    /** 注销已注册内存区域（被引用任务须已全部完成）；未注册时安全跳过 */
    bool Unregister(Client &client, std::string &err);
};
