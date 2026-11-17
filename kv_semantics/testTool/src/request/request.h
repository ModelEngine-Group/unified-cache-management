/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_client.h"
#include "request/request_orc.h"

/**
 * 单个测试任务的执行数据包：由 input.txt 的单 key 行或一个 batch 组装而成
 * （key 转换、value 块生成写 buffer、注册凭证绑定与 offset 组装由 Scheduler 完成）。
 */
struct Request {
    /** key 列表（key128 低 64 位） */
    std::vector<kv::CacheKey> keys;
    /** key 128 位整数的高 16 位列表（与 keys 逐项对应；get 读回日志拼接 key 字段用） */
    std::vector<std::uint16_t> keyHighs;
    /** 组装好的 kv client 请求项（绑定已注册 buffer 与 offset） */
    std::vector<kv::KVBuffer> entries;
    /** GET 回读比对基准（与 entries 逐项对应；空 vector 表示该 entry 无基准不比对） */
    std::vector<std::vector<std::uint8_t>> expected;
};

/**
 * 按 testmode 生成 value 块（尺寸等于 entry.length）：
 * mode1 系统生成（key128 低 64 位、高 16 位、offset、length、当前时间以空格拼接为种子串，重复填充至 length），
 * mode2 用行尾 value 字段作前缀、其余以 'A' 填充（value 为空时整块 'A'）。
 * @param entry 单条原始输入（length 与 value 的来源）
 * @param testmode 值生成模式（1 或 2）
 * @return 尺寸等于 entry.length 的块字节
 */
std::vector<std::uint8_t> BuildValueBlock(const RequestOrc &entry, int testmode);
