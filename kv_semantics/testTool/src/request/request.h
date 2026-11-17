/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include "kv_client.h"
#include "request/request_orc.h"

/**
 * 单个测试任务的执行数据包：由 input.txt 的单 key 行或一个 batch 组装而成
 * （key 转换、value 块生成写 buffer、注册凭证绑定与 offset 组装由 Scheduler 完成）。
 */
struct Request {
    /** 任务级节点标识 nsid（batch 头或单 key 行的 nsid；请求明细日志输出用） */
    std::uint64_t nsid{0};
    /** key 列表（key128 低 64 位） */
    std::vector<kv::CacheKey> keys;
    /** key 128 位整数的高 16 位列表（与 keys 逐项对应；请求明细日志分开显示 key 分量用） */
    std::vector<std::uint16_t> keyHighs;
    /** input 原文 key token 列表（与 keys 逐项对应；请求明细日志显示原始 key 用） */
    std::vector<std::string> rawKeys;
    /** 各 entry 的 value 前缀长度列表（与 keys 逐项对应；请求明细日志 value 读回长度用：
     *  mode1 为 BuildValueBlock 生成前缀的实际长度，mode2 为 entry.length 完整块长度） */
    std::vector<std::size_t> prefixLens;
    /** 组装好的 kv client 请求项（绑定已注册 buffer 与 offset） */
    std::vector<kv::KVBuffer> entries;
    /** GET 回读比对基准（与 entries 逐项对应；空 vector 表示该 entry 无基准不比对） */
    std::vector<std::vector<std::uint8_t>> expected;
};

/**
 * 按 testmode 生成待写入缓冲的 value 前缀（不补齐，尺寸 ≤ entry.length；前缀之后的
 * 缓冲内容保持启动 memset 的 'A' 底色或上次写入的残留）：
 * mode1 系统生成（key128 低 64 位、高 16 位、offset、length、时间戳以空格拼接为种子串，
 * 只写一次不重复填充；用户写的 value 不生效），
 * mode2 用行尾 value 字段作前缀（超 length 截断；value 为空时不写，整块保持底色）。
 * @param entry 单条原始输入（length 与 value 的来源）
 * @param testmode 值生成模式（1 或 2）
 * @param seedTime mode1 种子串的时间戳（batch 级在 for 外生成一次共用，同批全部 entry 一致）
 * @return 前缀字节（可能为空）
 */
std::vector<std::uint8_t> BuildValueBlock(const RequestOrc &entry, int testmode,
                                          std::time_t seedTime);
