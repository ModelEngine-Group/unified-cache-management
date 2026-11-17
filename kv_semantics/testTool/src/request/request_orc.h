/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <string>

/** 原始输入操作类型（input.txt 词汇表，均为 kv 操作；到执行侧 OpType 的映射由 Scheduler 完成） */
enum class OrcOp {
    GET,    // 读取（get）
    PUT,    // 写入（put）
    EXIST,  // 存在性查询（exist）
    DELETE  // 删除（delete）
};

/**
 * input.txt 单条指令的原始解析结果（原始格式）：只承载输入文件中的信息，不做执行侧加工。
 * key 解析为 128 位整数后按两个分量保存：低 64 位（CacheKey 取此值）与 >>64 后的低 16 位
 * （bit64-79，testmode=1 的 value 生成规则使用）。
 */
struct RequestOrc {
    /** 操作类型 */
    OrcOp op{OrcOp::GET};
    /** 节点标识 nsid（解析暂存，本流程不消费） */
    std::uint64_t nsid{0};
    /** key 128 位整数的低 64 位（CacheKey 取此值） */
    std::uint64_t keyLow64{0};
    /** key 128 位整数右移 64 位后的低 16 位（bit64-79；testmode=1 的 value 生成规则使用） */
    std::uint16_t keyHigh16{0};
    /** input 原文 key token（请求明细日志显示原始 key 用；与 keyLow64/keyHigh16 数值分量分开透传）
     */
    std::string rawKey;
    /** 块内偏移（入 KVBuffer.offset） */
    std::uint32_t offset{0};
    /** 块大小（值 buffer 需求尺寸） */
    std::uint32_t length{0};
    /** 行尾 value 字段（testmode=2 生效；testmode=1 下写了不生效） */
    std::string value;
};
