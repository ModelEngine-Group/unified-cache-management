/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */

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
 * key 解析为 128 位整数后按两个分量保存：低 64 位（CacheKey 以其十进制串构造）与 >>64 后的
 * 低 16 位（bit64-79，testmode=1 的 value 生成规则使用）。
 */
struct RequestOrc {
    /** 操作类型 */
    OrcOp op{OrcOp::GET};
    /** 请求级 namespace 名（batch 头或单 key 行的 ns；经 IoOptions.kvns 传给 kv client，
     *  由 kv client 哈希派生协议帧 namespace id，须与对端 namespace 名一致） */
    std::string ns;
    /** key 128 位整数的低 64 位（CacheKey 以其十进制串构造 string key） */
    std::uint64_t keyLow64{0};
    /** key 128 位整数右移 64 位后的低 16 位（bit64-79；testmode=1 的 value 生成规则使用） */
    std::uint16_t keyHigh16{0};
    /** input 原文 key token（明细日志显示原始 key 用） */
    std::string rawKey;
    /** 块内偏移（入 KVBuffer.offset） */
    std::uint32_t offset{0};
    /** 块大小（值 buffer 需求尺寸） */
    std::uint32_t length{0};
    /** 行尾 value 字段（testmode=2 生效；testmode=1 下写了不生效） */
    std::string value;
};
