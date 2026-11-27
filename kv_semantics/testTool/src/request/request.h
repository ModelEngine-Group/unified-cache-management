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
    /** 请求级 namespace 名（batch 头或单 key 行的 ns；经 IoOptions.kvns 传给 kv client，
     *  哈希派生协议帧 namespace——须与对端 namespace 名一致） */
    std::string ns;
    /** key 列表（key128 低 64 位） */
    std::vector<kv::CacheKey> keys;
    /** key 128 位整数的高 16 位列表（与 keys 逐项对应；请求明细日志分开显示 key 分量用） */
    std::vector<std::uint16_t> keyHighs;
    /** input 原文 key token 列表（与 keys 逐项对应；请求明细日志显示原始 key 用） */
    std::vector<std::string> rawKeys;
    /** 各 entry 的 value 前缀长度（明细日志 value 读回长度：mode1 为前缀实际长度，
     *  mode2 为完整块长度） */
    std::vector<std::size_t> prefixLens;
    /** 组装好的 kv client 请求项（绑定已注册 buffer 与 offset） */
    std::vector<kv::KVBuffer> entries;
    /** GET 回读比对基准（与 entries 逐项对应；空 vector 表示该 entry 无基准不比对） */
    std::vector<std::vector<std::uint8_t>> expected;
};

/**
 * 按 testmode 生成待写入缓冲的 value 前缀（不补齐，尺寸 ≤ entry.length）：
 * mode1 系统生成（key128 低 64 位、高 16 位、offset、length、时间戳以空格拼接为
 * 种子串，只写一次不重复填充），mode2 用行尾 value 字段作前缀（超 length 截断；
 * value 为空时不写，整块保持底色）。
 * @param entry 单条原始输入（length 与 value 的来源）
 * @param testmode 值生成模式（1 或 2）
 * @param seedTime mode1 种子串的时间戳（batch 级共用一次）
 * @return 前缀字节（可能为空）
 */
std::vector<std::uint8_t> BuildValueBlock(const RequestOrc &entry, int testmode,
                                          std::time_t seedTime);
