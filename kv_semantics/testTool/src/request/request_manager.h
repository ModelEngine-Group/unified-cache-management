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

#include <cstddef>
#include <string>
#include <variant>
#include <vector>
#include "request/request_orc.h"

/** 一个 kv 执行任务：input.txt 的单 key 行（1 条）或一个 batch（多行聚合为一次 kv 任务） */
struct OrcTask {
    /** 任务操作类型（batch 头或单 key 行的 optype） */
    OrcOp op{OrcOp::GET};
    /** 任务级节点标识（batch 头或单 key 行的 nsid） */
    std::uint64_t nsid{0};
    /** 任务内逐条原始数据（每条对应一个 kv entry） */
    std::vector<RequestOrc> entries;
};

/**
 * sleep 暂停步骤（sleep 行）：仅表示执行流程中暂停片刻，
 * 不是 kv 操作，不属于 OrcOp，也不参与 OrcTask / Request 组装。
 */
struct SleepStep {
    /** 暂停秒数 */
    std::uint32_t seconds{0};
};

/** 任务流中的一步：要么是 kv 任务（OrcTask），要么是 sleep 暂停步骤（SleepStep） */
using FlowStep = std::variant<OrcTask, SleepStep>;

/**
 * 输入管理器：解析 input.txt 全部指令并缓存为步骤序列，供执行侧按序取用。
 * 解析规则（见 input/readme.md）：行首尾 trim（空格与 \t）、空行跳过、sleep 行解析为独立暂停步骤、
 * batch 头（可带 nsid=x / nsid x，缺省 0）开始至 /batch 结束聚合（文件结束时未闭合的 batch
 * 属非法输入，报错退出）、单行先试四整数再试三整数；entry 字段须满足 kv 协议约束
 * （key 低 64 位非全 0、offset/length 512 字节对齐、length 非 0 且不超 24 位上限），
 * 违规启动即报错；
 * 任何非法输入数据（未知 op、字段解析失败等）不跳过不静默继续，直接返回启动错误提示管理员修正。
 */
class RequestManager {
public:
    /** 解析 input 文件全部指令到缓存；文件读取失败、行级解析失败（含协议约束校验失败，
     * fail-fast 不跳过不静默继续）或全文件无可执行步骤均返回 false */
    bool Load(const std::string &inputPath, std::string &err);

    /** 缓存的步骤数（单 key 行、batch、sleep 各计一步） */
    std::size_t StepCount() const;

    /** 取第 index 个步骤（index 须在 [0, StepCount())，由调用方保证） */
    const FlowStep &GetStep(std::size_t index) const;

private:
    /** 已解析的步骤序列（与 input 文件行序一致） */
    std::vector<FlowStep> steps_;
};
