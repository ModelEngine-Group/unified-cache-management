/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

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
 * batch 头（nsid=x / nsid x 两种形态）开始至 /batch 结束聚合、单行先试四整数再试三整数、
 * 解析失败的行警告并跳过（不影响其余输入）。
 */
class RequestManager {
public:
    /** 解析 input 文件全部指令到缓存；文件读取失败返回 false，行级解析失败仅警告跳过，全文件无可执行步骤返回 false */
    bool Load(const std::string &inputPath, std::string &err);

    /** 缓存的步骤数（单 key 行、batch、sleep 各计一步） */
    std::size_t StepCount() const;

    /** 取第 index 个步骤（index 须在 [0, StepCount())，由调用方保证） */
    const FlowStep &GetStep(std::size_t index) const;

private:
    /** 已解析的步骤序列（与 input 文件行序一致） */
    std::vector<FlowStep> steps_;
};
