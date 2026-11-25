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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include "kv_types.h"

namespace kv {

// Group 级动态流控（-2/+1 迟滞）：
//   - 初始化：groupNormalLimit = Σ min(config.maxInflightPerChannel,
//     ioQueueDepth)，groupBusyFloor = Σ min(config.busyFloor, perChannelLimit)，
//     currentLimit = groupNormalLimit。
//   - 合法 CQE busy：currentLimit - 2（下限 groupBusyFloor），复位稳定计时。
//   - 业务完整成功（含 query miss；探针/应用错误/部分失败/链路故障/超时除外）：
//     currentLimit + 1（上限 normalLimit）；达到 normalLimit 或稳定成功持续
//     >= stableSuccessMs → 解除降载、恢复全量。
//   - 非成功响应，或距上次成功 > stableSuccessMs（双语义：持续成功解除窗口 =
//     最大成功响应间隔）→ 复位稳定计数。
//   - 准入：actualInflight < currentLimit 才允许，达到即 fail-fast。
// 无等待队列、无 ResumeWaiters；降载态观测走 currentLimit < normalLimit
// 与结构化日志（无独立 congested 字段）。
class GroupFlowController {
public:
    GroupFlowController(const FlowControlConfig& config, std::uint32_t channelCount,
                       std::uint32_t ioQueueDepth);

    // 静态准入：达到 currentLimit 即 fail-fast。
    bool CanAcquire(std::uint32_t actualInflight) const;

    // 远端合法 CQE busy：预算 -2（下限 busyFloor），复位稳定计时。
    void OnBusy();

    // 业务完整成功：预算 +1（上限 normalLimit）+ 稳定成功解除判定。
    void OnBusinessSuccess(std::chrono::steady_clock::time_point now);

    // 非成功响应：复位稳定计数（不升载）。
    void OnNonSuccess();

    std::uint32_t CurrentLimit() const;
    std::uint32_t NormalLimit() const;
    std::uint32_t BusyFloor() const;
    bool Degraded() const;  // 派生：currentLimit < groupNormalLimit

    // busyFloor 必须小于实际（服务端 ioQueueDepth 收紧后的）每通道上限，
    // 否则 Init 拒绝（INVALID_ARGUMENT）。
    static Status ValidateConfig(const FlowControlConfig& config, std::uint32_t ioQueueDepth);

    // 工厂：先校验再构造，非法关系返回 INVALID_ARGUMENT。
    static Status Create(const FlowControlConfig& config, std::uint32_t channelCount,
                         std::uint32_t ioQueueDepth,
                         std::unique_ptr<GroupFlowController>& controller);

private:
    const FlowControlConfig config_;
    const std::uint32_t groupNormalLimit_{0};
    const std::uint32_t groupBusyFloor_{0};

    mutable std::mutex mu_;
    std::uint32_t currentLimit_{0};
    // 稳定成功窗口：从首个成功（或上次超间隔成功）起算的连续成功期。
    std::optional<std::chrono::steady_clock::time_point> stableSince_;
    std::chrono::steady_clock::time_point lastSuccessAt_{};
};

}  // namespace kv
