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
#include "group_flow_controller.h"
#include "logger.h"

namespace kv {

namespace {

// 每通道有效上限 = min(config.maxInflightPerChannel, ioQueueDepth)。
// ioQueueDepth 为 0 表示服务端未上报能力，直接用配置上限。
std::uint32_t EffectivePerChannelLimit(const FlowControlConfig& config,
                                      std::uint32_t ioQueueDepth)
{
    return ioQueueDepth == 0 ? config.maxInflightPerChannel
                             : std::min(config.maxInflightPerChannel, ioQueueDepth);
}

}  // namespace

GroupFlowController::GroupFlowController(const FlowControlConfig& config,
                                         std::uint32_t channelCount, std::uint32_t ioQueueDepth)
    : config_(config),
      groupNormalLimit_(EffectivePerChannelLimit(config, ioQueueDepth) * channelCount),
      // groupBusyFloor = Σ config.busyFloor；busyFloor < 每通道有效上限由
      // ValidateConfig/Create 在建组时保证。
      groupBusyFloor_(config.busyFloor * channelCount)
{
    currentLimit_ = groupNormalLimit_;
}

Status GroupFlowController::ValidateConfig(const FlowControlConfig& config,
                                            std::uint32_t ioQueueDepth)
{
    const auto effectiveLimit = EffectivePerChannelLimit(config, ioQueueDepth);
    if (config.busyFloor >= effectiveLimit) {
        return Status::Error(
            StatusCode::INVALID_ARGUMENT,
            "flow control busy_floor(" + std::to_string(config.busyFloor) +
                ") must be smaller than the effective per-channel ioQueueDepth(" +
                std::to_string(effectiveLimit) + ")");
    }
    return Status::OK();
}

Status GroupFlowController::Create(const FlowControlConfig& config, std::uint32_t channelCount,
                                   std::uint32_t ioQueueDepth,
                                   std::unique_ptr<GroupFlowController>& controller)
{
    auto status = ValidateConfig(config, ioQueueDepth);
    if (!status.ok()) { return status; }
    controller = std::make_unique<GroupFlowController>(config, channelCount, ioQueueDepth);
    return Status::OK();
}

bool GroupFlowController::CanAcquire(std::uint32_t actualInflight) const
{
    std::lock_guard<std::mutex> lock(mu_);
    return actualInflight < currentLimit_;
}

void GroupFlowController::OnBusy()
{
    std::lock_guard<std::mutex> lock(mu_);
    if (currentLimit_ > groupBusyFloor_) {
        // -2 迟滞，下限 groupBusyFloor。
        currentLimit_ = std::max(groupBusyFloor_,
                                 currentLimit_ > 2 ? currentLimit_ - 2 : groupBusyFloor_);
        KV_DEBUG("GroupFlowController ON_BUSY budget={} floor={} normal={}", currentLimit_,
                 groupBusyFloor_, groupNormalLimit_);
    }
    // 复位稳定计时：降载期间稳定窗口重新起算。
    stableSince_.reset();
}

void GroupFlowController::OnBusinessSuccess(std::chrono::steady_clock::time_point now)
{
    std::lock_guard<std::mutex> lock(mu_);
    // 双语义：距上次成功 > stableSuccessMs 视为新窗口的起点。
    if (!stableSince_.has_value() ||
        now - lastSuccessAt_ > std::chrono::milliseconds(config_.stableSuccessMs)) {
        stableSince_ = now;
    }
    lastSuccessAt_ = now;

    const bool reachedNormal =
        currentLimit_ == 0 ? false : currentLimit_ + 1 >= groupNormalLimit_;
    const bool stableSuccess =
        stableSince_.has_value() &&
        now - *stableSince_ >= std::chrono::milliseconds(config_.stableSuccessMs);
    if (reachedNormal || stableSuccess) {
        // 解除降载、恢复全量；稳定窗口重新起算。
        currentLimit_ = groupNormalLimit_;
        stableSince_ = now;
        KV_DEBUG("GroupFlowController load restored budget={} normal={}", currentLimit_,
                 groupNormalLimit_);
        return;
    }
    // 慢升：+1，上限 normalLimit。
    currentLimit_ = std::min(groupNormalLimit_, currentLimit_ + 1);
}

void GroupFlowController::OnNonSuccess()
{
    std::lock_guard<std::mutex> lock(mu_);
    // 非成功响应：复位稳定计数（不升载、不减载）。
    stableSince_.reset();
}

std::uint32_t GroupFlowController::CurrentLimit() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return currentLimit_;
}

std::uint32_t GroupFlowController::NormalLimit() const { return groupNormalLimit_; }

std::uint32_t GroupFlowController::BusyFloor() const { return groupBusyFloor_; }

bool GroupFlowController::Degraded() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return currentLimit_ < groupNormalLimit_;
}

}  // namespace kv
