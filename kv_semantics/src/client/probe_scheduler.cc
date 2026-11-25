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
#include "probe_scheduler.h"
#include <algorithm>
#include "logger.h"

namespace kv {

ClientProbeScheduler::ClientProbeScheduler(const ProbeConfig& config)
    : config_(config), budget_(config.DerivedInFlightLimit())
{
}

ClientProbeScheduler::~ClientProbeScheduler() { Stop(); }

void ClientProbeScheduler::RegisterTarget(ProbeTarget* target)
{
    if (target == nullptr) { return; }
    std::lock_guard<std::mutex> lock(runMu_);
    if (std::find(targets_.begin(), targets_.end(), target) == targets_.end()) {
        // 注册即重绑到 Client 级共享在途预算（全局上限派生，跨 Transport 共享）。
        target->SetBudget(&budget_);
        targets_.push_back(target);
    }
}

void ClientProbeScheduler::Start()
{
    if (running_.exchange(true, std::memory_order_acq_rel)) { return; }
    worker_ = std::thread(&ClientProbeScheduler::SchedulerLoop, this);
}

void ClientProbeScheduler::Stop()
{
    if (!running_.exchange(false, std::memory_order_acq_rel)) { return; }
    waitCv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

void ClientProbeScheduler::RunOnce()
{
    std::vector<ProbeTarget*> targets;
    {
        // 仅在捕获有限窗口时持锁（Visitor/Submit 锁外执行）。
        std::lock_guard<std::mutex> lock(runMu_);
        targets = targets_;
    }
    if (targets.empty()) { return; }

    const auto now = std::chrono::steady_clock::now();
    // 游标轮转驱动各 Transport 的目标（R(临时 refresh) 不创建额外线程）。
    const auto target = targets[transportCursor_ % targets.size()];
    transportCursor_ = (transportCursor_ + 1) % targets.size();
    target->RunOnce(now);
    rounds_.fetch_add(1, std::memory_order_release);
}

void ClientProbeScheduler::SchedulerLoop()
{
    // 启动即 RunOnce。
    RunOnce();
    while (running_.load(std::memory_order_acquire)) {
        const auto waitMs = config_.probePeriodMs + NextJitterMs();
        std::unique_lock<std::mutex> lock(waitMu_);
        waitCv_.wait_for(lock, std::chrono::milliseconds(waitMs),
                         [this] { return !running_.load(std::memory_order_acquire); });
        if (!running_.load(std::memory_order_acquire)) { break; }
        RunOnce();
    }
}

std::uint32_t ClientProbeScheduler::NextJitterMs() const
{
    // 有界 jitter ∈ [0, probePeriodMs/4]：时钟低位派生，避免随机设施。
    const auto bound = config_.probePeriodMs / 4;
    if (bound == 0) { return 0; }
    const auto entropy = static_cast<std::uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count() & 0xFFFFFFFFULL);
    return entropy % (bound + 1);
}

ProbeSchedulerStats ClientProbeScheduler::GetStats() const
{
    ProbeSchedulerStats stats;
    stats.rounds = rounds_.load(std::memory_order_acquire);
    stats.budgetActive = budget_.Active();
    stats.budgetLimit = budget_.Limit();
    std::lock_guard<std::mutex> lock(runMu_);
    stats.targets = targets_.size();
    return stats;
}

std::size_t ClientProbeScheduler::TargetCount() const
{
    std::lock_guard<std::mutex> lock(runMu_);
    return targets_.size();
}

}  // namespace kv
