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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include "task/probe_scheduler.h"

namespace kv {

// Client 级共享探针调度器：每 Client 至多 1 个探针线程，
// 线程数与 Transport 数无关；探针在途预算跨全部 Transport 共享。
struct ProbeSchedulerStats {
    std::uint64_t rounds{0};
    std::size_t targets{0};
    std::uint64_t budgetActive{0};
    std::uint64_t budgetLimit{0};
};

class ClientProbeScheduler {
public:
    explicit ClientProbeScheduler(const ProbeConfig& config);
    ~ClientProbeScheduler();

    ClientProbeScheduler(const ClientProbeScheduler&) = delete;
    ClientProbeScheduler& operator=(const ClientProbeScheduler&) = delete;

    // 注册一个 Transport 的 ProbeTarget（Init 建链后调用）。
    void RegisterTarget(ProbeTarget* target);
    // 启动即 RunOnce，之后每 probePeriodMs + 有界 jitter 轮询。
    void Start();
    void Stop();
    // 单线程串行执行一轮（worker 调用；测试亦可直接驱动）。
    void RunOnce();

    ProbeSchedulerStats GetStats() const;
    std::size_t TargetCount() const;
    // Client 级共享在途预算（注册目标时重绑给各 ProbeTarget）。
    ProbeBudget& GetBudget() { return budget_; }

private:
    void SchedulerLoop();
    // 有界 jitter ∈ [0, probePeriodMs/4]（非阻塞时钟派生，不引入随机设施）。
    std::uint32_t NextJitterMs() const;

    ProbeConfig config_;
    // 全局在途上限派生 = maxConcurrentTransports × maxChannelsPerTransport。
    ProbeBudget budget_;

    mutable std::mutex runMu_;  // 串行化 RunOnce
    std::mutex waitMu_;
    std::condition_variable waitCv_;
    std::thread worker_;
    std::atomic<bool> running_{false};

    std::vector<ProbeTarget*> targets_;
    std::size_t transportCursor_{0};
    std::atomic<std::uint64_t> rounds_{0};
};

}  // namespace kv
