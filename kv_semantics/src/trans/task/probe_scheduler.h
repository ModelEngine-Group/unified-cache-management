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
#include <unordered_map>
#include <vector>
#include "conn/connection_internal.h"
#include "conn/connection_manager.h"
#include "kv_types.h"

namespace kv {

// 全局在途探针预算：limit = maxConcurrentTransports × maxChannelsPerTransport
// （派生上限，不可独立配置）。CAS 计数，超限拒绝。
class ProbeBudget {
public:
    explicit ProbeBudget(std::uint64_t limit) : limit_(limit) {}

    bool TryAcquire()
    {
        auto current = active_.load(std::memory_order_acquire);
        while (current < limit_) {
            if (active_.compare_exchange_weak(current, current + 1,
                                              std::memory_order_acq_rel)) {
                return true;
            }
        }
        return false;
    }
    void Release() { active_.fetch_sub(1, std::memory_order_acq_rel); }

    std::uint64_t Active() const { return active_.load(std::memory_order_acquire); }
    std::uint64_t Limit() const { return limit_; }

private:
    const std::uint64_t limit_;
    std::atomic<std::uint64_t> active_{0};
};

// 探针租约：预算份额 + Channel probeInFlight 计数（经 channel_ 递减）。
// 完成/超时/取消/Shutdown/提交失败均幂等 Release。
class ProbeTaskLease {
public:
    ProbeTaskLease(ProbeBudget* budget, std::shared_ptr<ConnectionChannel> channel)
        : budget_(budget), channel_(std::move(channel))
    {
    }

    ~ProbeTaskLease()
    {
        // Shutdown/任务丢弃路径的兜底释放。
        Release();
    }

    // 幂等释放：预算归还 + Channel probeInFlight 清零，仅生效一次。
    bool Release()
    {
        if (released_.exchange(true, std::memory_order_acq_rel)) { return false; }
        if (budget_ != nullptr) { budget_->Release(); }
        if (channel_ != nullptr) { channel_->FinishProbe(); }
        return true;
    }

    const std::shared_ptr<ConnectionChannel>& Channel() const { return channel_; }
    bool Released() const { return released_.load(std::memory_order_acquire); }

private:
    ProbeBudget* budget_{nullptr};
    std::shared_ptr<ConnectionChannel> channel_;
    std::atomic<bool> released_{false};
};

// 探针提交接口：由传输层注入（KeepAlive TransportTask 路径）；
// 独立使用 AsuTransport 时由调用方提供调度与预算。
class IProbeSubmitter {
public:
    virtual ~IProbeSubmitter() = default;
    // lease 的所有权随提交转移；提交失败时由实现立即 Release。
    virtual Status SubmitProbe(const std::shared_ptr<ConnectionChannel>& channel,
                                std::shared_ptr<ProbeTaskLease> lease) = 0;
};

// 传输层探针目标结论统计（GetStats 观测面）。
struct ProbeTargetStats {
    std::uint64_t submitted{0};
    std::uint64_t success{0};
    std::uint64_t timeout{0};
    std::uint64_t flush{0};
    std::uint64_t inconclusive{0};
    std::uint64_t revives{0};
    std::uint64_t budgetExhausted{0};
    std::uint64_t outstanding{0};
};

// 无线程目标适配：被动 RunOnce（由 ClientProbeScheduler 驱动），
// 探针结果由 Executor::Poll 处理后经 HandleProbeResult 回灌。
class ProbeTarget {
public:
    // budget 可为空（独立使用时由调用方注入）；Client 注册时会重绑到
    // Client 级共享预算（每 Client 仅一份，全局在途上限派生）。
    ProbeTarget(ConnectionManager& manager, IProbeSubmitter& submitter, const ProbeConfig& config,
                ProbeBudget* budget = nullptr);

    // 绑定/重绑在途预算（Client 注册时指向调度器共享预算）。
    void SetBudget(ProbeBudget* budget);

    // 一轮被动探测：游标分批，每轮最多 maxChannelsPerTransport 个 Channel。
    // 探测白名单={ACTIVE 且 inflight==0，FAILED}；DRAINING 跳过；
    // 已有 probeInFlight 跳过；恢复窗口内整组跳过（含 FAILED，不复活）。
    void RunOnce(std::chrono::steady_clock::time_point now);

    // 探针三值结论处置（含 INCONCLUSIVE 保持语义）。
    void HandleProbeResult(const std::shared_ptr<ConnectionChannel>& channel, ProbeResult result,
                           std::chrono::steady_clock::time_point now);

    ProbeTargetStats GetStats() const;

private:
    struct RetryEntry {
        std::uint32_t retries{0};
        std::chrono::steady_clock::time_point nextAt{};
    };

    ConnectionManager& manager_;
    IProbeSubmitter& submitter_;
    ProbeConfig config_;

    mutable std::mutex mu_;
    ProbeBudget* budget_{nullptr};
    std::size_t cursor_{0};
    std::unordered_map<ConnectionChannel*, RetryEntry> retryEntries_;
    ProbeTargetStats stats_;
};

// 传输层探针暴露接口：ClientProbeScheduler 经 dynamic_cast 注册目标。
class IProbingTransport {
public:
    virtual ~IProbingTransport() = default;
    virtual ProbeTarget* GetProbeTarget() = 0;
};

}  // namespace kv
