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

ProbeTarget::ProbeTarget(ConnectionManager& manager, IProbeSubmitter& submitter,
                         const ProbeConfig& config, ProbeBudget* budget)
    : manager_(manager), submitter_(submitter), config_(config), budget_(budget)
{
}

void ProbeTarget::SetBudget(ProbeBudget* budget)
{
    std::lock_guard<std::mutex> lock(mu_);
    budget_ = budget;
}

void ProbeTarget::RunOnce(std::chrono::steady_clock::time_point now)
{
    std::vector<std::shared_ptr<ConnectionChannel>> snapshot;
    manager_.VisitChannels(
        [&snapshot](const std::shared_ptr<ConnectionChannel>& channel) {
            snapshot.push_back(channel);
        });
    if (snapshot.empty()) {
        cursor_ = 0;
        return;
    }

    // 游标分批：每轮最多扫描/提交 maxChannelsPerTransport 个 Channel。
    const auto batchLimit = static_cast<std::size_t>(config_.maxChannelsPerTransport);
    std::size_t visited = 0;
    std::size_t submitted = 0;
    for (std::size_t offset = 0;
         offset < snapshot.size() && visited < batchLimit; ++offset) {
        const auto& channel = snapshot[(cursor_ + offset) % snapshot.size()];
        ++visited;

        // 探针与恢复互斥：恢复窗口内跳过该组全部 Channel（含 FAILED），
        // 不提交新 PROBE、不发起复活。
        const auto group = channel->LockGroup();
        if (group && group->InRecoveryWindow()) { continue; }

        // m 轮 TIMEOUT 重试退避窗口。
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = retryEntries_.find(channel.get());
            if (it != retryEntries_.end() && now < it->second.nextAt) { continue; }
        }

        // 全局在途预算：耗尽即停止本轮提交。
        ProbeBudget* budget = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            budget = budget_;
        }
        if (budget == nullptr || !budget->TryAcquire()) {
            std::lock_guard<std::mutex> lock(mu_);
            stats_.budgetExhausted += 1;
            break;
        }

        // 探测资格：ACTIVE 且无业务在途，或 FAILED；DRAINING 排除；
        // 已有 probeInFlight 跳过。AcquireProbe 占位 probeInFlight。
        auto leased = manager_.AcquireProbe(channel);
        if (leased == nullptr) {
            if (budget != nullptr) { budget->Release(); }
            continue;
        }

        auto lease = std::make_shared<ProbeTaskLease>(budget, leased);
        const auto status = submitter_.SubmitProbe(leased, std::move(lease));
        if (!status.ok()) {
            // 提交失败立即 Release（幂等）。
            KV_DEBUG("ProbeTarget submit failed ch_id={} group_id={} code={}",
                     leased->GetChannelId(), leased->GetGroupId(),
                     static_cast<int>(status.code));
            continue;
        }
        submitted += 1;
        std::lock_guard<std::mutex> lock(mu_);
        stats_.submitted += 1;
        stats_.outstanding += 1;
    }
    cursor_ = (cursor_ + visited) % snapshot.size();
    (void)submitted;

    // 清理已被替换/移除 Channel 的重试簿记。
    std::lock_guard<std::mutex> lock(mu_);
    for (auto it = retryEntries_.begin(); it != retryEntries_.end();) {
        const bool stillPresent =
            std::find_if(snapshot.begin(), snapshot.end(),
                         [&it](const std::shared_ptr<ConnectionChannel>& channel) {
                             return channel.get() == it->first;
                         }) != snapshot.end();
        if (!stillPresent) {
            it = retryEntries_.erase(it);
        } else {
            ++it;
        }
    }
}

void ProbeTarget::HandleProbeResult(const std::shared_ptr<ConnectionChannel>& channel,
                                   ProbeResult result,
                                   std::chrono::steady_clock::time_point now)
{
    if (channel == nullptr) { return; }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stats_.outstanding > 0) { stats_.outstanding -= 1; }
    }

    switch (result) {
        case ProbeResult::FLUSH:
            // FLUSH（Send 同步返回 flush=QP 硬件 ERROR）：
            // FAILED 维持 FAILED；ACTIVE 按链路失败 ReportFailure（达 X 排空）。
            if (channel->GetState() == ChannelState::FAILED) {
                channel->SetLastProbeResult(ProbeResult::FLUSH);
            } else {
                manager_.ReportFailure(channel);
            }
            KV_INFO("ProbeTarget probe FLUSH ch_id={} group_id={} state_is_failed={}",
                    channel->GetChannelId(), channel->GetGroupId(),
                    channel->GetState() == ChannelState::FAILED);
            {
                std::lock_guard<std::mutex> lock(mu_);
                stats_.flush += 1;
            }
            break;

        case ProbeResult::SUCCESS: {
            // 合法 KeepAlive CQE 完整往返（opcode/CID/result 校验通过）。
            // FAILED 且组无恢复窗口 -> 复活（本地状态转移、计数器从零、
            // 无次数限制无冷却、不触发整组动作）。
            if (channel->GetState() == ChannelState::FAILED) {
                const auto group = channel->LockGroup();
                const bool inWindow = group && group->InRecoveryWindow();
                if (!inWindow && channel->Revive()) {
                    KV_INFO("ProbeTarget probe SUCCESS revived ch_id={} group_id={}",
                            channel->GetChannelId(), channel->GetGroupId());
                    std::lock_guard<std::mutex> lock(mu_);
                    stats_.revives += 1;
                }
            }
            // ACTIVE Channel 只完成探针：不升载、不清 errorCount。
            std::lock_guard<std::mutex> lock(mu_);
            stats_.success += 1;
            break;
        }

        case ProbeResult::TIMEOUT:
            // Send 已发出、probeTimeoutMs 内无响应。
            if (channel->GetState() == ChannelState::FAILED) {
                channel->SetLastProbeResult(ProbeResult::TIMEOUT);
                std::lock_guard<std::mutex> lock(mu_);
                stats_.timeout += 1;
                break;
            }
            // ACTIVE Channel：m 轮内重试，耗尽后 ReportFailure 排空。
            {
                std::uint32_t retries = 0;
                bool scheduleRetry = false;
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    auto& entry = retryEntries_[channel.get()];
                    entry.retries += 1;
                    retries = entry.retries;
                    if (entry.retries <= config_.probeRetryCount) {
                        entry.nextAt = now + std::chrono::milliseconds(config_.probeRetryBackoffMs);
                        scheduleRetry = true;
                        KV_DEBUG("ProbeTarget probe TIMEOUT retry scheduled ch_id={} retries={}",
                                 channel->GetChannelId(), entry.retries);
                    } else {
                        retryEntries_.erase(channel.get());
                    }
                    stats_.timeout += 1;
                }
                if (!scheduleRetry) {
                    KV_INFO("ProbeTarget probe TIMEOUT exhausted ch_id={} group_id={} "
                             "retries={} -> report failure",
                             channel->GetChannelId(), channel->GetGroupId(), retries);
                    // m 轮耗尽：按链路失败记账（X=1 即排空该 QP）。
                    manager_.ReportFailure(channel);
                }
            }
            break;

        case ProbeResult::INCONCLUSIVE:
        default:
            // INCONCLUSIVE/其他非链路结果：保持状态、lastProbeResult 不更新。
            std::lock_guard<std::mutex> lock(mu_);
            stats_.inconclusive += 1;
            break;
    }
}

ProbeTargetStats ProbeTarget::GetStats() const
{
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

}  // namespace kv
