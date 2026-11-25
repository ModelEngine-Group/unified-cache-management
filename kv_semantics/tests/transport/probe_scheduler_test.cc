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

// Shard 4 DT（transport 侧）：ProbeBudget 派生上限、ProbeTaskLease 幂等释放、
// ProbeTarget 空闲检测白名单、游标分批、恢复互斥、三值结论处置。
//
// 测试因子分解：
//   F1 Channel 资格 ∈ {ACTIVE空闲, ACTIVE忙, DRAINING, FAILED, 已有探针在途}
//   F2 恢复窗口 ∈ {开, 关}（窗口内整组跳过，含 FAILED）
//   F3 游标分批 ∈ {1, 2, 4 通道/轮}；预算耗尽即停
//   F4 探针结论 ∈ {FLUSH, SUCCESS, TIMEOUT, INCONCLUSIVE} × {ACTIVE, FAILED}
//   F5 m 轮重试 ∈ {0, 1}（ACTIVE 通道 TIMEOUT 耗尽后 ReportFailure）
// 断言面：submitted/budget 探针计数、Channel 状态/lastProbeResult/计数器、
//         probeInFlight 占位与释放、幂等 Release。

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "conn/connection_internal.h"
#include "conn/connection_manager.h"
#include "task/probe_scheduler.h"
#include "trans_provider.h"

namespace kv::test {
namespace {

class StubProvider : public TransProvider {
public:
    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                            uint32_t, std::vector<ConnectionHandle>& handles) override
    {
        handles.clear();
        for (std::uint32_t i = 0; i < qpNum; ++i) {
            handles.push_back(reinterpret_cast<ConnectionHandle>(static_cast<uintptr_t>(++sequence_)));
        }
        return Status::OK();
    }
    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>&) override { return {}; }
    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>&, uint32_t,
                             uint32_t) override
    {
        return {};
    }
    Status RegisterMemory(const std::vector<RegisterMemoryDesc>&,
                          std::vector<MRHandle>&) override
    {
        return Status::OK();
    }
    std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>&) override
    {
        return {};
    }
    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }
    std::vector<Status> FreeThread(const std::vector<ThreadHandle>&) override { return {}; }
    Status GetMemTokenId(MRHandle, uint32_t& tokenId) override
    {
        tokenId = 1;
        return Status::OK();
    }

private:
    std::uint64_t sequence_{0};
};

// 可编程探针提交桩：记录提交并保留租约供断言。
class RecordingSubmitter : public IProbeSubmitter {
public:
    std::atomic<int> submits{0};
    std::atomic<bool> failNext{false};
    std::mutex leasesMu;
    std::vector<std::shared_ptr<ProbeTaskLease>> leases;

    Status SubmitProbe(const std::shared_ptr<ConnectionChannel>& channel,
                      std::shared_ptr<ProbeTaskLease> lease) override
    {
        if (failNext.load()) { return Status::Error(StatusCode::CONNECTION_ERROR, "scripted"); }
        submits.fetch_add(1);
        std::lock_guard<std::mutex> lock(leasesMu);
        leases.push_back(std::move(lease));
        return Status::OK();
    }

    void ReleaseAll()
    {
        std::lock_guard<std::mutex> lock(leasesMu);
        for (auto& lease : leases) { lease->Release(); }
        leases.clear();
    }
};

ProbeConfig MakeProbeConfig(std::uint32_t maxChannelsPerTransport = 2,
                            std::uint32_t probeRetryCount = 0,
                            std::uint32_t probeRetryBackoffMs = 0)
{
    ProbeConfig config;
    config.maxChannelsPerTransport = maxChannelsPerTransport;
    config.probeRetryCount = probeRetryCount;
    config.probeRetryBackoffMs = probeRetryBackoffMs;
    return config;
}

NodeEndpoint MakeEndpoint(const std::string& ip)
{
    NodeEndpoint ep;
    ep.ip = ip;
    ep.port = 16666;
    return ep;
}

}  // namespace

// ─── ProbeBudget：CAS 预算与边界 ───

TEST(ProbeBudgetTest, AcquireReleasesAroundDerivedLimit)
{
    ProbeConfig probeConfig;
    EXPECT_EQ(probeConfig.DerivedInFlightLimit(), std::uint64_t{4} * 16);  // 默认 4x16=64

    ProbeBudget budget(2);
    EXPECT_TRUE(budget.TryAcquire());
    EXPECT_TRUE(budget.TryAcquire());
    EXPECT_FALSE(budget.TryAcquire());  // 达上限拒绝
    EXPECT_EQ(budget.Active(), std::uint64_t{2});
    budget.Release();
    EXPECT_EQ(budget.Active(), std::uint64_t{1});
    EXPECT_TRUE(budget.TryAcquire());  // 释放后可再取
    EXPECT_EQ(budget.Active(), std::uint64_t{2});
    budget.Release();
    budget.Release();
    EXPECT_EQ(budget.Active(), std::uint64_t{0});
}

TEST(ProbeBudgetTest, ZeroLimitRejectsEverything)
{
    ProbeBudget budget(0);
    EXPECT_FALSE(budget.TryAcquire());
    EXPECT_EQ(budget.Limit(), std::uint64_t{0});
}

// ─── ProbeTaskLease：幂等释放 ───

TEST(ProbeTaskLeaseTest, ReleaseIsIdempotentAndFreesProbeSlot)
{
    StubProvider provider;
    ProbeBudget budget(4);
    ConnectionGroup group(0, MakeEndpoint("10.0.0.1"));
    auto channel =
        std::make_shared<ConnectionChannel>(0, &group, nullptr, nullptr, nullptr);

    // RunOnce 语义：预算先 TryAcquire 再构造租约。
    ASSERT_TRUE(budget.TryAcquire());
    EXPECT_EQ(budget.Active(), std::uint64_t{1});
    {
        ProbeTaskLease lease(&budget, channel);
        EXPECT_TRUE(channel->TryStartProbe());  // AcquireProbe 的占位由测试模拟
        EXPECT_TRUE(lease.Release());
        EXPECT_TRUE(lease.Released());
        EXPECT_FALSE(channel->ProbeInFlight());  // probeInFlight 经 channel 递减
        EXPECT_EQ(budget.Active(), std::uint64_t{0});

        // 幂等：重复释放不再归还预算、不再触碰 channel。
        EXPECT_FALSE(lease.Release());
        EXPECT_EQ(budget.Active(), std::uint64_t{0});
    }
    // 析构路径同样幂等：不再归还预算。
    EXPECT_EQ(budget.Active(), std::uint64_t{0});
}

TEST(ProbeTaskLeaseTest, DestructorReleasesWhenForgotten)
{
    ProbeBudget budget(4);
    {
        ASSERT_TRUE(budget.TryAcquire());
        ProbeTaskLease lease(&budget, nullptr);  // 无 channel：仅预算
        (void)lease;
    }
    EXPECT_EQ(budget.Active(), std::uint64_t{0});
}

// ─── F1: 探测白名单（空闲检测、指定 QP 的资格校验） ───

TEST(ProbeTargetTest, RunOnceSubmitsOnlyEligibleChannels)
{
    StubProvider provider;
    ConnectionManager manager(provider, "");
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    std::vector<std::shared_ptr<ConnectionChannel>> channels;
    for (int i = 0; i < 4; ++i) {
        auto channel = manager.SelectConnection();
        ASSERT_NE(channel, nullptr);
        channels.push_back(channel);
    }
    for (auto& channel : channels) { channel->ReleaseInflight(); }

    // ch0：ACTIVE 空闲 -> 可探。
    // ch1：ACTIVE 忙（有业务在途）-> 跳过。
    channels[1]->IncrementInflight();
    // ch2：DRAINING -> 跳过。
    channels[2]->MarkForDrain();
    // ch3：FAILED -> 可探。
    channels[3]->MarkForDrain();
    channels[3]->MarkFailed();

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    ProbeTarget target(manager, submitter, MakeProbeConfig(16), &budget);
    target.RunOnce(std::chrono::steady_clock::now());

    EXPECT_EQ(submitter.submits.load(), 2);
    auto stats = target.GetStats();
    EXPECT_EQ(stats.submitted, std::uint64_t{2});
    EXPECT_EQ(stats.outstanding, std::uint64_t{2});

    // 提交对象即资格白名单成员。
    std::set<const ConnectionChannel*> submittedChannels;
    {
        std::lock_guard<std::mutex> lock(submitter.leasesMu);
        for (auto& lease : submitter.leases) {
            submittedChannels.insert(lease->Channel().get());
        }
    }
    EXPECT_TRUE(submittedChannels.count(channels[0].get()) != 0);
    EXPECT_TRUE(submittedChannels.count(channels[3].get()) != 0);
    EXPECT_EQ(submittedChannels.size(), std::size_t{2});

    // 已有 probeInFlight（未释放租约）：第二轮跳过在途通道，提交数不变。
    target.RunOnce(std::chrono::steady_clock::now());
    EXPECT_EQ(submitter.submits.load(), 2);

    // 释放后在途位清零：第三轮重新探测白名单成员。
    submitter.ReleaseAll();
    target.RunOnce(std::chrono::steady_clock::now());
    EXPECT_EQ(submitter.submits.load(), 4);
    submitter.ReleaseAll();
}

TEST(ProbeTargetTest, SubmitFailureReleasesLeaseImmediately)
{
    StubProvider provider;
    ConnectionManager manager(provider, "");
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    ProbeBudget budget(4);
    RecordingSubmitter submitter;
    submitter.failNext.store(true);
    ProbeTarget target(manager, submitter, MakeProbeConfig(), &budget);
    target.RunOnce(std::chrono::steady_clock::now());

    EXPECT_EQ(submitter.submits.load(), 0);      // 提交失败
    EXPECT_EQ(budget.Active(), std::uint64_t{0}); // 立即 Release
    auto stats = target.GetStats();
    EXPECT_EQ(stats.submitted, std::uint64_t{0});
}

// ─── F3: 游标分批与预算耗尽即停 ───

TEST(ProbeTargetTest, CursorBatchesChannelsAndBudgetStopsRound)
{
    StubProvider provider;
    ConnectionManager manager(provider, "");
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    std::vector<std::shared_ptr<ConnectionChannel>> channels;
    for (int i = 0; i < 4; ++i) {
        auto channel = manager.SelectConnection();
        ASSERT_NE(channel, nullptr);
        channel->ReleaseInflight();
        channels.push_back(channel);
    }

    ProbeBudget budget(2);  // 预算 2：本轮在途上限（分批上限 8 更宽松）
    RecordingSubmitter submitter;
    ProbeTarget target(manager, submitter, MakeProbeConfig(8), &budget);
    target.RunOnce(std::chrono::steady_clock::now());

    EXPECT_EQ(submitter.submits.load(), 2);  // 预算耗尽即停
    auto stats = target.GetStats();
    EXPECT_EQ(stats.submitted, std::uint64_t{2});
    EXPECT_EQ(stats.budgetExhausted, std::uint64_t{1});
    EXPECT_EQ(budget.Active(), std::uint64_t{2});  // 在途不越界

    submitter.ReleaseAll();
    target.RunOnce(std::chrono::steady_clock::now());
    // 游标推进：第二批覆盖剩余通道（含回绕）。
    EXPECT_EQ(submitter.submits.load(), 4);
    submitter.ReleaseAll();
}

TEST(ProbeTargetTest, CursorWrapsAfterFullSweep)
{
    StubProvider provider;
    ConnectionManager manager(provider, "");
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 3).ok());

    std::vector<std::shared_ptr<ConnectionChannel>> channels;
    for (int i = 0; i < 3; ++i) {
        auto channel = manager.SelectConnection();
        ASSERT_NE(channel, nullptr);
        channel->ReleaseInflight();
        channels.push_back(channel);
    }

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    // 每轮 1 个：游标分批逐通道推进并回绕。
    ProbeTarget target(manager, submitter, MakeProbeConfig(1), &budget);
    const auto now = std::chrono::steady_clock::now();
    target.RunOnce(now);
    target.RunOnce(now);
    target.RunOnce(now);
    EXPECT_EQ(submitter.submits.load(), 3);
    submitter.ReleaseAll();

    target.RunOnce(now);
    // 第 4 轮回绕到首个通道。
    EXPECT_EQ(submitter.submits.load(), 4);
    {
        std::lock_guard<std::mutex> lock(submitter.leasesMu);
        ASSERT_FALSE(submitter.leases.empty());
        EXPECT_EQ(submitter.leases.front()->Channel()->GetChannelId(),
                  channels[0]->GetChannelId());
    }
    submitter.ReleaseAll();
}

// ─── F2: 探针与恢复互斥（窗口内整组跳过，含 FAILED，不复活） ───

TEST(ProbeTargetTest, RecoveryWindowSkipsWholeGroupIncludingFailed)
{
    StubProvider provider;
    ConnectionManager manager(provider, "");
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    std::vector<std::shared_ptr<ConnectionChannel>> channels;
    for (int i = 0; i < 2; ++i) {
        auto channel = manager.SelectConnection();
        ASSERT_NE(channel, nullptr);
        channel->ReleaseInflight();
        channel->MarkForDrain();
        channel->MarkFailed();
        channels.push_back(channel);
    }
    {
        auto group = channels[0]->LockGroup();
        ASSERT_NE(group, nullptr);
        group->SetRecoveryWindow(true);  // 恢复窗口开启
    }

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    ProbeTarget target(manager, submitter, MakeProbeConfig(), &budget);
    target.RunOnce(std::chrono::steady_clock::now());
    EXPECT_EQ(submitter.submits.load(), 0);  // 整组跳过（含 FAILED）

    {
        auto group = channels[0]->LockGroup();
        group->SetRecoveryWindow(false);  // 窗口关闭
    }
    target.RunOnce(std::chrono::steady_clock::now());
    EXPECT_EQ(submitter.submits.load(), 2);
    submitter.ReleaseAll();
}

// ─── F4: 三值结论处置 ───

TEST(ProbeTargetTest, FlushKeepsFailedAndDrainsActive)
{
    StubProvider provider;
    ConnectionManager manager(provider, "", 5000 /*timeout*/, 1 /*X=1*/);
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto failed = manager.SelectConnection();
    auto active = manager.SelectConnection();
    ASSERT_NE(failed, nullptr);
    ASSERT_NE(active, nullptr);
    failed->ReleaseInflight();
    active->ReleaseInflight();
    failed->MarkForDrain();
    failed->MarkFailed();

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    ProbeTarget target(manager, submitter, MakeProbeConfig(), &budget);
    const auto now = std::chrono::steady_clock::now();

    target.HandleProbeResult(failed, ProbeResult::FLUSH, now);
    EXPECT_EQ(failed->GetState(), ChannelState::FAILED);  // 维持 FAILED
    EXPECT_EQ(failed->GetLastProbeResult(), ProbeResult::FLUSH);

    target.HandleProbeResult(active, ProbeResult::FLUSH, now);
    EXPECT_EQ(active->GetState(), ChannelState::FAILED);  // X=1：ReportFailure 排空

    auto stats = target.GetStats();
    EXPECT_EQ(stats.flush, std::uint64_t{2});
}

TEST(ProbeTargetTest, SuccessRevivesFailedWithFreshCounters)
{
    StubProvider provider;
    ConnectionManager manager(provider, "");
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto failed = manager.SelectConnection();
    auto active = manager.SelectConnection();
    ASSERT_NE(failed, nullptr);
    ASSERT_NE(active, nullptr);
    failed->ReleaseInflight();
    active->ReleaseInflight();
    failed->MarkForDrain();
    failed->MarkFailed();
    (void)failed->FetchAddErrorCount(5);  // 制造旧计数
    (void)active->FetchAddErrorCount(3);

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    ProbeTarget target(manager, submitter, MakeProbeConfig(), &budget);
    const auto now = std::chrono::steady_clock::now();

    target.HandleProbeResult(failed, ProbeResult::SUCCESS, now);
    EXPECT_EQ(failed->GetState(), ChannelState::ACTIVE);  // FAILED -> ACTIVE 复活
    EXPECT_EQ(failed->GetErrorCount(), std::uint32_t{0});  // 计数器从零

    // ACTIVE：只完成探针，不升载、不清 errorCount。
    target.HandleProbeResult(active, ProbeResult::SUCCESS, now);
    EXPECT_EQ(active->GetState(), ChannelState::ACTIVE);
    EXPECT_EQ(active->GetErrorCount(), std::uint32_t{3});  // 保留业务失败记账

    auto stats = target.GetStats();
    EXPECT_EQ(stats.success, std::uint64_t{2});
    EXPECT_EQ(stats.revives, std::uint64_t{1});
}

TEST(ProbeTargetTest, SuccessDoesNotReviveInsideRecoveryWindow)
{
    StubProvider provider;
    ConnectionManager manager(provider, "");
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    auto channel = manager.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();
    channel->MarkForDrain();
    channel->MarkFailed();
    {
        auto group = channel->LockGroup();
        group->SetRecoveryWindow(true);
    }

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    ProbeTarget target(manager, submitter, MakeProbeConfig(), &budget);
    target.HandleProbeResult(channel, ProbeResult::SUCCESS,
                              std::chrono::steady_clock::now());
    // 恢复窗口内探针结论不复活（恢复触发单向优先不可撤销）。
    EXPECT_EQ(channel->GetState(), ChannelState::FAILED);
    auto stats = target.GetStats();
    EXPECT_EQ(stats.revives, std::uint64_t{0});
}

TEST(ProbeTargetTest, TimeoutKeepsFailedAndDrainsActiveAfterRetries)
{
    StubProvider provider;
    ConnectionManager manager(provider, "", 5000, 1 /*X=1*/);
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto failed = manager.SelectConnection();
    auto active = manager.SelectConnection();
    ASSERT_NE(failed, nullptr);
    ASSERT_NE(active, nullptr);
    failed->ReleaseInflight();
    active->ReleaseInflight();
    failed->MarkForDrain();
    failed->MarkFailed();

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    // m=0（当前默认）：ACTIVE 通道首次 TIMEOUT 即耗尽 -> ReportFailure。
    ProbeTarget target(manager, submitter, MakeProbeConfig(16, 0, 0), &budget);
    const auto now = std::chrono::steady_clock::now();

    target.HandleProbeResult(failed, ProbeResult::TIMEOUT, now);
    EXPECT_EQ(failed->GetState(), ChannelState::FAILED);  // 维持 FAILED
    EXPECT_EQ(failed->GetLastProbeResult(), ProbeResult::TIMEOUT);

    target.HandleProbeResult(active, ProbeResult::TIMEOUT, now);
    EXPECT_EQ(active->GetState(), ChannelState::FAILED);  // 耗尽排空
    auto stats = target.GetStats();
    EXPECT_EQ(stats.timeout, std::uint64_t{2});
}

TEST(ProbeTargetTest, ActiveTimeoutRetriesWithinMBeforeDraining)
{
    StubProvider provider;
    ConnectionManager manager(provider, "", 5000, 1 /*X=1*/);
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    auto channel = manager.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    // m=1：首个 TIMEOUT 走一轮退避重试（backoff=50ms），第二个才排空。
    ProbeTarget target(manager, submitter, MakeProbeConfig(16, 1, 50), &budget);
    const auto now = std::chrono::steady_clock::now();

    target.HandleProbeResult(channel, ProbeResult::TIMEOUT, now);
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);  // m 轮内未排空
    EXPECT_EQ(channel->GetLastProbeResult(), ProbeResult::NONE);  // ACTIVE 不更新 lastProbe

    // 退避窗口未到：RunOnce 跳过该通道。
    target.RunOnce(now);
    EXPECT_EQ(submitter.submits.load(), 0);
    // 退避到期后重试同一通道。
    target.RunOnce(now + std::chrono::milliseconds(60));
    EXPECT_EQ(submitter.submits.load(), 1);
    submitter.ReleaseAll();

    // 第 2 次 TIMEOUT：retries=2 > m=1 -> 耗尽排空。
    target.HandleProbeResult(channel, ProbeResult::TIMEOUT,
                              now + std::chrono::milliseconds(60));
    EXPECT_EQ(channel->GetState(), ChannelState::FAILED);
}

TEST(ProbeTargetTest, InconclusiveKeepsStateAndLastResult)
{
    StubProvider provider;
    ConnectionManager manager(provider, "", 5000, 1);
    ASSERT_TRUE(manager.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto failed = manager.SelectConnection();
    auto active = manager.SelectConnection();
    ASSERT_NE(failed, nullptr);
    ASSERT_NE(active, nullptr);
    failed->ReleaseInflight();
    active->ReleaseInflight();
    failed->MarkForDrain();
    failed->MarkFailed();
    failed->SetLastProbeResult(ProbeResult::FLUSH);  // 制造既有结论

    ProbeBudget budget(16);
    RecordingSubmitter submitter;
    ProbeTarget target(manager, submitter, MakeProbeConfig(), &budget);
    const auto now = std::chrono::steady_clock::now();

    target.HandleProbeResult(failed, ProbeResult::INCONCLUSIVE, now);
    target.HandleProbeResult(active, ProbeResult::INCONCLUSIVE, now);

    EXPECT_EQ(failed->GetState(), ChannelState::FAILED);          // 状态保持
    EXPECT_EQ(failed->GetLastProbeResult(), ProbeResult::FLUSH);  // 结论不更新
    EXPECT_EQ(active->GetState(), ChannelState::ACTIVE);
    auto stats = target.GetStats();
    EXPECT_EQ(stats.inconclusive, std::uint64_t{2});
}

}  // namespace kv::test
