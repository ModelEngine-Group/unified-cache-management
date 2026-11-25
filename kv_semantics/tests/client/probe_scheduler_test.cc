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

// Shard 4 DT（client 侧）：ClientProbeScheduler 共享调度。
//
// 测试因子分解：
//   F1 Transport 数量 ∈ {1, 3, 8}（每 Client 仅 1 个共享探针线程，与数量无关）
//   F2 游标轮转     ∈ {rounds < targets, rounds == targets}
//   F3 共享预算     ∈ {未达上限, 达上限即停}（在途派生上限不越界）
//   F4 生命周期     ∈ {Start 幂等, Stop 后轮次不再推进}
// 断言面：TargetCount / rounds / 提交数 / 全局在途预算上限。

#include <atomic>
#include <functional>
#include <chrono>
#include <memory>
#include <set>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "conn/connection_internal.h"
#include "conn/connection_manager.h"
#include "probe_scheduler.h"
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
            handles.push_back(
                reinterpret_cast<ConnectionHandle>(static_cast<uintptr_t>(++sequence_)));
        }
        return Status::OK();
    }
    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>&) override
    {
        return {};
    }
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

// 共享记账提交桩：记录来自哪个 Transport 目标的提交。
class SharedRecordingSubmitter : public IProbeSubmitter {
public:
    std::atomic<int> submits{0};
    std::atomic<int> globalInflight{0};
    std::atomic<int> maxInflightSeen{0};

    Status SubmitProbe(const std::shared_ptr<ConnectionChannel>&,
                       std::shared_ptr<ProbeTaskLease> lease) override
    {
        submits.fetch_add(1);
        const auto current = globalInflight.fetch_add(1) + 1;
        int seen = maxInflightSeen.load();
        while (current > seen &&
               !maxInflightSeen.compare_exchange_weak(seen, current)) {
        }
        std::lock_guard<std::mutex> lock(mu);
        leases.push_back(std::move(lease));
        return Status::OK();
    }

    void ReleaseAll()
    {
        std::lock_guard<std::mutex> lock(mu);
        for (auto& lease : leases) {
            lease->Release();
            globalInflight.fetch_sub(1);
        }
        leases.clear();
    }

private:
    std::mutex mu;
    std::vector<std::shared_ptr<ProbeTaskLease>> leases;
};

NodeEndpoint MakeEndpoint(const std::string& ip)
{
    NodeEndpoint ep;
    ep.ip = ip;
    ep.port = 16666;
    return ep;
}

ProbeConfig SchedulerProbeConfig(std::uint32_t maxConcurrentTransports,
                                  std::uint32_t maxChannelsPerTransport,
                                  std::uint32_t probePeriodMs = 1000)
{
    ProbeConfig config;
    config.maxConcurrentTransports = maxConcurrentTransports;
    config.maxChannelsPerTransport = maxChannelsPerTransport;
    config.probePeriodMs = probePeriodMs;
    return config;
}

// 一个“Transport”目标 = ConnectionManager + submitter + ProbeTarget。
struct FakeTransportTarget {
    std::unique_ptr<StubProvider> provider;
    std::unique_ptr<ConnectionManager> manager;
    std::unique_ptr<SharedRecordingSubmitter> submitter;
    std::unique_ptr<ProbeTarget> target;
};

std::unique_ptr<FakeTransportTarget> MakeTarget(const std::string& ip,
                                                const ProbeConfig& targetConfig)
{
    auto unit = std::make_unique<FakeTransportTarget>();
    unit->provider = std::make_unique<StubProvider>();
    unit->manager = std::make_unique<ConnectionManager>(*unit->provider, "");
    if (!unit->manager->AddGroup(MakeEndpoint(ip), 2).ok()) { return nullptr; }
    for (int i = 0; i < 2; ++i) {
        auto channel = unit->manager->SelectConnection();
        if (channel == nullptr) { return nullptr; }
        channel->ReleaseInflight();
    }
    unit->submitter = std::make_unique<SharedRecordingSubmitter>();
    unit->target = std::make_unique<ProbeTarget>(*unit->manager, *unit->submitter, targetConfig);
    return unit;
}

bool WaitFor(std::function<bool()> predicate, int timeoutMs = 3000)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

}  // namespace

TEST(ClientProbeSchedulerTest, RegisterTargetsBindsSharedBudget)
{
    const auto targetConfig = SchedulerProbeConfig(1, 1024);
    ClientProbeScheduler scheduler(SchedulerProbeConfig(8, 8));  // 派生在途上限 64

    std::vector<std::unique_ptr<FakeTransportTarget>> units;
    for (int i = 0; i < 3; ++i) {
        auto unit = MakeTarget("10.0.0." + std::to_string(i + 1), targetConfig);
        ASSERT_NE(unit, nullptr);
        scheduler.RegisterTarget(unit->target.get());
        units.push_back(std::move(unit));
    }

    EXPECT_EQ(scheduler.TargetCount(), std::size_t{3});
    auto stats = scheduler.GetStats();
    EXPECT_EQ(stats.targets, std::size_t{3});
    EXPECT_EQ(stats.budgetLimit, std::uint64_t{64});  // 8 x 8 派生上限

    // 游标轮转：3 轮覆盖全部目标（预算 64 未满，不约束）。
    scheduler.RunOnce();
    scheduler.RunOnce();
    scheduler.RunOnce();
    int total = 0;
    for (auto& unit : units) { total += unit->submitter->submits.load(); }
    EXPECT_EQ(total, 6);
    EXPECT_EQ(scheduler.GetStats().rounds, std::uint64_t{3});

    for (auto& unit : units) { unit->submitter->ReleaseAll(); }
}

TEST(ClientProbeSchedulerTest, SharedBudgetCapsInflightAcrossTransports)
{
    // 派生上限 = 1 x 2 = 2；每目标每轮可提交 2 个 -> 第 2 个目标耗尽预算。
    const auto targetConfig = SchedulerProbeConfig(1, 1024);
    ClientProbeScheduler scheduler(SchedulerProbeConfig(1, 2));

    std::vector<std::unique_ptr<FakeTransportTarget>> units;
    for (int i = 0; i < 3; ++i) {
        auto unit = MakeTarget("10.0.0." + std::to_string(i + 1), targetConfig);
        ASSERT_NE(unit, nullptr);
        scheduler.RegisterTarget(unit->target.get());
        units.push_back(std::move(unit));
    }

    for (int round = 0; round < 3; ++round) { scheduler.RunOnce(); }

    int submitted = 0;
    for (auto& unit : units) { submitted += unit->submitter->submits.load(); }
    // 预算在途上限 2：三轮中只有预算允许的 2 个提交，在途不越界。
    EXPECT_EQ(submitted, 2);
    EXPECT_EQ(scheduler.GetStats().budgetActive, std::uint64_t{2});

    for (auto& unit : units) { unit->submitter->ReleaseAll(); }
    EXPECT_EQ(scheduler.GetStats().budgetActive, std::uint64_t{0});
}

TEST(ClientProbeSchedulerTest, StartRunsOnceImmediatelyThenPeriodicallyStopsCleanly)
{
    const auto targetConfig = SchedulerProbeConfig(1, 1024);
    ClientProbeScheduler scheduler(SchedulerProbeConfig(1, 1024, /*probePeriodMs=*/5));
    auto unit = MakeTarget("10.0.0.1", targetConfig);
    ASSERT_NE(unit, nullptr);
    scheduler.RegisterTarget(unit->target.get());

    scheduler.Start();   // 启动即 RunOnce
    scheduler.Start();   // 幂等：不重复起线程
    // 周期推进（含 jitter 上限 5/4ms）。
    EXPECT_TRUE(WaitFor([&] { return scheduler.GetStats().rounds >= 3; }, 4000));
    EXPECT_GE(unit->submitter->submits.load(), 1);

    const auto roundsAtStop = scheduler.GetStats().rounds;
    scheduler.Stop();
    const auto roundsAfterStop = scheduler.GetStats().rounds;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // Stop 后轮次不再推进（每 Client <= 1 线程，Stop 即收敛）。
    EXPECT_EQ(scheduler.GetStats().rounds, roundsAfterStop);
    EXPECT_GE(roundsAfterStop, roundsAtStop);
    unit->submitter->ReleaseAll();
}

TEST(ClientProbeSchedulerTest, EightTransportsShareOneSchedulerThread)
{
    // 大量 Transport 下每 Client 仅 1 个共享探针线程：
    // 8 目标经游标轮转全部被同一调度器驱动，无额外线程。
    const auto targetConfig = SchedulerProbeConfig(1, 1);  // 每轮每目标 1 个通道
    ClientProbeScheduler scheduler(SchedulerProbeConfig(8, 8, /*probePeriodMs=*/2));
    std::vector<std::unique_ptr<FakeTransportTarget>> units;
    for (int i = 0; i < 8; ++i) {
        auto unit = MakeTarget("10.0.1." + std::to_string(i + 1), targetConfig);
        ASSERT_NE(unit, nullptr);
        scheduler.RegisterTarget(unit->target.get());
        units.push_back(std::move(unit));
    }
    ASSERT_EQ(scheduler.TargetCount(), std::size_t{8});

    for (int round = 0; round < 16; ++round) { scheduler.RunOnce(); }

    // 游标轮转两圈：每个目标被驱动 2 轮、每轮 1 个通道。
    int totalSubmits = 0;
    for (auto& unit : units) { totalSubmits += unit->submitter->submits.load(); }
    EXPECT_EQ(totalSubmits, 16);
    EXPECT_EQ(scheduler.GetStats().budgetActive, std::uint64_t{16});

    for (auto& unit : units) { unit->submitter->ReleaseAll(); }
    EXPECT_EQ(scheduler.GetStats().budgetActive, std::uint64_t{0});
}

}  // namespace kv::test
