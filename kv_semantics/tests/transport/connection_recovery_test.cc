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

// Shard 2 DT：连接状态机、Endpoint 级整组恢复、节点三态、动态 Endpoint。
//
// 测试因子分解：
//   F1 Channel 状态 ∈ {ACTIVE, DRAINING, FAILED} × 迁移合法性（含非法边）
//   F2 触发条件     ∈ {部分 Channel 失败, 全部非 ACTIVE}
//   F3 恢复轮次     ∈ {首轮成功, 连续失败退避后成功, 组 inflight 未排空}
//   F4 三态         ∈ {GREEN, YELLOW, RED} × {updating=true/false}
//   F5 UpdateGroups ∈ {fail-fast, 删除无匹配, 删除有匹配, 新增, 全等skip,
//                       属性替换, 部分失败, 重复推送}
//   F6 选路         ∈ {排除集跨Group优先, 同Group兜底, 排空无候选}
// 断言面：状态迁移、句柄删除恰好一次、计数器清零、窗口标志、三态聚合、
//         fail-fast、幂等收敛。

#include "conn/connection_manager.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <set>
#include <algorithm>
#include <thread>
#include <future>
#include <vector>
#include <gtest/gtest.h>
#include "conn/connection_internal.h"
#include "kv_transport.h"
#include "trans_provider.h"

namespace kv::test {
namespace {

// 遵守 DeleteConnections 排空契约的可编程桩：仅记账、不产生写回。
class ScriptedTransProvider : public TransProvider {
public:
    std::atomic<int> createCount{0};
    std::atomic<int> deleteCount{0};
    std::atomic<int> createHandleCount{0};  // 成功建链产生的句柄数（对账用）
    std::atomic<int> createFailuresRemaining{0};
    ServerKvCapabilities serverCapabilities;
    // 按 remoteIp 决定建链结果的脚本钩子（缺省成功）。
    std::function<bool(const std::string&)> failByIp;
    // 建链时回调（用于在 UpdateGroups 窗口内观察快照）。
    std::function<void()> onCreate;
    // 建链调用顺序日志（FIFO / 公平性断言用），线程安全。
    std::mutex ipsMu;
    std::vector<std::string> createdIps;

    std::vector<std::string> GetCreatedIps()
    {
        std::lock_guard<std::mutex> lock(ipsMu);
        return createdIps;
    }

    Status GetServerCapabilities(ConnectionHandle, ServerKvCapabilities& capabilities) override
    {
        capabilities = serverCapabilities;
        return Status::OK();
    }

    Status CreateConnection(const std::string&, const std::string& remoteIp, uint32_t,
                            uint32_t qpNum, uint32_t,
                            std::vector<ConnectionHandle>& handles) override
    {
        const auto sequence = createCount.fetch_add(1) + 1;
        {
            std::lock_guard<std::mutex> lock(ipsMu);
            createdIps.push_back(remoteIp);
        }
        if (onCreate) { onCreate(); }
        if (failByIp && failByIp(remoteIp)) {
            return Status::Error(StatusCode::CONNECTION_ERROR, "scripted create failure");
        }
        if (createFailuresRemaining.load() > 0) {
            createFailuresRemaining.fetch_sub(1);
            return Status::Error(StatusCode::CONNECTION_ERROR, "scripted create failure");
        }
        handles.clear();
        for (std::uint32_t i = 0; i < qpNum; ++i) {
            handles.push_back(reinterpret_cast<ConnectionHandle>(
                static_cast<uintptr_t>(sequence * 100 + i + 1)));
        }
        createHandleCount.fetch_add(static_cast<int>(handles.size()));
        return Status::OK();
    }

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override
    {
        deleteCount.fetch_add(static_cast<int>(handles.size()));
        return std::vector<Status>(handles.size(), Status::OK());
    }

    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>&, uint32_t,
                             uint32_t) override
    {
        return {};
    }
    Status RegisterMemory(const std::vector<RegisterMemoryDesc>& descs,
                          std::vector<MRHandle>& handles) override
    {
        handles.clear();
        handles.reserve(descs.size());
        for (std::size_t index = 0; index < descs.size(); ++index) {
            handles.push_back(reinterpret_cast<MRHandle>(static_cast<std::uintptr_t>(index) +
                                                          static_cast<std::uintptr_t>(1)));
        }
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
};

TransportConfig FastRecoveryConfig()
{
    TransportConfig config;
    config.recovery.recoverCheckIntervalMs = 5;
    config.recovery.recoverInitialBackoffMs = 5;
    config.recovery.recoverMaxBackoffMs = 20;
    config.recovery.connectRetryPerRound = 1;  // 每轮 1+1=2 次尝试
    config.timeoutMs = 20;  // 任务超时（legacy 字段，恢复排空上界的一部分）
    config.connection.maxErrorCount = 1;
    return config;
}

NodeEndpoint MakeEndpoint(const std::string& ip, std::uint16_t port = 16666)
{
    NodeEndpoint ep;
    ep.ip = ip;
    ep.port = port;
    return ep;
}

// 轮询等待条件成立（上限 timeoutMs），避免固定 sleep 导致的抖动。
bool WaitFor(std::function<bool()> predicate, int timeoutMs = 2000)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}

}  // namespace

// ─── F1: Channel 状态机合法/非法迁移 ───

TEST(ConnectionChannelStateTest, InitStateIsActiveAndDrainFlowIsOrdered)
{
    ScriptedTransProvider provider;
    ConnectionGroup group(0, MakeEndpoint("10.0.0.1"));
    ConnectionChannel ch(0, &group, reinterpret_cast<ConnectionHandle>(0x1), &provider);

    EXPECT_EQ(ch.GetState(), ChannelState::ACTIVE);
    // 不存在 ACTIVE -> FAILED 直接边。
    EXPECT_FALSE(ch.MarkFailed());
    EXPECT_EQ(ch.GetState(), ChannelState::ACTIVE);
    // ACTIVE -> DRAINING。
    EXPECT_TRUE(ch.MarkForDrain());
    EXPECT_FALSE(ch.MarkForDrain());
}

TEST(ConnectionChannelStateTest, DrainingReachesFailedWhenInflightDrains)
{
    ScriptedTransProvider provider;
    ConnectionGroup group(0, MakeEndpoint("10.0.0.1"));
    ConnectionChannel ch(0, &group, reinterpret_cast<ConnectionHandle>(0x1), &provider);

    ch.IncrementInflight();
    ch.IncrementInflight();
    EXPECT_TRUE(ch.MarkForDrain());
    EXPECT_EQ(ch.GetState(), ChannelState::DRAINING);

    ch.ReleaseInflight();
    EXPECT_EQ(ch.GetState(), ChannelState::DRAINING);  // 仍在途
    ch.ReleaseInflight();
    EXPECT_EQ(ch.GetState(), ChannelState::FAILED);  // 排空即 FAILED
}

// ─── F2/F3: Endpoint 级整组恢复 ───

TEST(ConnectionRecoveryTest, RecoveryWindowOpensAndRebuildsWholeGroup)
{
    ScriptedTransProvider provider;
    auto config = FastRecoveryConfig();
    config.recovery.recoverInitialBackoffMs = 200;  // 首轮失败后窗口可观测
    ConnectionManager mgr(provider, "", config);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto ch0 = mgr.SelectConnection();
    auto ch1 = mgr.SelectConnection();
    ASSERT_NE(ch0, nullptr);
    ASSERT_NE(ch1, nullptr);
    ch0->ReleaseInflight();
    ch1->ReleaseInflight();
    const auto gid = ch0->GetGroupId();

    NodeHealthSnapshot snapshot;
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::GREEN);

    // 脚本：恢复重建首轮（1+K=2 次尝试）失败，窗口保持开启。
    provider.createFailuresRemaining.store(2);
    mgr.StartRecoverLoop();

    // 整组触发：两条 Channel 全部失败（X=1，各自单次失败）。
    mgr.ReportFailure(ch0);
    mgr.ReportFailure(ch1);
    EXPECT_EQ(ch0->GetState(), ChannelState::FAILED);
    EXPECT_EQ(ch1->GetState(), ChannelState::FAILED);

    // 窗口期：无候选 fail-fast，恢复窗口状态存在。
    EXPECT_TRUE(WaitFor([&] {
        return mgr.SelectConnection() == nullptr && mgr.HasRecoverState(gid);
    }));

    // 恢复完成：一次性替换全部新 Channel 对象。
    std::shared_ptr<ConnectionChannel> fresh0;
    std::shared_ptr<ConnectionChannel> fresh1;
    EXPECT_TRUE(WaitFor([&] {
        fresh0 = mgr.SelectConnection();
        fresh1 = mgr.SelectConnection();
        return fresh0 != nullptr && fresh1 != nullptr;
    }, 4000));
    mgr.StopRecoverLoop();
    if (fresh0) { fresh0->ReleaseInflight(); }
    if (fresh1) { fresh1->ReleaseInflight(); }

    ASSERT_NE(fresh0, nullptr);
    ASSERT_NE(fresh1, nullptr);
    // 新对象：不在旧对象集合内，计数器从零。
    EXPECT_NE(fresh0.get(), ch0.get());
    EXPECT_NE(fresh0.get(), ch1.get());
    EXPECT_NE(fresh1.get(), ch0.get());
    EXPECT_NE(fresh1.get(), ch1.get());
    EXPECT_EQ(fresh0->GetErrorCount(), 0u);
    EXPECT_EQ(fresh1->GetErrorCount(), 0u);
    EXPECT_EQ(fresh0->GetGroupId(), gid);

    // 旧句柄恰好删除一次（detached 后析构不再删）。
    EXPECT_EQ(provider.deleteCount.load(), 2);
    // 初始建链 1 次 + 失败轮 2 次 + 成功轮 1 次。
    EXPECT_EQ(provider.createCount.load(), 4);
    // 窗口关闭。
    EXPECT_FALSE(mgr.HasRecoverState(gid));

    mgr.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::GREEN);
}

TEST(ConnectionRecoveryTest, RecoveryRetriesWithBackoffUntilCreateSucceeds)
{
    ScriptedTransProvider provider;
    auto config = FastRecoveryConfig();
    ConnectionManager mgr(provider, "", config);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    auto ch = mgr.SelectConnection();
    ASSERT_NE(ch, nullptr);
    ch->ReleaseInflight();

    // 脚本：恢复重建连续失败 4 次后成功（覆盖指数退避重试轮次）。
    provider.createFailuresRemaining.store(4);
    mgr.StartRecoverLoop();
    mgr.ReportFailure(ch);

    std::shared_ptr<ConnectionChannel> fresh;
    EXPECT_TRUE(WaitFor([&] {
        fresh = mgr.SelectConnection();
        return fresh != nullptr;
    }, 4000));
    mgr.StopRecoverLoop();
    if (fresh) { fresh->ReleaseInflight(); }

    ASSERT_NE(fresh, nullptr);
    // 初始 1 次 + 4 次失败 + 1 次成功（K=1 时首轮 2 次尝试）。
    EXPECT_GE(provider.createCount.load(), 6);
    EXPECT_EQ(provider.deleteCount.load(), 1);
}

TEST(ConnectionRecoveryTest, RecoveryWaitsBoundedForInflightThenForcesTeardown)
{
    ScriptedTransProvider provider;
    auto config = FastRecoveryConfig();
    ConnectionManager mgr(provider, "", config);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    auto ch = mgr.SelectConnection();
    ASSERT_NE(ch, nullptr);
    // 保持一个在途请求不释放：排空有界等待。
    mgr.StartRecoverLoop();
    mgr.ReportFailure(ch);
    // 排空标志：DRAINING（inflight=1）。
    EXPECT_EQ(ch->GetState(), ChannelState::DRAINING);

    std::shared_ptr<ConnectionChannel> fresh;
    // 有界等待上限 = max(taskTimeoutMs=20, recoverInitialBackoffMs=5) 后强制断链重建，
    // 不死等。
    EXPECT_TRUE(WaitFor([&] {
        fresh = mgr.SelectConnection();
        return fresh != nullptr;
    }, 4000));
    mgr.StopRecoverLoop();

    ASSERT_NE(fresh, nullptr);
    EXPECT_NE(fresh.get(), ch.get());
    fresh->ReleaseInflight();
    // 旧句柄被强制断链（有界排空后由摘链方独占删除）。
    EXPECT_EQ(provider.deleteCount.load(), 1);
}

TEST(ConnectionRecoveryTest, SingleGroupFailureKeepsNodeYellowWithOtherActiveGroup)
{
    ScriptedTransProvider provider;
    auto config = FastRecoveryConfig();
    ConnectionManager mgr(provider, "", config);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.2"), 1).ok());

    auto failing = mgr.SelectConnection();
    ASSERT_NE(failing, nullptr);
    failing->ReleaseInflight();
    // 单 Group 恢复窗口不独立触发节点 RED：仍有 ACTIVE 链路 → YELLOW。
    provider.createFailuresRemaining.store(1000);
    mgr.StartRecoverLoop();
    mgr.ReportFailure(failing);

    EXPECT_TRUE(WaitFor([&] { return mgr.HasRecoverState(failing->GetGroupId()); }));
    NodeHealthSnapshot snapshot;
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::YELLOW);

    provider.createFailuresRemaining.store(0);
    EXPECT_TRUE(WaitFor([&] { return !mgr.HasRecoverState(failing->GetGroupId()); }));
    mgr.StopRecoverLoop();
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::GREEN);
}

// ─── F4: 三态聚合边界 ───

TEST(ConnectionHealthTest, SnapshotAggregatesGreenYellowRed)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    NodeHealthSnapshot snapshot;

    // 无 Group：RED。
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::RED);
    EXPECT_FALSE(snapshot.updating);

    TransportConfig config;
    ConnectionManager mgr2(provider, "", config);
    ASSERT_TRUE(mgr2.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());
    mgr2.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::GREEN);

    // 一条非 ACTIVE：YELLOW。
    auto ch0 = mgr2.SelectConnection();
    ASSERT_NE(ch0, nullptr);
    ch0->ReleaseInflight();
    ch0->MarkForDrain();
    mgr2.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::YELLOW);

    // 全部非 ACTIVE：RED（与无候选 fail-fast 一致）。
    auto ch1 = mgr2.SelectConnection();
    ASSERT_NE(ch1, nullptr);
    ch1->ReleaseInflight();
    ch1->MarkForDrain();
    ch1->MarkFailed();
    ch0->MarkFailed();
    mgr2.GetHealthSnapshot(snapshot);
    EXPECT_EQ(snapshot.health, NodeHealth::RED);
}

TEST(ConnectionHealthTest, StubTransportDefaultsToUnsupported)
{
    class StubTransport : public AsuTransport {
    public:
        Status Init(const TransportConfig&, std::shared_ptr<TransProvider>) override
        {
            return Status::OK();
        }
        Status Init(const std::string&, std::shared_ptr<TransProvider>) override
        {
            return Status::OK();
        }
        Status Shutdown() override { return Status::OK(); }
        Status CheckHealth() override { return Status::OK(); }
        Status Submit(const std::shared_ptr<TransportTask>&) override { return Status::OK(); }
        Status Cancel(TaskId) override { return Status::OK(); }
    };

    StubTransport stub;
    NodeHealthSnapshot snapshot;
    UpdateResult result;
    EXPECT_EQ(stub.GetConnectionHealth(snapshot).code, StatusCode::UNSUPPORTED);
    EXPECT_EQ(stub.UpdateEndpoints({}, {}, result).code, StatusCode::UNSUPPORTED);

    // 新增异步契约的默认实现：ready future + UNSUPPORTED，Stub 免实现。
    auto asyncFuture = stub.UpdateEndpointsAsync({}, {});
    ASSERT_EQ(asyncFuture.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    const auto asyncResult = asyncFuture.get();
    EXPECT_EQ(asyncResult.status.code, StatusCode::UNSUPPORTED);
}

// ─── F5: 动态 Endpoint（UpdateGroups） ───

TEST(ConnectionUpdateGroupsTest, EmptyPatchIsAtomicallyRejected)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    UpdateResult result;
    const auto status = mgr.UpdateGroups({}, {}, result);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    // 原子拒绝：不修改任何现有 Group。
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});
    EXPECT_EQ(provider.deleteCount.load(), 0);
    EXPECT_EQ(provider.createCount.load(), 1);
}

TEST(ConnectionUpdateGroupsTest, InvalidEndpointIsAtomicallyRejected)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    auto badPort = MakeEndpoint("10.0.0.2");
    badPort.port = 0;  // 端口 0 非法
    auto badIp = MakeEndpoint("");
    badIp.ip = "";

    UpdateResult result;
    EXPECT_EQ(mgr.UpdateGroups({badPort}, {}, result).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(mgr.UpdateGroups({}, {badIp}, result).code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});
    EXPECT_EQ(provider.createCount.load(), 1);
    EXPECT_EQ(provider.deleteCount.load(), 0);
}

TEST(ConnectionUpdateGroupsTest, RemoveWithoutMatchSkipsIdempotently)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    UpdateResult result;
    const auto status = mgr.UpdateGroups({}, {MakeEndpoint("10.9.9.9")}, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.removeResults.size(), std::size_t{1});
    EXPECT_EQ(result.removeResults.front().action, "skipped");
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});
    EXPECT_EQ(provider.deleteCount.load(), 0);

    // 重推同一 remove：仍 skip。
    UpdateResult again;
    (void)mgr.UpdateGroups({}, {MakeEndpoint("10.9.9.9")}, again);
    EXPECT_EQ(again.removeResults.front().action, "skipped");
}

TEST(ConnectionUpdateGroupsTest, RemoveExistingGroupDeregistersThenDisconnects)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());
    const auto created = provider.createCount.load();

    UpdateResult result;
    const auto status = mgr.UpdateGroups({}, {MakeEndpoint("10.0.0.1")}, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.removeResults.size(), std::size_t{1});
    EXPECT_EQ(result.removeResults.front().action, "removed");

    // 摘表 + 整组断链：无候选、句柄恰好删除。
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{0});
    EXPECT_EQ(mgr.SelectConnection(), nullptr);
    EXPECT_EQ(provider.deleteCount.load(), 2);
    EXPECT_EQ(provider.createCount.load(), created);
}

TEST(ConnectionUpdateGroupsTest, AddNewGroupCreatesAndServes)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    UpdateResult result;
    const auto status = mgr.UpdateGroups({MakeEndpoint("10.0.0.7")}, {}, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{1});
    EXPECT_EQ(result.addResults.front().action, "created");

    auto channel = mgr.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();
    EXPECT_EQ(channel->GetGroupId(), std::uint32_t{0});
}

TEST(ConnectionUpdateGroupsTest, AddIdenticalEndpointSkipsAndKeepsState)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    NodeEndpoint ep = MakeEndpoint("10.0.0.1");
    ep.attrs["tc"] = "3";
    ASSERT_TRUE(mgr.AddGroup(ep, 2).ok());
    const auto created = provider.createCount.load();

    auto before = mgr.SelectConnection();
    ASSERT_NE(before, nullptr);
    before->MarkForDrain();  // 制造组内状态，skip 必须保留状态/退避/流控。

    UpdateResult result;
    const auto status = mgr.UpdateGroups({ep}, {}, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{1});
    EXPECT_EQ(result.addResults.front().action, "skipped");
    EXPECT_EQ(provider.createCount.load(), created);  // 不重建

    // 状态保留：排空的 Channel 仍为 DRAINING，未被替换复活。
    EXPECT_EQ(before->GetState(), ChannelState::DRAINING);
    auto after = mgr.SelectConnection();
    ASSERT_NE(after, nullptr);  // 组内另一条 ACTIVE 链路可选
    EXPECT_NE(after.get(), before.get());
    EXPECT_EQ(after->GetGroupId(), before->GetGroupId());
    after->ReleaseInflight();
}

TEST(ConnectionUpdateGroupsTest, AddChangedAttributesReplacesOldGroup)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    auto original = MakeEndpoint("10.0.0.1");
    original.attrs["tc"] = "3";
    ASSERT_TRUE(mgr.AddGroup(original, 2).ok());

    auto modified = MakeEndpoint("10.0.0.1");
    modified.attrs["tc"] = "5";  // 属性不同：自动替换

    UpdateResult result;
    const auto status = mgr.UpdateGroups({modified}, {}, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{1});
    EXPECT_EQ(result.addResults.front().action, "replaced");

    // 禁止 endpoint 双 Group 并存：仍然只有一个 Group。
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});
    // 旧组走删除路径（2 个句柄），新组重建链（1 次调用创建 2 句柄）。
    EXPECT_EQ(provider.deleteCount.load(), 2);
    EXPECT_EQ(provider.createCount.load(), 2);
    auto channel = mgr.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();
}

TEST(ConnectionUpdateGroupsTest, PartialAddFailureIsBestEffortWithFailedEndpoints)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    // 脚本：10.0.0.2 建链失败，其余成功。
    provider.failByIp = [](const std::string& ip) { return ip == "10.0.0.2"; };

    UpdateResult result;
    const auto status =
        mgr.UpdateGroups({MakeEndpoint("10.0.0.1"), MakeEndpoint("10.0.0.2")}, {}, result);
    // best-effort：整体成功，部分失败如 failedEndpoints。
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{2});
    EXPECT_EQ(result.addResults[0].action, "created");
    EXPECT_EQ(result.addResults[1].action, "failed");
    ASSERT_EQ(result.failedEndpoints.size(), std::size_t{1});
    EXPECT_EQ(result.failedEndpoints.front().ip, "10.0.0.2");
    // 失败组自动回滚：不残留 Group，也不回滚已成功组。
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});
    auto channel = mgr.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();
}

TEST(ConnectionUpdateGroupsTest, RepushingSamePatchConvergesIdempotently)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 1).ok());

    const std::vector<NodeEndpoint> add{MakeEndpoint("10.0.0.2")};
    const std::vector<NodeEndpoint> remove{MakeEndpoint("10.0.0.1")};

    UpdateResult first;
    ASSERT_TRUE(mgr.UpdateGroups(add, remove, first).ok());
    EXPECT_EQ(first.removeResults.front().action, "removed");
    EXPECT_EQ(first.addResults.front().action, "created");

    const auto createdAfterFirst = provider.createCount.load();
    const auto deletedAfterFirst = provider.deleteCount.load();

    // 重推同一 patch：remove 无匹配 skip，add 全等 skip。
    UpdateResult second;
    ASSERT_TRUE(mgr.UpdateGroups(add, remove, second).ok());
    EXPECT_EQ(second.removeResults.front().action, "skipped");
    EXPECT_EQ(second.addResults.front().action, "skipped");
    EXPECT_EQ(provider.createCount.load(), createdAfterFirst);
    EXPECT_EQ(provider.deleteCount.load(), deletedAfterFirst);
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});
}

TEST(ConnectionUpdateGroupsTest, UpdatingFlagVisibleDuringWindowOnly)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    NodeHealthSnapshot duringUpdate;
    provider.onCreate = [&mgr, &duringUpdate]() {
        mgr.GetHealthSnapshot(duringUpdate);
    };

    UpdateResult result;
    const auto status = mgr.UpdateGroups({MakeEndpoint("10.0.0.3")}, {}, result);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(duringUpdate.updating);  // 配置窗口内 updating=true
    EXPECT_FALSE(result.updating);       // 窗口关闭后

    NodeHealthSnapshot after;
    mgr.GetHealthSnapshot(after);
    EXPECT_FALSE(after.updating);
    EXPECT_EQ(after.health, NodeHealth::GREEN);
}

// ─── 并发 DT ───

TEST(ConnectionConcurrencyTest, ConcurrentSelectReportAndUpdateIsSafe)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    workers.emplace_back([&] {
        while (!stop.load()) {
            auto channel = mgr.SelectConnection();
            if (channel) { channel->ReleaseInflight(); }
            auto idle = mgr.SelectConnection();
            if (idle) {
                mgr.ReportFailure(idle);
                mgr.ReportSuccess(idle);
                idle->ReleaseInflight();
            }
        }
    });
    workers.emplace_back([&] {
        int toggle = 0;
        while (!stop.load()) {
            UpdateResult result;
            NodeEndpoint extra = MakeEndpoint(toggle % 2 == 0 ? "10.0.0.8" : "10.0.0.9");
            if (toggle % 2 == 0) {
                (void)mgr.UpdateGroups({extra}, {}, result);
            } else {
                (void)mgr.UpdateGroups({}, {extra}, result);
            }
            ++toggle;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop.store(true);
    for (auto& worker : workers) { worker.join(); }

    // 关停后再对账：句柄账目守恒（建链句柄数 == 删除句柄数，恰好一次）。
    (void)mgr.Shutdown();
    EXPECT_EQ(provider.createHandleCount.load(), provider.deleteCount.load());
}


// ─── 异步 UpdateGroups（RecoverLoop 合并消费） ───

TEST(ConnectionUpdateGroupsAsyncTest, AddPatchCompletesOnWorker)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    mgr.StartRecoverLoop();

    auto future = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.60.1")}, {});
    ASSERT_EQ(future.wait_for(std::chrono::seconds(4)), std::future_status::ready);
    const auto result = future.get();
    ASSERT_TRUE(result.status.ok()) << result.status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{1});
    EXPECT_EQ(result.addResults.front().action, "created");
    EXPECT_FALSE(result.updating);

    auto channel = mgr.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();

    NodeHealthSnapshot snapshot;
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_FALSE(snapshot.updating);  // 窗口已闭合
    EXPECT_EQ(snapshot.health, NodeHealth::GREEN);

    mgr.StopRecoverLoop();
}

TEST(ConnectionUpdateGroupsAsyncTest, FailFastRejectsReadyFutureWithoutWork)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    mgr.StartRecoverLoop();

    auto empty = mgr.UpdateGroupsAsync({}, {});
    ASSERT_EQ(empty.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_EQ(empty.get().status.code, StatusCode::INVALID_ARGUMENT);

    auto badPort = MakeEndpoint("10.0.61.1");
    badPort.port = 0;
    auto invalid = mgr.UpdateGroupsAsync({badPort}, {});
    ASSERT_EQ(invalid.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_EQ(invalid.get().status.code, StatusCode::INVALID_ARGUMENT);

    // 原子拒绝：没有任何建链/入队副作用。
    EXPECT_EQ(provider.createCount.load(), 0);
    NodeHealthSnapshot snapshot;
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_FALSE(snapshot.updating);

    mgr.StopRecoverLoop();
}

TEST(ConnectionUpdateGroupsAsyncTest, StrictFIFOOrderingBetweenPatches)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    mgr.StartRecoverLoop();

    auto first = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.62.1")}, {});
    auto second = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.62.2")}, {});
    ASSERT_EQ(first.wait_for(std::chrono::seconds(4)), std::future_status::ready);
    ASSERT_EQ(second.wait_for(std::chrono::seconds(4)), std::future_status::ready);
    EXPECT_TRUE(first.get().status.ok());
    EXPECT_TRUE(second.get().status.ok());

    // 建链调用顺序 = 提交顺序（严格 FIFO，无合并）。
    const auto ips = provider.GetCreatedIps();
    auto posFirst = std::find(ips.begin(), ips.end(), "10.0.62.1");
    auto posSecond = std::find(ips.begin(), ips.end(), "10.0.62.2");
    ASSERT_NE(posFirst, ips.end());
    ASSERT_NE(posSecond, ips.end());
    EXPECT_LT(posFirst - ips.begin(), posSecond - ips.begin());

    mgr.StopRecoverLoop();
}

TEST(ConnectionUpdateGroupsAsyncTest, SpreadAddsDoNotStarveGroupRecovery)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.63.1"), 1).ok());
    auto victim = mgr.SelectConnection();
    ASSERT_NE(victim, nullptr);
    victim->ReleaseInflight();

    // 该组后续建链持续失败 -> 恢复窗口不断重试。
    provider.failByIp = [](const std::string& ip) { return ip == "10.0.63.1"; };
    mgr.StartRecoverLoop();
    mgr.ReportFailure(victim);  // X=1 -> FAILED -> 恢复窗口开启
    ASSERT_TRUE(WaitFor([&] { return mgr.HasRecoverState(victim->GetGroup()->GetGroupId()); }));

    // 2 个 add 分摊在两个 tick：每 tick 之间组恢复 sweep 照常推进。
    auto future = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.63.2"), MakeEndpoint("10.0.63.3")},
                                        {});
    ASSERT_EQ(future.wait_for(std::chrono::seconds(4)), std::future_status::ready);
    const auto result = future.get();
    ASSERT_TRUE(result.status.ok()) << result.status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{2});

    // 公平性：两次 add 的建链之间至少穿插一次恢复尝试。
    const auto ips = provider.GetCreatedIps();
    auto posAdd1 = std::find(ips.begin(), ips.end(), "10.0.63.2");
    auto posAdd2 = std::find(ips.begin(), ips.end(), "10.0.63.3");
    ASSERT_NE(posAdd1, ips.end());
    ASSERT_NE(posAdd2, ips.end());
    const auto recoveryBetween =
        std::find(posAdd1 + 1, posAdd2, "10.0.63.1");
    EXPECT_NE(recoveryBetween, posAdd2)
        << "group recovery must interleave between per-tick adds";

    mgr.StopRecoverLoop();
}

TEST(ConnectionUpdateGroupsAsyncTest, StopRecoverLoopCancelsPendingPatches)
{
    ScriptedTransProvider provider;
    std::atomic<bool> slowCreate{false};
    provider.onCreate = [&slowCreate]() {
        if (slowCreate.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
    };
    ConnectionManager mgr(provider, "");
    mgr.StartRecoverLoop();

    slowCreate.store(true);
    // patch1：2 个 add，第一个 add 的建链内 worker 会睡 500ms。
    auto first = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.64.1"), MakeEndpoint("10.0.64.2")},
                                      {});
    ASSERT_TRUE(WaitFor([&] { return provider.createCount.load() >= 1; }));

    // worker 正在 tick1 内：patch2 只能排队。
    auto second = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.64.3")}, {});
    EXPECT_EQ(second.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);

    // 停止（stop 先于 join 置位）：worker 完成本 tick 后退出，
    // active(patch1 剩 1 add) + queued(patch2) 全部终结为 CANCELED。
    mgr.StopRecoverLoop();

    ASSERT_EQ(first.wait_for(std::chrono::seconds(4)), std::future_status::ready);
    ASSERT_EQ(second.wait_for(std::chrono::seconds(4)), std::future_status::ready);
    EXPECT_EQ(first.get().status.code, StatusCode::CANCELED);
    EXPECT_EQ(second.get().status.code, StatusCode::CANCELED);

    NodeHealthSnapshot snapshot;
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_FALSE(snapshot.updating);
}

TEST(ConnectionUpdateGroupsAsyncTest, UpdatingFlagVisibleAcrossTicks)
{
    ScriptedTransProvider provider;
    provider.onCreate = []() { std::this_thread::sleep_for(std::chrono::milliseconds(200)); };
    ConnectionManager mgr(provider, "");
    mgr.StartRecoverLoop();

    auto future = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.65.1"), MakeEndpoint("10.0.65.2")},
                                        {});
    // 入队即置位：worker 处理第一个 add（200ms sleep）期间从外部观测。
    NodeHealthSnapshot snapshot;
    mgr.GetHealthSnapshot(snapshot);
    EXPECT_TRUE(snapshot.updating);

    ASSERT_EQ(future.wait_for(std::chrono::seconds(4)), std::future_status::ready);
    const auto result = future.get();
    ASSERT_TRUE(result.status.ok()) << result.status.message;
    EXPECT_FALSE(result.updating);

    mgr.GetHealthSnapshot(snapshot);
    EXPECT_FALSE(snapshot.updating);
    mgr.StopRecoverLoop();
}

TEST(ConnectionUpdateGroupsAsyncTest, SyncWrapperWorksWhileLoopRunning)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    mgr.StartRecoverLoop();

    // 同步包装 = 入队 + future.get()：循环运行时语义与内联一致。
    UpdateResult result;
    const auto status = mgr.UpdateGroups({MakeEndpoint("10.0.66.1")}, {}, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{1});
    EXPECT_EQ(result.addResults.front().action, "created");
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});

    mgr.StopRecoverLoop();
}

TEST(ConnectionUpdateGroupsAsyncTest, InlineFallbackWhenLoopNotRunning)
{
    ScriptedTransProvider provider;
    ConnectionManager mgr(provider, "");
    // 不启动循环：必须内联执行（返回已就绪 future），行为与旧同步接口一致。
    auto future = mgr.UpdateGroupsAsync({MakeEndpoint("10.0.67.1")}, {});
    ASSERT_EQ(future.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    const auto result = future.get();
    ASSERT_TRUE(result.status.ok()) << result.status.message;
    ASSERT_EQ(result.addResults.size(), std::size_t{1});
    EXPECT_EQ(result.addResults.front().action, "created");
    EXPECT_EQ(mgr.GetServerCapabilities().size(), std::size_t{1});
    EXPECT_FALSE(result.updating);
}

}  // namespace kv::test
