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

// Shard 5 DT：GroupFlowController 两级预算、-2/+1 迟滞、稳定成功解除、
// 静态准入 fail-fast、busyFloor/ioQueueDepth 关系校验、执行器/管理器接线。
//
// 测试因子分解：
//   F1 预算派生   ∈ {ioQueueDepth=0/小于配置/等于配置/大于配置} × {通道数 1,2}
//   F2 关系校验   ∈ {busyFloor < limit, == limit-1(边界), >= limit(拒绝)}
//   F3 ON_BUSY    ∈ {单次-2, 连续-2, 达 floor, floor 处保持}（迟滞快降）
//   F4 业务成功   ∈ {+1 慢升, 达 normal 恢复, 稳定成功恢复,
//                    成功间隔 > stableSuccessMs 重开窗口, 非成功复位}
//   F5 准入       ∈ {inflight = limit-1, = limit（边界）, 降载态 < floor}
//   F6 接线       ∈ {AddGroup 挂接/拒绝, 选路 fail-fast, 执行器 ON_BUSY/升载}

#include <chrono>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "task/trans_task_executor.h"
#undef private
#include "conn/connection_internal.h"
#include "conn/connection_manager.h"
#include "conn/group_flow_controller.h"
#include "trans_provider.h"

namespace kv::test {
namespace {

FlowControlConfig MakeFlowConfig(std::uint32_t maxInflightPerChannel = 10,
                                 std::uint32_t busyFloor = 4,
                                 std::uint64_t stableSuccessMs = 100)
{
    FlowControlConfig config;
    config.flowControlEnabled = true;
    config.maxInflightPerChannel = maxInflightPerChannel;
    config.busyFloor = busyFloor;
    config.stableSuccessMs = stableSuccessMs;
    return config;
}

}  // namespace

// ─── F1: 预算派生（等价类与边界） ───

TEST(GroupFlowControllerTest, NormalLimitDerivesFromConfigAndIoQueueDepth)
{
    {
        // ioQueueDepth=0（未上报能力）：直接用配置上限。2 通道 → 20。
        GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
        EXPECT_EQ(controller.NormalLimit(), std::uint32_t{20});
        EXPECT_EQ(controller.BusyFloor(), std::uint32_t{8});
        EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{20});
        EXPECT_FALSE(controller.Degraded());
    }
    {
        // ioQueueDepth < 配置：被服务端能力收紧。16 → 每通道 min(10,16)=10。
        GroupFlowController controller(MakeFlowConfig(10, 4), 2, 16);
        EXPECT_EQ(controller.NormalLimit(), std::uint32_t{20});
    }
    {
        // ioQueueDepth > 配置：配置上限生效（3 < 10）；busyFloor 由 Create 保证
        // 小于有效上限，此处用合法配置（busyFloor=2 < 3）验证派生。
        GroupFlowController controller(MakeFlowConfig(10, 2), 2, 3);
        EXPECT_EQ(controller.NormalLimit(), std::uint32_t{6});
        EXPECT_EQ(controller.BusyFloor(), std::uint32_t{4});  // Σ busyFloor = 2x2
    }
    {
        // ioQueueDepth == 配置（边界）：两者相等。
        GroupFlowController controller(MakeFlowConfig(10, 4), 1, 10);
        EXPECT_EQ(controller.NormalLimit(), std::uint32_t{10});
        EXPECT_EQ(controller.BusyFloor(), std::uint32_t{4});
    }
}

// ─── F2: busyFloor 关系校验（Init 拒绝边界） ───

TEST(GroupFlowControllerTest, BusyFloorMustBeSmallerThanEffectiveIoQueueDepth)
{
    std::unique_ptr<GroupFlowController> controller;

    // busyFloor < 有效上限：合法。
    EXPECT_TRUE(
        GroupFlowController::Create(MakeFlowConfig(10, 9), 2, 10, controller).ok());
    EXPECT_NE(controller, nullptr);

    // 边界：busyFloor == 有效上限-1 合法。
    controller.reset();
    EXPECT_TRUE(GroupFlowController::Create(MakeFlowConfig(4, 3), 1, 4, controller).ok());

    // busyFloor == 有效上限：拒绝（INVALID_ARGUMENT，禁止静默收紧）。
    controller.reset();
    EXPECT_EQ(GroupFlowController::Create(MakeFlowConfig(4, 4), 1, 4, controller).code,
              StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(controller, nullptr);

    // busyFloor > ioQueueDepth 收紧后的上限：拒绝。
    EXPECT_EQ(GroupFlowController::Create(MakeFlowConfig(10, 5), 1, 4, controller).code,
              StatusCode::INVALID_ARGUMENT);
}

// ─── F3: ON_BUSY 快降（-2 迟滞，下限 busyFloor） ───

TEST(GroupFlowControllerTest, OnBusyReducesBudgetByTwoDownToFloor)
{
    GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
    // normal=20, floor=8。
    controller.OnBusy();
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{18});
    EXPECT_TRUE(controller.Degraded());
    controller.OnBusy();
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{16});

    // 连续 busy 到达 floor：20-2x6=8。
    for (int i = 0; i < 4; ++i) { controller.OnBusy(); }
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{8});
    // 达 floor 后保持（下限 busyFloor）。
    controller.OnBusy();
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{8});
    EXPECT_EQ(controller.BusyFloor(), std::uint32_t{8});
}

TEST(GroupFlowControllerTest, OnBusySaturatesWhenLimitBelowFloorPlusTwo)
{
    // 临近下限的边界：limit=floor+1 → 一次 busy 直接落 floor（-2 不越下限）。
    GroupFlowController controller(MakeFlowConfig(10, 4), 1, 0);
    // normal=10, floor=4。5 次 busy: 10->8->6->4（第三次）->4->4。
    controller.OnBusy();
    controller.OnBusy();
    controller.OnBusy();
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{4});
    controller.OnBusy();
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{4});
}

// ─── F4: 业务成功慢升与稳定成功解除 ───

TEST(GroupFlowControllerTest, BusinessSuccessRaisesBudgetByOneUpToNormal)
{
    GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
    const auto now = std::chrono::steady_clock::now();
    controller.OnBusy();
    controller.OnBusy();   // 20 -> 18 -> 16
    ASSERT_EQ(controller.CurrentLimit(), std::uint32_t{16});

    // 慢升：+1/次（间隔小于 stableSuccessMs，不触发稳定解除）。
    controller.OnBusinessSuccess(now);
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{17});
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(10));
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{18});
}

TEST(GroupFlowControllerTest, ReachingNormalLimitRestoresFullBudget)
{
    GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
    const auto now = std::chrono::steady_clock::now();
    controller.OnBusy();   // 18
    controller.OnBusy();   // 16

    // +1 逐步逼近 normal：达到即解除降载、恢复全量。
    for (int i = 0; i < 3; ++i) {
        controller.OnBusinessSuccess(now + std::chrono::milliseconds(i));
    }
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{19});
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(5));
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{20});
    EXPECT_FALSE(controller.Degraded());
}

TEST(GroupFlowControllerTest, StableSuccessWindowRestoresFullBudget)
{
    // 稳定成功持续 >= stableSuccessMs（双语义之一）→ 恢复全量。
    GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
    const auto now = std::chrono::steady_clock::now();
    controller.OnBusy();
    controller.OnBusy();
    controller.OnBusy();   // 14
    ASSERT_EQ(controller.CurrentLimit(), std::uint32_t{14});

    // 窗口起算后持续成功 ≥ stableSuccessMs（=100ms）：在第 100ms 成功时解除。
    controller.OnBusinessSuccess(now);                          // 窗口起算，+1 -> 15
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(50));  // +1 -> 16
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{16});
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(100));  // 稳定达成
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{20});
    EXPECT_FALSE(controller.Degraded());
}

TEST(GroupFlowControllerTest, SuccessGapLargerThanStableWindowRestartsIt)
{
    // 双语义之二：距上次成功 > stableSuccessMs 视为窗口重启（最大成功响应间隔）。
    GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
    const auto now = std::chrono::steady_clock::now();
    controller.OnBusy();
    ASSERT_EQ(controller.CurrentLimit(), std::uint32_t{18});

    controller.OnBusinessSuccess(now);  // 窗口起算，+1 -> 19
    // 间隔 150ms > 100ms：窗口在 now+150ms 重启；+1 达 normal 亦解除降载。
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(150));
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{20});
    // 已恢复全量：保持 normalLimit。
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(210));
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{20});
}

TEST(GroupFlowControllerTest, NonSuccessResetsStableWindowWithoutRaising)
{
    GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
    const auto now = std::chrono::steady_clock::now();
    controller.OnBusy();
    controller.OnBusy();
    controller.OnBusy();   // 20 -> 14（远离 normal，避免 +1 直达恢复）
    ASSERT_EQ(controller.CurrentLimit(), std::uint32_t{14});

    // 建立稳定窗口中段。
    controller.OnBusinessSuccess(now);                          // +1 -> 15，窗口起算
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(60));  // +1 -> 16

    // 非成功（链路故障/超时/应用错误等）：复位稳定计数，不升载不减载。
    controller.OnNonSuccess();
    // 复位后窗口随下一次成功重开：80/100ms 各 +1，稳定窗口仅 20ms。
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(80));   // +1 -> 17
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(100));  // +1 -> 18
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{18});
    // 重开窗口（+80ms 起）在 180ms 处稳定达成（间隔均 < 100ms）。
    controller.OnBusinessSuccess(now + std::chrono::milliseconds(180));
    EXPECT_EQ(controller.CurrentLimit(), std::uint32_t{20});
}

// ─── F5: 静态准入 fail-fast（边界） ───

TEST(GroupFlowControllerTest, CanAcquireEnforcesCurrentLimitBoundary)
{
    GroupFlowController controller(MakeFlowConfig(10, 4), 2, 0);
    // normal=20。
    EXPECT_TRUE(controller.CanAcquire(19));   // limit-1：允许
    EXPECT_FALSE(controller.CanAcquire(20));  // == limit：达到即 fail-fast
    EXPECT_FALSE(controller.CanAcquire(21));  // 超限

    // 降载态：floor 内仍可准入。
    controller.OnBusy();
    controller.OnBusy();
    controller.OnBusy();
    ASSERT_EQ(controller.CurrentLimit(), std::uint32_t{14});
    EXPECT_TRUE(controller.CanAcquire(13));
    EXPECT_FALSE(controller.CanAcquire(14));
}

// ─── F6: 管理器/执行器接线 ───

namespace {

class StubFlowProvider : public TransProvider {
public:
    ServerKvCapabilities serverCapabilities;
    Status createOverride{Status::OK()};

    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                            uint32_t, std::vector<ConnectionHandle>& handles) override
    {
        handles.clear();
        if (!createOverride.ok()) { return createOverride; }
        for (std::uint32_t i = 0; i < qpNum; ++i) {
            handles.push_back(
                reinterpret_cast<ConnectionHandle>(static_cast<uintptr_t>(++sequence_)));
        }
        return Status::OK();
    }
    Status GetServerCapabilities(ConnectionHandle, ServerKvCapabilities& capabilities) override
    {
        capabilities = serverCapabilities;
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

NodeEndpoint MakeFlowEndpoint(const std::string& ip)
{
    NodeEndpoint ep;
    ep.ip = ip;
    ep.port = 16666;
    return ep;
}

TransportConfig FlowManagerConfig(const FlowControlConfig& flow)
{
    TransportConfig config;
    config.flowControl = flow;
    config.connection.maxErrorCount = 1;
    return config;
}

}  // namespace

TEST(GroupFlowWiringTest, ManagerAttachesControllerOnlyWhenEnabled)
{
    {
        // 未启用：无控制器，准入恒放行。
        StubFlowProvider provider;
        TransportConfig config = FlowManagerConfig(MakeFlowConfig());
        config.flowControl.flowControlEnabled = false;
        ConnectionManager manager(provider, "", config);
        ASSERT_TRUE(manager.AddGroup(MakeFlowEndpoint("10.0.0.1"), 2).ok());
        auto channel = manager.SelectConnection();
        ASSERT_NE(channel, nullptr);
        auto group = channel->LockGroup();
        ASSERT_NE(group, nullptr);
        EXPECT_EQ(group->GetFlowController(), nullptr);
        channel->ReleaseInflight();
    }
    {
        // 启用：AddGroup 挂接控制器（ioQueueDepth=16 → 每通道 min(10,16)=10）。
        StubFlowProvider provider;
        provider.serverCapabilities.ioQueueDepth = 16;
        ConnectionManager manager(provider, "", FlowManagerConfig(MakeFlowConfig()));
        ASSERT_TRUE(manager.AddGroup(MakeFlowEndpoint("10.0.0.1"), 2).ok());
        auto channel = manager.SelectConnection();
        ASSERT_NE(channel, nullptr);
        auto group = channel->LockGroup();
        ASSERT_NE(group, nullptr);
        const auto* controller = group->GetFlowController();
        ASSERT_NE(controller, nullptr);
        EXPECT_EQ(controller->NormalLimit(), std::uint32_t{20});
        channel->ReleaseInflight();
    }
}

TEST(GroupFlowWiringTest, ManagerRejectsGroupWhenBusyFloorNotBelowIoQueueDepth)
{
    StubFlowProvider provider;
    provider.serverCapabilities.ioQueueDepth = 4;  // 有效上限 4
    ConnectionManager manager(provider, "", FlowManagerConfig(MakeFlowConfig(10, 4)));
    const auto status = manager.AddGroup(MakeFlowEndpoint("10.0.0.1"), 2);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);  // busyFloor(4) >= 4
    EXPECT_EQ(manager.GetServerCapabilities().size(), std::size_t{0});  // 不上表
}

TEST(GroupFlowWiringTest, SelectionFailsFastAtGroupBudgetWhileChannelsHaveRoom)
{
    StubFlowProvider provider;
    provider.serverCapabilities.ioQueueDepth = 16;
    // maxInflightPerChannel=2, busyFloor=1, 2 通道：group normal=4, floor=2。
    ConnectionManager manager(provider, "", FlowManagerConfig(MakeFlowConfig(2, 1)));
    ASSERT_TRUE(manager.AddGroup(MakeFlowEndpoint("10.0.0.1"), 2).ok());

    // group 预算 4 -> 2（模拟合法 CQE busy 降载）。
    auto channelA = manager.SelectConnection();
    ASSERT_NE(channelA, nullptr);
    auto group = channelA->LockGroup();
    ASSERT_NE(group, nullptr);
    group->OnBusy();
    ASSERT_EQ(group->GetFlowController()->CurrentLimit(), std::uint32_t{2});

    // 预算 2 内仍可准入一条（group inflight=1）。
    auto channelB = manager.SelectConnection();
    ASSERT_NE(channelB, nullptr);
    ASSERT_NE(channelB.get(), channelA.get());  // per-channel 上限=2，落在另一条链路

    // group inflight=2 == currentLimit=2：达到即 fail-fast（各通道 per-channel
    // 余量仍在，但组预算已满）。
    EXPECT_EQ(manager.SelectConnection(), nullptr);
    EXPECT_EQ(group->GetFlowController()->CurrentLimit(), std::uint32_t{2});

    // 释放后在途回落，准入恢复。
    channelA->ReleaseInflight();
    auto resumed = manager.SelectConnection();
    ASSERT_NE(resumed, nullptr);
    resumed->ReleaseInflight();
    channelB->ReleaseInflight();
}

TEST(GroupFlowWiringTest, ExecutorBusyDrainsBudgetAndBusinessSuccessRaisesIt)
{
    // 执行器接线：合法 CQE busy -> OnBusy(-2)；业务完整成功 -> OnBusinessSuccess(+1)。
    class ScriptedProvider : public TransProvider {
    public:
        Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                                uint32_t, std::vector<ConnectionHandle>& handles) override
        {
            handles.clear();
            for (std::uint32_t i = 0; i < qpNum; ++i) {
                handles.push_back(reinterpret_cast<ConnectionHandle>(
                    static_cast<uintptr_t>(++sequence_)));
            }
            return Status::OK();
        }
        std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>&) override
        {
            return {};
        }
        std::vector<Status> Send(const std::vector<SendIoBatch>&, uint32_t,
                                 uint32_t) override
        {
            return std::vector<Status>(1, Status::OK());
        }
        Status RegisterMemory(const std::vector<RegisterMemoryDesc>&,
                              std::vector<MRHandle>& handles) override
        {
            handles.push_back(reinterpret_cast<MRHandle>(static_cast<uintptr_t>(1)));
            return Status::OK();
        }
        std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>&) override
        {
            return {};
        }
        Status AllocThread(uint32_t, const std::vector<uint32_t>&,
                           std::vector<ThreadHandle>&) override
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

    // executor 的 transProvider_ 是引用成员：必须绑定稳定的基类句柄
    //（派生->基类转换的临时量会悬垂）。
    std::shared_ptr<TransProvider> provider = std::make_shared<ScriptedProvider>();
    TransportConfig config;
    config.flowControl = MakeFlowConfig(64, 1);
    config.retry.busyRetryCount = 0;
    config.connection.maxErrorCount = 1;
    std::unique_ptr<ConnectionManager> manager =
        std::make_unique<ConnectionManager>(*provider, "", config);
    ASSERT_TRUE(manager->AddGroup(MakeFlowEndpoint("10.0.0.1"), 1).ok());
    std::unique_ptr<TransportTaskExecutor> executor =
        std::make_unique<TransportTaskExecutor>(config, provider, manager);
    ASSERT_TRUE(executor->sendBufferManager_.Init("flow send", MemoryType::HOST, 512, 8).ok());
    ASSERT_TRUE(executor->flagBufferManager_.Init("flow flag", MemoryType::HOST, 512, 8).ok());
    ASSERT_TRUE(executor->RegisterBufferMemory(executor->sendBufferManager_,
                                               executor->sendBufferMrHandle_)
                    .ok());
    ASSERT_TRUE(executor->RegisterBufferMemory(executor->flagBufferManager_,
                                               executor->flagBufferMrHandle_)
                    .ok());

    auto task = std::make_shared<TransportTask>();
    task->state.store(TransportTaskState::INFLIGHT, std::memory_order_release);
    task->deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    task->subBatchContexts->resize(1);
    auto& subBatch = (*task->subBatchContexts)[0];
    subBatch.opType = AsuOpType::BATCH_STORE;
    subBatch.cid = 100;
    subBatch.entryStatus.assign(1, Status::OK());
    subBatch.channel = manager->SelectConnection();
    ASSERT_NE(subBatch.channel, nullptr);
    subBatch.group = subBatch.channel->LockGroup();
    ASSERT_NE(subBatch.group, nullptr);
    subBatch.request = std::make_shared<KvBatchStoreRequest>();
    ASSERT_TRUE(executor->flagBufferManager_.Allocate(64, subBatch.flagBuffer).ok());
    subBatch.attemptDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    task->InitializeRemainingSubBatchCount();

    const auto* controller = subBatch.group->GetFlowController();
    ASSERT_NE(controller, nullptr);
    EXPECT_EQ(controller->CurrentLimit(), std::uint32_t{64});  // 1 通道 x min(64, 0)=64

    // 合法 CQE busy → 执行器分类 ON_BUSY → group 预算 -2。
    auto* cqe = reinterpret_cast<std::uint32_t*>(subBatch.flagBuffer.local_addr);
    cqe[3] = subBatch.cid | (0x731U << 17);
    executor->Poll(task);
    EXPECT_EQ(controller->CurrentLimit(), std::uint32_t{62});
    EXPECT_TRUE(controller->Degraded());
    // busyRetryCount=0：预算内不重试，直接以原始错误终结（不升载，复位稳定窗口）。
    EXPECT_EQ(subBatch.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(subBatch.status.code, StatusCode::CQE_RESOURCE_BUSY);

    (void)executor->Shutdown();
    (void)manager->Shutdown();
}

}  // namespace kv::test
