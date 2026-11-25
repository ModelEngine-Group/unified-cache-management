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

// Shard 3 DT：attempt 重试路由（BUSY 原链 / FAILOVER 换链 / RA）、退避计算、
// 双层 Deadline、迟到 CQE、PrepareRetryRequest 的一致性。
//
// 测试因子分解：
//   F1 错误类别   ∈ {CQE_RESOURCE_BUSY, CQE_WRITE_FAULT, ENTRY_RETRY_ADVISED,
//                     CQE_IO_TIMEOUT(attempt), CQE_KEY_NOT_EXIST(query miss)}
//   F2 重试预算   ∈ {M/N 边界 0, 1, 2, 耗尽}
//   F3 退避       ∈ {busy 增长型指数 min(max, base*2^(k-1)), 换链无退避}
//   F4 Deadline   ∈ {attempt 内超时, 任务总 Deadline 未发/挂起重试}
//   F5 迟到响应   ∈ {Channel 排空后到达的响应}
// 断言面：pendingRetryCategory / busyRetries / attemptedChannels / retryAt /
//         原始错误保留 / SQE CID+响应Buffer 同步 / Channel 健康记账。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "task/trans_task_executor.h"
#undef private
#include "conn/connection_internal.h"
#include "conn/connection_manager.h"
#include "trans_provider.h"

namespace kv::test {
namespace {

constexpr std::size_t kTestBufferSlotSize = 512;
constexpr std::size_t kTestBufferSlotNum = 64;

// 遵守 DeleteConnections 排空契约的可编程发送桩：只记账，不产生写回。
// CQE 内容由测试直接写入 flag buffer（控制响应时序与内容）。
class ScriptedSendProvider : public TransProvider {
public:
    std::atomic<int> createCount{0};
    std::atomic<int> deleteCount{0};
    std::atomic<int> sendCallCount{0};

    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                            uint32_t, std::vector<ConnectionHandle>& handles) override
    {
        const auto sequence = createCount.fetch_add(1) + 1;
        handles.clear();
        for (std::uint32_t i = 0; i < qpNum; ++i) {
            handles.push_back(reinterpret_cast<ConnectionHandle>(
                static_cast<uintptr_t>(sequence * 100 + i + 1)));
        }
        return Status::OK();
    }

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override
    {
        deleteCount.fetch_add(static_cast<int>(handles.size()));
        return std::vector<Status>(handles.size(), Status::OK());
    }

    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>& batches, uint32_t,
                             uint32_t) override
    {
        sendCallCount.fetch_add(1);
        // 默认逐 batch 返回 OK（发送同步成功）；响应内容仍由测试写 CQE 控制。
        return std::vector<Status>(batches.size(), Status::OK());
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

TransportConfig MakeExecutorConfig(std::uint32_t busyRetryCount, std::uint32_t failoverRetryCount)
{
    TransportConfig config;
    config.retry.busyRetryCount = busyRetryCount;
    config.retry.busyRetryBackoffBaseMs = 1;
    config.retry.busyRetryBackoffMaxMs = 100;
    config.retry.failoverRetryCount = failoverRetryCount;
    config.retry.taskTimeoutMs = 5000;
    config.retry.attemptTimeoutMs = 100;
    config.connection.maxErrorCount = 1;
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

class TransportTaskExecutorRetryTest : public ::testing::Test {
protected:
    // 显式参数化 SetUp：每个用例自行声明组数/预算。
    void SetupEnvironment(std::uint32_t groupCount, std::uint32_t qpNum,
                          std::uint32_t busyRetryCount, std::uint32_t failoverRetryCount)
    {
        // 注意：executor 的 transProvider_ 是引用成员，必须绑定到稳定存在的
        // shared_ptr<TransProvider>（基类句柄），避免派生->基类转换临时量悬垂。
        provider_ = std::make_shared<ScriptedSendProvider>();
        config_ = MakeExecutorConfig(busyRetryCount, failoverRetryCount);
        connManager_ = std::make_unique<ConnectionManager>(*provider_, "", config_);
        for (std::uint32_t index = 0; index < groupCount; ++index) {
            ASSERT_TRUE(
                connManager_->AddGroup(MakeEndpoint("10.0.0." + std::to_string(index + 1)), qpNum)
                    .ok());
        }
        executor_ = std::make_unique<TransportTaskExecutor>(config_, provider_, connManager_);
        auto status = executor_->sendBufferManager_.Init("test send buffer", MemoryType::HOST,
                                                          kTestBufferSlotSize, kTestBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        status = executor_->flagBufferManager_.Init("test flag buffer", MemoryType::HOST,
                                                    kTestBufferSlotSize, kTestBufferSlotNum);
        ASSERT_TRUE(status.ok()) << status.message;
        status = executor_->RegisterBufferMemory(executor_->sendBufferManager_,
                                                 executor_->sendBufferMrHandle_);
        ASSERT_TRUE(status.ok()) << status.message;
        status = executor_->RegisterBufferMemory(executor_->flagBufferManager_,
                                                 executor_->flagBufferMrHandle_);
        ASSERT_TRUE(status.ok()) << status.message;
    }

    void TearDown() override
    {
        if (executor_) { (void)executor_->Shutdown(); }
        if (connManager_) { (void)connManager_->Shutdown(); }
    }

    // 构造 INFLIGHT 任务：subBatch 已在 channel 上发送（flag buffer 已分配）。
    TransportTaskPtr MakeInFlightTask(std::size_t subBatchCount = 1,
                                       AsuOpType opType = AsuOpType::BATCH_STORE)
    {
        auto task = std::make_shared<TransportTask>();
        task->state.store(TransportTaskState::INFLIGHT, std::memory_order_release);
        task->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
        task->subBatchContexts->resize(subBatchCount);
        std::uint16_t nextCid = 100;
        for (auto& subBatchContext : *task->subBatchContexts) {
            subBatchContext.opType = opType;
            subBatchContext.cid = nextCid++;  // 非 0 才会被 Poll 匹配
            subBatchContext.entryStatus.assign(1, Status::OK());
            subBatchContext.channel = connManager_->SelectConnection();
            subBatchContext.group = subBatchContext.channel->LockGroup();
            if (subBatchContext.channel == nullptr) { return nullptr; }
            // 保留可重打包的合法原始请求（batch_number 与 entries 一致）。
            auto request = std::make_shared<KvBatchStoreRequest>();
            request->batch_number = 1;
            request->rflag = true;
            KvBatchStoreEntry requestEntry;
            requestEntry.key = CacheKey{std::byte{1}};
            requestEntry.buffer_addr = 0x1000;
            requestEntry.mr_key = 1;
            requestEntry.length = 512;  // 协议要求 512B 对齐
            request->entries.push_back(requestEntry);
            subBatchContext.request = std::move(request);
            const auto bufferStatus =
                executor_->flagBufferManager_.Allocate(64, subBatchContext.flagBuffer);
            if (!bufferStatus.ok()) { return nullptr; }
            subBatchContext.attemptDeadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        }
        task->InitializeRemainingSubBatchCount();
        return task;
    }

    // 向 subBatch 的响应缓冲写入指定 raw CQE 状态（0=OK）。
    void WriteCqe(TransportSubBatchContext& subBatchContext, std::uint32_t rawStatus)
    {
        auto* cqe = reinterpret_cast<std::uint32_t*>(subBatchContext.flagBuffer.local_addr);
        cqe[3] = subBatchContext.cid | (rawStatus << 17);
    }

    // 基类句柄：executor 的引用成员必须绑定到稳定对象（见 SetupEnvironment 注释）。
    std::shared_ptr<TransProvider> provider_;
    TransportConfig config_;
    std::unique_ptr<ConnectionManager> connManager_;
    std::unique_ptr<TransportTaskExecutor> executor_;
};

// ─── F1+F3: CQE_RESOURCE_BUSY → 原链重试，增长型退避，耗尽保留原始错误 ───

TEST_F(TransportTaskExecutorRetryTest, BusyRetriesOnSameChannelThenKeepsOriginalError)
{
    SetupEnvironment(1, 1, /*busyRetryCount=*/2, /*failoverRetryCount=*/0);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto channel = sb.channel;
    WriteCqe(sb, 0x731);  // CQE_RESOURCE_BUSY

    executor_->Poll(task);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::BUSY);
    EXPECT_EQ(sb.busyRetries, 1u);
    EXPECT_EQ(sb.channel.get(), channel.get());  // 原链不换
    EXPECT_EQ(channel->GetErrorCount(), 0u);      // busy 不记失败
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
    EXPECT_EQ(sb.status.code, StatusCode::CQE_RESOURCE_BUSY);
    // 失败 attempt 的资源已释放（等待退避，不占 Slot）。
    EXPECT_EQ(sb.flagBuffer.slot_index, UINT32_MAX);

    // 退避窗口未到：不启动新 attempt。
    EXPECT_FALSE(executor_->HasRetryWork(task));
    sb.retryAt = std::chrono::steady_clock::now();  // DT 直接推进退避时钟
    EXPECT_TRUE(executor_->HasRetryWork(task));

    const auto firstCid = sb.cid;
    executor_->ExecuteRetry(task);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::NONE);  // 发送前复位
    EXPECT_NE(sb.cid, firstCid);                             // 新 CID
    EXPECT_EQ(sb.channel.get(), channel.get());               // 仍是原链
    EXPECT_EQ(channel->GetInflightCount(), 1u);
    ASSERT_NE(sb.flagBuffer.slot_index, UINT32_MAX);

    // 第二次 busy：busyRetries=2，仍在预算内挂起。
    WriteCqe(sb, 0x731);
    executor_->Poll(task);
    EXPECT_EQ(sb.busyRetries, 2u);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::BUSY);

    // 第三次 busy：M=2 耗尽，最终 Busy 不换链，保留原始错误。
    sb.retryAt = std::chrono::steady_clock::now();
    executor_->ExecuteRetry(task);
    WriteCqe(sb, 0x731);
    executor_->Poll(task);
    EXPECT_EQ(sb.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(sb.status.code, StatusCode::CQE_RESOURCE_BUSY);
    EXPECT_EQ(sb.entryStatus[0].code, StatusCode::CQE_RESOURCE_BUSY);
    EXPECT_EQ(channel->GetErrorCount(), 0u);  // 全程不记链路失败
}

// ─── F1+F2: 链路故障换链，排除已试 Channel，N 次耗尽保留原始错误 ───

TEST_F(TransportTaskExecutorRetryTest, LinkFailureFailoversExcludingAttemptedChannels)
{
    SetupEnvironment(3, 1, /*busyRetryCount=*/0, /*failoverRetryCount=*/2);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto firstChannel = sb.channel;
    WriteCqe(sb, 0x280);  // CQE_WRITE_FAULT

    executor_->Poll(task);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::LINK_FAILURE);
    EXPECT_TRUE(executor_->HasRetryWork(task));  // 换链无退避
    ASSERT_EQ(sb.attemptedChannels.size(), std::size_t{1});
    EXPECT_EQ(sb.attemptedChannels[0], firstChannel.get());
    // X=1：故障链路已记账排空。
    EXPECT_EQ(firstChannel->GetState(), ChannelState::FAILED);

    executor_->ExecuteRetry(task);
    ASSERT_NE(sb.channel.get(), firstChannel.get());  // 排除已试 Channel
    auto secondChannel = sb.channel;
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::NONE);  // 已复位并发出新 attempt
    ASSERT_NE(sb.flagBuffer.slot_index, UINT32_MAX);

    WriteCqe(sb, 0x280);
    executor_->Poll(task);
    ASSERT_EQ(sb.attemptedChannels.size(), std::size_t{2});
    EXPECT_EQ(sb.attemptedChannels[1], secondChannel.get());

    executor_->ExecuteRetry(task);
    WriteCqe(sb, 0x280);
    executor_->Poll(task);
    // N=2 耗尽：不再换链，保留原始错误（CQE_WRITE_FAULT）。
    EXPECT_EQ(sb.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(sb.status.code, StatusCode::CQE_WRITE_FAULT);
    EXPECT_EQ(sb.entryStatus[0].code, StatusCode::CQE_WRITE_FAULT);
    ASSERT_EQ(sb.attemptedChannels.size(), std::size_t{3});
    EXPECT_TRUE(task->Done());
}

// ─── F1: ENTRY_RETRY_ADVISED → 换链无退避、不降载、不记失败 ───

TEST_F(TransportTaskExecutorRetryTest, RetryAdvisedFailoversImmediatelyWithoutAccounting)
{
    SetupEnvironment(2, 1, 0, 1);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto firstChannel = sb.channel;
    // RA 只存在于逐条目表达场景：CQE_CHECK_RESULT_BUFFER + 结果缓冲携带 RA(0x01)。
    WriteCqe(sb, 0x732);
    auto* cqe = reinterpret_cast<std::uint32_t*>(sb.flagBuffer.local_addr);
    cqe[4] = 0x1;  // entry0 结果 = 0x01（ENTRY_RETRY_ADVISED）

    executor_->Poll(task);
    EXPECT_EQ(sb.entryStatus[0].code, StatusCode::ENTRY_RETRY_ADVISED);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::LINK_FAILURE);
    EXPECT_TRUE(executor_->HasRetryWork(task));  // 无退避
    EXPECT_EQ(firstChannel->GetErrorCount(), 0u);  // RA 不记失败
    EXPECT_EQ(firstChannel->GetState(), ChannelState::ACTIVE);  // 不排空不降载

    const auto oldCid = sb.cid;
    executor_->ExecuteRetry(task);
    EXPECT_NE(sb.channel.get(), firstChannel.get());
    EXPECT_NE(sb.cid, oldCid);
}

// ─── F4: attempt 内无响应超时 → 合成 CQE_IO_TIMEOUT 按链路故障换链 ───

TEST_F(TransportTaskExecutorRetryTest, AttemptTimeoutSynthesizesIoTimeoutAndFailovers)
{
    SetupEnvironment(2, 1, 0, 1);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto firstChannel = sb.channel;
    sb.attemptDeadline = std::chrono::steady_clock::now();  // attempt 已到期，无 CQE

    executor_->Poll(task);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::LINK_FAILURE);
    EXPECT_EQ(sb.status.code, StatusCode::CQE_IO_TIMEOUT);
    EXPECT_EQ(sb.entryStatus[0].code, StatusCode::CQE_IO_TIMEOUT);
    EXPECT_EQ(firstChannel->GetState(), ChannelState::FAILED);  // X=1 记账排空

    executor_->ExecuteRetry(task);
    EXPECT_NE(sb.channel.get(), firstChannel.get());
    ASSERT_NE(sb.flagBuffer.slot_index, UINT32_MAX);  // 新 attempt 已就位
}

// ─── F5: Channel 排空后到达的响应 → 合成 IO_TIMEOUT，不以旧响应裁决 ───

TEST_F(TransportTaskExecutorRetryTest, LateCqeOnDrainedChannelIsIsolated)
{
    SetupEnvironment(2, 1, 0, 1);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto firstChannel = sb.channel;
    WriteCqe(sb, 0x00);  // 响应已写好（看似成功）
    firstChannel->MarkForDrain();  // 链路已被排空（迟到场景）

    executor_->Poll(task);
    // 迟到响应隔离：即使 CQE 已到也合成 IO_TIMEOUT，不采用旧响应裁决。
    EXPECT_EQ(sb.status.code, StatusCode::CQE_IO_TIMEOUT);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::LINK_FAILURE);
    EXPECT_NE(sb.entryStatus[0].code, StatusCode::OK);
}

// ─── F4: 任务总 Deadline 到期：挂起重试不再启动新 attempt，未完成 Entry=TIMEOUT ───

TEST_F(TransportTaskExecutorRetryTest, TaskDeadlineStopsNewAttemptsAndTimesOutUnsent)
{
    SetupEnvironment(1, 1, 1, 1);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto channel = sb.channel;
    WriteCqe(sb, 0x731);  // busy 挂起
    executor_->Poll(task);
    ASSERT_EQ(sb.pendingRetryCategory, RetryCategory::BUSY);

    task->deadline = std::chrono::steady_clock::now();  // 总 Deadline 到期
    sb.retryAt = std::chrono::steady_clock::now();

    EXPECT_FALSE(executor_->HasRetryWork(task));  // 不再启动新 attempt
    executor_->ExecuteRetry(task);
    EXPECT_NE(sb.pendingRetryCategory, RetryCategory::NONE);  // 仍挂起（未被驱动）

    executor_->Poll(task);
    // 未发超时：TIMEOUT 终结，不换链不记失败。
    EXPECT_EQ(sb.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(sb.status.code, StatusCode::TIMEOUT);
    EXPECT_EQ(sb.entryStatus[0].code, StatusCode::TIMEOUT);
    EXPECT_EQ(channel->GetErrorCount(), 0u);
    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->finalStatus.code, StatusCode::TIMEOUT);
}

// ─── F1: query miss 正常完成，不重试不记失败 ───

TEST_F(TransportTaskExecutorRetryTest, QueryMissCompletesWithoutRetry)
{
    SetupEnvironment(1, 1, 1, 1);
    auto task = MakeInFlightTask(1, AsuOpType::QUERY);
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto channel = sb.channel;
    WriteCqe(sb, 0x701);  // CQE_KEY_NOT_EXIST：QUERY 下为 query miss

    executor_->Poll(task);
    EXPECT_EQ(sb.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::NONE);
    EXPECT_EQ(channel->GetErrorCount(), 0u);
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
    EXPECT_TRUE(task->Done());
}

// ─── PrepareRetryRequest：CID 与响应 Buffer/MR 同步更新 ───

TEST_F(TransportTaskExecutorRetryTest, PrepareRetryRequestSyncsCidAndResponseBuffer)
{
    SetupEnvironment(2, 1, 0, 1);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    WriteCqe(sb, 0x716);  // CQE_IO_TIMEOUT 链路故障
    executor_->Poll(task);
    ASSERT_EQ(sb.pendingRetryCategory, RetryCategory::LINK_FAILURE);

    const auto oldCid = sb.cid;
    auto* request = static_cast<KvBatchStoreRequest*>(sb.request.get());
    const auto oldResponseAddr = request->response_buffer_addr;

    executor_->ExecuteRetry(task);

    EXPECT_NE(sb.cid, oldCid);
    ASSERT_NE(sb.flagBuffer.slot_index, UINT32_MAX);
    EXPECT_EQ(request->cid, sb.cid);  // SQE 中的 CID 已同步
    EXPECT_EQ(request->response_buffer_addr, sb.flagBuffer.device_addr);  // 响应 Buffer 同步
    EXPECT_EQ(request->response_mr_key, sb.flagBuffer.tokenId);
    EXPECT_NE(request->response_buffer_addr, oldResponseAddr);
    ASSERT_NE(sb.sendSge.slot_index, UINT32_MAX);  // SQE 已重打包进新 Slot
}

// ─── F2 边界：M=0（关闭 BUSY 重试）───

TEST_F(TransportTaskExecutorRetryTest, ZeroBusyRetryBudgetKeepsBusyErrorImmediately)
{
    SetupEnvironment(1, 1, /*busyRetryCount=*/0, 0);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    auto channel = sb.channel;  // 终态会释放 channel 引用，先捕获
    WriteCqe(sb, 0x731);

    executor_->Poll(task);
    EXPECT_EQ(sb.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(sb.status.code, StatusCode::CQE_RESOURCE_BUSY);
    EXPECT_EQ(sb.entryStatus[0].code, StatusCode::CQE_RESOURCE_BUSY);
    EXPECT_EQ(channel->GetErrorCount(), 0u);  // busy 全程不记链路失败
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
}

// ─── F2 边界：N=0（关闭换链）───

TEST_F(TransportTaskExecutorRetryTest, ZeroFailoverBudgetKeepsLinkErrorImmediately)
{
    SetupEnvironment(2, 1, 0, /*failoverRetryCount=*/0);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];
    WriteCqe(sb, 0x716);  // CQE_IO_TIMEOUT

    executor_->Poll(task);
    EXPECT_EQ(sb.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(sb.status.code, StatusCode::CQE_IO_TIMEOUT);
    EXPECT_TRUE(task->Done());
}

// ─── F2 混合类别边界：BUSY 与链路故障交替，预算独立单调、总有界 ───
// 防回归：类别振荡不得逃逸预算记账（每 attempt 恰好消耗 busyRetries 或
// attemptedChannels 之一；鸽笼保证总重试 <= M+N，不会重试到超时）。
TEST_F(TransportTaskExecutorRetryTest, MixedCategoryErrorsDrainIndependentBudgetsTerminally)
{
    SetupEnvironment(3, 1, /*busyRetryCount=*/2, /*failoverRetryCount=*/2);
    auto task = MakeInFlightTask();
    ASSERT_NE(task, nullptr);
    auto& sb = (*task->subBatchContexts)[0];

    // 交替序列：BUSY -> LINK_FAILURE -> BUSY -> LINK_FAILURE -> BUSY。

    // 第 1 次 BUSY：busyRetries=1，原链重试。
    WriteCqe(sb, 0x731);
    executor_->Poll(task);
    ASSERT_EQ(sb.pendingRetryCategory, RetryCategory::BUSY);
    sb.retryAt = std::chrono::steady_clock::now();  // DT 直接推进退避时钟
    executor_->ExecuteRetry(task);

    // 第 2 次 链路故障：attempted=1，换链（跨组优先）。
    WriteCqe(sb, 0x280);
    executor_->Poll(task);
    ASSERT_EQ(sb.pendingRetryCategory, RetryCategory::LINK_FAILURE);
    executor_->ExecuteRetry(task);

    // 第 3 次 BUSY：busyRetries=2，仍 <= M。
    WriteCqe(sb, 0x731);
    executor_->Poll(task);
    ASSERT_EQ(sb.pendingRetryCategory, RetryCategory::BUSY);
    EXPECT_EQ(sb.busyRetries, 2u);
    sb.retryAt = std::chrono::steady_clock::now();
    executor_->ExecuteRetry(task);

    // 第 4 次 链路故障：attempted=2，仍 <= N。
    WriteCqe(sb, 0x280);
    executor_->Poll(task);
    ASSERT_EQ(sb.pendingRetryCategory, RetryCategory::LINK_FAILURE);
    EXPECT_EQ(sb.attemptedChannels.size(), std::size_t{2});
    executor_->ExecuteRetry(task);

    // 第 5 次 BUSY：busyRetries(2) >= M(2) —— 任一预算耗尽即终止，
    // 不无限重试到超时；保留该次的原始错误（非笼统 TIMEOUT）。
    WriteCqe(sb, 0x731);
    executor_->Poll(task);
    EXPECT_EQ(sb.state, TransportSubBatchState::COMPLETED);
    EXPECT_EQ(sb.status.code, StatusCode::CQE_RESOURCE_BUSY);
    EXPECT_EQ(sb.entryStatus[0].code, StatusCode::CQE_RESOURCE_BUSY);
    EXPECT_EQ(sb.pendingRetryCategory, RetryCategory::NONE);
    EXPECT_TRUE(task->Done());

    // 总尝试数 = 1 初始 + M(2) 原链 + N(2) 换链 = 5：重试发送恰 4 次。
    EXPECT_EQ(static_cast<ScriptedSendProvider*>(provider_.get())->sendCallCount.load(), 4);
}

}  // namespace kv::test
