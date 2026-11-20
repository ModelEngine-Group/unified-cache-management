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
#include "task/task_manager.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "kv_client_impl.h"
#include "router/router.h"

namespace kv {
namespace {

constexpr NodeId kNodeA = 11;
constexpr NodeId kNodeB = 22;
constexpr NodeId kNodeC = 33;
constexpr std::uint64_t kWaitTimeoutMs = 500;
constexpr std::uint64_t kShortWaitTimeoutMs = 1;

BinaryKey MakeBinaryKey(std::uint64_t value)
{
    BinaryKey key{};
    std::memcpy(key.data(), &value, sizeof(value));
    return key;
}

ResolvedCacheKey MakeResolvedKey(std::uint64_t value)
{
    ResolvedCacheKey key;
    key.keyBytes = MakeBinaryKey(value);
    key.gnsid = 1;
    return key;
}

// 路由到传输层时使用的 key 视图（8 字节二进制内容）。
std::string KeyString(std::uint64_t value)
{
    const auto bytes = MakeBinaryKey(value);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

// 按 key 内容查表路由，未命中时回退到 fallback（kInvalidNodeId 表示不路由）。
class MapRouter final : public Router {
public:
    MapRouter(std::unordered_map<std::string, NodeId> owners, NodeId fallback)
        : Router(nullptr), owners_(std::move(owners)), fallback_(fallback)
    {
    }

private:
    NodeId RouteKey(const Key& key) const override
    {
        const auto iter = owners_.find(key);
        return iter == owners_.end() ? fallback_ : iter->second;
    }

    std::unordered_map<std::string, NodeId> owners_;
    NodeId fallback_;
};

// 可配置的传输层桩：立即完成或挂起等待测试手动触发完成。
class StubTransport final : public AsuTransport {
public:
    enum class Mode {
        kCompleteOnSubmit,
        kHold,
    };

    explicit StubTransport(Mode mode = Mode::kCompleteOnSubmit) : mode_(mode) {}

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

    Status Cancel(TaskId) override { return Status::OK(); }

    Status Submit(const TransportTaskPtr& task) override
    {
        submittedOpTypes.push_back(task->opType);
        submittedKeyCount += task->keys.size();
        submittedEntryCount += task->entries.size();
        for (const auto& key : task->keys) {
            submittedKeys.push_back(std::string(CacheKeyView(key)));
        }
        for (const auto& entry : task->entries) {
            submittedKeys.push_back(std::string(CacheKeyView(entry.key)));
            submittedOffsets.push_back(entry.offset);
        }
        if (!submitStatus.ok()) { return submitStatus; }

        if (mode_ == Mode::kHold) {
            heldTasks.push_back(task);
            return Status::OK();
        }
        if (task->onComplete) { task->onComplete(BuildResult(task)); }
        return Status::OK();
    }

    void CompleteHeld(std::size_t index)
    {
        const auto task = heldTasks.at(index);
        if (task->onComplete) { task->onComplete(BuildResult(task)); }
    }

    std::size_t HeldCount() const { return heldTasks.size(); }

    Status submitStatus{Status::OK()};
    // 非空时作为 query 的存在性结果，否则默认全部存在。
    std::vector<std::uint8_t> queryExists{};

    std::vector<AsuOpType> submittedOpTypes;
    std::size_t submittedKeyCount{0};
    std::size_t submittedEntryCount{0};
    // 按提交顺序记录的 key 内容与 entry offset，用于校验路由结果的组成与顺序。
    std::vector<std::string> submittedKeys;
    std::vector<std::uint32_t> submittedOffsets;

private:
    TaskResult BuildResult(const TransportTaskPtr& task) const
    {
        TaskResult result;
        result.status = Status::OK();
        result.entryStatus.assign(std::max(task->keys.size(), task->entries.size()), Status::OK());
        if (task->opType == AsuOpType::QUERY) {
            QueryResult queryResult;
            queryResult.exists =
                queryExists.empty() ? std::vector<std::uint8_t>(task->keys.size(), 1) : queryExists;
            result.queryResult = std::move(queryResult);
        }
        return result;
    }

    Mode mode_;
    std::vector<TransportTaskPtr> heldTasks;
};

std::shared_ptr<ViewSnapshot> MakeSnapshot(
    std::shared_ptr<Router> router,
    std::unordered_map<NodeId, std::shared_ptr<AsuTransport>> transports)
{
    auto snapshot = std::make_shared<ViewSnapshot>();
    snapshot->router = std::move(router);
    snapshot->transports = std::move(transports);
    for (const auto& item : snapshot->transports) { snapshot->asuIds.push_back(item.first); }
    return snapshot;
}

// 构造与 KvClientImpl 提交时一致的 ClientTask（entryStatus / queryResult 预分配）。
ClientTaskPtr MakeTask(AsuOpType opType, std::size_t itemCount,
                       std::shared_ptr<ViewSnapshot> snapshot)
{
    auto task = std::make_shared<ClientTask>();
    const auto now = std::chrono::steady_clock::now();
    task->opType = opType;
    task->viewSnapshot = std::move(snapshot);
    task->submittedAt = now;
    task->enqueuedAt = now;
    task->processingStartedAt = now;
    task->entryStatus.assign(itemCount, Status::OK());

    if (IsEntryBatchOp(opType)) {
        task->entries.reserve(itemCount);
        for (std::size_t index = 0; index < itemCount; ++index) {
            ResolvedKVBuffer entry;
            entry.key = MakeResolvedKey(index);
            entry.offset = static_cast<std::uint32_t>(index);
            task->entries.push_back(entry);
        }
        return task;
    }

    task->keys.reserve(itemCount);
    for (std::size_t index = 0; index < itemCount; ++index) {
        task->keys.push_back(MakeResolvedKey(index));
    }
    if (opType == AsuOpType::QUERY) { task->queryResult.exists.assign(itemCount, 0); }
    return task;
}

TaskId SubmitTask(ClientTaskManager& manager, const ClientTaskPtr& task)
{
    TaskId taskId = kInvalidTaskId;
    EXPECT_TRUE(manager.Submit(task, taskId).ok());
    return taskId;
}

// 把 8 字节二进制 key 还原为便于断言的整数。
std::uint64_t KeyValue(const std::string& key)
{
    std::uint64_t value = 0;
    if (key.size() < sizeof(value)) { return value; }
    std::memcpy(&value, key.data(), sizeof(value));
    return value;
}

std::size_t CountEntriesWithCode(const std::vector<Status>& entryStatus, StatusCode code)
{
    return static_cast<std::size_t>(
        std::count_if(entryStatus.begin(), entryStatus.end(),
                      [code](const Status& status) { return status.code == code; }));
}

// 构造一个用于直接调用静态完成接口的任务上下文（不经过路由与传输层）。
ClientTaskPtr MakeDirectTask(AsuOpType opType, std::size_t itemCount)
{
    auto task = std::make_shared<ClientTask>();
    task->opType = opType;
    task->entryStatus.assign(itemCount, Status::OK());
    if (opType == AsuOpType::QUERY) { task->queryResult.exists.assign(itemCount, 0); }
    return task;
}

// 追加一个已派发的传输层任务，并同步 remainingTransportTasks 计数。
TransportTaskPtr AddTransportTask(const ClientTaskPtr& task, std::vector<std::size_t> indices)
{
    auto transportTask = std::make_shared<TransportTask>();
    transportTask->originalIndices = std::move(indices);
    task->transportTasks.push_back(transportTask);
    task->remainingTransportTasks.fetch_add(1, std::memory_order_acq_rel);
    return transportTask;
}

TEST(ClientTaskManagerTest, ProcessNullTaskRejected)
{
    ClientTaskManager manager;
    EXPECT_EQ(manager.Process(nullptr).code, StatusCode::INVALID_ARGUMENT);
}

TEST(ClientTaskManagerTest, CheckReflectsTaskLifecycle)
{
    ClientTaskManager manager;
    // 未知任务视为已完成
    EXPECT_TRUE(manager.Check(1234));

    auto task = MakeTask(AsuOpType::LOAD, 2, nullptr);
    const auto taskId = SubmitTask(manager, task);
    // 提交后处于 PENDING，尚未完成
    EXPECT_FALSE(manager.Check(taskId));

    // 缺少 view 快照时 Process 失败，并直接把任务置为 COMPLETED
    EXPECT_EQ(manager.Process(task).code, StatusCode::NOT_INITIALIZED);
    EXPECT_TRUE(manager.Check(taskId));
    EXPECT_EQ(task->finalStatus.code, StatusCode::NOT_INITIALIZED);
    ASSERT_EQ(task->entryStatus.size(), 2U);
    for (const auto& entryStatus : task->entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::NOT_INITIALIZED);
    }
}

TEST(ClientTaskManagerTest, QueryTaskCompletesWithPrefixHits)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    transport->queryExists = {1, 1, 0};
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::QUERY, 3,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);

    ASSERT_TRUE(manager.Process(task).ok());
    EXPECT_TRUE(manager.Check(taskId));
    ASSERT_EQ(transport->submittedOpTypes.size(), 1U);
    EXPECT_EQ(transport->submittedOpTypes.front(), AsuOpType::QUERY);
    EXPECT_EQ(transport->submittedKeyCount, 3U);

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
    ASSERT_TRUE(result.queryResult.has_value());
    ASSERT_EQ(result.queryResult->exists.size(), 3U);
    EXPECT_EQ(result.queryResult->exists[0], 1);
    EXPECT_EQ(result.queryResult->exists[1], 1);
    EXPECT_EQ(result.queryResult->exists[2], 0);
    // 只统计从头开始连续命中的 key
    EXPECT_EQ(result.queryResult->prefixHitKeys, 2U);
    EXPECT_EQ(result.entryStatus.size(), 3U);
    // Wait 之后任务已从管理器移除
    EXPECT_TRUE(manager.Check(taskId));
}

TEST(ClientTaskManagerTest, WaitTimesOutWhileTransportPending)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>(StubTransport::Mode::kHold);
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::LOAD, 1,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);

    ASSERT_TRUE(manager.Process(task).ok());
    ASSERT_EQ(transport->HeldCount(), 1U);
    EXPECT_FALSE(manager.Check(taskId));

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, kShortWaitTimeoutMs, result).code, StatusCode::TIMEOUT);
    EXPECT_EQ(result.status.code, StatusCode::TIMEOUT);
    // 超时后任务同样被移除
    EXPECT_TRUE(manager.Check(taskId));

    // 迟到的完成回调不应使流程崩溃
    transport->CompleteHeld(0);
    EXPECT_TRUE(task->Done());
}

TEST(ClientTaskManagerTest, RoutesKeysToOwningTransports)
{
    ClientTaskManager manager;
    auto transportA = std::make_shared<StubTransport>();
    auto transportB = std::make_shared<StubTransport>();
    transportB->queryExists = {0};

    std::unordered_map<std::string, NodeId> owners;
    owners[KeyString(0)] = kNodeA;
    owners[KeyString(1)] = kNodeB;
    owners[KeyString(2)] = kNodeA;
    auto router = std::make_shared<MapRouter>(std::move(owners), kInvalidNodeId);
    auto task = MakeTask(AsuOpType::QUERY, 3,
                         MakeSnapshot(router, {
                                                  {kNodeA, transportA},
                                                  {kNodeB, transportB}
    }));
    const auto taskId = SubmitTask(manager, task);

    ASSERT_TRUE(manager.Process(task).ok());
    EXPECT_EQ(transportA->submittedKeyCount, 2U);
    EXPECT_EQ(transportB->submittedKeyCount, 1U);

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
    ASSERT_TRUE(result.queryResult.has_value());
    ASSERT_EQ(result.queryResult->exists.size(), 3U);
    // 各传输层的结果必须回填到路由前的原始下标
    EXPECT_EQ(result.queryResult->exists[0], 1);
    EXPECT_EQ(result.queryResult->exists[1], 0);
    EXPECT_EQ(result.queryResult->exists[2], 1);
    EXPECT_EQ(result.queryResult->prefixHitKeys, 1U);
}

TEST(ClientTaskManagerTest, UnroutedKeysFinishTaskImmediately)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    auto router =
        std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kInvalidNodeId);
    auto task = MakeTask(AsuOpType::DELETE, 2,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);

    ASSERT_TRUE(manager.Process(task).ok());
    EXPECT_TRUE(task->Done());
    // 无路由结果时不应向任何传输层派发任务
    EXPECT_TRUE(transport->submittedOpTypes.empty());

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
    EXPECT_EQ(result.entryStatus.size(), 2U);
}

TEST(ClientTaskManagerTest, MissingRoutedTransportFailsProcess)
{
    ClientTaskManager manager;
    auto transportB = std::make_shared<StubTransport>();
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    // 路由目标为 kNodeA，但快照里只有 kNodeB 的传输层
    auto task = MakeTask(AsuOpType::LOAD, 2,
                         MakeSnapshot(router, {
                                                  {kNodeB, transportB}
    }));
    const auto taskId = SubmitTask(manager, task);

    EXPECT_EQ(manager.Process(task).code, StatusCode::NOT_FOUND);
    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->finalStatus.code, StatusCode::NOT_FOUND);
    EXPECT_EQ(transportB->submittedEntryCount, 0U);

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, kWaitTimeoutMs, result).code, StatusCode::NOT_FOUND);
}

TEST(ClientTaskManagerTest, DispatchFailureMarksTaskPartialFailed)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    transport->submitStatus = Status::Error(StatusCode::CONNECTION_ERROR, "stub submit failure");
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::LOAD, 2,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);

    const auto status = manager.Process(task);
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    // 派发失败会附带 nodeId 上下文
    EXPECT_NE(status.message.find("nodeId="), std::string::npos);
    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->finalStatus.code, StatusCode::PARTIAL_FAILED);
    for (const auto& entryStatus : task->entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::CONNECTION_ERROR);
    }

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, kWaitTimeoutMs, result).code, StatusCode::PARTIAL_FAILED);
}

TEST(ClientTaskManagerTest, CompleteWithErrorFillsAllEntryStatus)
{
    auto task = MakeTask(AsuOpType::BATCH_STORE, 2, nullptr);
    const auto error = Status::Error(StatusCode::IO_ERROR, "prerequisite failed");

    ClientTaskManager::CompleteWithError(task, error);

    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->finalStatus.code, StatusCode::IO_ERROR);
    ASSERT_EQ(task->entryStatus.size(), 2U);
    for (const auto& entryStatus : task->entryStatus) {
        EXPECT_EQ(entryStatus.code, StatusCode::IO_ERROR);
        EXPECT_EQ(entryStatus.message, "prerequisite failed");
    }
}

TEST(ClientTaskManagerTest, DuplicateTransportCompletionIgnored)
{
    auto task = std::make_shared<ClientTask>();
    task->opType = AsuOpType::LOAD;
    task->entryStatus.assign(2, Status::OK());
    task->remainingTransportTasks.store(1);

    auto transportTask = std::make_shared<TransportTask>();
    transportTask->originalIndices = {0, 1};
    task->transportTasks.push_back(transportTask);

    TaskResult first;
    first.status = Status::OK();
    first.entryStatus = {Status::OK(), Status::Error(StatusCode::IO_ERROR, "entry failure")};
    ClientTaskManager::CompleteTransportTask(task, 0, first);

    EXPECT_TRUE(task->Done());
    // 单个 entry 失败不会把整体状态标记为 PARTIAL_FAILED，错误体现在 entryStatus 上
    EXPECT_EQ(task->finalStatus.code, StatusCode::OK);
    ASSERT_EQ(task->entryStatus.size(), 2U);
    EXPECT_EQ(task->entryStatus[0].code, StatusCode::OK);
    EXPECT_EQ(task->entryStatus[1].code, StatusCode::IO_ERROR);

    // 重复完成同一传输层任务必须被忽略，不能覆盖已写入的 entryStatus
    TaskResult duplicate;
    duplicate.status = Status::Error(StatusCode::CONNECTION_ERROR, "duplicate completion");
    duplicate.entryStatus.assign(2, duplicate.status);
    ClientTaskManager::CompleteTransportTask(task, 0, duplicate);

    EXPECT_EQ(task->entryStatus[0].code, StatusCode::OK);
    EXPECT_EQ(task->entryStatus[1].code, StatusCode::IO_ERROR);
    EXPECT_EQ(task->finalStatus.code, StatusCode::OK);
}

TEST(ClientTaskManagerTest, DrainRemovesCompletedTasks)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto snapshot = MakeSnapshot(router, {
                                             {kNodeA, transport}
    });

    auto first = MakeTask(AsuOpType::LOAD, 1, snapshot);
    auto second = MakeTask(AsuOpType::QUERY, 1, snapshot);
    const auto firstId = SubmitTask(manager, first);
    const auto secondId = SubmitTask(manager, second);
    ASSERT_TRUE(manager.Process(first).ok());
    ASSERT_TRUE(manager.Process(second).ok());

    EXPECT_TRUE(manager.Drain(kWaitTimeoutMs).ok());
    EXPECT_TRUE(manager.Check(firstId));
    EXPECT_TRUE(manager.Check(secondId));
}

TEST(ClientTaskManagerTest, DrainReportsTimeoutForPendingTask)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>(StubTransport::Mode::kHold);
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::LOAD, 1,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);
    ASSERT_TRUE(manager.Process(task).ok());

    EXPECT_EQ(manager.Drain(kShortWaitTimeoutMs).code, StatusCode::TIMEOUT);
    EXPECT_TRUE(manager.Check(taskId));

    transport->CompleteHeld(0);
}

// ===================== 任务注册表：Submit / Get / Remove =====================

TEST(ClientTaskManagerTest, SubmitRejectsNullTask)
{
    ClientTaskManager manager;
    TaskId taskId = 1234;

    const auto sharedStatus = manager.Submit(ClientTaskPtr{}, taskId);
    EXPECT_EQ(sharedStatus.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(sharedStatus.message.find("client task context is null"), std::string::npos);
    EXPECT_EQ(taskId, kInvalidTaskId);

    // 所有权版本的空指针同样必须被拒绝，且不留下任何注册项
    taskId = 5678;
    const auto uniqueStatus = manager.Submit(std::unique_ptr<ClientTask>{}, taskId);
    EXPECT_EQ(uniqueStatus.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(taskId, kInvalidTaskId);
    EXPECT_TRUE(manager.GetAll().empty());
}

TEST(ClientTaskManagerTest, SubmitRegistersTaskWithUniqueIncreasingIds)
{
    ClientTaskManager manager;
    auto first = MakeTask(AsuOpType::LOAD, 1, nullptr);
    auto second = MakeTask(AsuOpType::QUERY, 2, nullptr);

    const auto firstId = SubmitTask(manager, first);
    const auto secondId = SubmitTask(manager, second);
    EXPECT_NE(firstId, kInvalidTaskId);
    EXPECT_NE(secondId, kInvalidTaskId);
    EXPECT_LT(firstId, secondId);

    // 注册表按 id 索引，并且保存的就是调用方持有的同一个上下文对象
    EXPECT_EQ(manager.Get(firstId).get(), first.get());
    EXPECT_EQ(manager.Get(secondId).get(), second.get());
    EXPECT_EQ(manager.GetAll().size(), 2U);

    // 提交会把状态重置为 PENDING，并写入管理器分配的 id
    EXPECT_EQ(first->taskId, firstId);
    EXPECT_EQ(second->taskId, secondId);
    EXPECT_EQ(first->state.load(), ClientTaskState::PENDING);

    // 移除后 id 不会被复用，避免迟到的完成回调命中新任务
    ASSERT_TRUE(manager.Remove(firstId).ok());
    EXPECT_EQ(manager.Get(firstId), nullptr);
    auto third = MakeTask(AsuOpType::STORE, 1, nullptr);
    const auto thirdId = SubmitTask(manager, third);
    EXPECT_GT(thirdId, secondId);
}

TEST(ClientTaskManagerTest, SubmitAcceptsUniquePtrOwnership)
{
    ClientTaskManager manager;
    auto task = std::make_unique<ClientTask>();
    task->opType = AsuOpType::STORE;
    ResolvedKVBuffer entry;
    entry.key = MakeResolvedKey(0);
    task->entries.push_back(entry);
    task->entryStatus.assign(1, Status::OK());

    TaskId taskId = kInvalidTaskId;
    ASSERT_TRUE(manager.Submit(std::move(task), taskId).ok());

    auto stored = manager.Get(taskId);
    ASSERT_NE(stored, nullptr);
    EXPECT_EQ(stored->taskId, taskId);
    EXPECT_EQ(stored->opType, AsuOpType::STORE);
    ASSERT_EQ(stored->entries.size(), 1U);
    EXPECT_EQ(stored->state.load(), ClientTaskState::PENDING);
}

TEST(ClientTaskManagerTest, RemoveUnknownTaskReportsNotFound)
{
    ClientTaskManager manager;
    const auto missing = manager.Remove(4321);
    EXPECT_EQ(missing.code, StatusCode::TASK_NOT_FOUND);
    EXPECT_NE(missing.message.find("client task not found"), std::string::npos);

    const auto taskId = SubmitTask(manager, MakeTask(AsuOpType::LOAD, 1, nullptr));
    ASSERT_TRUE(manager.Remove(taskId).ok());
    // 重复移除同一个任务必须报告未找到，而不是静默成功
    EXPECT_EQ(manager.Remove(taskId).code, StatusCode::TASK_NOT_FOUND);
    EXPECT_TRUE(manager.GetAll().empty());
}

// ======================= Process 前置校验与空任务 =======================

TEST(ClientTaskManagerTest, ProcessWithoutTransportsFailsTask)
{
    ClientTaskManager manager;
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    // 快照里有路由器但没有可用传输层
    auto task = MakeTask(AsuOpType::LOAD, 2, MakeSnapshot(router, {}));
    const auto taskId = SubmitTask(manager, task);

    EXPECT_EQ(manager.Process(task).code, StatusCode::NOT_INITIALIZED);
    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->finalStatus.code, StatusCode::NOT_INITIALIZED);
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::NOT_INITIALIZED), 2U);

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, kWaitTimeoutMs, result).code, StatusCode::NOT_INITIALIZED);
}

TEST(ClientTaskManagerTest, ProcessWithoutRouterFailsTask)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    // 快照里有传输层但缺少路由器
    auto task = MakeTask(AsuOpType::QUERY, 2,
                         MakeSnapshot(nullptr, {
                                                   {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);

    EXPECT_EQ(manager.Process(task).code, StatusCode::NOT_INITIALIZED);
    EXPECT_TRUE(task->Done());
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::NOT_INITIALIZED), 2U);
    EXPECT_TRUE(transport->submittedOpTypes.empty());

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, kWaitTimeoutMs, result).code, StatusCode::NOT_INITIALIZED);
}

TEST(ClientTaskManagerTest, EmptyItemTaskCompletesWithoutDispatch)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::LOAD, 0,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);

    // 空批次没有任何 key/entry 需要路由，任务应立即成功结束
    ASSERT_TRUE(manager.Process(task).ok());
    EXPECT_TRUE(task->Done());
    EXPECT_TRUE(task->transportTasks.empty());
    EXPECT_EQ(task->finalStatus.code, StatusCode::OK);
    EXPECT_TRUE(transport->submittedOpTypes.empty());
    EXPECT_TRUE(manager.Check(taskId));

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
    EXPECT_TRUE(result.entryStatus.empty());
    EXPECT_FALSE(result.queryResult.has_value());
}

// ============================ 路由与派发 ============================

TEST(ClientTaskManagerTest, RouteValidationRunsBeforeAnyDispatch)
{
    ClientTaskManager manager;
    auto transportA = std::make_shared<StubTransport>();
    std::unordered_map<std::string, NodeId> owners;
    owners[KeyString(0)] = kNodeA;
    owners[KeyString(1)] = kNodeB;
    auto router = std::make_shared<MapRouter>(std::move(owners), kInvalidNodeId);
    // kNodeA 的传输层存在，kNodeB 缺失
    auto task = MakeTask(AsuOpType::QUERY, 2,
                         MakeSnapshot(router, {
                                                  {kNodeA, transportA}
    }));
    const auto taskId = SubmitTask(manager, task);

    const auto status = manager.Process(task);
    EXPECT_EQ(status.code, StatusCode::NOT_FOUND);
    EXPECT_NE(status.message.find("nodeId=" + std::to_string(kNodeB)), std::string::npos);
    // 路由整体校验失败时，可用的传输层也不能收到部分数据
    EXPECT_TRUE(transportA->submittedOpTypes.empty());
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::NOT_FOUND), 2U);

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, kWaitTimeoutMs, result).code, StatusCode::NOT_FOUND);
}

TEST(ClientTaskManagerTest, BatchStoreEntriesKeepOriginalOrderPerTransport)
{
    ClientTaskManager manager;
    auto transportA = std::make_shared<StubTransport>();
    auto transportB = std::make_shared<StubTransport>();
    std::unordered_map<std::string, NodeId> owners;
    owners[KeyString(0)] = kNodeA;
    owners[KeyString(1)] = kNodeB;
    owners[KeyString(2)] = kNodeA;
    auto router = std::make_shared<MapRouter>(std::move(owners), kInvalidNodeId);
    auto task = MakeTask(AsuOpType::BATCH_STORE, 3,
                         MakeSnapshot(router, {
                                                  {kNodeA, transportA},
                                                  {kNodeB, transportB}
    }));
    const auto taskId = SubmitTask(manager, task);

    ASSERT_TRUE(manager.Process(task).ok());
    // entry 批次按 key 内容路由，且同一节点内部保持原始下标顺序
    EXPECT_EQ(transportA->submittedEntryCount, 2U);
    EXPECT_EQ(transportB->submittedEntryCount, 1U);
    EXPECT_EQ(transportA->submittedOffsets, (std::vector<std::uint32_t>{0, 2}));
    EXPECT_EQ(transportB->submittedOffsets, (std::vector<std::uint32_t>{1}));
    ASSERT_EQ(transportA->submittedKeys.size(), 2U);
    EXPECT_EQ(KeyValue(transportA->submittedKeys[0]), 0U);
    EXPECT_EQ(KeyValue(transportA->submittedKeys[1]), 2U);
    ASSERT_EQ(transportB->submittedKeys.size(), 1U);
    EXPECT_EQ(KeyValue(transportB->submittedKeys[0]), 1U);

    // 派发后客户端不再保留 entry/key，避免重复持有设备缓冲区
    EXPECT_TRUE(task->entries.empty());
    EXPECT_TRUE(task->keys.empty());
    EXPECT_EQ(task->remainingTransportSendTasks.load(), 2U);
    ASSERT_EQ(task->transportTasks.size(), 2U);
    for (const auto& transportTask : task->transportTasks) {
        EXPECT_EQ(transportTask->opType, AsuOpType::BATCH_STORE);
        EXPECT_NE(transportTask->transport.lock(), nullptr);
    }

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
    EXPECT_EQ(result.entryStatus.size(), 3U);
    EXPECT_EQ(CountEntriesWithCode(result.entryStatus, StatusCode::OK), 3U);
    EXPECT_FALSE(result.queryResult.has_value());
}

TEST(ClientTaskManagerTest, DeleteRoutesByKeyAndReportsNoQueryResult)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::DELETE, 2,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);

    ASSERT_TRUE(manager.Process(task).ok());
    EXPECT_EQ(transport->submittedOpTypes.size(), 1U);
    EXPECT_EQ(transport->submittedOpTypes.front(), AsuOpType::DELETE);
    EXPECT_EQ(transport->submittedKeyCount, 2U);
    EXPECT_TRUE(transport->submittedOffsets.empty());
    ASSERT_EQ(task->transportTasks.size(), 1U);
    EXPECT_TRUE(task->transportTasks.front()->entries.empty());
    EXPECT_EQ(task->transportTasks.front()->keys.size(), 2U);

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
    EXPECT_EQ(result.entryStatus.size(), 2U);
    // 非查询任务的 Wait 结果不应携带 query 字段
    EXPECT_FALSE(result.queryResult.has_value());
}

TEST(ClientTaskManagerTest, DispatchInstallsSendCompletionCallbacks)
{
    ClientTaskManager manager;
    auto transportA = std::make_shared<StubTransport>();
    auto transportB = std::make_shared<StubTransport>();
    std::unordered_map<std::string, NodeId> owners;
    owners[KeyString(0)] = kNodeA;
    owners[KeyString(1)] = kNodeB;
    auto router = std::make_shared<MapRouter>(std::move(owners), kInvalidNodeId);
    auto task = MakeTask(AsuOpType::QUERY, 2,
                         MakeSnapshot(router, {
                                                  {kNodeA, transportA},
                                                  {kNodeB, transportB}
    }));
    const auto taskId = SubmitTask(manager, task);

    ASSERT_TRUE(manager.Process(task).ok());
    ASSERT_EQ(task->remainingTransportSendTasks.load(), 2U);
    for (const auto& transportTask : task->transportTasks) {
        ASSERT_TRUE(transportTask->onSendComplete != nullptr);
        transportTask->onSendComplete();
    }
    // 每个传输层任务的发送完成回调只应递减一次
    EXPECT_EQ(task->remainingTransportSendTasks.load(), 0U);

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
}

// ============================ 派发失败处理 ============================

TEST(ClientTaskManagerTest, UndispatchedTransportTasksAreCanceled)
{
    auto task = MakeDirectTask(AsuOpType::LOAD, 3);
    AddTransportTask(task, {0});
    AddTransportTask(task, {1});
    AddTransportTask(task, {2});
    ASSERT_EQ(task->remainingTransportTasks.load(), 3U);

    const auto dispatchStatus = Status::Error(StatusCode::CONNECTION_ERROR, "submit failed");
    ClientTaskManager::CompleteUndispatchedTransportTasks(task, 1, dispatchStatus);

    // 首个失败任务保留派发错误，其后未派发的任务统一标记为 CANCELED
    EXPECT_EQ(task->entryStatus[0].code, StatusCode::OK);
    EXPECT_EQ(task->entryStatus[1].code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(task->entryStatus[1].message, "submit failed");
    EXPECT_EQ(task->entryStatus[2].code, StatusCode::CANCELED);
    EXPECT_EQ(task->transportTasks[1]->finalStatus.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(task->transportTasks[2]->finalStatus.message,
              "transport task not dispatched after a dispatch failure");
    EXPECT_EQ(task->remainingTransportTasks.load(), 1U);
    // 仍有在途传输层任务时任务不能提前终结
    EXPECT_FALSE(task->Done());

    TaskResult result;
    result.status = Status::OK();
    result.entryStatus = {Status::OK()};
    ClientTaskManager::CompleteTransportTask(task, 0, result);

    EXPECT_TRUE(task->Done());
    // 只要存在失败的传输层任务，整体状态即为 PARTIAL_FAILED，成功分片保持 OK
    EXPECT_EQ(task->finalStatus.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(task->entryStatus[0].code, StatusCode::OK);
}

TEST(ClientTaskManagerTest, AllTransportsRejectingSubmitMarksTaskPartialFailed)
{
    ClientTaskManager manager;
    auto transportA = std::make_shared<StubTransport>();
    auto transportB = std::make_shared<StubTransport>();
    auto transportC = std::make_shared<StubTransport>();
    const std::vector<std::shared_ptr<StubTransport>> transports = {transportA, transportB,
                                                                    transportC};
    for (const auto& transport : transports) {
        transport->submitStatus =
            Status::Error(StatusCode::CONNECTION_ERROR, "stub submit failure");
    }

    // 每个节点各持有一个 key，便于在不依赖路由顺序的前提下断言
    std::unordered_map<std::string, NodeId> owners;
    owners[KeyString(0)] = kNodeA;
    owners[KeyString(1)] = kNodeB;
    owners[KeyString(2)] = kNodeC;
    auto router = std::make_shared<MapRouter>(std::move(owners), kInvalidNodeId);
    auto task = MakeTask(
        AsuOpType::QUERY, 3,
        MakeSnapshot(router,
                     {
                         {kNodeA, transportA},
                         {kNodeB, transportB},
                         {kNodeC, transportC}
    }));
    const auto taskId = SubmitTask(manager, task);

    const auto status = manager.Process(task);
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_NE(status.message.find("nodeId="), std::string::npos);
    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->finalStatus.code, StatusCode::PARTIAL_FAILED);

    // 首个派发失败后立即停止，后续传输层不应被调用
    std::size_t attemptedTransports = 0;
    for (const auto& transport : transports) {
        attemptedTransports += transport->submittedOpTypes.empty() ? 0U : 1U;
    }
    EXPECT_EQ(attemptedTransports, 1U);

    // 失败分片保留原始错误，其余分片被取消，与路由顺序无关
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::CONNECTION_ERROR), 1U);
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::CANCELED), 2U);

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, kWaitTimeoutMs, result).code, StatusCode::PARTIAL_FAILED);
}

// ====================== 传输层完成与结果汇总 ======================

TEST(ClientTaskManagerTest, QueryResultBackfillsOriginalIndices)
{
    auto task = MakeDirectTask(AsuOpType::QUERY, 3);
    // 传输层只负责原始下标 2、0，回填必须按 originalIndices 映射而不是按顺序拼接
    AddTransportTask(task, {2, 0});

    TaskResult result;
    result.status = Status::OK();
    result.entryStatus = {Status::OK(), Status::OK()};
    QueryResult queryResult;
    queryResult.exists = {1, 0};
    result.queryResult = queryResult;
    ClientTaskManager::CompleteTransportTask(task, 0, result);

    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->remainingTransportTasks.load(), 0U);
    EXPECT_EQ(task->finalStatus.code, StatusCode::OK);
    EXPECT_EQ(task->queryResult.exists[2], 1);
    EXPECT_EQ(task->queryResult.exists[0], 0);
    // 未参与本次分片的 key 保持初始值
    EXPECT_EQ(task->queryResult.exists[1], 0);
    EXPECT_EQ(task->queryResult.prefixHitKeys, 0U);
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::OK), 3U);
}

TEST(ClientTaskManagerTest, MissingQueryResultFailsCompletion)
{
    auto task = MakeDirectTask(AsuOpType::QUERY, 2);
    AddTransportTask(task, {0, 1});

    // 查询任务成功完成却没有携带 queryResult，属于传输层契约违约
    TaskResult result;
    result.status = Status::OK();
    result.entryStatus = {Status::OK(), Status::OK()};
    ClientTaskManager::CompleteTransportTask(task, 0, result);

    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->transportTasks[0]->finalStatus.code, StatusCode::INTERNAL_ERROR);
    EXPECT_EQ(task->transportTasks[0]->finalStatus.message, "transport query result is missing");
    EXPECT_EQ(task->finalStatus.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::INTERNAL_ERROR), 2U);
    // 结果缺失时不能把 exists 误写成默认值
    EXPECT_EQ(task->queryResult.exists, (std::vector<std::uint8_t>{0, 0}));
}

TEST(ClientTaskManagerTest, QueryResultSizeMismatchFailsCompletion)
{
    auto task = MakeDirectTask(AsuOpType::QUERY, 2);
    AddTransportTask(task, {0, 1});

    TaskResult result;
    result.status = Status::OK();
    result.entryStatus = {Status::OK()};
    QueryResult queryResult;
    // 2 个 key 只返回 1 个存在性结果
    queryResult.exists = {1};
    result.queryResult = queryResult;
    ClientTaskManager::CompleteTransportTask(task, 0, result);

    EXPECT_TRUE(task->Done());
    EXPECT_EQ(task->transportTasks[0]->finalStatus.message, "transport query result size mismatch");
    EXPECT_EQ(task->finalStatus.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(CountEntriesWithCode(task->entryStatus, StatusCode::INTERNAL_ERROR), 2U);
    EXPECT_EQ(task->queryResult.exists, (std::vector<std::uint8_t>{0, 0}));
}

TEST(ClientTaskManagerTest, MissingEntryStatusFallsBackToCompletionStatus)
{
    // 传输层未逐个返回 entry 状态时，成功结果用 OK 补齐
    auto okTask = MakeDirectTask(AsuOpType::LOAD, 2);
    AddTransportTask(okTask, {0, 1});
    TaskResult okResult;
    okResult.status = Status::OK();
    ClientTaskManager::CompleteTransportTask(okTask, 0, okResult);
    EXPECT_TRUE(okTask->Done());
    EXPECT_EQ(okTask->finalStatus.code, StatusCode::OK);
    EXPECT_EQ(CountEntriesWithCode(okTask->entryStatus, StatusCode::OK), 2U);

    // 失败结果则把同一个错误状态写入所有 entry
    auto failedTask = MakeDirectTask(AsuOpType::BATCH_STORE, 2);
    AddTransportTask(failedTask, {0, 1});
    TaskResult failedResult;
    failedResult.status = Status::Error(StatusCode::IO_ERROR, "transport io failure");
    ClientTaskManager::CompleteTransportTask(failedTask, 0, failedResult);
    EXPECT_TRUE(failedTask->Done());
    EXPECT_EQ(failedTask->finalStatus.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(CountEntriesWithCode(failedTask->entryStatus, StatusCode::IO_ERROR), 2U);
    EXPECT_EQ(failedTask->entryStatus[0].message, "transport io failure");
    EXPECT_EQ(failedTask->entryStatus[1].message, "transport io failure");
}

TEST(ClientTaskManagerTest, QueryPrefixHitKeysCountsLeadingHits)
{
    const std::vector<std::vector<std::uint8_t>> existsCases = {
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
        {0, 1, 1},
        {0, 0, 0},
    };
    const std::vector<std::uint32_t> expectedPrefixHits = {3, 2, 1, 0, 0};

    for (std::size_t caseIndex = 0; caseIndex < existsCases.size(); ++caseIndex) {
        SCOPED_TRACE(caseIndex);
        ClientTaskManager manager;
        auto transport = std::make_shared<StubTransport>();
        transport->queryExists = existsCases[caseIndex];
        auto router =
            std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
        auto task = MakeTask(AsuOpType::QUERY, 3,
                             MakeSnapshot(router, {
                                                      {kNodeA, transport}
        }));
        const auto taskId = SubmitTask(manager, task);

        ASSERT_TRUE(manager.Process(task).ok());
        TaskResult result;
        ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
        ASSERT_TRUE(result.queryResult.has_value());
        EXPECT_EQ(result.queryResult->exists, existsCases[caseIndex]);
        EXPECT_EQ(result.queryResult->prefixHitKeys, expectedPrefixHits[caseIndex]);
    }
}

// ========================== Wait / Drain 边界 ==========================

TEST(ClientTaskManagerTest, WaitUnknownTaskReportsNotFound)
{
    ClientTaskManager manager;
    TaskResult result;
    const auto status = manager.Wait(4321, kShortWaitTimeoutMs, result);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
    EXPECT_NE(status.message.find("task not found"), std::string::npos);
    // 未知任务在 Check 语义下视为已完成
    EXPECT_TRUE(manager.Check(4321));
}

TEST(ClientTaskManagerTest, WaitRemovesTaskFromRegistry)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>();
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::QUERY, 2,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);
    ASSERT_TRUE(manager.Process(task).ok());

    TaskResult result;
    ASSERT_TRUE(manager.Wait(taskId, kWaitTimeoutMs, result).ok());
    EXPECT_EQ(manager.Get(taskId), nullptr);
    EXPECT_TRUE(manager.GetAll().empty());

    // 同一个任务不能被重复等待
    TaskResult second;
    EXPECT_EQ(manager.Wait(taskId, kWaitTimeoutMs, second).code, StatusCode::TASK_NOT_FOUND);
}

TEST(ClientTaskManagerTest, WaitWithZeroTimeoutReportsTimeout)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>(StubTransport::Mode::kHold);
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::LOAD, 1,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);
    ASSERT_TRUE(manager.Process(task).ok());

    TaskResult result;
    EXPECT_EQ(manager.Wait(taskId, 0, result).code, StatusCode::TIMEOUT);
    EXPECT_EQ(result.status.code, StatusCode::TIMEOUT);
    EXPECT_NE(result.status.message.find("wait_ms=0"), std::string::npos);
    EXPECT_TRUE(manager.Check(taskId));

    transport->CompleteHeld(0);
    EXPECT_TRUE(task->Done());
}

TEST(ClientTaskManagerTest, DrainWithoutTasksSucceeds)
{
    ClientTaskManager manager;
    EXPECT_TRUE(manager.Drain(0).ok());
    EXPECT_TRUE(manager.Drain(kWaitTimeoutMs).ok());
    EXPECT_TRUE(manager.GetAll().empty());
}

TEST(ClientTaskManagerTest, DrainReportsTimeoutAndRemovesEveryTask)
{
    ClientTaskManager manager;
    auto heldTransport = std::make_shared<StubTransport>(StubTransport::Mode::kHold);
    auto fastTransport = std::make_shared<StubTransport>();
    std::unordered_map<std::string, NodeId> owners;
    owners[KeyString(0)] = kNodeA;
    owners[KeyString(1)] = kNodeB;
    auto router = std::make_shared<MapRouter>(std::move(owners), kInvalidNodeId);
    auto snapshot = MakeSnapshot(router, {
                                             {kNodeA, heldTransport},
                                             {kNodeB, fastTransport}
    });

    auto heldTask = MakeTask(AsuOpType::LOAD, 1, snapshot);
    auto fastTask = MakeTask(AsuOpType::LOAD, 1, snapshot);
    // 两个任务默认都使用 key 0，这里把第二个改到 kNodeB
    ASSERT_EQ(fastTask->entries.size(), 1U);
    fastTask->entries[0].key = MakeResolvedKey(1);

    const auto heldTaskId = SubmitTask(manager, heldTask);
    const auto fastTaskId = SubmitTask(manager, fastTask);
    ASSERT_TRUE(manager.Process(heldTask).ok());
    ASSERT_TRUE(manager.Process(fastTask).ok());
    ASSERT_EQ(heldTransport->HeldCount(), 1U);
    EXPECT_TRUE(manager.Check(fastTaskId));
    EXPECT_FALSE(manager.Check(heldTaskId));

    EXPECT_EQ(manager.Drain(kShortWaitTimeoutMs).code, StatusCode::TIMEOUT);
    // 超时也必须把已完成与未完成的任务全部移除，避免注册表泄漏
    EXPECT_TRUE(manager.GetAll().empty());
    TaskResult result;
    EXPECT_EQ(manager.Wait(heldTaskId, kShortWaitTimeoutMs, result).code,
              StatusCode::TASK_NOT_FOUND);

    // 迟到的完成回调仍然安全
    heldTransport->CompleteHeld(0);
    EXPECT_TRUE(heldTask->Done());
}

// ============================== 并发语义 ==============================

TEST(ClientTaskManagerTest, WaitUnblocksWhenHeldTransportCompletes)
{
    ClientTaskManager manager;
    auto transport = std::make_shared<StubTransport>(StubTransport::Mode::kHold);
    auto router = std::make_shared<MapRouter>(std::unordered_map<std::string, NodeId>{}, kNodeA);
    auto task = MakeTask(AsuOpType::LOAD, 1,
                         MakeSnapshot(router, {
                                                  {kNodeA, transport}
    }));
    const auto taskId = SubmitTask(manager, task);
    ASSERT_TRUE(manager.Process(task).ok());
    ASSERT_EQ(transport->HeldCount(), 1U);

    std::promise<Status> waitStatus;
    auto waitFuture = waitStatus.get_future();
    std::thread waiter([&manager, taskId, &waitStatus] {
        TaskResult result;
        waitStatus.set_value(manager.Wait(taskId, kWaitTimeoutMs * 20, result));
    });

    // 完成事件无论先于还是晚于 Wait 进入等待，都必须让等待方拿到成功结果
    transport->CompleteHeld(0);
    waiter.join();

    EXPECT_TRUE(waitFuture.get().ok());
    EXPECT_TRUE(task->Done());
    EXPECT_EQ(manager.Get(taskId), nullptr);
}

TEST(ClientTaskManagerTest, ConcurrentSubmitsAssignUniqueTaskIds)
{
    ClientTaskManager manager;
    constexpr std::size_t kThreadCount = 4;
    constexpr std::size_t kTasksPerThread = 32;
    std::vector<std::vector<TaskId>> taskIds(kThreadCount,
                                             std::vector<TaskId>(kTasksPerThread, kInvalidTaskId));

    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (std::size_t threadIndex = 0; threadIndex < kThreadCount; ++threadIndex) {
        threads.emplace_back([&manager, &taskIds, threadIndex] {
            for (std::size_t index = 0; index < kTasksPerThread; ++index) {
                auto task = std::make_shared<ClientTask>();
                task->opType = AsuOpType::LOAD;
                task->entryStatus.assign(1, Status::OK());
                (void)manager.Submit(task, taskIds[threadIndex][index]);
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }

    std::set<TaskId> uniqueIds;
    for (const auto& threadIds : taskIds) {
        for (auto taskId : threadIds) {
            EXPECT_NE(taskId, kInvalidTaskId);
            uniqueIds.insert(taskId);
        }
    }
    EXPECT_EQ(uniqueIds.size(), kThreadCount * kTasksPerThread);
    EXPECT_EQ(manager.GetAll().size(), kThreadCount * kTasksPerThread);
}

}  // namespace
}  // namespace kv
