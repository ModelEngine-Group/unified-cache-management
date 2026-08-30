/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit the persons to whom the Software is
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
 */
#include "task_manager_base_slot.h"
#include <atomic>
#include <gtest/gtest.h>
#include <mutex>
#include <set>
#include <thread>
#include <vector>
#include "client_task_manager.h"
#include "task_manager_base.h"
#include "transport_task_manager.h"

namespace UC::ASU {
namespace {

// ==================== Transport TaskManager Tests ====================

class TransportTaskManagerTest : public ::testing::Test {
protected:
    TransportTaskManager manager_;
};

// Submit with nullptr: verify defensive check
// Expected: Submit returns error, task_id remains kInvalidTaskId
TEST_F(TransportTaskManagerTest, SubmitNullContext_ReturnsError)
{
    TaskId task_id{kInvalidTaskId};
    auto status = manager_.Submit(nullptr, task_id);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(task_id, kInvalidTaskId);
}

// Get with invalid TaskId: query kInvalidTaskId and non-existent large ID
// Expected: Get returns nullptr
TEST_F(TransportTaskManagerTest, GetInvalidTaskId_ReturnsNull)
{
    auto retrieved = manager_.Get(kInvalidTaskId);
    EXPECT_EQ(retrieved, nullptr);

    retrieved = manager_.Get(99999);
    EXPECT_EQ(retrieved, nullptr);
}

// Remove with invalid TaskId: remove kInvalidTaskId and non-existent large ID
// Expected: Remove returns error
TEST_F(TransportTaskManagerTest, RemoveInvalidTaskId_ReturnsError)
{
    auto status = manager_.Remove(kInvalidTaskId);
    EXPECT_FALSE(status.ok());

    status = manager_.Remove(99999);
    EXPECT_FALSE(status.ok());
}

// Submit 100 tasks consecutively, verify all TaskIds are unique with no duplicates
// Expected: All 100 TaskIds are unique, set size = 100
TEST_F(TransportTaskManagerTest, SubmitMultipleTasks_TaskIdUnique)
{
    std::set<TaskId> task_ids;
    constexpr std::size_t kCount = 100;

    for (std::size_t i = 0; i < kCount; ++i) {
        auto ctx = std::make_unique<TransportTaskContext>();
        ctx->opType = static_cast<TransportOpType>(i % 4);

        TaskId task_id{kInvalidTaskId};
        auto status = manager_.Submit(std::move(ctx), task_id);
        ASSERT_TRUE(status.ok());

        EXPECT_EQ(task_ids.count(task_id), 0) << "TaskId " << task_id << " is not unique!";
        task_ids.insert(task_id);
    }

    EXPECT_EQ(task_ids.size(), kCount);
}

// Submit→Remove→Submit: submit new task after removing old task, verify slot reuse is correct
// Expected: new task_id ≠ old task_id, new task can be retrieved with Get
TEST_F(TransportTaskManagerTest, SubmitRemoveSubmit_TaskIdReuse)
{
    auto ctx1 = std::make_unique<TransportTaskContext>();
    TaskId task_id1{kInvalidTaskId};
    manager_.Submit(std::move(ctx1), task_id1);

    manager_.Remove(task_id1);

    auto ctx2 = std::make_unique<TransportTaskContext>();
    TaskId task_id2{kInvalidTaskId};
    manager_.Submit(std::move(ctx2), task_id2);

    EXPECT_NE(task_id1, task_id2);
    auto retrieved = manager_.Get(task_id2);
    EXPECT_NE(retrieved, nullptr);
}

// 16 threads concurrently submit 1600 tasks, verify no duplicate TaskIds under concurrency
// (fetch_add uniqueness) Expected: All 1600 TaskIds are unique, success_count = 1600
TEST_F(TransportTaskManagerTest, MultiThreadSubmit_TaskIdUnique)
{
    constexpr std::size_t kThreadNum = 16;
    constexpr std::size_t kOpsPerThread = 100;

    std::atomic<std::size_t> success_count{0};
    std::set<TaskId> all_task_ids;
    std::mutex ids_mutex;

    std::vector<std::thread> threads;
    threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                auto ctx = std::make_unique<TransportTaskContext>();

                TaskId task_id{kInvalidTaskId};
                auto status = manager_.Submit(std::move(ctx), task_id);

                if (status.ok()) {
                    success_count.fetch_add(1, std::memory_order_relaxed);

                    std::lock_guard<std::mutex> lock(ids_mutex);
                    EXPECT_EQ(all_task_ids.count(task_id), 0)
                        << "TaskId " << task_id << " is not unique!";
                    all_task_ids.insert(task_id);
                }
            }
        });
    }

    for (auto& th : threads) { th.join(); }

    EXPECT_EQ(success_count.load(), kThreadNum * kOpsPerThread);
    EXPECT_EQ(all_task_ids.size(), kThreadNum * kOpsPerThread);
}

// 16 threads concurrently submit 800 tasks, then 16 threads concurrently Get, verify Get
// correctness Expected: All submitted tasks can be retrieved with Get, get_success = 800, get_fail
// = 0
TEST_F(TransportTaskManagerTest, MultiThreadSubmitAndGet)
{
    constexpr std::size_t kThreadNum = 16;
    constexpr std::size_t kOpsPerThread = 50;

    std::vector<TaskId> submitted_ids;
    std::mutex ids_mutex;

    std::vector<std::thread> submit_threads;
    submit_threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        submit_threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                auto ctx = std::make_unique<TransportTaskContext>();

                TaskId task_id{kInvalidTaskId};
                auto status = manager_.Submit(std::move(ctx), task_id);
                if (status.ok()) {
                    std::lock_guard<std::mutex> lock(ids_mutex);
                    submitted_ids.push_back(task_id);
                }
            }
        });
    }

    for (auto& th : submit_threads) { th.join(); }

    std::atomic<std::size_t> get_success{0};
    std::atomic<std::size_t> get_fail{0};

    std::vector<std::thread> get_threads;
    get_threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        get_threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                std::lock_guard<std::mutex> lock(ids_mutex);
                if (submitted_ids.empty()) continue;
                TaskId task_id = submitted_ids.back();
                submitted_ids.pop_back();

                auto ctx = manager_.Get(task_id);
                if (ctx) {
                    get_success.fetch_add(1, std::memory_order_relaxed);
                } else {
                    get_fail.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : get_threads) { th.join(); }

    EXPECT_EQ(get_success.load(), kThreadNum * kOpsPerThread);
    EXPECT_EQ(get_fail.load(), 0);
}

// 16 threads concurrently execute Submit+Get+Remove mixed operations, verify no crash or race
// condition errors Expected: submit/get/remove counts all > 0, submit ≤ kTotalOps
TEST_F(TransportTaskManagerTest, MultiThreadSubmitGetRemove)
{
    constexpr std::size_t kThreadNum = 16;
    constexpr std::size_t kOpsPerThread = 30;
    constexpr std::size_t kTotalOps = kThreadNum * kOpsPerThread;

    std::atomic<std::size_t> submit_count{0};
    std::atomic<std::size_t> get_count{0};
    std::atomic<std::size_t> remove_count{0};
    std::atomic<std::size_t> last_task_id{kInvalidTaskId};

    std::vector<std::thread> threads;
    threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                auto ctx = std::make_unique<TransportTaskContext>();

                TaskId task_id{kInvalidTaskId};
                auto status = manager_.Submit(std::move(ctx), task_id);
                if (status.ok()) {
                    submit_count.fetch_add(1, std::memory_order_relaxed);
                    last_task_id.store(task_id, std::memory_order_relaxed);
                }

                TaskId check_id = last_task_id.load(std::memory_order_relaxed);
                if (check_id != kInvalidTaskId) {
                    auto retrieved = manager_.Get(check_id);
                    if (retrieved) { get_count.fetch_add(1, std::memory_order_relaxed); }

                    status = manager_.Remove(check_id);
                    if (status.ok()) { remove_count.fetch_add(1, std::memory_order_relaxed); }
                }
            }
        });
    }

    for (auto& th : threads) { th.join(); }

    EXPECT_GT(submit_count.load(), 0);
    EXPECT_GT(get_count.load(), 0);
    EXPECT_GT(remove_count.load(), 0);
    EXPECT_LE(submit_count.load(), kTotalOps);
}

// RecommendSlotCount returns power of 2 for different inputs, and >= kMinSlotCount=1024
// Expected: 1→1024, 100→1024, 1000→2048, 4096→8192
TEST_F(TransportTaskManagerTest, NormalizeSlotCount_AlwaysPowerOfTwo)
{
    using SlotManager = SlotTaskManagerBase<TransportTaskContext, TransportTaskState>;
    EXPECT_EQ(SlotManager::RecommendSlotCount(1), 1024);
    EXPECT_EQ(SlotManager::RecommendSlotCount(100), 1024);
    EXPECT_EQ(SlotManager::RecommendSlotCount(1000), 2048);
    EXPECT_EQ(SlotManager::RecommendSlotCount(4096), 8192);
}

// Double-check verification: after deleting task and submitting new task, Get with old task_id
// won't return stale data Expected: Get with old task_id returns nullptr (won't return new task's
// Context)
TEST_F(TransportTaskManagerTest, DoubleCheckInGet_PreventsStaleReturn)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = TransportOpType::LOAD;

    TaskId task_id{kInvalidTaskId};
    manager_.Submit(std::move(ctx), task_id);

    auto retrieved = manager_.Get(task_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->opType, TransportOpType::LOAD);

    manager_.Remove(task_id);

    auto new_ctx = std::make_unique<TransportTaskContext>();
    new_ctx->opType = TransportOpType::DELETE;
    TaskId new_task_id{kInvalidTaskId};
    manager_.Submit(std::move(new_ctx), new_task_id);

    auto stale = manager_.Get(task_id);
    EXPECT_EQ(stale, nullptr);
}

// Context state transition: PENDING→INFLIGHT→COMPLETED, verify atomic state read/write and Done()
// judgment Expected: initial PENDING, manually set INFLIGHT/COMPLETED, Done() returns true when
// COMPLETED
TEST_F(TransportTaskManagerTest, ContextStateTransition)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = TransportOpType::QUERY;

    TaskId task_id{kInvalidTaskId};
    manager_.Submit(std::move(ctx), task_id);

    auto retrieved = manager_.Get(task_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->state.load(), TransportTaskState::PENDING);

    retrieved->state.store(TransportTaskState::INFLIGHT, std::memory_order_release);
    EXPECT_EQ(retrieved->state.load(), TransportTaskState::INFLIGHT);

    retrieved->state.store(TransportTaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(retrieved->Done());
}

// GetAll() on empty manager: no tasks submitted, GetAll returns empty vector
TEST_F(TransportTaskManagerTest, GetAllEmpty_ReturnsEmpty)
{
    auto all = manager_.GetAll();
    EXPECT_TRUE(all.empty());
}

// GetAll() basic: submit 50 tasks, GetAll returns exactly 50 contexts with correct taskId
TEST_F(TransportTaskManagerTest, GetAllBasic_ReturnsAllTasks)
{
    std::set<TaskId> submitted_ids;
    constexpr std::size_t kCount = 50;

    for (std::size_t i = 0; i < kCount; ++i) {
        auto ctx = std::make_unique<TransportTaskContext>();
        ctx->opType = static_cast<TransportOpType>(i % 7);

        TaskId task_id{kInvalidTaskId};
        ASSERT_TRUE(manager_.Submit(std::move(ctx), task_id).ok());
        submitted_ids.insert(task_id);
    }

    auto all = manager_.GetAll();
    EXPECT_EQ(all.size(), kCount);

    for (const auto& ctx : all) {
        EXPECT_NE(ctx->taskId, kInvalidTaskId);
        EXPECT_TRUE(submitted_ids.count(ctx->taskId) > 0);
    }
}

// GetAll() after Remove: submit 30 tasks, remove 10, GetAll only returns the remaining 20
TEST_F(TransportTaskManagerTest, GetAllAfterRemove_ExcludesRemoved)
{
    std::vector<TaskId> all_ids;
    constexpr std::size_t kCount = 30;
    constexpr std::size_t kRemoveCount = 10;

    for (std::size_t i = 0; i < kCount; ++i) {
        auto ctx = std::make_unique<TransportTaskContext>();
        TaskId task_id{kInvalidTaskId};
        ASSERT_TRUE(manager_.Submit(std::move(ctx), task_id).ok());
        all_ids.push_back(task_id);
    }

    for (std::size_t i = 0; i < kRemoveCount; ++i) {
        ASSERT_TRUE(manager_.Remove(all_ids[i]).ok());
    }

    auto remaining = manager_.GetAll();
    EXPECT_EQ(remaining.size(), kCount - kRemoveCount);

    std::set<TaskId> remaining_ids;
    for (const auto& ctx : remaining) { remaining_ids.insert(ctx->taskId); }

    for (std::size_t i = kRemoveCount; i < kCount; ++i) {
        EXPECT_TRUE(remaining_ids.count(all_ids[i]) > 0);
    }
    for (std::size_t i = 0; i < kRemoveCount; ++i) {
        EXPECT_EQ(remaining_ids.count(all_ids[i]), 0);
    }
}

// GetAll() multi-thread: 8 threads concurrently submit tasks, then single-thread GetAll,
// verify all submitted tasks are present
TEST_F(TransportTaskManagerTest, MultiThreadSubmitAndGetAll)
{
    constexpr std::size_t kThreadNum = 8;
    constexpr std::size_t kOpsPerThread = 50;

    std::vector<TaskId> submitted_ids;
    std::mutex ids_mutex;

    std::vector<std::thread> threads;
    threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                auto ctx = std::make_unique<TransportTaskContext>();
                TaskId task_id{kInvalidTaskId};
                if (manager_.Submit(std::move(ctx), task_id).ok()) {
                    std::lock_guard<std::mutex> lock(ids_mutex);
                    submitted_ids.push_back(task_id);
                }
            }
        });
    }

    for (auto& th : threads) { th.join(); }

    auto all = manager_.GetAll();
    EXPECT_EQ(all.size(), submitted_ids.size());

    std::set<TaskId> all_ids;
    for (const auto& ctx : all) { all_ids.insert(ctx->taskId); }
    for (const auto& id : submitted_ids) {
        EXPECT_TRUE(all_ids.count(id) > 0) << "TaskId " << id << " not found in GetAll()";
    }
}

// Slot reuse produces different TaskId: submit→remove→submit on same slot,
// generation increments so new TaskId differs from old
TEST_F(TransportTaskManagerTest, SlotReuse_GenerationIncrements)
{
    std::vector<TaskId> ids;
    constexpr std::size_t kRounds = 20;

    for (std::size_t i = 0; i < kRounds; ++i) {
        auto ctx = std::make_unique<TransportTaskContext>();
        TaskId task_id{kInvalidTaskId};
        ASSERT_TRUE(manager_.Submit(std::move(ctx), task_id).ok());
        ids.push_back(task_id);

        ASSERT_TRUE(manager_.Remove(task_id).ok());
    }

    for (std::size_t i = 1; i < ids.size(); ++i) { EXPECT_NE(ids[i], ids[i - 1]); }
}

TEST_F(TransportTaskManagerTest, SlotFull_ReturnsError)
{
    using SmallSlotManager = SlotTaskManagerBase<TransportTaskContext, TransportTaskState>;
    SmallSlotManager small_manager(TransportTaskState::PENDING, "small_transport", 1024);

    const auto slotCount = SmallSlotManager::RecommendSlotCount(1024);

    std::atomic<std::size_t> success_count{0};
    std::atomic<std::size_t> fail_count{0};

    constexpr std::size_t kTryCount = 9000;

    for (std::size_t i = 0; i < kTryCount; ++i) {
        auto ctx = std::make_unique<TransportTaskContext>();
        TaskId task_id{kInvalidTaskId};
        auto status = small_manager.Submit(std::move(ctx), task_id);

        if (status.ok()) {
            success_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            fail_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    EXPECT_GT(success_count.load(), 0);
    EXPECT_GT(fail_count.load(), 0);
    EXPECT_EQ(success_count.load() + fail_count.load(), kTryCount);
    EXPECT_LE(success_count.load(), slotCount);
}

// ==================== Client TaskManager Tests ====================

class ClientTaskManagerTest : public ::testing::Test {
protected:
    ClientTaskManager manager_;
};

TEST_F(TransportTaskManagerTest, RemoveSuccess_ReturnsOK)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    TaskId task_id{kInvalidTaskId};
    ASSERT_TRUE(manager_.Submit(std::move(ctx), task_id).ok());

    auto status = manager_.Remove(task_id);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(manager_.Get(task_id), nullptr);
}

// 32 threads concurrently execute Client Submit+Get+Remove, verify no crash or race condition
// errors Expected: submit/get/remove counts all > 0, submit ≤ kTotalOps
TEST_F(ClientTaskManagerTest, MultiThreadSubmitGetRemove)
{
    constexpr std::size_t kThreadNum = 32;
    constexpr std::size_t kOpsPerThread = 1000;
    constexpr std::size_t kTotalOps = kThreadNum * kOpsPerThread;

    std::atomic<std::size_t> submit_count{0};
    std::atomic<std::size_t> get_count{0};
    std::atomic<std::size_t> remove_count{0};
    std::atomic<std::size_t> last_task_id{kInvalidTaskId};

    std::vector<std::thread> threads;
    threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                auto ctx = std::make_unique<ClientTaskContext>();

                TaskId task_id{kInvalidTaskId};
                auto status = manager_.Submit(std::move(ctx), task_id);
                if (status.ok()) {
                    submit_count.fetch_add(1, std::memory_order_relaxed);
                    last_task_id.store(task_id, std::memory_order_relaxed);
                }

                TaskId check_id = last_task_id.load(std::memory_order_relaxed);
                if (check_id != kInvalidTaskId) {
                    auto retrieved = manager_.Get(check_id);
                    if (retrieved) { get_count.fetch_add(1, std::memory_order_relaxed); }

                    status = manager_.Remove(check_id);
                    if (status.ok()) { remove_count.fetch_add(1, std::memory_order_relaxed); }
                }
            }
        });
    }

    for (auto& th : threads) { th.join(); }

    EXPECT_GT(submit_count.load(), 0);
    EXPECT_GT(get_count.load(), 0);
    EXPECT_GT(remove_count.load(), 0);
    EXPECT_LE(submit_count.load(), kTotalOps);
}

// GetAll() on empty ClientTaskManager: returns empty vector
TEST_F(ClientTaskManagerTest, GetAllEmpty_ReturnsEmpty)
{
    auto all = manager_.GetAll();
    EXPECT_TRUE(all.empty());
}

// GetAll() basic: submit 50 client tasks, GetAll returns exactly 50 contexts
TEST_F(ClientTaskManagerTest, GetAllBasic_ReturnsAllTasks)
{
    std::set<TaskId> submitted_ids;
    constexpr std::size_t kCount = 50;

    for (std::size_t i = 0; i < kCount; ++i) {
        auto ctx = std::make_unique<ClientTaskContext>();
        ctx->opType = static_cast<ClientOpType>(i % 3);

        TaskId task_id{kInvalidTaskId};
        ASSERT_TRUE(manager_.Submit(std::move(ctx), task_id).ok());
        submitted_ids.insert(task_id);
    }

    auto all = manager_.GetAll();
    EXPECT_EQ(all.size(), kCount);

    for (const auto& ctx : all) {
        EXPECT_NE(ctx->taskId, kInvalidTaskId);
        EXPECT_TRUE(submitted_ids.count(ctx->taskId) > 0);
    }
}

// GetAll() after Remove: submit 30 tasks, remove 10, GetAll only returns the remaining 20
TEST_F(ClientTaskManagerTest, GetAllAfterRemove_ExcludesRemoved)
{
    std::vector<TaskId> all_ids;
    constexpr std::size_t kCount = 30;
    constexpr std::size_t kRemoveCount = 10;

    for (std::size_t i = 0; i < kCount; ++i) {
        auto ctx = std::make_unique<ClientTaskContext>();
        TaskId task_id{kInvalidTaskId};
        ASSERT_TRUE(manager_.Submit(std::move(ctx), task_id).ok());
        all_ids.push_back(task_id);
    }

    for (std::size_t i = 0; i < kRemoveCount; ++i) {
        ASSERT_TRUE(manager_.Remove(all_ids[i]).ok());
    }

    auto remaining = manager_.GetAll();
    EXPECT_EQ(remaining.size(), kCount - kRemoveCount);

    std::set<TaskId> remaining_ids;
    for (const auto& ctx : remaining) { remaining_ids.insert(ctx->taskId); }

    for (std::size_t i = kRemoveCount; i < kCount; ++i) {
        EXPECT_TRUE(remaining_ids.count(all_ids[i]) > 0);
    }
    for (std::size_t i = 0; i < kRemoveCount; ++i) {
        EXPECT_EQ(remaining_ids.count(all_ids[i]), 0);
    }
}

// GetAll() concurrent with submit and remove: verify GetAll never returns deleted or stale contexts
TEST_F(ClientTaskManagerTest, MultiThreadSubmitRemoveAndGetAll)
{
    constexpr std::size_t kThreadNum = 8;
    constexpr std::size_t kOpsPerThread = 100;

    std::atomic<std::size_t> submit_count{0};
    std::atomic<std::size_t> remove_count{0};
    std::vector<TaskId> submitted_ids;
    std::mutex ids_mutex;

    std::vector<std::thread> submit_threads;
    submit_threads.reserve(kThreadNum);

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        submit_threads.emplace_back([&, tid]() {
            for (std::size_t op = 0; op < kOpsPerThread; ++op) {
                auto ctx = std::make_unique<ClientTaskContext>();
                TaskId task_id{kInvalidTaskId};
                if (manager_.Submit(std::move(ctx), task_id).ok()) {
                    submit_count.fetch_add(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(ids_mutex);
                    submitted_ids.push_back(task_id);
                }
            }
        });
    }

    for (auto& th : submit_threads) { th.join(); }

    std::vector<std::thread> remove_threads;
    remove_threads.reserve(kThreadNum);

    constexpr std::size_t kRemovePerThread = 50;
    std::vector<TaskId> ids_to_remove;
    {
        std::lock_guard<std::mutex> lock(ids_mutex);
        for (std::size_t i = 0; i < kThreadNum * kRemovePerThread && i < submitted_ids.size();
             ++i) {
            ids_to_remove.push_back(submitted_ids[i]);
        }
    }

    for (std::size_t tid = 0; tid < kThreadNum; ++tid) {
        remove_threads.emplace_back([&, tid]() {
            for (std::size_t op = 0;
                 op < kRemovePerThread && tid * kRemovePerThread + op < ids_to_remove.size();
                 ++op) {
                auto idx = tid * kRemovePerThread + op;
                if (manager_.Remove(ids_to_remove[idx]).ok()) {
                    remove_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : remove_threads) { th.join(); }

    auto all = manager_.GetAll();
    EXPECT_EQ(all.size(), submit_count.load() - remove_count.load());

    for (const auto& ctx : all) {
        EXPECT_NE(ctx->taskId, kInvalidTaskId);
        auto retrieved = manager_.Get(ctx->taskId);
        EXPECT_NE(retrieved, nullptr) << "GetAll returned stale context for TaskId " << ctx->taskId;
    }
}

}  // namespace
}  // namespace UC::ASU