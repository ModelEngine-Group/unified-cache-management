/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the TaskManager class (include/kvcache_task_manger.h) and the
 * TaskInfo / ThreadInfo / TaskWrapper structs. Uses the ACL test stub so worker
 * threads can call ObtainStream/DestoryStream without an Ascend device.
 */
#include <atomic>
#include <chrono>
#include <functional>
#include <gtest/gtest.h>
#include <thread>
#include <torch/torch.h>
#include <vector>
#include "kvcache_task_manger.h"
#include "kvcache_trans_ascend.h"
#include "kvcache_trans_ascend_stub.h"

namespace {

torch::Tensor MakeTopkTensor() { return torch::zeros({1, 1, 4}, torch::dtype(torch::kInt32)); }

}  // namespace

class TaskManagerTest : public ::testing::Test {
protected:
    void SetUp() override { acl_stub::Stub().Reset(); }
    void TearDown() override { acl_stub::Stub().Reset(); }
};

/* ---- structs ---- */

TEST_F(TaskManagerTest, ThreadInfoDefaultConstructor)
{
    ThreadInfo t;
    EXPECT_EQ(t.threadID_, static_cast<uint64_t>(-1));
    EXPECT_EQ(t.deviceID_, -1);
    EXPECT_EQ(t.stream_, nullptr);
    EXPECT_EQ(t.context_, nullptr);
    EXPECT_EQ(t.args_, nullptr);
}

TEST_F(TaskManagerTest, ThreadInfoParameterConstructor)
{
    ThreadInfo t(1, 2, reinterpret_cast<void*>(0x10), reinterpret_cast<void*>(0x20),
                 reinterpret_cast<void*>(0x30));
    EXPECT_EQ(t.threadID_, 1u);
    EXPECT_EQ(t.deviceID_, 2);
    EXPECT_EQ(t.stream_, reinterpret_cast<void*>(0x10));
    EXPECT_EQ(t.context_, reinterpret_cast<void*>(0x20));
    EXPECT_EQ(t.args_, reinterpret_cast<void*>(0x30));
}

TEST_F(TaskManagerTest, TaskInfoDefaultConstructorLeavesIntsUninitialized_BugDocumented)
{
    // BUG-FOCUS: TaskInfo() {} (kvcache_task_manger.h) has an EMPTY body, so the
    // POD members taskID / layerID are default-initialized (indeterminate for
    // non-class scope). This is UB. Documented: verify the vectors ARE
    // default-initialized (empty) while the ints are NOT zeroed.
    // When fixed (e.g. `int taskID{0}; int layerID{0};`), flip the ints to EXPECT_EQ 0.
    TaskInfo t;
    // Vectors are always default-initialized regardless of constructor body:
    EXPECT_TRUE(t.topkLenList.empty());
    EXPECT_TRUE(t.bsInIndex.empty());
    EXPECT_TRUE(t.reqIdList.empty());
    EXPECT_TRUE(t.indexerTopkList.empty());
    // taskID / layerID are indeterminate (bug) — we do not assert their value.
    // The following lines only compile-check that the fields exist & are int.
    static_assert(std::is_same_v<decltype(t.taskID), int>, "taskID must be int");
    static_assert(std::is_same_v<decltype(t.layerID), int>, "layerID must be int");
    SUCCEED();
}

TEST_F(TaskManagerTest, TaskInfoParameterConstructor)
{
    auto tensor = MakeTopkTensor();
    std::vector<int> lens = {4};
    std::vector<int> bs = {0};
    std::vector<std::string> reqs = {"req1"};
    TaskInfo t(7, 1, tensor, lens, bs, reqs);
    EXPECT_EQ(t.taskID, 7);
    EXPECT_EQ(t.layerID, 1);
    EXPECT_EQ(t.topkLenList, lens);
    EXPECT_EQ(t.bsInIndex, bs);
    EXPECT_EQ(t.reqIdList, reqs);
}

TEST_F(TaskManagerTest, TaskWrapperConstructorFromLvalue)
{
    auto tensor = MakeTopkTensor();
    std::vector<int> lens = {4};
    std::vector<int> bs = {0};
    std::vector<std::string> reqs = {"req1"};
    TaskInfo info(1, 0, tensor, lens, bs, reqs);
    std::function<void(const TaskInfo&, ThreadInfo&)> fn = [](const TaskInfo&, ThreadInfo&) {};
    TaskWrapper w(info, fn);
    EXPECT_NE(w.inputTaskInfo, nullptr);
    EXPECT_EQ(w.inputTaskInfo->taskID, 1);
    EXPECT_EQ(w.status.load(), TaskStatus::PENDING);
}

/* ---- end-to-end worker execution ---- */

TEST_F(TaskManagerTest, SubmitTaskExecutesCallback)
{
    std::atomic<int> executed{0};
    {
        TaskManager mgr(/*deviceID=*/0, /*dpRank=*/0, /*tpSize=*/1,
                        /*args=*/nullptr, /*threadNum=*/1, /*streamNum=*/1,
                        /*type=*/"kv", /*cpuToBind=*/{});
        auto tensor = MakeTopkTensor();
        std::vector<int> lens = {4};
        std::vector<int> bs = {0};
        std::vector<std::string> reqs = {"req1"};
        TaskInfo info(1, 0, tensor, lens, bs, reqs);
        mgr.SubmitTask(info, [&executed](const TaskInfo&, ThreadInfo&) {
            executed.fetch_add(1, std::memory_order_relaxed);
        });
        /* Wait for the worker to process the single task. */
        for (int i = 0; i < 200 && executed.load() == 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        EXPECT_EQ(executed.load(), 1);
    }
    EXPECT_EQ(acl_stub::Stub().destroyStreamCount.load(), 1);
}

TEST_F(TaskManagerTest, SubmitTaskSurvivesCallbackException)
{
    std::atomic<int> executed{0};
    {
        TaskManager mgr(0, 0, 1, nullptr, 1, 1, "kv", {});
        auto tensor = MakeTopkTensor();
        std::vector<int> lens = {4};
        std::vector<int> bs = {0};
        std::vector<std::string> reqs = {"req1"};
        TaskInfo info(1, 0, tensor, lens, bs, reqs);
        mgr.SubmitTask(info,
                       [](const TaskInfo&, ThreadInfo&) { throw std::runtime_error("boom"); });
        for (int i = 0; i < 200 && executed.load() == 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        /* The exception is caught inside ExecuteTask; worker must survive. */
        EXPECT_EQ(executed.load(), 0);
        /* Submit a second task to confirm the pool still works. */
        TaskInfo info2(2, 0, tensor, lens, bs, reqs);
        mgr.SubmitTask(info2, [&executed](const TaskInfo&, ThreadInfo&) {
            executed.fetch_add(1, std::memory_order_relaxed);
        });
        for (int i = 0; i < 200 && executed.load() == 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        EXPECT_EQ(executed.load(), 1);
    }
}

TEST_F(TaskManagerTest, DestructorJoinsAllWorkersWithoutHanging)
{
    /* Construct a pool with several workers; nothing submitted. */
    TaskManager mgr(0, 0, 1, nullptr, 3, 1, "kv", {});
    /* Destruct here; must not hang. */
    SUCCEED();
}
