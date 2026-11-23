/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the UCMKVPrefetch class declared in include/kvcache_prefetch.h.
 *
 * Current contract (taskid rewrite):
 *   - UCMLoadSSD submits a StoreV1::Load whose shards carry NO destination
 *     addresses (cache-warming prefetch for cache-style stores) and returns
 *     the task handle as int64_t (> 0) so the Python side can track it.
 *     Local / store rejections surface as negative UC::Status codes
 *     (Status::Underlying(): 0 = OK, -1 = Error, -50000 = InvalidParam,
 *      -50005 = NotFound, ...).  0 is never returned (invalid handle).
 *   - UCMCheck polls the store non-blockingly: 1 = finished, 0 = running,
 *     negative = Status error code.
 *   - UCMWait blocks once for the final outcome: 0 = success, negative =
 *     Status error code.  The handle is consumed by the wait.
 */
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include "../common/mock_store_v1.h"
#include "kvcache_prefetch.h"

using ucmprefetch::UCMKVPrefetch;
using ucmprefetch_test::MockStoreV1;

namespace {
UC::Detail::BlockId MakeBlockId(uint8_t v)
{
    UC::Detail::BlockId id{};
    for (auto& b : id) { b = std::byte{v}; }
    return id;
}

/* Status codes surfaced through the int64_t / int32_t returns. */
constexpr int64_t kStatusOk = 0;                      /* Status::OK            */
constexpr int64_t kStatusError = -1;                 /* Status::Error         */
constexpr int64_t kStatusInvalidParam = -50000;       /* Status::InvalidParam  */
constexpr int64_t kStatusNotFound = -50005;           /* Status::NotFound      */
}  // namespace

TEST(UCMKVPrefetch, StoreInitRejectsNullPointer)
{
    UCMKVPrefetch p;
    EXPECT_THROW(
        p.StoreInit(reinterpret_cast<uintptr_t>(nullptr), /*layerNum=*/78, /*tensorCount=*/4),
        std::runtime_error);
}

TEST(UCMKVPrefetch, StoreInitAcceptsValidPointer)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    /* No exception. */
    SUCCEED();
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsErrorWhenStoreNull)
{
    /* Without StoreInit the store pointer is null -> Status::Error code. */
    UCMKVPrefetch p;
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStatusError);
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsInvalidParamForEmptyInput)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_EQ(p.UCMLoadSSD(empty), kStatusInvalidParam);
    /* The store must not be touched for an empty request. */
    EXPECT_TRUE(store.calls.empty());
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsTaskHandleAndBuildsWarmTask)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/4, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1), MakeBlockId(2)};
    const auto ret = p.UCMLoadSSD(ids);
    /* > 0 task handle (first mock handle is 1), never a status code. */
    EXPECT_GT(ret, 0);
    ASSERT_EQ(store.calls.size(), 1u);
    EXPECT_EQ(store.calls[0].name, "Load");
    /* The task is sized blockIds.size() * layerNum = 2 * 4 = 8 shards. */
    ASSERT_EQ(store.calls[0].task.size(), 8u);
    /* Every shard is destination-less (cache-warm contract) and carries the
     * requested block ids with per-layer shard indices 0..layerNum-1. */
    for (size_t i = 0; i < store.calls[0].task.size(); ++i) {
        ASSERT_EQ(store.calls[0].task[i].addrs.size(), 4u);
        for (auto* addr : store.calls[0].task[i].addrs) { EXPECT_EQ(addr, nullptr); }
        EXPECT_EQ(store.calls[0].task[i].owner, ids[i / 4]);
        EXPECT_EQ(store.calls[0].task[i].index, i % 4);
    }
    EXPECT_EQ(store.calls[0].task.brief, "UCMLoadSSD");
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsStoreErrorOnFailure)
{
    MockStoreV1 store;
    store.loadFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStatusError);
    ASSERT_EQ(store.calls.size(), 1u);
    EXPECT_EQ(store.calls[0].name, "Load");
}

TEST(UCMKVPrefetch, UCMLoadSSDUsesConfiguredLayerNumForShardCount)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/3, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto ret = p.UCMLoadSSD(ids);
    EXPECT_GT(ret, 0);
    ASSERT_EQ(store.calls.size(), 1u);
    EXPECT_EQ(store.calls[0].task.size(), 3u);  // 1 block * 3 layers
}

TEST(UCMKVPrefetch, UCMLoadSSDRejectsStoreReturnedInvalidHandle)
{
    /* A store returning the invalid handle 0 must surface as Status::Error,
     * never as a positive task id. */
    struct ZeroHandleStore : MockStoreV1 {
        UC::Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc task) override
        {
            (void)task;
            return UC::Detail::TaskHandle{0};
        }
    } store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStatusError);
}

TEST(UCMKVPrefetch, UCMLoadSSDAndUCMWaitRoundtripWithoutBlocking)
{
    /* The historical bug #11 (UCMLoadSSD enqueued a handle nobody waited for,
     * UCMWait hung forever) must stay fixed: Load returns a usable handle
     * and Wait completes promptly on it. */
    MockStoreV1 store;
    auto start = std::chrono::steady_clock::now();
    {
        UCMKVPrefetch p;
        p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
        std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
        const auto handle = p.UCMLoadSSD(ids);
        ASSERT_GT(handle, 0);
        EXPECT_EQ(p.UCMWait(static_cast<uint64_t>(handle)), kStatusOk);
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    EXPECT_LT(elapsed.count(), 1000);
    ASSERT_EQ(store.calls.size(), 2u);
    EXPECT_EQ(store.calls[0].name, "Load");
    EXPECT_EQ(store.calls[1].name, "Wait");
    EXPECT_EQ(store.calls[1].handle, static_cast<UC::Detail::TaskHandle>(store.calls[0].handle));
}

/* ── UCMCheck ──────────────────────────────────────────────────────────── */

TEST(UCMKVPrefetch, UCMCheckReturnsDoneWhenTaskFinished)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto handle = p.UCMLoadSSD(ids);
    ASSERT_GT(handle, 0);
    /* Default mock semantics: task finished -> 1. */
    EXPECT_EQ(p.UCMCheck(static_cast<uint64_t>(handle)), 1);
    ASSERT_EQ(store.calls.size(), 2u);
    EXPECT_EQ(store.calls[1].name, "Check");
    EXPECT_EQ(store.calls[1].handle, static_cast<UC::Detail::TaskHandle>(handle));
}

TEST(UCMKVPrefetch, UCMCheckReturnsRunningWhenTaskInFlight)
{
    MockStoreV1 store;
    store.checkRunning = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto handle = p.UCMLoadSSD(ids);
    ASSERT_GT(handle, 0);
    EXPECT_EQ(p.UCMCheck(static_cast<uint64_t>(handle)), 0);
}

TEST(UCMKVPrefetch, UCMCheckReturnsErrorOnStoreFailure)
{
    MockStoreV1 store;
    store.checkFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto handle = p.UCMLoadSSD(ids);
    ASSERT_GT(handle, 0);
    EXPECT_EQ(p.UCMCheck(static_cast<uint64_t>(handle)), kStatusError);
}

TEST(UCMKVPrefetch, UCMCheckReturnsNotFoundForConsumedHandle)
{
    /* Mirrors TaskWrapper::Wait erasing the task: after the handle has been
     * consumed, Check surfaces NotFound. */
    MockStoreV1 store;
    store.checkNotFound = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto handle = p.UCMLoadSSD(ids);
    ASSERT_GT(handle, 0);
    EXPECT_EQ(p.UCMCheck(static_cast<uint64_t>(handle)), kStatusNotFound);
}

TEST(UCMKVPrefetch, UCMCheckRejectsInvalidHandle)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    EXPECT_EQ(p.UCMCheck(0), kStatusInvalidParam);
    /* The store must not be touched for an invalid handle. */
    EXPECT_TRUE(store.calls.empty());
}

TEST(UCMKVPrefetch, UCMCheckReturnsErrorWhenStoreNull)
{
    UCMKVPrefetch p;
    EXPECT_EQ(p.UCMCheck(1), kStatusError);
}

/* ── UCMWait ───────────────────────────────────────────────────────────── */

TEST(UCMKVPrefetch, UCMWaitReturnsOkOnSuccess)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto handle = p.UCMLoadSSD(ids);
    ASSERT_GT(handle, 0);
    EXPECT_EQ(p.UCMWait(static_cast<uint64_t>(handle)), kStatusOk);
}

TEST(UCMKVPrefetch, UCMWaitReturnsErrorOnStoreFailure)
{
    MockStoreV1 store;
    store.waitFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto handle = p.UCMLoadSSD(ids);
    ASSERT_GT(handle, 0);
    EXPECT_EQ(p.UCMWait(static_cast<uint64_t>(handle)), kStatusError);
}

TEST(UCMKVPrefetch, UCMWaitReturnsNotFoundForUnknownTask)
{
    MockStoreV1 store;
    store.waitNotFound = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    const auto handle = p.UCMLoadSSD(ids);
    ASSERT_GT(handle, 0);
    EXPECT_EQ(p.UCMWait(static_cast<uint64_t>(handle)), kStatusNotFound);
}

TEST(UCMKVPrefetch, UCMWaitRejectsInvalidHandle)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    EXPECT_EQ(p.UCMWait(0), kStatusInvalidParam);
    EXPECT_TRUE(store.calls.empty());
}

TEST(UCMKVPrefetch, UCMWaitReturnsErrorWhenStoreNull)
{
    UCMKVPrefetch p;
    EXPECT_EQ(p.UCMWait(1), kStatusError);
}

/* ── UCMLookup (unchanged contract) ────────────────────────────────────── */

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyWhenStoreNull)
{
    UCMKVPrefetch p;
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_TRUE(p.UCMLookup(ids).empty());
}

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyForEmptyInput)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_TRUE(p.UCMLookup(empty).empty());
}

TEST(UCMKVPrefetch, UCMLookupReturnsExistenceVector)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1), MakeBlockId(2), MakeBlockId(3)};
    auto result = p.UCMLookup(ids);
    EXPECT_EQ(result.size(), ids.size());
    for (auto v : result) { EXPECT_EQ(v, 1u); }
}

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyOnStoreFailure)
{
    MockStoreV1 store;
    store.lookupFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_TRUE(p.UCMLookup(ids).empty());
}

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyOnSizeMismatch)
{
    /* Force a mismatch by overriding Lookup to return a vector of a different
     * size. UCMLookup now returns an empty vector instead of throwing. */
    struct ShortStore : MockStoreV1 {
        UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
        {
            return std::vector<uint8_t>(num + 1, 1u);
        }
    } store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78, /*tensorCount=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_TRUE(p.UCMLookup(ids).empty());
}
