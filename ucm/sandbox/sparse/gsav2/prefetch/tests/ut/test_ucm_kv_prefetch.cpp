/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the UCMKVPrefetch class declared in include/kvcache_prefetch.h.
 *
 * After the ddr-load refactor (post 717d416) the class no longer owns a
 * background waiter thread: UCMLoadSSD now issues a synchronous UC::StoreV1::Load
 * and returns a PrefetchStatus int32_t instead of throwing / returning a task id,
 * and UCMLookup returns an empty vector on any error instead of throwing. The
 * async-wait infrastructure (mWaitQueue_/mTaskHandle/mAsyncWaiter_/UCMWait) that
 * previously leaked (bug #11: UCMLoadSSD never enqueued into mWaitQueue_) was
 * removed entirely, so the documented hang / never-processed behaviours are gone.
 */
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include "../common/mock_store_v1.h"
#include "kvcache_prefetch.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::UCMKVPrefetch;
using ucmprefetch_test::MockStoreV1;

namespace {
UC::Detail::BlockId MakeBlockId(uint8_t v)
{
    UC::Detail::BlockId id{};
    for (auto& b : id) { b = std::byte{v}; }
    return id;
}
}  // namespace

/* PrefetchStatus enum values returned as int32_t by UCMLoadSSD.
 * (The enum is private inside UCMKVPrefetch, so the int values are used.) */
namespace {
constexpr int32_t kStatusPrefetchDone = 0;  // GSA_PREFETCH_DONE
constexpr int32_t kStatusStoreEmpty = 1;    // GSA_STORE_EMPTY
constexpr int32_t kStatusBlockIdEmpty = 2;  // GSA_BLOCKID_EMPTY
constexpr int32_t kStatusHandleError = 3;   // GSA_HANDLE_ERROR
}  // namespace

TEST(UCMKVPrefetch, StoreInitRejectsNullPointer)
{
    UCMKVPrefetch p;
    EXPECT_THROW(p.StoreInit(reinterpret_cast<uintptr_t>(nullptr), /*layerNum=*/78),
                 std::runtime_error);
}

TEST(UCMKVPrefetch, StoreInitAcceptsValidPointer)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
    /* No exception. */
    SUCCEED();
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsStoreEmptyWhenStoreNull)
{
    /* Without StoreInit the store pointer is null -> GSA_STORE_EMPTY (no throw). */
    UCMKVPrefetch p;
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStatusStoreEmpty);
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsBlockIdEmptyForEmptyInput)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_EQ(p.UCMLoadSSD(empty), kStatusBlockIdEmpty);
}

TEST(UCMKVPrefetch, UCMLoadSSDSucceedsAndInvokesLoad)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/4);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1), MakeBlockId(2)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStatusPrefetchDone);
    /* UCMLoadSSD now uses UC::StoreV1::Load (sync) — exactly one Load call,
     * no LoadSSDToHost / WaitSSDToHost / ReleaseHostBuffer. */
    ASSERT_EQ(store.calls.size(), 1u);
    EXPECT_EQ(store.calls[0].name, "Load");
    /* The task is sized blockIds.size() * layerNum = 2 * 4 = 8 shards. */
    EXPECT_EQ(store.calls[0].task.size(), 8u);
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsHandleErrorOnStoreFailure)
{
    MockStoreV1 store;
    store.loadFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStatusHandleError);
    ASSERT_EQ(store.calls.size(), 1u);
    EXPECT_EQ(store.calls[0].name, "Load");
}

TEST(UCMKVPrefetch, UCMLoadSSDUsesConfiguredLayerNumForShardCount)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/3);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStatusPrefetchDone);
    ASSERT_EQ(store.calls.size(), 1u);
    EXPECT_EQ(store.calls[0].task.size(), 3u);  // 1 block * 3 layers
}

TEST(UCMKVPrefetch, UCMLoadSSDNoLongerEnqueuesAsyncWait_BugFixed)
{
    /* Previously (bug #11) UCMLoadSSD registered a handle in mTaskHandle but
     * never enqueued into mWaitQueue_, so the background AsyncWaitLoop never
     * processed the task and UCMWait blocked forever. The ddr-load refactor
     * removed the async-waiter infrastructure entirely: UCMLoadSSD now issues a
     * synchronous Load and returns immediately. Prove the fix: after UCMLoadSSD
     * the mock records exactly one Load call and no Wait/Release bookkeeping,
     * and the destructor returns promptly without joining any thread. */
    MockStoreV1 store;
    auto start = std::chrono::steady_clock::now();
    {
        UCMKVPrefetch p;
        p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
        std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
        (void)p.UCMLoadSSD(ids);
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    EXPECT_LT(elapsed.count(), 1000);
    ASSERT_EQ(store.calls.size(), 1u);
    EXPECT_EQ(store.calls[0].name, "Load");
}

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
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_TRUE(p.UCMLookup(empty).empty());
}

TEST(UCMKVPrefetch, UCMLookupReturnsExistenceVector)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
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
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
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
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), /*layerNum=*/78);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_TRUE(p.UCMLookup(ids).empty());
}
