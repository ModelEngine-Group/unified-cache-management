/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * DT (end-to-end): GSAPrefetchEngineC block-map lifecycle pipeline.
 *
 * Drives the engine through the public block-map lifecycle:
 *   SetBlocksMap -> SetTokensMap-skip -> DelBlocksMap -> GetReqUseBlockTable
 * and verifies the observable outputs are consistent:
 *   - the block table stored by SetBlocksMap is retrievable
 *   - DelBlocksMap clears the per-request state
 *   - the Get* accessors return the stored values
 *
 * All ACL H2D / scatter ops and the UCM store are stubbed so the lifecycle
 * runs deterministically on a plain host without spawning worker threads or
 * creating real shared memory (the full SubmitTask path is exercised via the
 * TaskManager UT, which is sufficient coverage for the thread-pool leg).
 */
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::BlockId;
using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

BlockId MakeBlock(uint8_t v)
{
    BlockId b{};
    for (auto& x : b) { x = std::byte{v}; }
    return b;
}

struct E2EFixture {
    static constexpr int kLayerNum = 1;
    static constexpr int kSparseLen = 4;
    static constexpr int kSpaTopkLen = 4;
    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;

    void Setup()
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        auto model = MakeModelInfo(kLayerNum, /*maxBlockLen=*/1024, kSpaTopkLen,
                                   /*streamNum=*/1, kSparseLen, 1, /*dpRank=*/dpRank,
                                   /*tpSize=*/1, /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/1);
        auto size = MakeSizeInfo(/*blockSize=*/2, /*numHead=*/1, /*kDim=*/2,
                                 /*vDim=*/2, /*elementSize=*/2);
        std::vector<int> topkLayerMap(kLayerNum);
        for (int i = 0; i < kLayerNum; ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());
    }
    void TearDown()
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }
};

}  // namespace

TEST(PrefetchE2EDt, SetBlocksMapThenGetRoundTripsStoredTable)
{
    E2EFixture f;
    f.Setup();
    std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
    std::vector<int> blockIdxs = {0, 1};
    std::vector<uintptr_t> addrs = {0x10, 0x20};
    f.engine->SetBlocksMap("req1", hashes, /*maxIdx=*/2, blockIdxs, addrs, addrs);
    EXPECT_TRUE(f.engine->mAllBlocksHash.count("req1"));
    EXPECT_EQ(f.engine->mPromptLen["req1"], 2);
    f.TearDown();
}

TEST(PrefetchE2EDt, DelBlocksMapClearsSparseSeqLenButNotHash_BugDocumented)
{
    // BUG-FOCUS: DelReqIDRun (kvcache_pre_scatter.cpp) erases every per-request
    // map EXCEPT mAllBlocksHash. This means stale block hashes leak across
    // requests after DelBlocksMap. Documented here so a future fix updates the
    // source AND this assertion.
    E2EFixture f;
    f.Setup();
    std::vector<BlockId> hashes = {MakeBlock(1)};
    std::vector<int> idxs = {0};
    std::vector<uintptr_t> addrs = {0x10};
    f.engine->SetBlocksMap("req1", hashes, 1, idxs, addrs, addrs);
    f.engine->mSparseSeqLen["req1"] = 4;
    f.engine->DelBlocksMap("req1");
    // mSparseSeqLen IS cleared by DelReqIDRun:
    EXPECT_FALSE(f.engine->mSparseSeqLen.count("req1"));
    // mAllBlocksHash is NOT cleared (bug). When fixed, flip to EXPECT_FALSE.
    EXPECT_TRUE(f.engine->mAllBlocksHash.count("req1"));
    // Idempotent: deleting again is a no-op (reqID already removed from set).
    f.engine->DelBlocksMap("req1");
    EXPECT_FALSE(f.engine->mSparseSeqLen.count("req1"));
    f.TearDown();
}

TEST(PrefetchE2EDt, DelBlocksMapQueuesDeletionWhenPrefetchRunning)
{
    E2EFixture f;
    f.Setup();
    f.engine->mIsPrefetchRunning.store(true);
    f.engine->mSparseSeqLen["req1"] = 4;
    f.engine->DelBlocksMap("req1");
    EXPECT_TRUE(f.engine->mDelSeqIds.count("req1"));
    EXPECT_TRUE(f.engine->mSparseSeqLen.count("req1"));
    f.TearDown();
}

TEST(PrefetchE2EDt, StoreV1RawAccessorsRoundTrip)
{
    // Note: E2EFixture::Setup() already calls SetStoreV1(store.get()), so
    // mStoreV1_ is NOT nullptr after Setup — it points at the mock store.
    E2EFixture f;
    f.Setup();
    EXPECT_EQ(f.engine->GetStoreV1Raw(), reinterpret_cast<uintptr_t>(f.store.get()));
    // Reset to nullptr and verify the setter works.
    f.engine->SetStoreV1(nullptr);
    EXPECT_EQ(f.engine->GetStoreV1Raw(), reinterpret_cast<uintptr_t>(nullptr));
    EXPECT_EQ(f.engine->GetSelfRaw(), reinterpret_cast<uintptr_t>(f.engine.get()));
    f.TearDown();
}

TEST(PrefetchE2EDt, SetLastIndexerBlockNoopWhenOffloadDisabled)
{
    E2EFixture f;
    f.Setup();
    std::vector<int> idxs = {0};
    std::vector<uintptr_t> addrs = {0x10};
    f.engine->SetLastIndexerBlock("req1", idxs, addrs);
    // No crash; indexer offload is off, nothing stored.
    EXPECT_FALSE(f.engine->mLastKVBlockAddress.count("req1"));
    f.TearDown();
}

TEST(PrefetchE2EDt, MultipleRequestsCoexist_HashLeakBugDocumented)
{
    // BUG-FOCUS: DelReqIDRun does not erase mAllBlocksHash, so after deleting
    // one request the hash map still reports the deleted request. Documented
    // here; when fixed, expect size 2 after the delete.
    E2EFixture f;
    f.Setup();
    for (int r = 0; r < 3; ++r) {
        std::string reqID = "req" + std::to_string(r);
        std::vector<BlockId> hashes = {MakeBlock(static_cast<uint8_t>(r + 1))};
        std::vector<int> idxs = {0};
        std::vector<uintptr_t> addrs = {static_cast<uintptr_t>(0x10 + r)};
        f.engine->SetBlocksMap(reqID, hashes, 1, idxs, addrs, addrs);
        f.engine->mSparseSeqLen[reqID] = 4;
    }
    EXPECT_EQ(f.engine->mAllBlocksHash.size(), 3u);
    EXPECT_EQ(f.engine->mSparseSeqLen.size(), 3u);
    // Delete one; mSparseSeqLen drops (DelReqIDRun clears it) but
    // mAllBlocksHash keeps the leaked entry (bug). When fixed: expect 2u.
    f.engine->DelBlocksMap("req1");
    EXPECT_EQ(f.engine->mSparseSeqLen.size(), 2u);
    EXPECT_EQ(f.engine->mAllBlocksHash.size(), 3u);  // leaked hash for "req1"
    EXPECT_TRUE(f.engine->mSparseSeqLen.count("req2"));
    EXPECT_TRUE(f.engine->mSparseSeqLen.count("req0"));
    f.TearDown();
}
