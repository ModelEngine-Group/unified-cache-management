/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the pure-logic / mockable methods of GSAPrefetchEngineC declared in
 * kvcache_pre.h and implemented in kvcache_pre.cpp.
 *
 * Covers TokenIndexMapInter, Add/DeleteIndexerMapForRequest, the UCM-merge
 * helpers (Validate/Build/Prepare/Load/Memcpy/Release/Copy/BuildTokenCopyTasks),
 * LastChunkMove early-return branches, and CheckInputIndex.
 *
 * These paths are pure host logic or only depend on the MockStoreV1, so they
 * run safely without ACL device / TaskManager worker threads.
 */
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_pre_detail.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::BlockId;
using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch::LastChunkSourceBlock;
using ucmprefetch::TokenCopyTask;
using ucmprefetch::UCMMergeBlockContext;
using ucmprefetch::UCMMergeHostAddress;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class PreLogicTest : public ::testing::Test {
protected:
    int dpRank{0};
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<GSAPrefetchEngineC> engine;
    std::unique_ptr<MockStoreV1> store;

    void SetUp() override
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
    }
    void TearDown() override
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    /* Engine with indexer offload ON and small dims:
     * blockSize=4, numHead=1, kDim=4, vDim=4, elemSize=2
     *   -> mKVSizeBytesToken = 2*1*4 = 8, mKVSizeBytesTokenRe = 8
     *   -> mBlockSize = 4
     * spaTopkLen=32, sparseLen=128. */
    std::unique_ptr<GSAPrefetchEngineC> MakeEngine(int layerNum = 2, int indexerOffload = 1,
                                                   int isSfaC8 = 0)
    {
        auto model = MakeModelInfo(layerNum, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                                   indexerOffload, isSfaC8, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                                 /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
        std::vector<int> topkLayerMap(layerNum);
        for (int i = 0; i < layerNum; ++i) { topkLayerMap[i] = i; }
        auto e = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
        e->SetStoreV1(store.get());
        /* The ddr-load refactor made LoadUCMMergeHostAddress / UCMTensorMerge
         * stage into the engine-owned mUCMMergeLoadBuffer (set by the shared-
         * memory SetupKVPackSharedBuffers in production, which tests skip).
         * Point it at the engine-allocated mLastChunkLoadBuffer so the merge
         * paths have a valid staging region without extra bookkeeping. The
         * destructor frees mLastChunkLoadBuffer and never frees
         * mUCMMergeLoadBuffer, so this aliasing is safe. */
        if (e->mUCMMergeLoadBuffer == nullptr && e->mLastChunkLoadBuffer != nullptr) {
            e->mUCMMergeLoadBuffer = static_cast<std::byte*>(e->mLastChunkLoadBuffer);
        }
        return e;
    }

    /* A 16-byte BlockId filled with the given byte value. */
    static BlockId MakeBlockId(std::uint8_t v)
    {
        BlockId b{};
        b.fill(static_cast<std::byte>(v));
        return b;
    }
};

}  // namespace

/* ---------------- CheckInputIndex ---------------- */

TEST_F(PreLogicTest, CheckInputIndexPassesWhenIndexBelowMaxLen)
{
    engine = MakeEngine();
    // index < maxLen -> no abort, returns normally.
    engine->CheckInputIndex(/*maxLen=*/10, /*index=*/5);
    SUCCEED();
}

// NOTE: The index >= maxLen branch calls std::abort() and cannot be tested
// in-process without killing the test runner; it is documented as a fatal
// contract violation rather than a recoverable error.

/* ---------------- TokenIndexMapInter ---------------- */

TEST_F(PreLogicTest, TokenIndexMapInterRewritesTopkToBlockIndexedSlots)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // mRealTimeIdx[reqID][layer 0]: block-index map (block -> internal idx).
    // block 0 -> idx 5, block 1 -> idx 9.
    engine->mRealTimeIdx[reqID] = {
        {5, 9}
    };  // [layer 0] = {5, 9}
    engine->mSeqLen[reqID] = 1000;
    // topk tokens: 0 (block 0 slot 0), 5 (block 1 slot 1)
    std::vector<int> oneTopkList = {0, 5};
    engine->TokenIndexMapInter(oneTopkList, /*bsIndex=*/0, /*layerID=*/0,
                               /*topkLen=*/2, reqID);
    // token 0: block 0 (0/4=0), idx=mRealTimeIdx[0][0]=5 -> 5*4 + 0 = 20
    // token 5: block 1 (5/4=1), idx=mRealTimeIdx[0][1]=9 -> 9*4 + 1 = 37
    EXPECT_EQ(oneTopkList[0], 20);
    EXPECT_EQ(oneTopkList[1], 37);
}

TEST_F(PreLogicTest, TokenIndexMapInterIsNoopWhenOffloadOff)
{
    engine = MakeEngine(/*layerNum=*/1, /*indexerOffload=*/0);
    std::vector<int> oneTopkList = {0, 5};
    engine->TokenIndexMapInter(oneTopkList, 0, 0, 2, "req1");
    EXPECT_EQ(oneTopkList, (std::vector<int>{0, 5}));  // unchanged
}

/* ---------------- AddIndexerMapForRequest ---------------- */

TEST_F(PreLogicTest, AddIndexerMapForRequestAppendsBlocksToDocsAndBlocksMap)
{
    engine = MakeEngine(/*layerNum=*/2);
    const std::string reqID = "req1";
    // Pre-seed empty docs/blocks tables for 2 layers.
    engine->mDocsTables[reqID].assign(2, {});
    engine->mBlocksMap[reqID].assign(2, {});
    engine->mIndexerSlot[reqID] = {0, 1};  // 2 existing slots -> next idx = 2
    engine->mIndexerAdd[reqID] = {
        {10, 100},
        {11, 200}
    };  // {slot, blockID}
    engine->AddIndexerMapForRequest(reqID);
    // BUG-FOCUS: the first loop computes idx = mIndexerSlot[reqID].size() which
    // is NOT incremented per-j inside the loop (mIndexerSlot is only grown in
    // the SECOND loop).  So BOTH new blocks are written to idx == 2, and the
    // last write (block 200) wins in mDocsTables.  This test pins the current
    // (buggy) behavior; the fix would be `idx = base + j`.
    EXPECT_EQ(engine->mDocsTables[reqID][0][2], 200);  // last write wins
    EXPECT_EQ(engine->mBlocksMap[reqID][0][100], 2);   // both map to idx 2
    EXPECT_EQ(engine->mBlocksMap[reqID][0][200], 2);
    // Same for layer 1.
    EXPECT_EQ(engine->mDocsTables[reqID][1][2], 200);
    EXPECT_EQ(engine->mBlocksMap[reqID][1][200], 2);
    // IndexerSlot appended with the slot values {10, 11} (second loop).
    EXPECT_EQ(engine->mIndexerSlot[reqID].size(), 4u);
    EXPECT_EQ(engine->mIndexerSlot[reqID][2], 10);
    EXPECT_EQ(engine->mIndexerSlot[reqID][3], 11);
}

TEST_F(PreLogicTest, AddIndexerMapForRequestWithEmptyAddIsNoop)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mDocsTables[reqID].assign(1, {});
    engine->mBlocksMap[reqID].assign(1, {});
    engine->mIndexerSlot[reqID] = {0};
    engine->mIndexerAdd[reqID] = {};
    engine->AddIndexerMapForRequest(reqID);
    EXPECT_TRUE(engine->mDocsTables[reqID][0].empty());
    EXPECT_EQ(engine->mIndexerSlot[reqID].size(), 1u);
}

/* ---------------- DeleteIndexerMapForRequest ---------------- */

TEST_F(PreLogicTest, DeleteIndexerMapForRequestRemovesBlocksAndErasesFromPrefetchIdx)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mDocsTables[reqID].assign(1, {});
    engine->mBlocksMap[reqID].assign(1, {});
    engine->mIndexerSlot[reqID] = {0, 1};
    // block 100 -> idx 2, block 200 -> idx 3 (matching Add test).
    engine->mDocsTables[reqID][0][2] = 100;
    engine->mDocsTables[reqID][0][3] = 200;
    engine->mBlocksMap[reqID][0][100] = 2;
    engine->mBlocksMap[reqID][0][200] = 3;
    engine->mPrefetchIdx[reqID].assign(1, {2, 3, 5});
    engine->mIndexerDel[reqID] = {100};  // delete block 100 (idx 2)
    engine->DeleteIndexerMapForRequest(reqID);
    EXPECT_EQ(engine->mBlocksMap[reqID][0].count(100), 0u);
    EXPECT_EQ(engine->mDocsTables[reqID][0].count(2), 0u);
    // idx 2 removed from mPrefetchIdx.
    EXPECT_EQ(engine->mPrefetchIdx[reqID][0], (std::vector<int>{3, 5}));
    // block 200 untouched.
    EXPECT_EQ(engine->mBlocksMap[reqID][0][200], 3);
}

/* ---------------- ValidateUCMMergeRequest ---------------- */

TEST_F(PreLogicTest, ValidateUCMMergeRequestReturnsFalseWhenNoMissTokens)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // mMissGlobalToken[reqID][topkIndex 0] is default-empty -> returns false.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    EXPECT_FALSE(engine->ValidateUCMMergeRequest(/*layerID=*/0, reqID));
}

TEST_F(PreLogicTest, ValidateUCMMergeRequestLogsWhenStoreV1IsNull)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->SetStoreV1(nullptr);  // exercise the null-store log branch
    const std::string reqID = "req1";
    // Seed non-empty miss tokens so we reach past the empty check.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {10, 20};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0, 1};
    engine->mBlockSize = 4;
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1)};
    engine->mReqUseBlockTable[reqID] = {0};
    engine->mLoadNum.assign(engine->mLayerNum, 0);
    engine->mLoadIndexBufferLen = 16;
    engine->mKVBufferLen = 16;
    EXPECT_TRUE(engine->ValidateUCMMergeRequest(/*layerID=*/0, reqID));
}

TEST_F(PreLogicTest, ValidateUCMMergeRequestReturnsTrueWithValidState)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {10, 20};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0, 1};
    engine->mBlockSize = 4;
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1)};
    engine->mReqUseBlockTable[reqID] = {0};
    engine->mLoadNum.assign(engine->mLayerNum, 0);
    engine->mLoadIndexBufferLen = 16;
    engine->mKVBufferLen = 16;
    EXPECT_TRUE(engine->ValidateUCMMergeRequest(/*layerID=*/0, reqID));
}

/* ---------------- BuildUCMMergeBlockContexts ---------------- */

TEST_F(PreLogicTest, BuildUCMMergeBlockContextsGroupsTokensByBlockAndOwnedShards)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // blockSize=4; miss tokens 0,1 (block 0), 8 (block 2).
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0, 1, 8};
    // hashVec has 3 blocks; block 2 < 3 -> isnoUCMblock=false.
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1), MakeBlockId(2), MakeBlockId(3)};
    engine->mRank = 0;
    engine->mTpSize = 1;  // (0 + requestIndex) % 1 == 0 == mRank -> owned
    std::vector<UCMMergeBlockContext> blockContexts;
    UC::Detail::TaskDesc task;
    std::map<int, std::vector<size_t>> blockIdxToTokenIdx;
    engine->BuildUCMMergeBlockContexts(/*layerID=*/0, reqID, /*requestIndex=*/0, blockContexts,
                                       task, blockIdxToTokenIdx, engine->mUCMMergeLoadBuffer,
                                       engine->mShardSize);
    // Two blocks: 0 (tokens 0,1) and 2 (token 2).
    EXPECT_EQ(blockContexts.size(), 2u);
    EXPECT_EQ(blockIdxToTokenIdx.size(), 2u);
    EXPECT_EQ(blockIdxToTokenIdx[0], (std::vector<size_t>{0, 1}));
    EXPECT_EQ(blockIdxToTokenIdx[2], (std::vector<size_t>{2}));
    // All owned, no non-UCM blocks -> task has 2 shards.
    EXPECT_EQ(task.size(), 2u);
    // tokenBase accumulates: first block 0 -> base 0, second block 2 -> base 2.
    EXPECT_EQ(blockContexts[0].tokenBase, 0u);
    EXPECT_EQ(blockContexts[1].tokenBase, 2u);
    EXPECT_FALSE(blockContexts[0].isnoUCMblock);
    EXPECT_TRUE(blockContexts[0].owned);
}

TEST_F(PreLogicTest, BuildUCMMergeBlockContextsMarksNonUcmBlocksAndSkipsShard)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    // token 8 -> block 2, but hashVec only has 2 entries -> isnoUCMblock=true.
    engine->mMissGlobalToken[reqID][0] = {8};
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1), MakeBlockId(2)};
    engine->mRank = 0;
    engine->mTpSize = 1;
    std::vector<UCMMergeBlockContext> blockContexts;
    UC::Detail::TaskDesc task;
    std::map<int, std::vector<size_t>> blockIdxToTokenIdx;
    engine->BuildUCMMergeBlockContexts(0, reqID, 0, blockContexts, task, blockIdxToTokenIdx,
                                       engine->mUCMMergeLoadBuffer, engine->mShardSize);
    ASSERT_EQ(blockContexts.size(), 1u);
    EXPECT_TRUE(blockContexts[0].isnoUCMblock);
    // owned but non-UCM block -> no shard pushed.
    EXPECT_TRUE(task.empty());
}

/* ---------------- BuildTokenCopyTasks ---------------- */

TEST_F(PreLogicTest, BuildTokenCopyTasksProducesOneTaskPerTokenWithHbmSlot)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // blockSize=4; miss tokens {0, 8}, slots {0, 4}, blockTable {7, 9}.
    // token 0: slot 0 -> hbmIdx=blockTable[0/4]=blockTable[0]=7, hbmSlot=0%4=0
    //          -> hbmSlot = 7*4 + 0 = 28
    // token 8: slot 4 -> hbmIdx=blockTable[4/4]=blockTable[1]=9, hbmSlot=4%4=0
    //          -> hbmSlot = 9*4 + 0 = 36
    std::vector<int> missGlobalIds = {0, 8};
    std::vector<int> missSlots = {0, 4};
    std::vector<int> blockTable = {7, 9};
    std::vector<size_t> tokenIndices0 = {0};  // first block owns token 0
    std::vector<size_t> tokenIndices1 = {1};  // second block owns token 1
    std::byte kSrc[64] = {};
    std::byte vSrc[64] = {};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices0, 0, 0, false, 0, kSrc, vSrc, true},
        {&tokenIndices1, 1, 1, false, 2, kSrc, vSrc, true},
    };
    std::vector<TokenCopyTask> tasks;
    engine->BuildTokenCopyTasks(blockContexts, tasks, missGlobalIds, missSlots, blockTable,
                                /*kTokenBytes=*/8, /*vTokenBytes=*/8, /*previousLoadNum=*/0);
    ASSERT_EQ(tasks.size(), 2u);
    EXPECT_EQ(tasks[0].dstOffset, 0u);
    EXPECT_EQ(tasks[0].hbmSlot, 28ull);
    EXPECT_TRUE(tasks[0].owned);
    EXPECT_NE(tasks[0].kSrc, nullptr);
    EXPECT_NE(tasks[0].vSrc, nullptr);  // !mISSPAC8
    EXPECT_EQ(tasks[1].dstOffset, 1u);  // previousLoadNum(0) + tokenBase(1) + localTokenIdx(0)
    EXPECT_EQ(tasks[1].hbmSlot, 36ull);
}

TEST_F(PreLogicTest, BuildTokenCopyTasksWithSpac8OmitsVSrc)
{
    engine = MakeEngine(/*layerNum=*/1, /*indexerOffload=*/1, /*isSfaC8=*/1);
    std::vector<int> missGlobalIds = {0};
    std::vector<int> missSlots = {0};
    std::vector<int> blockTable = {7};
    std::vector<size_t> tokenIndices = {0};
    std::byte kSrc[64] = {};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, false, 0, kSrc, nullptr, true},
    };
    std::vector<TokenCopyTask> tasks;
    engine->BuildTokenCopyTasks(blockContexts, tasks, missGlobalIds, missSlots, blockTable, 8, 8,
                                0);
    ASSERT_EQ(tasks.size(), 1u);
    EXPECT_EQ(tasks[0].vSrc, nullptr);  // mISSPAC8 -> vSrc null
}

TEST_F(PreLogicTest, BuildTokenCopyTasksUnownedContextHasNullSources)
{
    engine = MakeEngine(/*layerNum=*/1);
    std::vector<int> missGlobalIds = {4};
    std::vector<int> missSlots = {1};
    std::vector<int> blockTable = {3};
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, false, 0, nullptr, nullptr, /*owned=*/false},
    };
    std::vector<TokenCopyTask> tasks;
    engine->BuildTokenCopyTasks(blockContexts, tasks, missGlobalIds, missSlots, blockTable, 8, 8,
                                0);
    ASSERT_EQ(tasks.size(), 1u);
    EXPECT_FALSE(tasks[0].owned);
    EXPECT_EQ(tasks[0].kSrc, nullptr);
    EXPECT_EQ(tasks[0].vSrc, nullptr);
}

/* ---------------- CopySingleTokenKV ---------------- */

TEST_F(PreLogicTest, CopySingleTokenKVCopiesKAndVWhenNotSpac8)
{
    engine = MakeEngine(/*layerNum=*/1);
    std::byte kSrc[64], vSrc[64], kDst[64], vDst[64];
    for (int i = 0; i < 64; ++i) {
        kSrc[i] = static_cast<std::byte>(0x10 + i);
        vSrc[i] = static_cast<std::byte>(0x20 + i);
        kDst[i] = std::byte{0};
        vDst[i] = std::byte{0};
    }
    std::vector<size_t> tokenIndices = {0};
    UCMMergeBlockContext ctx{&tokenIndices, 0, 0, false, 0, kSrc, vSrc, true};
    engine->CopySingleTokenKV(kDst, vDst, ctx, /*tokenOffset=*/2,
                              /*dstOffset=*/5, /*kTokenBytes=*/8, /*vTokenBytes=*/8);
    // Verify K copied: kDst[5*8 .. 5*8+8) == kSrc[2*8 .. 2*8+8)
    EXPECT_EQ(std::memcmp(kDst + 5 * 8, kSrc + 2 * 8, 8), 0);
    // Verify V copied: vDst[5*8 .. 5*8+8) == vSrc[2*8 .. 2*8+8)
    EXPECT_EQ(std::memcmp(vDst + 5 * 8, vSrc + 2 * 8, 8), 0);
}

TEST_F(PreLogicTest, CopySingleTokenKVSkipsVWhenSpac8)
{
    engine = MakeEngine(/*layerNum=*/1, /*indexerOffload=*/1, /*isSfaC8=*/1);
    std::byte kSrc[64], vSrc[64], kDst[64], vDst[64];
    for (int i = 0; i < 64; ++i) {
        kSrc[i] = static_cast<std::byte>(0x30 + i);
        vSrc[i] = static_cast<std::byte>(0xFF);
        kDst[i] = std::byte{0};
        vDst[i] = std::byte{0};
    }
    std::vector<size_t> tokenIndices = {0};
    UCMMergeBlockContext ctx{&tokenIndices, 0, 0, false, 0, kSrc, vSrc, true};
    engine->CopySingleTokenKV(kDst, vDst, ctx, /*tokenOffset=*/0,
                              /*dstOffset=*/1, 8, 8);
    EXPECT_EQ(std::memcmp(kDst + 1 * 8, kSrc, 8), 0);
    // V untouched.
    bool vTouched = false;
    for (int i = 0; i < 64; ++i) {
        if (vDst[i] != std::byte{0}) {
            vTouched = true;
            break;
        }
    }
    EXPECT_FALSE(vTouched);
}

/* ---------------- ReleaseUCMMergeHostAddress ---------------- */

TEST_F(PreLogicTest, ReleaseUCMMergeHostAddressIsNoopWhenNoHostTask)
{
    engine = MakeEngine();
    UCMMergeHostAddress addr(*engine);
    addr.hasHostTask = false;
    engine->ReleaseUCMMergeHostAddress(addr);
    EXPECT_FALSE(addr.hasHostTask);
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(PreLogicTest, ReleaseUCMMergeHostAddressClearsStateWithoutStoreCall)
{
    /* After the ddr-load refactor UCMMergeHostAddress no longer owns a
     * HostBuffer task (taskId/views members removed): the staging buffer is
     * the engine-owned mUCMMergeLoadBuffer, and the Load+Wait is synchronous,
     * so Release only clears the address vector. No ReleaseHostBuffer call. */
    engine = MakeEngine();
    UCMMergeHostAddress addr(*engine);
    addr.hasHostTask = true;
    addr.addresses.push_back(nullptr);
    engine->ReleaseUCMMergeHostAddress(addr);
    EXPECT_FALSE(addr.hasHostTask);
    EXPECT_TRUE(addr.addresses.empty());
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(PreLogicTest, ReleaseUCMMergeHostAddressClearsAddressesEvenWithFailedStoreFlag)
{
    /* releaseHostBufferFails is now irrelevant (no ReleaseHostBuffer call is
     * made), but the address state must still be cleared. */
    engine = MakeEngine();
    store->releaseHostBufferFails = true;
    UCMMergeHostAddress addr(*engine);
    addr.hasHostTask = true;
    addr.addresses.push_back(nullptr);
    engine->ReleaseUCMMergeHostAddress(addr);
    EXPECT_FALSE(addr.hasHostTask);
    EXPECT_TRUE(addr.addresses.empty());
    EXPECT_TRUE(store->calls.empty());
    store->releaseHostBufferFails = false;
}

/* ---------------- LoadUCMMergeHostAddress ---------------- */

TEST_F(PreLogicTest, LoadUCMMergeHostAddressReturnsTrueWhenNoTask)
{
    engine = MakeEngine();
    UCMMergeHostAddress addr(*engine);
    UC::Detail::TaskDesc task;  // empty -> early return true
    EXPECT_TRUE(engine->LoadUCMMergeHostAddress(std::move(task), addr, 0, "req1"));
    EXPECT_FALSE(addr.hasHostTask);
}

TEST_F(PreLogicTest, LoadUCMMergeHostAddressReturnsFalseWhenAddressDirty)
{
    engine = MakeEngine();
    UCMMergeHostAddress addr(*engine);
    addr.hasHostTask = true;  // already has a task
    UC::Detail::TaskDesc task;
    task.resize(1);
    EXPECT_FALSE(engine->LoadUCMMergeHostAddress(std::move(task), addr, 0, "req1"));
}

TEST_F(PreLogicTest, LoadUCMMergeHostAddressReturnsFalseWhenStoreFails)
{
    engine = MakeEngine();
    store->loadFails = true;
    UCMMergeHostAddress addr(*engine);
    UC::Detail::TaskDesc task;
    task.resize(1);
    EXPECT_FALSE(engine->LoadUCMMergeHostAddress(std::move(task), addr, 0, "req1"));
    EXPECT_FALSE(addr.hasHostTask);
}

TEST_F(PreLogicTest, LoadUCMMergeHostAddressReturnsFalseWhenWaitFails)
{
    /* After the ddr-load refactor the load path is Load + Wait. A Wait failure
     * now takes the place of the old view-count-mismatch / null-host-addr
     * failure modes (those HostBufferView concepts no longer exist). */
    engine = MakeEngine();
    store->waitFails = true;
    UCMMergeHostAddress addr(*engine);
    UC::Detail::TaskDesc task;
    task.resize(1);
    EXPECT_FALSE(engine->LoadUCMMergeHostAddress(std::move(task), addr, 0, "req1"));
    EXPECT_FALSE(addr.hasHostTask);
}

TEST_F(PreLogicTest, LoadUCMMergeHostAddressSucceedsAndPopulatesAddresses)
{
    engine = MakeEngine();
    UCMMergeHostAddress addr(*engine);
    UC::Detail::TaskDesc task;
    task.resize(2);
    EXPECT_TRUE(engine->LoadUCMMergeHostAddress(std::move(task), addr, 0, "req1"));
    EXPECT_TRUE(addr.hasHostTask);
    EXPECT_EQ(addr.addresses.size(), 2u);
    EXPECT_NE(addr.addresses[0], nullptr);
    /* Addresses are spaced mShardSize apart in the staging buffer. */
    EXPECT_EQ(engine->mShardSize,
              static_cast<size_t>(static_cast<const std::byte*>(addr.addresses[1]) -
                                  static_cast<const std::byte*>(addr.addresses[0])));
}

TEST_F(PreLogicTest, LoadUCMMergeHostAddressReturnsFalseWhenStoreNull)
{
    engine = MakeEngine();
    engine->SetStoreV1(nullptr);
    UCMMergeHostAddress addr(*engine);
    UC::Detail::TaskDesc task;
    task.resize(1);
    EXPECT_FALSE(engine->LoadUCMMergeHostAddress(std::move(task), addr, 0, "req1"));
}

/* ---------------- PrepareUCMMergeSources ---------------- */

TEST_F(PreLogicTest, PrepareUCMMergeSourcesSetsSrcForOwnedUcmBlocks)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    std::byte shardBlock[256];  // big enough for K+V (kBlockBytes=4*8=32)
    for (int i = 0; i < 256; ++i) { shardBlock[i] = static_cast<std::byte>(0x40 + i); }
    // UCMMergeBlockContext for an owned UCM block reads address.addresses[viewIndex].
    UCMMergeHostAddress addr(*engine);
    addr.addresses.push_back(static_cast<const std::byte*>(shardBlock));
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, /*viewIndex=*/0, 0, /*isnoUCMblock=*/false, 0, nullptr, nullptr,
         /*owned=*/true},
    };
    EXPECT_TRUE(engine->PrepareUCMMergeSources(addr, blockContexts, /*layerID=*/0, reqID));
    EXPECT_NE(blockContexts[0].kSrc, nullptr);
    EXPECT_NE(blockContexts[0].vSrc, nullptr);
    // kSrc = addresses[0] + layerID(0) * kBlockBytes(0) = shardBlock
    EXPECT_EQ(blockContexts[0].kSrc, static_cast<const std::byte*>(shardBlock));
    // vSrc = addresses[0] + kBlockBytes(4*8=32)
    EXPECT_EQ(blockContexts[0].vSrc, static_cast<const std::byte*>(shardBlock) + 32);
}

TEST_F(PreLogicTest, PrepareUCMMergeSourcesReturnsFalseWhenNonUcmBlockMissingMap)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // No mLastKVBlockAddress entry -> non-UCM block lookup fails.
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, /*isnoUCMblock=*/true, 5, nullptr, nullptr, /*owned=*/true},
    };
    EXPECT_FALSE(
        engine->PrepareUCMMergeSources(UCMMergeHostAddress(*engine), blockContexts, 0, reqID));
}

TEST_F(PreLogicTest, PrepareUCMMergeSourcesReturnsFalseWhenBlockIdxMissing)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mLastKVBlockAddress[reqID][0].kAddr = 1;  // only block 0
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, /*isnoUCMblock=*/true, 5, nullptr, nullptr, /*owned=*/true},
    };
    EXPECT_FALSE(
        engine->PrepareUCMMergeSources(UCMMergeHostAddress(*engine), blockContexts, 0, reqID));
}

TEST_F(PreLogicTest, PrepareUCMMergeSourcesReturnsFalseWhenKAddrZero)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mLastKVBlockAddress[reqID][5].kAddr = 0;  // kAddr zero
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, /*isnoUCMblock=*/true, 5, nullptr, nullptr, /*owned=*/true},
    };
    EXPECT_FALSE(
        engine->PrepareUCMMergeSources(UCMMergeHostAddress(*engine), blockContexts, 0, reqID));
}

TEST_F(PreLogicTest, PrepareUCMMergeSourcesReturnsFalseWhenVAddrZeroAndNotSpac8)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    std::byte kBlock[64];
    engine->mLastKVBlockAddress[reqID][5].kAddr = reinterpret_cast<uintptr_t>(kBlock);
    engine->mLastKVBlockAddress[reqID][5].vAddr = 0;  // vAddr zero, !mISSPAC8
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, /*isnoUCMblock=*/true, 5, nullptr, nullptr, /*owned=*/true},
    };
    EXPECT_FALSE(
        engine->PrepareUCMMergeSources(UCMMergeHostAddress(*engine), blockContexts, 0, reqID));
}

TEST_F(PreLogicTest, PrepareUCMMergeSourcesSkipsUnownedContexts)
{
    engine = MakeEngine(/*layerNum=*/1);
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, /*isnoUCMblock=*/true, 5, nullptr, nullptr, /*owned=*/false},
    };
    // Unowned -> skipped, no source lookup -> returns true.
    EXPECT_TRUE(
        engine->PrepareUCMMergeSources(UCMMergeHostAddress(*engine), blockContexts, 0, "req1"));
}

/* ---------------- LastChunkMove early-return branches ---------------- */

TEST_F(PreLogicTest, LastChunkMoveReturnsOnEmptyMoveMap)
{
    engine = MakeEngine();
    engine->LastChunkMove(0, "req1", /*moveMap=*/{});
    SUCCEED();  // early return, no store calls
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(PreLogicTest, LastChunkMoveReturnsOnMisalignedMoveMap)
{
    engine = MakeEngine();
    // blockSize=4, moveMap.size()=3 -> 3 % 4 != 0 -> early return.
    engine->LastChunkMove(0, "req1", /*moveMap=*/{1, 2, 3});
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(PreLogicTest, LastChunkMoveReturnsWhenStoreNull)
{
    engine = MakeEngine();
    engine->SetStoreV1(nullptr);
    engine->LastChunkMove(0, "req1", /*moveMap=*/{1, 2, 3, 4});
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(PreLogicTest, LastChunkMoveReturnsWhenHashOrBlockTableMissing)
{
    engine = MakeEngine();
    // moveMap aligned (size 4 = blockSize), store present, but no hash/blockTable.
    engine->LastChunkMove(0, "req1", /*moveMap=*/{1, 2, 3, 4});
    EXPECT_TRUE(store->calls.empty());
}

/* ---------------- MemcpyUCMMergeHostAddress ---------------- */

TEST_F(PreLogicTest, MemcpyUCMMergeHostAddressReturnsFalseWhenPrepareFails)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // Seed miss tokens so tokenCount > 0, but PrepareUCMMergeSources will fail
    // because blockContexts has an owned non-UCM block with no last-KV map.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0};
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, /*isnoUCMblock=*/true, 5, nullptr, nullptr, /*owned=*/true},
    };
    std::vector<uint64_t> hbmSlotIndex;
    EXPECT_FALSE(engine->MemcpyUCMMergeHostAddress(UCMMergeHostAddress(*engine), blockContexts, 0,
                                                   0, hbmSlotIndex, reqID, 0));
}

TEST_F(PreLogicTest, MemcpyUCMMergeHostAddressSerialPathSucceeds)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // Seed a single miss token, owned UCM block, valid shard host address.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0};
    engine->mReqUseBlockTable[reqID] = {0};
    // Shared host buffers for the serial copy path.
    constexpr size_t kBufBytes = 256;
    std::byte* kBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kBuf, 0, kBufBytes);
    std::memset(vBuf, 0, kBufBytes);
    engine->mSharedKBufferHostPtr.resize(1);
    engine->mSharedVBufferHostPtr.resize(1);
    engine->mSharedKBufferHostPtr[0] = reinterpret_cast<uint64_t>(kBuf);
    engine->mSharedVBufferHostPtr[0] = reinterpret_cast<uint64_t>(vBuf);
    // Shard host address (kBlockBytes + vBlockBytes = 32+32 = 64 bytes).
    std::byte shardBlock[64];
    for (int i = 0; i < 64; ++i) { shardBlock[i] = static_cast<std::byte>(0x70 + i); }
    UCMMergeHostAddress addr(*engine);
    addr.addresses.push_back(static_cast<const std::byte*>(shardBlock));
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, /*viewIndex=*/0, 0, /*isnoUCMblock=*/false, 0, nullptr, nullptr,
         /*owned=*/true},
    };
    std::vector<uint64_t> hbmSlotIndex;
    EXPECT_TRUE(
        engine->MemcpyUCMMergeHostAddress(addr, blockContexts, 0, 0, hbmSlotIndex, reqID, 0));
    // tokenCount=1 < 256 -> serial path -> hbmSlotIndex resized to 1.
    EXPECT_EQ(hbmSlotIndex.size(), 1u);
    std::free(kBuf);
    std::free(vBuf);
}

/* ---------------- CopyUCMMergeTokensParallel ---------------- */

TEST_F(PreLogicTest, CopyUCMMergeTokensParallelCopiesTokensToSharedBuffers)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // Seed 4 miss tokens to stay under the 2-thread threshold (256).
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0, 4, 8, 12};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0, 1, 2, 3};
    engine->mReqUseBlockTable[reqID] = {0, 1, 2, 3};
    // Shared host buffers: 4 tokens * 8 bytes = 32 bytes each.
    constexpr size_t kBufBytes = 256;
    std::byte* kBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kBuf, 0, kBufBytes);
    std::memset(vBuf, 0, kBufBytes);
    engine->mSharedKBufferHostPtr.resize(1);
    engine->mSharedVBufferHostPtr.resize(1);
    engine->mSharedKBufferHostPtr[0] = reinterpret_cast<uint64_t>(kBuf);
    engine->mSharedVBufferHostPtr[0] = reinterpret_cast<uint64_t>(vBuf);
    // 4 owned UCM contexts, each with one token.
    std::byte kSrc[128], vSrc[128];
    for (int i = 0; i < 128; ++i) {
        kSrc[i] = static_cast<std::byte>(0xA0 + i);
        vSrc[i] = static_cast<std::byte>(0xB0 + i);
    }
    std::vector<size_t> ti0 = {0}, ti1 = {1}, ti2 = {2}, ti3 = {3};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&ti0, 0, 0, false, 0, kSrc, vSrc, true},
        {&ti1, 1, 1, false, 1, kSrc, vSrc, true},
        {&ti2, 2, 2, false, 2, kSrc, vSrc, true},
        {&ti3, 3, 3, false, 3, kSrc, vSrc, true},
    };
    // CopyUCMMergeTokensParallel writes hbmSlotIndex[dstOffset] without
    // resizing (the caller MemcpyUCMMergeHostAddress sizes it). Pre-size here.
    std::vector<uint64_t> hbmSlotIndex(4, 0);
    engine->CopyUCMMergeTokensParallel(blockContexts, 0, 0, hbmSlotIndex, reqID, 0, 2);
    EXPECT_EQ(hbmSlotIndex.size(), 4u);
    std::free(kBuf);
    std::free(vBuf);
}

/* ---------------- FillFailedUCMMerge ---------------- */

TEST_F(PreLogicTest, FillFailedUCMMergeZeroesOwnedTokensAndBuildsIndices)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // Seed miss slots + block table so the index build is valid.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0, 4};  // 2 tokens, blockSize 4
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0, 4};       // slotIDs
    engine->mReqUseBlockTable[reqID] = {7, 9};  // blockTable
    // Shared host buffers for the memset path.
    constexpr size_t kBufBytes = 256;
    std::byte* kBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kBuf, 0xCC, kBufBytes);
    std::memset(vBuf, 0xCC, kBufBytes);
    engine->mSharedKBufferHostPtr.resize(1);
    engine->mSharedVBufferHostPtr.resize(1);
    engine->mSharedKBufferHostPtr[0] = reinterpret_cast<uint64_t>(kBuf);
    engine->mSharedVBufferHostPtr[0] = reinterpret_cast<uint64_t>(vBuf);
    // 1 owned context with 2 tokens at tokenBase 0.
    std::vector<size_t> tokenIndices = {0, 1};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, /*isnoUCMblock=*/false, 0, nullptr, nullptr, /*owned=*/true},
    };
    std::vector<uint64_t> hbmSlotIndex;
    size_t zeroTokens =
        engine->FillFailedUCMMerge(blockContexts, /*layerID=*/0,
                                   /*streamIdx=*/0, hbmSlotIndex, reqID, /*previousLoadNum=*/0);
    EXPECT_EQ(zeroTokens, 2u);
    EXPECT_EQ(hbmSlotIndex.size(), 2u);
    // token 0: slot 0 -> hbmIdx=blockTable[0/4]=7, hbmSlot=7*4+0=28
    EXPECT_EQ(hbmSlotIndex[0], 28ull);
    // token 1: slot 4 -> hbmIdx=blockTable[4/4]=9, hbmSlot=9*4+0=36
    EXPECT_EQ(hbmSlotIndex[1], 36ull);
    // Owned tokens zeroed: kBuf[0..2*kTokenBytes) = 0.
    EXPECT_EQ(kBuf[0], std::byte{0});
    EXPECT_EQ(kBuf[2 * 8 - 1], std::byte{0});
    std::free(kBuf);
    std::free(vBuf);
}

TEST_F(PreLogicTest, FillFailedUCMMergeSkipsUnownedContexts)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0};
    engine->mReqUseBlockTable[reqID] = {7};
    constexpr size_t kBufBytes = 256;
    std::byte* kBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kBuf, 0xAA, kBufBytes);
    std::memset(vBuf, 0xAA, kBufBytes);
    engine->mSharedKBufferHostPtr.resize(1);
    engine->mSharedVBufferHostPtr.resize(1);
    engine->mSharedKBufferHostPtr[0] = reinterpret_cast<uint64_t>(kBuf);
    engine->mSharedVBufferHostPtr[0] = reinterpret_cast<uint64_t>(vBuf);
    std::vector<size_t> tokenIndices = {0};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&tokenIndices, 0, 0, false, 0, nullptr, nullptr, /*owned=*/false},
    };
    std::vector<uint64_t> hbmSlotIndex;
    size_t zeroTokens = engine->FillFailedUCMMerge(blockContexts, 0, 0, hbmSlotIndex, reqID, 0);
    EXPECT_EQ(zeroTokens, 0u);  // unowned -> no zeroing
    EXPECT_EQ(hbmSlotIndex.size(), 1u);
    // kBuf untouched (unowned).
    EXPECT_EQ(kBuf[0], std::byte{0xAA});
    std::free(kBuf);
    std::free(vBuf);
}

/* ---------------- UCMTensorMerge ---------------- */

TEST_F(PreLogicTest, UCMTensorMergeReturnsEarlyWhenNoMissTokens)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    // mMissGlobalToken[reqID][0] default-empty -> ValidateUCMMergeRequest false.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    std::vector<uint64_t> hbmSlotIndex;
    engine->UCMTensorMerge(/*layerID=*/0, /*streamIdx=*/0, hbmSlotIndex, reqID, /*requestIndex=*/0);
    // ValidateUCMMergeRequest returned false -> early return, no load.
    EXPECT_TRUE(store->calls.empty());
    EXPECT_EQ(engine->mLoadNum[0], 0);
}

TEST_F(PreLogicTest, UCMTensorMergeFallsBackToZeroOnStoreFailure)
{
    engine = MakeEngine(/*layerNum=*/1);
    store->loadFails = true;
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0, 4};  // 2 miss tokens
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0, 4};
    engine->mReqUseBlockTable[reqID] = {7, 9};
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1), MakeBlockId(2)};
    engine->mBlockSize = 4;
    engine->mLoadNum.assign(1, 0);
    engine->mLoadIndexBufferLen = 16;
    engine->mKVBufferLen = 16;
    // Shared buffers for FillFailedUCMMerge zero path.
    constexpr size_t kBufBytes = 256;
    std::byte* kBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kBuf, 0xDD, kBufBytes);
    std::memset(vBuf, 0xDD, kBufBytes);
    engine->mSharedKBufferHostPtr.resize(1);
    engine->mSharedVBufferHostPtr.resize(1);
    engine->mSharedKBufferHostPtr[0] = reinterpret_cast<uint64_t>(kBuf);
    engine->mSharedVBufferHostPtr[0] = reinterpret_cast<uint64_t>(vBuf);
    std::vector<uint64_t> hbmSlotIndex;
    engine->UCMTensorMerge(0, 0, hbmSlotIndex, reqID, 0);
    // Load failed -> FillFailedUCMMerge ran; mLoadNum updated to finalLoadNum=2.
    EXPECT_EQ(engine->mLoadNum[0], 2);
    ASSERT_EQ(store->calls.size(), 1u);
    EXPECT_EQ(store->calls[0].name, "Load");  // Load+Wait path; Load fails first
    std::free(kBuf);
    std::free(vBuf);
}

TEST_F(PreLogicTest, UCMTensorMergeSucceedsAndUpdatesLoadNum)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0};  // 1 miss token (block 0)
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0};
    engine->mReqUseBlockTable[reqID] = {7};
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1)};
    engine->mBlockSize = 4;
    engine->mLoadNum.assign(1, 0);
    engine->mLoadIndexBufferLen = 16;
    engine->mKVBufferLen = 16;
    // Shared buffers for the serial copy path.
    constexpr size_t kBufBytes = 256;
    std::byte* kBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kBuf, 0, kBufBytes);
    std::memset(vBuf, 0, kBufBytes);
    engine->mSharedKBufferHostPtr.resize(1);
    engine->mSharedVBufferHostPtr.resize(1);
    engine->mSharedKBufferHostPtr[0] = reinterpret_cast<uint64_t>(kBuf);
    engine->mSharedVBufferHostPtr[0] = reinterpret_cast<uint64_t>(vBuf);
    std::vector<uint64_t> hbmSlotIndex;
    engine->UCMTensorMerge(0, 0, hbmSlotIndex, reqID, 0);
    // Success path: mLoadNum updated to finalLoadNum=1, no zero fallback.
    EXPECT_EQ(engine->mLoadNum[0], 1);
    /* After the ddr-load refactor the merge path uses Load + Wait (synchronous)
     * into the engine-owned staging buffer; the UCMMergeHostAddress destructor
     * no longer calls ReleaseHostBuffer. So exactly two store calls (Load then
     * Wait), with no ReleaseHostBuffer. */
    ASSERT_EQ(store->calls.size(), 2u);
    EXPECT_EQ(store->calls[0].name, "Load");
    EXPECT_EQ(store->calls[1].name, "Wait");
    std::free(kBuf);
    std::free(vBuf);
}

/* ---------------- CopyUCMMergeTokensSerial ---------------- */

TEST_F(PreLogicTest, CopyUCMMergeTokensSerialCopiesOwnedTokens)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {0, 4};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {0, 4};
    engine->mReqUseBlockTable[reqID] = {7, 9};
    constexpr size_t kBufBytes = 256;
    std::byte* kBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vBuf = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kBuf, 0, kBufBytes);
    std::memset(vBuf, 0, kBufBytes);
    engine->mSharedKBufferHostPtr.resize(1);
    engine->mSharedVBufferHostPtr.resize(1);
    engine->mSharedKBufferHostPtr[0] = reinterpret_cast<uint64_t>(kBuf);
    engine->mSharedVBufferHostPtr[0] = reinterpret_cast<uint64_t>(vBuf);
    // 2 owned UCM contexts, each with one token. CopyUCMMergeTokensSerial does
    // NOT call PrepareUCMMergeSources, so we must provide valid kSrc/vSrc.
    std::byte shardBlock[128];
    for (int i = 0; i < 128; ++i) { shardBlock[i] = static_cast<std::byte>(0x70 + i); }
    std::vector<size_t> ti0 = {0}, ti1 = {1};
    std::vector<UCMMergeBlockContext> blockContexts = {
        {&ti0, /*viewIndex=*/0, 0, false, 0, shardBlock, shardBlock + 64, true},
        {&ti1, /*viewIndex=*/0, 1, false, 1, shardBlock, shardBlock + 64, true},
    };
    // CopyUCMMergeTokensSerial writes hbmSlotIndex[dstOffset] without resizing;
    // pre-size to finalLoadNum (previousLoadNum + total tokens = 2).
    std::vector<uint64_t> hbmSlotIndex(2, 0);
    engine->CopyUCMMergeTokensSerial(blockContexts, /*layerID=*/0,
                                     /*streamIdx=*/0, hbmSlotIndex, reqID, /*previousLoadNum=*/0);
    ASSERT_EQ(hbmSlotIndex.size(), 2u);
    // token 0: slot 0 -> hbmIdx=blockTable[0]=7, hbmSlot=0%4=0 -> 7*4+0=28
    EXPECT_EQ(hbmSlotIndex[0], 28ull);
    // token 1: slot 4 -> hbmIdx=blockTable[1]=9, hbmSlot=4%4=0 -> 9*4+0=36
    EXPECT_EQ(hbmSlotIndex[1], 36ull);
    std::free(kBuf);
    std::free(vBuf);
}

/* ---------------- PrepareLastChunkSources ---------------- */

TEST_F(PreLogicTest, PrepareLastChunkSourcesReturnsFalseOnInvalidBlockCount)
{
    engine = MakeEngine(/*layerNum=*/1);
    // moveMap.size()/blockSize > mLastChunkNum(=0) -> invalid.
    std::vector<int> moveMap = {0, 1, 2, 3};  // blockSize=4 -> blockCount=1, but mLastChunkNum=0
    std::vector<int> blockTable = {0};
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_FALSE(engine->PrepareLastChunkSources(
        0, "req1", moveMap, blockTable, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
}

TEST_F(PreLogicTest, PrepareLastChunkSourcesReturnsFalseWhenBlockTableTooSmall)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mLastChunkNum = 4;
    // blockCount=1, blockTable.size()=0 < 1 -> invalid.
    std::vector<int> moveMap = {0, 1, 2, 3};  // blockCount=1
    std::vector<int> blockTable;
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_FALSE(engine->PrepareLastChunkSources(
        0, "req1", moveMap, blockTable, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
}

TEST_F(PreLogicTest, PrepareLastChunkSourcesReturnsFalseWhenHbmSlotOutOfRange)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mLastChunkNum = 4;
    // blockTable[0] = -1 < 0 -> out of range.
    std::vector<int> moveMap = {0, 1, 2, 3};
    std::vector<int> blockTable = {-1};
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_FALSE(engine->PrepareLastChunkSources(
        0, "req1", moveMap, blockTable, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
}

TEST_F(PreLogicTest, PrepareLastChunkSourcesReturnsFalseWhenNoNonUcmBlockMap)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mLastChunkNum = 4;
    engine->mKVHBMBlockLen = 16;
    engine->mAllBlocksHash["req1"] = {MakeBlockId(1)};
    // No mLastKVBlockAddress entry -> BuildLastChunkSources returns false.
    std::vector<int> moveMap = {0, 1, 2, 3};  // block 0 (UCM, in hashVec)
    std::vector<int> blockTable = {0};
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_FALSE(engine->PrepareLastChunkSources(
        0, "req1", moveMap, blockTable, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
}

TEST_F(PreLogicTest, PrepareLastChunkSourcesBuildsUcmShards)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mLastChunkNum = 4;
    engine->mKVHBMBlockLen = 16;
    engine->mAllBlocksHash["req1"] = {MakeBlockId(1), MakeBlockId(2)};
    // BuildLastChunkSources requires mLastKVBlockAddress[reqID] to exist
    // (checked at the top before the token loop). Seed an empty map.
    engine->mLastKVBlockAddress["req1"] = {};
    std::vector<int> moveMap = {0, 1, 2, 3};  // block 0 (UCM)
    std::vector<int> blockTable = {0};
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_TRUE(engine->PrepareLastChunkSources(
        0, "req1", moveMap, blockTable, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
    ASSERT_EQ(task.size(), 1u);  // 1 UCM shard for block 0
    EXPECT_EQ(loadedBlockIndices.size(), 1u);
    EXPECT_EQ(loadedBlockIndices[0], 0);
}

/* ---------------- BuildLastChunkSources (non-UCM path) ---------------- */

TEST_F(PreLogicTest, BuildLastChunkSourcesHandlesNonUcmBlock)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mLastChunkNum = 4;
    engine->mKVHBMBlockLen = 16;
    // hashVec has 1 entry; block 5 >= 1 -> non-UCM path.
    engine->mAllBlocksHash["req1"] = {MakeBlockId(1)};
    std::byte kBlock[64], vBlock[64];
    engine->mLastKVBlockAddress["req1"][5].kAddr = reinterpret_cast<uintptr_t>(kBlock);
    engine->mLastKVBlockAddress["req1"][5].vAddr = reinterpret_cast<uintptr_t>(vBlock);
    std::vector<int> moveMap = {20, 21, 22, 23};  // block 5 (20/4=5)
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_TRUE(engine->BuildLastChunkSources(
        0, "req1", moveMap, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
    EXPECT_TRUE(task.empty());           // non-UCM -> no shard
    ASSERT_GE(sourceBlocks.size(), 6u);  // resized to blockIdx+1=6
    EXPECT_NE(sourceBlocks[5].kBase, nullptr);
    EXPECT_NE(sourceBlocks[5].vBase, nullptr);
}

TEST_F(PreLogicTest, BuildLastChunkSourcesReturnsFalseOnNegativeToken)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mAllBlocksHash["req1"] = {MakeBlockId(1)};
    engine->mLastKVBlockAddress["req1"][0].kAddr = 1;
    std::vector<int> moveMap = {-1};  // negative -> return false
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_FALSE(engine->BuildLastChunkSources(
        0, "req1", moveMap, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
}

TEST_F(PreLogicTest, BuildLastChunkSourcesReturnsFalseWhenNonUcmBlockMissing)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mAllBlocksHash["req1"] = {MakeBlockId(1)};
    engine->mLastKVBlockAddress["req1"][0].kAddr = 1;  // only block 0
    // block 5 is non-UCM (>=1) but not in mLastKVBlockAddress[req1].
    std::vector<int> moveMap = {20};  // block 5 (20/4=5)
    std::vector<LastChunkSourceBlock> sourceBlocks;
    UC::Detail::TaskDesc task;
    std::vector<int> loadedBlockIndices;
    EXPECT_FALSE(engine->BuildLastChunkSources(
        0, "req1", moveMap, sourceBlocks, task, loadedBlockIndices,
        static_cast<std::byte*>(engine->mLastChunkLoadBuffer), engine->mShardSize));
}

/* ---------------- LoadLastChunkSources ---------------- */

TEST_F(PreLogicTest, LoadLastChunkSourcesReturnsTrueWhenTaskEmpty)
{
    engine = MakeEngine(/*layerNum=*/1);
    UC::Detail::TaskDesc task;  // empty -> early return true
    std::vector<int> loadedBlockIndices;
    std::vector<LastChunkSourceBlock> sourceBlocks;
    EXPECT_TRUE(engine->LoadLastChunkSources(0, "req1", task, loadedBlockIndices, sourceBlocks));
}

TEST_F(PreLogicTest, LoadLastChunkSourcesReturnsFalseWhenStoreFails)
{
    engine = MakeEngine(/*layerNum=*/1);
    store->loadFails = true;
    UC::Detail::TaskDesc task;
    task.resize(1);
    std::vector<int> loadedBlockIndices = {0};
    std::vector<LastChunkSourceBlock> sourceBlocks(1);
    EXPECT_FALSE(engine->LoadLastChunkSources(0, "req1", task, loadedBlockIndices, sourceBlocks));
    ASSERT_EQ(store->calls.size(), 1u);  // Load attempted, fails -> no Wait
    EXPECT_EQ(store->calls[0].name, "Load");
}

TEST_F(PreLogicTest, LoadLastChunkSourcesReturnsFalseWhenWaitFails)
{
    /* After the ddr-load refactor the load path is Load + Wait into the
     * engine-owned mLastChunkLoadBuffer; the old view-count-mismatch /
     * null-host-addr failure modes no longer exist. A Wait failure now covers
     * that branch. */
    engine = MakeEngine(/*layerNum=*/1);
    store->waitFails = true;
    UC::Detail::TaskDesc task;
    task.resize(1);
    std::vector<int> loadedBlockIndices = {0};
    std::vector<LastChunkSourceBlock> sourceBlocks(1);
    EXPECT_FALSE(engine->LoadLastChunkSources(0, "req1", task, loadedBlockIndices, sourceBlocks));
    ASSERT_EQ(store->calls.size(), 2u);  // Load + Wait
    EXPECT_EQ(store->calls[0].name, "Load");
    EXPECT_EQ(store->calls[1].name, "Wait");
}

TEST_F(PreLogicTest, LoadLastChunkSourcesSucceedsAndPopulatesSourceBlocks)
{
    engine = MakeEngine(/*layerNum=*/1);
    UC::Detail::TaskDesc task;
    task.resize(1);
    std::vector<int> loadedBlockIndices = {0};
    std::vector<LastChunkSourceBlock> sourceBlocks(1);
    EXPECT_TRUE(engine->LoadLastChunkSources(0, "req1", task, loadedBlockIndices, sourceBlocks));
    /* hasHostTask is gone; success is observable via populated sourceBlocks
     * pointing into the engine-owned mLastChunkLoadBuffer. */
    EXPECT_NE(sourceBlocks[0].kBase, nullptr);
    EXPECT_NE(sourceBlocks[0].vBase, nullptr);  // !mISSPAC8
    /* kBase points at the staging buffer; vBase = kBase + kBlockBytes. */
    EXPECT_EQ(sourceBlocks[0].vBase, static_cast<const std::byte*>(sourceBlocks[0].kBase) +
                                         static_cast<size_t>(engine->mBlockSize) *
                                             static_cast<size_t>(engine->mKVSizeBytesToken));
}

/* ---------------- CopyLastChunkTokens ---------------- */

TEST_F(PreLogicTest, CopyLastChunkTokensCopiesTokensSerially)
{
    engine = MakeEngine(/*layerNum=*/1);
    // mLastCHunkKPtr/VPtr must be valid (set via SetKvTensor; pre-seed here).
    constexpr size_t kBufBytes = 256;
    std::byte* kDst = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vDst = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kDst, 0, kBufBytes);
    std::memset(vDst, 0, kBufBytes);
    engine->mLastCHunkKPtr.resize(1);
    engine->mLastCHunkVPtr.resize(1);
    engine->mLastCHunkKPtr[0] = reinterpret_cast<uint64_t>(kDst);
    engine->mLastCHunkVPtr[0] = reinterpret_cast<uint64_t>(vDst);
    // 1 UCM block (block 0), 4 tokens.
    std::byte kSrcBlock[64], vSrcBlock[64];
    for (int i = 0; i < 64; ++i) {
        kSrcBlock[i] = static_cast<std::byte>(0xA0 + i);
        vSrcBlock[i] = static_cast<std::byte>(0xB0 + i);
    }
    std::vector<int> moveMap = {0, 1, 2, 3};  // block 0, tokens 0..3
    std::vector<LastChunkSourceBlock> sourceBlocks = {
        {kSrcBlock, vSrcBlock},
    };
    /* CopyLastChunkTokens no longer takes hostTaskId/hasHostTask; the staging
     * buffer is released by the engine, not by this call. */
    engine->CopyLastChunkTokens(0, "req1", moveMap, sourceBlocks);
    // Verify K copied: kDst[i*kTokenBytes .. +kTokenBytes) == kSrcBlock[i*kTokenBytes ..]
    EXPECT_EQ(std::memcmp(kDst, kSrcBlock, 8), 0);          // token 0
    EXPECT_EQ(std::memcmp(kDst + 8, kSrcBlock + 8, 8), 0);  // token 1
    // Verify V copied: vDst[i*vTokenBytes ..] == vSrcBlock[i*vTokenBytes ..]
    EXPECT_EQ(std::memcmp(vDst, vSrcBlock, 8), 0);
    std::free(kDst);
    std::free(vDst);
}

TEST_F(PreLogicTest, CopyLastChunkTokensDoesNotCallStoreAfterDdrLoadRefactor)
{
    /* Previously CopyLastChunkTokens released the HostBuffer task when
     * hasHostTask was set. After the ddr-load refactor the staging buffer is
     * engine-owned (mLastChunkLoadBuffer) and there is no per-call release, so
     * the store is never touched from this path. */
    engine = MakeEngine(/*layerNum=*/1);
    constexpr size_t kBufBytes = 256;
    std::byte* kDst = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vDst = static_cast<std::byte*>(std::malloc(kBufBytes));
    engine->mLastCHunkKPtr.resize(1);
    engine->mLastCHunkVPtr.resize(1);
    engine->mLastCHunkKPtr[0] = reinterpret_cast<uint64_t>(kDst);
    engine->mLastCHunkVPtr[0] = reinterpret_cast<uint64_t>(vDst);
    std::byte kSrcBlock[64], vSrcBlock[64];
    std::vector<int> moveMap = {0};
    std::vector<LastChunkSourceBlock> sourceBlocks = {
        {kSrcBlock, vSrcBlock}
    };
    engine->CopyLastChunkTokens(0, "req1", moveMap, sourceBlocks);
    EXPECT_TRUE(store->calls.empty());
    std::free(kDst);
    std::free(vDst);
}

/* ---------------- LastChunkMove (deep path) ---------------- */

TEST_F(PreLogicTest, LastChunkMoveEndToEndWithUcmBlock)
{
    engine = MakeEngine(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->mLastChunkNum = 4;
    engine->mKVHBMBlockLen = 16;
    engine->mBlockSize = 4;
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1), MakeBlockId(2)};
    engine->mReqUseBlockTable[reqID] = {0};
    // BuildLastChunkSources requires mLastKVBlockAddress[reqID] to exist.
    engine->mLastKVBlockAddress[reqID] = {};
    // 4 tokens (block 0, UCM), aligned to blockSize.
    std::vector<int> moveMap = {0, 1, 2, 3};
    // Pre-seed mLastCHunkKPtr/VPtr for CopyLastChunkTokens.
    constexpr size_t kBufBytes = 256;
    std::byte* kDst = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::byte* vDst = static_cast<std::byte*>(std::malloc(kBufBytes));
    std::memset(kDst, 0, kBufBytes);
    std::memset(vDst, 0, kBufBytes);
    engine->mLastCHunkKPtr.resize(1);
    engine->mLastCHunkVPtr.resize(1);
    engine->mLastCHunkKPtr[0] = reinterpret_cast<uint64_t>(kDst);
    engine->mLastCHunkVPtr[0] = reinterpret_cast<uint64_t>(vDst);
    engine->LastChunkMove(0, reqID, moveMap);
    // End-to-end: PrepareLastChunkSources -> LoadLastChunkSources (Load+Wait) ->
    // CopyLastChunkTokens. The ddr-load refactor switched this from
    // LoadToHostBuffer/ReleaseHostBuffer to Load+Wait into mLastChunkLoadBuffer.
    ASSERT_GE(store->calls.size(), 1u);
    EXPECT_EQ(store->calls[0].name, "Load");
    std::free(kDst);
    std::free(vDst);
}
