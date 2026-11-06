/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the pure-logic prefetch-scatter methods of GSAPrefetchEngineC declared
 * in kvcache_pre.h and implemented in kvcache_pre_scatter.cpp.
 *
 * These methods (GetHitAndMissBlock, RunPrefetchH2D, DealLessPrefetchBlock,
 * LoadKVToHBM, FinishOneLayerPrefetch, SetTopkIndices, and the Get* accessors)
 * are pure host logic gated by mOpenIndexerOffload; they only touch engine
 * state and never touch ACL device ops.  Driving them directly exercises the
 * largest block of uncovered logic in kvcache_pre_scatter.cpp without needing
 * a live TaskManager worker pool or ACL device.
 */
#include <gtest/gtest.h>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch::PrefetchReqInfo;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class ScatterLogicTest : public ::testing::Test {
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

    /* Build an engine with indexer offload ON (so the scatter logic paths run)
       and the topkLayerMap pre-seeded to [0..layerNum-1]. */
    std::unique_ptr<GSAPrefetchEngineC> MakeEngine(int layerNum = 2)
    {
        auto model = MakeModelInfo(layerNum, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                                   /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                                 /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
        std::vector<int> topkLayerMap(layerNum);
        for (int i = 0; i < layerNum; ++i) { topkLayerMap[i] = i; }
        auto e = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
        return e;
    }

    /* Pre-seed the per-(reqID, layer) tables exactly as the engine would after
       SetBlocksMap + AddIndexerMap ran. This lets the scatter logic find the
       block tables / docs tables it expects without going through the heavy
       TaskManager + ACL load path. */
    void SeedBlockTables(const std::string& reqID, int layerNum,
                         const std::vector<int>& docsTableContents /* item -> blockID */)
    {
        engine->mDocsTables[reqID].assign(layerNum, {});
        engine->mBlocksMap[reqID].assign(layerNum, {});
        engine->mPrefetchIdx[reqID].assign(layerNum, {});
        engine->allNeedLoadBlock[reqID].assign(layerNum, {});
        engine->allMissIdxs[reqID].assign(layerNum, {});
        // mPrefetchBlockTableIdx[reqID][layerID] must be sized to >= hitBlocksIdx
        // entries, because FinishOneLayerPrefetch writes [pairIdx] directly.
        engine->mPrefetchBlockTableIdx[reqID].assign(
            layerNum, std::vector<int>(docsTableContents.size() + 8, -1));
        engine->mIndexerSlot[reqID].assign(layerNum, 0);
        for (int layer = 0; layer < layerNum; ++layer) {
            int idx = 0;
            for (int blockID : docsTableContents) {
                engine->mDocsTables[reqID][layer][idx] = blockID;
                engine->mBlocksMap[reqID][layer][blockID] = idx;
                engine->mPrefetchIdx[reqID][layer].push_back(idx);
                idx++;
            }
        }
        // mIndexerLoadResultT is a Tensor that FinishOneLayerPrefetch memcpy()s into.
        // Allocate enough int32 storage for the block table result.
        engine->mIndexerLoadResultT[reqID] = torch::full(
            {static_cast<int64_t>(docsTableContents.size() + 8)}, -1, torch::dtype(torch::kInt32));
    }
};

}  // namespace

/* ---------------- GetHitAndMissBlock ---------------- */

TEST_F(ScatterLogicTest, GetHitAndMissBlockClassifiesHitsAndMisses)
{
    engine = MakeEngine(/*layerNum=*/2);
    SeedBlockTables("req1", /*layerNum=*/2, /*docsTableContents=*/{10, 20});
    PrefetchReqInfo info{};
    info.topkLen = 4;
    info.reqID = "req1";
    info.layerID = 0;
    info.topkIndex = 0;
    info.bsIndex = 0;
    std::vector<int> topk = {0, 1, 5, 6};  // 0,1 -> hit (blocks 10,20); 5,6 -> miss
    std::unordered_set<int> hitBlocks;
    std::map<int, int> hitBlocksIdx;
    std::vector<int> missIdxs;
    engine->GetHitAndMissBlock(info, topk, hitBlocks, hitBlocksIdx, missIdxs);
    EXPECT_EQ(hitBlocks, (std::unordered_set<int>{10, 20}));
    EXPECT_EQ(hitBlocksIdx.size(), 2u);
    EXPECT_EQ(missIdxs, (std::vector<int>{5, 6}));
}

TEST_F(ScatterLogicTest, GetHitAndMissBlockIsNoopWhenIndexerOffloadOff)
{
    // Build an engine with indexer offload OFF.
    auto model = MakeModelInfo(/*layerNum=*/2, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/2);
    auto size = MakeSizeInfo();
    std::vector<int> topkLayerMap = {0, 1};
    engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                  /*rank=*/0, /*threadNum=*/1, std::vector<int>{0},
                                                  topkLayerMap, logFile->Path());
    PrefetchReqInfo info{};
    info.topkLen = 2;
    info.reqID = "req1";
    info.layerID = 0;
    std::vector<int> topk = {0, 1};
    std::unordered_set<int> hitBlocks;
    std::map<int, int> hitBlocksIdx;
    std::vector<int> missIdxs;
    engine->GetHitAndMissBlock(info, topk, hitBlocks, hitBlocksIdx, missIdxs);
    EXPECT_TRUE(hitBlocks.empty());
    EXPECT_TRUE(missIdxs.empty());
}

TEST_F(ScatterLogicTest, GetHitAndMissBlockAllHitsProducesNoMiss)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{10, 20, 30});
    PrefetchReqInfo info{};
    info.topkLen = 3;
    info.reqID = "req1";
    info.layerID = 0;
    std::vector<int> topk = {0, 1, 2};
    std::unordered_set<int> hitBlocks;
    std::map<int, int> hitBlocksIdx;
    std::vector<int> missIdxs;
    engine->GetHitAndMissBlock(info, topk, hitBlocks, hitBlocksIdx, missIdxs);
    EXPECT_EQ(hitBlocks, (std::unordered_set<int>{10, 20, 30}));
    EXPECT_TRUE(missIdxs.empty());
}

/* ---------------- RunPrefetchH2D ---------------- */

TEST_F(ScatterLogicTest, RunPrefetchH2DAllocatesFreeBlocksForMisses)
{
    engine = MakeEngine(/*layerNum=*/1);
    // docsTable: item 0->block 100, item 1->block 200, item 2->block 300
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{100, 200, 300});
    PrefetchReqInfo info{};
    info.topkLen = 3;
    info.reqID = "req1";
    info.layerID = 0;
    std::vector<int> topk = {0, 5, 6};  // 0 hit; 5,6 miss
    std::unordered_set<int> hitBlocks{100};
    std::map<int, int> hitBlocksIdx{
        {0, 100}
    };
    std::vector<int> missIdxs{5, 6};
    engine->RunPrefetchH2D(info, hitBlocks, hitBlocksIdx, missIdxs);
    // The two misses should be served from free blocks (200, 300), each inserted once.
    EXPECT_EQ(hitBlocks.size(), 3u);
    EXPECT_EQ(hitBlocksIdx.size(), 3u);
    // allNeedLoadBlock / allMissIdxs are populated for LoadKVToHBM.
    EXPECT_EQ(engine->allNeedLoadBlock["req1"][0].size(), 2u);
    EXPECT_EQ(engine->allMissIdxs["req1"][0].size(), 2u);
}

TEST_F(ScatterLogicTest, RunPrefetchH2DSkipsAlreadyHitBlocks)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{100, 100, 100});
    PrefetchReqInfo info{};
    info.topkLen = 3;
    info.reqID = "req1";
    info.layerID = 0;
    std::vector<int> topk = {0, 1, 2};
    std::unordered_set<int> hitBlocks{100};
    std::map<int, int> hitBlocksIdx{
        {0, 100}
    };
    std::vector<int> missIdxs{1, 2};
    engine->RunPrefetchH2D(info, hitBlocks, hitBlocksIdx, missIdxs);
    // All free blocks map to block 100 which is already a hit; none allocated.
    EXPECT_EQ(engine->allNeedLoadBlock["req1"][0].size(), 0u);
    // missIdxs is truncated to loadLen (=0).
    EXPECT_EQ(missIdxs.size(), 0u);
}

TEST_F(ScatterLogicTest, RunPrefetchH2DIsNoopWhenIndexerOffloadOff)
{
    auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/2);
    auto size = MakeSizeInfo();
    std::vector<int> topkLayerMap = {0};
    engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                  /*rank=*/0, /*threadNum=*/1, std::vector<int>{0},
                                                  topkLayerMap, logFile->Path());
    PrefetchReqInfo info{};
    info.reqID = "req1";
    info.layerID = 0;
    std::unordered_set<int> hitBlocks{1};
    std::map<int, int> hitBlocksIdx{
        {0, 1}
    };
    std::vector<int> missIdxs{2};
    engine->RunPrefetchH2D(info, hitBlocks, hitBlocksIdx, missIdxs);
    // No state mutated (offload off -> early return before touching tables).
    EXPECT_EQ(engine->allNeedLoadBlock.count("req1"), 0u);
}

/* ---------------- DealLessPrefetchBlock ---------------- */

TEST_F(ScatterLogicTest, DealLessPrefetchBlockFillsRemainingHitsFromDocsReverse)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{100, 200, 300});
    std::unordered_set<int> hitBlocks{100};
    std::map<int, int> hitBlocksIdx{
        {0, 100}
    };
    // topkLen=3 but only 1 hit so far; DealLess should add 200, 300 from reverse
    // traversal until hitBlocks.size() == topkLen.
    engine->DealLessPrefetchBlock(hitBlocks, hitBlocksIdx, "req1", /*layerID=*/0, /*topkLen=*/3);
    EXPECT_EQ(hitBlocks.size(), 3u);
    EXPECT_EQ(hitBlocksIdx.size(), 3u);
}

TEST_F(ScatterLogicTest, DealLessPrefetchBlockNoopWhenOffloadOff)
{
    auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/2);
    auto size = MakeSizeInfo();
    std::vector<int> topkLayerMap = {0};
    engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                  /*rank=*/0, /*threadNum=*/1, std::vector<int>{0},
                                                  topkLayerMap, logFile->Path());
    std::unordered_set<int> hitBlocks{1};
    std::map<int, int> hitBlocksIdx{
        {0, 1}
    };
    engine->DealLessPrefetchBlock(hitBlocks, hitBlocksIdx, "req1", /*layerID=*/0, /*topkLen=*/3);
    EXPECT_EQ(hitBlocks.size(), 1u);  // unchanged
}

/* ---------------- LoadKVToHBM ---------------- */

TEST_F(ScatterLogicTest, LoadKVToHBMSwapsBlocksAndUpdatesDocsTables)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{100, 200});
    // loadNPUBlockIDs=[100], missIdxs=[5]: replace block 100's doc entry.
    // Before: docsTable[5] doesn't exist, blocksMap[100]=0 (item 0).
    // After: blocksMap[reqID][0][100]=5, docsTable[reqID][0][5]=100, old docsTable[0] erased.
    engine->LoadKVToHBM(/*loadNPUBlockIDs=*/{100}, /*missIdxs=*/{5},
                        /*layerID=*/0, /*reqID=*/"req1");
    EXPECT_EQ(engine->mBlocksMap["req1"][0][100], 5);
    EXPECT_EQ(engine->mDocsTables["req1"][0][5], 100);
    EXPECT_EQ(engine->mDocsTables["req1"][0].count(0), 0u);  // old item erased
}

TEST_F(ScatterLogicTest, LoadKVToHBMEmptyListsIsNoop)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{100});
    engine->LoadKVToHBM(/*loadNPUBlockIDs=*/{}, /*missIdxs=*/{},
                        /*layerID=*/0, /*reqID=*/"req1");
    EXPECT_EQ(engine->mBlocksMap["req1"][0][100], 0);  // unchanged
}

/* ---------------- FinishOneLayerPrefetch ---------------- */

TEST_F(ScatterLogicTest, FinishOneLayerPrefetchSortsAndPublishesBlockTable)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{100, 200, 300});
    // hitBlocksIdx: item->blockID pairs (unsorted input)
    std::map<int, int> hitBlocksIdx{
        {2, 300},
        {0, 100},
        {1, 200}
    };
    std::vector<int> missIdxs;  // no loads
    ASSERT_TRUE(engine->mIndexerLoadResultT["req1"].defined());
    ASSERT_TRUE(engine->mIndexerLoadResultT["req1"].data_ptr() != nullptr);
    engine->FinishOneLayerPrefetch("req1", /*layerID=*/0, hitBlocksIdx, missIdxs);
    // mPrefetchBlockTableIdx[reqID][0][idx] holds the sorted item keys: 0,1,2.
    // (The vector is pre-sized; only the first 3 slots are written.)
    EXPECT_EQ(engine->mPrefetchBlockTableIdx["req1"][0][0], 0);
    EXPECT_EQ(engine->mPrefetchBlockTableIdx["req1"][0][1], 1);
    EXPECT_EQ(engine->mPrefetchBlockTableIdx["req1"][0][2], 2);
    // mPrefetchIdx is cleared after the call (then refilled with non-hit items).
    // Since all items were hits, mPrefetchIdx[reqID][0] is empty afterward.
    EXPECT_TRUE(engine->mPrefetchIdx["req1"][0].empty());
}

TEST_F(ScatterLogicTest, FinishOneLayerPrefetchRefillsPrefetchIdxWithNonHits)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedBlockTables("req1", /*layerNum=*/1, /*docsTableContents=*/{100, 200, 300, 400});
    // Only items 0,2 are hits; items 1,3 should be left in mPrefetchIdx.
    std::map<int, int> hitBlocksIdx{
        {0, 100},
        {2, 300}
    };
    std::vector<int> missIdxs;
    engine->FinishOneLayerPrefetch("req1", /*layerID=*/0, hitBlocksIdx, missIdxs);
    // mPrefetchIdx[reqID][0] should now hold the non-hit items {1, 3}.
    std::vector<int> refillIdx = engine->mPrefetchIdx["req1"][0];
    std::sort(refillIdx.begin(), refillIdx.end());
    EXPECT_EQ(refillIdx, (std::vector<int>{1, 3}));
}

/* ---------------- SetTopkIndices ---------------- */

TEST_F(ScatterLogicTest, SetTopkIndicesStoresTensorPerReqID)
{
    engine = MakeEngine(/*layerNum=*/1);
    auto tensor = torch::arange(0, 4, torch::dtype(torch::kInt32));
    engine->SetTopkIndices("req1", tensor);
    EXPECT_TRUE(torch::equal(engine->mAllTopkIndicesTensor["req1"], tensor));
    // Update with a different tensor.
    auto tensor2 = torch::arange(0, 8, torch::dtype(torch::kInt32));
    engine->SetTopkIndices("req1", tensor2);
    EXPECT_TRUE(torch::equal(engine->mAllTopkIndicesTensor["req1"], tensor2));
}

/* ---------------- Get* accessors ---------------- */

TEST_F(ScatterLogicTest, GetAccessorsReturnSeededState)
{
    engine = MakeEngine(/*layerNum=*/2);
    SeedBlockTables("req1", /*layerNum=*/2, /*docsTableContents=*/{10, 20});
    engine->mReqUseBlockTable["req1"] = {1, 2, 3};
    engine->mSparseSeqLen["req1"] = 42;
    // mHBMMangers[reqID][layerID] must exist for GetTokenBlockMap / GetBlockTokenMap.
    engine->mHBMMangers["req1"].push_back(std::make_unique<HBMManager>());
    engine->mHBMMangers["req1"].push_back(std::make_unique<HBMManager>());
    engine->mHBMMangers["req1"][0]->mGlobalToSlot = {7, 8};
    engine->mHBMMangers["req1"][0]->mSlotToGlobal = {9, 10};
    engine->mHBMMangers["req1"][1]->mGlobalToSlot = {11};
    engine->mHBMMangers["req1"][1]->mSlotToGlobal = {12};

    EXPECT_EQ(engine->GetTokenBlockMap(/*layerID=*/0, "req1"), (std::vector<int>{7, 8}));
    EXPECT_EQ(engine->GetBlockTokenMap(/*layerID=*/0, "req1"), (std::vector<int>{9, 10}));
    EXPECT_EQ(engine->GetTokenBlockMap(/*layerID=*/1, "req1"), (std::vector<int>{11}));
    EXPECT_EQ(engine->GetBlockTokenMap(/*layerID=*/1, "req1"), (std::vector<int>{12}));
    EXPECT_EQ(engine->GetIndexerTokensMap(/*layerID=*/0, "req1"), (std::map<int, int>{
                                                                      {0, 10},
                                                                      {1, 20}
    }));
    EXPECT_EQ(engine->GetIndexerBlockMap(/*layerID=*/0, "req1"), (std::map<int, int>{
                                                                     {10, 0},
                                                                     {20, 1}
    }));
    EXPECT_EQ(engine->GetReqUseBlockTable("req1"), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(engine->GetSparseSeqLen("req1"), 42);
}

TEST_F(ScatterLogicTest, GetSparseSeqLenReturnsZeroForUnknownReqID)
{
    engine = MakeEngine(/*layerNum=*/1);
    // operator[] default-constructs an int = 0 for unknown keys.
    EXPECT_EQ(engine->GetSparseSeqLen("ghost"), 0);
}

/* ---------------- DelReqIDRun integration ---------------- */

TEST_F(ScatterLogicTest, DelReqIDRunClearsPrefetchIdxAndBlocksMap)
{
    engine = MakeEngine(/*layerNum=*/2);
    SeedBlockTables("req1", /*layerNum=*/2, /*docsTableContents=*/{10, 20});
    engine->mSparseSeqLen["req1"] = 128;
    engine->mPromptLen["req1"] = 16;
    engine->mDelSeqIds.insert("req1");
    engine->DelReqIDRun();
    EXPECT_EQ(engine->mSparseSeqLen.count("req1"), 0u);
    EXPECT_EQ(engine->mPromptLen.count("req1"), 0u);
    EXPECT_EQ(engine->mBlocksMap.count("req1"), 0u);
    EXPECT_EQ(engine->mPrefetchIdx.count("req1"), 0u);
    // BUG-FOCUS: mAllBlocksHash is NOT cleared (documented bug #9).
    // (Not seeded here so it is gone anyway; covered in test_gsa_prefetch_engine.)
}

/* ---------------- SetKvTensor / SetKvTensorIndex ---------------- */

TEST_F(ScatterLogicTest, SetKvTensorStoresTensorsAndComputesLastChunkNum)
{
    engine = MakeEngine(/*layerNum=*/1);
    // blockSize=4, sparseLen=128 -> lastChunkNum = 128/4 = 32.
    auto kCaches = torch::arange(0, 16, torch::dtype(torch::kFloat32)).reshape({4, 4});
    auto vCaches = torch::arange(0, 16, torch::dtype(torch::kFloat32)).reshape({4, 4});
    auto lastChunkK = torch::arange(0, 8, torch::dtype(torch::kFloat32));
    auto lastChunkV = torch::arange(0, 8, torch::dtype(torch::kFloat32));
    engine->SetKvTensor(kCaches, vCaches, lastChunkK, lastChunkV, /*layerID=*/0);
    EXPECT_EQ(engine->mKVHBMBlockLen, 4);  // sizes()[0] of kCaches
    EXPECT_EQ(engine->mLastChunkNum, 32);  // mSparseLen(128) / mBlockSize(4)
    EXPECT_EQ(engine->mKcacheTensor[0].sizes()[0], 4);
    EXPECT_NE(engine->mKcachePtr[0], 0ull);
    EXPECT_NE(engine->mLastCHunkKPtr[0], 0ull);
    EXPECT_NE(engine->mVcachePtr[0], 0ull);  // !mISSPAC8 -> V stored
    EXPECT_NE(engine->mLastCHunkVPtr[0], 0ull);
}

TEST_F(ScatterLogicTest, SetKvTensorIndexIsNoopWhenIndexerOffloadOff)
{
    auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/2);
    auto size = MakeSizeInfo();
    std::vector<int> topkLayerMap = {0};
    engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                  /*rank=*/0, /*threadNum=*/1, std::vector<int>{0},
                                                  topkLayerMap, logFile->Path());
    auto idx = torch::arange(0, 4, torch::dtype(torch::kInt32));
    engine->SetKvTensorIndex(/*layerID=*/0, idx);
    // offload off -> early return: indexer containers stay at their ctor defaults
    // (mIndexerTensor / mIndexerPtr are NOT resized when offload is off).
    EXPECT_TRUE(engine->mIndexerPtr.empty());
    EXPECT_TRUE(engine->mIndexerTensor.empty());
    EXPECT_EQ(engine->mDeviceIndexerLen, 0);
}

TEST_F(ScatterLogicTest, SetKvTensorIndexStoresIndexerAndComputesSize)
{
    engine = MakeEngine(/*layerNum=*/1);
    // indexerCache: a 2D tensor whose [0] has element_size=4 and numel=4 -> 16 bytes/block.
    auto idx = torch::arange(0, 16, torch::dtype(torch::kInt32)).reshape({4, 4});
    engine->SetKvTensorIndex(/*layerID=*/0, idx);
    EXPECT_EQ(engine->mDeviceIndexerLen, 4);
    EXPECT_EQ(engine->mIndexerSizeBytesBlock, 16u);
    EXPECT_EQ(engine->mIndexerTensor[0].sizes()[0], 4);
    EXPECT_NE(engine->mIndexerPtr[0], 0ull);
}

/* ---------------- LoadNonUcmIndexerBlock ---------------- */

TEST_F(ScatterLogicTest, LoadNonUcmIndexerBlockReturnsFalseWhenNoLastKVBlockMap)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    std::uint8_t dst[16] = {0};
    bool ok = engine->LoadNonUcmIndexerBlock("req1", /*layerID=*/0, /*blockIdx=*/2,
                                             /*deviceBlockId=*/2, dst);
    EXPECT_FALSE(ok);  // no mLastKVBlockAddress entry
}

TEST_F(ScatterLogicTest, LoadNonUcmIndexerBlockReturnsFalseWhenBlockIdxMissing)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    // Register a last-block address for blockIdx 5 only.
    engine->mLastKVBlockAddress["req1"][5].indexerAddr =
        reinterpret_cast<uintptr_t>(new std::uint8_t[16]{});
    std::uint8_t dst[16] = {0};
    bool ok = engine->LoadNonUcmIndexerBlock("req1", /*layerID=*/0, /*blockIdx=*/2,
                                             /*deviceBlockId=*/2, dst);
    EXPECT_FALSE(ok);  // blockIdx 2 not in map
}

TEST_F(ScatterLogicTest, LoadNonUcmIndexerBlockCopiesAndReturnsTrueOnSuccess)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    // Source indexer block per layer (layerID 0 -> offset 0).
    auto* src = new std::uint8_t[32]();  // 2 layers worth
    for (int i = 0; i < 16; ++i) { src[i] = static_cast<std::uint8_t>(0xAB); }
    engine->mLastKVBlockAddress["req1"][2].indexerAddr = reinterpret_cast<uintptr_t>(src);
    std::uint8_t dst[16] = {0};
    bool ok = engine->LoadNonUcmIndexerBlock("req1", /*layerID=*/0, /*blockIdx=*/2,
                                             /*deviceBlockId=*/2, dst);
    EXPECT_TRUE(ok);
    for (int i = 0; i < 16; ++i) { EXPECT_EQ(dst[i], 0xAB); }
    delete[] src;
}

TEST_F(ScatterLogicTest, LoadNonUcmIndexerBlockReturnsFalseOnMemcpyFailure)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    acl_stub::Stub().memcpyFails = true;
    auto* src = new std::uint8_t[16]();
    engine->mLastKVBlockAddress["req1"][2].indexerAddr = reinterpret_cast<uintptr_t>(src);
    std::uint8_t dst[16] = {0};
    bool ok = engine->LoadNonUcmIndexerBlock("req1", /*layerID=*/0, /*blockIdx=*/2,
                                             /*deviceBlockId=*/2, dst);
    EXPECT_FALSE(ok);  // aclrtMemcpy failed
    delete[] src;
}

/* ---------------- SubmitIndexerInitLoadTask ---------------- */

TEST_F(ScatterLogicTest, SubmitIndexerInitLoadTaskReturnsTrueOnEmptyTask)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->SetStoreV1(store.get());
    UC::Detail::TaskDesc task;  // empty task -> early return true
    bool ok = engine->SubmitIndexerInitLoadTask(task, /*reqID=*/"req1", /*layerID=*/0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(store->calls.size(), 0u);  // Load not invoked
}

TEST_F(ScatterLogicTest, SubmitIndexerInitLoadTaskReturnsFalseWhenStoreFails)
{
    engine = MakeEngine(/*layerNum=*/1);
    store->loadFails = true;
    engine->SetStoreV1(store.get());
    UC::Detail::TaskDesc task;
    task.resize(1);  // non-empty task -> Load invoked -> fails (no Wait)
    bool ok = engine->SubmitIndexerInitLoadTask(task, /*reqID=*/"req1", /*layerID=*/0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(store->calls.size(), 1u);
    EXPECT_EQ(store->calls[0].name, "Load");
}

TEST_F(ScatterLogicTest, SubmitIndexerInitLoadTaskReturnsFalseWhenWaitFails)
{
    /* After the ddr-load refactor SubmitIndexerInitLoadTask issues Load + Wait
     * (replacing the old LoadToDevice). A Wait failure is a new failure mode. */
    engine = MakeEngine(/*layerNum=*/1);
    store->waitFails = true;
    engine->SetStoreV1(store.get());
    UC::Detail::TaskDesc task;
    task.resize(1);  // non-empty task -> Load succeeds, Wait fails
    bool ok = engine->SubmitIndexerInitLoadTask(task, /*reqID=*/"req1", /*layerID=*/0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(store->calls.size(), 2u);  // Load + Wait
    EXPECT_EQ(store->calls[0].name, "Load");
    EXPECT_EQ(store->calls[1].name, "Wait");
}
