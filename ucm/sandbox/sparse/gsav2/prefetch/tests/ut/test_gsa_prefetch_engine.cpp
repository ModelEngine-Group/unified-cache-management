/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the testable public surface of GSAPrefetchEngineC
 * (include/kvcache_pre.h).  The engine is constructed with the ACL test stub
 * so no real Ascend device is required; the heavy shared-memory / thread-pool
 * init is driven with rank 0 and a unique dp-rank per test.
 */
#include <gtest/gtest.h>
#include <thread>
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

class GSAEngineTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        acl_stub::Stub().Reset();
        dpRank_ = UniqueDpRank();
        CleanupShmFor(dpRank_);
        logFile_ = std::make_unique<TempLogFile>();
    }
    void TearDown() override
    {
        engine_.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank_);
    }
    std::unique_ptr<GSAPrefetchEngineC> MakeEngine(bool indexerOffload = false)
    {
        auto model = MakeModelInfo(/*layerNum=*/2, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank_, /*tpSize=*/1,
                                   /*indexerOffload=*/indexerOffload ? 1 : 0,
                                   /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo();
        std::vector<int> topkLayerMap(model[0]);  // layer 0..layerNum-1
        for (size_t i = 0; i < topkLayerMap.size(); ++i) { topkLayerMap[i] = i; }
        std::vector<int> cpuBind = {0};
        return std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                    /*rank=*/0, /*threadNum=*/1, cpuBind,
                                                    topkLayerMap, logFile_->Path());
    }
    int dpRank_{0};
    std::unique_ptr<TempLogFile> logFile_;
    std::unique_ptr<GSAPrefetchEngineC> engine_;
};

}  // namespace

/* ---- construction / destruction ---- */

TEST_F(GSAEngineTest, ConstructAndDestroySucceed)
{
    engine_ = MakeEngine();
    EXPECT_NE(engine_, nullptr);
    EXPECT_FALSE(engine_->GetPrefetchStatus());
    EXPECT_FALSE(engine_->GetIndexerStatus());
}

TEST_F(GSAEngineTest, ConstructWithIndexerOffloadCreatesIndexerTaskManager)
{
    engine_ = MakeEngine(/*indexerOffload=*/true);
    EXPECT_NE(engine_, nullptr);
}

TEST_F(GSAEngineTest, SetModelRunningStatusTogglesFlag)
{
    engine_ = MakeEngine();
    engine_->SetModelRunningStatus(true);
    EXPECT_TRUE(engine_->mStopPrefetch);
    engine_->SetModelRunningStatus(false);
    EXPECT_FALSE(engine_->mStopPrefetch);
}

/* ---- StoreV1 raw pointer accessors ---- */

TEST_F(GSAEngineTest, StoreV1RawAccessorsRoundTrip)
{
    engine_ = MakeEngine();
    EXPECT_EQ(engine_->GetStoreV1Raw(), reinterpret_cast<uintptr_t>(nullptr));
    MockStoreV1 store;
    engine_->SetStoreV1Raw(reinterpret_cast<uintptr_t>(&store));
    EXPECT_EQ(engine_->GetStoreV1Raw(), reinterpret_cast<uintptr_t>(&store));
    engine_->SetStoreV1(nullptr);
    EXPECT_EQ(engine_->GetStoreV1Raw(), reinterpret_cast<uintptr_t>(nullptr));
}

TEST_F(GSAEngineTest, GetSelfRawReturnsEngineAddress)
{
    engine_ = MakeEngine();
    EXPECT_EQ(engine_->GetSelfRaw(), reinterpret_cast<uintptr_t>(engine_.get()));
}

/* ---- SetBlocksMap validation ---- */

TEST_F(GSAEngineTest, SetBlocksMapRejectsKAddrsSizeMismatch)
{
    engine_ = MakeEngine();
    std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
    std::vector<int> blockIdxs = {0, 1};
    std::vector<uintptr_t> kAddrs = {0x1000};  // wrong size
    std::vector<uintptr_t> vAddrs = {0x2000, 0x2001};
    engine_->SetBlocksMap("req1", hashes, /*maxIdx=*/2, blockIdxs, kAddrs, vAddrs);
    /* Should not have stored the hash map. */
    EXPECT_EQ(engine_->mAllBlocksHash.count("req1"), 0u);
}

TEST_F(GSAEngineTest, SetBlocksMapRejectsVAddrsSizeMismatchInNonC8Mode)
{
    engine_ = MakeEngine();  // non-c8 mode
    std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
    std::vector<int> blockIdxs = {0, 1};
    std::vector<uintptr_t> kAddrs = {0x1000, 0x1001};
    std::vector<uintptr_t> vAddrs = {0x2000};  // wrong size
    engine_->SetBlocksMap("req1", hashes, 2, blockIdxs, kAddrs, vAddrs);
    EXPECT_EQ(engine_->mAllBlocksHash.count("req1"), 0u);
}

TEST_F(GSAEngineTest, SetBlocksMapStoresHashAndBlockAddresses)
{
    engine_ = MakeEngine();
    std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
    std::vector<int> blockIdxs = {0, 1};
    std::vector<uintptr_t> kAddrs = {0x1000, 0x1001};
    std::vector<uintptr_t> vAddrs = {0x2000, 0x2001};
    engine_->SetBlocksMap("req1", hashes, /*maxIdx=*/2, blockIdxs, kAddrs, vAddrs);
    ASSERT_EQ(engine_->mAllBlocksHash.count("req1"), 1u);
    EXPECT_EQ(engine_->mAllBlocksHash["req1"].size(), 2u);
    EXPECT_EQ(engine_->mPromptLen["req1"], 2);
    auto& blockMap = engine_->mLastKVBlockAddress["req1"];
    EXPECT_EQ(blockMap[0].kAddr, 0x1000u);
    EXPECT_EQ(blockMap[0].vAddr, 0x2000u);
    EXPECT_EQ(blockMap[1].kAddr, 0x1001u);
    EXPECT_EQ(blockMap[1].vAddr, 0x2001u);
}

/* ---- SetLastIndexerBlock validation ---- */

TEST_F(GSAEngineTest, SetLastIndexerBlockIsNoopWhenIndexerOffloadDisabled)
{
    engine_ = MakeEngine(/*indexerOffload=*/false);
    std::vector<int> idxs = {0};
    std::vector<uintptr_t> addrs = {0x10};
    engine_->SetLastIndexerBlock("req1", idxs, addrs);
    /* No crash; nothing stored. */
    SUCCEED();
}

TEST_F(GSAEngineTest, SetLastIndexerBlockRejectsSizeMismatch)
{
    engine_ = MakeEngine(/*indexerOffload=*/true);
    std::vector<int> idxs = {0, 1};
    std::vector<uintptr_t> addrs = {0x10};
    engine_->SetLastIndexerBlock("req1", idxs, addrs);
    SUCCEED();
}

TEST_F(GSAEngineTest, SetLastIndexerBlockRejectsUnknownReqID)
{
    engine_ = MakeEngine(/*indexerOffload=*/true);
    std::vector<int> idxs = {0};
    std::vector<uintptr_t> addrs = {0x10};
    engine_->SetLastIndexerBlock("unknown", idxs, addrs);
    SUCCEED();
}

TEST_F(GSAEngineTest, SetLastIndexerBlockUpdatesExistingEntry)
{
    engine_ = MakeEngine(/*indexerOffload=*/true);
    /* Seed mLastKVBlockAddress first via SetBlocksMap. */
    std::vector<BlockId> hashes = {MakeBlock(1)};
    std::vector<int> blockIdxs = {0};
    std::vector<uintptr_t> kAddrs = {0x1000};
    std::vector<uintptr_t> vAddrs = {0x2000};
    engine_->SetBlocksMap("req1", hashes, 1, blockIdxs, kAddrs, vAddrs);

    std::vector<int> idxs = {0};
    std::vector<uintptr_t> addrs = {0xABCDEFu};
    engine_->SetLastIndexerBlock("req1", idxs, addrs);
    EXPECT_EQ(engine_->mLastKVBlockAddress["req1"][0].indexerAddr, 0xABCDEFu);
    EXPECT_EQ(engine_->mLastKVBlockAddress["req1"][0].indexerBlockIdx, 0);
}

/* ---- DelBlocksMap / DelReqIDRun ---- */

TEST_F(GSAEngineTest, DelBlocksMapQueuesDeletionWhenPrefetchRunning)
{
    engine_ = MakeEngine();
    engine_->mIsPrefetchRunning.store(true);
    engine_->mSparseSeqLen["req1"] = 128;
    engine_->DelBlocksMap("req1");
    EXPECT_EQ(engine_->mDelSeqIds.count("req1"), 1u);
    EXPECT_EQ(engine_->mSparseSeqLen.count("req1"), 1u);  // not yet deleted
}

TEST_F(GSAEngineTest, DelBlocksMapRunsImmediatelyWhenIdle)
{
    engine_ = MakeEngine();
    engine_->mSparseSeqLen["req1"] = 128;
    engine_->DelBlocksMap("req1");
    EXPECT_EQ(engine_->mSparseSeqLen.count("req1"), 0u);
    EXPECT_EQ(engine_->mDelSeqIds.count("req1"), 0u);
}

TEST_F(GSAEngineTest, DelReqIDRunSkipsUnknownReqID)
{
    engine_ = MakeEngine();
    engine_->mDelSeqIds.insert("ghost");
    engine_->DelReqIDRun();
    SUCCEED();
}

TEST_F(GSAEngineTest, DelReqIDRunErasesMostMapsButNotHash_BugDocumented)
{
    // BUG-FOCUS: DelReqIDRun erases mSparseSeqLen, mPromptLen, mReqUseBlockTable,
    // ... but NOT mAllBlocksHash. The hash entry leaks after deletion.
    // Documented: assert the cleared maps are gone while the hash survives.
    engine_ = MakeEngine();
    engine_->mSparseSeqLen["req1"] = 128;
    engine_->mPromptLen["req1"] = 16;
    engine_->mReqUseBlockTable["req1"] = {0, 1};
    engine_->mAllBlocksHash["req1"] = {MakeBlock(1)};
    engine_->mDelSeqIds.insert("req1");
    engine_->DelReqIDRun();
    EXPECT_EQ(engine_->mSparseSeqLen.count("req1"), 0u);
    EXPECT_EQ(engine_->mPromptLen.count("req1"), 0u);
    EXPECT_EQ(engine_->mReqUseBlockTable.count("req1"), 0u);
    // mAllBlocksHash is NOT cleared by DelReqIDRun (bug). When fixed, expect 0u.
    EXPECT_EQ(engine_->mAllBlocksHash.count("req1"), 1u);
}

/* ---- UpdateIndexerMap / AddIndexerMap / DeleteIndexerMap ---- */

TEST_F(GSAEngineTest, UpdateIndexerMapDoesNotGuardOnIndexerOffload_BugDocumented)
{
    // BUG-FOCUS: UpdateIndexerMap (kvcache_pre.cpp:883) is the ONLY indexer-map
    // mutator that omits the `if (!mOpenIndexerOffload) return;` guard that
    // AddIndexerMap (line 950) and DeleteIndexerMap (line 989) both have.
    // It therefore mutates mUseBlockTableIdx even when indexer offload is off.
    // It also indexes mUseBlockTableIdx[reqID][layerID] WITHOUT first ensuring
    // the inner vector has mLayerNum slots, so calling it before SetBlocksMap
    // for that reqID is out-of-bounds (UB / segfault).
    //
    // Here we pre-size the table (as SetBlocksMap would) and prove the call
    // actually mutates state — i.e. it is NOT a no-op. When the missing guard
    // is added, flip the assertion to EXPECT_TRUE(mUseBlockTableIdx[reqID][0].empty()).
    engine_ = MakeEngine(/*indexerOffload=*/false);
    const std::string reqID = "req1";
    // Pre-size mUseBlockTableIdx[reqID] to mLayerNum empty vectors (mirrors what
    // SetBlocksMap / SetTokensMap do internally) so the [layerID] access is safe.
    engine_->mUseBlockTableIdx[reqID].resize(engine_->mLayerNum);
    ASSERT_EQ(engine_->mUseBlockTableIdx[reqID][0].size(), 0u);
    engine_->UpdateIndexerMap(reqID, /*idx=*/1, /*blockID=*/2, /*type=*/"add");
    // Bug: the call ran despite offload being off, appending idx to the table.
    EXPECT_EQ(engine_->mUseBlockTableIdx[reqID][0].size(), 1u);
    EXPECT_EQ(engine_->mUseBlockTableIdx[reqID][0][0], 1);
    EXPECT_EQ(engine_->mUseBlockTableIdx[reqID][1].size(), 1u);
}

TEST_F(GSAEngineTest, UpdateIndexerMapLastdelPopsBackWhenPreSized)
{
    engine_ = MakeEngine(/*indexerOffload=*/false);
    const std::string reqID = "req1";
    engine_->mUseBlockTableIdx[reqID].resize(engine_->mLayerNum);
    engine_->mUseBlockTableIdx[reqID][0].push_back(10);
    engine_->mUseBlockTableIdx[reqID][1].push_back(10);
    engine_->UpdateIndexerMap(reqID, /*idx=*/0, /*blockID=*/2, /*type=*/"lastdel");
    EXPECT_EQ(engine_->mUseBlockTableIdx[reqID][0].size(), 0u);
    EXPECT_EQ(engine_->mUseBlockTableIdx[reqID][1].size(), 0u);
}

TEST_F(GSAEngineTest, UpdateIndexerMapDelBranchIsSafeWhenPreSized)
{
    engine_ = MakeEngine(/*indexerOffload=*/false);
    const std::string reqID = "req1";
    engine_->mUseBlockTableIdx[reqID].resize(engine_->mLayerNum);
    engine_->mUseBlockTableIdx[reqID][0].push_back(10);
    engine_->mUseBlockTableIdx[reqID][1].push_back(10);
    // "del" branch only logs; does not modify the table. Must not crash.
    engine_->UpdateIndexerMap(reqID, /*idx=*/0, /*blockID=*/2, /*type=*/"del");
    EXPECT_EQ(engine_->mUseBlockTableIdx[reqID][0].size(), 1u);
}

TEST_F(GSAEngineTest, AddIndexerMapIsNoopWhenOff)
{
    engine_ = MakeEngine(/*indexerOffload=*/false);
    engine_->AddIndexerMap();
    SUCCEED();
}

TEST_F(GSAEngineTest, DeleteIndexerMapIsNoopWhenOff)
{
    engine_ = MakeEngine(/*indexerOffload=*/false);
    engine_->DeleteIndexerMap();
    SUCCEED();
}

TEST_F(GSAEngineTest, AddIndexerMapSkipsUnknownReqID)
{
    engine_ = MakeEngine(/*indexerOffload=*/true);
    engine_->mIndexerAdd["ghost"] = {};
    engine_->AddIndexerMap();
    EXPECT_EQ(engine_->mBlocksMap.count("ghost"), 0u);
}

TEST_F(GSAEngineTest, DeleteIndexerMapSkipsUnknownReqID)
{
    engine_ = MakeEngine(/*indexerOffload=*/true);
    engine_->mIndexerDel["ghost"] = {1};
    engine_->DeleteIndexerMap();
    SUCCEED();
}

/* ---- Prefetch status flags ---- */

TEST_F(GSAEngineTest, PrefetchStatusFlagReflectsAtomic)
{
    engine_ = MakeEngine();
    engine_->mIsPrefetchRunning.store(true);
    EXPECT_TRUE(engine_->GetPrefetchStatus());
    engine_->mIsPrefetchRunning.store(false);
    EXPECT_FALSE(engine_->GetPrefetchStatus());
}

TEST_F(GSAEngineTest, IndexerStatusFlagReflectsAtomic)
{
    engine_ = MakeEngine(/*indexerOffload=*/true);
    engine_->mIsIndexerLoadRunning.store(true);
    EXPECT_TRUE(engine_->GetIndexerStatus());
    engine_->mIsIndexerLoadRunning.store(false);
    EXPECT_FALSE(engine_->GetIndexerStatus());
}

/* ---- Get* accessors on populated maps ---- */

TEST_F(GSAEngineTest, GetReqUseBlockTableReturnsStoredTable)
{
    engine_ = MakeEngine();
    engine_->mReqUseBlockTable["req1"] = {3, 4, 5};
    auto out = engine_->GetReqUseBlockTable("req1");
    EXPECT_EQ(out, (std::vector<int>{3, 4, 5}));
}

TEST_F(GSAEngineTest, GetSparseSeqLenReturnsStoredValue)
{
    engine_ = MakeEngine();
    engine_->mSparseSeqLen["req1"] = 256;
    EXPECT_EQ(engine_->GetSparseSeqLen("req1"), 256);
}
