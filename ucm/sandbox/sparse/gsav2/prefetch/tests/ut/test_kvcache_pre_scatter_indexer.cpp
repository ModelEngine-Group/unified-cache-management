/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the indexer-load chain and host-buffer scatter paths of
 * GSAPrefetchEngineC implemented in kvcache_pre_scatter.cpp:
 *   LoadIndexerLayer / LoadIndexerRequest / BuildIndexerLoadTask /
 *   CopyNonUCMIndexerBlock / LoadIndexerTask / UCMIndexerLoadAsync /
 *   IndexerLoadProcess / RunOneBsPrefetch / ValidateAndPublishHbmSlotIndex /
 *   PreSwapTokens / CopyPreSwapTokens.
 *
 * These paths are pure host logic or only depend on the MockStoreV1 + the ACL
 * test stub (aclrtMemcpy / aclrtSetCurrentContext), so they run safely under
 * qemu without the TaskManager worker pool.
 */
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_task_manger.h"
#include "kvcache_trans_ascend.h"

using ::TaskInfo;
using ::ThreadInfo;
using ucmprefetch::BlockId;
using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class ScatterIndexerTest : public ::testing::Test {
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
        return e;
    }

    static BlockId MakeBlockId(std::uint8_t v)
    {
        BlockId b{};
        b.fill(static_cast<std::byte>(v));
        return b;
    }

    /* Seed mIndexerPtr / mIndexerSizeBytesBlock as SetKvTensorIndex would. */
    void SeedIndexer(int layerNum)
    {
        engine->mIndexerSizeBytesBlock = 16;  // 4 tokens * 4 bytes
        engine->mIndexerPtr.assign(layerNum, 0);
        for (int l = 0; l < layerNum; ++l) {
            auto* buf = new std::uint8_t[256]();
            engine->mIndexerPtr[l] = reinterpret_cast<uint64_t>(buf);
        }
    }

    void ReleaseIndexer(int layerNum)
    {
        for (int l = 0; l < layerNum; ++l) {
            if (engine->mIndexerPtr[l] != 0) {
                delete[] reinterpret_cast<std::uint8_t*>(engine->mIndexerPtr[l]);
                engine->mIndexerPtr[l] = 0;
            }
        }
    }
};

}  // namespace

/* ---------------- LoadIndexerLayer ---------------- */

TEST_F(ScatterIndexerTest, LoadIndexerLayerReturnsFalseWhenIndexerBaseNull)
{
    engine = MakeEngine(/*layerNum=*/1);
    // mIndexerPtr[0] stays 0 (no SetKvTensorIndex) -> early return false.
    size_t total = 0;
    EXPECT_FALSE(engine->LoadIndexerLayer(/*layerID=*/0, total));
}

TEST_F(ScatterIndexerTest, LoadIndexerLayerWithNoBlocksReturnsTrue)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(/*layerNum=*/1);
    // allNeedLoadBlock empty -> loop body never runs -> returns true.
    size_t total = 0;
    EXPECT_TRUE(engine->LoadIndexerLayer(0, total));
    EXPECT_EQ(total, 0u);
    ReleaseIndexer(1);
}

TEST_F(ScatterIndexerTest, LoadIndexerLayerDrivesLoadIndexerRequest)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(/*layerNum=*/1);
    const std::string reqID = "req1";
    // Seed one block to load with its hash; miss idxs must match size.
    engine->allNeedLoadBlock[reqID] = {{5}};  // layer 0 -> [5]
    engine->allMissIdxs[reqID] = {{0}};       // layer 0 -> [0]
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1), MakeBlockId(2), MakeBlockId(3),
                                     MakeBlockId(4), MakeBlockId(5), MakeBlockId(6)};
    size_t total = 0;
    EXPECT_TRUE(engine->LoadIndexerLayer(0, total));
    // task non-empty (1 UCM shard) -> LoadIndexerTask -> mock Load (Load+Wait path).
    ASSERT_EQ(store->calls.size(), 2u);  // Load + Wait
    EXPECT_EQ(store->calls[0].name, "Load");
    EXPECT_EQ(total, 1u);
    ReleaseIndexer(1);
}

/* ---------------- LoadIndexerRequest ---------------- */

TEST_F(ScatterIndexerTest, LoadIndexerRequestReturnsTrueWhenNoBlocks)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    std::uint8_t base[64] = {};
    size_t total = 0;
    EXPECT_TRUE(engine->LoadIndexerRequest("req1", /*hbmBlockIDs=*/{}, 0, base, total));
    ReleaseIndexer(1);
}

TEST_F(ScatterIndexerTest, LoadIndexerRequestReturnsFalseWhenNoHash)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    std::uint8_t base[64] = {};
    size_t total = 0;
    EXPECT_FALSE(engine->LoadIndexerRequest("req1", {1}, 0, base, total));
    ReleaseIndexer(1);
}

TEST_F(ScatterIndexerTest, LoadIndexerRequestReturnsFalseWhenNoMissIdxs)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    engine->mAllBlocksHash["req1"] = {MakeBlockId(1)};
    std::uint8_t base[64] = {};
    size_t total = 0;
    EXPECT_FALSE(engine->LoadIndexerRequest("req1", {1}, 0, base, total));
    ReleaseIndexer(1);
}

TEST_F(ScatterIndexerTest, LoadIndexerRequestReturnsFalseOnSizeMismatch)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    engine->mAllBlocksHash["req1"] = {MakeBlockId(1)};
    engine->allMissIdxs["req1"] = {
        {0, 1}
    };  // 2 misses vs 1 hbm block
    std::uint8_t base[64] = {};
    size_t total = 0;
    EXPECT_FALSE(engine->LoadIndexerRequest("req1", {1}, 0, base, total));
    ReleaseIndexer(1);
}

TEST_F(ScatterIndexerTest, LoadIndexerRequestReturnsTrueWhenTaskEmpty)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    const std::string reqID = "req1";
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1)};
    engine->allMissIdxs[reqID] = {{0}};
    // missBlockIdx 0 < hashVec.size() 1 -> builds a shard -> task non-empty.
    std::uint8_t base[64] = {};
    size_t total = 0;
    EXPECT_TRUE(engine->LoadIndexerRequest(reqID, {0}, 0, base, total));
    ASSERT_EQ(store->calls.size(), 2u);  // Load + Wait
    ReleaseIndexer(1);
}

/* ---------------- BuildIndexerLoadTask ---------------- */

TEST_F(ScatterIndexerTest, BuildIndexerLoadTaskBuildsShardsForUcmBlocks)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    const std::string reqID = "req1";
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1), MakeBlockId(2)};
    std::uint8_t base[64] = {};
    UC::Detail::TaskDesc task;
    size_t total = 0;
    EXPECT_TRUE(engine->BuildIndexerLoadTask(reqID, /*layerID=*/0, base,
                                             /*hbmBlockIDs=*/{0, 1}, /*missBlockIdxs=*/{0, 1},
                                             engine->mAllBlocksHash[reqID], task, total));
    EXPECT_EQ(task.size(), 2u);
    ReleaseIndexer(1);
}

TEST_F(ScatterIndexerTest, BuildIndexerLoadTaskSkipsOutOfRangeMissIdx)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    const std::string reqID = "req1";
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1)};  // only 1 hash entry
    std::uint8_t base[64] = {};
    UC::Detail::TaskDesc task;
    size_t total = 0;
    // missBlockIdx 5 >= hashVec.size() 1, and 5 >= lastUCMBlockIdx 1 -> non-UCM.
    // CopyNonUCMIndexerBlock needs mLastKVBlockAddress[reqID][5] -> not found -> false.
    EXPECT_FALSE(engine->BuildIndexerLoadTask(reqID, 0, base, {0}, {5},
                                              engine->mAllBlocksHash[reqID], task, total));
    ReleaseIndexer(1);
}

TEST_F(ScatterIndexerTest, BuildIndexerLoadTaskHandlesNegativeMissIdx)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(1);
    const std::string reqID = "req1";
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1)};
    std::uint8_t base[64] = {};
    UC::Detail::TaskDesc task;
    size_t total = 0;
    // missBlockIdx -1: -1 < 0 -> out-of-range log + continue (no shard, no abort).
    EXPECT_TRUE(engine->BuildIndexerLoadTask(reqID, 0, base, {0}, {-1},
                                             engine->mAllBlocksHash[reqID], task, total));
    EXPECT_TRUE(task.empty());
    ReleaseIndexer(1);
}

/* ---------------- CopyNonUCMIndexerBlock ---------------- */

TEST_F(ScatterIndexerTest, CopyNonUCMIndexerBlockReturnsFalseWhenNoMap)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    std::uint8_t dst[16] = {};
    size_t total = 0;
    EXPECT_FALSE(engine->CopyNonUCMIndexerBlock("req1", 0, 2, 2, static_cast<void*>(dst), total));
}

TEST_F(ScatterIndexerTest, CopyNonUCMIndexerBlockReturnsFalseWhenBlockIdxMissing)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    engine->mLastKVBlockAddress["req1"][5].indexerAddr =
        reinterpret_cast<uintptr_t>(new std::uint8_t[16]());
    std::uint8_t dst[16] = {};
    size_t total = 0;
    EXPECT_FALSE(engine->CopyNonUCMIndexerBlock("req1", 0, 2, 2, static_cast<void*>(dst), total));
}

TEST_F(ScatterIndexerTest, CopyNonUCMIndexerBlockSucceedsAndIncrementsTotal)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    auto* src = new std::uint8_t[32]();
    for (int i = 0; i < 16; ++i) { src[i] = static_cast<std::uint8_t>(0xCD); }
    engine->mLastKVBlockAddress["req1"][2].indexerAddr = reinterpret_cast<uintptr_t>(src);
    std::uint8_t dst[16] = {};
    size_t total = 0;
    EXPECT_TRUE(engine->CopyNonUCMIndexerBlock("req1", /*layerID=*/0,
                                               /*missBlockIdx=*/2, /*hbmBlockID=*/2,
                                               static_cast<void*>(dst), total));
    EXPECT_EQ(total, 1u);
    for (int i = 0; i < 16; ++i) { EXPECT_EQ(dst[i], 0xCD); }
    delete[] src;
}

TEST_F(ScatterIndexerTest, CopyNonUCMIndexerBlockReturnsFalseOnMemcpyFailure)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mIndexerSizeBytesBlock = 16;
    acl_stub::Stub().memcpyFails = true;
    auto* src = new std::uint8_t[16]();
    engine->mLastKVBlockAddress["req1"][2].indexerAddr = reinterpret_cast<uintptr_t>(src);
    std::uint8_t dst[16] = {};
    size_t total = 0;
    EXPECT_FALSE(engine->CopyNonUCMIndexerBlock("req1", 0, 2, 2, static_cast<void*>(dst), total));
    delete[] src;
}

/* ---------------- LoadIndexerTask ---------------- */

TEST_F(ScatterIndexerTest, LoadIndexerTaskSucceedsAndAccumulatesTotal)
{
    engine = MakeEngine(/*layerNum=*/1);
    UC::Detail::TaskDesc task;
    task.resize(2);
    size_t total = 0;
    EXPECT_TRUE(engine->LoadIndexerTask("req1", 0, task, total));
    EXPECT_EQ(total, 2u);
    ASSERT_EQ(store->calls.size(), 2u);  // Load + Wait
}

TEST_F(ScatterIndexerTest, LoadIndexerTaskReturnsFalseWhenStoreFails)
{
    engine = MakeEngine(/*layerNum=*/1);
    store->loadFails = true;
    UC::Detail::TaskDesc task;
    task.resize(1);
    size_t total = 0;
    EXPECT_FALSE(engine->LoadIndexerTask("req1", 0, task, total));
}

/* ---------------- UCMIndexerLoadAsync ---------------- */

TEST_F(ScatterIndexerTest, UCMIndexerLoadAsyncReturnsWhenContextNull)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->UCMIndexerLoadAsync(/*context=*/nullptr, /*stream=*/nullptr);
    // No store calls, no crash.
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(ScatterIndexerTest, UCMIndexerLoadAsyncReturnsWhenSetContextFails)
{
    engine = MakeEngine(/*layerNum=*/1);
    acl_stub::Stub().setCurrentContextFails = true;
    // Use a malloc'd context pointer (engine.mContext is valid already, but
    // pass our own to avoid touching the engine's runtime).
    void* ctx = std::malloc(sizeof(int));
    engine->UCMIndexerLoadAsync(ctx, nullptr);
    std::free(ctx);
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(ScatterIndexerTest, UCMIndexerLoadAsyncDrivesLoadIndexerLayer)
{
    engine = MakeEngine(/*layerNum=*/1);
    SeedIndexer(/*layerNum=*/1);
    const std::string reqID = "req1";
    engine->allNeedLoadBlock[reqID] = {{0}};  // layer 0 -> block 0
    engine->allMissIdxs[reqID] = {{0}};
    engine->mAllBlocksHash[reqID] = {MakeBlockId(1)};
    // Use the engine's own mContext (set up by InitAclRuntime).
    ASSERT_NE(engine->mContext, nullptr);
    engine->UCMIndexerLoadAsync(engine->mContext, /*stream=*/nullptr);
    ASSERT_EQ(store->calls.size(), 2u);  // Load + Wait
    EXPECT_EQ(store->calls[0].name, "Load");
    ReleaseIndexer(1);
}

/* ---------------- IndexerLoadProcess ---------------- */

TEST_F(ScatterIndexerTest, IndexerLoadProcessReturnsEarlyWhenOffloadOff)
{
    engine = MakeEngine(/*layerNum=*/1, /*indexerOffload=*/0);
    TaskInfo taskInfo{};
    ThreadInfo threadInfo{};
    EXPECT_EQ(engine->IndexerLoadProcess(taskInfo, threadInfo), 0);
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(ScatterIndexerTest, IndexerLoadProcessSkipsUnknownReqID)
{
    engine = MakeEngine(/*layerNum=*/1);
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"ghost"};  // not in mDocsTables -> skip + continue
    taskInfo.topkLenList = {4};
    ThreadInfo threadInfo{};
    EXPECT_EQ(engine->IndexerLoadProcess(taskInfo, threadInfo), 0);
    EXPECT_TRUE(store->calls.empty());
}

TEST_F(ScatterIndexerTest, IndexerLoadProcessSkipsZeroTopkLen)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mDocsTables["req1"].assign(1, {});
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"req1"};
    taskInfo.topkLenList = {0};  // topkLen <= 0 -> skip
    ThreadInfo threadInfo{};
    EXPECT_EQ(engine->IndexerLoadProcess(taskInfo, threadInfo), 0);
    EXPECT_TRUE(store->calls.empty());
}

/* ---------------- RunOneBsPrefetch ---------------- */

TEST_F(ScatterIndexerTest, RunOneBsPrefetchEndToEndWithHits)
{
    engine = MakeEngine(/*layerNum=*/2);
    const std::string reqID = "req1";
    // Seed docs/blocks tables: item 0->block 10, item 1->block 20.
    for (int l = 0; l < 2; ++l) {
        engine->mDocsTables[reqID].push_back({
            {0, 10},
            {1, 20}
        });
        engine->mBlocksMap[reqID].push_back({
            {10, 0},
            {20, 1}
        });
        engine->mPrefetchIdx[reqID].push_back({0, 1});
        engine->allNeedLoadBlock[reqID].push_back({});
        engine->allMissIdxs[reqID].push_back({});
        engine->mPrefetchBlockTableIdx[reqID].push_back(std::vector<int>(4, -1));
    }
    engine->mIndexerSlot[reqID].assign(2, 0);
    engine->mIndexerLoadResultT[reqID] = torch::full({2, 4}, -1, torch::dtype(torch::kInt32));
    // topkList: [layer 0] = [0,1], [layer 1] = [0,1] (both hits).
    auto topkList = torch::tensor({0, 1, 0, 1}, torch::dtype(torch::kInt32)).reshape({2, 1, 2});
    engine->RunOneBsPrefetch(reqID, /*topkLen=*/2, /*topkIndex=*/0, topkList);
    // All hits -> no loads, mPrefetchIdx cleared then refilled (empty since all hit).
    EXPECT_TRUE(engine->mPrefetchIdx[reqID][0].empty());
    EXPECT_TRUE(store->calls.empty());
}

/* ---------------- ValidateAndPublishHbmSlotIndex ---------------- */

TEST_F(ScatterIndexerTest, ValidateAndPublishHbmSlotIndexPublishesOnSuccess)
{
    engine = MakeEngine(/*layerNum=*/1);
    // mLoadIndexHost is a single Tensor indexed by [streamIdx]; make it 2D.
    engine->mLoadIndexHost = torch::zeros({1, 16}, torch::dtype(torch::kInt64));
    engine->mLoadNum.assign(1, 2);
    engine->mLoadIndexBufferLen = 16;
    std::vector<uint64_t> hbmSlotIndex = {100ull, 200ull};
    engine->ValidateAndPublishHbmSlotIndex(/*streamIdx=*/0, /*layerID=*/0, hbmSlotIndex);
    // Verify the memcpy published the slot indices into mLoadIndexHost.
    auto accessor = engine->mLoadIndexHost.accessor<int64_t, 2>();
    EXPECT_EQ(accessor[0][0], 100);
    EXPECT_EQ(accessor[0][1], 200);
}

/* ---------------- PreSwapTokens / CopyPreSwapTokens ---------------- */

TEST_F(ScatterIndexerTest, PreSwapTokensCopiesAppendedTokensToSpaBuffer)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mTopkIndexerNum = 1;
    const int sparseLen = 4;
    const int spaMaxBs = 4;
    const int mtpToken = 1;
    // useToken buffer: sparseLen * mTopkIndexerNum ints = 4 ints, values 0..3.
    std::vector<int32_t> useBuf(sparseLen * engine->mTopkIndexerNum);
    std::iota(useBuf.begin(), useBuf.end(), 0);
    // spaToken buffer: spaMaxBs * sparseLen ints (big enough).
    std::vector<int32_t> spaBuf(spaMaxBs * sparseLen, -1);
    std::vector<uintptr_t> useTokenPtrs = {reinterpret_cast<uintptr_t>(useBuf.data())};
    std::vector<int32_t> seqLens = {static_cast<int32_t>(sparseLen)};
    std::vector<std::vector<int>> swapIndex = {
        {0, 1}
    };  // swapRange [0,1)
    std::vector<uintptr_t> spaTokensPtr = {reinterpret_cast<uintptr_t>(spaBuf.data())};
    std::vector<std::string> reqIDs = {"req1"};
    engine->PreSwapTokens(useTokenPtrs, seqLens, swapIndex, spaTokensPtr, reqIDs, sparseLen,
                          spaMaxBs, mtpToken);
    // After PreSwapTokens: spaBuf[0..sparseLen) == useBuf (copied), then the
    // appended tail overwrites the last (changeLen) slots with new ids.
    // maxID = 3, start = 4, end = sparseLen(4) + mtpToken(1) - 1 = 4,
    // change_len = end - start = 0 -> clamped to mtpToken = 1.
    // needAppend = {4}. swapRange [0,1) -> index 0: copy useBuf to spaBuf[0],
    //   then write needAppend at offset sparseLen - changeLen + mtpToken - 1 - 0
    //   = 4 - 1 + 1 - 1 - 0 = 3 -> spaBuf[3] = 4.
    EXPECT_EQ(spaBuf[0], useBuf[0]);
    EXPECT_EQ(spaBuf[1], useBuf[1]);
    EXPECT_EQ(spaBuf[2], useBuf[2]);
    EXPECT_EQ(spaBuf[3], 4);  // appended token id
}

TEST_F(ScatterIndexerTest, CopyPreSwapTokensCopiesRange)
{
    engine = MakeEngine(/*layerNum=*/1);
    engine->mTopkIndexerNum = 1;
    const int sparseLen = 4;
    const int spaMaxBs = 4;
    const int mtpToken = 1;
    std::vector<int32_t> useBuf = {10, 11, 12, 13};
    std::vector<int32_t> spaBuf(spaMaxBs * sparseLen, -1);
    std::vector<int> needAppend = {99};
    engine->CopyPreSwapTokens(reinterpret_cast<uintptr_t>(useBuf.data()),
                              reinterpret_cast<uintptr_t>(spaBuf.data()), /*swapRange=*/{0, 2},
                              needAppend, sparseLen, spaMaxBs, mtpToken, /*changeLen=*/1);
    // index 0: spaBuf[0..4) = useBuf, then needAppend at offset
    //   sparseLen - changeLen + mtpToken - 1 - 0%1 = 4-1+1-1-0 = 3 -> spaBuf[3] = 99.
    // index 1: spaCpuPtr at spaBuf[(0*4+1)*4] = spaBuf[4]; copy useBuf to spaBuf[4..8),
    //   then needAppend at spaBuf[4 + 3] = spaBuf[7] = 99.
    EXPECT_EQ(spaBuf[0], 10);
    EXPECT_EQ(spaBuf[3], 99);
    EXPECT_EQ(spaBuf[4], 10);
    EXPECT_EQ(spaBuf[5], 11);
    EXPECT_EQ(spaBuf[7], 99);
}
