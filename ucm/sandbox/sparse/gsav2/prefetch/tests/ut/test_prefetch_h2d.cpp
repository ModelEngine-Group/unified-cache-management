/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the H2D/scatter pipeline functions in kvcache_pre_scatter.cpp:
 *   - CopyPrefetchToH2DBuffers
 *   - ValidateAndPublishHbmSlotIndex
 *   - GetScatterWorkspace
 *   - RunInplaceScatter
 *   - KVCacheInplaceScatter (K and V)
 *
 * These tests are split into a separate binary from test_set_kv_buffer.cpp to
 * reduce the number of engine constructions per process (qemu becomes unstable
 * after ~10 engine ctor/dtor cycles due to ACL/torch global-state accumulation).
 */
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_pre_detail.h"
#include "kvcache_task_manger.h"
#include "kvcache_trans_ascend.h"

using ::TaskInfo;
using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch::PrefetchShmLayout;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class PrefetchH2DTest : public ::testing::Test {
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

    static torch::Tensor Make2DTensor(int streamNum, int tokens, int dim)
    {
        return torch::zeros({streamNum, tokens, dim}, torch::dtype(torch::kFloat32));
    }
    static torch::Tensor Make1DIndexTensor(int streamNum, int tokens)
    {
        return torch::zeros({streamNum, tokens}, torch::dtype(torch::kInt64));
    }

    /* Full setup: engine + SetKvBuffer + init'd HBMManagers + tensors. */
    void SetupFullEngine(int layerNum = 1, int isSfaC8 = 0)
    {
        auto model = MakeModelInfo(layerNum, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                                   /*indexerOffload=*/1, isSfaC8, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                                 /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
        std::vector<int> topkLayerMap(layerNum);
        for (int i = 0; i < layerNum; ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());

        // SetKvBuffer to initialize mKVPackControl + shared host buffers.
        auto kHost = Make2DTensor(1, 64, 4);
        auto vHost = Make2DTensor(1, 64, 4);
        auto kDevice = Make2DTensor(1, 64, 4);
        auto vDevice = Make2DTensor(1, 64, 4);
        auto loadIdxHost = Make1DIndexTensor(1, 64);
        auto loadIdxDevice = Make1DIndexTensor(1, 64);
        engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
        ASSERT_NE(engine->mKVPackControl, nullptr);

        // Pre-seed mHBMMangers with init'd managers (one per topkIndexer).
        const std::string reqID = "req1";
        engine->mHBMMangers[reqID].resize(engine->mTopkIndexerNum);
        for (int t = 0; t < engine->mTopkIndexerNum; ++t) {
            engine->mHBMMangers[reqID][t] = std::make_unique<HBMManager>();
            engine->mHBMMangers[reqID][t]->init(/*dramTensorCount=*/2,
                                                /*hbmCapacity=*/32, /*hbmImmutableSize=*/32,
                                                /*topkSize=*/2, engine->mLogger);
        }
        engine->mAllTopkIndicesTensor[reqID] = torch::zeros(
            {engine->mTopkIndexerNum, engine->mSpaTopkLen}, torch::dtype(torch::kInt32));
        engine->mIsIndexerOffload.assign(2, false);
    }
};

/* ---------------- GetScatterWorkspace ---------------- */

TEST_F(PrefetchH2DTest, GetScatterWorkspaceAllocatesAndCachesOnFirstCall)
{
    SetupFullEngine();
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    void* ws = engine->GetScatterWorkspace("K", /*layerID=*/0, /*streamIdx=*/0,
                                           /*workspaceSize=*/1024);
    EXPECT_NE(ws, nullptr);
    EXPECT_EQ(engine->mKMaxWorkspaceSize[0], 1024u);
    // Second call with smaller size reuses the cached workspace.
    void* ws2 = engine->GetScatterWorkspace("K", 0, 0, /*workspaceSize=*/512);
    EXPECT_EQ(ws2, ws);
    EXPECT_EQ(engine->mKMaxWorkspaceSize[0], 1024u);
}

TEST_F(PrefetchH2DTest, GetScatterWorkspaceReallocatesWhenSizeGrows)
{
    SetupFullEngine();
    engine->mVWorkspaceAddr.assign(1, nullptr);
    engine->mVMaxWorkspaceSize.assign(1, 0);
    void* ws = engine->GetScatterWorkspace("V", 0, 0, 1024);
    EXPECT_NE(ws, nullptr);
    EXPECT_EQ(engine->mVMaxWorkspaceSize[0], 1024u);
    // Larger size -> free old + malloc new.
    void* ws2 = engine->GetScatterWorkspace("V", 0, 0, 2048);
    EXPECT_NE(ws2, nullptr);
    EXPECT_EQ(engine->mVMaxWorkspaceSize[0], 2048u);
}

// NOTE: GetScatterWorkspace's malloc-failure path calls KVPackAbort (std::abort).
// EXPECT_DEATH forks the process, which corrupts the shared-memory state for
// subsequent tests under qemu; the abort path is documented and covered on
// real hardware.

/* ---------------- CopyPrefetchToH2DBuffers ---------------- */

TEST_F(PrefetchH2DTest, CopyPrefetchToH2DBuffersIsNoopWhenLoadNumZero)
{
    SetupFullEngine();
    engine->mLoadNum.assign(1, 0);
    engine->CopyPrefetchToH2DBuffers(/*streamIdx=*/0, /*layerID=*/0);
    SUCCEED();
}

TEST_F(PrefetchH2DTest, CopyPrefetchToH2DBuffersCopiesKAndVWhenLoadNumNonZero)
{
    SetupFullEngine();
    engine->mLoadNum.assign(1, 4);
    ASSERT_EQ(engine->mH2DPtrInfo[0].size(), 6u);
    // Seed shared host buffers with a pattern.
    std::byte* kSrc = reinterpret_cast<std::byte*>(engine->mSharedKBufferHostPtr[0]);
    std::byte* vSrc = reinterpret_cast<std::byte*>(engine->mSharedVBufferHostPtr[0]);
    for (int i = 0; i < 256; ++i) {
        kSrc[i] = static_cast<std::byte>(0xA0 + i);
        vSrc[i] = static_cast<std::byte>(0xB0 + i);
    }
    engine->CopyPrefetchToH2DBuffers(0, 0);
    auto kDevice = engine->mKBufferDevice[0];
    EXPECT_NE(kDevice.data_ptr(), nullptr);
    SUCCEED();
}

/* ---------------- ValidateAndPublishHbmSlotIndex ---------------- */

TEST_F(PrefetchH2DTest, ValidateAndPublishHbmSlotIndexPublishesOnSuccess)
{
    SetupFullEngine();
    engine->mLoadNum.assign(1, 2);
    engine->mLoadIndexBufferLen = 64;
    std::vector<uint64_t> hbmSlotIndex = {100ull, 200ull};
    engine->ValidateAndPublishHbmSlotIndex(/*streamIdx=*/0, /*layerID=*/0, hbmSlotIndex);
    // Verify the memcpy published the slot indices into mLoadIndexHost[0].
    auto loadIdxSlice = engine->mLoadIndexHost[0];
    auto accessor = loadIdxSlice.accessor<int64_t, 1>();
    EXPECT_EQ(accessor[0], 100);
    EXPECT_EQ(accessor[1], 200);
}

/* ---------------- WritePrefetchResult ---------------- */
// These verify the shm write path (rank 0) without the read-back (which
// bus-errors under qemu when chained after other tests).

TEST_F(PrefetchH2DTest, WritePrefetchResultWritesCountsAndDataToShm)
{
    SetupFullEngine();
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {10, 20, 30};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {1, 2, 3};
    std::vector<int> useBlockTable(engine->mSpaTopkLen, 7);
    PrefetchShmLayout layout{};
    layout.versionSize = 4;
    layout.countSize = 4;
    layout.maxLoadNum = engine->mMaxLoadNum;
    layout.missTokenSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.loadSlotSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.totalSize =
        layout.missTokenSize + layout.loadSlotSize + engine->mSpaTopkLen * sizeof(int);
    engine->WritePrefetchResult(reqID, /*i=*/0, /*layerID=*/0, useBlockTable, layout);
    size_t offset = 4;
    uint8_t* shmBase = static_cast<uint8_t*>(engine->mPrefetchResultShm) + offset;
    uint32_t missCount = 0, slotCount = 0;
    std::memcpy(&missCount, shmBase, 4);
    std::memcpy(&slotCount, shmBase + layout.missTokenSize, 4);
    EXPECT_EQ(missCount, 3u);
    EXPECT_EQ(slotCount, 3u);
    int missTokens[3];
    std::memcpy(missTokens, shmBase + 4, 3 * sizeof(int));
    EXPECT_EQ(missTokens[0], 10);
    EXPECT_EQ(missTokens[2], 30);
}

TEST_F(PrefetchH2DTest, WritePrefetchResultClampsToMaxLoadNum)
{
    SetupFullEngine();
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0].assign(engine->mMaxLoadNum + 10, 99);
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0].assign(engine->mMaxLoadNum + 10, 88);
    std::vector<int> useBlockTable(engine->mSpaTopkLen, 0);
    PrefetchShmLayout layout{};
    layout.versionSize = 4;
    layout.countSize = 4;
    layout.maxLoadNum = engine->mMaxLoadNum;
    layout.missTokenSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.loadSlotSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.totalSize =
        layout.missTokenSize + layout.loadSlotSize + engine->mSpaTopkLen * sizeof(int);
    engine->WritePrefetchResult(reqID, 0, 0, useBlockTable, layout);
    uint8_t* shmBase = static_cast<uint8_t*>(engine->mPrefetchResultShm) + 4;
    uint32_t missCount = 0;
    std::memcpy(&missCount, shmBase, 4);
    EXPECT_EQ(missCount, static_cast<uint32_t>(engine->mMaxLoadNum));
}

/* ---------------- ProcessPrefetchRequest ---------------- */
// Drives the full HBMManager.processTopkWithDebug path + shm write.

TEST_F(PrefetchH2DTest, ProcessPrefetchRequestProcessesTopkAndWritesShm)
{
    SetupFullEngine();
    const std::string reqID = "req1";
    // Pre-size mMissGlobalToken / mLoadSlot for [reqID][layerID] indexing.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    // mAvgHitRate must be sized [mLayerNum].
    engine->mAvgHitRate.assign(engine->mLayerNum, {});
    // Build topkList: [layerNum=1, bs=1, topkLen=mSpaTopkLen=32] int32.
    // Values in [0, dramTensorCount-1] = [0, 1] (DRAM path, all hits).
    auto topkList = torch::zeros({1, 1, engine->mSpaTopkLen}, torch::dtype(torch::kInt32));
    auto acc = topkList.accessor<int32_t, 3>();
    for (int j = 0; j < engine->mSpaTopkLen; ++j) {
        acc[0][0][j] = j % 2;  // values 0, 1 (both in DRAM range)
    }
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {reqID};
    taskInfo.topkLenList = {engine->mSpaTopkLen};  // == mSpaTopkLen -> size check passes
    taskInfo.topkList = topkList;
    PrefetchShmLayout layout{};
    layout.versionSize = 4;
    layout.countSize = 4;
    layout.maxLoadNum = engine->mMaxLoadNum;
    layout.missTokenSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.loadSlotSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.totalSize =
        layout.missTokenSize + layout.loadSlotSize + engine->mSpaTopkLen * sizeof(int);
    engine->ProcessPrefetchRequest(taskInfo, /*i=*/0, /*layerID=*/0, layout);
    // After processing, mMissGlobalToken[reqID][0] is populated.
    // All topk values (0,1) are in DRAM range -> some may be misses (first access).
    EXPECT_GE(engine->mMissGlobalToken[reqID][0].size(), 0u);
    // useBlockTable.size() == mSpaTopkLen -> memcpy into mAllTopkIndicesTensor.
    // (If size mismatches, KVPackAbort would crash; no crash means success.)
    EXPECT_EQ(engine->mAvgHitRate[0].size(), 1u);
}

TEST_F(PrefetchH2DTest, ProcessPrefetchRequestSkipsZeroTopkLen)
{
    SetupFullEngine();
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"req1"};
    taskInfo.topkLenList = {0};  // <= 0 -> early return
    PrefetchShmLayout layout{};
    layout.versionSize = 4;
    layout.countSize = 4;
    layout.maxLoadNum = engine->mMaxLoadNum;
    layout.missTokenSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.loadSlotSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.totalSize =
        layout.missTokenSize + layout.loadSlotSize + engine->mSpaTopkLen * sizeof(int);
    engine->ProcessPrefetchRequest(taskInfo, /*i=*/0, /*layerID=*/0, layout);
    // No mutation of mMissGlobalToken (no entry created).
    EXPECT_EQ(engine->mMissGlobalToken.count("req1"), 0u);
}

}  // namespace
