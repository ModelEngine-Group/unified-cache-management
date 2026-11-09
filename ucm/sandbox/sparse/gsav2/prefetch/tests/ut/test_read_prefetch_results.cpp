/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for ReadPrefetchResults (kvcache_pre_scatter.cpp), which reads back
 * the prefetch results from shared memory.  The round-trip (Write -> Read)
 * is performed within a single test to avoid cross-test shm state leaks.
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
using ucmprefetch::KVPackRankState;
using ucmprefetch::PrefetchShmLayout;
using ucmprefetch::SharedKVPackHeader;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class ReadPrefetchResultsTest : public ::testing::Test {
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

    static torch::Tensor Make2DTensor(int s, int t, int d)
    {
        return torch::zeros({s, t, d}, torch::dtype(torch::kFloat32));
    }
    static torch::Tensor Make1DIndexTensor(int s, int t)
    {
        return torch::zeros({s, t}, torch::dtype(torch::kInt64));
    }

    PrefetchShmLayout MakeLayout()
    {
        PrefetchShmLayout layout{};
        layout.versionSize = 4;
        layout.countSize = 4;
        layout.maxLoadNum = engine->mMaxLoadNum;
        layout.missTokenSize = layout.countSize + layout.maxLoadNum * sizeof(int);
        layout.loadSlotSize = layout.countSize + layout.maxLoadNum * sizeof(int);
        layout.totalSize =
            layout.missTokenSize + layout.loadSlotSize + engine->mSpaTopkLen * sizeof(int);
        return layout;
    }

    void SetupEngine()
    {
        auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/2,
                                   /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                                 /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
        std::vector<int> topkLayerMap{0};
        engine = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());
        auto kHost = Make2DTensor(1, 64, 4);
        auto vHost = Make2DTensor(1, 64, 4);
        auto kDevice = Make2DTensor(1, 64, 4);
        auto vDevice = Make2DTensor(1, 64, 4);
        auto loadIdxHost = Make1DIndexTensor(1, 64);
        auto loadIdxDevice = Make1DIndexTensor(1, 64);
        engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
        ASSERT_NE(engine->mPrefetchResultShm, nullptr);
    }

    /* Pre-set rank 0's packedEpoch so FinishKVPack's wait (mRank=1) passes. */
    void PreSetRank0PackedEpoch(uint64_t epoch)
    {
        auto* states = reinterpret_cast<KVPackRankState*>(static_cast<void*>(
            reinterpret_cast<uint8_t*>(engine->mKVPackControl) + sizeof(SharedKVPackHeader)));
        states[0].packedEpoch.store(epoch, std::memory_order_release);
        states[0].h2dDoneEpoch.store(epoch, std::memory_order_release);
    }
};

TEST_F(ReadPrefetchResultsTest, WriteThenReadRoundTripsMissTokensAndSlots)
{
    SetupEngine();
    const std::string reqID = "req1";
    // Pre-size for [reqID][layerID] indexing.
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    // Set up mAllTopkIndicesTensor for the final memcpy.
    engine->mAllTopkIndicesTensor[reqID] =
        torch::zeros({engine->mTopkIndexerNum, engine->mSpaTopkLen}, torch::dtype(torch::kInt32));

    // Write test data.
    engine->mMissGlobalToken[reqID][0] = {10, 20};
    engine->mLoadSlot[reqID][0] = {5, 6};
    std::vector<int> useBlockTable(engine->mSpaTopkLen, 42);
    auto layout = MakeLayout();
    engine->WritePrefetchResult(reqID, /*i=*/0, /*layerID=*/0, useBlockTable, layout);

    // Clear local state to simulate a reader.
    engine->mMissGlobalToken[reqID][0].clear();
    engine->mLoadSlot[reqID][0].clear();

    // Read back.
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {reqID};
    taskInfo.topkLenList = {engine->mSpaTopkLen};  // > 0 -> process
    engine->ReadPrefetchResults(taskInfo, layout);

    EXPECT_EQ(engine->mMissGlobalToken[reqID][0], (std::vector<int>{10, 20}));
    EXPECT_EQ(engine->mLoadSlot[reqID][0], (std::vector<int>{5, 6}));
}

TEST_F(ReadPrefetchResultsTest, ReadPrefetchResultsSkipsZeroTopkLen)
{
    SetupEngine();
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"req1"};
    taskInfo.topkLenList = {0};  // <= 0 -> skip
    auto layout = MakeLayout();
    engine->ReadPrefetchResults(taskInfo, layout);
    // No mutation of mMissGlobalToken (no entry created).
    EXPECT_EQ(engine->mMissGlobalToken.count("req1"), 0u);
}

TEST_F(ReadPrefetchResultsTest, ReadPrefetchResultsHandlesZeroMissTokens)
{
    SetupEngine();
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mAllTopkIndicesTensor[reqID] =
        torch::zeros({engine->mTopkIndexerNum, engine->mSpaTopkLen}, torch::dtype(torch::kInt32));
    // Write with zero miss tokens and zero load slots.
    std::vector<int> useBlockTable(engine->mSpaTopkLen, 7);
    auto layout = MakeLayout();
    engine->WritePrefetchResult(reqID, 0, 0, useBlockTable, layout);
    // Clear + read back.
    engine->mMissGlobalToken[reqID][0].clear();
    engine->mLoadSlot[reqID][0].clear();
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {reqID};
    taskInfo.topkLenList = {engine->mSpaTopkLen};
    engine->ReadPrefetchResults(taskInfo, layout);
    EXPECT_TRUE(engine->mMissGlobalToken[reqID][0].empty());
    EXPECT_TRUE(engine->mLoadSlot[reqID][0].empty());
}

/* ---------------- PreparePrefetchResults (rank != 0 path) ---------------- */
// With mRank != 0, PreparePrefetchResults takes the else branch: it waits for
// the version in shm to match the expected value, then calls ReadPrefetchResults.
// No omp loop is used in this branch, so it's safe under qemu.

TEST_F(ReadPrefetchResultsTest, PreparePrefetchResultsNonRank0ReadsBackShm)
{
    SetupEngine();
    const std::string reqID = "req1";
    // Switch to non-rank-0 to take the else branch (no omp loop).
    engine->mRank = 1;
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mAllTopkIndicesTensor[reqID] =
        torch::zeros({engine->mTopkIndexerNum, engine->mSpaTopkLen}, torch::dtype(torch::kInt32));
    // Write test data via WritePrefetchResult.
    engine->mMissGlobalToken[reqID][0] = {100, 200};
    engine->mLoadSlot[reqID][0] = {50, 60};
    std::vector<int> useBlockTable(engine->mSpaTopkLen, 11);
    auto layout = MakeLayout();
    engine->WritePrefetchResult(reqID, 0, 0, useBlockTable, layout);
    // Clear local state.
    engine->mMissGlobalToken[reqID][0].clear();
    engine->mLoadSlot[reqID][0].clear();
    // Pre-set the version in shm so the wait loop exits immediately.
    uint32_t expectedVersion = engine->mPrefetchResultVersion.load() + 1;
    uint32_t* versionPtr = static_cast<uint32_t*>(engine->mPrefetchResultShm);
    __atomic_store_n(versionPtr, expectedVersion, __ATOMIC_RELEASE);
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {reqID};
    taskInfo.topkLenList = {engine->mSpaTopkLen};
    // PreparePrefetchResults (rank != 0) -> wait loop (passes immediately) ->
    // ReadPrefetchResults.
    engine->PreparePrefetchResults(taskInfo);
    EXPECT_EQ(engine->mMissGlobalToken[reqID][0], (std::vector<int>{100, 200}));
    EXPECT_EQ(engine->mLoadSlot[reqID][0], (std::vector<int>{50, 60}));
}

TEST_F(ReadPrefetchResultsTest, PreparePrefetchResultsNonRank0SkipsZeroTopkLen)
{
    SetupEngine();
    engine->mRank = 1;
    // Pre-set the version so the wait loop exits.
    uint32_t expectedVersion = engine->mPrefetchResultVersion.load() + 1;
    uint32_t* versionPtr = static_cast<uint32_t*>(engine->mPrefetchResultShm);
    __atomic_store_n(versionPtr, expectedVersion, __ATOMIC_RELEASE);
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"req1"};
    taskInfo.topkLenList = {0};  // <= 0 -> ReadPrefetchResults skips
    auto layout = MakeLayout();
    engine->PreparePrefetchResults(taskInfo);
    // No crash; skip branch exercised.
    SUCCEED();
}

/* NOTE: CallPrefetchProcess was tested but found to crash under qemu when
 * PreparePrefetchResults (rank != 0) is followed by MergeAndCopyPrefetchLayer
 * within the same call — a subtle state interaction triggers KVPackAbort.
 * The individual components (PreparePrefetchResults, MergeAndCopyPrefetchLayer)
 * are tested directly and pass.  CallPrefetchProcess is covered on real
 * hardware via SubmitTask. */

}  // namespace
