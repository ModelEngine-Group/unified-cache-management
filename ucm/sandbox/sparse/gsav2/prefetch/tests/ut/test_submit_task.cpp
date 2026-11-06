/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for SubmitTask / SubmitTaskIndexer (kvcache_pre_scatter.cpp), which submit
 * work to the TaskManager worker thread.  Uses mRank=1 (non-rank-0) to avoid
 * the omp parallel loop in PreparePrefetchResults (which crashes under qemu).
 * With empty reqIDList, the worker's CallPrefetchProcess does minimal work.
 */
#include <chrono>
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
using ucmprefetch::SharedKVPackHeader;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class SubmitTaskTest : public ::testing::Test {
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
        // Use rank 1 to avoid the omp loop in PreparePrefetchResults (rank 0).
        engine->mRank = 1;
        engine->mLoadNum.assign(engine->mLayerNum, 0);
        engine->mLoadIndexBufferLen = 64;
        // Pre-set version so PreparePrefetchResults' wait loop exits immediately.
        uint32_t expectedVersion = engine->mPrefetchResultVersion.load() + 1;
        uint32_t* versionPtr = static_cast<uint32_t*>(engine->mPrefetchResultShm);
        __atomic_store_n(versionPtr, expectedVersion, __ATOMIC_RELEASE);
        // Pre-set rank 0's packedEpoch/h2dDoneEpoch so FinishKVPack's wait passes.
        auto* states = reinterpret_cast<KVPackRankState*>(static_cast<void*>(
            reinterpret_cast<uint8_t*>(engine->mKVPackControl) + sizeof(SharedKVPackHeader)));
        states[0].packedEpoch.store(1, std::memory_order_release);
        states[0].h2dDoneEpoch.store(1, std::memory_order_release);
    }

    /* Wait for mIsPrefetchRunning to become false (worker thread finished). */
    void WaitForPrefetchDone(int timeoutMs = 5000)
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (engine->mIsPrefetchRunning.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void WaitForIndexerLoadDone(int timeoutMs = 5000)
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (engine->mIsIndexerLoadRunning.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

// NOTE: SubmitTaskWithEmptyReqListCompletesViaWorker was removed because the
// worker thread's CallPrefetchProcess (mRank=1) crashes under qemu — the same
// PreparePrefetchResults + MergeAndCopyPrefetchLayer interaction that affects
// the direct CallPrefetchProcess test.  SubmitTask is covered on real hardware.

TEST_F(SubmitTaskTest, SubmitTaskIndexerReturnsEarlyWhenIndexerOffloadClosed)
{
    SetupEngine();
    engine->mOpenIndexerOffload = false;  // early return
    std::vector<int> topkLensInput;
    std::vector<std::string> reqIDList;
    std::vector<torch::Tensor> topkLists;
    engine->SubmitTaskIndexer(topkLensInput, reqIDList, topkLists, /*stepTime=*/1);
    // No worker spawned; mIsIndexerLoadRunning stays false.
    EXPECT_FALSE(engine->mIsIndexerLoadRunning.load());
}

TEST_F(SubmitTaskTest, SubmitTaskIndexerWithEmptyReqListCompletesViaWorker)
{
    SetupEngine();
    engine->mOpenIndexerOffload = true;
    std::vector<int> topkLensInput;
    std::vector<std::string> reqIDList;
    std::vector<torch::Tensor> topkLists;
    engine->SubmitTaskIndexer(topkLensInput, reqIDList, topkLists, /*stepTime=*/1);
    WaitForIndexerLoadDone();
    EXPECT_FALSE(engine->mIsIndexerLoadRunning.load());
}

TEST_F(SubmitTaskTest, SubmitTaskIndexerWithNonEmptyReqListCopiesBlockTable)
{
    SetupEngine();
    engine->mOpenIndexerOffload = true;
    const std::string reqID = "req1";
    // Pre-seed mUseBlockTableIdx and mPrefetchBlockTableIdx for the memcpy loop.
    engine->mUseBlockTableIdx[reqID].assign(engine->mLayerNum, std::vector<int>(32, 0));
    engine->mPrefetchBlockTableIdx[reqID].assign(engine->mLayerNum, std::vector<int>(32, 42));
    std::vector<int> topkLensInput = {32};
    std::vector<std::string> reqIDList = {reqID};
    std::vector<torch::Tensor> topkLists = {torch::zeros({32}, torch::dtype(torch::kInt32))};
    engine->SubmitTaskIndexer(topkLensInput, reqIDList, topkLists, /*stepTime=*/1);
    // Verify the memcpy happened: mUseBlockTableIdx should now contain 42.
    EXPECT_EQ(engine->mUseBlockTableIdx[reqID][0][0], 42);
    WaitForIndexerLoadDone();
    EXPECT_FALSE(engine->mIsIndexerLoadRunning.load());
}

}  // namespace
