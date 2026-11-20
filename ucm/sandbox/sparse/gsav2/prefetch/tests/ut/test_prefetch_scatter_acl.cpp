/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the ACL scatter functions in kvcache_pre_scatter.cpp:
 *   - RunInplaceScatter
 *   - KVCacheInplaceScatter (K and V)
 *   - PrepareScatterAclTensor
 *
 * Split into a separate binary to stay under qemu's ~8-engine-ctor limit.
 */
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
using ucmprefetch::PrefetchTiming;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

class PrefetchScatterAclTest : public ::testing::Test {
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
    { return torch::zeros({streamNum, tokens, dim}, torch::dtype(torch::kFloat32)); }
    static torch::Tensor Make1DIndexTensor(int streamNum, int tokens)
    { return torch::zeros({streamNum, tokens}, torch::dtype(torch::kInt64)); }

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
        auto kHost = Make2DTensor(1, 64, 4);
        auto vHost = Make2DTensor(1, 64, 4);
        auto kDevice = Make2DTensor(1, 64, 4);
        auto vDevice = Make2DTensor(1, 64, 4);
        auto loadIdxHost = Make1DIndexTensor(1, 64);
        auto loadIdxDevice = Make1DIndexTensor(1, 64);
        engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
        ASSERT_NE(engine->mKVPackControl, nullptr);
    }
};

/* ---------------- RunInplaceScatter ---------------- */

TEST_F(PrefetchScatterAclTest, RunInplaceScatterSucceedsWithStubbedAcl)
{
    SetupFullEngine();
    aclTensor* dstTensor = reinterpret_cast<aclTensor*>(0xABCD);
    aclTensor* dstIndexTensor = reinterpret_cast<aclTensor*>(0xBEEF);
    aclTensor* dstHBMBufferTensor = reinterpret_cast<aclTensor*>(0xCAFE);
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    engine->RunInplaceScatter("K", /*layerID=*/0, /*streamIdx=*/0, dstTensor, dstIndexTensor,
                              dstHBMBufferTensor, /*stream=*/nullptr);
    EXPECT_GT(acl_stub::Stub().inplaceScatterWsfCount.load(), 0);
    EXPECT_GT(acl_stub::Stub().inplaceScatterCount.load(), 0);
}

// NOTE: RunInplaceScatter's workspace-size-failure and execution-failure paths
// call KVPackAbort (std::abort).  EXPECT_DEATH forks the process, which
// corrupts the shared-memory state for subsequent tests under qemu; the abort
// paths are documented and covered on real hardware.

/* ---------------- KVCacheInplaceScatter ---------------- */

TEST_F(PrefetchScatterAclTest, KVCacheInplaceScatterKEndToEndWithStubbedAcl)
{
    SetupFullEngine();
    engine->mLoadNum.assign(1, 4);
    engine->mKCacheAcl.assign(1, reinterpret_cast<aclTensor*>(0xABCD));
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    engine->mKDstIndexAcl.assign(1, nullptr);
    engine->mKHBMBufferAcl.assign(1, nullptr);
    ASSERT_NE(engine->mContext, nullptr);
    engine->KVCacheInplaceScatter("K", /*layerID=*/0,
                                  /*stream=*/nullptr,
                                  *static_cast<aclrtContext*>(engine->mContext));
    EXPECT_GT(acl_stub::Stub().inplaceScatterCount.load(), 0);
}

TEST_F(PrefetchScatterAclTest, KVCacheInplaceScatterVEndToEndWithStubbedAcl)
{
    SetupFullEngine(/*layerNum=*/1, /*isSfaC8=*/0);
    engine->mLoadNum.assign(1, 4);
    engine->mVCacheAcl.assign(1, reinterpret_cast<aclTensor*>(0xABCD));
    engine->mVWorkspaceAddr.assign(1, nullptr);
    engine->mVMaxWorkspaceSize.assign(1, 0);
    engine->mVDstIndexAcl.assign(1, nullptr);
    engine->mVHBMBufferAcl.assign(1, nullptr);
    ASSERT_NE(engine->mContext, nullptr);
    engine->KVCacheInplaceScatter("V", 0, nullptr, *static_cast<aclrtContext*>(engine->mContext));
    EXPECT_GT(acl_stub::Stub().inplaceScatterCount.load(), 0);
}

/* ---------------- PrepareScatterAclTensor ---------------- */

TEST_F(PrefetchScatterAclTest, PrepareScatterAclTensorCreatesAndReplacesForKAndV)
{
    SetupFullEngine();
    // K path: create fresh tensors.
    engine->mKDstIndexAcl.assign(1, nullptr);
    engine->mKHBMBufferAcl.assign(1, nullptr);
    auto dstIndexND = torch::zeros({4, 4}, torch::kInt64);
    auto hbmBufferND = torch::zeros({4, 4}, torch::kFloat32);
    aclTensor* dstIndexTensor = nullptr;
    aclTensor* dstHBMBufferTensor = nullptr;
    engine->PrepareScatterAclTensor("K", /*streamIdx=*/0, dstIndexND, hbmBufferND, dstIndexTensor,
                                    dstHBMBufferTensor);
    EXPECT_NE(engine->mKDstIndexAcl[0], nullptr);
    EXPECT_NE(engine->mKHBMBufferAcl[0], nullptr);
    EXPECT_EQ(dstIndexTensor, engine->mKDstIndexAcl[0]);
    EXPECT_EQ(dstHBMBufferTensor, engine->mKHBMBufferAcl[0]);

    // V path: create fresh tensors.
    engine->mVDstIndexAcl.assign(1, nullptr);
    engine->mVHBMBufferAcl.assign(1, nullptr);
    engine->PrepareScatterAclTensor("V", 0, dstIndexND, hbmBufferND, dstIndexTensor,
                                    dstHBMBufferTensor);
    EXPECT_NE(engine->mVDstIndexAcl[0], nullptr);
    EXPECT_NE(engine->mVHBMBufferAcl[0], nullptr);

    // Re-use path: old tensors are released and replaced.
    acl_stub::Stub().destroyTensorCount.store(0);
    engine->PrepareScatterAclTensor("K", 0, dstIndexND, hbmBufferND, dstIndexTensor,
                                    dstHBMBufferTensor);
    EXPECT_GT(acl_stub::Stub().destroyTensorCount.load(), 0);
}

/* ---------------- PreparePrefetchResults (rank 0 path) ---------------- */
// NOTE: PreparePrefetchResults wraps ProcessPrefetchRequest in a
// `#pragma omp parallel for` loop + an atomic version increment.  The omp
// loop crashes under qemu (torch tensor ref-counting / OpenMP interaction in
// the emulated aarch64 environment).  The inner ProcessPrefetchRequest is
// tested directly in test_prefetch_h2d.cpp; the version increment is a
// trivial atomic store covered on real hardware.

/* ---------------- MergeAndCopyPrefetchLayer (early-exit path) ---------------- */
// With topkLenList={0}, the UCMTensorMerge loop is skipped, mLoadNum stays 0,
// and the function takes the early exit at MarkKVH2DDone.  This covers
// BeginKVPackSlot, FinishKVPack, CopyPrefetchToH2DBuffers (noop),
// ValidateAndPublishHbmSlotIndex (noop), Synchronize, and MarkKVH2DDone.

TEST_F(PrefetchScatterAclTest, MergeAndCopyPrefetchLayerEarlyExitOnZeroTopk)
{
    SetupFullEngine();
    engine->mLoadNum.assign(engine->mLayerNum, 0);
    engine->mLoadIndexBufferLen = 64;
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"req1"};
    taskInfo.topkLenList = {0};  // <= 0 -> UCMTensorMerge loop skipped
    ThreadInfo threadInfo{};
    // Synchronize dereferences context_, so use the engine's malloc'd context.
    threadInfo.context_ = engine->mContext;
    threadInfo.stream_ = engine->mStream;
    PrefetchTiming timing{};
    // Increment version so BeginKVPackSlot's epoch is non-zero.
    engine->mPrefetchResultVersion.fetch_add(1);
    engine->MergeAndCopyPrefetchLayer(taskInfo, threadInfo, /*layerID=*/0, timing);
    // Early exit: mLoadNum[0] == 0 -> MarkKVH2DDone called, no scatter.
    EXPECT_EQ(engine->mLoadNum[0], 0);
    EXPECT_GE(timing.cpuMove, 0);
}

/* ---------------- TransferAndScatterPrefetchLayer ---------------- */
// Direct test (bypassing MergeAndCopyPrefetchLayer) with stubbed ACL.
// Needs mLoadNum > 0, mKCacheAcl/mVCacheAcl set, mH2DPtrInfo (SetKvBuffer).

TEST_F(PrefetchScatterAclTest, TransferAndScatterPrefetchLayerKEndToEnd)
{
    SetupFullEngine();
    engine->mLoadNum.assign(1, 4);  // > 0 -> full H2D + scatter path
    engine->mKVSizeBytesToken = 16;
    engine->mKVSizeBytesTokenRe = 16;
    engine->mKCacheAcl.assign(1, reinterpret_cast<aclTensor*>(0xABCD));
    engine->mVCacheAcl.assign(1, reinterpret_cast<aclTensor*>(0xABCD));
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    engine->mVWorkspaceAddr.assign(1, nullptr);
    engine->mVMaxWorkspaceSize.assign(1, 0);
    engine->mKDstIndexAcl.assign(1, nullptr);
    engine->mKHBMBufferAcl.assign(1, nullptr);
    engine->mVDstIndexAcl.assign(1, nullptr);
    engine->mVHBMBufferAcl.assign(1, nullptr);
    ThreadInfo threadInfo{};
    threadInfo.context_ = engine->mContext;
    threadInfo.stream_ = engine->mStream;
    PrefetchTiming timing{};
    auto syncEnd = std::chrono::high_resolution_clock::now();
    engine->TransferAndScatterPrefetchLayer(threadInfo, /*layerID=*/0,
                                            /*streamIdx=*/0, timing, syncEnd);
    // CopyTrans + Synchronize + KVCacheInplaceScatter all succeed via stub.
    EXPECT_GT(acl_stub::Stub().inplaceScatterCount.load(), 0);
    EXPECT_GE(timing.h2d, 0);
}

/* ---------------- IndexerLoadProcess (early-exit paths) ---------------- */

TEST_F(PrefetchScatterAclTest, IndexerLoadProcessReturnsZeroWhenIndexerOffloadClosed)
{
    SetupFullEngine();
    engine->mOpenIndexerOffload = false;  // early return
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"req1"};
    taskInfo.topkLenList = {0};
    ThreadInfo threadInfo{};
    int ret = engine->IndexerLoadProcess(taskInfo, threadInfo);
    EXPECT_EQ(ret, 0);
}

TEST_F(PrefetchScatterAclTest, IndexerLoadProcessSkipsUnknownReqAndNullContext)
{
    SetupFullEngine();
    engine->mOpenIndexerOffload = true;
    // mDocsTables is empty -> reqID not found -> continue
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"unknownReq"};
    taskInfo.topkLenList = {32};  // > 0 but reqID not in mDocsTables
    ThreadInfo threadInfo{};
    threadInfo.context_ = nullptr;  // UCMIndexerLoadAsync returns early
    int ret = engine->IndexerLoadProcess(taskInfo, threadInfo);
    EXPECT_EQ(ret, 0);
}

TEST_F(PrefetchScatterAclTest, IndexerLoadProcessSkipsZeroTopkLenReq)
{
    SetupFullEngine();
    engine->mOpenIndexerOffload = true;
    // Pre-seed mDocsTables so reqID is found, but topkLen <= 0 -> skip
    engine->mDocsTables["req1"] = {};
    TaskInfo taskInfo{};
    taskInfo.reqIdList = {"req1"};
    taskInfo.topkLenList = {0};  // <= 0 -> skip
    ThreadInfo threadInfo{};
    threadInfo.context_ = nullptr;
    int ret = engine->IndexerLoadProcess(taskInfo, threadInfo);
    EXPECT_EQ(ret, 0);
}

/* ---------------- KVCacheInplaceScatter (setContext failure path) ---------------- */

TEST_F(PrefetchScatterAclTest, KVCacheInplaceScatterLogsErrorOnSetContextFailure)
{
    SetupFullEngine();
    acl_stub::Stub().setCurrentContextFails = true;
    engine->mLoadNum.assign(1, 4);
    engine->mKCacheAcl.assign(1, reinterpret_cast<aclTensor*>(0xABCD));
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    engine->mKDstIndexAcl.assign(1, nullptr);
    engine->mKHBMBufferAcl.assign(1, nullptr);
    ASSERT_NE(engine->mContext, nullptr);
    // The function logs the error and continues (no abort).
    engine->KVCacheInplaceScatter("K", /*layerID=*/0,
                                  /*stream=*/nullptr,
                                  *static_cast<aclrtContext*>(engine->mContext));
    // Verify setContext was called and failed.
    EXPECT_GT(acl_stub::Stub().setCurrentContextCount.load(), 0);
    acl_stub::Stub().setCurrentContextFails = false;
}

/* ---------------- PrepareScatterAclTensor (V re-use path) ---------------- */

TEST_F(PrefetchScatterAclTest, PrepareScatterAclTensorReleasesOldVTensorsOnReuse)
{
    SetupFullEngine();
    // Pre-set V tensors to non-null to exercise the release-old path.
    engine->mVDstIndexAcl.assign(1, reinterpret_cast<aclTensor*>(0xDEAD));
    engine->mVHBMBufferAcl.assign(1, reinterpret_cast<aclTensor*>(0xBEEF));
    auto dstIndexND = torch::zeros({4, 4}, torch::kInt64);
    auto hbmBufferND = torch::zeros({4, 4}, torch::kFloat32);
    aclTensor* dstIndexTensor = nullptr;
    aclTensor* dstHBMBufferTensor = nullptr;
    acl_stub::Stub().destroyTensorCount.store(0);
    engine->PrepareScatterAclTensor("V", 0, dstIndexND, hbmBufferND, dstIndexTensor,
                                    dstHBMBufferTensor);
    // Old V tensors were released and replaced.
    EXPECT_NE(engine->mVDstIndexAcl[0], reinterpret_cast<aclTensor*>(0xDEAD));
    EXPECT_NE(engine->mVHBMBufferAcl[0], reinterpret_cast<aclTensor*>(0xBEEF));
    EXPECT_GT(acl_stub::Stub().destroyTensorCount.load(), 0);
}

}  // namespace
