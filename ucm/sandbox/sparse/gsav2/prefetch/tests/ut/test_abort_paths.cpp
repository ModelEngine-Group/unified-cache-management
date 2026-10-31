/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the KVPackAbort error paths in kvcache_pre_scatter.cpp.  These paths
 * call KVPackAbort -> std::abort, so they use EXPECT_DEATH.  Each test creates
 * its own engine (unique dpRank) so the fork's child process doesn't leak shm
 * state to subsequent tests.
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
using ucmprefetch::PrefetchTiming;
using ucmprefetch_test::CleanupShmFor;
using ucmprefetch_test::MakeModelInfo;
using ucmprefetch_test::MakeSizeInfo;
using ucmprefetch_test::MockStoreV1;
using ucmprefetch_test::TempLogFile;
using ucmprefetch_test::UniqueDpRank;

namespace {

/* Helper: create a fully-set-up engine (SetKvBuffer + HBMManagers). */
static std::unique_ptr<GSAPrefetchEngineC> MakeFullEngine(int dpRank, TempLogFile& logFile,
                                                          MockStoreV1& store)
{
    auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
    auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                             /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
    std::vector<int> topkLayerMap{0};
    auto engine = std::make_unique<GSAPrefetchEngineC>(
        model, size, /*isLog=*/false,
        /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile.Path());
    engine->SetStoreV1(&store);
    auto kHost = torch::zeros({1, 64, 4}, torch::kFloat32);
    auto vHost = torch::zeros({1, 64, 4}, torch::kFloat32);
    auto kDevice = torch::zeros({1, 64, 4}, torch::kFloat32);
    auto vDevice = torch::zeros({1, 64, 4}, torch::kFloat32);
    auto loadIdxHost = torch::zeros({1, 64}, torch::kInt64);
    auto loadIdxDevice = torch::zeros({1, 64}, torch::kInt64);
    engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
    /* The ddr-load refactor made BuildUCMMergeBlockContexts / UCMTensorMerge
     * stage into the engine-owned mUCMMergeLoadBuffer (set by the shared-memory
     * SetupKVPackSharedBuffers in production, which tests skip). Point it at
     * the engine-allocated mLastChunkLoadBuffer so the merge / abort paths have
     * a valid staging region without extra bookkeeping. */
    if (engine->mUCMMergeLoadBuffer == nullptr && engine->mLastChunkLoadBuffer != nullptr) {
        engine->mUCMMergeLoadBuffer = static_cast<std::byte*>(engine->mLastChunkLoadBuffer);
    }
    return engine;
}

/* ---- RunInplaceScatter abort paths ---- */

TEST(AbortPaths, RunInplaceScatterAbortsOnWorkspaceSizeFailure)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    acl_stub::Stub().inplaceScatterWsfFails = true;
    aclTensor* dst = reinterpret_cast<aclTensor*>(0xABCD);
    aclTensor* idx = reinterpret_cast<aclTensor*>(0xBEEF);
    aclTensor* hbm = reinterpret_cast<aclTensor*>(0xCAFE);
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    EXPECT_DEATH(engine->RunInplaceScatter("K", 0, 0, dst, idx, hbm, nullptr), "");
}

TEST(AbortPaths, RunInplaceScatterAbortsOnExecutionFailure)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    acl_stub::Stub().inplaceScatterFails = true;
    aclTensor* dst = reinterpret_cast<aclTensor*>(0xABCD);
    aclTensor* idx = reinterpret_cast<aclTensor*>(0xBEEF);
    aclTensor* hbm = reinterpret_cast<aclTensor*>(0xCAFE);
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    EXPECT_DEATH(engine->RunInplaceScatter("K", 0, 0, dst, idx, hbm, nullptr), "");
}

/* ---- ValidateAndPublishHbmSlotIndex abort paths ---- */

TEST(AbortPaths, ValidateAbortsOnSizeMismatch)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    engine->mLoadNum.assign(1, 4);  // 4 != hbmSlotIndex.size() (2)
    engine->mLoadIndexBufferLen = 64;
    std::vector<uint64_t> hbmSlotIndex = {100ull, 200ull};  // size 2
    EXPECT_DEATH(engine->ValidateAndPublishHbmSlotIndex(0, 0, hbmSlotIndex), "");
}

TEST(AbortPaths, ValidateAbortsOnSizeOverflow)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    engine->mLoadNum.assign(1, 2);
    engine->mLoadIndexBufferLen = 1;  // hbmSlotIndex.size() (2) > 1
    std::vector<uint64_t> hbmSlotIndex = {100ull, 200ull};
    EXPECT_DEATH(engine->ValidateAndPublishHbmSlotIndex(0, 0, hbmSlotIndex), "");
}

/* ---- TransferAndScatterPrefetchLayer abort paths ---- */

TEST(AbortPaths, TransferAbortsOnKCopyTransFailure)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    engine->mLoadNum.assign(1, 4);
    engine->mKVSizeBytesToken = 16;
    engine->mKVSizeBytesTokenRe = 16;
    engine->mKCacheAcl.assign(1, reinterpret_cast<aclTensor*>(0xABCD));
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    engine->mKDstIndexAcl.assign(1, nullptr);
    engine->mKHBMBufferAcl.assign(1, nullptr);
    acl_stub::Stub().memcpyAsyncFails = true;  // CopyTrans fails
    ThreadInfo threadInfo{};
    threadInfo.context_ = engine->mContext;
    threadInfo.stream_ = engine->mStream;
    PrefetchTiming timing{};
    auto syncEnd = std::chrono::high_resolution_clock::now();
    EXPECT_DEATH(engine->TransferAndScatterPrefetchLayer(threadInfo, 0, 0, timing, syncEnd), "");
}

/* ---- GetScatterWorkspace abort path ---- */

TEST(AbortPaths, GetScatterWorkspaceAbortsOnMallocFailure)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    acl_stub::Stub().mallocFails = true;
    engine->mKWorkspaceAddr.assign(1, nullptr);
    engine->mKMaxWorkspaceSize.assign(1, 0);
    EXPECT_DEATH(engine->GetScatterWorkspace("K", 0, 0, 1024), "");
}

/* ---- ValidateUCMMergeRequest abort paths ---- */

TEST(AbortPaths, ValidateUCMMergeRequestAbortsOnMissSizeMismatch)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {1, 2, 3};  // size 3
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {1, 2};  // size 2 != 3
    EXPECT_DEATH(engine->ValidateUCMMergeRequest(0, reqID), "");
}

TEST(AbortPaths, ValidateUCMMergeRequestAbortsOnMissingBlocksHash)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    const std::string reqID = "req1";
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {1, 2, 3};
    engine->mLoadSlot[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mLoadSlot[reqID][0] = {1, 2, 3};  // same size, but no mAllBlocksHash
    // mAllBlocksHash doesn't have reqID -> KVPackAbort
    EXPECT_DEATH(engine->ValidateUCMMergeRequest(0, reqID), "");
}

/* ---- BuildUCMMergeBlockContexts abort path ---- */

TEST(AbortPaths, BuildUCMMergeBlockContextsAbortsOnNegativeMissGlobalId)
{
    int dpRank = UniqueDpRank();
    CleanupShmFor(dpRank);
    TempLogFile logFile;
    MockStoreV1 store;
    acl_stub::Stub().Reset();
    auto engine = MakeFullEngine(dpRank, logFile, store);
    const std::string reqID = "req1";
    engine->mBlockSize = 4;
    engine->mAllBlocksHash[reqID] = {ucmprefetch::BlockId{}, ucmprefetch::BlockId{},
                                     ucmprefetch::BlockId{}};
    engine->mMissGlobalToken[reqID].assign(engine->mTopkIndexerNum, {});
    engine->mMissGlobalToken[reqID][0] = {-1};  // negative -> KVPackAbort
    std::vector<ucmprefetch::UCMMergeBlockContext> blockContexts;
    UC::Detail::TaskDesc task;
    std::map<int, std::vector<size_t>> blockIdxToTokenIdx;
    EXPECT_DEATH(
        engine->BuildUCMMergeBlockContexts(0, reqID, 0, blockContexts, task, blockIdxToTokenIdx,
                                           engine->mUCMMergeLoadBuffer, engine->mShardSize),
        "");
}

}  // namespace
