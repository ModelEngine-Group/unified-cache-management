/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for SetKvBuffer (kvcache_pre_scatter.cpp), which initializes the host
 * staging buffers and the KV pack control shared memory.  These tests verify
 * the success and the mISSPAC8 branch; they rely on shm_open/mmap succeeding
 * (host-only, safe under qemu).
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

class SetKvBufferTest : public ::testing::Test {
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

    /* Build tensors shaped [streamNum, tokens, dim] so [streamIdx] indexing works. */
    static torch::Tensor Make2DTensor(int streamNum, int tokens, int dim)
    { return torch::zeros({streamNum, tokens, dim}, torch::dtype(torch::kFloat32)); }
    static torch::Tensor Make1DIndexTensor(int streamNum, int tokens)
    { return torch::zeros({streamNum, tokens}, torch::dtype(torch::kInt64)); }
};

}  // namespace

TEST_F(SetKvBufferTest, SetKvBufferInitializesHostBuffersAndKvPackControl)
{
    auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
    auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                             /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
    std::vector<int> topkLayerMap = {0};
    engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                  /*rank=*/0, /*threadNum=*/1, std::vector<int>{0},
                                                  topkLayerMap, logFile->Path());
    engine->SetStoreV1(store.get());

    // kBufferHost/vBufferDevice: [streamNum=1, tokens=8, dim=4].
    auto kHost = Make2DTensor(1, 8, 4);
    auto vHost = Make2DTensor(1, 8, 4);
    auto kDevice = Make2DTensor(1, 8, 4);
    auto vDevice = Make2DTensor(1, 8, 4);
    auto loadIdxHost = Make1DIndexTensor(1, 8);
    auto loadIdxDevice = Make1DIndexTensor(1, 8);
    engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
    // mKVPackControl should now be non-null (InitKVPackControl succeeded).
    ASSERT_NE(engine->mKVPackControl, nullptr);
    EXPECT_EQ(engine->mKVBufferLen, 8);  // kHost[0].sizes()[0]
    EXPECT_EQ(engine->mLoadIndexBufferLen, 8);
    EXPECT_EQ(engine->mH2DPtrInfo.size(), 1u);
    EXPECT_EQ(engine->mH2DPtrInfo[0].size(), 6u);  // H2D_INFO_SIZE
    EXPECT_NE(engine->mSharedKBufferHostPtr[0], 0ull);
    EXPECT_NE(engine->mSharedVBufferHostPtr[0], 0ull);
    EXPECT_NE(engine->mKBufferHostPtr[0], 0ull);
    EXPECT_NE(engine->mVBufferHostPtr[0], 0ull);
}

TEST_F(SetKvBufferTest, SetKvBufferWithSpac8SkipsVBuffers)
{
    auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/1, /*isSfaC8=*/1, /*maxBs=*/2);
    auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                             /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
    std::vector<int> topkLayerMap = {0};
    engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                  /*rank=*/0, /*threadNum=*/1, std::vector<int>{0},
                                                  topkLayerMap, logFile->Path());
    engine->SetStoreV1(store.get());

    auto kHost = Make2DTensor(1, 8, 4);
    auto vHost = Make2DTensor(1, 8, 4);  // ignored (mISSPAC8)
    auto kDevice = Make2DTensor(1, 8, 4);
    auto vDevice = Make2DTensor(1, 8, 4);  // ignored
    auto loadIdxHost = Make1DIndexTensor(1, 8);
    auto loadIdxDevice = Make1DIndexTensor(1, 8);
    engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
    ASSERT_NE(engine->mKVPackControl, nullptr);
    // mISSPAC8 -> mSharedVBufferHostPtr / mVBufferHostPtr NOT resized.
    EXPECT_TRUE(engine->mSharedVBufferHostPtr.empty());
    EXPECT_EQ(engine->mVBufferHostPtr.size(), 0u);
    // But K buffers are set.
    EXPECT_NE(engine->mSharedKBufferHostPtr[0], 0ull);
    EXPECT_NE(engine->mKBufferHostPtr[0], 0ull);
}

TEST_F(SetKvBufferTest, SetKvBufferIsIdempotentOnKvPackControl)
{
    auto model = MakeModelInfo(/*layerNum=*/1, /*maxBlockLen=*/4096,
                               /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/128,
                               /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                               /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
    auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1,
                             /*kDim=*/4, /*vDim=*/4, /*elementSize=*/2);
    std::vector<int> topkLayerMap = {0};
    engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                  /*rank=*/0, /*threadNum=*/1, std::vector<int>{0},
                                                  topkLayerMap, logFile->Path());
    engine->SetStoreV1(store.get());

    auto kHost = Make2DTensor(1, 8, 4);
    auto vHost = Make2DTensor(1, 8, 4);
    auto kDevice = Make2DTensor(1, 8, 4);
    auto vDevice = Make2DTensor(1, 8, 4);
    auto loadIdxHost = Make1DIndexTensor(1, 8);
    auto loadIdxDevice = Make1DIndexTensor(1, 8);
    engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
    auto* firstControl = engine->mKVPackControl;
    ASSERT_NE(firstControl, nullptr);
    // Second call: mKVPackControl != nullptr -> InitKVPackControl NOT re-called.
    engine->SetKvBuffer(kHost, vHost, kDevice, vDevice, loadIdxHost, loadIdxDevice);
    EXPECT_EQ(engine->mKVPackControl, firstControl);  // same pointer
}
