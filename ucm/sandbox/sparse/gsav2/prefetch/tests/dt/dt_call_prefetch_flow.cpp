/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * DT (end-to-end): CallPrefetchProcess full-pipeline correctness.
 *
 * Drives the engine through the complete decode-step pipeline using ONLY the
 * public setup surface, exactly as production does:
 *
 *   SetKvTensor (HBM KV cache + last-chunk staging)
 *   -> SetKvBuffer (H2D host/device buffers + KVPack shared staging init)
 *   -> SetTopkIndices / SetBlocksMap (UCM block hash table + non-UCM block
 *      addresses)
 *   -> SetTokensMap (creates the per-request HBMManager with the production
 *      LRU preload mapping, then runs LastChunkMove: 4 UCM blocks spliced
 *      into the last-chunk staging via LoadOnToken and H2D-copied into the
 *      HBM KV cache tensor)
 *   -> CallPrefetchProcess:
 *        PreparePrefetchResults   (HBM manager: topk -> hits/misses/slots,
 *                                  LRU eviction, useBlockTable, shm publish)
 *        MergeAndCopyPrefetchLayer(whole-batch UCMTensorMergeBatch: one
 *                                  LoadOnToken task per K/V leg, engine-side
 *                                  non-UCM local copies, hbmSlotIndex build)
 *        CopyPrefetchToH2DBuffers / ValidateAndPublishHbmSlotIndex
 *        TransferAndScatterPrefetchLayer (H2D CopyTrans + InplaceScatter)
 *
 * Focus areas (per the task): HBM manager hit/miss bookkeeping, LRU eviction
 * order across decode steps, KV Load address correctness (LoadOnToken shard
 * dst == shared staging window + offset, filter ranges == in-block offsets),
 * H2D data flow (staging -> host buffer -> device buffer -> slot index) and
 * the whole-batch merge of multiple requests into disjoint staging windows.
 *
 * All ACL ops run through the stub (real memcpys, counted scatter calls);
 * the store is MockStoreV1 whose LoadOnToken splices deterministic payloads,
 * so every byte in the pipeline has an expected value.
 */
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "../common/mock_store_v1.h"
#include "../common/test_fixture.h"
#include "kvcache_pre.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::BlockId;
using ucmprefetch::GSAPrefetchEngineC;
using ucmprefetch::kKTensorIndex;
using ucmprefetch::kStoreTensorCount;
using ucmprefetch::kVTensorIndex;
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

bool VecEq(const std::vector<int>& actual, const std::vector<int>& expected)
{
    return actual == expected;
}

/* Layout constants shared by the fixture and the assertions.
 * Token bytes = elemSize(2) * numHead(1) * kDim(2) = 4; block = 4 tokens. */
struct FlowFixture {
    static constexpr int kLayerNum = 1;
    static constexpr int kStreamNum = 1;
    static constexpr int kBlockSize = 4;
    static constexpr int kNumHead = 1;
    static constexpr int kKDim = 2;
    static constexpr int kVDim = 2;
    static constexpr int kElemSize = 2;
    static constexpr int kSpaTopkLen = 4;
    static constexpr int kSparseLen = 16;
    static constexpr int kDramTensorCount = 32;  // prompt tokens: gids 0..31
    static constexpr int kUcmBlockCount = 7;     // blocks 0..6 UCM, 7 non-UCM
    static constexpr int kKVHBMBlocks = 64;
    static constexpr int kBufferTokens = 64;
    static constexpr int kTokenBytes = kElemSize * kNumHead * kKDim;

    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;
    /* HBM block table for the 4 sparse blocks (capacity 16 slots). */
    const std::vector<int> blockTable = {10, 20, 30, 40};
    /* Non-UCM block payloads (block 7 holds the last chunk on device). */
    std::vector<uint8_t> nonUcmK;
    std::vector<uint8_t> nonUcmV;
    /* ACL stream/context handed to CallPrefetchProcess via ThreadInfo. */
    aclrtStream streamArr[1]{};
    aclrtContext ctx{};
    ThreadInfo threadInfo;

    void Setup()
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        auto model = MakeModelInfo(kLayerNum, /*maxBlockLen=*/4096, kSpaTopkLen, kStreamNum,
                                   kSparseLen, /*prefetchType=*/1, dpRank,
                                   /*tpSize=*/1, /*indexerOffload=*/0, /*isSfaC8=*/0,
                                   /*maxBs=*/2);
        auto size = MakeSizeInfo(kBlockSize, kNumHead, kKDim, kVDim, kElemSize);
        std::vector<int> topkLayerMap(kLayerNum);
        for (int i = 0; i < kLayerNum; ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());

        /* HBM KV cache tensor: 64 blocks * 16 bytes; last-chunk staging:
         * 4 blocks * 16 bytes. SetKvTensor also derives mKVHBMBlockLen and
         * the last-chunk pointers used by LastChunkMove. */
        auto kCaches = torch::zeros({kKVHBMBlocks, kBlockSize, kKDim}, torch::dtype(torch::kInt16));
        auto vCaches = torch::zeros({kKVHBMBlocks, kBlockSize, kVDim}, torch::dtype(torch::kInt16));
        auto lastChunkK =
            torch::zeros({kSparseLen / kBlockSize, kBlockSize, kKDim}, torch::dtype(torch::kInt16));
        auto lastChunkV =
            torch::zeros({kSparseLen / kBlockSize, kBlockSize, kVDim}, torch::dtype(torch::kInt16));
        engine->SetKvTensor(kCaches, vCaches, lastChunkK, lastChunkV, /*layerID=*/0);

        /* H2D host/device buffers + KVPack shared staging (real POSIX shm).
         * 64 tokens * 2 dim * 4B = 512B per stream, staging window 512B. */
        auto kHost =
            torch::zeros({kStreamNum, kBufferTokens, kKDim}, torch::dtype(torch::kFloat32));
        auto vHost =
            torch::zeros({kStreamNum, kBufferTokens, kVDim}, torch::dtype(torch::kFloat32));
        auto kDev = torch::zeros({kStreamNum, kBufferTokens, kKDim}, torch::dtype(torch::kFloat32));
        auto vDev = torch::zeros({kStreamNum, kBufferTokens, kVDim}, torch::dtype(torch::kFloat32));
        auto idxHost = torch::zeros({kStreamNum, kBufferTokens}, torch::dtype(torch::kInt64));
        auto idxDev = torch::zeros({kStreamNum, kBufferTokens}, torch::dtype(torch::kInt64));
        engine->SetKvBuffer(kHost, vHost, kDev, vDev, idxHost, idxDev);
        ASSERT_NE(engine->mKVPackControl, nullptr);

        nonUcmK.assign(32, 0x77);
        nonUcmV.assign(32, 0x88);

        aclrtCreateContext(&ctx, 0);
        aclrtCreateStream(&streamArr[0]);
        threadInfo = ThreadInfo(0, 0, streamArr, &ctx, nullptr);
    }

    void TearDown()
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    /* Public-API per-request setup: topk indices tensor, UCM hash table
     * (blocks 0..6) + non-UCM block address (block 7), HBM manager creation
     * via SetTokensMap (which also runs LastChunkMove). */
    void SetupRequest(const std::string& reqID, const std::vector<int>& initTopk)
    {
        auto topIndices =
            torch::zeros({engine->mTopkIndexerNum, kSpaTopkLen}, torch::dtype(torch::kInt32));
        engine->SetTopkIndices(reqID, topIndices);

        std::vector<BlockId> hashes;
        std::vector<int> blockIdxs;
        std::vector<uintptr_t> kAddrs;
        std::vector<uintptr_t> vAddrs;
        for (int b = 0; b < kUcmBlockCount; ++b) {
            hashes.push_back(MakeBlock(static_cast<uint8_t>(b + 1)));
            blockIdxs.push_back(b);
            kAddrs.push_back(0x1000 + static_cast<uintptr_t>(b) * 0x100);
            vAddrs.push_back(0x2000 + static_cast<uintptr_t>(b) * 0x100);
        }
        blockIdxs.push_back(kUcmBlockCount);
        kAddrs.push_back(reinterpret_cast<uintptr_t>(nonUcmK.data()));
        vAddrs.push_back(reinterpret_cast<uintptr_t>(nonUcmV.data()));
        engine->SetBlocksMap(reqID, hashes, kDramTensorCount, blockIdxs, kAddrs, vAddrs);

        auto qTopk = torch::zeros({1, 1, static_cast<int64_t>(initTopk.size())},
                                  torch::dtype(torch::kInt32));
        std::copy(initTopk.begin(), initTopk.end(), qTopk.data_ptr<int>());
        std::vector<int> slots;
        engine->SetTokensMap(/*layerID=*/0, reqID, qTopk, blockTable, kDramTensorCount, slots,
                             /*isNeedInit=*/true);
    }

    /* TaskInfo shaped like SubmitTask builds it: topkList is
     * [topkIndexerNum][bs][spaTopkLen] int32. */
    TaskInfo MakeFlowTaskInfo(const std::vector<std::string>& reqIds,
                              const std::vector<std::vector<int>>& topks)
    {
        const int64_t bs = static_cast<int64_t>(reqIds.size());
        torch::Tensor topkList =
            torch::zeros({engine->mTopkIndexerNum, bs, kSpaTopkLen}, torch::dtype(torch::kInt32));
        std::vector<int> lens;
        for (size_t i = 0; i < reqIds.size(); ++i) {
            lens.push_back(static_cast<int>(topks[i].size()));
            int* row = topkList[0][static_cast<int64_t>(i)].data_ptr<int>();
            std::copy(topks[i].begin(), topks[i].end(), row);
        }
        std::vector<int> bsInIndex;
        std::vector<std::string> ids = reqIds;
        return TaskInfo(0, -1, topkList, lens, bsInIndex, ids);
    }

    size_t LoadOnTokenCount() const
    {
        size_t n = 0;
        for (const auto& call : store->calls) {
            if (call.name == "LoadOnToken") { ++n; }
        }
        return n;
    }

    const MockStoreV1::CallRecord* FindLastMergeLoad(const std::string& brief) const
    {
        for (auto it = store->calls.rbegin(); it != store->calls.rend(); ++it) {
            if (it->name == "LoadOnToken" && it->task.brief == brief) { return &*it; }
        }
        return nullptr;
    }

    /* ---- byte/int readers over engine-owned tensors ---- */
    static const uint8_t* TensorBytes(const torch::Tensor& t)
    {
        return reinterpret_cast<const uint8_t*>(t.data_ptr());
    }
    const uint8_t* SharedK() const
    {
        return reinterpret_cast<const uint8_t*>(engine->mSharedKBufferHostPtr[0]);
    }
    const uint8_t* SharedV() const
    {
        return reinterpret_cast<const uint8_t*>(engine->mSharedVBufferHostPtr[0]);
    }
    const int64_t* LoadIndexHost() const
    {
        return reinterpret_cast<const int64_t*>(engine->mLoadIndexHost[0].data_ptr());
    }
    const int64_t* LoadIndexDevice() const
    {
        return reinterpret_cast<const int64_t*>(engine->mLoadIndexDevice[0].data_ptr());
    }
    const int* TopkIndices(const std::string& reqID) const
    {
        return engine->mAllTopkIndicesTensor.at(reqID)[0].data_ptr<int>();
    }
    void ExpectTopkIndices(const std::string& reqID, const std::vector<int>& expected) const
    {
        const int* ptr = TopkIndices(reqID);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(ptr[i], expected[i]) << reqID << " topkIndices[" << i << "]";
        }
    }
    HBMManagerLayered* Manager(const std::string& reqID) const
    {
        return engine->mHBMMangers.at(reqID)[0].get();
    }
};

}  // namespace

/* =====================================================================
 * Test 1: three decode steps through the full CallPrefetchProcess pipeline.
 *
 * Token/slot model (blockSize=4, dramTensorCount=32, capacity=16,
 * preload maps slots 0..15 <-> gids 0..15, LRU head=0 tail=15):
 *   step 1 topk {0,17,18,29}: gid0 hit; 17,18 miss UCM block 4; 29 miss
 *          non-UCM block 7 -> LRU tail eviction slots 15,14,13
 *   step 2 topk {17,18}:      both hit (loaded by step 1) -> zero loads
 *   step 3 topk {19}:         fresh miss -> victim is the NEW tail slot 12
 *          (proof the step-2 hits reordered the LRU list)
 * ===================================================================== */
TEST(PrefetchFlowDt, ThreeDecodeStepsLoadHitEvictCorrectness)
{
    FlowFixture f;
    f.Setup();
    f.SetupRequest("req1", {0});
    ASSERT_NE(f.engine->mHBMMangers.at("req1")[0], nullptr);

    /* ---- SetTokensMap ran LastChunkMove: 4 UCM blocks (gids 0..15) were
     * spliced into the last-chunk staging by LoadOnToken (mock fill =
     * shardIdx+1 / shardIdx+2) and H2D-copied into the HBM KV cache tensor
     * at blockTable offsets {10,20,30,40}. */
    ASSERT_EQ(f.LoadOnTokenCount(), 2u);  // K leg + V leg
    const auto* lastChunkK =
        reinterpret_cast<const uint8_t*>(f.engine->mLastCHunkKBuffer[0].data_ptr());
    const auto* lastChunkV =
        reinterpret_cast<const uint8_t*>(f.engine->mLastCHunkVBuffer[0].data_ptr());
    for (int block = 0; block < 4; ++block) {
        for (int b = 0; b < 16; ++b) {
            EXPECT_EQ(lastChunkK[block * 16 + b], 1u + static_cast<uint8_t>(block))
                << "last-chunk K block " << block << " byte " << b;
            EXPECT_EQ(lastChunkV[block * 16 + b], 2u + static_cast<uint8_t>(block))
                << "last-chunk V block " << block << " byte " << b;
        }
    }
    const auto* kCache = reinterpret_cast<const uint8_t*>(f.engine->mKcacheTensor[0].data_ptr());
    const auto* vCache = reinterpret_cast<const uint8_t*>(f.engine->mVcacheTensor[0].data_ptr());
    for (int i = 0; i < 4; ++i) {
        const size_t off = static_cast<size_t>(f.blockTable[i]) * 16;
        EXPECT_EQ(kCache[off], 1u + static_cast<uint8_t>(i))
            << "HBM KV cache K at blockTable[" << i << "]";
        EXPECT_EQ(vCache[off], 2u + static_cast<uint8_t>(i))
            << "HBM KV cache V at blockTable[" << i << "]";
    }

    /* ================= decode step 1: mixed hit / UCM miss / non-UCM miss */
    const size_t loadsBefore = f.LoadOnTokenCount();
    const int scatterBefore = acl_stub::Stub().inplaceScatterCount.load();
    TaskInfo step1 = f.MakeFlowTaskInfo(
        {
            "req1"
    },
        {{0, 17, 18, 29}});
    f.engine->CallPrefetchProcess(step1, f.threadInfo);

    /* -- HBM manager bookkeeping: hit + misses + LRU eviction -- */
    EXPECT_TRUE(VecEq(f.engine->mMissGlobalToken["req1"][0], {17, 18, 29}));
    EXPECT_TRUE(VecEq(f.engine->mLoadSlot["req1"][0], {15, 14, 13}));
    {
        HBMManagerLayered* mgr = f.Manager("req1");
        EXPECT_EQ(mgr->getSlotForGlobal(0), 0);    // hit token stayed resident
        EXPECT_EQ(mgr->getSlotForGlobal(17), 15);  // misses landed at victims
        EXPECT_EQ(mgr->getSlotForGlobal(18), 14);
        EXPECT_EQ(mgr->getSlotForGlobal(29), 13);
        EXPECT_EQ(mgr->getSlotForGlobal(15), -1);  // evicted LRU tail gids
        EXPECT_EQ(mgr->getSlotForGlobal(14), -1);
        EXPECT_EQ(mgr->getSlotForGlobal(13), -1);
        EXPECT_EQ(mgr->getSlotForGlobal(12), 12);  // untouched so far
        EXPECT_TRUE(mgr->validateLruChain());
        EXPECT_TRUE(mgr->validateMapping());
    }
    f.ExpectTopkIndices("req1", {0, 15, 14, 13});

    /* -- KV Load: one LoadOnToken task per leg, address contract -- */
    ASSERT_EQ(f.LoadOnTokenCount() - loadsBefore, 2u);
    const auto* kLoad = f.FindLastMergeLoad("UCMTensorMergeLoadOnTokenK");
    const auto* vLoad = f.FindLastMergeLoad("UCMTensorMergeLoadOnTokenV");
    ASSERT_NE(kLoad, nullptr);
    ASSERT_NE(vLoad, nullptr);
    ASSERT_EQ(kLoad->task.size(), 1u);  // single UCM block (block 4)
    ASSERT_EQ(kLoad->filters.size(), 1u);
    EXPECT_EQ(kLoad->task[0].addrs[0],
              reinterpret_cast<void*>(static_cast<uintptr_t>(f.engine->mSharedKBufferHostPtr[0])));
    ASSERT_EQ(kLoad->filters[0].ranges.size(), kStoreTensorCount);
    ASSERT_EQ(kLoad->filters[0].ranges[kKTensorIndex].size(), 2u);
    /* tokens 17,18 -> in-block offsets 1,2 -> byte offsets 4,8; dst offsets
     * 0,4 relative to the shard dst (= staging token window 0..1). */
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][0].offset, 4u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][0].length, 4u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][0].dstOffset, 0u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][1].offset, 8u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][1].length, 4u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][1].dstOffset, 4u);
    ASSERT_EQ(vLoad->task.size(), 1u);
    EXPECT_EQ(vLoad->task[0].addrs[0],
              reinterpret_cast<void*>(static_cast<uintptr_t>(f.engine->mSharedVBufferHostPtr[0])));
    ASSERT_EQ(vLoad->filters[0].ranges[kVTensorIndex].size(), 2u);

    /* -- merge output: mLoadNum + hbmSlotIndex published to load-index
     * host AND device tensors. slot 15/14/13 -> blockTable[3]=40 ->
     * 40*4+{3,2,1} = {163,162,161}. */
    EXPECT_EQ(f.engine->mLoadNum[0], 3u);
    EXPECT_EQ(f.LoadIndexHost()[0], 163);
    EXPECT_EQ(f.LoadIndexHost()[1], 162);
    EXPECT_EQ(f.LoadIndexHost()[2], 161);
    EXPECT_EQ(f.LoadIndexDevice()[0], 163);
    EXPECT_EQ(f.LoadIndexDevice()[1], 162);
    EXPECT_EQ(f.LoadIndexDevice()[2], 161);

    /* -- data flow: staging shm -> H2D host buffer -> device buffer.
     * tokens 0,1 (UCM block 4, K shard 0 -> fill 0x01, V -> 0x02);
     * token 2 (non-UCM block 7) locally copied from nonUcmK/V (0x77/0x88). */
    for (int i = 0; i < 12; ++i) {
        const uint8_t expectK = (i < 8) ? 0x01 : 0x77;
        const uint8_t expectV = (i < 8) ? 0x02 : 0x88;
        EXPECT_EQ(f.SharedK()[i], expectK) << "staging K byte " << i;
        EXPECT_EQ(f.SharedV()[i], expectV) << "staging V byte " << i;
        EXPECT_EQ(f.TensorBytes(f.engine->mKBufferHost[0])[i], expectK) << "H2D K host byte " << i;
        EXPECT_EQ(f.TensorBytes(f.engine->mVBufferHost[0])[i], expectV) << "H2D V host byte " << i;
        EXPECT_EQ(f.TensorBytes(f.engine->mKBufferDevice[0])[i], expectK)
            << "H2D K device byte " << i;
        EXPECT_EQ(f.TensorBytes(f.engine->mVBufferDevice[0])[i], expectV)
            << "H2D V device byte " << i;
    }
    /* K + V scatter ops issued on the ACL stream (stubbed, counted). */
    EXPECT_EQ(acl_stub::Stub().inplaceScatterCount.load() - scatterBefore, 2);

    /* ================= decode step 2: all hits -> zero loads ============ */
    const size_t loadsAfterStep1 = f.LoadOnTokenCount();
    TaskInfo step2 = f.MakeFlowTaskInfo(
        {
            "req1"
    },
        {{17, 18}});
    f.engine->CallPrefetchProcess(step2, f.threadInfo);

    EXPECT_EQ(f.LoadOnTokenCount(), loadsAfterStep1);  // nothing loaded
    EXPECT_EQ(f.engine->mLoadNum[0], 0u);
    EXPECT_TRUE(f.engine->mMissGlobalToken["req1"][0].empty());
    EXPECT_TRUE(f.engine->mLoadSlot["req1"][0].empty());
    /* topkIndices = hit slots [15,14] + LRU-head padding [13,0]. */
    f.ExpectTopkIndices("req1", {15, 14, 13, 0});
    /* Published slot index untouched by the zero-load step. */
    EXPECT_EQ(f.LoadIndexHost()[0], 163);
    {
        HBMManagerLayered* mgr = f.Manager("req1");
        EXPECT_EQ(mgr->getSlotForGlobal(17), 15);
        EXPECT_EQ(mgr->getSlotForGlobal(18), 14);
        EXPECT_EQ(mgr->getSlotForGlobal(12), 12);  // still resident
        EXPECT_TRUE(mgr->validateLruChain());
    }

    /* ================= decode step 3: fresh miss from the new tail ====== */
    TaskInfo step3 = f.MakeFlowTaskInfo({"req1"}, {{19}});
    f.engine->CallPrefetchProcess(step3, f.threadInfo);

    EXPECT_EQ(f.engine->mLoadNum[0], 1u);
    EXPECT_TRUE(VecEq(f.engine->mMissGlobalToken["req1"][0], {19}));
    /* Step-2 hits moved slots 15/14 to the LRU head, so the eviction victim
     * is the NEW tail slot 12 (not 15/14/13 again). */
    EXPECT_TRUE(VecEq(f.engine->mLoadSlot["req1"][0], {12}));
    EXPECT_EQ(f.LoadIndexHost()[0], 160);  // 40*4 + 0
    f.ExpectTopkIndices("req1", {12, 14, 15, 13});
    {
        HBMManagerLayered* mgr = f.Manager("req1");
        EXPECT_EQ(mgr->getSlotForGlobal(19), 12);
        EXPECT_EQ(mgr->getSlotForGlobal(12), -1);  // old resident evicted
        EXPECT_TRUE(mgr->validateLruChain());
        EXPECT_TRUE(mgr->validateMapping());
    }
    /* Fresh epoch re-used the staging window from token 0: mock K fill. */
    EXPECT_EQ(f.SharedK()[0], 0x01);
    EXPECT_EQ(f.SharedK()[1], 0x01);
    EXPECT_EQ(f.TensorBytes(f.engine->mKBufferHost[0])[0], 0x01);

    f.TearDown();
}

/* =====================================================================
 * Test 2: two requests in ONE CallPrefetchProcess (bs=2) go through the
 * whole-batch merge: exactly one LoadOnToken task per leg whose shards
 * concatenate both requests' UCM blocks, each request landing in its own
 * disjoint staging window (reqA tokens 0..2, reqB tokens 3..5).
 *
 *   reqA topk {0,17,18,29}: hits 0; UCM misses 17,18 (block 4); non-UCM 29
 *   reqB topk {0,19,21,23}: hits 0; UCM misses 19 (block 4), 21,23 (block 5)
 * Both managers evict their own LRU tails 15,14,13.
 * ===================================================================== */
TEST(PrefetchFlowDt, TwoRequestBatchSharesOneTaskPerLayerLeg)
{
    FlowFixture f;
    f.Setup();
    f.SetupRequest("reqA", {0});
    f.SetupRequest("reqB", {0});

    const size_t loadsBefore = f.LoadOnTokenCount();
    ASSERT_EQ(loadsBefore, 4u);  // LastChunkMove K+V per request

    TaskInfo batch = f.MakeFlowTaskInfo(
        {
            "reqA", "reqB"
    },
        {{0, 17, 18, 29}, {0, 19, 21, 23}});
    f.engine->CallPrefetchProcess(batch, f.threadInfo);

    /* -- per-request manager bookkeeping stayed independent -- */
    EXPECT_TRUE(VecEq(f.engine->mMissGlobalToken["reqA"][0], {17, 18, 29}));
    EXPECT_TRUE(VecEq(f.engine->mLoadSlot["reqA"][0], {15, 14, 13}));
    EXPECT_TRUE(VecEq(f.engine->mMissGlobalToken["reqB"][0], {19, 21, 23}));
    EXPECT_TRUE(VecEq(f.engine->mLoadSlot["reqB"][0], {15, 14, 13}));
    EXPECT_EQ(f.Manager("reqA")->getSlotForGlobal(17), 15);
    EXPECT_EQ(f.Manager("reqB")->getSlotForGlobal(23), 13);
    f.ExpectTopkIndices("reqA", {0, 15, 14, 13});
    f.ExpectTopkIndices("reqB", {0, 15, 14, 13});

    /* -- cumulative load number and published slot indices (reqA window
     * first, then reqB): all six slots are 15/14/13 -> blockTable[3]=40. -- */
    EXPECT_EQ(f.engine->mLoadNum[0], 6u);
    const int64_t* idx = f.LoadIndexHost();
    EXPECT_EQ(idx[0], 163);
    EXPECT_EQ(idx[1], 162);
    EXPECT_EQ(idx[2], 161);
    EXPECT_EQ(idx[3], 163);
    EXPECT_EQ(idx[4], 162);
    EXPECT_EQ(idx[5], 161);
    EXPECT_EQ(f.LoadIndexDevice()[5], 161);

    /* -- exactly one task per leg, three shards concatenated:
     * shard 0 = reqA block 4 (tokens 17,18, staging tokens 0,1)
     * shard 1 = reqB block 4 (token 19,     staging token 3)
     * shard 2 = reqB block 5 (tokens 21,23, staging tokens 4,5) -- */
    ASSERT_EQ(f.LoadOnTokenCount() - loadsBefore, 2u);
    const auto* kLoad = f.FindLastMergeLoad("UCMTensorMergeLoadOnTokenK");
    const auto* vLoad = f.FindLastMergeLoad("UCMTensorMergeLoadOnTokenV");
    ASSERT_NE(kLoad, nullptr);
    ASSERT_NE(vLoad, nullptr);
    ASSERT_EQ(kLoad->task.size(), 3u);
    ASSERT_EQ(kLoad->filters.size(), 3u);
    const uintptr_t kBase = static_cast<uintptr_t>(f.engine->mSharedKBufferHostPtr[0]);
    const uintptr_t vBase = static_cast<uintptr_t>(f.engine->mSharedVBufferHostPtr[0]);
    EXPECT_EQ(static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(kLoad->task[0].addrs[0])), kBase);
    EXPECT_EQ(static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(kLoad->task[1].addrs[0])),
              kBase + 3 * FlowFixture::kTokenBytes);
    EXPECT_EQ(static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(kLoad->task[2].addrs[0])),
              kBase + 4 * FlowFixture::kTokenBytes);
    EXPECT_EQ(static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(vLoad->task[1].addrs[0])),
              vBase + 3 * FlowFixture::kTokenBytes);
    /* shard 0 (reqA block 4): tokens 17,18 -> in-block byte offsets 4,8. */
    ASSERT_EQ(kLoad->filters[0].ranges[kKTensorIndex].size(), 2u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][0].offset, 4u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][0].dstOffset, 0u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][1].offset, 8u);
    EXPECT_EQ(kLoad->filters[0].ranges[kKTensorIndex][1].dstOffset, 4u);
    /* shard 1 (reqB block 4): token 19 -> in-block byte offset 12. */
    ASSERT_EQ(kLoad->filters[1].ranges[kKTensorIndex].size(), 1u);
    EXPECT_EQ(kLoad->filters[1].ranges[kKTensorIndex][0].offset, 12u);
    EXPECT_EQ(kLoad->filters[1].ranges[kKTensorIndex][0].dstOffset, 0u);
    /* shard 2 (reqB block 5): tokens 21,23 -> in-block byte offsets 4,12. */
    ASSERT_EQ(kLoad->filters[2].ranges[kKTensorIndex].size(), 2u);
    EXPECT_EQ(kLoad->filters[2].ranges[kKTensorIndex][0].offset, 4u);
    EXPECT_EQ(kLoad->filters[2].ranges[kKTensorIndex][0].dstOffset, 0u);
    EXPECT_EQ(kLoad->filters[2].ranges[kKTensorIndex][1].offset, 12u);
    EXPECT_EQ(kLoad->filters[2].ranges[kKTensorIndex][1].dstOffset, 4u);

    /* -- staging content per disjoint window (K fill = shardIdx+1,
     * V fill = shardIdx+2; reqA token 2 locally copied 0x77/0x88): -- */
    const uint8_t expectK[6] = {0x01, 0x01, 0x77, 0x02, 0x03, 0x03};
    const uint8_t expectV[6] = {0x02, 0x02, 0x88, 0x03, 0x04, 0x04};
    for (int t = 0; t < 6; ++t) {
        for (int b = 0; b < FlowFixture::kTokenBytes; ++b) {
            const size_t i = static_cast<size_t>(t) * FlowFixture::kTokenBytes + b;
            EXPECT_EQ(f.SharedK()[i], expectK[t]) << "staging K token " << t;
            EXPECT_EQ(f.SharedV()[i], expectV[t]) << "staging V token " << t;
            EXPECT_EQ(f.TensorBytes(f.engine->mKBufferHost[0])[i], expectK[t])
                << "H2D K host token " << t;
            EXPECT_EQ(f.TensorBytes(f.engine->mKBufferDevice[0])[i], expectK[t])
                << "H2D K device token " << t;
            EXPECT_EQ(f.TensorBytes(f.engine->mVBufferDevice[0])[i], expectV[t])
                << "H2D V device token " << t;
        }
    }
    /* One K + one V scatter for the whole 6-token batch. */
    EXPECT_EQ(acl_stub::Stub().inplaceScatterCount.load(), 2);

    f.TearDown();
}
