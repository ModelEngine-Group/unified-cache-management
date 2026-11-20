/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * MST: cross-module flow for UCMTensorMerge.
 *
 * UCMTensorMerge ties together HBM miss-bookkeeping (mMissGlobalToken /
 * mLoadSlot), the UCM store (LoadOnToken + Wait via mock), the multi-rank
 * KVPack shared staging buffers, and the CPU token-merge loop.  We drive it
 * with a mock store and the ACL stub, and verify both the success path
 * (hbmSlotIndex filled, loadNum advanced) and the failure fallback path
 * (FillFailedUCMMerge zeroes owned tokens while preserving all-rank indices).
 */
#include <cstring>
#include <gtest/gtest.h>
#include <vector>
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

struct MergeFixture {
    static constexpr int kBlockSize = 4;
    static constexpr int kNumHead = 1;
    static constexpr int kKDim = 2;
    static constexpr int kVDim = 2;
    static constexpr int kElemSize = 2;  // bytes
    static constexpr int kLayerNum = 1;
    // kTokenBytes = 2*1*2 = 4; kBlockBytes = 4*4 = 16; vBlockBytes = 16
    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;
    std::vector<uint8_t> sharedK;
    std::vector<uint8_t> sharedV;
    std::vector<uint8_t> nonUcmK;
    std::vector<uint8_t> nonUcmV;

    void Setup()
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        /* Token splices go through LoadOnToken, whose mock payload is
         * per-shard/tensor and needs no per-block host payload config. */
        auto model = MakeModelInfo(kLayerNum, 4096, /*spaTopkLen=*/32,
                                   /*streamNum=*/1, /*sparseLen=*/16, 1, /*dpRank=*/dpRank,
                                   /*tpSize=*/1, /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(kBlockSize, kNumHead, kKDim, kVDim, kElemSize);
        std::vector<int> topkLayerMap(kLayerNum);
        for (int i = 0; i < kLayerNum; ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                      /*rank=*/0, /*threadNum=*/1,
                                                      /*cpuToBind=*/std::vector<int>{0},
                                                      topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());
        /* UCMTensorMerge now splices owned UCM tokens via LoadOnToken directly
         * into the shared staging buffers below; no engine-owned merge staging
         * buffer (mUCMMergeLoadBuffer) is involved any more. */

        // Shared K/V staging buffers (engine reads via mSharedKBufferHostPtr).
        sharedK.assign(64, 0xAA);
        sharedV.assign(64, 0xBB);
        engine->mSharedKBufferHostPtr.assign(1, reinterpret_cast<uint64_t>(sharedK.data()));
        engine->mSharedVBufferHostPtr.assign(1, reinterpret_cast<uint64_t>(sharedV.data()));
        engine->mLoadIndexBufferLen = 256;
        engine->mKVBufferLen = 256;
        engine->mLoadNum.assign(kLayerNum, 0);
    }

    void TearDown()
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    // Seed: 2 UCM blocks, miss tokens spanning them, slot/block tables.
    void SeedMissScenario()
    {
        std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
        std::vector<int> blockIdxs = {0, 1};
        std::vector<uintptr_t> addrs = {0x1000, 0x2000};
        engine->SetBlocksMap("req1", hashes, /*maxIdx=*/2, blockIdxs, addrs, addrs);
        // 4 misses, 2 per block: tokens 0,1,4,5 -> blocks 0,0,1,1
        engine->mMissGlobalToken["req1"].assign(kLayerNum, {
                                                                {0, 1, 4, 5}
        });
        engine->mLoadSlot["req1"].assign(kLayerNum, {
                                                         {0, 1, 8, 9}
        });
        engine->mReqUseBlockTable["req1"] = {10, 20};  // block table
    }

    /* Seed one request with the standard 2-UCM-block / 4-miss pattern (tokens
     * 0,1,4,5, slots 0,1,4,5), used by the whole-batch merge tests. */
    void SeedTwoBlockRequest(const std::string& reqID)
    {
        std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
        std::vector<int> blockIdxs = {0, 1};
        std::vector<uintptr_t> addrs = {0x1000, 0x2000};
        engine->SetBlocksMap(reqID, hashes, /*maxIdx=*/2, blockIdxs, addrs, addrs);
        engine->mMissGlobalToken[reqID].assign(kLayerNum, {{0, 1, 4, 5}});
        engine->mLoadSlot[reqID].assign(kLayerNum, {{0, 1, 4, 5}});
        engine->mReqUseBlockTable[reqID] = {10, 20};
    }

    /* Minimal TaskInfo carrying just the request IDs for the batch merge. */
    TaskInfo MakeBatchTaskInfo(std::vector<std::string> reqIds)
    {
        std::vector<int> topkLens(reqIds.size(), 1);
        std::vector<int> bsInIndex;
        torch::Tensor emptyTopk;
        return TaskInfo(0, 0, emptyTopk, topkLens, bsInIndex, reqIds);
    }
};

}  // namespace

TEST(UCMMergeFlow, SuccessFillsHbmSlotIndexAndAdvancesLoadNum)
{
    MergeFixture f;
    f.Setup();
    f.SeedMissScenario();
    /* Use in-range slots: slot/blockSize must stay below blockTable.size()=2
     * inside CopyUCMMergeTokensSerial. The index encoding itself is asserted
     * in the test below. */
    f.engine->mLoadSlot["req1"][0] = {0, 1, 4, 5};

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMerge(/*layerID=*/0, /*streamIdx=*/0, hbmSlotIndex,
                             /*reqID=*/"req1", /*requestIndex=*/0);

    // previousLoadNum=0 + 4 misses -> finalLoadNum=4.
    ASSERT_EQ(hbmSlotIndex.size(), 4u);
    EXPECT_EQ(f.engine->mLoadNum[0], 4u);
    // Owned UCM tokens were spliced by the mock's LoadOnToken into the
    // shared staging buffers (shard 0 -> 0x01, shard 1 -> 0x02).
    for (int i = 0; i < 2 * 4; ++i) {
        EXPECT_EQ(f.sharedK[i], 0x01u) << "shared K byte " << i;
        EXPECT_EQ(f.sharedV[i], 0x02u) << "shared V byte " << i;
    }
    for (int i = 2 * 4; i < 4 * 4; ++i) {
        EXPECT_EQ(f.sharedK[i], 0x02u) << "shared K byte " << i;
        EXPECT_EQ(f.sharedV[i], 0x03u) << "shared V byte " << i;
    }
    f.TearDown();
}

TEST(UCMMergeFlow, SuccessWithInrangeSlotsVerifiesIndexEncoding)
{
    MergeFixture f;
    f.Setup();
    f.SeedMissScenario();
    // Use slots that keep slot/blockSize < blockTable.size()=2: slots 0,1,4,5.
    f.engine->mLoadSlot["req1"][0] = {0, 1, 4, 5};

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMerge(0, 0, hbmSlotIndex, "req1", 0);

    ASSERT_EQ(hbmSlotIndex.size(), 4u);
    // token idx 0: slot 0 -> hbmIdx=blockTable[0]=10, hbmSlot=0 -> 10*4+0=40
    EXPECT_EQ(hbmSlotIndex[0], 40ull);
    // token idx 1: slot 1 -> hbmIdx=blockTable[0]=10, hbmSlot=1 -> 10*4+1=41
    EXPECT_EQ(hbmSlotIndex[1], 41ull);
    // token idx 2 (global 4): slot 4 -> 4/4=1 -> blockTable[1]=20, hbmSlot=0 -> 20*4+0=80
    EXPECT_EQ(hbmSlotIndex[2], 80ull);
    // token idx 3 (global 5): slot 5 -> 5/4=1 -> blockTable[1]=20, hbmSlot=1 -> 81
    EXPECT_EQ(hbmSlotIndex[3], 81ull);
    f.TearDown();
}

TEST(UCMMergeFlow, StoreLoadFailureFallsBackToZeroedOwnedTokens)
{
    MergeFixture f;
    f.Setup();
    f.SeedMissScenario();
    f.engine->mLoadSlot["req1"][0] = {0, 1, 4, 5};
    f.store->loadOnTokenFails = true;
    // Clear shared buffers so we can detect the zero-fill fallback.
    std::memset(f.sharedK.data(), 0xCC, f.sharedK.size());
    std::memset(f.sharedV.data(), 0xDD, f.sharedV.size());

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMerge(0, 0, hbmSlotIndex, "req1", 0);

    ASSERT_EQ(hbmSlotIndex.size(), 4u);
    // hbmSlotIndex is still populated for all tokens (all-rank invariant).
    EXPECT_EQ(hbmSlotIndex[0], 40ull);
    // Owned tokens (rank 0 owns context index 0 since (0+0)%1==0==mRank) are zeroed.
    for (size_t i = 0; i < 4; ++i) {
        for (int b = 0; b < 4; ++b) {  // 4 bytes per token
            EXPECT_EQ(f.sharedK[i * 4 + b], 0u)
                << "owned K token " << i << " byte " << b << " not zeroed";
        }
    }
    f.TearDown();
}

TEST(UCMMergeFlow, EmptyMissListIsEarlyReturn)
{
    MergeFixture f;
    f.Setup();
    f.SeedMissScenario();
    f.engine->mMissGlobalToken["req1"][0].clear();  // no misses
    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMerge(0, 0, hbmSlotIndex, "req1", 0);
    // ValidateUCMMergeRequest returns false on empty misses; nothing written.
    EXPECT_TRUE(hbmSlotIndex.empty());
    f.TearDown();
}

TEST(UCMMergeFlow, NonUcmBlockUsesLastKVBlockAddress)
{
    MergeFixture f;
    f.Setup();
    f.SeedMissScenario();
    // Add a non-UCM block (blockIdx 2) by extending misses with tokens 8,9
    // and registering its K/V address.
    f.engine->mMissGlobalToken["req1"][0] = {0, 1, 4, 5, 8, 9};
    // hbmSlotIndex encoding = blockTable[slotID / blockSize] * blockSize + slotID % blockSize.
    // To route tokens 8,9 (idx 4,5) into block 2 (blockTable[2]=30), their slotID
    // must be 8,9 so slotID/blockSize == 2.
    f.engine->mLoadSlot["req1"][0] = {0, 1, 4, 5, 8, 9};
    f.engine->mReqUseBlockTable["req1"] = {10, 20, 30};
    f.nonUcmK.assign(16, 0x77);
    f.nonUcmV.assign(16, 0x88);
    auto& blockMap = f.engine->mLastKVBlockAddress["req1"];
    blockMap[2].kAddr = reinterpret_cast<uintptr_t>(f.nonUcmK.data());
    blockMap[2].vAddr = reinterpret_cast<uintptr_t>(f.nonUcmV.data());

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMerge(0, 0, hbmSlotIndex, "req1", 0);

    ASSERT_EQ(hbmSlotIndex.size(), 6u);
    // Tokens 8,9 (idx 4,5): slotID 8,9 -> blockTable[8/4=2]=30 -> 30*4+0=120, 30*4+1=121
    EXPECT_EQ(hbmSlotIndex[4], 120ull);
    EXPECT_EQ(hbmSlotIndex[5], 121ull);
    f.TearDown();
}

/* ---- Whole-batch merge (one LoadOnToken task per leg for all requests) ---- */

TEST(UCMMergeFlow, BatchTwoRequestsSubmitOneTaskPerLegWithCorrectAddresses)
{
    MergeFixture f;
    f.Setup();
    f.SeedTwoBlockRequest("req0");
    f.SeedTwoBlockRequest("req1");

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMergeBatch(f.MakeBatchTaskInfo({"req0", "req1"}), {0, 1}, 0, 0,
        hbmSlotIndex);

    /* Exactly one task per leg (K then V), each concatenating both requests'
     * owned blocks: req0 block0/block1 then req1 block0/block1. The call log
     * also contains the Wait() calls that follow each LoadOnToken. */
    std::vector<const MockStoreV1::CallRecord*> loads;
    for (const auto& call : f.store->calls) {
        if (call.name == "LoadOnToken") { loads.push_back(&call); }
    }
    ASSERT_EQ(loads.size(), 2u);
    const auto& kTask = loads[0]->task;
    const auto& kFilters = loads[0]->filters;
    ASSERT_EQ(kTask.size(), 4u);
    ASSERT_EQ(kFilters.size(), 4u);
    /* Per-shard dst = the block's landing window in the shared staging:
     * req0 tokens occupy [0,4), req1 tokens [4,8); block1 of each request
     * lands at tokenBase 2. */
    EXPECT_EQ(kTask[0].addrs[0], f.sharedK.data());
    EXPECT_EQ(kTask[1].addrs[0], f.sharedK.data() + 2 * 4);
    EXPECT_EQ(kTask[2].addrs[0], f.sharedK.data() + 4 * 4);
    EXPECT_EQ(kTask[3].addrs[0], f.sharedK.data() + 6 * 4);
    const auto& vTask = loads[1]->task;
    ASSERT_EQ(vTask.size(), 4u);
    EXPECT_EQ(vTask[0].addrs[0], f.sharedV.data());
    EXPECT_EQ(vTask[2].addrs[0], f.sharedV.data() + 4 * 4);
    /* K filter of shard 0 (req0 block0, tokens 0,1): offsets are in-block
     * token offsets, dstOffsets are relative to the shard dst base. */
    ASSERT_EQ(kFilters[0].ranges.size(), ucmprefetch::kStoreTensorCount);
    ASSERT_EQ(kFilters[0].ranges[0].size(), 2u);
    EXPECT_EQ(kFilters[0].ranges[0][0].offset, 0u);
    EXPECT_EQ(kFilters[0].ranges[0][0].length, 4u);
    EXPECT_EQ(kFilters[0].ranges[0][0].dstOffset, 0u);
    EXPECT_EQ(kFilters[0].ranges[0][1].offset, 4u);
    EXPECT_EQ(kFilters[0].ranges[0][1].dstOffset, 4u);
    /* Shard 1 (req0 block1, global tokens 4,5 -> in-block offsets 4%4=0,
     * 5%4=1 -> byte offsets 0,4). */
    EXPECT_EQ(kFilters[1].ranges[0][0].offset, 0u);
    EXPECT_EQ(kFilters[1].ranges[0][1].offset, 4u);

    /* Mock fill = shard index + 1 (K leg): [0,8)=0x01 [8,16)=0x02 [16,24)=0x03
     * [24,32)=0x04; V leg = shard index + 2. */
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(f.sharedK[i], 1u + static_cast<uint8_t>(i / 8)) << "shared K byte " << i;
        EXPECT_EQ(f.sharedV[i], 2u + static_cast<uint8_t>(i / 8)) << "shared V byte " << i;
    }

    ASSERT_EQ(hbmSlotIndex.size(), 8u);
    EXPECT_EQ(f.engine->mLoadNum[0], 8u);
    /* Both requests used slots 0,1,4,5 -> indices {40,41,80,81} twice. */
    EXPECT_EQ(hbmSlotIndex[0], 40ull);
    EXPECT_EQ(hbmSlotIndex[3], 81ull);
    EXPECT_EQ(hbmSlotIndex[4], 40ull);
    EXPECT_EQ(hbmSlotIndex[7], 81ull);
    f.TearDown();
}

TEST(UCMMergeFlow, BatchStoreLoadFailureZeroesBothRequests)
{
    MergeFixture f;
    f.Setup();
    f.SeedTwoBlockRequest("req0");
    f.SeedTwoBlockRequest("req1");
    f.store->loadOnTokenFails = true;
    std::memset(f.sharedK.data(), 0xCC, f.sharedK.size());
    std::memset(f.sharedV.data(), 0xDD, f.sharedV.size());

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMergeBatch(f.MakeBatchTaskInfo({"req0", "req1"}), {0, 1}, 0, 0,
        hbmSlotIndex);

    /* All 8 owned tokens of both requests are zero-filled; hbmSlotIndex is
     * still populated for every token (all-rank invariant). */
    ASSERT_EQ(hbmSlotIndex.size(), 8u);
    EXPECT_EQ(f.engine->mLoadNum[0], 8u);
    for (size_t i = 0; i < 8; ++i) {
        for (int b = 0; b < 4; ++b) {
            EXPECT_EQ(f.sharedK[i * 4 + b], 0u) << "owned K token " << i << " byte " << b;
            EXPECT_EQ(f.sharedV[i * 4 + b], 0u) << "owned V token " << i << " byte " << b;
        }
    }
    EXPECT_EQ(hbmSlotIndex[0], 40ull);
    EXPECT_EQ(hbmSlotIndex[7], 81ull);
    f.TearDown();
}

TEST(UCMMergeFlow, BatchMixedUcmAndNonUcmRequestsKeepLocalCopies)
{
    MergeFixture f;
    f.Setup();
    /* req0: plain 2-UCM-block request. */
    f.SeedTwoBlockRequest("req0");
    /* req1: 2 UCM blocks + 1 non-UCM block (tokens 8,9 via slots 8,9,
     * blockTable[2]=30, sourced from mLastKVBlockAddress). */
    f.SeedTwoBlockRequest("req1");
    f.engine->mMissGlobalToken["req1"][0] = {0, 1, 4, 5, 8, 9};
    f.engine->mLoadSlot["req1"][0] = {0, 1, 4, 5, 8, 9};
    f.engine->mReqUseBlockTable["req1"] = {10, 20, 30};
    f.nonUcmK.assign(16, 0x77);
    f.nonUcmV.assign(16, 0x88);
    auto& blockMap = f.engine->mLastKVBlockAddress["req1"];
    blockMap[2].kAddr = reinterpret_cast<uintptr_t>(f.nonUcmK.data());
    blockMap[2].vAddr = reinterpret_cast<uintptr_t>(f.nonUcmV.data());

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMergeBatch(f.MakeBatchTaskInfo({"req0", "req1"}), {0, 1}, 0, 0,
        hbmSlotIndex);

    ASSERT_EQ(hbmSlotIndex.size(), 10u);
    EXPECT_EQ(f.engine->mLoadNum[0], 10u);
    /* req0 window [0,16): UCM spliced (shards 0,1 of the K task). */
    for (int i = 0; i < 8; ++i) { EXPECT_EQ(f.sharedK[i], 0x01u) << "req0 K byte " << i; }
    for (int i = 8; i < 16; ++i) { EXPECT_EQ(f.sharedK[i], 0x02u) << "req0 K byte " << i; }
    /* req1 window [16,40): UCM tokens 0..3 spliced (shards 2,3 -> 0x03/0x04),
     * non-UCM tokens 4,5 locally copied from nonUcmK (0x77). */
    for (int i = 16; i < 24; ++i) { EXPECT_EQ(f.sharedK[i], 0x03u) << "req1 K byte " << i; }
    for (int i = 24; i < 32; ++i) { EXPECT_EQ(f.sharedK[i], 0x04u) << "req1 K byte " << i; }
    for (int i = 32; i < 40; ++i) { EXPECT_EQ(f.sharedK[i], 0x77u) << "req1 K byte " << i; }
    for (int i = 32; i < 40; ++i) { EXPECT_EQ(f.sharedV[i], 0x88u) << "req1 V byte " << i; }
    /* Non-UCM tokens: slotID 8,9 -> blockTable[2]=30 -> 120, 121. */
    EXPECT_EQ(hbmSlotIndex[8], 120ull);
    EXPECT_EQ(hbmSlotIndex[9], 121ull);
    f.TearDown();
}
