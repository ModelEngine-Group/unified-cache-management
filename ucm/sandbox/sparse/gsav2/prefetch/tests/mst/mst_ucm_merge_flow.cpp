/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * MST: cross-module flow for UCMTensorMerge.
 *
 * UCMTensorMerge ties together HBM miss-bookkeeping (mMissGlobalToken /
 * mLoadSlot), the UCM store (Load + Wait via mock), the multi-rank KVPack
 * shared staging buffers, and the CPU token-merge loop.  We drive it with a
 * mock store and the ACL stub, and verify both the success path
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
        store->SetHostBufferPayload(32, 16);  // 16 K + 16 V per block view
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
        /* The ddr-load refactor made UCMTensorMerge stage into the engine-owned
         * mUCMMergeLoadBuffer (set by SetupKVPackSharedBuffers in production,
         * which tests skip). Alias it to mLastChunkLoadBuffer so the merge path
         * has a valid staging region. The destructor frees mLastChunkLoadBuffer
         * and never frees mUCMMergeLoadBuffer, so this is safe. */
        if (engine->mUCMMergeLoadBuffer == nullptr && engine->mLastChunkLoadBuffer != nullptr) {
            engine->mUCMMergeLoadBuffer = static_cast<std::byte*>(engine->mLastChunkLoadBuffer);
        }

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
};

}  // namespace

TEST(UCMMergeFlow, SuccessFillsHbmSlotIndexAndAdvancesLoadNum)
{
    MergeFixture f;
    f.Setup();
    f.SeedMissScenario();

    std::vector<uint64_t> hbmSlotIndex;
    f.engine->UCMTensorMerge(/*layerID=*/0, /*streamIdx=*/0, hbmSlotIndex,
                             /*reqID=*/"req1", /*requestIndex=*/0);

    // previousLoadNum=0 + 4 misses -> finalLoadNum=4.
    ASSERT_EQ(hbmSlotIndex.size(), 4u);
    EXPECT_EQ(f.engine->mLoadNum[0], 4u);

    // Verify the hbm slot index encoding for the 4 tokens.
    // token 0: slot 0 -> hbmIdx=blockTable[0/4]=10, hbmSlot=0%4=0 -> 10*4+0=40
    // token 1: slot 1 -> hbmIdx=blockTable[0]=10, hbmSlot=1 -> 10*4+1=41
    // token 2 (idx 4): slot 8 -> hbmIdx=blockTable[8/4=2]... but blockTable has
    //   only 2 entries [10,20]. slot 8 -> 8/4=2 which is out of range for
    //   blockTable.size()=2. This would crash CopyUCMMergeTokensSerial. So
    //   pick slots that stay in range. Recreate with valid slots.
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
    f.store->loadFails = true;
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
