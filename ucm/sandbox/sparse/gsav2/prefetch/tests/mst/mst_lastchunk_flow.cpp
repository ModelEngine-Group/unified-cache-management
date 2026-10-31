/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * MST: cross-module flow for LastChunkMove.
 *
 * LastChunkMove spans four collaborating modules:
 *   - GSAPrefetchEngineC (orchestration + HBM slot bookkeeping)
 *   - UC::StoreV1 (mocked) for the UCM block load path
 *   - kvcache_trans_ascend (CopyTrans / Synchronize via ACL stub)
 *   - the per-token CPU copy loop (CopyLastChunkTokens, OpenMP)
 *
 * The test sets up a small deterministic scenario with two UCM blocks and one
 * non-UCM block, drives LastChunkMove and verifies that the final HBM buffer
 * receives the correct per-block payload through the UCM load + CPU merge + H2D
 * copy pipeline, including the store-failure fallback path.
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

struct LastChunkFixture {
    int blockSize = 4;
    int numHead = 1;
    int kDim = 2;
    int vDim = 2;
    int elemSize = 2;  // bytes
    int layerNum = 1;
    // mKVSizeBytesToken = elemSize*numHead*kDim = 2*1*2 = 4
    // mKVSizeBytesTokenRe = elemSize*numHead*vDim = 2*1*2 = 4
    // kBlockBytes = blockSize*kTokenBytes = 4*4 = 16
    // vBlockBytes = 4*4 = 16
    int kTokenBytes = 4;
    int vTokenBytes = 4;
    int kBlockBytes = 16;
    int vBlockBytes = 16;

    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;
    std::vector<uint8_t> kHbm;
    std::vector<uint8_t> vHbm;
    std::vector<uint8_t> kLastChunk;
    std::vector<uint8_t> vLastChunk;
    std::vector<uint8_t> nonUcmK;
    std::vector<uint8_t> nonUcmV;

    void Setup()
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        // Each UCM host buffer must hold K+V for one block.
        store->SetHostBufferPayload(static_cast<size_t>(kBlockBytes + vBlockBytes),
                                    static_cast<size_t>(kBlockBytes));

        auto model = MakeModelInfo(layerNum, /*maxBlockLen=*/4096,
                                   /*spaTopkLen=*/32, /*streamNum=*/1, /*sparseLen=*/16,
                                   /*prefetchType=*/1, /*dpRank=*/dpRank, /*tpSize=*/1,
                                   /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(blockSize, numHead, kDim, vDim, elemSize);
        std::vector<int> topkLayerMap(layerNum);
        for (int i = 0; i < layerNum; ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                      /*rank=*/0, /*threadNum=*/1,
                                                      /*cpuToBind=*/std::vector<int>{0},
                                                      topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());

        // HBM KV cache: 100 slots * 16 bytes.
        kHbm.assign(100 * kBlockBytes, 0);
        vHbm.assign(100 * vBlockBytes, 0);
        engine->mKcachePtr.assign(layerNum, reinterpret_cast<uint64_t>(kHbm.data()));
        engine->mVcachePtr.assign(layerNum, reinterpret_cast<uint64_t>(vHbm.data()));
        // Last chunk staging: 3 blocks worth.
        kLastChunk.assign(3 * kBlockBytes, 0);
        vLastChunk.assign(3 * vBlockBytes, 0);
        engine->mLastCHunkKPtr.assign(layerNum, reinterpret_cast<uint64_t>(kLastChunk.data()));
        engine->mLastCHunkVPtr.assign(layerNum, reinterpret_cast<uint64_t>(vLastChunk.data()));
        engine->mLastChunkNum = 3;
        engine->mKVHBMBlockLen = 100;

        // Non-UCM block 2 source buffers (K + V).
        nonUcmK.assign(kBlockBytes, 0xAB);
        nonUcmV.assign(vBlockBytes, 0xCD);
    }

    void TearDown()
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    void SeedBlocksMap()
    {
        // 2 UCM blocks (hashes) + the non-UCM block address for blockIdx 2.
        std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
        std::vector<int> blockIdxs = {0, 1};
        std::vector<uintptr_t> kAddrs = {
            reinterpret_cast<uintptr_t>(nonUcmK.data()),
            reinterpret_cast<uintptr_t>(nonUcmK.data())};  // placeholder, unused for blocks 0/1
        std::vector<uintptr_t> vAddrs = {reinterpret_cast<uintptr_t>(nonUcmV.data()),
                                         reinterpret_cast<uintptr_t>(nonUcmV.data())};
        engine->SetBlocksMap("req1", hashes, /*maxIdx=*/2, blockIdxs, kAddrs, vAddrs);
        // Register the non-UCM block (blockIdx 2) K/V addresses.
        auto& blockMap = engine->mLastKVBlockAddress["req1"];
        blockMap[2].kAddr = reinterpret_cast<uintptr_t>(nonUcmK.data());
        blockMap[2].vAddr = reinterpret_cast<uintptr_t>(nonUcmV.data());
        // blockTable: which HBM slot each of the 3 blocks lands in.
        engine->mReqUseBlockTable["req1"] = {10, 20, 30};
    }
};

}  // namespace

TEST(LastChunkMoveFlow, UcmAndNonUcmBlocksLandInCorrectHbmSlots)
{
    LastChunkFixture f;
    f.Setup();
    f.SeedBlocksMap();

    // moveMap: 12 tokens spanning blocks 0,1 (UCM) and 2 (non-UCM).
    std::vector<int> moveMap = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    // moveMap is normally `manager.mSlotToGlobal`; it must be blockCount*blockSize.
    ASSERT_EQ(moveMap.size() % f.blockSize, 0u);

    f.engine->LastChunkMove(/*layerID=*/0, "req1", moveMap);

    // UCM block 0 -> HBM slot 10. Mock fills view[0] with 0x01.
    for (int i = 0; i < f.kBlockBytes; ++i) {
        EXPECT_EQ(f.kHbm[10 * f.kBlockBytes + i], 0x01u)
            << "K byte " << i << " mismatch for UCM block 0";
    }
    // UCM block 1 -> HBM slot 20. Mock fills view[1] with 0x02.
    for (int i = 0; i < f.kBlockBytes; ++i) {
        EXPECT_EQ(f.kHbm[20 * f.kBlockBytes + i], 0x02u)
            << "K byte " << i << " mismatch for UCM block 1";
    }
    // Non-UCM block 2 -> HBM slot 30. Source is nonUcmK (0xAB).
    for (int i = 0; i < f.kBlockBytes; ++i) {
        EXPECT_EQ(f.kHbm[30 * f.kBlockBytes + i], 0xABu)
            << "K byte " << i << " mismatch for non-UCM block 2";
    }
    // V path (non-c8 mode): non-UCM V is 0xCD.
    for (int i = 0; i < f.vBlockBytes; ++i) { EXPECT_EQ(f.vHbm[30 * f.vBlockBytes + i], 0xCDu); }
    /* After the ddr-load refactor LastChunkMove uses Load + Wait into the
     * engine-owned mLastChunkLoadBuffer (no LoadToHostBuffer / ReleaseHostBuffer).
     * The mock Load fills each UCM shard's staging slot with a per-shard byte
     * pattern (shard 0 -> 0x01, shard 1 -> 0x02). */
    bool hasLoad = false, hasWait = false, hasRelease = false;
    for (const auto& c : f.store->calls) {
        if (c.name == "Load") { hasLoad = true; }
        if (c.name == "Wait") { hasWait = true; }
        if (c.name == "ReleaseHostBuffer") { hasRelease = true; }
    }
    EXPECT_TRUE(hasLoad);
    EXPECT_TRUE(hasWait);
    EXPECT_FALSE(hasRelease);  // no per-call release after the refactor

    f.TearDown();
}

TEST(LastChunkMoveFlow, EmptyMoveMapIsNoop)
{
    LastChunkFixture f;
    f.Setup();
    f.SeedBlocksMap();
    std::vector<int> empty;
    f.engine->LastChunkMove(0, "req1", empty);
    // HBM untouched.
    for (int i = 0; i < f.kBlockBytes; ++i) { EXPECT_EQ(f.kHbm[10 * f.kBlockBytes + i], 0u); }
    f.TearDown();
}

TEST(LastChunkMoveFlow, NullStoreLogsAndReturns)
{
    LastChunkFixture f;
    f.Setup();
    f.SeedBlocksMap();
    f.engine->SetStoreV1(nullptr);
    std::vector<int> moveMap = {0, 1, 2, 3};
    f.engine->LastChunkMove(0, "req1", moveMap);
    // No store calls expected; HBM untouched.
    EXPECT_TRUE(f.store->calls.empty());
    f.TearDown();
}

TEST(LastChunkMoveFlow, StoreLoadFailureAbortsBeforeCopy)
{
    LastChunkFixture f;
    f.Setup();
    f.SeedBlocksMap();
    f.store->loadFails = true;
    std::vector<int> moveMap = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    f.engine->LastChunkMove(0, "req1", moveMap);
    // On load failure, LastChunkMove returns early; HBM stays untouched.
    for (int i = 0; i < f.kBlockBytes; ++i) { EXPECT_EQ(f.kHbm[10 * f.kBlockBytes + i], 0u); }
    f.TearDown();
}

TEST(LastChunkMoveFlow, UnknownReqIDLogsAndReturns)
{
    LastChunkFixture f;
    f.Setup();
    f.SeedBlocksMap();
    std::vector<int> moveMap = {0, 1, 2, 3};
    f.engine->LastChunkMove(0, "ghost", moveMap);
    f.TearDown();
}

TEST(LastChunkMoveFlow, NegativeTokenInMoveMapReturnsEarly)
{
    LastChunkFixture f;
    f.Setup();
    f.SeedBlocksMap();
    std::vector<int> moveMap = {0, 1, -1, 3};
    f.engine->LastChunkMove(0, "req1", moveMap);
    // BuildLastChunkSources returns false on negative token; HBM untouched.
    for (int i = 0; i < f.kBlockBytes; ++i) { EXPECT_EQ(f.kHbm[10 * f.kBlockBytes + i], 0u); }
    f.TearDown();
}
