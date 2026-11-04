/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * MST: cross-module flow for the indexer offload path (IndexerInitLoad,
 * SetIndexerMap, UpdateIndexerMap, AddIndexerMap, DeleteIndexerMap).  These
 * tie together GSAPrefetchEngineC's indexer bookkeeping maps, the UCM store
 * (Load + Wait via mock) and the ACL memcpy path (aclrtMemcpy via stub).
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

torch::Tensor MakeDsaSBT(int layers, int sparseLen)
{
    return torch::zeros({layers, sparseLen}, torch::dtype(torch::kInt32));
}

struct IndexerFixture {
    int layerNum = 2;
    int sparseLen = 8;
    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;
    std::vector<uint8_t> indexerDevice;

    void Setup()
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        auto model = MakeModelInfo(layerNum, 4096, /*spaTopkLen=*/32,
                                   /*streamNum=*/1, sparseLen, 1, /*dpRank=*/dpRank,
                                   /*tpSize=*/1, /*indexerOffload=*/1, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1, /*kDim=*/2,
                                 /*vDim=*/2, /*elementSize=*/2);
        std::vector<int> topkLayerMap(layerNum);
        for (int i = 0; i < layerNum; ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                      /*rank=*/0, /*threadNum=*/1,
                                                      /*cpuToBind=*/std::vector<int>{0},
                                                      topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());

        // Device indexer buffer (32 bytes per block * 8 blocks).
        engine->mIndexerSizeBytesBlock = 32;
        engine->mDeviceIndexerLen = 8;
        indexerDevice.assign(8 * 32, 0);
        engine->mIndexerPtr.assign(layerNum, reinterpret_cast<uint64_t>(indexerDevice.data()));
    }

    void TearDown()
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    void SeedBlocksMap(int blockCount)
    {
        std::vector<BlockId> hashes;
        std::vector<int> blockIdxs;
        std::vector<uintptr_t> addrs;
        for (int i = 0; i < blockCount; ++i) {
            hashes.push_back(MakeBlock(static_cast<uint8_t>(i + 1)));
            blockIdxs.push_back(i);
            addrs.push_back(0x1000 + i);
        }
        engine->SetBlocksMap("req1", hashes, blockCount, blockIdxs, addrs, addrs);
    }
};

}  // namespace

TEST(IndexerFlow, SetIndexerMapPopulatesAllLayers)
{
    IndexerFixture f;
    f.Setup();
    f.SeedBlocksMap(4);
    auto dsaSBT = MakeDsaSBT(f.layerNum, f.sparseLen);
    std::vector<int> blockTable = {0, 1, 2, 3};
    std::vector<std::vector<int>> remain(f.layerNum, std::vector<int>{0, 1});
    std::vector<std::vector<int>> prefetch(f.layerNum, std::vector<int>{2, 3});
    std::vector<int> blockSlot = {10, 20, 30, 40};

    f.engine->SetIndexerMap("req1", blockTable, remain, prefetch, blockSlot, dsaSBT);
    // Each layer must have its maps populated.
    for (int l = 0; l < f.layerNum; ++l) {
        EXPECT_EQ(f.engine->mDocsTables["req1"][l].size(), 4u);
        EXPECT_EQ(f.engine->mBlocksMap["req1"][l].size(), 4u);
    }
    EXPECT_EQ(f.engine->mIndexerSlot["req1"], blockSlot);
    f.TearDown();
}

TEST(IndexerFlow, UpdateIndexerMapAddAppendsAcrossLayers)
{
    IndexerFixture f;
    f.Setup();
    f.SeedBlocksMap(4);
    auto dsaSBT = MakeDsaSBT(f.layerNum, f.sparseLen);
    std::vector<int> blockTable = {0, 1, 2, 3};
    std::vector<std::vector<int>> remain(f.layerNum, std::vector<int>{0, 1});
    std::vector<std::vector<int>> prefetch(f.layerNum, std::vector<int>{2, 3});
    std::vector<int> blockSlot = {10, 20, 30, 40};
    f.engine->SetIndexerMap("req1", blockTable, remain, prefetch, blockSlot, dsaSBT);

    f.engine->UpdateIndexerMap("req1", /*idx=*/99, /*blockID=*/7, "add");
    for (int l = 0; l < f.layerNum; ++l) {
        EXPECT_EQ(f.engine->mUseBlockTableIdx["req1"][l].back(), 99);
    }
    f.TearDown();
}

TEST(IndexerFlow, UpdateIndexerMapLastDelPopsAcrossLayers)
{
    IndexerFixture f;
    f.Setup();
    f.SeedBlocksMap(4);
    auto dsaSBT = MakeDsaSBT(f.layerNum, f.sparseLen);
    std::vector<int> blockTable = {0, 1, 2, 3};
    std::vector<std::vector<int>> remain(f.layerNum, std::vector<int>{0, 1});
    std::vector<std::vector<int>> prefetch(f.layerNum, std::vector<int>{2, 3});
    std::vector<int> blockSlot = {10, 20, 30, 40};
    f.engine->SetIndexerMap("req1", blockTable, remain, prefetch, blockSlot, dsaSBT);

    size_t before = f.engine->mUseBlockTableIdx["req1"][0].size();
    f.engine->UpdateIndexerMap("req1", 0, 0, "lastdel");
    EXPECT_EQ(f.engine->mUseBlockTableIdx["req1"][0].size(), before - 1);
    f.TearDown();
}

TEST(IndexerFlow, IndexerInitLoadLoadsUcmBlocksViaStore)
{
    IndexerFixture f;
    f.Setup();
    f.SeedBlocksMap(2);
    // blockTable[0]=0,1 (ucm); slabIdx tells us block 2 (non-ucm, absent).
    std::vector<std::vector<int>> blockTable(f.layerNum, {0, 1});
    std::vector<std::vector<int>> slabIdx(f.layerNum, {0, 1});
    f.engine->IndexerInitLoad(blockTable, slabIdx, "req1");
    bool hasLoad = false;
    for (const auto& c : f.store->calls) {
        if (c.name == "Load") { hasLoad = true; }
    }
    EXPECT_TRUE(hasLoad);
    f.TearDown();
}

TEST(IndexerFlow, IndexerInitLoadWithoutHashReturnsEarly)
{
    IndexerFixture f;
    f.Setup();
    // Do not seed the hash map.
    std::vector<std::vector<int>> blockTable(f.layerNum, {0});
    std::vector<std::vector<int>> slabIdx(f.layerNum, {0});
    f.engine->IndexerInitLoad(blockTable, slabIdx, "ghost");
    // No store calls because the hash lookup failed.
    EXPECT_TRUE(f.store->calls.empty());
    f.TearDown();
}

TEST(IndexerFlow, IndexerInitLoadStoreFailureIsLoggedNotCrashed)
{
    IndexerFixture f;
    f.Setup();
    f.SeedBlocksMap(2);
    /* After the ddr-load refactor the indexer load path is Load + Wait. A Load
     * failure returns false internally and is logged, not crashed. */
    f.store->loadFails = true;
    std::vector<std::vector<int>> blockTable(f.layerNum, {0, 1});
    std::vector<std::vector<int>> slabIdx(f.layerNum, {0, 1});
    // Should return false internally and not crash.
    f.engine->IndexerInitLoad(blockTable, slabIdx, "req1");
    bool hasLoad = false;
    for (const auto& c : f.store->calls) {
        if (c.name == "Load") { hasLoad = true; }
    }
    EXPECT_TRUE(hasLoad);
    f.TearDown();
}

TEST(IndexerFlow, AddDeleteIndexerMapOnKnownReqIDMutatesMaps)
{
    IndexerFixture f;
    f.Setup();
    f.SeedBlocksMap(4);
    auto dsaSBT = MakeDsaSBT(f.layerNum, f.sparseLen);
    std::vector<int> blockTable = {0, 1, 2, 3};
    std::vector<std::vector<int>> remain(f.layerNum, std::vector<int>{0, 1});
    std::vector<std::vector<int>> prefetch(f.layerNum, std::vector<int>{2, 3});
    std::vector<int> blockSlot = {10, 20, 30, 40};
    f.engine->SetIndexerMap("req1", blockTable, remain, prefetch, blockSlot, dsaSBT);

    f.engine->mIndexerAdd["req1"].emplace_back(50, 7);
    f.engine->AddIndexerMap();
    EXPECT_GT(f.engine->mIndexerSlot["req1"].size(), 4u);
    EXPECT_FALSE(f.engine->mIndexerAdd.count("req1"));

    f.engine->mIndexerDel["req1"] = {0};
    f.engine->DeleteIndexerMap();
    EXPECT_FALSE(f.engine->mIndexerDel.count("req1"));
    f.TearDown();
}
