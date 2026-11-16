/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * MST: block-map lifecycle flow across SetBlocksMap / SetLastIndexerBlock /
 * SetTokensMap-skip / DelBlocksMap. Exercises the engine's request bookkeeping
 * end-to-end without requiring the full ACL scatter pipeline.
 */
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

struct LifecycleFixture {
    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;

    void Setup(bool indexerOffload = false)
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        auto model = MakeModelInfo(/*layerNum=*/2, 4096, /*spaTopkLen=*/32,
                                   /*streamNum=*/1, /*sparseLen=*/16, 1, /*dpRank=*/dpRank,
                                   /*tpSize=*/1, /*indexerOffload=*/indexerOffload ? 1 : 0,
                                   /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo();
        std::vector<int> topkLayerMap(model[0]);
        for (size_t i = 0; i < topkLayerMap.size(); ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                      /*rank=*/0, /*threadNum=*/1,
                                                      /*cpuToBind=*/std::vector<int>{0},
                                                      topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());
    }
    void TearDown()
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }
};

}  // namespace

TEST(BlockMapLifecycle, SetBlocksMapThenDelBlocksMapClearsState)
{
    // BUG-FOCUS: DelReqIDRun erases mSparseSeqLen but NOT mAllBlocksHash.
    // Documented: assert the sparseSeqLen is cleared while the hash leaks.
    LifecycleFixture f;
    f.Setup();
    std::vector<BlockId> hashes = {MakeBlock(1), MakeBlock(2)};
    std::vector<int> idxs = {0, 1};
    std::vector<uintptr_t> addrs = {0x10, 0x20};
    f.engine->SetBlocksMap("req1", hashes, 2, idxs, addrs, addrs);
    // Pretend SetTokensMap ran by seeding mSparseSeqLen so DelReqIDRun clears it.
    f.engine->mSparseSeqLen["req1"] = 128;
    ASSERT_TRUE(f.engine->mAllBlocksHash.count("req1"));
    f.engine->DelBlocksMap("req1");
    // mSparseSeqLen IS cleared:
    EXPECT_FALSE(f.engine->mSparseSeqLen.count("req1"));
    // mAllBlocksHash is NOT cleared (bug). When fixed, flip to EXPECT_FALSE.
    EXPECT_TRUE(f.engine->mAllBlocksHash.count("req1"));
    f.TearDown();
}

TEST(BlockMapLifecycle, ConcurrentDelRequestsAreSerializedByMutex)
{
    LifecycleFixture f;
    f.Setup();
    f.engine->mSparseSeqLen["req1"] = 1;
    f.engine->mSparseSeqLen["req2"] = 2;
    // DelBlocksMap takes mDelReqMutex internally; concurrent deletes must not
    // corrupt the shared mDelSeqIds set.
    auto t1 = std::thread([&] { f.engine->DelBlocksMap("req1"); });
    auto t2 = std::thread([&] { f.engine->DelBlocksMap("req2"); });
    t1.join();
    t2.join();
    EXPECT_FALSE(f.engine->mSparseSeqLen.count("req1"));
    EXPECT_FALSE(f.engine->mSparseSeqLen.count("req2"));
    f.TearDown();
}

TEST(BlockMapLifecycle, SetLastIndexerBlockFlowWithIndexerOffload)
{
    LifecycleFixture f;
    f.Setup(/*indexerOffload=*/true);
    std::vector<BlockId> hashes = {MakeBlock(1)};
    std::vector<int> idxs = {0};
    std::vector<uintptr_t> addrs = {0x10};
    f.engine->SetBlocksMap("req1", hashes, 1, idxs, addrs, addrs);
    std::vector<int> blockIdxs = {0};
    std::vector<uintptr_t> indexerAddrs = {0xDEADBEEFu};
    f.engine->SetLastIndexerBlock("req1", blockIdxs, indexerAddrs);
    EXPECT_EQ(f.engine->mLastKVBlockAddress["req1"][0].indexerAddr, 0xDEADBEEFu);
    f.TearDown();
}

TEST(BlockMapLifecycle, DelBlocksMapWhileRunningQueuesOnly)
{
    LifecycleFixture f;
    f.Setup();
    f.engine->mIsPrefetchRunning.store(true);
    f.engine->mSparseSeqLen["req1"] = 1;
    f.engine->DelBlocksMap("req1");
    // Running flag means DelReqIDRun is skipped; state remains.
    EXPECT_TRUE(f.engine->mSparseSeqLen.count("req1"));
    EXPECT_TRUE(f.engine->mDelSeqIds.count("req1"));
    f.TearDown();
}
