/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * MST: IndexerLoadProcess decode-step correctness (indexer offload path).
 *
 * Complements mst_indexer_flow (init path: SetIndexerMap / UpdateIndexerMap /
 * IndexerInitLoad) by driving the DECODE task body directly and verifying,
 * per layer, the data it produces:
 *
 *   - output block table: the sorted (topkIdx -> NPU block) window written
 *     into mIndexerLoadResultT (the Python-shared prefetch block-table
 *     tensor): hits keep their physical block, misses are allocated from the
 *     free-slot list (mPrefetchIdx) in order.
 *   - token bookkeeping: mPrefetchBlockTableIdx (the topkIdx list consumed by
 *     the next SubmitTaskIndexer), mDocsTables/mBlocksMap remapping
 *     (LoadKVToHBM) and the rebuilt free list mPrefetchIdx.
 *   - indexer cache bytes: a UCM miss block must land in the store-loaded
 *     payload (MockStoreV1::Load fill) and a non-UCM miss block must be an
 *     exact aclrtMemcpy of the per-layer slice of the host slab registered
 *     via SetLastIndexerBlock; untouched blocks stay zero; an all-hit step
 *     must not reload anything (byte-identical cache) and a step whose free
 *     list is exhausted must fall back to already-cached blocks
 *     (DealLessPrefetchBlock) instead of overwriting live data.
 *
 * Layout: 8 physical NPU blocks x 32 bytes per layer (2 layers, per-layer
 * independent cache buffers), 6 UCM hash blocks (blockIdx 0..5), block 6 is
 * non-UCM with a host slab source. SetIndexerMap installs:
 *   remain  = {0, 1}         -> physical 0, 1
 *   prefetch(free) = {4, 5}   -> physical 6, 7   (evictable candidates)
 *   last = blockSlot.size()-1 = 7 -> physical 3 (useBlockTable tail entry)
 * so mDocsTables = {0->0, 1->1, 4->6, 5->7} and mPrefetchIdx = [4, 5].
 */
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
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

BlockId MakeBlockId(uint8_t v)
{
    BlockId b{};
    for (auto& x : b) { x = std::byte{v}; }
    return b;
}

constexpr int kBlockBytes = 32;  // mIndexerSizeBytesBlock for this fixture
constexpr int kPhysBlocks = 8;    // mDeviceIndexerLen
constexpr int kUcmBlocks = 6;     // blockIdx 0..5 are UCM, 6 is non-UCM

struct IndexerLoadFixture {
    static constexpr int kLayerNum = 2;

    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;
    /* Per-layer independent NPU indexer cache stand-ins. */
    std::vector<uint8_t> cache[kLayerNum];
    /* Host slab backing the non-UCM block (block 6): layer l reads
     * slab[l * kBlockBytes .. +kBlockBytes). */
    std::vector<uint8_t> slab;
    torch::Tensor dsa;  // (layers, sparseLen) block-table tensor
    aclrtContext ctx{};

    void Setup()
    {
        acl_stub::Stub().Reset();
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        /* Block-level Load writes bytesPerBlock_ per shard; the indexer block
         * is 32 bytes. */
        store->SetHostBufferPayload(kBlockBytes, kBlockBytes);
        auto model = MakeModelInfo(kLayerNum, /*maxBlockLen=*/4096, /*spaTopkLen=*/32,
                                    /*streamNum=*/1, /*sparseLen=*/8, /*prefetchType=*/1,
                                    /*dpRank=*/dpRank, /*tpSize=*/1, /*indexerOffload=*/1,
                                    /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(/*blockSize=*/4, /*numHead=*/1, /*kDim=*/2, /*vDim=*/2,
                                  /*elementSize=*/2);
        std::vector<int> topkLayerMap(kLayerNum);
        for (int i = 0; i < kLayerNum; ++i) { topkLayerMap[i] = i; }
        engine = std::make_unique<GSAPrefetchEngineC>(model, size, /*isLog=*/false,
                                                      /*rank=*/0, /*threadNum=*/1,
                                                      std::vector<int>{0}, topkLayerMap,
                                                      logFile->Path());
        engine->SetStoreV1(store.get());

        for (int l = 0; l < kLayerNum; ++l) {
            cache[l].assign(static_cast<size_t>(kPhysBlocks) * kBlockBytes, 0);
        }
        engine->mIndexerSizeBytesBlock = kBlockBytes;
        engine->mDeviceIndexerLen = kPhysBlocks;
        engine->mIndexerPtr.assign(kLayerNum, 0);
        for (int l = 0; l < kLayerNum; ++l) {
            engine->mIndexerPtr[l] = reinterpret_cast<uint64_t>(cache[l].data());
        }

        /* Block hashes: 6 UCM blocks (blockIdx 0..5) + non-UCM block 6 whose
         * indexer source lives on the host slab. */
        slab.assign(2 * static_cast<size_t>(kBlockBytes) + kBlockBytes, 0);
        for (size_t i = 0; i < slab.size(); ++i) {
            slab[i] = static_cast<uint8_t>(0xA0 + (i & 0x3F));
        }
        std::vector<BlockId> hashes;
        std::vector<int> blockIdxs;
        std::vector<uintptr_t> addrs;
        for (int b = 0; b < kUcmBlocks; ++b) {
            hashes.push_back(MakeBlockId(static_cast<uint8_t>(b + 1)));
            blockIdxs.push_back(b);
            addrs.push_back(0x1000 + static_cast<uintptr_t>(b) * 0x100);
        }
        blockIdxs.push_back(kUcmBlocks);
        addrs.push_back(reinterpret_cast<uintptr_t>(slab.data()));
        engine->SetBlocksMap("req1", hashes, /*maxIdx=*/32, blockIdxs, addrs, addrs);
        engine->SetLastIndexerBlock("req1", std::vector<int>{kUcmBlocks},
                                    std::vector<uintptr_t>{reinterpret_cast<uintptr_t>(slab.data())});

        dsa = torch::zeros({kLayerNum, 8}, torch::dtype(torch::kInt32));
        /* SetIndexerMapForLayer consumes blockTableList in segment order:
         *   [0 .. prefetchLen)                -> physical blocks of the FREE
         *                                         (prefetch idx) entries
         *   [prefetchLen .. +remainLen)       -> physical blocks of the
         *                                         stay-resident (remain) window
         *   back()                            -> physical block of the last
         *                                         block-table tail entry
         * With blockTable {6, 7, 0, 1, 3}:
         *   idx 4 -> phys 6, idx 5 -> phys 7 (free candidates)
         *   idx 0 -> phys 0, idx 1 -> phys 1 (stay-resident window)
         *   tail idx 7 -> phys 3 */
        std::vector<int> blockTable = {6, 7, 0, 1, 3};
        std::vector<std::vector<int>> remain(kLayerNum, std::vector<int>{0, 1});
        std::vector<std::vector<int>> prefetch(kLayerNum, std::vector<int>{4, 5});
        std::vector<int> blockSlot(kPhysBlocks, 0);
        engine->SetIndexerMap("req1", blockTable, remain, prefetch, blockSlot, dsa);

        aclrtCreateContext(&ctx, 0);
    }

    void TearDown()
    {
        engine.reset();
        store.reset();
        acl_stub::Stub().Reset();
        CleanupShmFor(dpRank);
    }

    /* Builds the per-request indexer topk tensor (one row per layer) and runs
     * one synchronous IndexerLoadProcess over it. */
    void RunDecodeStep(const std::vector<int>& topk)
    {
        TaskInfo info;
        info.reqIdList = {"req1"};
        info.topkLenList = {static_cast<int>(topk.size())};
        auto tensor = torch::zeros({kLayerNum, static_cast<int64_t>(topk.size())},
                                    torch::dtype(torch::kInt32));
        for (int l = 0; l < kLayerNum; ++l) {
            int* row = tensor.data_ptr<int>() + l * static_cast<int>(topk.size());
            for (size_t i = 0; i < topk.size(); ++i) { row[i] = topk[i]; }
        }
        info.indexerTopkList = {tensor};
        ThreadInfo threadInfo(0, 0, nullptr, &ctx, nullptr);
        ASSERT_EQ(engine->IndexerLoadProcess(info, threadInfo), 0);
    }

    size_t LoadCallCount() const
    {
        size_t n = 0;
        for (const auto& c : store->calls) {
            if (c.name == "Load") { ++n; }
        }
        return n;
    }

    const uint8_t* CacheBlock(int layer, int phys) const { return &cache[layer][phys * kBlockBytes]; }
    const uint8_t* SlabSlice(int layer) const { return &slab[static_cast<size_t>(layer) * kBlockBytes]; }

    void ExpectBlockFill(int layer, int phys, uint8_t fill) const
    {
        const uint8_t* block = CacheBlock(layer, phys);
        for (int i = 0; i < kBlockBytes; ++i) {
            EXPECT_EQ(block[i], fill) << "layer " << layer << " phys " << phys << " byte " << i;
        }
    }

    void ExpectBlockMem(int layer, int phys, const uint8_t* expected) const
    {
        const uint8_t* block = CacheBlock(layer, phys);
        for (int i = 0; i < kBlockBytes; ++i) {
            EXPECT_EQ(block[i], expected[i]) << "layer " << layer << " phys " << phys
                                              << " byte " << i;
        }
    }

    void ExpectDsaRow(int layer, const std::vector<int>& expected) const
    {
        const int* row = dsa.data_ptr<int>() + layer * dsa.size(1);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(row[i], expected[i]) << "layer " << layer << " dsa[" << i << "]";
        }
    }

    void ExpectIdxList(int layer, const std::vector<int>& expected) const
    {
        const auto& list = engine->mPrefetchBlockTableIdx.at("req1")[layer];
        ASSERT_EQ(list.size(), expected.size()) << "layer " << layer;
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(list[i], expected[i]) << "layer " << layer << " idx[" << i << "]";
        }
    }

    void ExpectFreeList(int layer, const std::vector<int>& expected) const
    {
        const auto& list = engine->mPrefetchIdx.at("req1")[layer];
        ASSERT_EQ(list.size(), expected.size()) << "layer " << layer << " free size";
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(list[i], expected[i]) << "layer " << layer << " free[" << i << "]";
        }
    }

    /* mDocsTables as sorted (idx -> phys) pairs. */
    void ExpectDocTable(int layer, const std::vector<std::pair<int, int>>& expected) const
    {
        const auto& table = engine->mDocsTables.at("req1")[layer];
        ASSERT_EQ(table.size(), expected.size()) << "layer " << layer << " doc size";
        size_t i = 0;
        for (const auto& kv : table) {  // std::map iterates in idx order
            EXPECT_EQ(kv.first, expected[i].first) << "layer " << layer << " key " << i;
            EXPECT_EQ(kv.second, expected[i].second) << "layer " << layer << " val " << i;
            ++i;
        }
    }

    /* Baseline state right after SetIndexerMap. */
    void ExpectInitialState() const
    {
        for (int l = 0; l < kLayerNum; ++l) {
            ExpectDocTable(l, {{0, 0}, {1, 1}, {4, 6}, {5, 7}});
            ExpectFreeList(l, {4, 5});
            ExpectIdxList(l, {0, 1, 7});
            ExpectDsaRow(l, {0, 1, 3});
        }
    }
};

}  // namespace

/* =====================================================================
 * Test 1: one decode step over topk {0, 2, 6}.
 *   0 hit (phys 0), 2 miss -> free phys 6 (UCM, store-loaded fill 0x01),
 *   6 miss -> free phys 7 (non-UCM, aclrtMemcpy of the per-layer slab
 *   slice). Output block table must keep the hit at phys 0 and place the
 *   misses on the freed candidates; the free list shrinks to the untouched
 *   idx 1.
 * ===================================================================== */
TEST(IndexerLoadProcess, DecodeStepAllocatesFreeBlocksAndLoadsCacheBytes)
{
    IndexerLoadFixture f;
    f.Setup();
    f.ExpectInitialState();

    const int memcpyBefore = acl_stub::Stub().memcpyCount.load();
    f.RunDecodeStep({0, 2, 6});

    for (int l = 0; l < IndexerLoadFixture::kLayerNum; ++l) {
        /* Output block table: (0->0, 2->6, 6->7) sorted by idx. */
        f.ExpectDsaRow(l, {0, 6, 7});
        f.ExpectIdxList(l, {0, 2, 6});
        /* Doc table remapped: idx 4/5 evicted, idx 2/6 installed. */
        f.ExpectDocTable(l, {{0, 0}, {1, 1}, {2, 6}, {6, 7}});
        EXPECT_EQ(f.engine->mBlocksMap.at("req1")[l].at(6), 2);
        EXPECT_EQ(f.engine->mBlocksMap.at("req1")[l].at(7), 6);
        /* Free list = doc entries not referenced by this step's topk. */
        f.ExpectFreeList(l, {1});
        /* Cache bytes: UCM miss block got the store payload (task shard 0 ->
         * fill 1); non-UCM miss block got this layer's slab slice; hit and
         * untouched blocks stay zero. */
        f.ExpectBlockFill(l, 6, 0x01);
        f.ExpectBlockMem(l, 7, f.SlabSlice(l));
        f.ExpectBlockFill(l, 0, 0x00);
        f.ExpectBlockFill(l, 1, 0x00);
        f.ExpectBlockFill(l, 3, 0x00);
    }
    /* One store Load per layer (the UCM miss) + one aclrtMemcpy per layer
     * (the non-UCM miss). */
    EXPECT_EQ(f.LoadCallCount(), 2u);
    EXPECT_EQ(acl_stub::Stub().memcpyCount.load() - memcpyBefore, 2);
    f.TearDown();
}

/* =====================================================================
 * Test 2: an all-hit step reloads nothing. After the first step loaded
 * phys 6/7, replaying the same topk must keep the cache byte-identical,
 * issue no store Load / memcpy, and leave every table untouched.
 * ===================================================================== */
TEST(IndexerLoadProcess, AllHitStepLeavesCacheAndTablesUntouched)
{
    IndexerLoadFixture f;
    f.Setup();
    f.RunDecodeStep({0, 2, 6});

    std::vector<std::vector<uint8_t>> cacheBefore;
    for (int l = 0; l < IndexerLoadFixture::kLayerNum; ++l) {
        cacheBefore.emplace_back(f.cache[l]);
    }
    const auto loadsBefore = f.LoadCallCount();
    const int memcpyBefore = acl_stub::Stub().memcpyCount.load();

    f.RunDecodeStep({0, 2, 6});

    for (int l = 0; l < IndexerLoadFixture::kLayerNum; ++l) {
        EXPECT_EQ(0, std::memcmp(f.cache[l].data(), cacheBefore[l].data(), cacheBefore[l].size()))
            << "layer " << l << " cache bytes changed on an all-hit step";
        f.ExpectDocTable(l, {{0, 0}, {1, 1}, {2, 6}, {6, 7}});
        f.ExpectFreeList(l, {1});
        f.ExpectIdxList(l, {0, 2, 6});
        f.ExpectDsaRow(l, {0, 6, 7});
    }
    EXPECT_EQ(f.LoadCallCount(), loadsBefore);
    EXPECT_EQ(acl_stub::Stub().memcpyCount.load(), memcpyBefore);
    f.TearDown();
}

/* =====================================================================
 * Test 3: free list exhausted -> DealLessPrefetchBlock falls back to
 * cached blocks. Step 1 leaves the free list at [idx 1] (phys 1); step 2
 * topk {0, 3, 4} misses idx 3 and 4 but only phys 1 is evictable: idx 3
 * takes phys 1, idx 4 is dropped and the window is padded with the most
 * significant cached idx (6 -> phys 7). The just-loaded UCM payload at
 * phys 6 survives (it only becomes a free candidate for later steps).
 * ===================================================================== */
TEST(IndexerLoadProcess, ExhaustedFreeListFallsBackToCachedBlocks)
{
    IndexerLoadFixture f;
    f.Setup();
    f.RunDecodeStep({0, 2, 6});

    f.RunDecodeStep({0, 3, 4});

    for (int l = 0; l < IndexerLoadFixture::kLayerNum; ++l) {
        /* Window: hit 0 (phys 0), newly loaded idx 3 (phys 1), fallback
         * idx 6 (phys 7). The dropped miss idx 4 must not appear. */
        f.ExpectDsaRow(l, {0, 1, 7});
        f.ExpectIdxList(l, {0, 3, 6});
        f.ExpectDocTable(l, {{0, 0}, {2, 6}, {3, 1}, {6, 7}});
        /* Free list = doc entries not referenced: idx 2 (phys 6) is the new
         * candidate; the fallback idx 6 is referenced so it stays resident. */
        f.ExpectFreeList(l, {2});
        /* Cache bytes: phys 1 reloaded by the store (UCM miss idx 3), phys 6
         * keeps step-1 payload, phys 7 keeps the step-1 slab slice. */
        f.ExpectBlockFill(l, 1, 0x01);
        f.ExpectBlockFill(l, 6, 0x01);
        f.ExpectBlockMem(l, 7, f.SlabSlice(l));
    }
    /* Step 1: 2 loads (per layer) + 2 memcpys. Step 2: miss idx 3 is UCM so
     * 2 more loads, and no non-UCM copy happens. */
    EXPECT_EQ(f.LoadCallCount(), 4u);
    EXPECT_EQ(acl_stub::Stub().memcpyCount.load(), 2);
    f.TearDown();
}
