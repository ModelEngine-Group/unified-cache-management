/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * DT (end-to-end): CallPrefetchProcess LRU + HBM KV cache byte correctness.
 *
 * Complements dt_call_prefetch_flow (pipeline shape / address contracts) by
 * verifying, per decode step, the DATA the pipeline produces:
 *
 *   - output tokens: the useBlockTable (topkIndices) row, miss gid list and
 *     miss slot list must match a reference LRU model that reproduces
 *     HBMManager semantics exactly (dedup keep-first, hit promotes to head,
 *     miss evicts the LRU tail slot and re-inserts it at head, output pads
 *     with unused slots from the LRU head up to the immutable window size).
 *   - KV cache bytes: with Stub().realScatter the stubbed aclnnInplaceScatter
 *     performs the real host-side dst[index[i][j], j] = src[i][j], so after
 *     every step each of the 16 HBM slots must contain EXACTLY the bytes of
 *     the token the model says is resident there -- loaded either by
 *     LastChunkMove at setup (distinct mock payload per (block, token)) or
 *     by the latest decode-step load (UCM blocks via MockStoreV1
 *     distinctTokenPayload, non-UCM block via the fixture's per-token
 *     payload). A wrong load index, a transposed token inside the staging
 *     window, a K/V leg swap or a stale slot would all show up as a byte
 *     mismatch.
 *
 * Model constants mirror the engine setup: blockSize=4, dram domain gids
 * 0..31, HBM capacity 16 slots (sparse blocks mapped through blockTable),
 * immutable window = spaTopkLen = 4, one layer / one stream / tpSize 1
 * (rank 0 owns every UCM block).
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <random>
#include <set>
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

constexpr int kTokenBytes = 4;  // elemSize(2) * numHead(1) * kDim(2)

void ExpectBytesAt(const uint8_t* actual, const std::array<uint8_t, kTokenBytes>& expected,
                   const std::string& what, size_t index)
{
    SCOPED_TRACE(what + "[" + std::to_string(index) + "]");
    for (int i = 0; i < kTokenBytes; ++i) {
        EXPECT_EQ(actual[i], expected[i]) << what << "[" << index << "] byte " << i;
    }
}

/* ------------------------------------------------------------------------
 * Reference LRU model (oracle) for one request.
 * Reproduces processTopkWithDebug + processTopkMisses + buildTopkOutput and
 * additionally tracks, per HBM slot, the K/V bytes that the cache tensor
 * must hold (from the latest load of the resident gid).
 * ------------------------------------------------------------------------ */
class RefLru {
public:
    static constexpr int kCapacity = 16;  // HBM slots == sparseLen
    static constexpr int kImmutable = 4;  // == spaTopkLen
    static constexpr int kDramGids = 32;  // gids 0..31 (blocks 0..7)
    static constexpr int kUcmBlocks = 7;  // blocks 0..6 are UCM, 7 is not
    static constexpr int kBlockSize = 4;

    struct StepExpect {
        std::vector<int> missGids;
        std::vector<int> missSlots;
        std::vector<int> output;  // expected useBlockTable / topkIndices row
        std::vector<int64_t> loadIndex;
        std::vector<std::array<uint8_t, kTokenBytes>> kRows;  // staging rows
        std::vector<std::array<uint8_t, kTokenBytes>> vRows;  // staging rows
        int scatterDelta = 0;
        int loadOnTokenDelta = 0;
    };

    RefLru(const std::vector<int>& blockTable, const std::vector<uint8_t>& nonUcmK,
           const std::vector<uint8_t>& nonUcmV)
        : blockTable_(blockTable), nonUcmK_(nonUcmK), nonUcmV_(nonUcmV)
    {
        /* Identity preload: slot s <-> gid s for s in 0..15. Gids outside
         * the preloaded window start as absent (-1) -- a zero value would
         * alias slot 0 and wrongly report hits for gids 16..31. Payloads
         * come from LastChunkMove, whose LoadOnToken uses one shard per
         * sparse block b (shard order = block order) with the distinct mock
         * fill: K = b*16 + t*4 + 1, V = b*16 + t*4 + 2. */
        gidSlot_.fill(-1);
        /* Preload LRU order anchored by HBMManager::init (kvcache_hbm_manger.h
         * :94-99): mLruHead=0, mLruTail=capacity-1, mLruNext[i]=i+1 -- the list
         * order is [0, 1, ..., 15], slot 0 is the MRU head and slot 15 the
         * tail. First miss evicts slot 15 (confirmed by dt_call_prefetch_flow:
         * miss {17,18,29} -> mLoadSlot {15,14,13}). */
        for (int s = 0; s < kCapacity; ++s) {
            order_.push_back(s);
            gidSlot_[s] = s;
            slotGid_[s] = s;
            const int b = s / kBlockSize;
            const int t = s % kBlockSize;
            kSlot_[s].fill(static_cast<uint8_t>((b * 16 + t * 4 + 1) & 0xFF));
            vSlot_[s].fill(static_cast<uint8_t>((b * 16 + t * 4 + 2) & 0xFF));
        }
    }

    int slotOf(int gid) const { return (gid >= 0 && gid < kDramGids) ? gidSlot_[gid] : -1; }
    int gidAt(int slot) const { return slotGid_[slot]; }
    /* i-th slot from the LRU head (MRU first). */
    int orderAt(size_t i) const { return order_.at(i); }

    /* Flat cache-token destination of an HBM slot: slot s lives in sparse
     * block s / blockSize, which the request's blockTable maps to a cache
     * block; token position inside = s % blockSize. */
    int64_t flatDst(int slot) const
    {
        return static_cast<int64_t>(blockTable_[slot / kBlockSize]) * kBlockSize +
               (slot % kBlockSize);
    }

    const std::array<uint8_t, kTokenBytes>& slotK(int slot) const { return kSlot_[slot]; }
    const std::array<uint8_t, kTokenBytes>& slotV(int slot) const { return vSlot_[slot]; }

    /* Advances the model by one decode step over `topk` (topkLen entries).
     * `shardBase` is this request's first shard ordinal inside the
     * whole-batch LoadOnToken task (0 for a single-request batch). */
    StepExpect Apply(const std::vector<int>& topk, int shardBase = 0)
    {
        StepExpect e;
        std::set<int> seen;
        std::vector<int> dedupSlots;  // slot per deduped entry, -1 until loaded
        std::vector<size_t> missAt;   // dedup position of each miss
        for (int gid : topk) {
            if (!seen.insert(gid).second) { continue; }  // duplicate: keep first
            const int slot = slotOf(gid);
            if (slot >= 0) {
                moveToFront(slot);
                dedupSlots.push_back(slot);
            } else {
                e.missGids.push_back(gid);
                missAt.push_back(dedupSlots.size());
                dedupSlots.push_back(-1);
            }
        }
        /* Misses in order: evict the LRU tail slot, move in, promote to head
         * (processTopkMisses removes the victim from the list and re-inserts
         * it at the head). */
        for (size_t m = 0; m < e.missGids.size(); ++m) {
            const int victim = order_.back();
            order_.pop_back();
            const int oldGid = slotGid_[victim];
            if (oldGid >= 0 && oldGid < kDramGids) { gidSlot_[oldGid] = -1; }
            const int gid = e.missGids[m];
            slotGid_[victim] = gid;
            gidSlot_[gid] = victim;
            e.missSlots.push_back(victim);
            order_.push_front(victim);
            dedupSlots[missAt[m]] = victim;
        }
        /* Output = deduped slots (all resolve to valid slots in this
         * workload), then padding from the LRU head skipping used slots
         * until the immutable window is full (buildTopkOutput). */
        std::set<int> used;
        for (int s : dedupSlots) {
            e.output.push_back(s);
            used.insert(s);
        }
        for (int s : order_) {
            if (static_cast<int>(e.output.size()) >= kImmutable) { break; }
            if (used.insert(s).second) { e.output.push_back(s); }
        }
        /* Staging rows: the merge walks blocks in ascending order, tokens in
         * miss order within each block. UCM tokens are spliced by the mock
         * store with fill = (batchShardIdx * 16 + tensorIdx + inBlock*4 + 1)
         * where batchShardIdx counts UCM shards across the whole batch;
         * non-UCM tokens are copied from the fixture payload. */
        if (!e.missGids.empty()) {
            std::map<int, std::vector<size_t>> byBlock;
            for (size_t m = 0; m < e.missGids.size(); ++m) {
                byBlock[e.missGids[m] / kBlockSize].push_back(m);
            }
            int ucmSeen = 0;
            for (const auto& entry : byBlock) {
                const int block = entry.first;
                int shard = -1;
                if (block < kUcmBlocks) { shard = shardBase + ucmSeen++; }
                for (size_t m : entry.second) {
                    const int gid = e.missGids[m];
                    const int t = gid % kBlockSize;
                    const int slot = e.missSlots[m];
                    std::array<uint8_t, kTokenBytes> k, v;
                    if (block < kUcmBlocks) {
                        k.fill(static_cast<uint8_t>((shard * 16 + 0 + t * 4 + 1) & 0xFF));
                        v.fill(static_cast<uint8_t>((shard * 16 + 1 + t * 4 + 1) & 0xFF));
                    } else {
                        for (int i = 0; i < kTokenBytes; ++i) {
                            k[i] = nonUcmK_[t * kTokenBytes + i];
                            v[i] = nonUcmV_[t * kTokenBytes + i];
                        }
                    }
                    e.kRows.push_back(k);
                    e.vRows.push_back(v);
                    kSlot_[slot] = k;
                    vSlot_[slot] = v;
                    e.loadIndex.push_back(flatDst(slot));
                }
            }
            e.scatterDelta = 2;  // K leg + V leg
            e.loadOnTokenDelta = (ucmSeen > 0) ? 2 : 0;
        }
        return e;
    }

    /* Number of UCM blocks among this step's misses (for the shardBase of
     * the next request in a multi-request batch). */
    static int CountUcmBlocks(const StepExpect& e)
    {
        std::set<int> blocks;
        for (int gid : e.missGids) {
            const int b = gid / kBlockSize;
            if (b < kUcmBlocks) { blocks.insert(b); }
        }
        return static_cast<int>(blocks.size());
    }

private:
    void moveToFront(int slot)
    {
        for (auto it = order_.begin(); it != order_.end(); ++it) {
            if (*it == slot) {
                order_.erase(it);
                break;
            }
        }
        order_.push_front(slot);
    }

    std::vector<int> blockTable_;
    const std::vector<uint8_t>& nonUcmK_;
    const std::vector<uint8_t>& nonUcmV_;
    std::array<int, kDramGids> gidSlot_{};
    std::array<int, kCapacity> slotGid_{};
    std::deque<int> order_;  // front = MRU (LRU head)
    std::array<std::array<uint8_t, kTokenBytes>, kCapacity> kSlot_{};
    std::array<std::array<uint8_t, kTokenBytes>, kCapacity> vSlot_{};
};

/* ------------------------------------------------------------------------
 * Fixture: full public-API engine setup, bf16 tensors everywhere (element
 * size matches between cache and H2D buffers so the real-scatter emulator
 * copies full elements), distinct mock payloads, real scatter enabled.
 * ------------------------------------------------------------------------ */
struct KVFixture {
    static constexpr int kLayerNum = 1;
    static constexpr int kStreamNum = 1;
    static constexpr int kBlockSize = 4;
    static constexpr int kNumHead = 1;
    static constexpr int kKDim = 2;
    static constexpr int kVDim = 2;
    static constexpr int kElemSize = 2;
    static constexpr int kSpaTopkLen = 4;
    static constexpr int kSparseLen = 16;
    static constexpr int kDramTensorCount = 32;
    static constexpr int kUcmBlockCount = 7;
    static constexpr int kKVHBMBlocks = 64;
    static constexpr int kBufferTokens = 64;

    int dpRank;
    std::unique_ptr<TempLogFile> logFile;
    std::unique_ptr<MockStoreV1> store;
    std::unique_ptr<GSAPrefetchEngineC> engine;
    std::vector<uint8_t> nonUcmK;
    std::vector<uint8_t> nonUcmV;
    aclrtStream streamArr[1]{};
    aclrtContext ctx{};
    ThreadInfo threadInfo;

    void Setup()
    {
        acl_stub::Stub().Reset();
        acl_stub::Stub().realScatter = true;
        dpRank = UniqueDpRank();
        CleanupShmFor(dpRank);
        logFile = std::make_unique<TempLogFile>();
        store = std::make_unique<MockStoreV1>();
        store->distinctTokenPayload = true;
        auto model = MakeModelInfo(kLayerNum, /*maxBlockLen=*/4096, kSpaTopkLen, kStreamNum,
                                   kSparseLen, /*prefetchType=*/1, dpRank, /*tpSize=*/1,
                                   /*indexerOffload=*/0, /*isSfaC8=*/0, /*maxBs=*/2);
        auto size = MakeSizeInfo(kBlockSize, kNumHead, kKDim, kVDim, kElemSize);
        std::vector<int> topkLayerMap(kLayerNum, 0);
        engine = std::make_unique<GSAPrefetchEngineC>(
            model, size, /*isLog=*/false,
            /*rank=*/0, /*threadNum=*/1, std::vector<int>{0}, topkLayerMap, logFile->Path());
        engine->SetStoreV1(store.get());

        auto kCaches =
            torch::zeros({kKVHBMBlocks, kBlockSize, kKDim}, torch::dtype(torch::kBFloat16));
        auto vCaches =
            torch::zeros({kKVHBMBlocks, kBlockSize, kVDim}, torch::dtype(torch::kBFloat16));
        auto lastChunkK = torch::zeros({kSparseLen / kBlockSize, kBlockSize, kKDim},
                                       torch::dtype(torch::kBFloat16));
        auto lastChunkV = torch::zeros({kSparseLen / kBlockSize, kBlockSize, kVDim},
                                       torch::dtype(torch::kBFloat16));
        engine->SetKvTensor(kCaches, vCaches, lastChunkK, lastChunkV, /*layerID=*/0);

        auto kHost =
            torch::zeros({kStreamNum, kBufferTokens, kKDim}, torch::dtype(torch::kBFloat16));
        auto vHost =
            torch::zeros({kStreamNum, kBufferTokens, kVDim}, torch::dtype(torch::kBFloat16));
        auto kDev =
            torch::zeros({kStreamNum, kBufferTokens, kKDim}, torch::dtype(torch::kBFloat16));
        auto vDev =
            torch::zeros({kStreamNum, kBufferTokens, kVDim}, torch::dtype(torch::kBFloat16));
        auto idxHost = torch::zeros({kStreamNum, kBufferTokens}, torch::dtype(torch::kInt64));
        auto idxDev = torch::zeros({kStreamNum, kBufferTokens}, torch::dtype(torch::kInt64));
        engine->SetKvBuffer(kHost, vHost, kDev, vDev, idxHost, idxDev);
        ASSERT_NE(engine->mKVPackControl, nullptr);

        /* Non-UCM block (block 7) payload: byte value encodes (leg, token,
         * intra-token position) so a transposed or leg-swapped copy is
         * detectable. */
        nonUcmK.assign(32, 0);
        nonUcmV.assign(32, 0);
        for (int t = 0; t < kBlockSize; ++t) {
            for (int i = 0; i < kTokenBytes; ++i) {
                nonUcmK[t * kTokenBytes + i] = static_cast<uint8_t>(0x40 + t * 8 + i);
                nonUcmV[t * kTokenBytes + i] = static_cast<uint8_t>(0x80 + t * 8 + i);
            }
        }

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

    void SetupRequest(const std::string& reqID, const std::vector<int>& initTopk,
                      const std::vector<int>& blockTable)
    {
        auto topIndices =
            torch::zeros({engine->mTopkIndexerNum, kSpaTopkLen}, torch::dtype(torch::kInt32));
        engine->SetTopkIndices(reqID, topIndices);

        std::vector<BlockId> hashes;
        std::vector<int> blockIdxs;
        std::vector<uintptr_t> kAddrs;
        std::vector<uintptr_t> vAddrs;
        for (int b = 0; b < kUcmBlockCount; ++b) {
            hashes.push_back(BlockId{});
            for (auto& x : hashes.back()) { x = std::byte{static_cast<uint8_t>(b + 1)}; }
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

    /* ---- readers over engine-owned tensors / shared staging ---- */
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
    const uint8_t* CacheK() const
    {
        return reinterpret_cast<const uint8_t*>(engine->mKcacheTensor[0].data_ptr());
    }
    const uint8_t* CacheV() const
    {
        return reinterpret_cast<const uint8_t*>(engine->mVcacheTensor[0].data_ptr());
    }
    static const uint8_t* TensorBytes(const torch::Tensor& t)
    {
        return reinterpret_cast<const uint8_t*>(t.data_ptr());
    }
    const int* TopkIndices(const std::string& reqID) const
    {
        return engine->mAllTopkIndicesTensor.at(reqID)[0].data_ptr<int>();
    }
    HBMManagerLayered* Manager(const std::string& reqID) const
    {
        return engine->mHBMMangers.at(reqID)[0].get();
    }
    size_t LoadOnTokenCount() const
    {
        size_t n = 0;
        for (const auto& call : store->calls) {
            if (call.name == "LoadOnToken") { ++n; }
        }
        return n;
    }
    size_t CacheBytes() const
    {
        return static_cast<size_t>(engine->mKcacheTensor[0].numel()) *
               static_cast<size_t>(engine->mKcacheTensor[0].element_size());
    }
};

/* Verify one decode step of a single request against the model: vectors,
 * staging/H2D/load-index rows (at this request's window offset inside the
 * batch staging) and every HBM slot's cache bytes. mLoadNum is the
 * whole-batch total and is asserted by the callers. */
void VerifySingleRequestStep(KVFixture& f, const RefLru& model, const RefLru::StepExpect& e,
                             const std::string& reqID, const std::string& stepTag,
                             size_t rowOffset = 0)
{
    SCOPED_TRACE(stepTag);
    ASSERT_EQ(f.engine->mMissGlobalToken[reqID][0], e.missGids);
    ASSERT_EQ(f.engine->mLoadSlot[reqID][0], e.missSlots);
    /* useBlockTable published into mAllTopkIndicesTensor. */
    const int* topkRow = f.TopkIndices(reqID);
    ASSERT_EQ(static_cast<size_t>(RefLru::kImmutable), e.output.size());
    for (int i = 0; i < RefLru::kImmutable; ++i) {
        EXPECT_EQ(topkRow[i], e.output[i]) << reqID << " topkIndices[" << i << "]";
    }
    for (size_t i = 0; i < e.loadIndex.size(); ++i) {
        const size_t row = rowOffset + i;
        EXPECT_EQ(f.LoadIndexHost()[row], e.loadIndex[i]) << "loadIndexHost[" << row << "]";
        EXPECT_EQ(f.LoadIndexDevice()[row], e.loadIndex[i]) << "loadIndexDevice[" << row << "]";
        /* staging -> H2D host -> H2D device rows. */
        ExpectBytesAt(f.SharedK() + row * kTokenBytes, e.kRows[i], "stagingK", row);
        ExpectBytesAt(f.SharedV() + row * kTokenBytes, e.vRows[i], "stagingV", row);
        ExpectBytesAt(KVFixture::TensorBytes(f.engine->mKBufferHost[0]) + row * kTokenBytes,
                      e.kRows[i], "h2dKHost", row);
        ExpectBytesAt(KVFixture::TensorBytes(f.engine->mVBufferHost[0]) + row * kTokenBytes,
                      e.vRows[i], "h2dVHost", row);
        ExpectBytesAt(KVFixture::TensorBytes(f.engine->mKBufferDevice[0]) + row * kTokenBytes,
                      e.kRows[i], "h2dKDev", row);
        ExpectBytesAt(KVFixture::TensorBytes(f.engine->mVBufferDevice[0]) + row * kTokenBytes,
                      e.vRows[i], "h2dVDev", row);
    }
    /* Manager bookkeeping vs the model for the whole gid domain. */
    HBMManagerLayered* mgr = f.Manager(reqID);
    for (int gid = 0; gid < RefLru::kDramGids; ++gid) {
        EXPECT_EQ(mgr->getSlotForGlobal(gid), model.slotOf(gid)) << "gid " << gid;
    }
    EXPECT_TRUE(mgr->validateLruChain());
    EXPECT_TRUE(mgr->validateMapping());
    /* The real-scatter emulation must have placed every slot's resident
     * token bytes into the cache tensor. */
    for (int slot = 0; slot < RefLru::kCapacity; ++slot) {
        const int64_t flat = model.flatDst(slot);
        ExpectBytesAt(f.CacheK() + flat * kTokenBytes, model.slotK(slot), "cacheK",
                      static_cast<size_t>(slot));
        ExpectBytesAt(f.CacheV() + flat * kTokenBytes, model.slotV(slot), "cacheV",
                      static_cast<size_t>(slot));
    }
}

}  // namespace

/* =====================================================================
 * Test 1: seeded multi-step churn through the full CallPrefetchProcess.
 * 40 steps of random topk (len 1..4, gids 0..31, duplicates possible)
 * drive hits, misses, evictions and re-loads; every step is verified
 * against the reference model down to the cache bytes.
 * ===================================================================== */
TEST(PrefetchKVLruDt, SeededChurnKeepsLruSlotsAndCacheBytesInSync)
{
    KVFixture f;
    f.Setup();
    const std::vector<int> blockTable = {10, 20, 30, 40};
    f.SetupRequest("req1", {0}, blockTable);
    ASSERT_NE(f.Manager("req1"), nullptr);

    /* Setup sanity: LastChunkMove spliced the four sparse blocks into the
     * last-chunk staging with per-(block, token) distinct payloads and
     * H2D-copied them into the cache at the blockTable offsets. */
    ASSERT_EQ(f.LoadOnTokenCount(), 2u);  // K leg + V leg
    RefLru model(blockTable, f.nonUcmK, f.nonUcmV);
    for (int slot = 0; slot < RefLru::kCapacity; ++slot) {
        const int64_t flat = model.flatDst(slot);
        ASSERT_EQ(
            0, std::memcmp(f.CacheK() + flat * kTokenBytes, model.slotK(slot).data(), kTokenBytes))
            << "setup cache K slot " << slot;
        ASSERT_EQ(
            0, std::memcmp(f.CacheV() + flat * kTokenBytes, model.slotV(slot).data(), kTokenBytes))
            << "setup cache V slot " << slot;
    }

    std::mt19937 rng(20261111);
    std::uniform_int_distribution<int> gidDist(0, RefLru::kDramGids - 1);
    for (int step = 0; step < 40; ++step) {
        const int n = 1 + (gidDist(rng) % RefLru::kImmutable);
        std::vector<int> topk;
        topk.reserve(RefLru::kImmutable);
        for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
        const RefLru::StepExpect e = model.Apply(topk);

        const size_t loadsBefore = f.LoadOnTokenCount();
        const int scatterBefore = acl_stub::Stub().inplaceScatterCount.load();
        TaskInfo task = f.MakeFlowTaskInfo({"req1"}, {topk});
        f.engine->CallPrefetchProcess(task, f.threadInfo);

        EXPECT_EQ(f.LoadOnTokenCount() - loadsBefore, static_cast<size_t>(e.loadOnTokenDelta))
            << "step " << step;
        EXPECT_EQ(acl_stub::Stub().inplaceScatterCount.load() - scatterBefore, e.scatterDelta)
            << "step " << step;
        /* Single-request batch: mLoadNum equals this request's row count. */
        EXPECT_EQ(f.engine->mLoadNum[0], static_cast<uint32_t>(e.loadIndex.size()))
            << "step " << step;
        VerifySingleRequestStep(f, model, e, "req1", "step " + std::to_string(step));
    }
    f.TearDown();
}

/* =====================================================================
 * Test 2: two requests in one batch. Each request has its own manager
 * (own LRU) and its own blockTable, so the two requests scatter into
 * disjoint cache regions. The whole-batch merge lays out req1's staging
 * window before req2's, and the single LoadOnToken task concatenates the
 * UCM shards of both requests (shard index continues across requests).
 * ===================================================================== */
TEST(PrefetchKVLruDt, TwoRequestBatchScattersIntoDisjointCacheRegions)
{
    KVFixture f;
    f.Setup();
    const std::vector<int> table1 = {10, 20, 30, 40};
    const std::vector<int> table2 = {11, 21, 31, 41};
    f.SetupRequest("req1", {0}, table1);
    f.SetupRequest("req2", {0}, table2);

    RefLru m1(table1, f.nonUcmK, f.nonUcmV);
    RefLru m2(table2, f.nonUcmK, f.nonUcmV);

    /* req1: miss 17 (block 4), miss 20 (block 5), miss 29 (non-UCM block 7),
     * hit 3.  req2: miss 21 (block 5), miss 28 (non-UCM), hit 5 + dup 5. */
    const std::vector<int> topk1 = {17, 20, 29, 3};
    const std::vector<int> topk2 = {21, 5, 28, 5};
    const RefLru::StepExpect e1 = m1.Apply(topk1, /*shardBase=*/0);
    const RefLru::StepExpect e2 = m2.Apply(topk2, /*shardBase=*/RefLru::CountUcmBlocks(e1));

    ASSERT_EQ(e1.missGids, (std::vector<int>{17, 20, 29}));
    ASSERT_EQ(e2.missGids, (std::vector<int>{21, 28}));
    /* Batch staging layout: req1 rows then req2 rows; load index likewise. */
    std::vector<int64_t> loadIndex = e1.loadIndex;
    std::vector<std::array<uint8_t, kTokenBytes>> kRows = e1.kRows;
    std::vector<std::array<uint8_t, kTokenBytes>> vRows = e1.vRows;
    loadIndex.insert(loadIndex.end(), e2.loadIndex.begin(), e2.loadIndex.end());
    kRows.insert(kRows.end(), e2.kRows.begin(), e2.kRows.end());
    vRows.insert(vRows.end(), e2.vRows.begin(), e2.vRows.end());

    const size_t loadsBefore = f.LoadOnTokenCount();
    const int scatterBefore = acl_stub::Stub().inplaceScatterCount.load();
    TaskInfo task = f.MakeFlowTaskInfo({"req1", "req2"}, {topk1, topk2});
    f.engine->CallPrefetchProcess(task, f.threadInfo);

    /* One LoadOnToken task per leg covering BOTH requests' UCM blocks
     * (req1: blocks 4,5; req2: block 5). */
    EXPECT_EQ(f.LoadOnTokenCount() - loadsBefore, 2u);
    EXPECT_EQ(acl_stub::Stub().inplaceScatterCount.load() - scatterBefore, 2);
    EXPECT_EQ(f.engine->mLoadNum[0], static_cast<uint32_t>(loadIndex.size()));
    for (size_t i = 0; i < loadIndex.size(); ++i) {
        EXPECT_EQ(f.LoadIndexHost()[i], loadIndex[i]) << "batch loadIndexHost[" << i << "]";
        EXPECT_EQ(f.LoadIndexDevice()[i], loadIndex[i]) << "batch loadIndexDevice[" << i << "]";
        ExpectBytesAt(f.SharedK() + i * kTokenBytes, kRows[i], "batch stagingK", i);
        ExpectBytesAt(f.SharedV() + i * kTokenBytes, vRows[i], "batch stagingV", i);
    }
    /* Per-request vectors / managers / cache regions (region2 is disjoint
     * from region1, so each model's slot bytes pin its own region). req2's
     * staging window starts after req1's rows (whole-batch merge). */
    VerifySingleRequestStep(f, m1, e1, "req1", "two-req req1", /*rowOffset=*/0);
    VerifySingleRequestStep(f, m2, e2, "req2", "two-req req2", e1.loadIndex.size());
    f.TearDown();
}

/* =====================================================================
 * Test 3: an all-hit step must not touch the cache: no loads, no scatter,
 * byte-identical cache tensors afterwards.
 * ===================================================================== */
TEST(PrefetchKVLruDt, AllHitStepLeavesCacheBytesUntouched)
{
    KVFixture f;
    f.Setup();
    const std::vector<int> blockTable = {10, 20, 30, 40};
    f.SetupRequest("req1", {0}, blockTable);
    RefLru model(blockTable, f.nonUcmK, f.nonUcmV);

    /* Churn a few steps so the cache holds post-setup payloads. */
    const std::vector<std::vector<int>> warmup = {
        {0, 17, 18, 29},
        {17, 18},
        {19, 30, 21, 31},
    };
    for (size_t i = 0; i < warmup.size(); ++i) {
        const RefLru::StepExpect e = model.Apply(warmup[i]);
        TaskInfo task = f.MakeFlowTaskInfo({"req1"}, {warmup[i]});
        f.engine->CallPrefetchProcess(task, f.threadInfo);
        EXPECT_EQ(f.engine->mLoadNum[0], static_cast<uint32_t>(e.loadIndex.size()))
            << "warmup " << i;
        VerifySingleRequestStep(f, model, e, "req1", "warmup " + std::to_string(i));
    }

    /* Snapshot both cache tensors. */
    const size_t cacheBytes = f.CacheBytes();
    std::vector<uint8_t> kBefore(f.CacheK(), f.CacheK() + cacheBytes);
    std::vector<uint8_t> vBefore(f.CacheV(), f.CacheV() + cacheBytes);

    /* All-hit step: the four most recently used tokens. */
    std::vector<int> hitTopk;
    for (int i = 0; i < RefLru::kImmutable; ++i) {
        hitTopk.push_back(model.gidAt(model.orderAt(static_cast<size_t>(i))));
    }
    for (int gid : hitTopk) { ASSERT_GE(model.slotOf(gid), 0); }
    const RefLru::StepExpect e = model.Apply(hitTopk);
    ASSERT_TRUE(e.missGids.empty());

    const size_t loadsBefore = f.LoadOnTokenCount();
    const int scatterBefore = acl_stub::Stub().inplaceScatterCount.load();
    TaskInfo task = f.MakeFlowTaskInfo({"req1"}, {hitTopk});
    f.engine->CallPrefetchProcess(task, f.threadInfo);

    EXPECT_EQ(f.LoadOnTokenCount(), loadsBefore);
    EXPECT_EQ(acl_stub::Stub().inplaceScatterCount.load(), scatterBefore);
    EXPECT_EQ(f.engine->mLoadNum[0], 0u);
    EXPECT_TRUE(f.engine->mMissGlobalToken["req1"][0].empty());
    EXPECT_TRUE(f.engine->mLoadSlot["req1"][0].empty());
    EXPECT_EQ(0, std::memcmp(f.CacheK(), kBefore.data(), cacheBytes));
    EXPECT_EQ(0, std::memcmp(f.CacheV(), vBefore.data(), cacheBytes));
    /* topkIndices still resolves to the hit slots (+ padding). */
    const int* topkRow = f.TopkIndices("req1");
    for (int i = 0; i < RefLru::kImmutable; ++i) {
        EXPECT_EQ(topkRow[i], e.output[i]) << "all-hit topkIndices[" << i << "]";
    }
    f.TearDown();
}
