/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * MST: cross-module differential suite for kvcache_hbm_manger_adapt.h.
 *
 * HBMManagerLayered is the tunable extension of HBMManager
 * (kvcache_hbm_manger.h).  Its contract: when init is called with default
 * runtime params (Lru + Uniform, no dedup), every externally observable
 * behaviour must be identical to HBMManager for the same init shape and the
 * same processTopkWithDebug call sequence.  The differential tests below
 * drive both implementations through identical sequences and compare:
 *   - output / missGlobalIds / missSlots per call;
 *   - the full public mapping arrays (mGlobalToSlot, mSlotToGlobal);
 *   - validateLruChain / validateMapping;
 *   - every getter, plus getSlotForGlobal / getGidForSlot probes across the
 *     whole gid/slot domain (regular, pinned-formula and out-of-range);
 *   - mHitRate once the 2047 sampling point has been crossed.
 *
 * The remaining tests cover the adapt-only surface: init runtime-param
 * resolution (evict mode / window plan / decay lambda / window steps /
 * output dedup), the window-size clamping regression (init used
 * `win == mHbmImmutableSize`, a no-op comparison instead of an assignment),
 * decay-mode output padding and windowSteps pruning.
 *
 * The HBMAdaptGolden suite drives an independent "golden" model written
 * from the behaviour contract with naive containers (std::list recency
 * chain, std::map decay state, full backward victim scans, checked fill
 * walks).  It contains none of the header's fast paths (fused LruNode,
 * mEvictTail cache, call-start boundary skip, generation flags), so
 * adapt == golden agreement validates those optimizations themselves.
 * It covers: Lru (incl. the 2048-miss break and the 2047 hit-rate
 * sampling point), decay (ranking / tie-breaks / prune / window clamp /
 * output = real-topk + m-window history + -1 pad), and the
 * outputDedup duplicate-free semantics in both modes.
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <gtest/gtest.h>
#include <list>
#include <map>
#include <random>
#include <set>
#include <vector>
#include "kvcache_hbm_manger.h"
#include "kvcache_hbm_manger_adapt.h"
#include "kvcache_log.h"

namespace {

Logger kNullLogger{};  // logger disabled by default ctor

/* Init shape shared by both implementations under comparison. */
struct Shape {
    int dram;
    int cap;
    int immu;
    int topk;
    int maxTotal;
    bool preload;
};

void InitRef(HBMManager& m, const Shape& s)
{
    m.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal);
}

void InitAdapt(HBMManagerLayered& m, const Shape& s, int layerId = -1,
               int evictMode = HBMManagerLayered::kAutoEvict,
               int windowPlan = HBMManagerLayered::kAutoPlan,
               float decayLambda = HBMManagerLayered::kAutoLambda,
               int windowSteps = HBMManagerLayered::kAutoSteps,
               int outputDedup = HBMManagerLayered::kAutoDedup)
{
    m.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal, layerId, evictMode,
           windowPlan, decayLambda, windowSteps, outputDedup);
}

/* Full externally-observable state comparison (mHitRate excluded: it is only
 * defined after the 2047 sampling point, see the dedicated test below). */
void CompareState(const HBMManager& ref, const HBMManagerLayered& adapt)
{
    ASSERT_TRUE(ref.validateLruChain());
    ASSERT_TRUE(ref.validateMapping());
    ASSERT_TRUE(adapt.validateLruChain());
    ASSERT_TRUE(adapt.validateMapping());

    EXPECT_EQ(ref.isInitialized(), adapt.isInitialized());
    EXPECT_EQ(ref.getDramTensorCount(), adapt.getDramTensorCount());
    EXPECT_EQ(ref.getHbmCapacity(), adapt.getHbmCapacity());
    EXPECT_EQ(ref.getHbmImmutableSize(), adapt.getHbmImmutableSize());
    EXPECT_EQ(ref.getMaxTotalSlots(), adapt.getMaxTotalSlots());
    EXPECT_EQ(ref.getTopkSize(), adapt.getTopkSize());
    EXPECT_EQ(ref.getPinnedSlotCount(), adapt.getPinnedSlotCount());

    ASSERT_EQ(ref.mGlobalToSlot, adapt.mGlobalToSlot);
    ASSERT_EQ(ref.mSlotToGlobal, adapt.mSlotToGlobal);

    const int dram = ref.getDramTensorCount();
    const int cap = ref.getHbmCapacity();
    const int maxTotal = ref.getMaxTotalSlots();
    for (int gid = -2; gid <= dram + (maxTotal - cap) + 3; ++gid) {
        EXPECT_EQ(ref.getSlotForGlobal(gid), adapt.getSlotForGlobal(gid)) << "gid " << gid;
    }
    for (int slot = -1; slot <= maxTotal + 2; ++slot) {
        EXPECT_EQ(ref.getGidForSlot(slot), adapt.getGidForSlot(slot)) << "slot " << slot;
    }
}

/* Drive both implementations with the same topk and compare per-call
 * results plus the full state afterwards. */
void RunBoth(HBMManager& ref, HBMManagerLayered& adapt, const std::vector<int>& topk)
{
    std::vector<int> outRef, outAdapt, missGidRef, missGidAdapt, missSlotRef, missSlotAdapt;
    ref.processTopkWithDebug(topk, outRef, missGidRef, missSlotRef, kNullLogger);
    adapt.processTopkWithDebug(topk, outAdapt, missGidAdapt, missSlotAdapt, kNullLogger);
    ASSERT_EQ(outRef, outAdapt);
    ASSERT_EQ(missGidRef, missGidAdapt);
    ASSERT_EQ(missSlotRef, missSlotAdapt);
    CompareState(ref, adapt);
}

}  // namespace

/* ================= default params => identical to HBMManager ================= */

TEST(HBMAdaptEquivalence, InitStateMatchesWithDefaultParams)
{
    const Shape shapes[] = {
        {1024, 256, 128, 64, 512, true},
        {1024, 256, 128, 64, 512, false},
        {8, 4, 2, 2, 8, true},
        {64, 32, 16, 16, 48, false},
    };
    for (const auto& s : shapes) {
        HBMManager ref;
        HBMManagerLayered adapt;
        InitRef(ref, s);
        InitAdapt(adapt, s);
        ASSERT_TRUE(adapt.isInitialized());
        EXPECT_FALSE(adapt.isDecayMode());
        EXPECT_STREQ(adapt.getEvictModeName(), "lru");
        EXPECT_STREQ(adapt.getWindowPlanName(), "uniform");
        EXPECT_EQ(adapt.getWindowSize(), s.immu);
        CompareState(ref, adapt);
    }
}

TEST(HBMAdaptEquivalence, ScriptedFlowsMatchAcrossBranches)
{
    HBMManager ref;
    HBMManagerLayered adapt;
    InitRef(ref, {8, 4, 2, 2, 8, true});
    InitAdapt(adapt, {8, 4, 2, 2, 8, true});

    /* hits on preloaded identities */
    RunBoth(ref, adapt, {0, 1, 2});
    /* misses trigger eviction from the LRU tail */
    RunBoth(ref, adapt, {5, 6});
    /* in-range pinned gids (slots 4..7 via gids 8..11) */
    RunBoth(ref, adapt, {8, 9, 10});
    /* negative gids occupy dedup positions without emitting slots */
    RunBoth(ref, adapt, {-1, -1, 3});
    /* duplicates of every class: in-range regular + in-range pinned */
    RunBoth(ref, adapt, {5, 5, 8, 8, 9});
    /* out-of-range pinned gid (slot = 12 - 8 + 4 = 8 >= maxTotal 8):
     * raw slot written, no pin, NOT deduped when repeated */
    RunBoth(ref, adapt, {12, 12, 0});
    /* mixed storm */
    RunBoth(ref, adapt, {7, 13, -5, 4, 4, 9, 11, 11, 2, 14});
}

TEST(HBMAdaptEquivalence, MissThresholdBreakMatches)
{
    /* preload=false + gids beyond any mapping => pure misses; both
     * implementations must stop collecting at 2048 misses and resize the
     * returned miss vectors to the actually-loaded prefix identically. */
    HBMManager ref;
    HBMManagerLayered adapt;
    InitRef(ref, {16384, 1024, 512, 64, 4096, false});
    InitAdapt(adapt, {16384, 1024, 512, 64, 4096, false});

    std::vector<int> topk;
    topk.reserve(3000);
    /* regular gid domain ([0, dram)), no preload => every gid is a miss */
    for (int i = 0; i < 3000; ++i) { topk.push_back(i); }

    std::vector<int> outRef, outAdapt, missGidRef, missGidAdapt, missSlotRef, missSlotAdapt;
    ref.processTopkWithDebug(topk, outRef, missGidRef, missSlotRef, kNullLogger);
    adapt.processTopkWithDebug(topk, outAdapt, missGidAdapt, missSlotAdapt, kNullLogger);

    ASSERT_EQ(missGidRef.size(), 2048u);
    EXPECT_EQ(outRef, outAdapt);
    EXPECT_EQ(missGidRef, missGidAdapt);
    EXPECT_EQ(missSlotRef, missSlotAdapt);
    CompareState(ref, adapt);
}

TEST(HBMAdaptEquivalence, HitRateSamplingMatchesPast2047)
{
    HBMManager ref;
    HBMManagerLayered adapt;
    InitRef(ref, {8192, 4096, 2048, 64, 8192, true});
    InitAdapt(adapt, {8192, 4096, 2048, 64, 8192, true});

    /* below the sampling point: adapt defines mHitRate as 0.0f; the
     * original leaves it indeterminate (documented intentional difference,
     * never compared pre-sampling). */
    {
        std::vector<int> oR, oA, mR, mA, sR, sA;
        ref.processTopkWithDebug({0, 1}, oR, mR, sR, kNullLogger);
        adapt.processTopkWithDebug({0, 1}, oA, mA, sA, kNullLogger);
        EXPECT_FLOAT_EQ(adapt.mHitRate, 0.0f);
    }

    /* 3000 distinct REGULAR gids: identities 0..256 hit, 257+ miss.  The
     * deduped index 2047 must be a regular gid for the sampling check to
     * fire (it lives in the regular branch only; a pinned gid at 2047
     * skips it forever).  The miss threshold (2048) breaks the loop later,
     * at gid 2304, without affecting the sampled value. */
    std::vector<int> topk;
    topk.reserve(3000);
    for (int i = 0; i < 3000; ++i) { topk.push_back(i); }

    std::vector<int> outRef, outAdapt, missGidRef, missGidAdapt, missSlotRef, missSlotAdapt;
    ref.processTopkWithDebug(topk, outRef, missGidRef, missSlotRef, kNullLogger);
    adapt.processTopkWithDebug(topk, outAdapt, missGidAdapt, missSlotAdapt, kNullLogger);

    EXPECT_EQ(outRef, outAdapt);
    EXPECT_EQ(missGidRef, missGidAdapt);
    EXPECT_EQ(missSlotRef, missSlotAdapt);
    EXPECT_FLOAT_EQ(ref.mHitRate, adapt.mHitRate);
    /* misses among deduped positions 0..2047: gids 257..2047 = 1791 */
    EXPECT_NEAR(ref.mHitRate, 1.0f - 1791.0f / 2048.0f, 1e-6);
    CompareState(ref, adapt);
}

TEST(HBMAdaptEquivalence, RandomFuzzMatchesAcrossShapes)
{
    const Shape shapes[] = {
        {8, 4, 2, 2, 8, true},
        {1024, 256, 128, 64, 512, true},
        {64, 32, 16, 16, 48, false},
    };
    for (const auto& s : shapes) {
        HBMManager ref;
        HBMManagerLayered adapt;
        InitRef(ref, s);
        InitAdapt(adapt, s);

        std::mt19937 rng(98765 + s.dram);
        /* gid domain spans negatives, regular, in-range pinned and
         * out-of-range pinned so every classify branch is exercised. */
        std::uniform_int_distribution<int> gidDist(-3,
                                                   s.dram + (s.maxTotal - s.cap) + 3);
        std::uniform_int_distribution<int> lenDist(0, 70);
        for (int iter = 0; iter < 200; ++iter) {
            std::vector<int> topk;
            const int n = lenDist(rng);
            topk.reserve(n);
            for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
            ASSERT_NO_FATAL_FAILURE(RunBoth(ref, adapt, topk)) << "shape dram=" << s.dram
                                                               << " iter " << iter;
        }
    }
}

TEST(HBMAdaptEquivalence, AllPinnedEdgeMatches)
{
    /* Pin every available pinned slot, then serve regular misses from the
     * regular zone: the eviction "no evictable slot" dead-drop path and the
     * pinned LRU interleaving must stay identical. */
    HBMManager ref;
    HBMManagerLayered adapt;
    InitRef(ref, {8, 4, 2, 4, 8, false});
    InitAdapt(adapt, {8, 4, 2, 4, 8, false});

    RunBoth(ref, adapt, {8, 9, 10, 11});
    ASSERT_EQ(ref.getPinnedSlotCount(), 4);
    RunBoth(ref, adapt, {5});
    RunBoth(ref, adapt, {8, 5, 6});
}

/* ================= adapt-only: init 参数解析 ================= */

TEST(HBMAdaptConfig, DefaultParamsResolveLruUniform)
{
    HBMManagerLayered m;
    m.init(1024, 256, 128, 64, kNullLogger, true, 512);
    EXPECT_FALSE(m.isDecayMode());
    EXPECT_STREQ(m.getEvictModeName(), "lru");
    EXPECT_STREQ(m.getWindowPlanName(), "uniform");
    EXPECT_EQ(m.getWindowSize(), 128);
    EXPECT_FALSE(m.isOutputDedup());
    /* 显式开启 decay 的实例不受其他实例影响；kAuto 哨兵各项取默认值 */
    HBMManagerLayered second;
    second.init(1024, 256, 128, 64, kNullLogger, true, 512, -1, /*evictMode=*/1);
    EXPECT_TRUE(second.isDecayMode());
    EXPECT_STREQ(second.getEvictModeName(), "decay");
    EXPECT_FALSE(second.isNoCapMode()) << "kAutoSteps resolves to default 20";
    EXPECT_FALSE(second.isOutputDedup());
}

TEST(HBMAdaptConfig, EvictModeParamVariants)
{
    /* 1 = Decay；0 / 2 / kAutoEvict / 其他负值 = Lru（原始方案） */
    {
        HBMManagerLayered m;
        m.init(1024, 256, 128, 64, kNullLogger, true, 512, -1, /*evictMode=*/1);
        EXPECT_TRUE(m.isDecayMode());
    }
    for (int v : {0, 2, -2, HBMManagerLayered::kAutoEvict, 7}) {
        HBMManagerLayered m;
        m.init(1024, 256, 128, 64, kNullLogger, true, 512, -1, v);
        EXPECT_FALSE(m.isDecayMode()) << "value " << v;
    }
}

TEST(HBMAdaptConfig, WindowPlanParamAndLayerTable)
{
    /* pure table functions */
    EXPECT_EQ(gsa_layer_window::getWindowSize4096(-1), 5120);
    EXPECT_EQ(gsa_layer_window::getWindowSize4096(10), 3072);
    EXPECT_EQ(gsa_layer_window::getWindowSize4096(30), 4096);
    EXPECT_EQ(gsa_layer_window::getWindowSize4096(50), 5120);
    EXPECT_EQ(gsa_layer_window::getWindowSize4096(70), 3072);
    EXPECT_EQ(gsa_layer_window::getWindowSize5120(-1), 5120);
    EXPECT_EQ(gsa_layer_window::getWindowSize5120(10), 4096);
    EXPECT_EQ(gsa_layer_window::getWindowSize5120(40), 5120);
    EXPECT_EQ(gsa_layer_window::getWindowSize5120(70), 4096);

    /* windowPlan=1 (Budget4096): layer 30 -> table 4096, immu 8192 -> no clamp */
    {
        HBMManagerLayered m;
        m.init(1024, 16384, 8192, 64, kNullLogger, true, 32768, /*layerId=*/30,
               HBMManagerLayered::kAutoEvict, /*windowPlan=*/1);
        EXPECT_STREQ(m.getWindowPlanName(), "4096");
        EXPECT_EQ(m.getWindowSize(), 4096);
    }
    /* windowPlan=2 (Budget5120): layer 50 -> 5120 */
    {
        HBMManagerLayered m;
        m.init(1024, 16384, 8192, 64, kNullLogger, true, 32768, /*layerId=*/50,
               HBMManagerLayered::kAutoEvict, /*windowPlan=*/2);
        EXPECT_STREQ(m.getWindowPlanName(), "5120");
        EXPECT_EQ(m.getWindowSize(), 5120);
    }
    /* windowPlan=0 / 3 / kAutoPlan: Uniform regardless of layer */
    for (int plan : {0, 3, HBMManagerLayered::kAutoPlan}) {
        HBMManagerLayered m;
        m.init(1024, 256, 128, 64, kNullLogger, true, 512, /*layerId=*/30,
               HBMManagerLayered::kAutoEvict, plan);
        EXPECT_STREQ(m.getWindowPlanName(), "uniform") << "plan " << plan;
        EXPECT_EQ(m.getWindowSize(), 128) << "plan " << plan;
    }
}

/* Regression: init used `win == mHbmImmutableSize` (a no-op comparison), so
 * a table value above the immutable size was never clamped down. */
TEST(HBMAdaptConfig, WindowSizeClampedToImmutableAndTopk)
{
    {
        HBMManagerLayered m;
        /* layer -1 -> table 5120, immu 2048: must clamp DOWN to 2048. */
        m.init(1024, 4096, 2048, 64, kNullLogger, true, 8192, /*layerId=*/-1,
               HBMManagerLayered::kAutoEvict, /*windowPlan=*/1);
        EXPECT_EQ(m.getWindowSize(), 2048);
        /* setWindowSize applies the same clamp */
        m.setWindowSize(9999);
        EXPECT_EQ(m.getWindowSize(), 2048);
    }
    {
        HBMManagerLayered m;
        /* layer 0 -> table 3072 < topk 4000: must clamp UP to topk. */
        m.init(1024, 8192, 6144, 4000, kNullLogger, true, 16384, /*layerId=*/0,
               HBMManagerLayered::kAutoEvict, /*windowPlan=*/1);
        EXPECT_EQ(m.getWindowSize(), 4000);
        m.setWindowSize(100);
        EXPECT_EQ(m.getWindowSize(), 4000);
    }
}

/* ================= adapt-only: output dedup (outputDedup=1) ================= */

/* Default off: the two documented duplicate scenarios stay byte-identical
 * to the original (the switch must not change the default-param contract). */
TEST(HBMAdaptDedup, DisabledByDefaultKeepsOriginalSemantics)
{
    HBMManagerLayered m;
    m.init(4, 2, 2, 2, kNullLogger, true, 4);
    EXPECT_FALSE(m.isOutputDedup());
    std::vector<int> output, missGids, missSlots;
    m.processTopkWithDebug({0, 2, 3}, output, missGids, missSlots, kNullLogger);
    ASSERT_EQ(output.size(), 3u);
    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 1);
    EXPECT_EQ(output[2], 0) << "duplicate must remain when the switch is off";
}

/* Case B removed: with dedup on, the rotation-evicted hit position is
 * refreshed to its final residency (-1), no stale duplicate remains. */
TEST(HBMAdaptDedup, RotationDuplicateRemovedWhenEnabled)
{
    for (int mode = 0; mode < 2; ++mode) {
        HBMManagerLayered m;
        m.init(4, 2, 2, 2, kNullLogger, true, 4, -1, /*evictMode=*/mode,
               HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
               HBMManagerLayered::kAutoSteps, /*outputDedup=*/1);
        ASSERT_TRUE(m.isOutputDedup()) << "mode " << mode;
        std::vector<int> output, missGids, missSlots;
        /* gid 0 hits slot 0; miss 2 loads slot 1; miss 3 evicts slot 0 and
         * reloads it -> with dedup: scratch = [-1, 1, 0] (hit refreshed to
         * -1), output has no duplicate. */
        m.processTopkWithDebug({0, 2, 3}, output, missGids, missSlots, kNullLogger);
        ASSERT_GE(output.size(), 2u) << "mode " << mode;
        /* section 1 = [1, 0]: the loaded miss slots, unique */
        EXPECT_EQ(output[0], 1) << "mode " << mode;
        EXPECT_EQ(output[1], 0) << "mode " << mode;
        for (size_t i = 0; i < output.size(); ++i) {
            for (size_t j = i + 1; j < output.size(); ++j) {
                EXPECT_FALSE(output[i] == output[j] && output[i] != -1)
                    << "mode " << mode << ": duplicate " << output[i];
            }
        }
        EXPECT_TRUE(m.validateLruChain());
        EXPECT_TRUE(m.validateMapping());
    }
}

/* Case A removed: with dedup on, a repeated out-of-range pinned gid emits
 * its raw slot exactly once (both modes). */
TEST(HBMAdaptDedup, RawDuplicateRemovedWhenEnabled)
{
    for (int mode = 0; mode < 2; ++mode) {
        HBMManagerLayered m;
        m.init(8, 4, 4, 8, kNullLogger, false, 8, -1, /*evictMode=*/mode,
               HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
               HBMManagerLayered::kAutoSteps, /*outputDedup=*/1);
        std::vector<int> output, missGids, missSlots;
        /* raw slot 8 emitted once; gid 0 misses into tail slot 3 */
        m.processTopkWithDebug({12, 12, 0}, output, missGids, missSlots, kNullLogger);
        ASSERT_GE(output.size(), 3u) << "mode " << mode;
        EXPECT_EQ(output[0], 8) << "mode " << mode;
        EXPECT_NE(output[1], 8) << "mode " << mode << ": raw slot repeated";
        EXPECT_EQ(output[1], 3) << "mode " << mode;
        EXPECT_EQ(missGids.size(), 1u);
    }
}

/* With dedup on, rotation-dense fuzzing must never produce any duplicated
 * non-(-1) output value, in either mode. */
TEST(HBMAdaptDedup, FuzzNoDuplicatesWhenEnabled)
{
    for (int mode = 0; mode < 2; ++mode) {
        HBMManagerLayered m;
        m.init(4, 2, 2, 2, kNullLogger, true, 4, -1, /*evictMode=*/mode,
               HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
               HBMManagerLayered::kAutoSteps, /*outputDedup=*/1);
        std::mt19937 rng(4242);
        /* negatives, regular [0,4), valid pinned gids 4..5 (slots 2..3) and
         * raw pinned gid 6 (slot 4, repeatable) across 1..10 gids per call */
        std::uniform_int_distribution<int> gidDist(-2, 6);
        std::uniform_int_distribution<int> lenDist(1, 10);
        for (int iter = 0; iter < 500; ++iter) {
            std::vector<int> topk;
            const int n = lenDist(rng);
            for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
            std::vector<int> output, missGids, missSlots;
            m.processTopkWithDebug(topk, output, missGids, missSlots, kNullLogger);
            ASSERT_TRUE(m.validateLruChain()) << "mode " << mode << " iter " << iter;
            ASSERT_TRUE(m.validateMapping()) << "mode " << mode << " iter " << iter;
            std::vector<int> seen;
            for (int v : output) {
                if (v == -1) { continue; }
                const bool dup = std::find(seen.begin(), seen.end(), v) != seen.end();
                ASSERT_FALSE(dup) << "mode " << mode << " iter " << iter << ": slot " << v
                                 << " duplicated";
                seen.push_back(v);
            }
        }
    }
}

/* ================= adapt-only: decay behaviour ================= */

TEST(HBMAdaptDecay, ExplicitParamsResolveModes)
{
    {
        /* evictMode=0：显式 Lru */
        HBMManagerLayered m;
        m.init(1024, 256, 128, 64, kNullLogger, true, 512, -1, /*evictMode=*/0);
        EXPECT_FALSE(m.isDecayMode());
    }
    {
        /* evictMode=1：显式 Decay */
        HBMManagerLayered m;
        m.init(1024, 256, 128, 64, kNullLogger, true, 512, -1, /*evictMode=*/1);
        EXPECT_TRUE(m.isDecayMode());
    }
    {
        /* windowPlan=0：显式 Uniform */
        HBMManagerLayered m;
        m.init(1024, 256, 128, 64, kNullLogger, true, 512, -1, HBMManagerLayered::kAutoEvict,
               /*windowPlan=*/0);
        EXPECT_STREQ(m.getWindowPlanName(), "uniform");
    }
}

TEST(HBMAdaptDecay, OutputPaddedToImmutableSize)
{
    HBMManagerLayered m;
    m.init(1024, 64, 32, 8, kNullLogger, true, 512, -1, /*evictMode=*/1);
    ASSERT_TRUE(m.isDecayMode());

    std::vector<int> output, missGids, missSlots;
    std::vector<int> topk = {5};
    m.processTopkWithDebug(topk, output, missGids, missSlots, kNullLogger);

    /* decay output is always exactly mHbmImmutableSize, tail padded with -1 */
    ASSERT_EQ(output.size(), 32u);
    EXPECT_EQ(output[0], 5);
    for (size_t i = 1; i < output.size(); ++i) { EXPECT_EQ(output[i], -1) << "idx " << i; }
    EXPECT_TRUE(m.validateLruChain());
    EXPECT_TRUE(m.validateMapping());
}

TEST(HBMAdaptDecay, WindowStepsPruneStaleSlots)
{
    HBMManagerLayered m;
    m.init(1024, 64, 32, 8, kNullLogger, true, 512, -1, /*evictMode=*/1,
           HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
           /*windowSteps=*/2);
    ASSERT_TRUE(m.isDecayMode());

    std::vector<int> output, missGids, missSlots;
    /* step 1: touch slot 5 only */
    m.processTopkWithDebug({5}, output, missGids, missSlots, kNullLogger);
    EXPECT_EQ(m.getActiveSlotCount(), 1);
    /* step 2: touch slot 6 only; slot 5 is age 1 (< steps 2) -> alive */
    m.processTopkWithDebug({6}, output, missGids, missSlots, kNullLogger);
    EXPECT_EQ(m.getActiveSlotCount(), 2);
    /* step 3: touch slot 6 only; slot 5 is age 2 (>= steps 2) -> pruned */
    m.processTopkWithDebug({6}, output, missGids, missSlots, kNullLogger);
    EXPECT_EQ(m.getActiveSlotCount(), 1);
    /* slot 5 must not reappear in the history fill of step 3 */
    EXPECT_EQ(output.size(), 32u);
    EXPECT_EQ(output[0], 6);
    for (size_t i = 1; i < output.size(); ++i) {
        EXPECT_NE(output[i], 5) << "stale slot resurrected at idx " << i;
        EXPECT_NE(output[i], 6) << "hit slot duplicated at idx " << i;
    }
}

TEST(HBMAdaptDecay, NoCapModeViaSetters)
{
    HBMManagerLayered m;
    m.init(1024, 64, 32, 8, kNullLogger, true, 512);
    EXPECT_FALSE(m.isNoCapMode());
    m.setWindowSteps(0);
    EXPECT_TRUE(m.isNoCapMode());
    /* setWindowSteps clamps non-positive values to the 20 default */
    m.setWindowSteps(-5);
    EXPECT_FALSE(m.isNoCapMode());
    m.setEvictMode(HbmEvictMode::Decay);
    EXPECT_TRUE(m.isDecayMode());
    m.setEvictMode(HbmEvictMode::Lru);
    EXPECT_FALSE(m.isDecayMode());
}

/* decay-mode state stays internally consistent under randomized flows */
TEST(HBMAdaptDecay, RandomFuzzKeepsInvariants)
{
    HBMManagerLayered m;
    m.init(512, 64, 32, 16, kNullLogger, true, 128, -1, /*evictMode=*/1);
    ASSERT_TRUE(m.isDecayMode());
    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> gidDist(-3, 64 + (128 - 64) + 3);
    std::uniform_int_distribution<int> lenDist(0, 40);
    for (int iter = 0; iter < 150; ++iter) {
        std::vector<int> topk;
        const int n = lenDist(rng);
        topk.reserve(n);
        for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
        std::vector<int> output, missGids, missSlots;
        m.processTopkWithDebug(topk, output, missGids, missSlots, kNullLogger);
        ASSERT_TRUE(m.validateLruChain()) << "iter " << iter;
        ASSERT_TRUE(m.validateMapping()) << "iter " << iter;
        /* real topk is never cut: output is padded UP to immutable (32) and
         * may legitimately exceed it when hits+loads outnumber the immutable
         * budget; over-length outputs are rejected by the consumer. */
        ASSERT_GE(output.size(), 32u) << "iter " << iter;
        /* miss vectors must stay parallel */
        ASSERT_EQ(missGids.size(), missSlots.size()) << "iter " << iter;
    }
}

/* Output composition: [real topk][m20 decay history][-1 pad to immutable].
 * Layer -1 under plan 4096 -> window 5120 < immutable 8192: the history
 * budget is the layer window, the -1 padding always closes at the
 * immutable size the consumer (mSpaTopkLen) requires. */
TEST(HBMAdaptDecay, OutputCompositionTopkHistoryThenPad)
{
    HBMManagerLayered m;
    m.init(6000, 8192, 8192, 64, kNullLogger, true, 16384, /*layerId=*/-1,
           /*evictMode=*/1, /*windowPlan=*/1);
    ASSERT_TRUE(m.isDecayMode());
    EXPECT_EQ(m.getWindowSize(), 5120);

    std::vector<int> output, missGids, missSlots;
    /* step 1: hits on preloaded identities 0..2; no history yet */
    m.processTopkWithDebug({0, 1, 2}, output, missGids, missSlots, kNullLogger);
    ASSERT_EQ(output.size(), 8192u);
    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 1);
    EXPECT_EQ(output[2], 2);
    for (size_t i = 3; i < output.size(); ++i) { ASSERT_EQ(output[i], -1) << "idx " << i; }

    /* step 2: hit 3; history = {0,1,2} (age 1 < 20 steps, tie on score ->
     * slot ascending), then -1 to the immutable size */
    m.processTopkWithDebug({3}, output, missGids, missSlots, kNullLogger);
    ASSERT_EQ(output.size(), 8192u);
    EXPECT_EQ(output[0], 3);
    EXPECT_EQ(output[1], 0);
    EXPECT_EQ(output[2], 1);
    EXPECT_EQ(output[3], 2);
    for (size_t i = 4; i < output.size(); ++i) { ASSERT_EQ(output[i], -1) << "idx " << i; }
}

/* Regression: the real-topk section of buildTopkOutputDecay used to write
 * mAddedSlotFlag[slot] for every emitted slot.  Out-of-range pinned gids put
 * their RAW slot (>= mMaxTotalSlots) into topkSlotIds, so that write was an
 * out-of-bounds heap access.  Flags for valid slots are already set by
 * classifyTopkGid / handlePinnedGid / processTopkMisses; the output builder
 * must not touch the flag array in the topk section at all. */
TEST(HBMAdaptDecay, OutOfRangePinnedGidKeepsOutputBuilderSafe)
{
    /* dram 8, cap 4, maxTotal 8: pinned gid domain is [8, 12); gid 12 maps
     * to raw slot 12 - 8 + 4 = 8 >= maxTotalSlots -> verbatim passthrough. */
    HBMManagerLayered m;
    m.init(8, 4, 2, 2, kNullLogger, false, 8, -1, /*evictMode=*/1);
    ASSERT_TRUE(m.isDecayMode());

    /* randomized flows mixing every gid class, incl. the raw-slot gid 12..13 */
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> gidDist(-2, 14);
    for (int iter = 0; iter < 300; ++iter) {
        std::vector<int> topk;
        for (int i = 0; i < 8; ++i) { topk.push_back(gidDist(rng)); }
        std::vector<int> output, missGids, missSlots;
        m.processTopkWithDebug(topk, output, missGids, missSlots, kNullLogger);
        ASSERT_TRUE(m.validateLruChain()) << "iter " << iter;
        ASSERT_TRUE(m.validateMapping()) << "iter " << iter;
        /* real topk is never cut: >= immutable, longer when hits exceed it */
        ASSERT_GE(output.size(), 2u) << "iter " << iter;
    }

    /* deterministic single call: the raw slot value appears verbatim */
    std::vector<int> output, missGids, missSlots;
    m.processTopkWithDebug({12, 0}, output, missGids, missSlots, kNullLogger);
    ASSERT_GE(output.size(), 2u);
    EXPECT_EQ(output[0], 8);
}

/* Regression: the trailing output.resize(mHbmImmutableSize) used to cut the
 * real-topk section down when topkSize > immutable (init only asserts
 * topkSize <= hbmCapacity).  The original buildTopkOutput never truncates;
 * over-length outputs must be left for the consumer to reject. */
TEST(HBMAdaptDecay, RealTopkNeverTruncated)
{
    HBMManagerLayered m;
    m.init(64, 32, 4, 8, kNullLogger, true, 64, -1, /*evictMode=*/1);
    std::vector<int> output, missGids, missSlots;
    /* preloaded identities 0..7 all hit: 8 real topk slots, immutable is 4 */
    m.processTopkWithDebug({0, 1, 2, 3, 4, 5, 6, 7}, output, missGids, missSlots, kNullLogger);
    ASSERT_EQ(output.size(), 8u);
    for (int i = 0; i < 8; ++i) { EXPECT_EQ(output[i], i) << "idx " << i; }
    EXPECT_TRUE(m.validateLruChain());
    EXPECT_TRUE(m.validateMapping());
}

/* Output duplicate audit.  Besides -1, the output CAN contain duplicated
 * slot ids, in exactly two scenarios, both mirrored from HBMManager:
 *
 *  A) repeated out-of-range pinned gid: its raw slot (transformedSlot >=
 *     mMaxTotalSlots) passes through verbatim and lies outside the
 *     mSeenGidFlag dedup domain [0, dram + maxTotal - cap), so each
 *     occurrence emits the same raw value again;
 *
 *  B) in-call eviction rotation: a just-hit slot can be picked as a miss
 *     victim later in the SAME call (tiny capacity forces rotation).  The
 *     backfill only refreshes MISS positions; the hit position keeps the
 *     now-stale slot value while the new gid's backfill writes the same
 *     slot -> the slot appears twice.  The original has the identical
 *     structure and the identical duplicate.
 *
 * Everything else is duplicate-free by construction: valid regular/pinned
 * gids are generation-deduped, gid classes occupy disjoint slot ranges,
 * mGlobalToSlot is injective, and the history/pad sections are flag-gated.
 * The tests below pin both scenarios deterministically and verify the
 * adapt output (duplicates included) matches the original byte-for-byte. */
TEST(HBMAdaptEquivalence, OutputDuplicateSemanticsMatchOriginal)
{

    /* ---- Case B: deterministic in-call eviction rotation ---- */
    {
        /* dram 4, cap 2, preload: identities 0..1.  topk {0, 2, 3}: gid 0
         * hits slot 0; miss 2 loads slot 1; miss 3 evicts slot 0 (LRU tail)
         * and reloads it -> scratch = [0, 1, 0]. */
        HBMManager ref;
        HBMManagerLayered adapt;
        InitRef(ref, {4, 2, 2, 2, 4, true});
        InitAdapt(adapt, {4, 2, 2, 2, 4, true});

        std::vector<int> oR, oA, mR, mA, sR, sA;
        ref.processTopkWithDebug({0, 2, 3}, oR, mR, sR, kNullLogger);
        adapt.processTopkWithDebug({0, 2, 3}, oA, mA, sA, kNullLogger);
        EXPECT_EQ(oA, oR);
        ASSERT_EQ(oA.size(), 3u);
        EXPECT_EQ(oA[0], 0);
        EXPECT_EQ(oA[1], 1);
        EXPECT_EQ(oA[2], 0) << "in-call eviction rotation must duplicate slot 0";
    }

    /* ---- Case A: deterministic repeated raw pinned slot, both modes ---- */
    {
        /* dram 8, cap 4, maxTotal 8: dedup domain [0, 12); gid 12 maps to
         * raw slot 8 (verbatim, repeatable); gid 0 misses into tail slot 3. */
        for (int mode = 0; mode < 2; ++mode) {
            HBMManagerLayered m;
            m.init(8, 4, 4, 8, kNullLogger, false, 8, -1, /*evictMode=*/mode);
            std::vector<int> output, missGids, missSlots;
            m.processTopkWithDebug({12, 12, 0}, output, missGids, missSlots, kNullLogger);
            ASSERT_GE(output.size(), 3u) << "mode " << mode;
            EXPECT_EQ(output[0], 8) << "mode " << mode;
            EXPECT_EQ(output[1], 8) << "mode " << mode;
            EXPECT_EQ(output[2], 3) << "mode " << mode;
        }
    }
}

/* Rotation-dense differential fuzz: duplicates stay byte-identical between
 * the original and adapt.  cap 2 with topk up to 10 gids forces constant
 * rotation, mixing negatives, regular, valid pinned (gids [4,6) -> slots
 * [2,4)) and raw pinned (gid 6 -> slot 4). */
TEST(HBMAdaptEquivalence, OutputDuplicateFuzzMatchesOriginal)
{
    HBMManager ref;
    HBMManagerLayered adapt;
    InitRef(ref, {4, 2, 2, 2, 4, true});
    InitAdapt(adapt, {4, 2, 2, 2, 4, true});
    std::mt19937 rng(777);
    std::uniform_int_distribution<int> gidDist(-2, 6);
    std::uniform_int_distribution<int> lenDist(1, 10);
    for (int iter = 0; iter < 500; ++iter) {
        std::vector<int> topk;
        const int n = lenDist(rng);
        for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
        std::vector<int> oR, oA, mR, mA, sR, sA;
        ref.processTopkWithDebug(topk, oR, mR, sR, kNullLogger);
        adapt.processTopkWithDebug(topk, oA, mA, sA, kNullLogger);
        ASSERT_EQ(oA, oR) << "iter " << iter;
        ASSERT_EQ(mA, mR) << "iter " << iter;
        ASSERT_EQ(sA, sR) << "iter " << iter;
    }
}

/* ================= golden reference model ================= */

namespace {

/* Independent, deliberately naive re-implementation of the adapt spec,
 * written from the behaviour contract rather than from the header's code:
 *   - recency order in a std::list (front = most recent);
 *   - decay state in std::map (score / last-seen step);
 *   - eviction victims found by a full backward scan of the chain;
 *   - LRU fill by a checked walk over the whole chain.
 * None of the header's fast paths exist here (fused LruNode, mEvictTail
 * cache, call-start boundary skip, generation flags), so adapt == golden
 * agreement validates those optimizations themselves, not just the output
 * contract.  Float paths mirror the header exactly on purpose (iterative
 * pow table, touch recurrence, 1e-3f liveness threshold) so the comparison
 * is bit-exact.
 *
 * dedupOn=false reproduces the original's observable quirks (repeated raw
 * slots; stale hit slots after in-call rotation).  dedupOn=true implements
 * the documented fix: per-call raw-slot dedup + post-miss residency
 * refresh. */
class GoldenHbm {
public:
    int dram = 0;
    int cap = 0;
    int immu = 0;
    int maxTotal = 0;
    bool decay = false;
    int steps = 20;
    float lam = 0.85f;
    int window = 0;
    bool dedupOn = false;

    std::vector<int> globalToSlot;
    std::vector<int> slotToGlobal;
    std::vector<char> pinned;
    std::list<int> chain;  // front = most recently used
    std::map<int, float> score;
    std::map<int, int> last;
    float hitRate = 0.0f;
    int pinnedCount = 0;

    void init(int dramT, int capT, int immuT, int maxTotalSlots, bool preload, bool decayMode,
              int windowSteps, float lambda, int windowSize, bool dedupOnFlag)
    {
        dram = dramT;
        cap = capT;
        immu = immuT;
        maxTotal = (maxTotalSlots > cap) ? maxTotalSlots : cap;
        decay = decayMode;
        steps = windowSteps;
        lam = lambda;
        window = windowSize;
        dedupOn = dedupOnFlag;

        globalToSlot.assign(dram, -1);
        slotToGlobal.assign(cap, -1);
        pinned.assign(maxTotal, 0);
        chain.clear();
        for (int i = 0; i < cap; ++i) { chain.push_back(i); }
        if (preload) { preloadSlotMapping(); }

        const int n = (steps > 0) ? std::max(steps, 64) : 512;
        mPowT.assign(n + 1, 1.0f);
        for (int k = 1; k <= n; ++k) { mPowT[k] = mPowT[k - 1] * lam; }

        score.clear();
        last.clear();
        mStep = 0;
        pinnedCount = 0;
        hitRate = 0.0f;
    }

    float powLambda(int k) const
    {
        if (k >= 0 && k < (int)mPowT.size()) { return mPowT[k]; }
        return std::pow(lam, (float)k);
    }

    bool alive(int slot) const
    {
        auto it = last.find(slot);
        if (it == last.end() || it->second < 0) { return false; }
        const int age = mStep - it->second;
        if (age < 0) { return false; }
        if (steps > 0) { return age < steps; }
        return score.at(slot) * powLambda(age) >= 1e-3f;
    }

    void touch(int slot)
    {
        auto it = last.find(slot);
        if (it != last.end() && it->second >= 0) {
            const int age = mStep - it->second;
            if (age > 0) { score[slot] *= powLambda(age); }
        }
        score[slot] += 1.0f;
        last[slot] = mStep;
    }

    void prune()
    {
        for (auto it = last.begin(); it != last.end();) {
            const int slot = it->first;
            if (it->second < 0 || !alive(slot)) {
                it = last.erase(it);
                score.erase(slot);
            } else {
                ++it;
            }
        }
    }

    void moveToFront(int slot)
    {
        for (auto it = chain.begin(); it != chain.end(); ++it) {
            if (*it == slot) {
                chain.splice(chain.begin(), chain, it);
                return;
            }
        }
        assert(false && "moveToFront on a slot that is not in the chain");
    }

    int slotForGlobal(int gid) const
    {
        if (gid >= dram) { return gid - dram + cap; }
        if (gid >= 0 && gid < dram) { return globalToSlot[gid]; }
        return -1;
    }

    int gidForSlot(int slot) const
    {
        if (slot >= 0 && slot < cap) { return slotToGlobal[slot]; }
        if (slot >= cap && slot < maxTotal && pinned[slot] != 0) { return slot - cap + dram; }
        return -1;
    }

    void process(const std::vector<int>& topkIds, std::vector<int>& output,
                 std::vector<int>& missGids, std::vector<int>& missSlots)
    {
        if (decay) { ++mStep; }
        output.clear();
        missGids.clear();
        missSlots.clear();
        CallCtx c;
        classifyPhase(topkIds, c, missGids);
        evictAndBackfillPhase(c, missGids, missSlots);
        if (dedupOn) { refreshPhase(c); }
        if (decay) {
            for (int slot : missSlots) { touch(slot); }
            prune();
        }
        outputPhase(c, output);
    }

private:
    /* 与头文件一致的字面常量（金标准必须镜像这些行为契约值）。 */
    static constexpr int kMissLimit = 2048;       /* 单调用 miss 上限，到达即 break */
    static constexpr int kHitRatePos = kMissLimit - 1;  /* 命中率采样去重序号 */
    static constexpr int kPreloadIdentityLimit = 256;   /* 恒等预载上界（含 256） */

    /* 单调用 scratch：各去重位置的 gid / 槽值与 miss 回填序号。 */
    struct CallCtx {
        std::set<int> seen;
        std::set<int> rawSeen;
        std::vector<int> posGid;
        std::vector<int> posSlot;
        std::vector<int> missPos;
        int deduped = 0;
        int misses = 0;
    };

    void classifyPhase(const std::vector<int>& topkIds, CallCtx& c, std::vector<int>& missGids)
    {
        const int gidBound = dram + (maxTotal - cap);
        for (int gid : topkIds) {
            if (gid >= 0 && gid < gidBound) {
                if (c.seen.count(gid) != 0) { continue; }
                c.seen.insert(gid);
            }
            if (gid >= dram) {
                const int slot = gid - dram + cap;
                if (slot >= maxTotal) {
                    if (dedupOn && c.rawSeen.count(slot) != 0) { continue; }
                    c.rawSeen.insert(slot);
                    c.posGid.push_back(gid);
                    c.posSlot.push_back(slot);
                    ++c.deduped;
                    continue;
                }
                if (pinned[slot] == 0) {
                    pinned[slot] = 1;
                    chain.push_front(slot);
                    ++pinnedCount;
                } else {
                    moveToFront(slot);
                }
                c.posGid.push_back(gid);
                c.posSlot.push_back(slot);
                if (decay) { touch(slot); }
                ++c.deduped;
                continue;
            }
            if (gid < 0) {
                c.posGid.push_back(gid);
                c.posSlot.push_back(-1);
                ++c.deduped;
                continue;
            }
            const int slot = globalToSlot[gid];
            if (slot != -1) {
                moveToFront(slot);
                c.posGid.push_back(gid);
                c.posSlot.push_back(slot);
                if (decay) { touch(slot); }
            } else {
                missGids.push_back(gid);
                c.missPos.push_back((int)c.posGid.size());
                c.posGid.push_back(gid);
                c.posSlot.push_back(-1);
                ++c.misses;
                if (c.misses == kMissLimit) { break; }
            }
            if (c.deduped == kHitRatePos) {
                hitRate = 1.0f - c.misses * 1.0f / (float)kMissLimit;
            }
            ++c.deduped;
        }
    }

    void evictAndBackfillPhase(CallCtx& c, std::vector<int>& missGids, std::vector<int>& missSlots)
    {
        int loaded = 0;
        for (size_t i = 0; i < missGids.size(); ++i) {
            int victim = -1;
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                if (pinned[*it] == 0) { victim = *it; break; }
            }
            if (victim == -1) { continue; }
            const int oldGid = slotToGlobal[victim];
            if (oldGid >= 0 && oldGid < dram) { globalToSlot[oldGid] = -1; }
            if (decay) {
                score[victim] = 0.0f;
                last[victim] = -1;
            }
            slotToGlobal[victim] = missGids[i];
            globalToSlot[missGids[i]] = victim;
            missSlots.push_back(victim);
            ++loaded;
            moveToFront(victim);
        }
        missGids.resize(loaded);
        for (int j = 0; j < loaded; ++j) { c.posSlot[c.missPos[j]] = globalToSlot[missGids[j]]; }
    }

    /* The gid that trips the miss limit does not increment deduped: its slot
     * is loaded into HBM (observable in missSlots / the mapping state), but
     * its scratch position lies outside [0, deduped) and is NOT emitted - the
     * header breaks out of classify before counting it. */
    void refreshPhase(CallCtx& c)
    {
        for (int i = 0; i < c.deduped; ++i) {
            const int gid = c.posGid[i];
            if (gid < 0 || gid >= dram) { continue; }
            const int live = globalToSlot[gid];
            if (c.posSlot[i] != live) { c.posSlot[i] = live; }
        }
    }

    void outputPhase(CallCtx& c, std::vector<int>& output)
    {
        std::set<int> flagged;
        int realTopk = 0;
        for (int i = 0; i < c.deduped; ++i) {
            const int s = c.posSlot[i];
            if (s < 0) { continue; }
            output.push_back(s);
            ++realTopk;
            if (s < maxTotal) { flagged.insert(s); }
        }
        if (!decay) {
            emitLruFill(c, output, flagged, immu - realTopk);
        } else {
            emitDecayHistory(c, output, flagged, window - realTopk);
            while ((int)output.size() < immu) { output.push_back(-1); }
        }
    }

    void emitLruFill(CallCtx& c, std::vector<int>& output, std::set<int>& flagged, int needed)
    {
        int cnt = 0;
        for (int slot : chain) {
            if (cnt >= needed) { break; }
            if (flagged.count(slot) != 0) { continue; }
            output.push_back(slot);
            flagged.insert(slot);
            ++cnt;
        }
    }

    void emitDecayHistory(CallCtx& c, std::vector<int>& output, std::set<int>& flagged,
                          int histNeeded)
    {
        if (histNeeded <= 0) { return; }
        std::vector<std::pair<float, int>> ranked;
        for (const auto& kv : last) {
            const int slot = kv.first;
            if (flagged.count(slot) != 0) { continue; }
            if (kv.second < 0) { continue; }
            if (!alive(slot)) { continue; }
            const int age = mStep - kv.second;
            const float eff = score.at(slot) * powLambda(age);
            if (eff <= 0.0f) { continue; }
            ranked.emplace_back(eff, slot);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      if (a.first != b.first) { return a.first > b.first; }
                      return a.second < b.second;
                  });
        if ((int)ranked.size() > histNeeded) { ranked.resize(histNeeded); }
        for (const auto& kv : ranked) {
            if ((int)output.size() >= window) { break; }
            if (flagged.count(kv.second) != 0) { continue; }
            output.push_back(kv.second);
            flagged.insert(kv.second);
        }
    }

    void preloadSlotMapping()
    {
        int slotIdx = 0;
        for (; slotIdx <= kPreloadIdentityLimit && slotIdx < cap && slotIdx < dram; ++slotIdx) {
            globalToSlot[slotIdx] = slotIdx;
            slotToGlobal[slotIdx] = slotIdx;
        }
        const int identityCount = slotIdx;
        for (int gid = dram - 1; slotIdx < cap && gid >= identityCount; --gid, ++slotIdx) {
            globalToSlot[gid] = slotIdx;
            slotToGlobal[slotIdx] = gid;
        }
    }

    std::vector<float> mPowT;
    int mStep = 0;
};

/* Full externally-observable state comparison against the golden model. */
void CompareGoldenState(const HBMManagerLayered& adapt, const GoldenHbm& golden)
{
    ASSERT_TRUE(adapt.validateLruChain());
    ASSERT_TRUE(adapt.validateMapping());
    ASSERT_EQ(adapt.mGlobalToSlot, golden.globalToSlot);
    ASSERT_EQ(adapt.mSlotToGlobal, golden.slotToGlobal);
    ASSERT_EQ(adapt.getPinnedSlotCount(), golden.pinnedCount);
    const int dram = adapt.getDramTensorCount();
    const int cap = adapt.getHbmCapacity();
    const int maxTotal = adapt.getMaxTotalSlots();
    for (int gid = -2; gid <= dram + (maxTotal - cap) + 3; ++gid) {
        ASSERT_EQ(adapt.getSlotForGlobal(gid), golden.slotForGlobal(gid)) << "gid " << gid;
    }
    for (int slot = -1; slot <= maxTotal + 2; ++slot) {
        ASSERT_EQ(adapt.getGidForSlot(slot), golden.gidForSlot(slot)) << "slot " << slot;
    }
}

/* Drive adapt and the golden model with the same topk; compare every
 * externally observable result of the call plus the full state after it.
 * outAdapt / missSlotsAdapt (optional) receive adapt's results for
 * exact-value assertions. */
void RunGolden(HBMManagerLayered& adapt, GoldenHbm& golden, const std::vector<int>& topk,
              std::vector<int>* outAdapt = nullptr, std::vector<int>* missSlotsAdapt = nullptr)
{
    std::vector<int> outA, outG, missGidA, missGidG, missSlotA, missSlotG;
    adapt.processTopkWithDebug(topk, outA, missGidA, missSlotA, kNullLogger);
    golden.process(topk, outG, missGidG, missSlotG);
    ASSERT_EQ(outA, outG);
    ASSERT_EQ(missGidA, missGidG);
    ASSERT_EQ(missSlotA, missSlotG);
    if (adapt.isDecayMode()) { ASSERT_EQ(adapt.getActiveSlotCount(), (int)golden.last.size()); }
    if (outAdapt != nullptr) { *outAdapt = outA; }
    if (missSlotsAdapt != nullptr) { *missSlotsAdapt = missSlotA; }
    CompareGoldenState(adapt, golden);
}

void AssertNoDuplicateSlots(const std::vector<int>& v)
{
    std::set<int> seen;
    for (int s : v) {
        if (s == -1) { continue; }
        ASSERT_TRUE(seen.insert(s).second) << "duplicate slot " << s;
    }
}

/* decay outputs: the -1 padding must be one contiguous tail section */
void AssertPadIsContiguousTail(const std::vector<int>& v)
{
    bool pad = false;
    for (int s : v) {
        if (s == -1) {
            pad = true;
        } else {
            ASSERT_FALSE(pad) << "slot " << s << " emitted after -1 padding started";
        }
    }
}

}  // namespace

/* ============ golden differential tests (LRU) ============ */

/* Lru mode (default params), scripted large shape: adapt, original and golden
 * must agree everywhere.  The golden model has no fast paths, so agreement
 * also validates mEvictTail and the call-start boundary skip under mixes of
 * hits / misses / valid pins / raw pins / negatives / repeats. */
TEST(HBMAdaptGolden, LruScriptedMixMatchesGolden)
{
    HBMManagerLayered adapt;
    GoldenHbm golden;
    const Shape s = {4096, 1024, 512, 64, 4096, true};
    InitAdapt(adapt, s);
    golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, /*decay=*/false,
                /*steps=*/20, /*lambda=*/0.85f, /*window=*/s.immu, /*dedup=*/false);
    const std::vector<std::vector<int>> calls = {
        {0, 1, 2},                /* preloaded identity hits */
        {5000, 5001, 5002},       /* valid pinned gids -> slots 1928..1930 */
        {60000, 60000, 3},        /* raw out-of-range pin, repeat, hit */
        {-1, 4095, 4096},         /* negative, reverse-preloaded hit, first pinned gid */
        {0, 5000, 60000, -5, 300} /* mixed; 300 is not preloaded -> miss */
    };
    for (const auto& topk : calls) { RunGolden(adapt, golden, topk); }
}

/* Lru mode (default params), fuzz small shapes: triple agreement of original /
 * adapt / golden on output, missGlobalIds and missSlots per call. */
TEST(HBMAdaptGolden, LruFuzzMatchesGoldenAndOriginal)
{
    const Shape shapes[] = {{64, 32, 16, 8, 48, true},
                            {256, 64, 32, 16, 128, false},
                            {4, 2, 2, 2, 4, true}};
    for (const auto& s : shapes) {
        HBMManager ref;
        HBMManagerLayered adapt;
        GoldenHbm golden;
        InitRef(ref, s);
        InitAdapt(adapt, s);
        golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, /*decay=*/false, /*steps=*/20,
                    /*lambda=*/0.85f, /*window=*/s.immu, /*dedup=*/false);
        std::mt19937 rng(2024 + s.dram);
        std::uniform_int_distribution<int> gidDist(-3, s.dram + (s.maxTotal - s.cap) + 3);
        std::uniform_int_distribution<int> lenDist(0, 40);
        for (int iter = 0; iter < 200; ++iter) {
            std::vector<int> topk;
            const int n = lenDist(rng);
            for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
            std::vector<int> oR, oA, oG, mR, mA, mG, sR, sA, sG;
            ref.processTopkWithDebug(topk, oR, mR, sR, kNullLogger);
            adapt.processTopkWithDebug(topk, oA, mA, sA, kNullLogger);
            golden.process(topk, oG, mG, sG);
            ASSERT_EQ(oA, oG) << "shape dram " << s.dram << " iter " << iter;
            ASSERT_EQ(oR, oG) << "original must match golden, shape dram " << s.dram << " iter "
                              << iter;
            ASSERT_EQ(mA, mG) << "iter " << iter;
            ASSERT_EQ(sA, sG) << "iter " << iter;
        }
        CompareGoldenState(adapt, golden);
    }
}

/* cap==1 回归：唯一常规槽即链头（prev=-1），victim 换入重载后 mEvictTail
 * 不得丢失（早期 prevEvictable(-1) 会把缓存置成 -1，此后所有 miss 被永久
 * 丢弃，而原版逐次重扫可反复驱逐同一槽）。同调用连续 miss 必须全部装载，
 * 三方（原版 / adapt / golden）逐调用一致。 */
TEST(HBMAdaptGolden, LruSingleCapacitySlotRepeatedEvictions)
{
    const Shape s = {4, 1, 1, 1, 4, true};
    HBMManager ref;
    HBMManagerLayered adapt;
    GoldenHbm golden;
    InitRef(ref, s);
    InitAdapt(adapt, s);
    golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, /*decay=*/false, /*steps=*/20,
                /*lambda=*/0.85f, /*window=*/s.immu, /*dedup=*/false);
    const std::vector<std::vector<int>> calls = {
        {1, 2, 3},           /* 3 consecutive misses: the only slot evicted+reloaded 3x */
        {4, 5, 2},           /* new pins at slots 1/2 + a miss */
        {0, 3, 5, 6, -1},    /* miss + pin re-touch + new pin + negative */
        {7, 7, 2, 0, 3},     /* raw out-of-range pin repeated + regular mix */
        {2, 1, 0, 3},
    };
    for (const auto& topk : calls) {
        std::vector<int> oR, oA, oG, mR, mA, mG, sR, sA, sG;
        ref.processTopkWithDebug(topk, oR, mR, sR, kNullLogger);
        adapt.processTopkWithDebug(topk, oA, mA, sA, kNullLogger);
        golden.process(topk, oG, mG, sG);
        ASSERT_EQ(oA, oG);
        ASSERT_EQ(oR, oG) << "original must match golden";
        ASSERT_EQ(mA, mG);
        ASSERT_EQ(sA, sG);
    }
    CompareGoldenState(adapt, golden);
}

/* Miss-threshold break + hit-rate sampling point, validated against the
 * golden model.  With preload: position 2047 is a miss -> sampled
 * mHitRate = 1 - 1791/2048 while the miss stack keeps growing to 2048.
 * Without preload: the gid at position 2047 IS the 2048th miss, so the
 * break fires before sampling and mHitRate stays 0. */
TEST(HBMAdaptGolden, LruMissThresholdAndHitRateMatchGolden)
{
    std::vector<int> topk;
    for (int gid = 0; gid < 3000; ++gid) { topk.push_back(gid); }

    {
        HBMManagerLayered adapt;
        GoldenHbm golden;
        const Shape s = {16384, 1024, 512, 64, 4096, true};
        InitAdapt(adapt, s);
        golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, false, 20, 0.85f, s.immu,
                    false);
        std::vector<int> oA, oG, mA, mG, sA, sG;
        adapt.processTopkWithDebug(topk, oA, mA, sA, kNullLogger);
        golden.process(topk, oG, mG, sG);
        ASSERT_EQ(mA.size(), 2048u) << "break at the 2048-miss threshold";
        ASSERT_EQ(mA, mG);
        ASSERT_EQ(sA, sG);
        ASSERT_EQ(oA, oG);
        EXPECT_FLOAT_EQ(adapt.mHitRate, golden.hitRate);
        EXPECT_FLOAT_EQ(adapt.mHitRate, 1.0f - 1791.0f / 2048.0f);
        CompareGoldenState(adapt, golden);
    }
    {
        HBMManagerLayered adapt;
        GoldenHbm golden;
        const Shape s = {16384, 1024, 512, 64, 4096, false};
        InitAdapt(adapt, s);
        golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, false, 20, 0.85f, s.immu,
                    false);
        std::vector<int> oA, oG, mA, mG, sA, sG;
        adapt.processTopkWithDebug(topk, oA, mA, sA, kNullLogger);
        golden.process(topk, oG, mG, sG);
        ASSERT_EQ(mA.size(), 2048u);
        ASSERT_EQ(oA, oG);
        ASSERT_EQ(mA, mG);
        ASSERT_EQ(sA, sG);
        EXPECT_FLOAT_EQ(adapt.mHitRate, 0.0f) << "break precedes the sampling point";
        EXPECT_FLOAT_EQ(adapt.mHitRate, golden.hitRate);
        CompareGoldenState(adapt, golden);
    }
}

/* ============ golden differential tests (decay) ============ */

/* Decay mode default config (steps 20, lambda 0.85, uniform window),
 * scripted ranking / tie-break / window / pad composition.  Output layout
 * verified exactly: [real topk][m-window history ranked by eff desc,
 * slot asc][-1 pad to immutable]. */
TEST(HBMAdaptGolden, DecayScriptedRankingMatchesGolden)
{
    HBMManagerLayered adapt;
    GoldenHbm golden;
    const Shape s = {256, 64, 32, 8, 128, true};
    adapt.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal, -1,
               /*evictMode=*/1);
    golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, /*decay=*/true, /*steps=*/20,
                /*lambda=*/0.85f, /*window=*/adapt.getWindowSize(), /*dedup=*/false);
    ASSERT_EQ(adapt.getWindowSize(), 32);
    /* Expected real-topk + history prefix per call:
     *  c1 {0,1,2}: identity hits, history empty -> -1 pad
     *  c2 {3}:     history = age-1 ties {0,1,2}, slot asc
     *  c3 {3}:     repeat touch keeps the ranking
     *  c4 {0}:     slot 3 (score 0.85+1) outranks age-3 slots (eff 0.85^3) */
    const std::vector<std::vector<int>> calls = {{0, 1, 2}, {3}, {3}, {0}};
    const std::vector<std::vector<int>> prefixes = {
        {0, 1, 2}, {3, 0, 1, 2}, {3, 0, 1, 2}, {0, 3, 1, 2}};
    std::vector<int> out;
    for (size_t c = 0; c < calls.size(); ++c) {
        RunGolden(adapt, golden, calls[c], &out);
        ASSERT_EQ(out.size(), 32u) << "call " << c;
        EXPECT_EQ(std::vector<int>(out.begin(), out.begin() + prefixes[c].size()), prefixes[c])
            << "call " << c;
        for (int i = (int)prefixes[c].size(); i < 32; ++i) { EXPECT_EQ(out[i], -1) << c; }
    }
    EXPECT_EQ(adapt.getActiveSlotCount(), 4);
    CompareGoldenState(adapt, golden);
}

/* Decay mode default config, fuzz: adapt == golden on every call. */
TEST(HBMAdaptGolden, DecayFuzzMatchesGolden)
{
    const Shape shapes[] = {{64, 32, 16, 8, 48, true}, {256, 64, 32, 16, 128, false}};
    for (const auto& f : shapes) {
        HBMManagerLayered a;
        GoldenHbm g;
        a.init(f.dram, f.cap, f.immu, f.topk, kNullLogger, f.preload, f.maxTotal, -1,
               /*evictMode=*/1);
        g.init(f.dram, f.cap, f.immu, f.maxTotal, f.preload, true, 20, 0.85f,
               a.getWindowSize(), false);
        std::mt19937 rng(6789 + f.dram);
        std::uniform_int_distribution<int> gidDist(-2, f.dram + (f.maxTotal - f.cap) + 3);
        std::uniform_int_distribution<int> lenDist(0, 40);
        for (int iter = 0; iter < 300; ++iter) {
            std::vector<int> topk;
            const int n = lenDist(rng);
            for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
            RunGolden(a, g, topk);
        }
    }
}

/* windowSteps=2: strict age window (age < steps).  Misses 40/41
 * load into slots 31/30; two idle hits age them to 2 (== steps) so they
 * are pruned and the third call's history is empty. */
TEST(HBMAdaptGolden, DecayWindowStepsTwoMatchesGolden)
{
    HBMManagerLayered adapt;
    GoldenHbm golden;
    const Shape s = {64, 32, 16, 8, 48, true};
    InitAdapt(adapt, s, /*layerId=*/-1, /*evictMode=*/1, HBMManagerLayered::kAutoPlan,
              HBMManagerLayered::kAutoLambda, /*windowSteps=*/2);
    ASSERT_TRUE(adapt.isDecayMode());
    golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, true, /*steps=*/2, 0.85f,
                adapt.getWindowSize(), false);
    /* miss 40 -> slot 31, miss 41 -> slot 30 (tail-most non-pinned) */
    std::vector<int> out, missSlots;
    RunGolden(adapt, golden, {40, 41}, &out, &missSlots);
    ASSERT_EQ(missSlots, (std::vector<int>{31, 30}));
    /* hit 0 twice: ages of 31/30 reach 2 (== steps) -> pruned, the
     * history of the third call is empty */
    RunGolden(adapt, golden, {0});
    RunGolden(adapt, golden, {0});
    ASSERT_EQ(adapt.getActiveSlotCount(), 1);
    ASSERT_EQ((int)golden.last.size(), 1);
    std::mt19937 rng(1111);
    std::uniform_int_distribution<int> gidDist(-2, 70);
    std::uniform_int_distribution<int> lenDist(0, 24);
    for (int iter = 0; iter < 200; ++iter) {
        std::vector<int> topk;
        const int n = lenDist(rng);
        for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
        RunGolden(adapt, golden, topk);
    }
}

/* windowSteps=0: no-cap mode, liveness via eff >= 1e-3 instead of
 * a step window. */
TEST(HBMAdaptGolden, DecayNoCapModeMatchesGolden)
{
    HBMManagerLayered adapt;
    GoldenHbm golden;
    const Shape s = {64, 32, 16, 8, 48, false};
    InitAdapt(adapt, s, /*layerId=*/-1, /*evictMode=*/1, HBMManagerLayered::kAutoPlan,
              HBMManagerLayered::kAutoLambda, /*windowSteps=*/0);
    ASSERT_TRUE(adapt.isNoCapMode());
    golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, true, /*steps=*/0, 0.85f,
                adapt.getWindowSize(), false);
    RunGolden(adapt, golden, {40, 41, 42});
    RunGolden(adapt, golden, {43});
    RunGolden(adapt, golden, {44});
    std::mt19937 rng(2222);
    std::uniform_int_distribution<int> gidDist(-2, 70);
    std::uniform_int_distribution<int> lenDist(0, 24);
    for (int iter = 0; iter < 300; ++iter) {
        std::vector<int> topk;
        const int n = lenDist(rng);
        for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
        RunGolden(adapt, golden, topk);
    }
}

/* windowPlan=1 (Budget4096, layer 30 -> 4096) + decayLambda=0.5:
 * bigger-than-immutable window clamped to 4096, faster score decay. */
TEST(HBMAdaptGolden, DecayPlan4096LambdaHalfMatchesGolden)
{
    const Shape s = {1024, 4608, 4096, 64, 4608, true};
    HBMManagerLayered adapt;
    adapt.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal,
               /*layerId=*/30, /*evictMode=*/1, /*windowPlan=*/1, /*decayLambda=*/0.5f);
    ASSERT_EQ(adapt.getWindowSize(), 4096);
    GoldenHbm golden;
    golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, true, /*steps=*/20,
                /*lambda=*/0.5f, 4096, false);
    /* identity hits only: empty history, full -1 pad to 4096 */
    std::vector<int> out;
    RunGolden(adapt, golden, {0, 1, 2, 3, 4, 5, 6, 7}, &out);
    ASSERT_EQ(out.size(), 4096u);
    EXPECT_EQ(std::vector<int>(out.begin(), out.begin() + 8),
              (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    for (int i = 8; i < 4096; ++i) { EXPECT_EQ(out[i], -1) << "idx " << i; }
    /* reverse-preloaded hit 500 (slot 780) + age-1 history 0..7 (eff 0.5,
     * tie -> slot asc) */
    RunGolden(adapt, golden, {500}, &out);
    ASSERT_EQ(out.size(), 4096u);
    EXPECT_EQ(out[0], 780);
    EXPECT_EQ(std::vector<int>(out.begin() + 1, out.begin() + 9),
              (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    for (int i = 9; i < 4096; ++i) { EXPECT_EQ(out[i], -1) << "idx " << i; }
    CompareGoldenState(adapt, golden);
    std::mt19937 rng(3333);
    std::uniform_int_distribution<int> gidDist(-2, 1030);
    std::uniform_int_distribution<int> lenDist(1, 6);
    for (int iter = 0; iter < 40; ++iter) {
        std::vector<int> topk;
        const int n = lenDist(rng);
        for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
        RunGolden(adapt, golden, topk);
    }
}

/* ============ golden differential tests (output dedup) ============ */

/* outputDedup=1, scripted: the golden model implements the
 * documented duplicate-free semantics (raw-slot per-call dedup + post-miss
 * residency refresh) and adapt must match it exactly in both evict modes.
 *  - rotation {0,2,3} on cap 2: stale hit refreshed to -1 -> output {1,0}
 *  - raw {12,12,0}: Lru {8,3,0,1}, decay {8,3,-1,-1}
 * Structural invariants: no duplicate slot ids; Lru outputs contain no -1
 * at all; decay outputs have -1 only as one contiguous tail section. */
TEST(HBMAdaptGolden, DedupOnScriptedMatchesGolden)
{
    for (int mode = 0; mode < 2; ++mode) {
        {
            HBMManagerLayered m;
            GoldenHbm golden;
            const Shape s = {4, 2, 2, 2, 4, true};
            m.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal, -1, mode,
                   HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
                   HBMManagerLayered::kAutoSteps, /*outputDedup=*/1);
            ASSERT_TRUE(m.isOutputDedup());
            golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, mode == 1, 20, 0.85f,
                        m.getWindowSize(), /*dedup=*/true);
            std::vector<int> out;
            RunGolden(m, golden, {0, 2, 3}, &out);
            ASSERT_EQ(out, (std::vector<int>{1, 0})) << "mode " << mode;
            AssertNoDuplicateSlots(out);
            AssertPadIsContiguousTail(out);
        }
        {
            HBMManagerLayered m;
            GoldenHbm golden;
            const Shape s = {8, 4, 4, 8, 8, false};
            m.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal, -1, mode,
                   HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
                   HBMManagerLayered::kAutoSteps, /*outputDedup=*/1);
            golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, mode == 1, 20, 0.85f,
                        m.getWindowSize(), /*dedup=*/true);
            std::vector<int> out;
            RunGolden(m, golden, {12, 12, 0}, &out);
            if (mode == 0) {
                /* Lru: [raw 8][backfill 3][chain fill 0, 1] */
                ASSERT_EQ(out, (std::vector<int>{8, 3, 0, 1})) << "mode " << mode;
                for (int v : out) { ASSERT_GE(v, 0) << "Lru output must not contain -1"; }
            } else {
                /* decay: window 4, no history candidate -> pad to 4 */
                ASSERT_EQ(out, (std::vector<int>{8, 3, -1, -1})) << "mode " << mode;
                AssertPadIsContiguousTail(out);
            }
            AssertNoDuplicateSlots(out);
        }
    }
}

/* outputDedup=1, fuzz: golden agreement plus duplicate-free output
 * structure in both evict modes. */
TEST(HBMAdaptGolden, DedupOnFuzzNoDuplicatesMatchesGolden)
{
    const Shape shapes[] = {{4, 2, 2, 2, 4, true}, {64, 32, 16, 8, 48, false}};
    for (int mode = 0; mode < 2; ++mode) {
        for (const auto& s : shapes) {
            HBMManagerLayered adapt;
            GoldenHbm golden;
            adapt.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal, -1,
                       mode, HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
                       HBMManagerLayered::kAutoSteps, /*outputDedup=*/1);
            golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, mode == 1, 20, 0.85f,
                        adapt.getWindowSize(), /*dedup=*/true);
            std::mt19937 rng(9090 + s.dram + mode);
            std::uniform_int_distribution<int> gidDist(-2, s.dram + (s.maxTotal - s.cap) + 2);
            std::uniform_int_distribution<int> lenDist(1, 12);
            for (int iter = 0; iter < 400; ++iter) {
                std::vector<int> topk;
                const int n = lenDist(rng);
                for (int i = 0; i < n; ++i) { topk.push_back(gidDist(rng)); }
                std::vector<int> oA;
                RunGolden(adapt, golden, topk, &oA);
                AssertNoDuplicateSlots(oA);
                if (mode == 1) {
                    AssertPadIsContiguousTail(oA);
                } else {
                    for (int v : oA) { ASSERT_GE(v, 0) << "iter " << iter; }
                }
            }
            CompareGoldenState(adapt, golden);
        }
    }
}

/* ============ init 运行期参数解析 ============ */

/* init 显式运行期参数（decayLambda / windowSteps / outputDedup）即最终
 * 配置；kAuto* 哨兵回落默认值（lambda 0.85 / steps 20 / 不去重，与
 * kvcache_hbm_manger.h 的原始方案一致）。lambda 取值通过 no-cap 模式可
 * 观测：λ=0.5 时分值 1 的槽在 age 10 处 eff=0.5^10<1e-3 被裁剪（若默认
 * λ=0.9 生效则 age 10 仍存活，活跃数会多 1）。 */
TEST(HBMAdaptGolden, InitRuntimeParamsResolve)
{
    /* 全部哨兵/默认：仅显式 outputDedup=1 生效 */
    {
        HBMManagerLayered m;
        GoldenHbm golden;
        const Shape s = {4, 2, 2, 2, 4, true};
        m.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal, -1,
               /*evictMode=*/0, HBMManagerLayered::kAutoPlan, HBMManagerLayered::kAutoLambda,
               HBMManagerLayered::kAutoSteps, /*outputDedup=*/1);
        ASSERT_TRUE(m.isOutputDedup());
        golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, /*decay=*/false, 20, 0.85f,
                    m.getWindowSize(), /*dedup=*/true);
        std::vector<int> out;
        RunGolden(m, golden, {0, 2, 3}, &out);
        ASSERT_EQ(out, (std::vector<int>{1, 0}));
    }
    /* 显式覆盖：evictMode=1 + lambda 0.5 / steps 0 / dedup 0 组合生效 */
    {
        HBMManagerLayered m;
        GoldenHbm golden;
        const Shape s = {64, 32, 16, 8, 48, false};
        m.init(s.dram, s.cap, s.immu, s.topk, kNullLogger, s.preload, s.maxTotal, -1,
               /*evictMode=*/1, HBMManagerLayered::kAutoPlan,
               /*decayLambda=*/0.5f, /*windowSteps=*/0, /*outputDedup=*/0);
        EXPECT_TRUE(m.isNoCapMode()) << "explicit steps=0 resolves to no-cap";
        EXPECT_FALSE(m.isOutputDedup()) << "explicit dedup=0 resolves to off";
        golden.init(s.dram, s.cap, s.immu, s.maxTotal, s.preload, /*decay=*/true, /*steps=*/0,
                    /*lambda=*/0.5f, m.getWindowSize(), /*dedup=*/false);
        /* 11 个不同 gid 各 miss 一次：age 达 10 的首个槽被裁剪 */
        for (int gid = 40; gid <= 50; ++gid) { RunGolden(m, golden, {gid}); }
        EXPECT_EQ(m.getActiveSlotCount(), 10)
            << "lambda=0.5: first slot pruned at age 10 (default 0.9 would keep 11 active)";
    }
}
