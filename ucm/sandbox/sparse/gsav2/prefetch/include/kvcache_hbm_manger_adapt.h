/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef GSA_HBM_MANAGER_ADAPT_H
#define GSA_HBM_MANAGER_ADAPT_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
#include <kvcache_log.h>

#ifdef DEBUG
#define HBM_ASSERT(f) assert(f)
#else
#define HBM_ASSERT(f) ((void)0)
#endif

// ================= 可配置项  ===============

enum class HbmEvictMode {
    Lru = 0,
    Decay = 1,
};

enum class HbmWindowPlan {
    Uniform = 0,
    Budget4096 = 1,
    Budget5120 = 2,
};

namespace gsa_layer_window {

/* 层窗口预算表：三档窗口大小（槽位计），由窗口计划 + 层号分段选取。 */
inline constexpr int kWindowSmall = 3072;
inline constexpr int kWindowMedium = 4096;
inline constexpr int kWindowLarge = 5120;

/* 层号分段边界（含边界层取该段窗口）。 */
inline constexpr int kBandEdge4096Low = 21;
inline constexpr int kBandEdge4096Mid = 33;
inline constexpr int kBandEdge4096High = 65;
inline constexpr int kBandEdge5120Low = 18;
inline constexpr int kBandEdge5120High = 62;

/* 衰减默认参数：init 未显式指定时的取值（默认 Lru + Uniform 下不参与
 * 任何行为，仅 decay 模式读取）。 */
inline constexpr float kDefaultDecayLambda = 0.85f;
inline constexpr int kDefaultWindowSteps = 20;

inline int getWindowSize4096(int layerId)
{
    if (layerId < 0) { return kWindowLarge; }
    if (layerId <= kBandEdge4096Low) { return kWindowSmall; }
    if (layerId <= kBandEdge4096Mid) { return kWindowMedium; }
    if (layerId <= kBandEdge4096High) { return kWindowLarge; }
    return kWindowSmall;
}

inline int getWindowSize5120(int layerId)
{
    if (layerId < 0) { return kWindowLarge; }
    if (layerId <= kBandEdge5120Low) { return kWindowMedium; }
    if (layerId <= kBandEdge5120High) { return kWindowLarge; }
    return kWindowMedium;
}

inline int getWindowSize(HbmWindowPlan plan, int layerId, int immutableSize)
{
    switch (plan) {
        case HbmWindowPlan::Budget4096: return getWindowSize4096(layerId);
        case HbmWindowPlan::Budget5120: return getWindowSize5120(layerId);
        case HbmWindowPlan::Uniform:
        default: return immutableSize;
    }
}

} // namespace gsa_layer_window

// ================= HBM 管理器 =================
/* HBMManagerLayered：kvcache_hbm_manger.h 中 HBMManager 的可调参扩展。
 * init 未显式开启任何扩展（默认参数，Lru + Uniform）时，查找/换入/输出
 * 逻辑与 HBMManager 逐分支等价（同一调用序列产生完全相同的外部可观测
 * 状态）；decay/window/dedup 仅在 init 显式开启后生效。
 *
 * 与原版一致的实现要点：
 *   - 去重用代际标记数组 mSeenGidFlag（uint8，255 回绕全清），仅覆盖
 *     [0, mDramTensorCount + mMaxTotalSlots - mHbmCapacity) 域；负 gid
 *     与超界 pinned gid 不去重（负 gid 占序号不出槽，与原版一致）；
 *   - scratch（slot 序列 / miss 序号）复用类成员，热路径零堆分配；
 *   - hitRate 仅在常规 gid 分支的去重序号 2047 处采样；
 *   - miss 上限 2048 到达即 break。
 *
 * 与原版一致的已知输出重复（默认行为，刻意镜像）：
 *   A) 重复超界 pinned gid 的 raw 槽原样重复输出（不在去重域内）；
 *   B) 小容量同调用驱逐轮转：hit 槽被本调用后续 miss 驱逐重载，hit 位
 *      置保留过期槽值与 miss 回填值重复。
 * init 传入 outputDedup=1 或 setOutputDedup(true) 可消除两者：raw 槽
 * 每调用去重 + miss 处理后按最终驻留校验各位置槽值；该开关默认关闭，
 * 关闭时行为与原版逐字节一致。 */
class HBMManagerLayered {
public:
    /* 显式参数哨兵：传 kAuto*（或越界值）表示该项取默认值（原始方案
     * Lru + Uniform，与 HBMManager 行为一致）。 */
    static constexpr int kAutoEvict = -1;
    static constexpr int kAutoPlan = -1;
    static constexpr float kAutoLambda = -1.0f;
    static constexpr int kAutoSteps = -1;
    static constexpr int kAutoDedup = -1;

    HBMManagerLayered() : mInitialized(false) {}

    /* init 输入即全部运行期开关：
     *   evictMode    0 = Lru / 1 = Decay
     *   windowPlan   0 = Uniform / 1 = Budget4096 / 2 = Budget5120
     *   decayLambda  (0, 1]，decay 模式的分值衰减系数
     *   windowSteps  >= 0，历史窗口步数（0 = no-cap，按分数阈值裁剪）
     *   outputDedup  0/1，输出去重开关
     * 各项传 kAuto*（或越界值）则取默认值：Lru + Uniform + lambda 0.85 +
     * steps 20 + 不去重（与 HBMManager 等价的原始方案）。 */
    void init(int dramTensorCount, int hbmCapacity, int hbmImmutableSize, int topkSize,
              Logger& mLogger, bool preloadHbm = true, int maxTotalSlots = 32768,
              int layerId = -1, int evictMode = kAutoEvict, int windowPlan = kAutoPlan,
              float decayLambda = kAutoLambda, int windowSteps = kAutoSteps,
              int outputDedup = kAutoDedup)
    {
        HBM_ASSERT(dramTensorCount > 0);
        HBM_ASSERT(hbmCapacity > 0);
        HBM_ASSERT(hbmImmutableSize > 0);
        HBM_ASSERT(topkSize > 0);
        HBM_ASSERT(hbmImmutableSize <= hbmCapacity);
        HBM_ASSERT(topkSize <= hbmCapacity);

        mDramTensorCount = dramTensorCount;
        mHbmCapacity = hbmCapacity;
        mHbmImmutableSize = hbmImmutableSize;
        mTopkSize = topkSize;
        mMaxTotalSlots = (maxTotalSlots > hbmCapacity) ? maxTotalSlots : hbmCapacity;
        mPinnedSlotCount = 0;
        mHitRate = 0.0f;

        resolveRuntimeConfig(evictMode, windowPlan, decayLambda, windowSteps, outputDedup);

        mLayerId = layerId;
        int win = gsa_layer_window::getWindowSize(mWindowPlan, layerId, hbmImmutableSize);
        if (win < mTopkSize) { win = mTopkSize; }
        if (win > mHbmImmutableSize) { win = mHbmImmutableSize; }
        mWindowSize = win;
        logInitConfig(mLogger);

        allocateState();
        if (preloadHbm) { preloadSlotMapping(); }
        initLruChain();
        /* 最尾可驱逐节点缓存：fresh 链全为常规非 pinned 槽，尾即候选。
         * 之后由 moveToHead / processTopkMisses 增量维护。 */
        mEvictTail = mLruTail;
        mAddedSlotGeneration = 0;
        mInitialized = true;
    }

    /* 预载映射：slot [0, min(256, mDramTensorCount-1)] 恒等映射 global，
     * 剩余 slot 倒序回填 [mDramTensorCount-1 ...] 倒序的 gid，止步于恒等区间
     *（越界回填会产生幽灵 slot：validateMapping 必败、LRU 误驱逐、双槽
     * gather）。剩余 slot 保持 -1（冷填按需启用）。 */
    void preloadSlotMapping()
    {
        int slotIdx = 0;
        for (; slotIdx <= kPreloadIdentityLimit && slotIdx < mHbmCapacity &&
               slotIdx < mDramTensorCount;
             ++slotIdx) {
            mGlobalToSlot[slotIdx] = slotIdx;
            mSlotToGlobal[slotIdx] = slotIdx;
        }
        const int identityCount = slotIdx;
        for (int gid = mDramTensorCount - 1;
             slotIdx < mHbmCapacity && gid >= identityCount; --gid, ++slotIdx) {
            mGlobalToSlot[gid] = slotIdx;
            mSlotToGlobal[slotIdx] = gid;
        }
    }

    // LRU 链表仅串常规 slots (0 <-> ... <-> mHbmCapacity-1)；pinned 按需动态加入。
    void initLruChain()
    {
        mLruHead = 0;
        mLruTail = mHbmCapacity - 1;
        for (int i = 0; i < mHbmCapacity; ++i) {
            mLru[i].prev = (i == 0) ? -1 : i - 1;
            mLru[i].next = (i == mHbmCapacity - 1) ? -1 : i + 1;
        }
    }

    /* topk 查找主路径：与 HBMManager::processTopkWithDebug 同构——代际标记
     * 去重 + hit/miss 分类合流单循环 + miss 换入 + 输出组装；decay 模式仅
     * 在分类/换入路径上挂 touch/clear 钩子并在输出段分流。 */
    void processTopkWithDebug(const std::vector<int>& topkIds, std::vector<int>& output,
                              std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                              Logger& mLogger)
    {
        HBM_ASSERT(mInitialized);
        mTopkSize = topkIds.size();
        missGlobalIds.reserve(mTopkSize);
        missSlots.reserve(mTopkSize);
        bumpGeneration();
        /* 记录调用起点的链头：所有本调用触碰过的节点（hit 移头/pinned 头插/
         * miss 载入头插）都被搬进链头段，未触碰段保持原相对顺序。若起点
         * 未被触碰，输出填充段可直接从起点开始遍历，跳过整个已触碰头段。 */
        mCallStartHead = mLruHead;
        bumpSeenGeneration();
        if (mEvictMode == HbmEvictMode::Decay) { ++mDecayStep; }
        mScratchSlotIds.assign(mTopkSize, -1);
        mScratchMissIndices.clear();
        if (mOutputDedup) {
            mScratchGids.assign(mTopkSize, -1);
            mScratchRawSlots.clear();
        }
        int dedupedCount = 0;
        for (int i = 0; i < mTopkSize; ++i) {
            if (!classifyTopkGid(topkIds[i], i, dedupedCount, missGlobalIds, mLogger)) { break; }
        }
        processTopkMisses(mScratchSlotIds, missGlobalIds, missSlots, mScratchMissIndices,
                          mLogger);
        if (mOutputDedup) { refreshResidentSlots(dedupedCount); }
        if (mEvictMode == HbmEvictMode::Decay) {
            touchLoadedVictims(missSlots);
            pruneDecay();
            buildTopkOutputDecay(mScratchSlotIds, dedupedCount, output);
        } else {
            buildTopkOutputLru(mScratchSlotIds, dedupedCount, output);
        }
    }

    int getSlotForGlobal(int gid) const
    {
        if (gid >= mDramTensorCount) { return gid - mDramTensorCount + mHbmCapacity; }
        if (gid >= 0 && gid < mDramTensorCount) { return mGlobalToSlot[gid]; }
        return -1;
    }

    // pinned slot -> gid: 由公式推导，gid = slot - mHbmCapacity + mDramTensorCount
    int getGidForSlot(int slot) const
    {
        if (slot >= 0 && slot < mHbmCapacity) { return mSlotToGlobal[slot]; }
        if (slot >= mHbmCapacity && slot < mMaxTotalSlots &&
            (mLru[slot].flags & kLruNodePinned)) {
            return slot - mHbmCapacity + mDramTensorCount;
        }
        return -1;
    }

    bool isInitialized() const { return mInitialized; }
    int getTopkSize() const { return mTopkSize; }
    int getHbmCapacity() const { return mHbmCapacity; }
    int getHbmImmutableSize() const { return mHbmImmutableSize; }
    int getWindowSize() const { return mWindowSize; }
    int getLayerId() const { return mLayerId; }
    int getDramTensorCount() const { return mDramTensorCount; }
    int getPinnedSlotCount() const { return mPinnedSlotCount; }
    int getMaxTotalSlots() const { return mMaxTotalSlots; }
    int getActiveSlotCount() const { return (int)mActiveSlots.size(); }
    bool isDecayMode() const { return mEvictMode == HbmEvictMode::Decay; }

    const char* getEvictModeName() const
    {
        return (mEvictMode == HbmEvictMode::Decay) ? "decay" : "lru";
    }

    const char* getWindowPlanName() const
    {
        switch (mWindowPlan) {
            case HbmWindowPlan::Budget4096: return "4096";
            case HbmWindowPlan::Budget5120: return "5120";
            default: return "uniform";
        }
    }

    void setEvictMode(HbmEvictMode mode) { mEvictMode = mode; }

    void setWindowSize(int win)
    {
        if (win < mTopkSize) { win = mTopkSize; }
        if (win > mHbmImmutableSize) { win = mHbmImmutableSize; }
        mWindowSize = win;
    }

    void setDecayLambda(float lam) { mDecayLambda = lam; rebuildPowTable(); }
    /* 输出去重的运行时开关；开启后输出无重复槽 ID：raw 超界 pinned 槽
     * 每调用去重 + hit 槽驻留校验。默认关闭。 */
    void setOutputDedup(bool on) { mOutputDedup = on; }
    bool isOutputDedup() const { return mOutputDedup; }
    /* 与 init 参数语义保持一致（windowSteps=0 => no-cap）：0 合法，仅负数
     * 回落默认 20。原先 `steps > 0` 会把 0 吞成 20，使 no-cap 模式无法
     * 通过 setter 进入。 */
    void setWindowSteps(int steps)
    {
        mWindowSteps = (steps >= 0) ? steps : 20;
        rebuildPowTable();
    }
    bool isNoCapMode() const { return mWindowSteps == 0; }

    bool validateLruChain() const
    {
        int expectedCount = mHbmCapacity + mPinnedSlotCount;
        int curr = mLruHead;
        int count = 0;
        int safetyLimit = mMaxTotalSlots + 10;

        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLru[curr].next;
        }

        if (count >= safetyLimit || count != expectedCount) return false;

        curr = mLruTail;
        count = 0;
        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLru[curr].prev;
        }

        return count == expectedCount;
    }

    bool validateMapping() const
    {
        // 检查常规 slot 映射
        for (int slot = 0; slot < mHbmCapacity; ++slot) {
            int gid = mSlotToGlobal[slot];
            if (gid == -1) continue;
            if (gid < -1 || gid >= mDramTensorCount) return false;
            if (mGlobalToSlot[gid] != slot) return false;
        }
        // 检查已初始化的 pinned slot 映射 (gid 由公式推导)
        for (int slot = mHbmCapacity; slot < mMaxTotalSlots; ++slot) {
            if (!(mLru[slot].flags & kLruNodePinned)) continue;
            int expectedGid = slot - mHbmCapacity + mDramTensorCount;
            // pinned slot 的 gid 必须与公式一致
            if (expectedGid < mDramTensorCount) return false;
        }
        return true;
    }

    // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]
    // LastChunkMove 等外部代码仅遍历此范围
    std::vector<int> mSlotToGlobal;
    std::vector<int> mGlobalToSlot;
    float mHitRate;

private:
    int mDramTensorCount;
    int mHbmCapacity;
    int mHbmImmutableSize;
    int mTopkSize;
    int mPinnedSlotCount;
    int mMaxTotalSlots;
    int mLayerId;
    int mWindowSize;
    HbmEvictMode mEvictMode = HbmEvictMode::Lru;
    HbmWindowPlan mWindowPlan = HbmWindowPlan::Uniform;

    /* 融合 LRU 节点：16 字节恰好一个节点内 prev/next/本调用代际/固定槽位，
     * 链跳、命中标记、pinned 判别均单 cache line 命中（对齐后无 padding）。 */
    struct LruNode {
        int prev;
        int next;
        uint32_t gen;      // 本调用代际标记（原 mAddedSlotFlag）
        uint32_t flags;    // bit0: kLruNodePinned（原 mIsPinned）
    };
    static constexpr uint32_t kLruNodePinned = 1u;
    std::vector<LruNode> mLru;

    // 去重标记数组（代际模式；与 mAddedSlotFlag 的 INT_MAX 翻转不同：uint8 空间
    // 小，代际到 255 时全量清零回 1，杜绝 8bit 空间的陈旧别名命中）
    std::vector<uint8_t> mSeenGidFlag;
    // 每调用 scratch：同尺寸 assign / clear 复用，免逐步堆分配
    std::vector<int> mScratchSlotIds;
    std::vector<int> mScratchMissIndices;
    // 输出去重（outputDedup=1）专用 scratch：各去重位置的 gid 快照
    // 与本调用出现过的 raw 超界 pinned 槽列表；关闭时不触碰，零开销
    std::vector<int> mScratchGids;
    std::vector<int> mScratchRawSlots;
    bool mOutputDedup = false;

    int mLruHead;
    int mLruTail;
    int mCallStartHead;
    int mEvictTail;
    uint32_t mAddedSlotGeneration;
    int mSeenGidGeneration;
    bool mInitialized;

    std::vector<float> mDecayScore;
    std::vector<int> mDecayLastStep;
    std::vector<bool> mActiveFlag;
    std::vector<int> mActiveSlots;
    int mDecayStep = 0;
    float mDecayLambda = gsa_layer_window::kDefaultDecayLambda;
    int mWindowSteps = gsa_layer_window::kDefaultWindowSteps;

    /* ---- 内部常量（行为契约的一部分，测试按字面值固化） ---- */
    static constexpr int kMissThreshold = 2048;       /* 单调用 miss 上限，到达即 break */
    static constexpr int kHitRateSamplePos = kMissThreshold - 1;  /* 命中率采样去重序号 */
    static constexpr int kSeenGidGenerationMax = 255; /* uint8 去重代际回绕点（全清回 1） */
    static constexpr int kPreloadIdentityLimit = 256; /* 恒等预载上界（原版行为，含 256） */
    static constexpr int kMinPowTableSize = 64;       /* 有窗口模式的最小幂表长度 */
    static constexpr float kDecayTouchIncrement = 1.0f; /* 每次触达的分值增量 */
    static constexpr float kMinDecayScore = 1e-3f;   /* no-cap 模式的存活分值阈值 */
    static constexpr int kPowTableSize = 512;         /* no-cap 模式的幂表长度 */
    std::vector<float> mPowTable;
    std::vector<std::pair<float, int>> mRankedBuf;

    /* 历史候选全序：有效分值降序，平分按槽位升序（确定性输出）。 */
    struct RankEffDesc {
        bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const
        {
            if (a.first != b.first) { return a.first > b.first; }
            return a.second < b.second;
        }
    };

    /* ---- init 子步骤 ---- */

    /* 运行期配置解析：init 显式参数即唯一配置来源（合法域内直接生效），
     * kAuto* / 越界值取默认值——原始方案 Lru + Uniform + lambda 0.85 +
     * steps 20 + 不去重（与 HBMManager 等价）。 */
    void resolveRuntimeConfig(int evictMode, int windowPlan, float decayLambda, int windowSteps,
                              int outputDedup)
    {
        mEvictMode = (evictMode == kAutoEvict)
                        ? HbmEvictMode::Lru
                        : ((evictMode == 1) ? HbmEvictMode::Decay : HbmEvictMode::Lru);
        mWindowPlan = (windowPlan == kAutoPlan)
                        ? HbmWindowPlan::Uniform
                        : ((windowPlan == 1) ? HbmWindowPlan::Budget4096
                                             : ((windowPlan == 2) ? HbmWindowPlan::Budget5120
                                                                  : HbmWindowPlan::Uniform));
        mDecayLambda = (decayLambda > 0.0f && decayLambda <= 1.0f)
                            ? decayLambda
                            : gsa_layer_window::kDefaultDecayLambda;
        mWindowSteps = (windowSteps >= 0) ? windowSteps : gsa_layer_window::kDefaultWindowSteps;
        mOutputDedup = (outputDedup >= 0) ? (outputDedup != 0) : false;
    }

    void logInitConfig(Logger& mLogger)
    {
        mLogger.log(
            LogLevel::DEBUG,
            "|HBM Manager Layered| Init layer: %d, evictMode: %s, windowPlan: %s, "
            "mDramTensorCount: %d, mHbmCapacity: %d, mHbmImmutableSize: %d, "
            "mWindowSize: %d, mTopkSize: %d, mMaxTotalSlots: %d, lambda: %.2f, steps: %d, "
            "outputDedup: %d\n",
            mLayerId, (mEvictMode == HbmEvictMode::Decay) ? "decay" : "lru",
            (mWindowPlan == HbmWindowPlan::Budget4096)
                ? "4096"
                : (mWindowPlan == HbmWindowPlan::Budget5120) ? "5120" : "uniform",
            mDramTensorCount, mHbmCapacity, mHbmImmutableSize, mWindowSize, mTopkSize,
            mMaxTotalSlots, mDecayLambda, mWindowSteps, (int)mOutputDedup);
    }

    /* 按模式分配状态数组；decay 的分数/活跃集/幂表仅在该模式驻留，
     * Lru 模式零开销。 */
    void allocateState()
    {
        mGlobalToSlot.assign(mDramTensorCount, -1);
        // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]；pinned slot 的 gid
        // 由公式推导: gid = slot - mHbmCapacity + mDramTensorCount
        mSlotToGlobal.assign(mHbmCapacity, -1);
        /* LRU 节点融合数组：原版 mLruPrev/mLruNext/mIsPinned/mAddedSlotFlag
         * 四个平行数组在链跳/命中标记/固定槽判别时每步跨数组随机访问
         * （2-3 次 cache miss/跳，输出填充段每调用 ~immutable 跳）。融合
         * 为 16B 单节点后每次链跳只触碰 1 条 cache line。外部可观测行为
         * 不变。 */
        mLru.assign(mMaxTotalSlots, LruNode{-1, -1, 0, 0});
        // 去重标记数组：覆盖常规 gid 域与 pinned gid 公式域两段连续区间。
        mSeenGidFlag.assign(mDramTensorCount + mMaxTotalSlots - mHbmCapacity, 0);
        mSeenGidGeneration = 0;
        if (mEvictMode == HbmEvictMode::Decay) {
            mDecayScore.assign(mMaxTotalSlots, 0.0f);
            mDecayLastStep.assign(mMaxTotalSlots, -1);
            mActiveFlag.assign(mMaxTotalSlots, false);
            mActiveSlots.clear();
            mActiveSlots.reserve((size_t)mWindowSteps * mTopkSize);
            rebuildPowTable();
        } else {
            mDecayScore.clear();
            mDecayLastStep.clear();
            mActiveFlag.clear();
            mActiveSlots.clear();
        }
        mDecayStep = 0;
    }

    /* ---- 每调用子步骤 ---- */

    /* 槽位代际推进：UINT32_MAX 后回绕到 1（0 保留给 init 后首调用，避免
     * 与"从未标记"的初值撞车）。 */
    void bumpGeneration()
    {
        mAddedSlotGeneration = (mAddedSlotGeneration == UINT32_MAX) ? 1
                                                                    : mAddedSlotGeneration + 1;
    }

    /* uint8 去重代际推进：到回绕点全量清零重来，杜绝 8bit 空间的陈旧
     * 别名命中。 */
    void bumpSeenGeneration()
    {
        if (mSeenGidGeneration == kSeenGidGenerationMax) {
            std::fill(mSeenGidFlag.begin(), mSeenGidFlag.end(), 0);
            mSeenGidGeneration = 1;
        } else {
            ++mSeenGidGeneration;
        }
    }

    /* 输出去重（outputDedup=1）：miss 换入完成后按最终驻留刷新
     * 各位置的槽值。修掉"同调用驱逐轮转"重复（场景 B）：刚命中的槽若被
     * 本调用后续 miss 驱逐重载，hit 位置的槽值已过期（旧 gid 的位置将
     * gather 到新 gid 的数据），按 mGlobalToSlot 刷新为 -1；miss 位置本就
     * 是终值，刷新幂等。默认关闭时本段完全跳过，行为与原版逐字节一致。 */
    void refreshResidentSlots(int dedupedCount)
    {
        for (int i = 0; i < dedupedCount; ++i) {
            int gid = mScratchGids[i];
            if (gid < 0 || gid >= mDramTensorCount) { continue; }
            int live = mGlobalToSlot[gid];
            if (mScratchSlotIds[i] != live) { mScratchSlotIds[i] = live; }
        }
    }

    void touchLoadedVictims(const std::vector<int>& missSlots)
    {
        for (int slot : missSlots) { touchDecay(slot); }
    }

    inline float powLambda(int k) const
    {
        if (k >= 0 && k < (int)mPowTable.size()) { return mPowTable[k]; }
        return std::pow(mDecayLambda, (float)k);
    }

    inline bool isDecayAlive(int slot, int age) const
    {
        if (age < 0) { return false; }
        if (mWindowSteps > 0) { return age < mWindowSteps; }
        return mDecayScore[slot] * powLambda(age) >= kMinDecayScore;
    }

    void rebuildPowTable()
    {
        int n = (mWindowSteps > 0) ? std::max(mWindowSteps, kMinPowTableSize) : kPowTableSize;
        mPowTable.resize(n + 1);
        mPowTable[0] = 1.0f;
        for (int k = 1; k <= n; ++k) {
            mPowTable[k] = mPowTable[k - 1] * mDecayLambda;
        }
    }

    /* 单 gid 分类入 scratch：去重命中跳过 / pinned gid 换入并出槽 / 负 gid
     * 占序号不出槽 / 常规 gid hit 出槽或 miss 入栈（kMissThreshold 上限
     * break）。hit rate 在去重序号 kHitRateSamplePos 处采样。返回 false =
     * miss 上限到达，停止循环。 */
    bool classifyTopkGid(int gid, int origIndex, int& dedupedCount,
                         std::vector<int>& missGlobalIds, Logger& mLogger)
    {
        /* 去重开关开启时记录每个去重位置的 gid，供 miss 后的驻留校验使用；
         * 重复 gid 提前返回不占位，同一下标会被下一个新 gid 覆写。 */
        if (mOutputDedup) { mScratchGids[dedupedCount] = gid; }
        if (markGidSeen(gid)) {
            mLogger.log(LogLevel::ERROR,
                        "|HBM Manager| Duplicate gid in topkIds: %d at index: %d, "
                        "skipping\n",
                        gid, origIndex);
            return true;
        }
        if (gid >= mDramTensorCount) {
            int transformedSlot = gid - mDramTensorCount + mHbmCapacity;
            return handlePinnedGid(gid, transformedSlot, dedupedCount, mLogger);
        }
        if (gid < 0) {
            mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| Topk is abnormal value: %d \n", gid);
            ++dedupedCount;
            return true;
        }
        int slot = mGlobalToSlot[gid];
        if (slot != -1) {
            moveToHead(slot);
            mScratchSlotIds[dedupedCount] = slot;
            mLru[slot].gen = mAddedSlotGeneration;
            if (mEvictMode == HbmEvictMode::Decay) { touchDecay(slot); }
        } else {
            missGlobalIds.push_back(gid);
            mScratchMissIndices.push_back(dedupedCount);
            if ((int)mScratchMissIndices.size() == kMissThreshold) { return false; }
        }
        if (dedupedCount == kHitRateSamplePos) {
            mHitRate = 1 - mScratchMissIndices.size() * 1.0 / kMissThreshold;
        }
        ++dedupedCount;
        return true;
    }

    /* 域内 gid 去重：已见返回 true（调用方跳过并记日志），未见则打代际
     * 标记返回 false。负 gid 与超界 pinned gid 不在域内，不去重（与原版
     * 一致）。 */
    bool markGidSeen(int gid)
    {
        const int gidBound = static_cast<int>(mSeenGidFlag.size());
        if (gid < 0 || gid >= gidBound) { return false; }
        if (mSeenGidFlag[gid] == mSeenGidGeneration) { return true; }
        mSeenGidFlag[gid] = mSeenGidGeneration;
        return false;
    }

    /* pinned gid 换入：首次置位 + LRU 头插，否则移头；超界 pinned 的原始 slot
     * 值照写（不参与 LRU，行为不变）。 */
    bool handlePinnedGid(int gid, int transformedSlot, int& dedupedCount, Logger& mLogger)
    {
        if (transformedSlot >= mMaxTotalSlots) {
            return emitRawPinnedSlot(gid, transformedSlot, dedupedCount, mLogger);
        }
        pinOrTouch(transformedSlot);
        mScratchSlotIds[dedupedCount] = transformedSlot;
        mLru[transformedSlot].gen = mAddedSlotGeneration;
        if (mEvictMode == HbmEvictMode::Decay) { touchDecay(transformedSlot); }
        ++dedupedCount;
        return true;
    }

    void pinOrTouch(int slot)
    {
        if (!(mLru[slot].flags & kLruNodePinned)) {
            mLru[slot].flags |= kLruNodePinned;
            insertToHead(slot);
            mPinnedSlotCount++;
        } else {
            moveToHead(slot);
        }
    }

    /* 超界 pinned gid 的 raw slot 原样透传（不参与 LRU 与输出去重域）。
     * 输出去重开启时对 raw 槽做每调用小列表去重（场景 A：raw 槽不在
     * mSeenGidFlag 去重域内，重复超界 pinned gid 会原样重复输出；异常
     * 路径罕见，线性扫描足够）。重复出现直接跳过、不占去重位，与常规
     * 重复 gid 的语义一致。 */
    bool emitRawPinnedSlot(int gid, int transformedSlot, int& dedupedCount, Logger& mLogger)
    {
        mLogger.log(LogLevel::ERROR,
                    "|HBM Manager| Pinned slot %d exceeds maxTotalSlots %d, gid: %d\n",
                    transformedSlot, mMaxTotalSlots, gid);
        if (mOutputDedup) {
            for (int s : mScratchRawSlots) {
                if (s == transformedSlot) { return true; }
            }
            mScratchRawSlots.push_back(transformedSlot);
        }
        mScratchSlotIds[dedupedCount] = transformedSlot;
        ++dedupedCount;
        return true;
    }

    /* miss 换入与回填：victim 自 LRU 尾部驱逐（跳过 pinned）。missGlobalIds 与
     * missIndices 平行入栈（missIndices[j] 为第 j 个 miss 的去重后序号），回填
     * 直接取 missGlobalIds[j]，无需 deduped 序列二次查表。 */
    void processTopkMisses(std::vector<int>& topkSlotIds,
                           std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                           const std::vector<int>& missIndices, Logger& mLogger)
    {
        const int missCount = (int)missGlobalIds.size();
        int loadedCount = 0;
        for (int i = 0; i < missCount; ++i) {
            if (mEvictTail == -1) {
                mLogger.log(LogLevel::ERROR,
                            "|HBM Manager| No evictable slot for miss gid: %d, "
                            "remaining %d misses dropped\n",
                            missGlobalIds[i], missCount - i);
                continue;
            }
            missSlots.push_back(evictAndLoad(mEvictTail, missGlobalIds[i], mLogger));
            loadedCount++;
        }
        missGlobalIds.resize(loadedCount);
        backfillMissSlots(topkSlotIds, missGlobalIds, missIndices);
    }

    /* victim 直接取 mEvictTail（最尾非 pinned 节点，与原版自 mLruTail
     * 走链跳过 pinned 的结果恒等）。原版逐 miss 重走 pinned 尾部聚簇
     * （O(miss*pinned)），此处增量维护摊还 O(1)：victim 即将被钉到链头，
     * 新的最尾可驱逐节点从其原前驱回退查找（跳过的 pinned 节点此后
     * 不再被重复扫描）。返回装载了 newGid 的 victim 槽位。 */
    int evictAndLoad(int victimSlot, int newGid, Logger& mLogger)
    {
        const int victimOldPrev = mLru[victimSlot].prev;
        int oldGid = mSlotToGlobal[victimSlot];
        if (oldGid >= 0 && oldGid < mDramTensorCount) {
            mGlobalToSlot[oldGid] = -1;
        } else if (oldGid == -1) {
            mLogger.log(LogLevel::WARNING, "|HBM Manager| Evicting unallocated slot: %d\n",
                        victimSlot);
        } else {
            mLogger.log(
                LogLevel::ERROR,
                "|HBM Manager| Evicting slot with out-of-range gid: %d in slot: %d\n",
                oldGid, victimSlot);
        }
        if (mEvictMode == HbmEvictMode::Decay) { clearDecay(victimSlot); }

        mSlotToGlobal[victimSlot] = newGid;
        mGlobalToSlot[newGid] = victimSlot;
        removeFromList(victimSlot);
        insertToHead(victimSlot);
        /* victim 原前驱为 -1（victim 即链头，唯一常规槽场景）时，回退查找
         * 为空：victim 已重载并仍在链头，它自身就是新的最尾可驱逐节点。 */
        mEvictTail = (victimOldPrev == -1) ? victimSlot : prevEvictable(victimOldPrev);
        return victimSlot;
    }

    /* 回填各 miss 位置的本调用最终槽值并打代际标记。 */
    void backfillMissSlots(std::vector<int>& topkSlotIds, const std::vector<int>& missGlobalIds,
                           const std::vector<int>& missIndices)
    {
        const int loadedCount = (int)missGlobalIds.size();
        for (int j = 0; j < loadedCount; ++j) {
            int slot = mGlobalToSlot[missGlobalIds[j]];
            topkSlotIds[missIndices[j]] = slot;
            if (slot >= 0 && slot < mMaxTotalSlots) {
                mLru[slot].gen = mAddedSlotGeneration;
            }
        }
    }

    /* 真实 topk 段：输出本调用 hit + 回填 miss 的槽（去重序，负值跳过，
     * raw 槽原样透传）。返回段长。 */
    int emitRealTopk(const std::vector<int>& topkSlotIds, int dedupedSize,
                     std::vector<int>& output)
    {
        int realTopkSlotNum = 0;
        for (int i = 0; i < dedupedSize; ++i) {
            if (topkSlotIds[i] < 0) { continue; }
            output.push_back(topkSlotIds[i]);
            realTopkSlotNum += 1;
        }
        return realTopkSlotNum;
    }

    void buildTopkOutputLru(const std::vector<int>& topkSlotIds, int dedupedSize,
                            std::vector<int>& output)
    {
        output.reserve(mHbmImmutableSize);
        int realTopkSlotNum = emitRealTopk(topkSlotIds, dedupedSize, output);
        fillFromUntouchedChain(output, mHbmImmutableSize - realTopkSlotNum);
    }

    /* 填充段：自 LRU 链头方向收未触碰槽补满 slotsNeeded。 */
    void fillFromUntouchedChain(std::vector<int>& output, int slotsNeeded)
    {
        int count = 0;
        if (mCallStartHead != -1 && mLru[mCallStartHead].gen != mAddedSlotGeneration) {
            /* 快路径：起点未触碰 => 未触碰段恰从起点开始（触碰过的节点全在
             * 其前方），无需逐节点查代际，直接顺序收满。 */
            int curr = mCallStartHead;
            while (curr != -1 && count < slotsNeeded) {
                output.push_back(curr);
                mLru[curr].gen = mAddedSlotGeneration;
                count++;
                curr = mLru[curr].next;
            }
        } else {
            /* 慢路径（起点本身被触碰——如起点恰为命中且未移位的链头）：
             * 与原版一致的自链头检查式遍历。 */
            int curr = mLruHead;
            while (curr != -1 && count < slotsNeeded) {
                if (mLru[curr].gen != mAddedSlotGeneration) {
                    output.push_back(curr);
                    mLru[curr].gen = mAddedSlotGeneration;
                    count++;
                }
                curr = mLru[curr].next;
            }
        }
    }

    /* decay 输出三段组装：[真实 topk（本步 hit + 回填 miss 槽，依去重序）]
     * [m20 历史窗口（mActiveSlots 中 age < mWindowSteps 的存活槽，按有效
     * 分值降序，预算 mWindowSize - 真实topk 数——不同层窗口不一致）]
     * [-1 补齐到 mHbmImmutableSize（== 生产端 mSpaTopkLen，消费端按精确
     * 长度校验，短/超长均被拒收）]。 */
    void buildTopkOutputDecay(const std::vector<int>& topkSlotIds, int dedupedSize,
                              std::vector<int>& output)
    {
        output.clear();
        output.reserve(mHbmImmutableSize);
        /* 不在此写 mAddedSlotFlag：有效 slot 的代际标记已在
         * classifyTopkGid / handlePinnedGid / processTopkMisses 回填中置位，
         * 而超界 pinned gid 的 raw slot（>= mMaxTotalSlots）会原样进入本段，
         * 对其写 flag 属越界堆写；raw slot 也永远不会出现在 mActiveSlots
         * 候选中，无需标记。 */
        const int realTopkSlotNum = emitRealTopk(topkSlotIds, dedupedSize, output);
        const int histNeeded = mWindowSize - realTopkSlotNum;
        if (histNeeded > 0 && !mActiveSlots.empty()) {
            collectRankedHistory(histNeeded);
            emitHistorySlots(output);
        }
        while ((int)output.size() < mHbmImmutableSize) {
            output.push_back(-1);
        }
        /* 真实 topk 段不做截断（原版 buildTopkOutput 亦不截断）：topkSize
         * 可大于 mHbmImmutableSize（init 仅断言 <= mHbmCapacity），超长输出
         * 由消费端按 mSpaTopkLen 契约拒收；截断反而让消费端拿到残缺的
         * gather 表。 */
    }

    /* 历史候选收集：未标记、存活且有效分值为正的活跃槽，按 (eff 降序,
     * slot 升序) 全序排序后截断到 histNeeded。 */
    void collectRankedHistory(int histNeeded)
    {
        mRankedBuf.clear();
        for (int slot : mActiveSlots) {
            if (mLru[slot].gen == mAddedSlotGeneration) { continue; }
            int age = mDecayStep - mDecayLastStep[slot];
            if (!isDecayAlive(slot, age)) { continue; }
            float eff = mDecayScore[slot] * powLambda(age);
            if (eff <= 0.0f) { continue; }
            mRankedBuf.emplace_back(eff, slot);
        }
        auto& ranked = mRankedBuf;
        if ((int)ranked.size() > histNeeded) {
            std::nth_element(ranked.begin(), ranked.begin() + histNeeded, ranked.end(),
                             RankEffDesc());
            ranked.resize(histNeeded);
        }
        std::sort(ranked.begin(), ranked.end(), RankEffDesc());
    }

    void emitHistorySlots(std::vector<int>& output)
    {
        for (const auto& kv : mRankedBuf) {
            if ((int)output.size() >= mWindowSize) { break; }
            if (mLru[kv.second].gen == mAddedSlotGeneration) { continue; }
            output.push_back(kv.second);
            mLru[kv.second].gen = mAddedSlotGeneration;
        }
    }

    inline void touchDecay(int slot)
    {
        if (mEvictMode != HbmEvictMode::Decay) { return; }
        if (slot < 0 || slot >= mMaxTotalSlots) { return; }
        if (mDecayLastStep[slot] >= 0) {
            int age = mDecayStep - mDecayLastStep[slot];
            if (age > 0) {
                mDecayScore[slot] *= powLambda(age);
            }
        }
        mDecayScore[slot] += kDecayTouchIncrement;
        mDecayLastStep[slot] = mDecayStep;
        if (!mActiveFlag[slot]) {
            mActiveFlag[slot] = true;
            mActiveSlots.push_back(slot);
        }
    }

    inline void clearDecay(int slot)
    {
        if (mEvictMode != HbmEvictMode::Decay) { return; }
        if (slot < 0 || slot >= mMaxTotalSlots) { return; }
        mDecayScore[slot] = 0.0f;
        mDecayLastStep[slot] = -1;
    }

    void pruneDecay()
    {
        if (mEvictMode != HbmEvictMode::Decay) { return; }
        int write = 0;
        for (int read = 0; read < (int)mActiveSlots.size(); ++read) {
            int slot = mActiveSlots[read];
            int last = mDecayLastStep[slot];
            int age = mDecayStep - last;
            if (last < 0 || !isDecayAlive(slot, age)) {
                mActiveFlag[slot] = false;
                mDecayScore[slot] = 0.0f;
                mDecayLastStep[slot] = -1;
                continue;
            }
            mActiveSlots[write++] = slot;
        }
        mActiveSlots.resize(write);
    }

    inline void removeFromList(int slot)
    {
        LruNode& n = mLru[slot];
        if (n.prev != -1) {
            mLru[n.prev].next = n.next;
        } else {
            mLruHead = n.next;
        }
        if (n.next != -1) {
            mLru[n.next].prev = n.prev;
        } else {
            mLruTail = n.prev;
        }
    }

    inline void insertToHead(int slot)
    {
        LruNode& n = mLru[slot];
        n.prev = -1;
        n.next = mLruHead;
        if (mLruHead != -1) mLru[mLruHead].prev = slot;
        mLruHead = slot;
        if (mLruTail == -1) mLruTail = slot;
    }

    inline void moveToHead(int slot)
    {
        if (slot == mLruHead) return;
        /* 被移头的是最尾可驱逐节点时需刷新缓存：读取原前驱须在摘链前。 */
        const bool wasEvictTail = (slot == mEvictTail);
        const int oldPrev = mLru[slot].prev;
        removeFromList(slot);
        insertToHead(slot);
        if (wasEvictTail) { mEvictTail = prevEvictable(oldPrev); }
    }

    /* 自 from 起向链头方向回退，跳过 pinned 节点，返回首个可驱逐节点。
     * 摊还分析：每个 pinned 节点相对候选序单调后移，至多被跳过 O(1) 次。 */
    inline int prevEvictable(int from) const
    {
        int c = from;
        while (c != -1 && (mLru[c].flags & kLruNodePinned)) { c = mLru[c].prev; }
        return c;
    }
};

#endif  // GSA_HBM_MANAGER_ADAPT_H
