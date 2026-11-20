/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef ATB_KV_CACHE_PRE_DETAIL_H
#define ATB_KV_CACHE_PRE_DETAIL_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include "ucmstore_v1.h"

namespace ucmprefetch {

class GSAPrefetchEngineC;

// 典型 CPU cache line 大小（字节），用于隔离跨进程共享结构的伪共享。
constexpr size_t kKVCacheLineSize = 64;

// KVPack 跨进程共享布局（详见 kvcache_pre_detail.h 的使用者）。
struct alignas(kKVCacheLineSize) KVPackRankState {
    std::atomic<uint64_t> packedEpoch{0};
    std::atomic<uint64_t> h2dDoneEpoch{0};
};

struct alignas(kKVCacheLineSize) SharedKVPackHeader {
    std::atomic<uint32_t> initialized{0};
    std::atomic<int32_t> fatalError{0};
};

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "KV pack requires lock-free uint64_t atomics");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "KV pack requires lock-free uint32_t atomics");
static_assert(std::atomic<int32_t>::is_always_lock_free,
              "KV pack requires lock-free int32_t atomics");

inline int KVPackFail(SharedKVPackHeader* header)
{
    header->fatalError.store(-1, std::memory_order_release);
    return -1;
}

// Coverage flush helper: flushes gcov data before abort (for death-test coverage).
// Declared without weak so the linker pulls __gcov_dump from libgcov.a; the
// -lgcov link flag is added by run_tests.sh only in COVERAGE mode.

#if defined(UCM_PREFETCH_COVERAGE)
extern "C" void __gcov_dump(void);
#endif

[[noreturn]] inline void KVPackAbort(SharedKVPackHeader* header)
{
    if (header != nullptr) { header->fatalError.store(-1, std::memory_order_release); }
#if defined(UCM_PREFETCH_COVERAGE)
    // Flush coverage data before abort so death-test lines are counted.
    __gcov_dump();
#endif
    std::abort();
}

/* mUpdateReqMutex 争用/持锁埋点：计数常开（单次约 60~90ns，可忽略），dump 由
 * UCM_PREFETCH_LOCK_PROFILE 环境变量门控。只做性能观测，不参与同步语义。 */
struct UReqLockStats {
    std::atomic<uint64_t> acquisitions{0};  // 获取次数
    std::atomic<uint64_t> contended{0};     // 等锁超过阈值的次数（真实争用）
    std::atomic<uint64_t> waitNs{0};        // 累计等锁时长（调用 lock -> 获得锁）
    std::atomic<uint64_t> holdNs{0};        // 累计持锁时长（获得锁 -> 析构释放）

    void Reset()
    {
        acquisitions.store(0, std::memory_order_relaxed);
        contended.store(0, std::memory_order_relaxed);
        waitNs.store(0, std::memory_order_relaxed);
        holdNs.store(0, std::memory_order_relaxed);
    }
};

// 判定"真实争用"的等锁阈值：无竞争获取约 30ns 量级，超过 1µs 必有其他持锁者。
constexpr uint64_t kUReqLockContendThresholdNs = 1000;

// std::lock_guard 的埋点等价物：wait 从调用 lock 前起算，hold 到析构释放。
class UReqLockGuard {
public:
    UReqLockGuard(std::mutex& mutex, UReqLockStats& stats) : mutex_(mutex), stats_(stats)
    {
        const auto waitStart = std::chrono::steady_clock::now();
        mutex_.lock();
        const auto acquired = std::chrono::steady_clock::now();
        const uint64_t waitNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(acquired - waitStart).count());
        stats_.acquisitions.fetch_add(1, std::memory_order_relaxed);
        stats_.waitNs.fetch_add(waitNs, std::memory_order_relaxed);
        if (waitNs > kUReqLockContendThresholdNs) {
            stats_.contended.fetch_add(1, std::memory_order_relaxed);
        }
        holdStart_ = acquired;
    }

    ~UReqLockGuard()
    {
        const auto holdEnd = std::chrono::steady_clock::now();
        const uint64_t holdNs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(holdEnd - holdStart_).count());
        stats_.holdNs.fetch_add(holdNs, std::memory_order_relaxed);
        mutex_.unlock();
    }

    UReqLockGuard(const UReqLockGuard&) = delete;
    UReqLockGuard& operator=(const UReqLockGuard&) = delete;

private:
    std::mutex& mutex_;
    UReqLockStats& stats_;
    std::chrono::steady_clock::time_point holdStart_{};
};

// mUpdateReqMutex 全部获取站点（埋点归属）。顺序与 DumpUReqLockStats 的
// 名称表一一对应，新增站点必须两处同步。
enum class UReqLockSite : size_t {
    WorkerCleanup = 0,        // MultiBSThreadFun 清理段（DelReqIDRun 冲刷）
    DelBlocksMap,             // Python del_blocks_map（延迟删除门控）
    SetBlocksMap,             // Python set_blocks_map
    SetLastIndexerBlock,      // Python set_last_indexer_block
    SetIndexerMap,            // Python set_indexer_map
    UpdateIndexerMap,         // Python update_indexer_map
    SubmitTaskIndexer,        // Python submit_task_indexer
    SetTopkIndices,           // Python set_topk_indices（Tier-0 后仅包 map 赋值）
    ResetReqTokenState,       // SetTokensMap 子调用
    ProcessTokensMapTopk,     // SetTokensMap 子调用
    RunLastChunkMoveGuarded,  // SetTokensMap 子调用
    IndexerInitLoad,          // Python indexer_init_load
    SubmitTask,               // Python submit_prefetch_task 提交段
    IndexerLoadProcess,       // indexer worker 表事务（IO 段出锁）
    IndexerLoadResolve,       // indexer worker 装载计划解析（表查找+任务组装）
    ProcessPrefetchRequest,   // prefetch worker（rank0，OMP 域内最小临界区）
    ReadPrefetchResults,      // prefetch worker（rank!=0）
    Count
};

struct LastKVBlockAddress {
    std::uintptr_t kAddr{0};
    std::uintptr_t vAddr{0};
    std::uintptr_t indexerAddr{0};
    int indexerBlockIdx{-1};
};

struct UCMMergeBlockContext {
    const std::vector<size_t>* tokenIndices;
    size_t tokenBase;
    bool isnoUCMblock;
    int blockIdx;
    const std::byte* kSrc;
    const std::byte* vSrc;
    bool owned;
};

/* UCM store cache-slot 布局（SharedIndexerKVCacheLayout）：[K][V][indexer][scale]
 * 四段。LoadOnToken filters 必须有 kStoreTensorCount 组 range，K 在
 * kKTensorIndex、V 在 kVTensorIndex（indexer/scale 组留空）。 */
constexpr size_t kKTensorIndex = 0;
constexpr size_t kVTensorIndex = 1;
constexpr size_t kIndexerTensorIndex = 2;
constexpr size_t kScaleTensorIndex = 3;
constexpr size_t kStoreTensorCount = 4;

/* LoadOnToken plan：K/V 各自独立的 shard 列表 + per-shard TokenFilter。每 shard
 * 携带一个目的窗口首地址；TokenRange{offset, length, dstOffset} 描述单 token：
 * offset 为源块内偏移，length 为单 token 字节数，dstOffset 相对 shard 首地址。 */
struct UCMTokenLoadPlan {
    UC::Detail::TaskDesc kTask;
    std::vector<UC::Detail::TokenFilter> kFilters;
    UC::Detail::TaskDesc vTask;
    std::vector<UC::Detail::TokenFilter> vFilters;
};

struct UCMMergeTimeStatus {
    uint64_t preLoadUs = 0;
    uint64_t loadToHostUs = 0;
    uint64_t preMemcpyUs = 0;
    uint64_t memcpyUs = 0;
};

struct PrefetchTopkView {
    const int* base = nullptr;
    int64_t stride0 = 0;
    int64_t stride1 = 0;
    int64_t numel = 0;
};

inline int64_t PrefetchTopkReadableNumel(const torch::Tensor& t)
{
    if (!t.defined()) { return 0; }
    const int64_t elemBytes = static_cast<int64_t>(t.element_size());
    if (elemBytes <= 0) { return 0; }
    const int64_t storageElems = 
        static_cast<int64_t>(t.storage().nbytes()) / elemBytes;
    return storageElems - static_cast<int64_t>(t.storage_offset());
}

/* UCMTensorMergeBatch 的整批请求态：contexts 持有指向 tokenMap 节点的
 * 指针，须与 tokenMap 同生命周期（std::map 节点地址稳定，move 安全）。 */
struct UCMMergeRequestState {
    std::string reqID;
    std::map<int, std::vector<size_t>> tokenMap;
    std::vector<UCMMergeBlockContext> contexts;
    size_t previousLoadNum = 0;
};

struct LastChunkSourceBlock {
    const std::byte* kBase{nullptr};
    const std::byte* vBase{nullptr};
};

using PrefetchTimePoint = std::chrono::high_resolution_clock::time_point;

struct PrefetchShmLayout {
    uint32_t versionSize;
    uint32_t countSize;
    uint32_t maxLoadNum;
    size_t missTokenSize;
    size_t loadSlotSize;
    size_t totalSize;
};

struct PrefetchTiming {
    uint64_t cpuMove = 0;
    uint64_t loadToHostUs = 0;
    uint64_t preMemcpyUs = 0;
    uint64_t memcpyUs = 0;
    uint64_t sync = 0;
    uint64_t h2d = 0;
    uint64_t ops = 0;
};

/* 非 rank0 等待 rank0 发布 prefetch 结果的超时：默认 30s，环境变量
 * UCM_PREFETCH_SYNC_TIMEOUT_MS 可覆盖（毫秒）。原 1s 在大 batch 下易误超时。 */
constexpr uint64_t kPrefetchSyncDefaultTimeoutMs = 30000;

/* 解析上述超时（微秒）：环境变量缺失/为空/非法（<=0 或解析失败）时回落默认，
 * 非法值向 stderr 打 WARN。逐次解析不缓存（每步一次，代价可忽略），保证
 * 测试与运行期改值即时生效。 */
uint64_t ResolvePrefetchSyncTimeoutUs();

struct TokenCopyTask {
    const std::byte* kSrc;
    const std::byte* vSrc;
    size_t dstOffset;
    uint64_t hbmSlot;
    /* True when this rank must memcpy the token locally: the block is owned
     * by this rank and is NOT a UCM block (UCM tokens are placed directly by
     * the store's LoadOnToken). */
    bool localCopy;
};

}  // namespace ucmprefetch

#endif
