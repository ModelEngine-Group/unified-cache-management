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
#include <string>
#include <unordered_map>
#include <vector>
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

/* UCM store cache-slot tensor layout expected by the GSA deployment
 * (SharedIndexerKVCacheLayout, li_c8 mode): the layer block cache slot is
 * laid out as [full K cache][full V cache][full indexer cache][full scale
 * cache], matching the indexer load task whose shards carry
 * {nullptr, nullptr, indexerAddr, nullptr}. LoadOnToken filters must
 * therefore have exactly kStoreTensorCount range groups, with K ranges under
 * kKTensorIndex and V ranges under kVTensorIndex (indexer/scale groups left
 * empty). */
constexpr size_t kKTensorIndex = 0;
constexpr size_t kVTensorIndex = 1;
constexpr size_t kIndexerTensorIndex = 2;
constexpr size_t kScaleTensorIndex = 3;
constexpr size_t kStoreTensorCount = 4;

/* LoadOnToken plan. K and V are staged into two separate destination buffers
 * (mLastCHunkKPtr/VPtr, mSharedKBufferHostPtr/VBufferHostPtr), so each tensor
 * gets its own shard list plus a per-shard TokenFilter. Every shard carries
 * exactly one host address: the base address of that shard's destination
 * window, i.e. dst base + firstDstPos * tokenBytes. Every TokenRange
 * {offset, length, dstOffset} describes one token: offset is the token's
 * offset inside the source block's tensor, length is the per-token K/V byte
 * size (mKVSizeBytesToken / mKVSizeBytesTokenRe) and dstOffset is relative
 * to the shard's destination base address (the store copies to
 * shard.addrs.front() + dstOffset). */
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
    uint64_t memcpy = 0;
    uint64_t sync = 0;
    uint64_t h2d = 0;
    uint64_t ops = 0;
};

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
