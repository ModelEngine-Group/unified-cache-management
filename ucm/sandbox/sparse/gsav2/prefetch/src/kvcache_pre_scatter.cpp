/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <kvcache_log.h>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "kvcache_pre.h"
#include "kvcache_pre_detail.h"
#include "securec.h"

#define H2D_INFO_SIZE 6

namespace ucmprefetch {

void GSAPrefetchEngineC::KVCacheInplaceScatter(std::string type, int layerID, aclrtStream stream,
                                               aclrtContext context)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start KVCacheInplaceScatter "
                "type: %s, load num: %d \n",
                mDecodeStep, type.c_str(), mLoadNum[layerID]);

    int lastDim = 0;
    int streamIdx = layerID % mStreamNum;
    torch::Tensor hbmBufferND;
    aclTensor* dstTensor;

    if (type == "K") {
        lastDim = mKDim;
        hbmBufferND = mKBufferDevice[streamIdx].slice(0, 0, mLoadNum[layerID]);
        dstTensor = mKCacheAcl[layerID];
    } else {
        lastDim = mVDim;
        hbmBufferND = mVBufferDevice[streamIdx].slice(0, 0, mLoadNum[layerID]);
        dstTensor = mVCacheAcl[layerID];
    }

    auto ret = aclrtSetCurrentContext(context);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s \n", ret,
                    errMsg ? errMsg : "Unknown error");
    }

    torch::Tensor dstIndexND = mLoadIndexDevice[streamIdx]
                                   .slice(0, 0, mLoadNum[layerID])
                                   .view({-1, 1})
                                   .expand({mLoadNum[layerID], lastDim});
    aclTensor* dstIndexTensor;
    aclTensor* dstHBMBufferTensor;
    PrepareScatterAclTensor(type, streamIdx, dstIndexND, hbmBufferND, dstIndexTensor,
                            dstHBMBufferTensor);

    RunInplaceScatter(type, layerID, streamIdx, dstTensor, dstIndexTensor, dstHBMBufferTensor,
                      stream);

    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End KVCacheInplaceScatter "
                "type: %s \n",
                mDecodeStep, type.c_str());
}

void GSAPrefetchEngineC::PrepareScatterAclTensor(const std::string& type, int streamIdx,
                                                 torch::Tensor& dstIndexND,
                                                 torch::Tensor& hbmBufferND,
                                                 aclTensor*& dstIndexTensor,
                                                 aclTensor*& dstHBMBufferTensor)
{
    if (type == "K") {
        if (mKDstIndexAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mKDstIndexAcl[streamIdx]);
            mKDstIndexAcl[streamIdx] = nullptr;
        }
        if (mKHBMBufferAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mKHBMBufferAcl[streamIdx]);
            mKHBMBufferAcl[streamIdx] = nullptr;
        }

        mKDstIndexAcl[streamIdx] = AtTensorToAclTensorShared(dstIndexND);
        mKHBMBufferAcl[streamIdx] = AtTensorToAclTensorShared(hbmBufferND);
        dstIndexTensor = mKDstIndexAcl[streamIdx];
        dstHBMBufferTensor = mKHBMBufferAcl[streamIdx];
    } else {
        if (mVDstIndexAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mVDstIndexAcl[streamIdx]);
            mVDstIndexAcl[streamIdx] = nullptr;
        }
        if (mVHBMBufferAcl[streamIdx] != nullptr) {
            ReleaseAclTensor(mVHBMBufferAcl[streamIdx]);
            mVHBMBufferAcl[streamIdx] = nullptr;
        }

        mVDstIndexAcl[streamIdx] = AtTensorToAclTensorShared(dstIndexND);
        mVHBMBufferAcl[streamIdx] = AtTensorToAclTensorShared(hbmBufferND);
        dstIndexTensor = mVDstIndexAcl[streamIdx];
        dstHBMBufferTensor = mVHBMBufferAcl[streamIdx];
    }
}

void* GSAPrefetchEngineC::GetScatterWorkspace(const std::string& type, int layerID, int streamIdx,
                                              uint64_t workspaceSize)
{
    uint64_t& maxWorkspaceSize =
        type == "K" ? mKMaxWorkspaceSize[streamIdx] : mVMaxWorkspaceSize[streamIdx];

    void*& workspaceAddr = type == "K" ? mKWorkspaceAddr[streamIdx] : mVWorkspaceAddr[streamIdx];

    if (workspaceSize > maxWorkspaceSize) {
        if (maxWorkspaceSize > 0) {
            auto ret = aclrtFree(workspaceAddr);
            if (ret != ACL_SUCCESS) {
                const char* errMsg = aclGetRecentErrMsg();
                mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtFree Error: %d, Error: %s\n",
                            ret, errMsg ? errMsg : "Unknown error");
            }
        }

        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtMalloc Error: %d, Erroe: %s \n",
                        ret, errMsg ? errMsg : "Unknown error");
            KVPackAbort(mKVPackControl);
        } else {
            mLogger.log(LogLevel::DEBUG,
                        "|KVCache Prefetch| aclrtMalloc success for layer: %d, "
                        "stype: %s\n",
                        layerID, type.c_str());
        }

        maxWorkspaceSize = workspaceSize;
    }

    return workspaceAddr;
}

void GSAPrefetchEngineC::RunInplaceScatter(const std::string& type, int layerID, int streamIdx,
                                           aclTensor* dstTensor, aclTensor* dstIndexTensor,
                                           aclTensor* dstHBMBufferTensor, aclrtStream stream)
{
    int64_t dim = 0;
    int64_t reduce = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto ret = aclnnInplaceScatterGetWorkspaceSize(
        dstTensor, dim, dstIndexTensor, dstHBMBufferTensor, reduce, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| InplaceScatterGetWorkspaceSize "
                    "Error: %d, Error: %s\n",
                    ret, errMsg ? errMsg : "Unknown error");
        KVPackAbort(mKVPackControl);
    }

    void* workspaceAddr = GetScatterWorkspace(type, layerID, streamIdx, workspaceSize);

    ret = aclnnInplaceScatter(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch|  aclnnGather Kcache Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
        KVPackAbort(mKVPackControl);
    }
}

int GSAPrefetchEngineC::CallPrefetchProcess(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    std::lock_guard<std::mutex> lock(mMutexUseToken);
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    int ret = 0;

    // 判别式：预取结果准备失败（已打日志、结果清零、版本照常递增）时记录返回码，
    // 后续层循环仍需完成以维持跨 rank 协议完整（空结果天然走零加载路径）。
    if (PreparePrefetchResults(taskInfo) != 0) { ret = -1; }

    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| "
                "all layer average hit rate: %f for bs: %d\n",
                mDecodeStep, average2d(mAvgHitRate), oneRunBsLen);
    for (size_t i = 0; i < mAvgHitRate.size(); ++i) { mAvgHitRate[i].clear(); }
    auto managerEnd = std::chrono::high_resolution_clock::now();
    auto managerDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(managerEnd - start);
    PrefetchTiming timing;

    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        MergeAndCopyPrefetchLayer(taskInfo, threadInfo, layerID, timing);
    }
    // 收尾同步失败打日志（Synchronize 内部已记录）并上抛返回码，进程不终止。
    if (Synchronize(threadInfo.context_, threadInfo.stream_, mStreamNum, mLogger) != 0) {
        ret = -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish async pretch, "
                "Synchronize cost: %lu, manager time:%lu, "
                "cpu merge time: %lu, memcpy time: %lu, "
                "sync time: %lu, h2d time: %lu, ops time: %lu\n",
                mDecodeStep, duration.count(), managerDuration.count(), timing.cpuMove,
                timing.memcpyUs, timing.sync, timing.h2d, timing.ops);
    return ret;
}

int GSAPrefetchEngineC::PreparePrefetchResults(const TaskInfo& taskInfo)
{
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    PrefetchShmLayout layout{};
    layout.versionSize = 4;
    layout.countSize = 4;
    layout.maxLoadNum = mMaxLoadNum;
    layout.missTokenSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.loadSlotSize = layout.countSize + layout.maxLoadNum * sizeof(int);
    layout.totalSize = layout.missTokenSize + layout.loadSlotSize + mSpaTopkLen * sizeof(int);
    uint32_t* versionPtr = static_cast<uint32_t*>(mPrefetchResultShm);
    if (mRank == 0) {
        return PublishPrefetchResultsRank0(taskInfo, oneRunBsLen, layout, versionPtr);
    }
    return SyncPrefetchResultsFromRank0(taskInfo, layout, versionPtr);
}

int GSAPrefetchEngineC::PublishPrefetchResultsRank0(const TaskInfo& taskInfo, uint32_t oneRunBsLen,
                                                    const PrefetchShmLayout& layout,
                                                    uint32_t* versionPtr)
{
    /* 提前识别：进入并行域前完成张量合法性判别并一次性提取只读视图，
     * OMP 域内不再触碰任何 torch API（消除 data_ptr dtype 抛异常、下标
     * 越界抛 IndexError 与张量切片堆分配三类异常/终止源）。 */
    PrefetchTopkView topkView;
    if (AcquirePrefetchTopkBase(taskInfo, oneRunBsLen, topkView) != 0) {
        ZeroFillAllPrefetchResults(oneRunBsLen, layout, versionPtr);
        return -1;
    }
    int failedCount = 0;
#pragma omp parallel for num_threads(mThreadNum) proc_bind(master) reduction(+ : failedCount)
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        /* OMP 线程独占 scratch：oneTopkList 同尺寸 assign 覆写、useBlockTable
         * clear 保留 capacity，将 per-(req,layer) 的两笔 ~20KB 堆分配收敛为
         * 每 layerID 迭代一次。 */
        std::vector<int> oneTopkListScratch;
        std::vector<int> useBlockTableScratch;
        for (size_t i = 0; i < oneRunBsLen; i++) {
            if (ProcessPrefetchRequest(taskInfo, topkView, i, layerID, layout, oneTopkListScratch,
                                       useBlockTableScratch) != 0) {
                failedCount += 1;
            }
        }
    }
    __atomic_store_n(versionPtr, mPrefetchResultVersion.fetch_add(1) + 1, __ATOMIC_RELEASE);
    mLogger.log(LogLevel::INFO, "Rank %d synced upload version: %u, %u\n", mRank,
                mPrefetchResultVersion.load(), __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE));
    if (failedCount > 0) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| %d (req,layer) results skipped "
                    "this step\n",
                    mDecodeStep, failedCount);
        return -1;
    }
    return 0;
}

int GSAPrefetchEngineC::AcquirePrefetchTopkBase(const TaskInfo& taskInfo, uint32_t oneRunBsLen,
                                                PrefetchTopkView& topkView)
{
    const torch::Tensor& topkList = taskInfo.topkList;
    const bool topkDefined = topkList.defined();
    const int64_t topkDim = topkDefined ? topkList.dim() : 0;
    /* mMaxBs 同时界定 shm 结果区容量（mPrefetchResultShmSize），
     * 越界批次必须在写 shm 前拒绝。 */
    const bool topkOk =
        topkDefined && topkDim >= 2 &&
        static_cast<int64_t>(topkList.size(0)) >= static_cast<int64_t>(mTopkIndexerNum) &&
        static_cast<int64_t>(topkList.size(1)) >= static_cast<int64_t>(oneRunBsLen) &&
        taskInfo.topkLenList.size() >= static_cast<size_t>(oneRunBsLen) &&
        static_cast<uint64_t>(oneRunBsLen) <= static_cast<uint64_t>(mMaxBs) &&
        topkList.scalar_type() == torch::kInt;
    if (!topkOk) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| invalid topkList tensor: "
                    "defined=%d dim=%lld size0=%lld size1=%lld dtype=%d bsLen=%u maxBs=%zu "
                    "topkIndexerNum=%d\n",
                    mDecodeStep, topkDefined ? 1 : 0, static_cast<long long>(topkDim),
                    topkDim >= 1 ? static_cast<long long>(topkList.size(0)) : -1LL,
                    topkDim >= 2 ? static_cast<long long>(topkList.size(1)) : -1LL,
                    topkDefined ? static_cast<int>(topkList.scalar_type()) : -1, oneRunBsLen,
                    mMaxBs, mTopkIndexerNum);
        return -1;
    }
    topkView.base = topkList.data_ptr<int>();
    topkView.stride0 = topkList.stride(0);
    topkView.stride1 = topkList.stride(1);
    topkView.numel = PrefetchTopkReadableNumel(topkList);
    return 0;
}

void GSAPrefetchEngineC::ZeroFillAllPrefetchResults(uint32_t oneRunBsLen,
                                                    const PrefetchShmLayout& layout,
                                                    uint32_t* versionPtr)
{
    std::cerr << "|KVCache Prefetch| [ERROR] PreparePrefetchResults: invalid topkList, "
                 "results zero-filled for this step\n";
    if (mPrefetchResultShm == nullptr) { return; }
    /* 结果区整体清零 = 每请求零 miss，消费侧按空结果降级；
     * 版本照常递增，保住 rank!=0 的等待协议（否则对端超时 abort）。 */
    std::memset(
        static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize, 0,
        static_cast<size_t>(oneRunBsLen) * static_cast<size_t>(mTopkIndexerNum) * layout.totalSize);
    __atomic_store_n(versionPtr, mPrefetchResultVersion.fetch_add(1) + 1, __ATOMIC_RELEASE);
}

uint64_t ResolvePrefetchSyncTimeoutUs()
{
    const char* envVal = std::getenv("UCM_PREFETCH_SYNC_TIMEOUT_MS");
    if (envVal == nullptr || envVal[0] == '\0') { return kPrefetchSyncDefaultTimeoutMs * 1000ULL; }
    char* parseEnd = nullptr;
    const long parsedMs = std::strtol(envVal, &parseEnd, 10);
    if (parseEnd == envVal || parsedMs <= 0) {
        std::cerr << "|KVCache Prefetch| [WARN] invalid UCM_PREFETCH_SYNC_TIMEOUT_MS='" << envVal
                  << "', fallback to " << kPrefetchSyncDefaultTimeoutMs << "ms\n";
        return kPrefetchSyncDefaultTimeoutMs * 1000ULL;
    }
    return static_cast<uint64_t>(parsedMs) * 1000ULL;
}

int GSAPrefetchEngineC::SyncPrefetchResultsFromRank0(const TaskInfo& taskInfo,
                                                     const PrefetchShmLayout& layout,
                                                     uint32_t* versionPtr)
{
    /* L-9：超时默认 30s（UCM_PREFETCH_SYNC_TIMEOUT_MS 可覆盖，100us 粒度轮询），
     * 超时判定同时复核版本号，消除"恰在边界发布成功仍被判超时"的误杀竞态；
     * fatal 对端提前置错时立即退出，不等超时。 */
    const uint64_t timeoutUs = ResolvePrefetchSyncTimeoutUs();
    uint32_t expectedVersion = mPrefetchResultVersion.load() + 1;
    uint64_t waitedUs = 0;
    while (__atomic_load_n(versionPtr, __ATOMIC_ACQUIRE) != expectedVersion &&
           waitedUs < timeoutUs) {
        if (mKVPackControl->fatalError.load(std::memory_order_acquire) != 0) {
            KVPackAbort(mKVPackControl);
        }
        usleep(100);
        waitedUs += 100;
    }
    if (waitedUs >= timeoutUs && __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE) != expectedVersion) {
        mLogger.log(LogLevel::ERROR,
                    "Rank %d timeout(%llu ms) waiting for prefetch result, version: %u, "
                    "expected: %u\n",
                    mRank, static_cast<unsigned long long>(timeoutUs / 1000),
                    __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE), expectedVersion);
        KVPackAbort(mKVPackControl);
    }
    ReadPrefetchResults(taskInfo, layout);
    mPrefetchResultVersion.store(expectedVersion);
    mLogger.log(LogLevel::INFO, "Rank %d synced prefetch results, version: %u\n", mRank,
                __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE));
    return 0;
}

int GSAPrefetchEngineC::ProcessPrefetchRequest(const TaskInfo& taskInfo,
                                               const PrefetchTopkView& topkView, size_t i,
                                               int layerID, const PrefetchShmLayout& layout,
                                               std::vector<int>& oneTopkListScratch,
                                               std::vector<int>& useBlockTableScratch)
{
    const std::string& oneReqID = taskInfo.reqIdList[i];
    const int* topkPtr = nullptr;
    int oneTopkLen = 0;
    const int state =
        AcquirePrefetchRequestState(taskInfo, topkView, i, layerID, layout, topkPtr, oneTopkLen);
    if (state == 0) { return 0; }
    if (state < 0) { return -1; }
    // miss/loadSlot 取 map 槽位引用（RunHbmTopkLookup 就地演化，供后续
    // MergeAndCopyPrefetchLayer 消费）；find 判别杜绝插空+越界 UB。
    auto missIt = mMissGlobalToken.find(oneReqID);
    auto loadIt = mLoadSlot.find(oneReqID);
    if (missIt == mMissGlobalToken.end() || loadIt == mLoadSlot.end() ||
        static_cast<size_t>(layerID) >= missIt->second.size() ||
        static_cast<size_t>(layerID) >= loadIt->second.size()) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| miss/loadSlot state missing: "
                    "req_id: %s, layerID: %d, rank: %d, i: %zu\n",
                    mDecodeStep, oneReqID.c_str(), layerID, mRank, i);
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    std::vector<int>& missGlobalIds = missIt->second[static_cast<size_t>(layerID)];
    std::vector<int>& loadSlots = loadIt->second[static_cast<size_t>(layerID)];
    oneTopkListScratch.assign(topkPtr, topkPtr + oneTopkLen);
    useBlockTableScratch.clear();
    if (!RunHbmTopkLookup(oneReqID, oneTopkListScratch, layerID, static_cast<int>(i), oneTopkLen,
                          useBlockTableScratch, missGlobalIds, loadSlots)) {
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    return FinishPrefetchRequest(i, layerID, layout, oneReqID, useBlockTableScratch, missGlobalIds,
                                 loadSlots);
}

// 尾段：尺寸校验 + mAllTopkIndicesTensor 拷贝（最小锁临界区）+ shm 写结果。
// 返回 0=成功，-1=失败（尺寸不匹配/拷贝失败均已清槽）。
int GSAPrefetchEngineC::FinishPrefetchRequest(
    size_t i, int layerID, const PrefetchShmLayout& layout, const std::string& oneReqID,
    std::vector<int>& useBlockTable, std::vector<int>& missGlobalIds, std::vector<int>& loadSlots)
{
    // useBlockTable 尺寸与 mSpaTopkLen 不一致属可恢复场景（短/年轻请求 sparse
    // 窗口不足），清槽跳过、消费侧按零 miss 降级。
    if (static_cast<uint32_t>(useBlockTable.size()) != mSpaTopkLen) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| useBlockTable size mismatch, skip: "
                    "req_id: %s, layerID: %d, mSpaTopkLen: %d, useBlockTableSize: %zu\n",
                    mDecodeStep, oneReqID.c_str(), layerID, mSpaTopkLen, useBlockTable.size());
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    // mAllTopkIndicesTensor 读侧与写侧统一 mUpdateReqMutex 协议，锁缩到
    // find+memcpy 最小临界区（本调用点位于 OMP 并行域）。
    int copyRet = 0;
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::ProcessPrefetchRequest));
        copyRet = CopyUseBlockTableToIndicesTensor(oneReqID, layerID, useBlockTable);
    }
    if (copyRet != 0) {
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    WritePrefetchResult(i, layerID, useBlockTable, layout, missGlobalIds, loadSlots);
    return 0;
}

/* 校验单请求查找输入：topkLen / 平坦读边界。返回 1=就绪（topkPtr/oneTopkLen
 * 填充），0=topkLen 空（软跳过，已清槽），-1=非法（已日志+清槽）。 */
int GSAPrefetchEngineC::AcquirePrefetchRequestState(const TaskInfo& taskInfo,
                                                    const PrefetchTopkView& topkView, size_t i,
                                                    int layerID, const PrefetchShmLayout& layout,
                                                    const int*& topkPtr, int& oneTopkLen)
{
    const std::string& oneReqID = taskInfo.reqIdList[i];
    oneTopkLen = taskInfo.topkLenList[i];
    if (oneTopkLen <= 0) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| topk len is zero: "
                    "%d for req_id: %s, rank: %d, taskID: %d, i: %zu\n",
                    mDecodeStep, oneTopkLen, oneReqID.c_str(), mRank, taskInfo.taskID, i);
        // 清零槽位避免消费侧读到上一 step 的陈旧结果（零 miss = 本步无加载）。
        ClearPrefetchResultSlot(i, layerID, layout);
        return 0;
    }
    // 平坦读边界：topkLen 以外的长度越界一律拒绝（防 OOB 读与脏数据扩散）。
    const int64_t topkOffset = static_cast<int64_t>(layerID) * topkView.stride0 +
                               static_cast<int64_t>(i) * topkView.stride1;
    if (topkView.base == nullptr ||
        topkOffset + static_cast<int64_t>(oneTopkLen) > topkView.numel) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| topk view out of range: "
                    "req_id: %s, layerID: %d, i: %zu, offset: %lld, topkLen: %d, numel: %lld\n",
                    mDecodeStep, oneReqID.c_str(), layerID, i, static_cast<long long>(topkOffset),
                    oneTopkLen, static_cast<long long>(topkView.numel));
        ClearPrefetchResultSlot(i, layerID, layout);
        return -1;
    }
    topkPtr = topkView.base + topkOffset;
    return 1;
}

HBMManagerLayered* GSAPrefetchEngineC::FindReqLayerManager(const std::string& reqID, int layerID)
{
    /* find 判别：杜绝 map operator[] 在 OMP 失败路径上的并发插入竞态；
     * managers 与 miss/loadSlot 容器在 SetTokensMap 中同步扩容，命中即全部就绪。 */
    auto managersIt = mHBMMangers.find(reqID);
    if (managersIt == mHBMMangers.end() ||
        static_cast<size_t>(layerID) >= managersIt->second.size() ||
        !managersIt->second[static_cast<size_t>(layerID)]) {
        return nullptr;
    }
    return managersIt->second[static_cast<size_t>(layerID)].get();
}

bool GSAPrefetchEngineC::RunHbmTopkLookup(const std::string& oneReqID,
                                          std::vector<int>& oneTopkList, int layerID, int bsIndex,
                                          int oneTopkLen, std::vector<int>& useBlockTable,
                                          std::vector<int>& missGlobalIds,
                                          std::vector<int>& loadSlots)
{
    HBMManagerLayered* manager = FindReqLayerManager(oneReqID, layerID);
    if (manager == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| HBM manager is not initialized: "
                    "req_id: %s, layerID: %d, rank: %d, bsIndex: %d\n",
                    mDecodeStep, oneReqID.c_str(), layerID, mRank, bsIndex);
        return false;
    }
    /* missGlobalIds/loadSlots 由调用方检索好的槽位引用直通，本函数不再访问
     * mMissGlobalToken/mLoadSlot 两张 map。 */
    missGlobalIds.clear();
    loadSlots.clear();
    if (mOpenIndexerOffload && static_cast<size_t>(bsIndex) < mIsIndexerOffload.size() &&
        mIsIndexerOffload[static_cast<size_t>(bsIndex)]) {
        TokenIndexMapInter(oneTopkList, bsIndex, layerID, oneTopkLen, oneReqID);
    }
    manager->processTopkWithDebug(oneTopkList, useBlockTable, missGlobalIds, loadSlots, mLogger);
    if (missGlobalIds.size() != loadSlots.size() ||
        missGlobalIds.size() > static_cast<size_t>(mMaxLoadNum)) {
        KVPackAbort(mKVPackControl);
    }
    mAvgHitRate[layerID].emplace_back(manager->mHitRate);
    /* average() 是 O(步数) 线性聚合，DEBUG 关闭时白算；enabledFor 门控后
     * emplace 刚完成，mAvgHitRate[layerID] 必非空，可直调。 */
    if (mLogger.enabledFor(LogLevel::DEBUG)) {
        mLogger.log(LogLevel::DEBUG,
                    "Decode step: %u, |KVCache Prefetch| expand topk len: "
                    "%lu and hit rate %f and average hit rate: %f, for layer: %d req_id: %s,  \n",
                    mDecodeStep, oneTopkList.size(), manager->mHitRate,
                    average(mAvgHitRate[layerID]), layerID, oneReqID.c_str());
    }
    return true;
}

void GSAPrefetchEngineC::WritePrefetchResult(size_t i, int layerID,
                                             const std::vector<int>& useBlockTable,
                                             const PrefetchShmLayout& layout,
                                             const std::vector<int>& missGlobalIds,
                                             const std::vector<int>& loadSlots)
{
    size_t reqLayerOffset = (i * mTopkIndexerNum + layerID) * layout.totalSize;
    uint8_t* shmBase =
        static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize + reqLayerOffset;
    uint32_t missTokenCount =
        std::min(missGlobalIds.size(), static_cast<size_t>(layout.maxLoadNum));
    std::memcpy(shmBase, &missTokenCount, layout.countSize);

    if (missTokenCount > 0) {
        std::memcpy(shmBase + layout.countSize, missGlobalIds.data(), missTokenCount * sizeof(int));
    }

    uint32_t loadSlotCount = std::min(loadSlots.size(), static_cast<size_t>(layout.maxLoadNum));
    std::memcpy(shmBase + layout.missTokenSize, &loadSlotCount, layout.countSize);
    if (loadSlotCount > 0) {
        std::memcpy(shmBase + layout.missTokenSize + layout.countSize, loadSlots.data(),
                    loadSlotCount * sizeof(int));
    }
    std::memcpy(shmBase + layout.missTokenSize + layout.loadSlotSize, useBlockTable.data(),
                useBlockTable.size() * sizeof(int));
}

void GSAPrefetchEngineC::ClearPrefetchResultSlot(size_t i, int layerID,
                                                 const PrefetchShmLayout& layout)
{
    if (mPrefetchResultShm == nullptr) { return; }
    const size_t slotOffset = layout.versionSize + (i * static_cast<size_t>(mTopkIndexerNum) +
                                                    static_cast<size_t>(layerID)) *
                                                       layout.totalSize;
    std::memset(static_cast<uint8_t*>(mPrefetchResultShm) + slotOffset, 0, layout.totalSize);
}

// shm 槽内单段"定长计数 + 变长数据"读回（count 为 0 时仅清空）。
void ReadShmCountedSlot(const uint8_t* base, size_t countOffset, size_t dataOffset,
                        uint32_t countSize, std::vector<int>& out)
{
    uint32_t count = 0;
    memcpy(&count, base + countOffset, countSize);
    out.clear();
    if (count > 0) {
        out.resize(count);
        memcpy(out.data(), base + dataOffset, count * sizeof(int));
    }
}

void GSAPrefetchEngineC::ReadPrefetchResults(const TaskInfo& taskInfo,
                                             const PrefetchShmLayout& layout)
{
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    // 循环外一次性 scratch：resize 后逐槽 memcpy 全量覆写，免逐 (req,layer) 分配。
    std::vector<int> useBlockTable(static_cast<size_t>(mSpaTopkLen));
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        for (size_t i = 0; i < oneRunBsLen; i++) {
            if (taskInfo.topkLenList[i] <= 0) { continue; }
            ReadOneReqLayerResults(taskInfo, layout, i, layerID, useBlockTable);
        }
    }
}

// 单 (req,layer) 的 shm 槽位读回 + 拷贝到 mAllTopkIndicesTensor。
void GSAPrefetchEngineC::ReadOneReqLayerResults(const TaskInfo& taskInfo,
                                                const PrefetchShmLayout& layout, size_t i,
                                                int layerID, std::vector<int>& useBlockTable)
{
    const std::string& oneReqID = taskInfo.reqIdList[i];
    // find 判别取槽位引用（判别失败打日志跳过，消费侧按零加载降级）。
    auto missIt = mMissGlobalToken.find(oneReqID);
    auto loadIt = mLoadSlot.find(oneReqID);
    if (missIt == mMissGlobalToken.end() || loadIt == mLoadSlot.end() ||
        static_cast<size_t>(layerID) >= missIt->second.size() ||
        static_cast<size_t>(layerID) >= loadIt->second.size()) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| read: miss/loadSlot state "
                    "missing: req_id: %s, layerID: %d, rank: %d, i: %zu\n",
                    mDecodeStep, oneReqID.c_str(), layerID, mRank, i);
        return;
    }
    std::vector<int>& missGlobalIds = missIt->second[static_cast<size_t>(layerID)];
    std::vector<int>& loadSlots = loadIt->second[static_cast<size_t>(layerID)];
    size_t reqLayerOffset = (i * mTopkIndexerNum + layerID) * layout.totalSize;
    uint8_t* shmBase =
        static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize + reqLayerOffset;
    ReadShmCountedSlot(shmBase, 0, layout.countSize, layout.countSize, missGlobalIds);
    ReadShmCountedSlot(shmBase, layout.missTokenSize, layout.missTokenSize + layout.countSize,
                       layout.countSize, loadSlots);
    memcpy(useBlockTable.data(), shmBase + layout.missTokenSize + layout.loadSlotSize,
           mSpaTopkLen * sizeof(int));
    // 读侧统一 mUpdateReqMutex 协议（写侧 SetTopkIndices、erase 侧 DelReqIDRun
    // 均持该锁）；判别失败清空本 (req,layer) 的 miss/slot，按零加载降级。
    int copyRet = 0;
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::ReadPrefetchResults));
        copyRet = CopyUseBlockTableToIndicesTensor(oneReqID, layerID, useBlockTable);
    }
    if (copyRet != 0) {
        missGlobalIds.clear();
        loadSlots.clear();
    }
}

void GSAPrefetchEngineC::MergeAndCopyPrefetchLayer(const TaskInfo& taskInfo, ThreadInfo& threadInfo,
                                                   int layerID, PrefetchTiming& timing)
{
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    mLoadNum[layerID] = 0;
    int streamIdx = layerID % mStreamNum;
    const uint64_t epoch = (uint64_t(mPrefetchResultVersion.load()) << 32) | uint64_t(layerID + 1);
    if (BeginKVPackSlot(streamIdx, epoch) != 0) { KVPackAbort(mKVPackControl); }
    std::vector<uint64_t> hbmSlotIndex;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    /* Whole-batch merge：本层全部有效请求合成一个 LoadOnToken 任务（后端 IO
     * 重叠）；staging 容量按请求累计校验（ValidateUCMMergeRequest）。 */
    std::vector<uint32_t> requestIndices;
    requestIndices.reserve(oneRunBsLen);
    for (size_t i = 0; i < oneRunBsLen; i++) {
        if (taskInfo.topkLenList[i] > 0) { requestIndices.push_back(static_cast<uint32_t>(i)); }
    }
    UCMTensorMergeBatch(taskInfo, requestIndices, layerID, streamIdx, hbmSlotIndex);
    if (FinishKVPack(streamIdx, epoch) != 0) { KVPackAbort(mKVPackControl); }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    timing.cpuMove +=
        std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    CopyPrefetchToH2DBuffers(streamIdx, layerID);
    ValidateAndPublishHbmSlotIndex(streamIdx, layerID, hbmSlotIndex);
    auto memcpyEnd = std::chrono::high_resolution_clock::now();
    timing.memcpyUs +=
        std::chrono::duration_cast<std::chrono::microseconds>(memcpyEnd - cpuEnd).count();
    if (Synchronize(threadInfo.context_, static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx,
                    1, mLogger) != 0) {
        KVPackAbort(mKVPackControl);
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| H2D size for all request %d for layer: %d\n",
                mDecodeStep, mLoadNum[layerID], layerID);
    auto syncEnd = std::chrono::high_resolution_clock::now();
    timing.sync +=
        std::chrono::duration_cast<std::chrono::microseconds>(syncEnd - memcpyEnd).count();
    if (mLoadNum[layerID] == 0) {
        MarkKVH2DDone(streamIdx, epoch);
        return;
    }
    TransferAndScatterPrefetchLayer(threadInfo, layerID, streamIdx, timing, syncEnd);
    MarkKVH2DDone(streamIdx, epoch);
}

void GSAPrefetchEngineC::CopyPrefetchToH2DBuffers(int streamIdx, int layerID)
{
    if (mLoadNum[layerID] != 0) {
        std::memcpy(mH2DPtrInfo[streamIdx][0],
                    reinterpret_cast<const void*>(mSharedKBufferHostPtr[streamIdx]),
                    static_cast<size_t>(mLoadNum[layerID]) * mKVSizeBytesToken);
        if (!mISSPAC8) {
            std::memcpy(mH2DPtrInfo[streamIdx][1],
                        reinterpret_cast<const void*>(mSharedVBufferHostPtr[streamIdx]),
                        static_cast<size_t>(mLoadNum[layerID]) * mKVSizeBytesTokenRe);
        }
    }
}

void GSAPrefetchEngineC::ValidateAndPublishHbmSlotIndex(int streamIdx, int layerID,
                                                        std::vector<uint64_t>& hbmSlotIndex)
{
    uint32_t indexNum = hbmSlotIndex.size();
    if ((uint32_t)mLoadNum[layerID] != hbmSlotIndex.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] hbmSlotIndex is not equer to mLoadNum[layerID] !!! layer_id = %d, "
                    "hbmSlotIndex.size() = %d, mLoadNum[layerID] = %d\n",
                    layerID, hbmSlotIndex.size(), mLoadNum[layerID]);
        KVPackAbort(mKVPackControl);
    }
    if (hbmSlotIndex.size() > (uint32_t)mLoadIndexBufferLen) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] copy is greater than bound !!! layer_id %d, "
                    "hbmSlotIndex.size()  %d, mLoadIndexBufferLen %d\n",
                    layerID, hbmSlotIndex.size(), mLoadIndexBufferLen);
        KVPackAbort(mKVPackControl);
    }
    std::memcpy(mLoadIndexHost[streamIdx].data_ptr(), hbmSlotIndex.data(),
                indexNum * sizeof(int64_t));
}

void GSAPrefetchEngineC::TransferAndScatterPrefetchLayer(ThreadInfo& threadInfo, int layerID,
                                                         int streamIdx, PrefetchTiming& timing,
                                                         PrefetchTimePoint syncEnd)
{
    size_t kSizeBytes = mLoadNum[layerID] * mKVSizeBytesToken;
    size_t vSizeBytes = mLoadNum[layerID] * mKVSizeBytesTokenRe;
    size_t indexSizeBytes = mLoadNum[layerID] * sizeof(uint64_t);
    int32_t ret = CopyTrans(static_cast<void**>(&mH2DPtrInfo[streamIdx][0]),
                            static_cast<void**>(&mH2DPtrInfo[streamIdx][2]), kSizeBytes, 1, "h2d",
                            static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx);
    if (ret == -1) {
        mLogger.log(LogLevel::ERROR, "[ERROR] mKBufferDevice CopyTrans H2D ERROR !!!\n");
        KVPackAbort(mKVPackControl);
    }
    if (!mISSPAC8) {
        ret = CopyTrans(static_cast<void**>(&mH2DPtrInfo[streamIdx][1]),
                        static_cast<void**>(&mH2DPtrInfo[streamIdx][3]), vSizeBytes, 1, "h2d",
                        static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx);
        if (ret == -1) {
            mLogger.log(LogLevel::ERROR, "[ERROR] mVBufferDevice CopyTrans H2D ERROR !!!\n");
            KVPackAbort(mKVPackControl);
        }
    }
    ret = CopyTrans(static_cast<void**>(&mH2DPtrInfo[streamIdx][4]),
                    static_cast<void**>(&mH2DPtrInfo[streamIdx][5]), indexSizeBytes, 1, "h2d",
                    static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx);
    if (ret == -1) {
        mLogger.log(LogLevel::ERROR, "[ERROR] mLoadIndexDevice CopyTrans H2D ERROR !!!\n");
        KVPackAbort(mKVPackControl);
    }
    if (Synchronize(threadInfo.context_, static_cast<aclrtStream*>(threadInfo.stream_) + streamIdx,
                    1, mLogger) != 0) {
        KVPackAbort(mKVPackControl);
    }
    auto h2dEnd = std::chrono::high_resolution_clock::now();
    timing.h2d += std::chrono::duration_cast<std::chrono::microseconds>(h2dEnd - syncEnd).count();
    KVCacheInplaceScatter("K", layerID, static_cast<aclrtStream*>(threadInfo.stream_)[streamIdx],
                          *static_cast<aclrtContext*>(threadInfo.context_));
    if (!mISSPAC8) {
        KVCacheInplaceScatter("V", layerID,
                              static_cast<aclrtStream*>(threadInfo.stream_)[streamIdx],
                              *static_cast<aclrtContext*>(threadInfo.context_));
    }
    auto opsEnd = std::chrono::high_resolution_clock::now();
    timing.ops += std::chrono::duration_cast<std::chrono::microseconds>(opsEnd - h2dEnd).count();
}

int GSAPrefetchEngineC::IndexerLoadProcess(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    if (!mOpenIndexerOffload) { return 0; }
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t runBsLen = taskInfo.reqIdList.size();
    int ret = 0;
    /* 段1（表事务，持锁）：hit/miss 判定与六张 indexer 表演化；段2 的 ms 级 DDR IO
     * 出锁，共享面仅剩 mAllBlocksHash/mLastKVBlockAddress（mini 段保护）。
     * allNeedLoadBlock/allMissIdxs 为 worker 私有（erase 被 mIsIndexerLoadRunning
     * 门控，该位在本函数返回后才清零）。 */
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::IndexerLoadProcess));
        allNeedLoadBlock.clear();
        allMissIdxs.clear();
        for (size_t i = 0; i < runBsLen; i++) {
            if (mDocsTables.find(taskInfo.reqIdList[i]) == mDocsTables.end() ||
                taskInfo.topkLenList[i] <= 0) {
                mLogger.log(LogLevel::ERROR,
                            "Decode step: %u, |KVCache Prefetch| topk len is zero: %d\n",
                            mDecodeStep, taskInfo.topkLenList[i]);
                continue;
            }
            allMissIdxs.insert({taskInfo.reqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
            allNeedLoadBlock.insert(
                {taskInfo.reqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
            // 判别式：单请求 topk 张量非法（已打日志）时记失败返回码并跳过，其余请求继续。
            if (RunOneBsPrefetch(taskInfo.reqIdList[i], taskInfo.topkLenList[i], i,
                                 taskInfo.indexerTopkList[i]) != 0) {
                ret = -1;
                continue;
            }
        }
    }
    UCMIndexerLoadAsync(threadInfo.context_, threadInfo.stream_);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish indexer load cost: %lu\n", mDecodeStep,
                duration.count());
    return ret;
}

void GSAPrefetchEngineC::DealLessPrefetchBlock(std::unordered_set<int>& hitBlocks,
                                               std::map<int, int>& hitBlocksIdx, std::string reqID,
                                               int layerID, int topkLen)
{
    if (!mOpenIndexerOffload) { return; }
    for (auto it = mDocsTables[reqID][layerID].rbegin(); it != mDocsTables[reqID][layerID].rend();
         ++it) {
        if (hitBlocks.find(it->second) == hitBlocks.end() && hitBlocks.size() < (uint32_t)topkLen) {
            hitBlocks.insert(it->second);
            hitBlocksIdx.insert(std::make_pair(it->first, it->second));
        }
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| hitBlocks after deal len: %d\n", mDecodeStep,
                hitBlocks.size());
}

int GSAPrefetchEngineC::RunOneBsPrefetch(std::string reqID, int topkLen, int topkIndex,
                                         const torch::Tensor& topkList)
{
    if (!mOpenIndexerOffload) { return 0; }
    // 进入并行域前完成张量判别并提取只读视图，OMP 域内不触碰 torch API。
    PrefetchTopkView topkView;
    if (AcquireIndexerTopkBase(topkList, reqID, topkLen, topkView) != 0) { return -1; }
#pragma omp parallel for num_threads(4) proc_bind(master)
    for (int i = 0; i < mLayerNum; i++) {
        std::unordered_set<int> hitBlocks;
        std::map<int, int> hitBlocksIdx;
        std::vector<int> missIdxs;
        PrefetchReqInfo oneBsInfo;
        oneBsInfo.topkLen = topkLen;
        oneBsInfo.reqID = reqID;
        oneBsInfo.layerID = i;
        const int* topk_ptr = topkView.base + static_cast<int64_t>(i) * topkView.stride0;
        std::vector<int> oneTopkList(topk_ptr, topk_ptr + topkLen);
        GetHitAndMissBlock(oneBsInfo, oneTopkList, hitBlocks, hitBlocksIdx, missIdxs);
        if (missIdxs.size() != 0 && hitBlocksIdx.size() < (uint32_t)topkLen) {
            RunPrefetchH2D(oneBsInfo, hitBlocks, hitBlocksIdx, missIdxs);
        }
        if (hitBlocksIdx.size() != (uint32_t)topkLen) {
            mLogger.log(LogLevel::DEBUG,
                        "Decode step: %u, "
                        "|KVCache Prefetch| successIndex size is error!!! "
                        "mPrefetchIdx[reqID][layerID] size: %lu mDocsTables[reqID][i] size: %lu, "
                        "hitBlocksIdx size: %lu, "
                        "mBlocksMap[reqID][layerID]: %lu, "
                        "missIdxs size: %lu\n",
                        mDecodeStep, mPrefetchIdx[reqID][i].size(), mDocsTables[reqID][i].size(),
                        hitBlocksIdx.size(), mBlocksMap[reqID][i].size(), missIdxs.size());
            DealLessPrefetchBlock(hitBlocks, hitBlocksIdx, reqID, i, topkLen);
        }
        FinishOneLayerPrefetch(reqID, i, hitBlocksIdx, missIdxs);
    }
    return 0;
}

int GSAPrefetchEngineC::AcquireIndexerTopkBase(const torch::Tensor& topkList,
                                               const std::string& reqID, int topkLen,
                                               PrefetchTopkView& topkView)
{
    const bool topkDefined = topkList.defined();
    const int64_t topkDim = topkDefined ? topkList.dim() : 0;
    bool topkOk = topkDefined && topkDim >= 1 &&
                  static_cast<int64_t>(topkList.size(0)) >= static_cast<int64_t>(mLayerNum) &&
                  topkList.scalar_type() == torch::kInt;
    if (topkOk) {
        /* 平坦读边界：末层末行读取范围不得超过张量元素数（防 OOB 读）。 */
        const int64_t stride0 = topkList.stride(0);
        topkOk = topkLen >= 0 && static_cast<int64_t>(mLayerNum) - 1 >= 0 &&
                 (static_cast<int64_t>(mLayerNum) - 1) * stride0 + static_cast<int64_t>(topkLen) <=
                     PrefetchTopkReadableNumel(topkList);
    }
    if (!topkOk) {
        mLogger.log(
            LogLevel::ERROR,
            "Decode step: %u, |KVCache Prefetch| invalid indexer topkList tensor: "
            "reqID=%s defined=%d dim=%lld size0=%lld dtype=%d layerNum=%d topkLen=%d\n",
            mDecodeStep, reqID.c_str(), topkDefined ? 1 : 0, static_cast<long long>(topkDim),
            topkDim >= 1 ? static_cast<long long>(topkList.size(0)) : -1LL,
            topkDefined ? static_cast<int>(topkList.scalar_type()) : -1, mLayerNum, topkLen);
        std::cerr << "|KVCache Prefetch| [ERROR] RunOneBsPrefetch: invalid indexer topkList, "
                     "layer run skipped, reqID="
                  << reqID << std::endl;
        return -1;
    }
    topkView.base = topkList.data_ptr<int>();
    topkView.stride0 = topkList.stride(0);
    topkView.numel = topkList.numel();
    return 0;
}

void GSAPrefetchEngineC::FinishOneLayerPrefetch(const std::string& reqID, int layerID,
                                                std::map<int, int>& hitBlocksIdx,
                                                std::vector<int>& missIdxs)
{
    int successIndex = 0;
    std::vector<int> blockTableResult;
    std::vector<std::pair<int, int>> sortResult(hitBlocksIdx.size());
    for (auto it = hitBlocksIdx.begin(); it != hitBlocksIdx.end(); it++) {
        sortResult[successIndex] = std::make_pair(it->first, it->second);
        successIndex += 1;
    }
    std::sort(sortResult.begin(), sortResult.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    int32_t pairIdx = 0;
    for (auto onePair : sortResult) {
        blockTableResult.emplace_back(onePair.second);
        mPrefetchBlockTableIdx[reqID][layerID][pairIdx] = onePair.first;
        pairIdx += 1;
    }
    LoadKVToHBM(allNeedLoadBlock[reqID][layerID], missIdxs, layerID, reqID);
    std::memcpy(mIndexerLoadResultT[reqID][layerID].data_ptr(), blockTableResult.data(),
                successIndex * sizeof(int));
    mPrefetchIdx[reqID][layerID].clear();
    for (auto it = mDocsTables[reqID][layerID].begin(); it != mDocsTables[reqID][layerID].end();
         it++) {
        if (hitBlocksIdx.find(it->first) != hitBlocksIdx.end()) {
            continue;
        } else {
            mPrefetchIdx[reqID][layerID].push_back(it->first);
        }
    }
}

void GSAPrefetchEngineC::GetHitAndMissBlock(PrefetchReqInfo oneBsInfo, std::vector<int>& topkList,
                                            std::unordered_set<int>& hitBlocks,
                                            std::map<int, int>& hitBlocksIdx,
                                            std::vector<int>& missIdxs)
{
    if (!mOpenIndexerOffload) { return; }
    int topkLen = oneBsInfo.topkLen;
    int layerID = oneBsInfo.layerID;
    std::string reqID = oneBsInfo.reqID;

    for (int j = 0; j < topkLen; j++) {
        int32_t item = topkList[j];
        if (mDocsTables[reqID][layerID].find(item) != mDocsTables[reqID][layerID].end()) {
            int blockID = mDocsTables[reqID][layerID][item];
            hitBlocks.insert(blockID);
            hitBlocksIdx.insert(std::make_pair(item, blockID));
        } else {
            missIdxs.push_back(item);
        }
    }
    if ((hitBlocks.size() + missIdxs.size()) != (uint32_t)topkLen) {
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| Decode step: %u, Rank: %d, reqID: %s, layer: %d, hit size: "
                    "%lu, miss size: %lu , topkLen: %d, not equal error\n",
                    mDecodeStep, mRank, reqID.c_str(), layerID, hitBlocks.size(), missIdxs.size(),
                    topkLen);
    }
}

void GSAPrefetchEngineC::RunPrefetchH2D(PrefetchReqInfo oneBsInfo,
                                        std::unordered_set<int>& hitBlocks,
                                        std::map<int, int>& hitBlocksIdx,
                                        std::vector<int>& missIdxs)
{
    if (!mOpenIndexerOffload) { return; }
    int layerID = oneBsInfo.layerID;
    std::string reqID = oneBsInfo.reqID;

    int oneFreeBlockLen = mPrefetchIdx[reqID][layerID].size();
    std::vector<int> oneFreeBlockTable;

    uint32_t index = 0;
    std::vector<int> onePrefetchIdx = mPrefetchIdx[reqID][layerID];
    int oneFreeBlockIndex = 0;
    while (oneFreeBlockIndex < oneFreeBlockLen && index < missIdxs.size()) {
        int oneFreeBlockID = mDocsTables[reqID][layerID][onePrefetchIdx[oneFreeBlockIndex]];
        if (hitBlocks.find(oneFreeBlockID) != hitBlocks.end()) {
            oneFreeBlockIndex += 1;
            continue;
        } else {
            oneFreeBlockTable.push_back(oneFreeBlockID);
            hitBlocks.insert(oneFreeBlockID);
            hitBlocksIdx.insert(std::make_pair(missIdxs[index], oneFreeBlockID));
            index += 1;
            oneFreeBlockIndex += 1;
        }
    }
    uint32_t loadLen = oneFreeBlockTable.size();
    missIdxs.erase(missIdxs.begin() + loadLen, missIdxs.end());
    allNeedLoadBlock[reqID][layerID] = oneFreeBlockTable;
    allMissIdxs[reqID][layerID] = missIdxs;
}

void GSAPrefetchEngineC::LoadKVToHBM(std::vector<int> loadNPUBlockIDs, std::vector<int> missIdxs,
                                     int layerID, std::string reqID)
{
    for (size_t i = 0; i < loadNPUBlockIDs.size(); i++) {
        int oriIdx = mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]];
        mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]] = missIdxs[i];
        mDocsTables[reqID][layerID].erase(oriIdx);
        mDocsTables[reqID][layerID][missIdxs[i]] = loadNPUBlockIDs[i];
    }
}

void GSAPrefetchEngineC::IndexerInitLoad(std::vector<std::vector<int>>& blockTable,
                                         std::vector<std::vector<int>>& slabIdx, std::string reqID)
{
    if (!mOpenIndexerOffload) { return; }
    /* 整程锁收窄为存在性校验 mini 段（含子层任务组装 mini 段，
     * SubmitIndexerInitLoadTask 的 store Load/Wait 锁外）。 */
    mLogger.log(LogLevel::INFO, "Decode step: %u, |KVCache Prefetch| Start IndexerInitLoad\n",
                mDecodeStep);
    auto start = std::chrono::high_resolution_clock::now();
    int number = blockTable[0].size();

    auto ret = aclrtSetCurrentContext(*(static_cast<aclrtContext*>(mContext)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch|  aclrtSetCurrentContext Error:"
                    " %d, Error: %s\n",
                    ret, errMsg ? errMsg : "Unknown error");
        return;
    }

    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::IndexerInitLoad));
        auto hashIt = mAllBlocksHash.find(reqID);
        if (hashIt == mAllBlocksHash.end() || hashIt->second.empty()) {
            mLogger.log(LogLevel::ERROR, "[ERROR] IndexerInitLoad no hash, reqID=%s\n",
                        reqID.c_str());
            return;
        }
    }

    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        if (!ProcessIndexerInitLoadLayer(blockTable, slabIdx, reqID, layerID, number)) { return; }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| End IndexerInitLoad  load cost: %lu\n",
                mDecodeStep, static_cast<unsigned long>(duration.count()));
}

bool GSAPrefetchEngineC::ProcessIndexerInitLoadLayer(std::vector<std::vector<int>>& blockTable,
                                                     std::vector<std::vector<int>>& slabIdx,
                                                     const std::string& reqID, int layerID,
                                                     int number)
{
    if (static_cast<size_t>(layerID) >= mIndexerPtr.size() ||
        static_cast<size_t>(layerID) >= blockTable.size() ||
        static_cast<size_t>(layerID) >= slabIdx.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad layerID out of range: reqID=%s layerID=%d "
                    "indexerPtrSize=%zu blockTableSize=%zu slabIdxSize=%zu\n",
                    reqID.c_str(), layerID, mIndexerPtr.size(), blockTable.size(), slabIdx.size());
        return false;
    }
    auto* indexerBaseAddr = reinterpret_cast<std::uint8_t*>(mIndexerPtr[layerID]);
    if (indexerBaseAddr == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad indexer base address is nullptr, layerID=%d\n",
                    layerID);
        return false;
    }
    UC::Detail::TaskDesc task;
    task.reserve(number);
    task.brief = "UCMIndexerInitLoad";
    /* mini 段：mAllBlocksHash 查找 + shard 组装（hashVec 引用与
     * mLastKVBlockAddress 查找不跨段）锁内；store Load/Wait 锁外。 */
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::IndexerInitLoad));
        auto hashIt = mAllBlocksHash.find(reqID);
        if (hashIt == mAllBlocksHash.end()) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] IndexerInitLoad reqID not found in mAllBlocksHash: "
                        "reqID=%s layerID=%d\n",
                        reqID.c_str(), layerID);
            return false;
        }
        if (!BuildIndexerInitLoadShards(blockTable, slabIdx, reqID, layerID, number,
                                        indexerBaseAddr, hashIt->second, task)) {
            return false;
        }
    }
    return SubmitIndexerInitLoadTask(task, reqID, layerID);
}

bool GSAPrefetchEngineC::BuildIndexerInitLoadShards(std::vector<std::vector<int>>& blockTable,
                                                    std::vector<std::vector<int>>& slabIdx,
                                                    const std::string& reqID, int layerID,
                                                    int number, std::uint8_t* indexerBaseAddr,
                                                    const std::vector<BlockId>& hashVec,
                                                    UC::Detail::TaskDesc& task)
{
    const size_t shardIndex = static_cast<size_t>(layerID);
    for (int i = 0; i < number; i++) {
        if (!TryAppendInitShard(i, blockTable, slabIdx, reqID, layerID, number, indexerBaseAddr,
                                hashVec, shardIndex, task)) {
            return false;
        }
    }
    return true;
}

// 单块分类：UCM 块追加 shard（地址按 store 张量槽位数 padding，ghost 槽留
// nullptr）；非 UCM 块走 LoadNonUcmIndexerBlock；非法输入日志跳过。
bool GSAPrefetchEngineC::TryAppendInitShard(int i, std::vector<std::vector<int>>& blockTable,
                                            std::vector<std::vector<int>>& slabIdx,
                                            const std::string& reqID, int layerID, int number,
                                            std::uint8_t* indexerBaseAddr,
                                            const std::vector<BlockId>& hashVec, size_t shardIndex,
                                            UC::Detail::TaskDesc& task)
{
    if (static_cast<size_t>(i) >= blockTable[layerID].size() ||
        static_cast<size_t>(i) >= slabIdx[layerID].size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad row too short: reqID=%s layerID=%d i=%d "
                    "blockRow=%zu slabRow=%zu\n",
                    reqID.c_str(), layerID, i, blockTable[layerID].size(), slabIdx[layerID].size());
        return true;
    }
    int deviceBlockId = blockTable[layerID][i];
    int blockIdx = slabIdx[layerID][i];
    if (deviceBlockId < 0 || deviceBlockId >= mDeviceIndexerLen) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad deviceBlockId out of range, "
                    "reqID=%s layerID=%d deviceBlockId=%d mDeviceIndexerLen=%d\n",
                    reqID.c_str(), layerID, deviceBlockId, mDeviceIndexerLen);
        return true;
    }
    auto* indexerAddr =
        indexerBaseAddr + static_cast<size_t>(deviceBlockId) * mIndexerSizeBytesBlock;
    if (blockIdx >= static_cast<int>(hashVec.size())) {
        return LoadNonUcmIndexerBlock(reqID, layerID, blockIdx, deviceBlockId, indexerAddr);
    }
    if (blockIdx < 0) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad blockIdx out of range, "
                    "reqID=%s layerID=%d blockIdx=%d hashNum=%zu\n",
                    reqID.c_str(), layerID, blockIdx, hashVec.size());
        return true;
    }
    std::vector<void*> tensorAddrs(mStoreTensorCount, nullptr);
    tensorAddrs[kIndexerTensorIndex] = indexerAddr;
    task.emplace_back(UC::Detail::Shard{hashVec[static_cast<size_t>(blockIdx)], shardIndex,
                                        std::move(tensorAddrs)});
    return true;
}

bool GSAPrefetchEngineC::SubmitIndexerInitLoadTask(UC::Detail::TaskDesc& task,
                                                   const std::string& reqID, int layerID)
{
    if (task.empty()) { return true; }
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad mStoreV1_ is nullptr, reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    const size_t loadNumber = task.size();
    auto loadResult = mStoreV1_->Load(std::move(task));
    if (!loadResult) {
        const std::string error = loadResult.Error().ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad Load failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    auto waitStatus = mStoreV1_->Wait(loadResult.Value());
    if (waitStatus.Failure()) {
        const std::string error = waitStatus.ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad Wait failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    return true;
}

bool GSAPrefetchEngineC::LoadNonUcmIndexerBlock(const std::string& reqID, int layerID, int blockIdx,
                                                int deviceBlockId, std::uint8_t* indexerAddr)
{
    auto lastIt = mLastKVBlockAddress.find(reqID);
    if (lastIt == mLastKVBlockAddress.end()) {
        mLogger.log(LogLevel::ERROR, "|IndexerInitLoad| IndexerInitLoad no non-UCM block map \n");
        return false;
    }

    auto blockAddressIt = lastIt->second.find(blockIdx);
    if (blockAddressIt == lastIt->second.end()) {
        mLogger.log(LogLevel::ERROR,
                    "|IndexerInitLoad| IndexerInitLoad non-UCM block not found \n");
        return false;
    }

    const auto& blockAddress = blockAddressIt->second;
    const auto* indexerBase = reinterpret_cast<const std::uint8_t*>(blockAddress.indexerAddr);
    const void* lastIndexerAddr = static_cast<const void*>(
        indexerBase + static_cast<size_t>(layerID) * mIndexerSizeBytesBlock);

    auto copyRet = aclrtMemcpy(indexerAddr, mIndexerSizeBytesBlock, lastIndexerAddr,
                               mIndexerSizeBytesBlock, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad last block memcpy failed, "
                    "reqID=%s layerID=%d blockIdx=%d deviceBlockId=%d "
                    "ret=%d error=%s \n",
                    reqID.c_str(), layerID, blockIdx, deviceBlockId, copyRet,
                    errMsg ? errMsg : "Unknown error");
        return false;
    }

    return true;
}

void GSAPrefetchEngineC::UCMIndexerLoadAsync(void* context, void* stream)
{
    (void)stream;
    mLogger.log(LogLevel::INFO, "Decode step: %u, |KVCache Prefetch| Start UCMIndexerLoadAsync\n",
                mDecodeStep);
    auto start = std::chrono::high_resolution_clock::now();

    if (context == nullptr) {
        mLogger.log(LogLevel::ERROR, "[ERROR] UCMIndexerLoadAsync context is nullptr\n");
        return;
    }

    auto ret = aclrtSetCurrentContext(*(static_cast<aclrtContext*>(context)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
        return;
    }

    size_t totalNumber = 0;
    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        if (!LoadIndexerLayer(layerID, totalNumber)) { return; }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| End UCMIndexerLoadAsync "
                "blockNum=%zu load cost: %lu us\n",
                mDecodeStep, totalNumber, static_cast<unsigned long>(duration.count()));
}

bool GSAPrefetchEngineC::LoadIndexerLayer(int layerID, size_t& totalNumber)
{
    auto* indexerBaseAddr = reinterpret_cast<std::uint8_t*>(mIndexerPtr[layerID]);
    if (indexerBaseAddr == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync indexer base address is nullptr, "
                    "layerID=%d\n",
                    layerID);
        return false;
    }
    for (auto it = allNeedLoadBlock.begin(); it != allNeedLoadBlock.end(); ++it) {
        const std::string& reqID = it->first;
        const auto& hbmBlockIDs = it->second[layerID];
        if (!LoadIndexerRequest(reqID, hbmBlockIDs, layerID, indexerBaseAddr, totalNumber)) {
            return false;
        }
    }
    return true;
}

bool GSAPrefetchEngineC::LoadIndexerRequest(const std::string& reqID,
                                            const std::vector<int>& hbmBlockIDs, int layerID,
                                            std::uint8_t* indexerBaseAddr, size_t& totalNumber)
{
    int number = static_cast<int>(hbmBlockIDs.size());
    if (number == 0) { return true; }
    UC::Detail::TaskDesc task;
    task.reserve(number);
    task.brief = "UCMIndexerLoad";
    if (!ResolveIndexerLoadTask(reqID, hbmBlockIDs, layerID, indexerBaseAddr, task, totalNumber)) {
        return false;
    }
    if (task.empty()) { return true; }
    return LoadIndexerTask(reqID, layerID, task, totalNumber);
}

// mini 段：mAllBlocksHash/mLastKVBlockAddress/allMissIdxs 查找与任务组装锁内
// （防 rehash 引用失效）；ms 级 store Load/Wait 留给调用方锁外执行。
bool GSAPrefetchEngineC::ResolveIndexerLoadTask(const std::string& reqID,
                                                const std::vector<int>& hbmBlockIDs, int layerID,
                                                std::uint8_t* indexerBaseAddr,
                                                UC::Detail::TaskDesc& task, size_t& totalNumber)
{
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::IndexerLoadResolve));
    auto hashIt = mAllBlocksHash.find(reqID);
    if (hashIt == mAllBlocksHash.end()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync no hash, "
                    "reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    auto missIt = allMissIdxs.find(reqID);
    if (missIt == allMissIdxs.end()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync no miss indexes, "
                    "reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    const auto& hashVec = hashIt->second;
    const auto& missBlockIdxs = missIt->second[layerID];
    if (hbmBlockIDs.size() != missBlockIdxs.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync size mismatch, "
                    "reqID=%s layerID=%d hbmNum=%zu missNum=%zu\n",
                    reqID.c_str(), layerID, hbmBlockIDs.size(), missBlockIdxs.size());
        return false;
    }
    return BuildIndexerLoadTask(reqID, layerID, indexerBaseAddr, hbmBlockIDs, missBlockIdxs,
                                hashVec, task, totalNumber);
}

bool GSAPrefetchEngineC::BuildIndexerLoadTask(const std::string& reqID, int layerID,
                                              std::uint8_t* indexerBaseAddr,
                                              const std::vector<int>& hbmBlockIDs,
                                              const std::vector<int>& missBlockIdxs,
                                              const std::vector<BlockId>& hashVec,
                                              UC::Detail::TaskDesc& task, size_t& totalNumber)
{
    const int lastUCMBlockIdx = static_cast<int>(hashVec.size());
    size_t shardIndex = static_cast<size_t>(layerID);

    for (size_t i = 0; i < hbmBlockIDs.size(); i++) {
        int hbmBlockID = hbmBlockIDs[i];
        int missBlockIdx = missBlockIdxs[i];
        void* indexerAddr = static_cast<void*>(indexerBaseAddr + static_cast<size_t>(hbmBlockID) *
                                                                     mIndexerSizeBytesBlock);

        if (missBlockIdx >= lastUCMBlockIdx) {
            if (!CopyNonUCMIndexerBlock(reqID, layerID, missBlockIdx, hbmBlockID, indexerAddr,
                                        totalNumber)) {
                return false;
            }
            continue;
        }

        if (missBlockIdx < 0 || static_cast<size_t>(missBlockIdx) >= hashVec.size()) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] UCMIndexerLoadAsync missBlockIdx out of range, "
                        "reqID=%s layerID=%d missBlockIdx=%d, hashNum=%zu\n",
                        reqID.c_str(), layerID, missBlockIdx, hashVec.size());
            continue;
        }
        /* Same slot-padded addrs contract as the init load: the store's H2D
         * path indexes addrs[i] for every tensorSizes_ slot. */
        std::vector<void*> tensorAddrs(mStoreTensorCount, nullptr);
        tensorAddrs[kIndexerTensorIndex] = indexerAddr;
        task.emplace_back(
            UC::Detail::Shard{hashVec[missBlockIdx], shardIndex, std::move(tensorAddrs)});
    }
    return true;
}

bool GSAPrefetchEngineC::CopyNonUCMIndexerBlock(const std::string& reqID, int layerID,
                                                int missBlockIdx, int hbmBlockID, void* indexerAddr,
                                                size_t& totalNumber)
{
    auto lastIt = mLastKVBlockAddress.find(reqID);
    if (lastIt == mLastKVBlockAddress.end()) {
        mLogger.log(LogLevel::ERROR, "|IndexerInitLoad| IndexerInitLoad no non-UCM block map \n");
        return false;
    }
    auto blockAddressIt = lastIt->second.find(missBlockIdx);
    if (blockAddressIt == lastIt->second.end()) {
        mLogger.log(LogLevel::ERROR, "|IndexerInitLoad| IndexerInitLoad no non-UCM block map \n");
        return false;
    }
    const auto& blockAddress = blockAddressIt->second;
    const auto* indexerBase = reinterpret_cast<const std::uint8_t*>(blockAddress.indexerAddr);
    const void* lastIndexerAddr = static_cast<const void*>(
        indexerBase + static_cast<size_t>(layerID) * mIndexerSizeBytesBlock);
    auto copyRet = aclrtMemcpy(indexerAddr, mIndexerSizeBytesBlock, lastIndexerAddr,
                               mIndexerSizeBytesBlock, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync last block memcpy failed, "
                    "reqID=%s layerID=%d missBlockIdx=%d hbmBlockID=%d "
                    "ret=%d error=%s\n",
                    reqID.c_str(), layerID, missBlockIdx, hbmBlockID, copyRet,
                    errMsg ? errMsg : "Unknown error");
        return false;
    }
    totalNumber++;
    return true;
}

bool GSAPrefetchEngineC::LoadIndexerTask(const std::string& reqID, int layerID,
                                         UC::Detail::TaskDesc& task, size_t& totalNumber)
{
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync mStoreV1_ is nullptr, reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    const size_t loadNumber = task.size();
    auto loadResult = mStoreV1_->Load(std::move(task));
    if (!loadResult) {
        const std::string error = loadResult.Error().ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync Load failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    auto waitStatus = mStoreV1_->Wait(loadResult.Value());
    if (waitStatus.Failure()) {
        const std::string error = waitStatus.ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync Wait failed, "
                    "reqID=%s layerID=%d loadNumber=%zu status=%s\n",
                    reqID.c_str(), layerID, loadNumber, error.c_str());
        return false;
    }
    totalNumber += loadNumber;
    return true;
}

void GSAPrefetchEngineC::SetKvTensor(torch::Tensor kCachesTensor, torch::Tensor vCachesTensor,
                                     torch::Tensor lastChunkKBuffer, torch::Tensor lastChunkvBuffer,
                                     int layerID)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start SetKvTensor layerID: %u\n", mDecodeStep,
                layerID);

    // init token an block tensor size
    if (mKVHBMBlockLen == 0) { mKVHBMBlockLen = kCachesTensor.sizes()[0]; }
    mKcacheTensor[layerID] = kCachesTensor;
    mKcachePtr[layerID] = reinterpret_cast<uint64_t>(kCachesTensor.data_ptr());
    at::Tensor kCachesTensorND = kCachesTensor.view({-1, mKDim});
    mKCacheAcl[layerID] = AtTensorToAclTensorShared(kCachesTensorND);
    mLastChunkNum = mSparseLen / mBlockSize;
    mLastCHunkKBuffer[layerID] = lastChunkKBuffer;
    mLastCHunkKPtr[layerID] = reinterpret_cast<uint64_t>(mLastCHunkKBuffer[layerID].data_ptr());
    if (!mISSPAC8) {
        mVcacheTensor[layerID] = vCachesTensor;
        mVcachePtr[layerID] = reinterpret_cast<uint64_t>(vCachesTensor.data_ptr());
        at::Tensor vCachesTensorND = vCachesTensor.view({-1, mVDim});
        mVCacheAcl[layerID] = AtTensorToAclTensorShared(vCachesTensorND);
        mLastCHunkVBuffer[layerID] = lastChunkvBuffer;
        mLastCHunkVPtr[layerID] = reinterpret_cast<uint64_t>(mLastCHunkVBuffer[layerID].data_ptr());
    }
    mLogger.log(LogLevel::INFO, "Decode step: %u, |KVCache Prefetch| SetKvTensor for layer: %d\n",
                mDecodeStep, layerID);
}

void GSAPrefetchEngineC::SetKvTensorIndex(int layerID, torch::Tensor indexCache)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start SetKvTensorIndex layerID: %u\n",
                mDecodeStep, layerID);
    if (!mOpenIndexerOffload) { return; }
    if (mDeviceIndexerLen == 0) {
        mDeviceIndexerLen = indexCache.sizes()[0];
        mIndexerSizeBytesBlock = indexCache[0].element_size() * indexCache[0].numel();
        mLogger.log(LogLevel::INFO,
                    "Decode step: %u, "
                    "|KVCache Prefetch| init indexer kvcache, mDeviceIndexerLen: %d, "
                    "mIndexerSizeBytesBlock: %d\n",
                    mDecodeStep, mDeviceIndexerLen, mIndexerSizeBytesBlock);
    }
    mIndexerTensor[layerID] = indexCache;
    mIndexerPtr[layerID] = reinterpret_cast<uint64_t>(indexCache.data_ptr());
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| SetKvTensorIndex for layer: %d\n", mDecodeStep,
                layerID);
}

enum class H2DInfoIndex {
    KBufferHost = 0,
    VBufferHost,
    KBufferDevice,
    VBufferDevice,
    LoadIndexHost,
    LoadIndexDevice
};

void GSAPrefetchEngineC::SetKvBuffer(torch::Tensor kBufferHost, torch::Tensor vBufferHost,
                                     torch::Tensor kBufferDevice, torch::Tensor vBufferDevice,
                                     torch::Tensor loadIndexHost, torch::Tensor loadIndexDevice)
{
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| Start SetKvBuffer\n",
                mDecodeStep);
    mKBufferHost = kBufferHost;
    mKBufferDevice = kBufferDevice;
    mLoadIndexHost = loadIndexHost;
    mLoadIndexDevice = loadIndexDevice;
    if (!mISSPAC8) {
        mVBufferHost = vBufferHost;
        mVBufferDevice = vBufferDevice;
    }
    if (mKVBufferLen == 0) {
        mKVBufferLen = mKBufferHost[0].sizes()[0];
        mLoadIndexBufferLen = mLoadIndexHost[0].sizes()[0];
    }
    mH2DPtrInfo.resize(mStreamNum);
    for (int i = 0; i < mStreamNum; i++) {
        mH2DPtrInfo[i].resize(H2D_INFO_SIZE);
        if (!mISSPAC8) {
            mVBufferHostPtr[i] = reinterpret_cast<uint64_t>(mVBufferHost[i].data_ptr());
            mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::VBufferHost)] =
                mVBufferHost[i].data_ptr();
            mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::VBufferDevice)] =
                mVBufferDevice[i].data_ptr();
        }
        mKBufferHostPtr[i] = reinterpret_cast<uint64_t>(mKBufferHost[i].data_ptr());
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::KBufferHost)] = mKBufferHost[i].data_ptr();
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::KBufferDevice)] =
            mKBufferDevice[i].data_ptr();
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::LoadIndexHost)] =
            mLoadIndexHost[i].data_ptr();
        mH2DPtrInfo[i][static_cast<size_t>(H2DInfoIndex::LoadIndexDevice)] =
            mLoadIndexDevice[i].data_ptr();
    }
    if (mKVPackControl == nullptr && InitKVPackControl(mKVPackDpRank) != 0) {
        mLogger.log(LogLevel::ERROR, "|KVPack| shared staging initialization failed\n");
        std::abort();
    }
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| SetKvBuffer\n", mDecodeStep);
}

void GSAPrefetchEngineC::SubmitTask(std::vector<int>& topkLensInput, std::vector<int>& bsIndexInput,
                                    std::vector<std::string>& reqIDList, torch::Tensor& topkLists,
                                    std::vector<bool>& isIndexerOffload, std::vector<int>& seqLens,
                                    uint32_t stepTime)
{
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| start SubmitTask batch size: %lu\n",
                mDecodeStep, reqIDList.size());
    auto start = std::chrono::high_resolution_clock::now();
    mDecodeStep = stepTime;
    // 埋点 dump 触发点：PROFILE 开启时按间隔周期输出争用画像。
    if (mUReqLockProfileEnabled && static_cast<uint64_t>(mDecodeStep) >= mUReqLockNextDumpStep) {
        DumpUReqLockStats();
        mUReqLockNextDumpStep = static_cast<uint64_t>(mDecodeStep) + mUReqLockDumpIntervalSteps;
    }
    // set_kv_buffer 未完成（KVPack 控制块为空）时提交会在 worker 路径空指针
    // 崩溃，直接日志+跳过本步提交。
    if (mKVPackControl == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| set_kv_buffer not done yet, "
                    "skip prefetch task\n",
                    mDecodeStep);
        std::cerr << "|KVCache Prefetch| [ERROR] set_kv_buffer not done, prefetch task skipped\n";
        return;
    }
    TaskInfo inputInfo(totalTaskNum, -1, topkLists, topkLensInput, bsIndexInput, reqIDList);
    mIsPrefetchRunning.store(true, std::memory_order_release);
    mIsIndexerOffload = isIndexerOffload;
    SyncIndexerSubmitState(reqIDList, seqLens);
    mTaskManger->SubmitTask(inputInfo, MultiBSThreadFun);
    totalTaskNum += 1;
    auto endLoadPre = std::chrono::high_resolution_clock::now();
    auto durationLoadPre =
        std::chrono::duration_cast<std::chrono::microseconds>(endLoadPre - start);
    mLogger.log(
        LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| finish SubmitTask batch size: %lu, cost: %lu\n",
        mDecodeStep, reqIDList.size(), durationLoadPre.count());
}

// 提交期同步 indexer 状态（mRealTimeIdx/mSeqLen）到 prefetch worker 可见面。
// 与 worker 清理段 erase 并发，对称加锁；DEBUG 日志移出临界区。
void GSAPrefetchEngineC::SyncIndexerSubmitState(std::vector<std::string>& reqIDList,
                                                std::vector<int>& seqLens)
{
    struct SubmitTaskDebugSize {
        std::string reqId;
        uint32_t realTimeIdxSize;
        uint32_t useBlockTableIdxSize;
    };
    std::vector<SubmitTaskDebugSize> debugSizes;
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SubmitTask));
        for (uint32_t i = 0; i < reqIDList.size(); i++) {
            if (mOpenIndexerOffload && mIsIndexerOffload[i]) {
                mRealTimeIdx[reqIDList[i]].clear();
                mRealTimeIdx[reqIDList[i]] = mUseBlockTableIdx[reqIDList[i]];
                mSeqLen[reqIDList[i]] = seqLens[i];
                debugSizes.push_back(
                    {reqIDList[i], static_cast<uint32_t>(mRealTimeIdx[reqIDList[i]][0].size()),
                     static_cast<uint32_t>(mUseBlockTableIdx[reqIDList[i]][0].size())});
            }
        }
    }
    for (size_t k = 0; k < debugSizes.size(); ++k) {
        mLogger.log(LogLevel::DEBUG,
                    "Decode step: %u, "
                    "|KVCache Prefetch| SubmitTask mRealTimeIdx: %u, mUseBlockTableIdx: %u "
                    "for req_id: %s\n",
                    mDecodeStep, debugSizes[k].realTimeIdxSize, debugSizes[k].useBlockTableIdxSize,
                    debugSizes[k].reqId.c_str());
    }
}

void GSAPrefetchEngineC::SubmitTaskIndexer(std::vector<int>& topkLensInput,
                                           std::vector<std::string>& reqIDList,
                                           std::vector<torch::Tensor>& topkLists, uint32_t stepTime)
{
    if (!mOpenIndexerOffload) { return; }
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| start SubmitTaskIndexer batch size: %lu\n",
                mDecodeStep, reqIDList.size());
    auto start = std::chrono::high_resolution_clock::now();
    /* mPrefetchBlockTableIdx 唯一写者为 indexer worker 的
     * FinishOneLayerPrefetch（表事务锁内），读侧对称加锁；is_indexer_load
     * 门控保证提交时 indexer worker 空闲，锁无竞争。 */
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SubmitTaskIndexer));
    for (uint32_t i = 0; i < reqIDList.size(); i++) {
        for (int layerID = 0; layerID < mLayerNum; layerID++) {
            std::memcpy(mUseBlockTableIdx[reqIDList[i]][layerID].data(),
                        mPrefetchBlockTableIdx[reqIDList[i]][layerID].data(),
                        topkLensInput[i] * sizeof(int));
        }
    }
    mDecodeStep = stepTime;
    TaskInfo inputInfo;
    inputInfo.indexerTopkList = topkLists;
    inputInfo.topkLenList = topkLensInput;
    inputInfo.reqIdList = reqIDList;
    mIsIndexerLoadRunning.store(true, std::memory_order_release);
    mTaskManger->SubmitTask(inputInfo, MultiBSThreadFunIndexer);
    totalTaskNum += 1;
    auto endLoadPre = std::chrono::high_resolution_clock::now();
    auto durationLoadPre =
        std::chrono::duration_cast<std::chrono::microseconds>(endLoadPre - start);
    mLogger.log(
        LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| finish SubmitTaskIndexer batch size: %lu, cost: %lu\n",
        mDecodeStep, reqIDList.size(), durationLoadPre.count());
}

void GSAPrefetchEngineC::SetTopkIndices(std::string reqID, torch::Tensor& topIndices)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start SetTopkIndices reqID: %s\n", mDecodeStep,
                reqID.c_str());
    /* mAllTopkIndicesTensor 读写统一 mUpdateReqMutex 协议（读者：worker 树；
     * erase 侧：DelReqIDRun）。新 req 首个 decode step 的 map 插入必然与
     * worker 并发交叠，锁不可省；收窄到仅包 map 赋值，End 日志移出临界区。 */
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SetTopkIndices));
        mAllTopkIndicesTensor[reqID] = topIndices;
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End SetTopkIndices reqID: %s\n", mDecodeStep,
                reqID.c_str());
}

/* PreSwapTokens 分解辅助：逐层计算 swap 追加的末尾边界 appendEnds[layerID]。
 * decay 模式取 (req,layer) 管理器的分层窗口（Budget5120 表 + clamp，如
 * sparseLen=5120 时 layer0 → 4096）；Lru / 管理器未就绪 / 窗口越界时退化为
 * 整段 sparseLen（与原行为一致）。find 判别与 FindReqLayerManager 同协议。 */
void GSAPrefetchEngineC::CollectLayerAppendEnds(const std::string& reqID, int sparseLen,
                                                std::vector<int>& appendEnds)
{
    appendEnds.assign(static_cast<size_t>(mTopkIndexerNum), sparseLen);
    for (int layerID = 0; layerID < mTopkIndexerNum; ++layerID) {
        HBMManagerLayered* manager = FindReqLayerManager(reqID, layerID);
        if (manager == nullptr || !manager->isDecayMode()) { continue; }
        int win = manager->getWindowSize();
        win = std::min(std::max(win, 1), sparseLen);
        appendEnds[static_cast<size_t>(layerID)] = win;
    }
}

/* CopyPreSwapTokens 的追加边界（decay 分层窗口）：
 *   Lru：appendEnd == sparseLen，整段拷贝 + 尾部追加，与原实现逐字节一致。
 *   Decay：appendEnd == 该层窗口 W < sparseLen（如 sparseLen=5120、layer0
 *     需要 4096 => W=4096）：
 *     - 整段拷贝改为前对齐窗口拷贝 [0, W)，尾部 [W, sparseLen) 显式补 -1
 *       padding（不依赖 use 缓冲区尾段内容，恒为 -1）；
 *     - 最近 token 落在窗口末尾 [W-x, W)（x 为本次追加数），padding 不被
 *       触碰；增量装不下时截尾保留最新 W 个 token。
 * 追加计数 count = changeLen - mtpToken + 1 + index % mtpToken，写入偏移恒为
 * appendEnd - count（与原公式 sparseLen - changeLen + mtpToken - 1 - index%mtpToken
 * 代数等价，end 换成 appendEnd）。 */
void GSAPrefetchEngineC::CopyPreSwapTokens(uintptr_t useTokenBase, uintptr_t spaTokenBase,
                                           const std::vector<int>& swapRange,
                                           const std::vector<int>& needAppend, int sparseLen,
                                           int spaMaxBs, int mtpToken, int changeLen,
                                           const std::vector<int>& appendEnds)
{
#pragma omp parallel for num_threads(8) proc_bind(master)
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        void* usePtr =
            reinterpret_cast<void*>(useTokenBase + sparseLen * layerID * sizeof(int32_t));
        const int appendEnd = (static_cast<size_t>(layerID) < appendEnds.size())
                                  ? appendEnds[static_cast<size_t>(layerID)]
                                  : sparseLen;
        for (int index = swapRange[0]; index < swapRange[1]; index++) {
            int32_t* spaCpuPtr = reinterpret_cast<int32_t*>(spaTokenBase) +
                                 (size_t)(layerID * spaMaxBs + index) * sparseLen;
            if (appendEnd < sparseLen) {
                std::memcpy(spaCpuPtr, usePtr, appendEnd * sizeof(int32_t));
                std::fill(spaCpuPtr + appendEnd, spaCpuPtr + sparseLen, -1);
            } else {
                std::memcpy(spaCpuPtr, usePtr, sparseLen * sizeof(int32_t));
            }
            int count = changeLen - mtpToken + 1 + index % mtpToken;
            const int32_t* src = needAppend.data();
            if (count > appendEnd) {
                mLogger.log(LogLevel::WARNING,
                            "|KVCache Prefetch| PreSwapTokens append %d tokens exceeds layer %d "
                            "window %d, keeping the newest %d\n",
                            count, layerID, appendEnd, appendEnd);
                src += count - appendEnd;
                count = appendEnd;
            }
            std::memcpy(spaCpuPtr + appendEnd - count, src, count * sizeof(int));
        }
    }
}

void GSAPrefetchEngineC::PreSwapTokens(std::vector<uintptr_t>& useTokenPtrs,
                                       std::vector<int32_t>& seqLens,
                                       std::vector<std::vector<int>>& swapIndex,
                                       std::vector<uintptr_t>& spaTokensPtr,
                                       std::vector<std::string>& reqIDs, int sparseLen,
                                       int spaMaxBs, int mtpToken)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch|"
                " Start PreSwapTokens, sparseLen: %d, spaMaxBs: %d, mtpToken: %d\n",
                mDecodeStep, sparseLen, spaMaxBs, mtpToken);
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = aclrtSetCurrentContext(*(static_cast<aclrtContext*>(mContext)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
    }
    uint64_t max_tokens_time = 0;
    uint64_t cpu_copy_time = 0;
    for (size_t bs = 0; bs < useTokenPtrs.size(); bs++) {
        SwapOneBsTokens(bs, useTokenPtrs, seqLens, swapIndex, spaTokensPtr, reqIDs[bs], sparseLen,
                        spaMaxBs, mtpToken, max_tokens_time, cpu_copy_time);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End PreSwapTokens, "
                "bs: %lu total time: %lu max_tokens_time: %lu cpu_copy_time: %lu\n",
                mDecodeStep, useTokenPtrs.size(), duration.count(), max_tokens_time, cpu_copy_time);
}

// 单 bs 的 swap token 增量计算与拷贝（计时累加到出参）。
void GSAPrefetchEngineC::SwapOneBsTokens(size_t bs, std::vector<uintptr_t>& useTokenPtrs,
                                         std::vector<int32_t>& seqLens,
                                         std::vector<std::vector<int>>& swapIndex,
                                         std::vector<uintptr_t>& spaTokensPtr,
                                         const std::string& reqID, int sparseLen, int spaMaxBs,
                                         int mtpToken, uint64_t& maxTokensUs, uint64_t& cpuCopyUs)
{
    auto start_max = std::chrono::high_resolution_clock::now();
    const int32_t* data = reinterpret_cast<const int32_t*>(useTokenPtrs[bs]);
    int maxID = *std::max_element(data, data + sparseLen * mTopkIndexerNum);
    int start = maxID + 1;
    int end = seqLens[bs] + mtpToken - 1;
    int change_len = end - start;
    std::vector<int> needAppend;
    if (change_len <= mtpToken - 1) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] PreSwapTokens change len < 0 !!!!!! start %d, end %d\n", start, end);
        change_len = mtpToken;
    }
    mLogger.log(
        LogLevel::DEBUG,
        "Decode step: %u, |KVCache Prefetch| Start PreSwapTokens, change_len: %d, maxID: %d\n",
        mDecodeStep, change_len, maxID);
    needAppend.resize(change_len);
    std::iota(needAppend.begin(), needAppend.end(), start);
    auto max_end = std::chrono::high_resolution_clock::now();
    maxTokensUs +=
        std::chrono::duration_cast<std::chrono::microseconds>(max_end - start_max).count();
    /* decay 分层窗口：逐层追加边界（Lru 时全为 sparseLen，行为与原实现一致） */
    std::vector<int> appendEnds;
    CollectLayerAppendEnds(reqID, sparseLen, appendEnds);
    CopyPreSwapTokens(useTokenPtrs[bs], spaTokensPtr[0], swapIndex[bs], needAppend, sparseLen,
                      spaMaxBs, mtpToken, change_len, appendEnds);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    cpuCopyUs += std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - max_end).count();
}

void GSAPrefetchEngineC::DelBlocksMap(std::string reqID)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start DelBlocksMap, reqID: %s\n", mDecodeStep,
                reqID.c_str());
    {
        UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::DelBlocksMap));
        mDelSeqIds.insert(reqID);
        // Only run the deletion when neither prefetch nor indexer load is running,
        // so that the shared per-request maps are not erased mid-update.
        if (!GetPrefetchStatus() && !GetIndexerStatus()) { DelReqIDRun(); }
    }
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End DelBlocksMap, reqID: %s\n", mDecodeStep,
                reqID.c_str());
}

void GSAPrefetchEngineC::DelReqIDRun()
{
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| Start DelReqIDRun\n",
                mDecodeStep);
    for (auto it = mDelSeqIds.begin(); it != mDelSeqIds.end(); it++) {
        if (mSparseSeqLen.find(*it) == mSparseSeqLen.end()) {
            mLogger.log(LogLevel::INFO,
                        "Decode step: %u, |KVCache Prefetch| req: %s not in mSparseSeqLen\n",
                        mDecodeStep, (*it).c_str());
            continue;
        } else {
            mAllTopkIndicesTensor.erase(*it);
            mSparseSeqLen.erase(*it);
            mAllReqIdSlots.erase(*it);
            mPromptLen.erase(*it);
            mHBMMangers.erase(*it);
            mMissGlobalToken.erase(*it);
            mLoadSlot.erase(*it);
            mReqUseBlockTable.erase(*it);
            /* L-4 修复：SetBlocksMap 写入的两兄弟成员一并清理，
             * 否则每完成请求泄漏 16B×块数（BlockId/LastKVBlockAddress 条目）。 */
            mAllBlocksHash.erase(*it);
            mLastKVBlockAddress.erase(*it);
            mBlocksMap.erase(*it);
            mDocsTables.erase(*it);
            mPrefetchIdx.erase(*it);
            mIndexerLoadResultT.erase(*it);
            mPrefetchBlockTableIdx.erase(*it);
            mUseBlockTableIdx.erase(*it);
            mRealTimeIdx.erase(*it);
            mSeqLen.erase(*it);
            allNeedLoadBlock.erase(*it);
            allMissIdxs.erase(*it);
            mLogger.log(LogLevel::INFO,
                        "Decode step: %u, |KVCache Prefetch| delete info of req: %s\n", mDecodeStep,
                        (*it).c_str());
        }
    }
    mDelSeqIds.clear();
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch| End DelReqIDRun\n",
                mDecodeStep);
}

bool GSAPrefetchEngineC::GetPrefetchStatus()
{
    return mIsPrefetchRunning.load(std::memory_order_acquire);
}

bool GSAPrefetchEngineC::GetIndexerStatus()
{
    return mIsIndexerLoadRunning.load(std::memory_order_acquire);
}

void GSAPrefetchEngineC::WaitPrefetchIdle(uint64_t timeoutMs)
{
    // 谓词式 wait_for：唤醒后重查原子位，虚假唤醒免疫。
    std::unique_lock<std::mutex> lock(mTaskDoneMutex);
    mPrefetchDoneCv.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this] {
        return !mIsPrefetchRunning.load(std::memory_order_acquire);
    });
}

void GSAPrefetchEngineC::WaitIndexerIdle(uint64_t timeoutMs)
{
    std::unique_lock<std::mutex> lock(mTaskDoneMutex);
    mIndexerDoneCv.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this] {
        return !mIsIndexerLoadRunning.load(std::memory_order_acquire);
    });
}
}  // namespace ucmprefetch
