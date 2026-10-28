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
// 0表示使用原来的HOST预取、组合，1表示优化后的UCM统一管理
#define IS_UCM_BUFFER_OPEN 1

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

    PreparePrefetchResults(taskInfo);

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
    Synchronize(threadInfo.context_, threadInfo.stream_, mStreamNum, mLogger);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish async pretch, "
                "Synchronize cost: %lu, manager time:%lu, "
                "cpu merge time: %lu, memcpy time: %lu, "
                "sync time: %lu, h2d time: %lu, ops time: %lu\n",
                mDecodeStep, duration.count(), managerDuration.count(), timing.cpuMove,
                timing.memcpy, timing.sync, timing.h2d, timing.ops);
    return 0;
}

void GSAPrefetchEngineC::PreparePrefetchResults(const TaskInfo& taskInfo)
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
#pragma omp parallel for num_threads(mThreadNum) proc_bind(master)
        for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
            for (size_t i = 0; i < oneRunBsLen; i++) {
                ProcessPrefetchRequest(taskInfo, i, layerID, layout);
            }
        }
        __atomic_store_n(versionPtr, mPrefetchResultVersion.fetch_add(1) + 1, __ATOMIC_RELEASE);
        mLogger.log(LogLevel::INFO, "Rank %d synced upload version: %u, %u\n", mRank,
                    mPrefetchResultVersion.load(), __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE));
    } else {
        uint32_t expectedVersion = mPrefetchResultVersion.load() + 1;
        int retryCount = 0;
        while (__atomic_load_n(versionPtr, __ATOMIC_ACQUIRE) != expectedVersion &&
               retryCount < 10000) {
            if (mKVPackControl->fatalError.load(std::memory_order_acquire) != 0) {
                KVPackAbort(mKVPackControl);
            }
            usleep(100);
            retryCount++;
        }
        if (retryCount >= 10000) {
            mLogger.log(LogLevel::ERROR,
                        "Rank %d timeout waiting for prefetch result, version: %u, expected: %u\n",
                        mRank, __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE), expectedVersion);
            KVPackAbort(mKVPackControl);
        }
        ReadPrefetchResults(taskInfo, layout);
        mPrefetchResultVersion.store(expectedVersion);
        mLogger.log(LogLevel::INFO, "Rank %d synced prefetch results, version: %u\n", mRank,
                    __atomic_load_n(versionPtr, __ATOMIC_ACQUIRE));
    }
}

void GSAPrefetchEngineC::ProcessPrefetchRequest(const TaskInfo& taskInfo, size_t i, int layerID,
                                                const PrefetchShmLayout& layout)
{
    std::string oneReqID = taskInfo.reqIdList[i];
    int oneTopkLen = taskInfo.topkLenList[i];
    if (oneTopkLen <= 0) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| topk len is zero: "
                    "%d for req_id: %s, rank: %d, taskID: %d, i: %d\n",
                    mDecodeStep, oneTopkLen, oneReqID.c_str(), mRank, taskInfo.taskID, i);
        return;
    }
    int* topkPtr = taskInfo.topkList[layerID][i].data_ptr<int>();
    std::vector<int> oneTopkList(topkPtr, topkPtr + oneTopkLen);
    mMissGlobalToken[oneReqID][layerID].clear();
    mLoadSlot[oneReqID][layerID].clear();
    std::vector<int> useBlockTable;
    if (mOpenIndexerOffload && mIsIndexerOffload[i]) {
        TokenIndexMapInter(oneTopkList, i, layerID, oneTopkLen, oneReqID);
    }

    mHBMMangers[oneReqID][layerID]->processTopkWithDebug(oneTopkList, useBlockTable,
                                                         mMissGlobalToken[oneReqID][layerID],
                                                         mLoadSlot[oneReqID][layerID], mLogger);
    if (mMissGlobalToken[oneReqID][layerID].size() != mLoadSlot[oneReqID][layerID].size() ||
        mMissGlobalToken[oneReqID][layerID].size() > static_cast<size_t>(mMaxLoadNum)) {
        KVPackAbort(mKVPackControl);
    }
    mAvgHitRate[layerID].emplace_back(mHBMMangers[oneReqID][layerID]->mHitRate);
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| expand topk len: "
                "%lu and hit rate %f and average hit rate: %f, for layer: %d req_id: %s,  \n",
                mDecodeStep, oneTopkList.size(), mHBMMangers[oneReqID][layerID]->mHitRate,
                average(mAvgHitRate[layerID]), layerID, oneReqID.c_str());
    if (mSpaTopkLen != useBlockTable.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] mAllTopkIndicesTensor greater than mSpaTopkLen !!! layer_id %d, "
                    "useBlockTable size %d\n",
                    layerID, mSpaTopkLen, useBlockTable.size());
        KVPackAbort(mKVPackControl);
    } else {
        std::memcpy(mAllTopkIndicesTensor[oneReqID][layerID].data_ptr(), useBlockTable.data(),
                    useBlockTable.size() * sizeof(int));
    }
    WritePrefetchResult(oneReqID, i, layerID, useBlockTable, layout);
}

void GSAPrefetchEngineC::WritePrefetchResult(const std::string& oneReqID, size_t i, int layerID,
                                             const std::vector<int>& useBlockTable,
                                             const PrefetchShmLayout& layout)
{
    size_t reqLayerOffset = (i * mTopkIndexerNum + layerID) * layout.totalSize;
    uint8_t* shmBase =
        static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize + reqLayerOffset;
    uint32_t missTokenCount =
        std::min((size_t)mMissGlobalToken[oneReqID][layerID].size(), (size_t)layout.maxLoadNum);
    std::memcpy(shmBase, &missTokenCount, layout.countSize);

    if (missTokenCount > 0) {
        std::memcpy(shmBase + layout.countSize, mMissGlobalToken[oneReqID][layerID].data(),
                    missTokenCount * sizeof(int));
    }

    uint32_t loadSlotCount =
        std::min((size_t)mLoadSlot[oneReqID][layerID].size(), (size_t)layout.maxLoadNum);
    std::memcpy(shmBase + layout.missTokenSize, &loadSlotCount, layout.countSize);
    if (loadSlotCount > 0) {
        std::memcpy(shmBase + layout.missTokenSize + layout.countSize,
                    mLoadSlot[oneReqID][layerID].data(), loadSlotCount * sizeof(int));
    }
    std::memcpy(shmBase + layout.missTokenSize + layout.loadSlotSize, useBlockTable.data(),
                useBlockTable.size() * sizeof(int));
}

void GSAPrefetchEngineC::ReadPrefetchResults(const TaskInfo& taskInfo,
                                             const PrefetchShmLayout& layout)
{
    uint32_t oneRunBsLen = taskInfo.reqIdList.size();
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        for (size_t i = 0; i < oneRunBsLen; i++) {
            if (taskInfo.topkLenList[i] <= 0) { continue; }
            std::string oneReqID = taskInfo.reqIdList[i];
            size_t reqLayerOffset = (i * mTopkIndexerNum + layerID) * layout.totalSize;
            uint8_t* shmBase =
                static_cast<uint8_t*>(mPrefetchResultShm) + layout.versionSize + reqLayerOffset;
            uint32_t missTokenCount = 0;
            memcpy(&missTokenCount, shmBase, layout.countSize);
            mMissGlobalToken[oneReqID][layerID].clear();
            if (missTokenCount > 0) {
                mMissGlobalToken[oneReqID][layerID].resize(missTokenCount);
                memcpy(mMissGlobalToken[oneReqID][layerID].data(), shmBase + layout.countSize,
                       missTokenCount * sizeof(int));
            }
            uint32_t loadSlotCount = 0;
            memcpy(&loadSlotCount, shmBase + layout.missTokenSize, layout.countSize);
            mLoadSlot[oneReqID][layerID].clear();
            if (loadSlotCount > 0) {
                mLoadSlot[oneReqID][layerID].resize(loadSlotCount);
                memcpy(mLoadSlot[oneReqID][layerID].data(),
                       shmBase + layout.missTokenSize + layout.countSize,
                       loadSlotCount * sizeof(int));
            }
            std::vector<int> useBlockTable(mSpaTopkLen);
            memcpy(useBlockTable.data(), shmBase + layout.missTokenSize + layout.loadSlotSize,
                   mSpaTopkLen * sizeof(int));
            std::memcpy(mAllTopkIndicesTensor[oneReqID][layerID].data_ptr(), useBlockTable.data(),
                        useBlockTable.size() * sizeof(int));
        }
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
    for (size_t i = 0; i < oneRunBsLen; i++) {
        std::string oneReqID = taskInfo.reqIdList[i];
        int oneTopkLen = taskInfo.topkLenList[i];
        if (oneTopkLen <= 0) { continue; }
        mLogger.log(LogLevel::DEBUG, "IS_UCM_BUFFER_OPEN is: %d\n", IS_UCM_BUFFER_OPEN);
        UCMTensorMerge(layerID, streamIdx, hbmSlotIndex, oneReqID, static_cast<uint32_t>(i));
    }
    if (FinishKVPack(streamIdx, epoch) != 0) { KVPackAbort(mKVPackControl); }
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    timing.cpuMove +=
        std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    CopyPrefetchToH2DBuffers(streamIdx, layerID);
    ValidateAndPublishHbmSlotIndex(streamIdx, layerID, hbmSlotIndex);
    auto memcpyEnd = std::chrono::high_resolution_clock::now();
    timing.memcpy +=
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
    std::lock_guard<std::mutex> lock(mUpdateReqMutex);
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t runBsLen = taskInfo.reqIdList.size();
    allNeedLoadBlock.clear();
    allMissIdxs.clear();
    for (size_t i = 0; i < runBsLen; i++) {
        if (mDocsTables.find(taskInfo.reqIdList[i]) == mDocsTables.end() ||
            taskInfo.topkLenList[i] <= 0) {
            mLogger.log(LogLevel::ERROR,
                        "Decode step: %u, |KVCache Prefetch| topk len is zero: %d\n", mDecodeStep,
                        taskInfo.topkLenList[i]);
            continue;
        }
        allMissIdxs.insert({taskInfo.reqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        allNeedLoadBlock.insert({taskInfo.reqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        RunOneBsPrefetch(taskInfo.reqIdList[i], taskInfo.topkLenList[i], i,
                         taskInfo.indexerTopkList[i]);
    }
    UCMIndexerLoadAsync(threadInfo.context_, threadInfo.stream_);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish indexer load cost: %lu\n", mDecodeStep,
                duration.count());
    return 0;
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

void GSAPrefetchEngineC::RunOneBsPrefetch(std::string reqID, int topkLen, int topkIndex,
                                          const torch::Tensor& topkList)
{
    if (!mOpenIndexerOffload) { return; }
#pragma omp parallel for num_threads(4) proc_bind(master)
    for (int i = 0; i < mLayerNum; i++) {
        std::unordered_set<int> hitBlocks;
        std::map<int, int> hitBlocksIdx;
        std::vector<int> missIdxs;
        PrefetchReqInfo oneBsInfo;
        oneBsInfo.topkLen = topkLen;
        oneBsInfo.reqID = reqID;
        oneBsInfo.layerID = i;
        int* topk_ptr = topkList[i].data_ptr<int>();
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

    std::ostringstream oss;
    oss << "Decode step: " << mDecodeStep << " Rank: " << mRank << " reqID: " << reqID
        << " layerID: " << layerID << " topk len: " << topkLen << " topk: ";
    for (int j = 0; j < topkLen; j++) {
        int32_t item = topkList[j];
        oss << item << " ";
        if (mDocsTables[reqID][layerID].find(item) != mDocsTables[reqID][layerID].end()) {
            int blockID = mDocsTables[reqID][layerID][item];
            hitBlocks.insert(blockID);
            hitBlocksIdx.insert(std::make_pair(item, blockID));
        } else {
            missIdxs.push_back(item);
        }
    }
    oss << "------\n";
    mLogger.log(LogLevel::DEBUG, oss.str().c_str());
    oss.str("");
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

    auto hashIt = mAllBlocksHash.find(reqID);
    if (hashIt == mAllBlocksHash.end() || hashIt->second.empty()) {
        mLogger.log(LogLevel::ERROR, "[ERROR] IndexerInitLoad no hash, reqID=%s\n", reqID.c_str());
        return;
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
    auto* indexerBaseAddr = reinterpret_cast<std::uint8_t*>(mIndexerPtr[layerID]);
    if (indexerBaseAddr == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad indexer base address is nullptr, layerID=%d\n",
                    layerID);
        return false;
    }
    const auto& hashVec = mAllBlocksHash.at(reqID);
    UC::Detail::TaskDesc task;
    task.reserve(number);
    task.brief = "UCMIndexerInitLoadToDevice";
    size_t shardIndex = static_cast<size_t>(layerID);
    for (int i = 0; i < number; i++) {
        int deviceBlockId = blockTable[layerID][i];
        int blockIdx = slabIdx[layerID][i];
        if (deviceBlockId < 0 || deviceBlockId >= mDeviceIndexerLen) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] IndexerInitLoad deviceBlockId out of range, "
                        "reqID=%s layerID=%d deviceBlockId=%d mDeviceIndexerLen=%d\n",
                        reqID.c_str(), layerID, deviceBlockId, mDeviceIndexerLen);
            continue;
        }
        auto* indexerAddr =
            indexerBaseAddr + static_cast<size_t>(deviceBlockId) * mIndexerSizeBytesBlock;
        if (blockIdx >= static_cast<int>(hashVec.size())) {
            if (!LoadNonUcmIndexerBlock(reqID, layerID, blockIdx, deviceBlockId, indexerAddr)) {
                return false;
            }
            continue;
        }
        if (blockIdx < 0 || static_cast<size_t>(blockIdx) >= hashVec.size()) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] IndexerInitLoad blockIdx out of range, "
                        "reqID=%s layerID=%d blockIdx=%d hashNum=%zu\n",
                        reqID.c_str(), layerID, blockIdx, hashVec.size());
            continue;
        }
        task.emplace_back(UC::Detail::Shard{
            hashVec[blockIdx], shardIndex, std::vector<void*>{nullptr, nullptr, indexerAddr}
        });
    }
    return SubmitIndexerInitLoadTask(task, reqID, layerID);
}

bool GSAPrefetchEngineC::SubmitIndexerInitLoadTask(UC::Detail::TaskDesc& task,
                                                   const std::string& reqID, int layerID)
{
    if (task.empty()) { return true; }
    const size_t loadNumber = task.size();
    auto status = mStoreV1_->LoadToDevice(std::move(task));
    if (status.Failure()) {
        const std::string error = status.ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] IndexerInitLoad LoadToDevice failed, "
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

    UC::Detail::TaskDesc task;
    task.reserve(number);
    task.brief = "UCMIndexerLoadToDevice";
    if (!BuildIndexerLoadTask(reqID, layerID, indexerBaseAddr, hbmBlockIDs, missBlockIdxs, hashVec,
                              task, totalNumber)) {
        return false;
    }

    if (task.empty()) { return true; }
    return LoadIndexerTask(reqID, layerID, task, totalNumber);
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
        task.emplace_back(UC::Detail::Shard{
            hashVec[missBlockIdx], shardIndex, std::vector<void*>{nullptr, nullptr, indexerAddr}
        });
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
    const size_t loadNumber = task.size();
    auto status = mStoreV1_->LoadToDevice(std::move(task));
    if (status.Failure()) {
        const std::string error = status.ToString();
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] UCMIndexerLoadAsync LoadToDevice failed, "
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
    TaskInfo inputInfo(totalTaskNum, -1, topkLists, topkLensInput, bsIndexInput, reqIDList);
    mIsPrefetchRunning.store(true, std::memory_order_release);
    mIsIndexerOffload = isIndexerOffload;
    for (uint32_t i = 0; i < reqIDList.size(); i++) {
        if (mOpenIndexerOffload && mIsIndexerOffload[i]) {
            mRealTimeIdx[reqIDList[i]].clear();
            mRealTimeIdx[reqIDList[i]] = mUseBlockTableIdx[reqIDList[i]];
            mSeqLen[reqIDList[i]] = seqLens[i];
            mLogger.log(LogLevel::DEBUG,
                        "Decode step: %u, "
                        "|KVCache Prefetch| SubmitTask mRealTimeIdx: %u, mUseBlockTableIdx: %u for "
                        "req_id: %s\n",
                        mDecodeStep, mRealTimeIdx[reqIDList[i]][0].size(),
                        mUseBlockTableIdx[reqIDList[i]][0].size(), reqIDList[i].c_str());
        }
    }
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

void GSAPrefetchEngineC::SubmitTaskIndxer(std::vector<int>& topkLensInput,
                                          std::vector<std::string>& reqIDList,
                                          std::vector<torch::Tensor>& topkLists, uint32_t stepTime)
{
    if (!mOpenIndexerOffload) { return; }
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| start SubmitTaskIndxer batch size: %lu\n",
                mDecodeStep, reqIDList.size());
    auto start = std::chrono::high_resolution_clock::now();
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
        "Decode step: %u, |KVCache Prefetch| finish SubmitTaskIndxer batch size: %lu, cost: %lu\n",
        mDecodeStep, reqIDList.size(), durationLoadPre.count());
}

void GSAPrefetchEngineC::SetTopkIndies(std::string reqID, torch::Tensor& topIndices)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start SetTopkIndies reqID: %s\n", mDecodeStep,
                reqID.c_str());
    mAllTopkIndicesTensor[reqID] = topIndices;
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End SetTopkIndies reqID: %s\n", mDecodeStep,
                reqID.c_str());
}

void GSAPrefetchEngineC::CopyPreSwapTokens(uintptr_t useTokenBase, uintptr_t spaTokenBase,
                                           const std::vector<int>& swapRange,
                                           const std::vector<int>& needAppend, int sparseLen,
                                           int spaMaxBs, int mtpToken, int changeLen)
{
#pragma omp parallel for num_threads(8) proc_bind(master)
    for (int layerID = 0; layerID < mTopkIndexerNum; layerID++) {
        void* usePtr =
            reinterpret_cast<void*>(useTokenBase + sparseLen * layerID * sizeof(int32_t));
        for (int index = swapRange[0]; index < swapRange[1]; index++) {
            void* spaCpuPtr = reinterpret_cast<void*>(
                spaTokenBase + (size_t)(layerID * spaMaxBs + index) * sparseLen * sizeof(int32_t));
            std::memcpy(spaCpuPtr, usePtr, sparseLen * sizeof(int32_t));
            std::memcpy(reinterpret_cast<uint32_t*>(spaCpuPtr) + sparseLen - changeLen + mtpToken -
                            1 - index % mtpToken,
                        needAppend.data(),
                        (changeLen - mtpToken + 1 + index % mtpToken) * sizeof(int));
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
        auto start_max = std::chrono::high_resolution_clock::now();
        const int32_t* data = reinterpret_cast<const int32_t*>(useTokenPtrs[bs]);
        int maxID = *std::max_element(data, data + sparseLen * mTopkIndexerNum);
        int start = maxID + 1;
        int end = seqLens[bs] + mtpToken - 1;
        int change_len = end - start;
        std::vector<int> needAppend;
        if (change_len <= mtpToken - 1) {
            mLogger.log(LogLevel::ERROR,
                        "[ERROR] PreSwapTokens change len < 0 !!!!!! start %d, end %d\n", start,
                        end);
            change_len = mtpToken;
        }
        mLogger.log(
            LogLevel::DEBUG,
            "Decode step: %u, |KVCache Prefetch| Start PreSwapTokens, change_len: %d, maxID: %d\n",
            mDecodeStep, change_len, maxID);
        needAppend.resize(change_len);
        std::iota(needAppend.begin(), needAppend.end(), start);
        auto max_end = std::chrono::high_resolution_clock::now();
        max_tokens_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(max_end - start_max).count();
        CopyPreSwapTokens(useTokenPtrs[bs], spaTokensPtr[0], swapIndex[bs], needAppend, sparseLen,
                          spaMaxBs, mtpToken, change_len);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_copy_time +=
            std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - max_end).count();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End PreSwapTokens, "
                "bs: %lu total time: %lu max_tokens_time: %lu cpu_copy_time: %lu\n",
                mDecodeStep, useTokenPtrs.size(), duration.count(), max_tokens_time, cpu_copy_time);
}

void GSAPrefetchEngineC::DelBlocksMap(std::string reqID)
{
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| Start DelBlocksMap, reqID: %s\n", mDecodeStep,
                reqID.c_str());
    mDelReqMutex.lock();
    mDelSeqIds.insert(reqID);
    if (!GetPrefetchStatus()) { DelReqIDRun(); }
    mDelReqMutex.unlock();
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
            mBlocksMap.erase(*it);
            mDocsTables.erase(*it);
            mIndexerSlot.erase(*it);
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

void GSAPrefetchEngineC::SetModelRunningStatus(bool flag) { mStopPrefetch = flag; }

std::vector<int> GSAPrefetchEngineC::GetTokenBlockMap(int layerID, std::string reqID)
{
    return mHBMMangers[reqID][layerID]->mGlobalToSlot;
}

std::vector<int> GSAPrefetchEngineC::GetBlockTokenMap(int layerID, std::string reqID)
{
    return mHBMMangers[reqID][layerID]->mSlotToGlobal;
}

std::map<int, int> GSAPrefetchEngineC::GetIndexerTokensMap(int layerID, std::string reqID)
{
    return mDocsTables[reqID][layerID];
}

std::map<int, int> GSAPrefetchEngineC::GetIndexerBlockMap(int layerID, std::string reqID)
{
    return mBlocksMap[reqID][layerID];
}

std::vector<int> GSAPrefetchEngineC::GetReqUseBlockTable(std::string reqID)
{
    return mReqUseBlockTable[reqID];
}

int GSAPrefetchEngineC::GetSparseSeqLen(std::string reqID) { return mSparseSeqLen[reqID]; }
}  // namespace ucmprefetch
