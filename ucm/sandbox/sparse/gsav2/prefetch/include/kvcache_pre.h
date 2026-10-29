/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef ATB_KV_CACHE_PRE_H
#define ATB_KV_CACHE_PRE_H
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <queue>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "kvcache_hbm_manger.h"
#include "kvcache_pre_detail.h"
#include "kvcache_task_manger.h"
#include "ucmstore_v1.h"

namespace py = pybind11;

namespace ucmprefetch {
using BlockId = std::array<std::byte, 16>; /* 16-byte block hash */
typedef struct {
    int topkLen;
    std::string reqID;
    int layerID;
    int topkIndex;
    int bsIndex;
} PrefetchReqInfo;

void MultiBSThreadFun(const TaskInfo& taskInfo, ThreadInfo& threadInfo);
void MultiBSThreadFunIndexer(const TaskInfo& taskInfo, ThreadInfo& threadInfo);

float average(const std::vector<float>& vec);
float average2d(const std::vector<std::vector<float>>& matrix);

class __attribute__((visibility("hidden"))) GSAPrefetchEngineC {
    friend struct UCMMergeHostAddress;

public:
    std::mutex mMutexUseToken;
    // Serializes access to the per-request maps (mReqUseBlockTable, mDocsTables,
    // mBlocksMap, mIndexerSlot, ...) that are touched concurrently by
    // DelBlocksMap (Python thread), SetIndexerMap (Python thread) and
    // IndexerLoadProcess (indexer worker thread).
    std::mutex mUpdateReqMutex;
    bool mStopPrefetch = false;
    std::atomic<bool> mIsPrefetchRunning{false};
    std::atomic<bool> mIsIndexerLoadRunning{false};
    int mRank = -1;
    uint32_t mSpaTopkLen = 5120;
    std::vector<std::vector<float>> mAvgHitRate;
    // task manager class
    std::unique_ptr<TaskManager> mTaskManger;
    std::unique_ptr<TaskManager> mIndexerTaskManger;
    std::vector<int> mPrefetchLen;
    std::vector<bool> mIsIndexerOffload;
    std::vector<void*> mDumpKBuffer;
    std::vector<void*> mDumpVBuffer;
    std::map<std::string, std::vector<std::vector<int>>> mUseBlockTableIdx;
    std::map<std::string, std::vector<std::vector<int>>> mRealTimeIdx;
    std::map<std::string, std::vector<std::vector<int>>> mPrefetchBlockTableIdx;
    std::map<std::string, int> mSeqLen;

    /* UCM store for Host-only KV load */
    UC::StoreV1* mStoreV1_{nullptr};
    bool mISSPAC8{false};

public:
    ~GSAPrefetchEngineC();

    GSAPrefetchEngineC(std::vector<int>& modelInfo, std::vector<int>& sizeinfo, bool isLog,
                       int rank, int threadNum, std::vector<int> cpuToBind,
                       std::vector<int> topkLayerMap, std::string logFileName);

    // Initialize/Update HBM manager mapping
    // Hash map setup
    void SetBlocksMap(const std::string& reqID, const std::vector<BlockId>& blocksHash, int maxIdx,
                      const std::vector<int>& blockIdxs, const std::vector<uintptr_t>& kAddrs,
                      const std::vector<uintptr_t>& vAddrs);

    void LastChunkMove(int layerID, const std::string& reqID, const std::vector<int>& moveMap);

    void SetTokensMap(int layerID, const std::string& reqID, torch::Tensor& qTopkTensor,
                      const std::vector<int>& remainBlockTable, int maxIdx,
                      const std::vector<int>& slots, bool isNeedInit);

    // kvcache dump
    void SetKvTensor(torch::Tensor kCachesTensor, torch::Tensor vCachesTensor,
                     torch::Tensor lastChunkKBuffer, torch::Tensor lastChunkvBuffer, int layerID);

    void SetKvTensorIndex(int layerID, torch::Tensor indexCache);

    void SetKvBuffer(torch::Tensor kBufferHost, torch::Tensor vBufferHost,
                     torch::Tensor kBufferDevice, torch::Tensor vBufferDevice,
                     torch::Tensor loadIndexHost, torch::Tensor loadIndexDevice);

    // ============================== 新增的一些函数 ==============================
    // UC::Detail::BlockId ToBlockId(const std::string& hash);
    void UCMTensorMerge(int layerID, int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
                        const std::string& reqID, uint32_t requestIndex);
    /* Host-only KV load from UCM */
    int LoadMissKVFromUCMHost(int reqID, int layerID, const std::vector<int>& missIdxs,
                              void* kBufferOffset, void* vBufferOffset);
    void SetStoreV1(UC::StoreV1* store) { mStoreV1_ = store; }

    void SetStoreV1Raw(uintptr_t storePtr) { mStoreV1_ = reinterpret_cast<UC::StoreV1*>(storePtr); }

    uintptr_t GetStoreV1Raw() const { return reinterpret_cast<uintptr_t>(mStoreV1_); }

    uintptr_t GetSelfRaw() const { return reinterpret_cast<uintptr_t>(this); }

    void KVCacheInplaceScatter(std::string type, int layerID, aclrtStream stream,
                               aclrtContext context);

    void DealLessPrefetchBlock(std::unordered_set<int>& hitBlocks, std::map<int, int>& hitBlocksIdx,
                               std::string reqID, int layerID, int topkLen);

    // Trigger deletion
    void DelBlocksMap(std::string reqID);

    void DelReqIDRun();

    // Task assignment and execution
    void SubmitTask(std::vector<int>& topkLensInput, std::vector<int>& bsIndexInput,
                    std::vector<std::string>& reqIDList, torch::Tensor& topkLists,
                    std::vector<bool>& isIndexerOffload, std::vector<int>& seqLens,
                    uint32_t stepTime);

    int CallPrefetchProcess(const TaskInfo& taskInfo, ThreadInfo& threadInfo);

    // set output
    void SetTopkIndices(std::string reqID, torch::Tensor& topIndices);

    void PreSwapTokens(std::vector<uintptr_t>& useTokenPtrs, std::vector<int32_t>& seqLens,
                       std::vector<std::vector<int>>& swapIndex,
                       std::vector<uintptr_t>& spaTokensPtr, std::vector<std::string>& reqIDs,
                       int sparseLen, int spaMaxBs, int mtpToken);

    void TopkExpend(std::vector<int>& topkLensInput, std::string reqID);

    // Prefetch running status
    bool GetPrefetchStatus();

    void SetModelRunningStatus(bool flag);

    std::vector<int> GetTokenBlockMap(int layerID, std::string reqID);

    std::vector<int> GetBlockTokenMap(int layerID, std::string reqID);

    std::map<int, std::pair<int, int>> GetPrefetchToken(int layerID, std::string reqID);

    std::vector<int> GetReqUseBlockTable(std::string reqID);

    int GetSparseSeqLen(std::string reqID);

    // indexer offload
    void SetIndexerMap(std::string reqID, std::vector<int>& blockTableList,
                       std::vector<std::vector<int>>& remainIdx,
                       std::vector<std::vector<int>>& prefetchIdx, std::vector<int>& blockSlot,
                       torch::Tensor& dsaSBT);

    void UpdateIndexerMap(std::string reqID, int idx, int blockID, std::string type);

    void AddIndexerMap();

    void DeleteIndexerMap();

    int IndexerLoadProcess(const TaskInfo& taskInfo, ThreadInfo& threadInfo);

    void RunOneBsPrefetch(std::string reqID, int topkLen, int topkIndex,
                          const torch::Tensor& topkList);

    void GetHitAndMissBlock(PrefetchReqInfo oneBsInfo, std::vector<int>& topkList,
                            std::unordered_set<int>& hitBlocks, std::map<int, int>& hitBlocksIdx,
                            std::vector<int>& missIdxs);

    void RunPrefetchH2D(PrefetchReqInfo oneBsInfo, std::unordered_set<int>& hitBlocks,
                        std::map<int, int>& hitBlocksIdx, std::vector<int>& missIdxs);

    void LoadKVToHBM(std::vector<int> loadNPUBlockIDs, std::vector<int> missIdxs, int layerID,
                     std::string reqID);

    void UCMIndexerLoadAsync(void* context, void* stream);

    void SubmitTaskIndexer(std::vector<int>& topkLensInput, std::vector<std::string>& reqIDList,
                           std::vector<torch::Tensor>& topkLists, uint32_t stepTime);

    void IndexerInitLoad(std::vector<std::vector<int>>& blockTable,
                         std::vector<std::vector<int>>& slabIdx, std::string reqID);

    void TokenIndexMapInter(std::vector<int>& oneTopkList, int bsIndex, int layerID, int topkLen,
                            std::string reqID);

    bool GetIndexerStatus();

    std::map<int, int> GetIndexerBlockMap(int layerID, std::string reqID);

    std::map<int, int> GetIndexerTokensMap(int layerID, std::string reqID);

    void SetLastIndexerBlock(const std::string& reqID, const std::vector<int>& blockIdxs,
                             const std::vector<uintptr_t>& indexerAddrs);

private:
#ifdef KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC
public:
#endif
    // HBM manager mapping
    // tokens topks
    // mTokenBlockMaps  LN[{reqID, [token-idx]}]
    // mBlockTokenMaps  LN[{reqID, {blockID, [token-idx]}}]
    // mPrefetchTokens LN[{reqID, {token-idx, (blockID, slot)}}]
    std::map<std::string, std::vector<int>> mAllReqIdSlots;
    std::map<std::string, std::vector<int>> mReqUseBlockTable;
    std::map<std::string, int> mPromptLen;
    std::map<std::string, int> mSparseSeqLen;
    std::map<std::string, std::vector<std::unique_ptr<HBMManager>>> mHBMMangers;

    // kv_load_info
    std::map<std::string, std::vector<std::vector<int>>> mMissGlobalToken;
    std::map<std::string, std::vector<std::vector<int>>> mLoadSlot;

    // Shared memory for cross-rank prefetch result synchronization
    void* mPrefetchResultShm;
    size_t mPrefetchResultShmSize;
    std::atomic<uint32_t> mPrefetchResultVersion;
    bool mIsPrefetchResultShmCreator;
    std::string mPrefetchResultShmName;
    SharedKVPackHeader* mKVPackControl{nullptr};
    size_t mKVPackControlSize{0};
    std::string mKVPackControlName;
    bool mIsKVPackControlCreator{false};
    std::vector<uint64_t> mKVPackPreviousEpoch;
    int mKVPackDpRank{-1};
    std::vector<uint64_t> mSharedKBufferHostPtr;
    std::vector<uint64_t> mSharedVBufferHostPtr;
    int mMaxLoadNum = 2048;

    // kv_load_info
    std::map<std::string, std::vector<std::map<int, int>>> mDocsTables;
    std::map<std::string, std::vector<std::map<int, int>>> mBlocksMap;
    std::map<std::string, std::vector<std::vector<int>>> mPrefetchIdx;
    std::map<std::string, std::vector<std::vector<int>>> allNeedLoadBlock;
    std::map<std::string, std::vector<std::vector<int>>> allMissIdxs;
    std::map<std::string, std::vector<int>> mIndexerSlot;
    std::map<std::string, std::vector<std::pair<int, int>>> mIndexerAdd;
    std::map<std::string, std::vector<int>> mIndexerDel;

    //  output tensor
    std::map<std::string, torch::Tensor> mAllTopkIndicesTensor;
    std::map<std::string, torch::Tensor> mIndexerLoadResultT;

    // delete info
    std::unordered_set<std::string> mDelSeqIds;

    // hash id
    std::map<std::string, std::vector<BlockId>> mAllBlocksHash;

    // kv cache tensor and ptr
    std::vector<uint64_t> mKcachePtr;
    std::vector<uint64_t> mVcachePtr;
    std::vector<torch::Tensor> mKcacheTensor;
    std::vector<torch::Tensor> mVcacheTensor;
    torch::Tensor mKBufferHost;
    torch::Tensor mVBufferHost;
    torch::Tensor mKBufferDevice;
    torch::Tensor mVBufferDevice;
    torch::Tensor mLoadIndexHost;
    torch::Tensor mLoadIndexDevice;
    std::vector<uint64_t> mKBufferHostPtr;
    std::vector<uint64_t> mVBufferHostPtr;
    std::vector<std::vector<void*>> mH2DPtrInfo;
    std::vector<int> mLoadNum;
    std::vector<torch::Tensor> mLastCHunkKBuffer;
    std::vector<torch::Tensor> mLastCHunkVBuffer;
    std::vector<uint64_t> mLastCHunkKPtr;
    std::vector<uint64_t> mLastCHunkVPtr;
    std::vector<torch::Tensor> mIndexerTensor;
    std::vector<uint64_t> mIndexerPtr;
    uint32_t mIndexerSizeBytesBlock = 0;
    int mLocalWindows = 5;

    int mSparseLen = 0;
    void* mStream;
    void* mContext;
    bool mOpenIndexerOffload = false;

    // ops value
    std::vector<aclTensor*> mKCacheAcl;
    std::vector<aclTensor*> mVCacheAcl;
    std::vector<void*> mKWorkspaceAddr;
    std::vector<void*> mVWorkspaceAddr;
    std::vector<uint64_t> mKMaxWorkspaceSize;
    std::vector<uint64_t> mVMaxWorkspaceSize;
    std::vector<aclTensor*> mKDstIndexAcl;
    std::vector<aclTensor*> mKHBMBufferAcl;
    std::vector<aclTensor*> mVDstIndexAcl;
    std::vector<aclTensor*> mVHBMBufferAcl;

    // model information
    int mLayerNum;
    uint32_t mTensorElemSize = 0;
    uint32_t mKVSizeBytesBlock = 0;
    uint32_t mKVSizeBytesBlockRe = 0;
    uint32_t mKVSizeBytesToken = 0;
    uint32_t mKVSizeBytesTokenRe = 0;
    int mMaxLen = 131072;
    int mBlockSize = 128;
    int mKDim = 512;
    int mVDim = 64;
    int mNumHead = 1;
    std::vector<int> mTopkLayerMap;
    int mTopkIndexerNum = 21;
    size_t mTpSize = 16;
    size_t mMaxBs = 64;

    // Runtime information
    uint32_t mDecodeStep = 0;
    int totalTaskNum = 0;
    int mThreadNum = 16;
    int mStreamNum = 4;

    // log information
    Logger mLogger;
    bool mIsLog = false;

    // ptr bound
    uint32_t mIndexerSparseLen = 0;
    int mKVHostBlockLen = 0;
    int mKVHBMBlockLen = 0;
    int mLastChunkNum = 0;
    int mKVBufferLen = 0;
    int mLoadIndexBufferLen = 0;
    int mDeviceIndexerLen = 0;

    std::unordered_map<std::string, std::unordered_map<int, LastKVBlockAddress>>
        mLastKVBlockAddress;

private:
#ifdef KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC
public:
#endif
    // Abnormality determination
    void CheckInputIndex(uint32_t maxLen, uint32_t index);
    int InitKVPackControl(int dpRank);
    void CleanupKVPackControl();
    int BeginKVPackSlot(int streamIdx, uint64_t epoch);
    int FinishKVPack(int streamIdx, uint64_t epoch);
    void MarkKVH2DDone(int streamIdx, uint64_t epoch);

    // KVPack 初始化分解辅助方法（huge_method 序号47）
    const char* WaitKVPackControlInitialized(const std::chrono::steady_clock::time_point& deadline);
    const char* CreateKVPackControlFile(size_t bytes, int& fd);
    const char* OpenKVPackControlFollower(const std::chrono::steady_clock::time_point& deadline,
                                          size_t bytes, int& fd);
    void InitKVPackSharedStates(size_t stateCount);
    void SetupKVPackSharedBuffers(size_t controlBytes, size_t streamCount, size_t kBytes,
                                  size_t vBytes);

    // CopyUCMMergeTokensParallel 分解辅助（huge_method 序号48）
    void BuildTokenCopyTasks(const std::vector<UCMMergeBlockContext>& blockContexts,
                             std::vector<TokenCopyTask>& tokenCopyTasks,
                             const std::vector<int>& missGlobalIds,
                             const std::vector<int>& missSlots, const std::vector<int>& blockTable,
                             size_t kTokenBytes, size_t vTokenBytes, size_t previousLoadNum);

    void InitModelInfo(std::vector<int>& modelInfo, bool isLog, std::vector<int> cpuToBind,
                       std::vector<int> topkLayerMap, int rank, int threadNum);
    void InitBufferInfo(std::vector<int>& sizeinfo);
    void InitAclRuntime();
    void InitPrefetchResultShm(std::vector<int>& modelInfo);
    void CreatePrefetchResultShm();
    void AttachPrefetchResultShm();

    bool LoadIndexerLayer(int layerID, size_t& totalNumber);
    bool LoadIndexerRequest(const std::string& reqID, const std::vector<int>& hbmBlockIDs,
                            int layerID, std::uint8_t* indexerBaseAddr, size_t& totalNumber);
    bool BuildIndexerLoadTask(const std::string& reqID, int layerID, std::uint8_t* indexerBaseAddr,
                              const std::vector<int>& hbmBlockIDs,
                              const std::vector<int>& missBlockIdxs,
                              const std::vector<BlockId>& hashVec, UC::Detail::TaskDesc& task,
                              size_t& totalNumber);
    bool CopyNonUCMIndexerBlock(const std::string& reqID, int layerID, int missBlockIdx,
                                int hbmBlockID, void* indexerAddr, size_t& totalNumber);
    bool LoadIndexerTask(const std::string& reqID, int layerID, UC::Detail::TaskDesc& task,
                         size_t& totalNumber);

    // 从KVCacheInplaceScatter拆分出来
    void PrepareScatterAclTensor(const std::string& type, int streamIdx, torch::Tensor& dstIndexND,
                                 torch::Tensor& hbmBufferND, aclTensor*& dstIndexTensor,
                                 aclTensor*& dstHBMBufferTensor);

    void* GetScatterWorkspace(const std::string& type, int layerID, int streamIdx,
                              uint64_t workspaceSize);

    void RunInplaceScatter(const std::string& type, int layerID, int streamIdx,
                           aclTensor* dstTensor, aclTensor* dstIndexTensor,
                           aclTensor* dstHBMBufferTensor, aclrtStream stream);

    // 0923 开始
    // UCMTensorMerge
    bool LoadUCMMergeHostAddress(UC::Detail::TaskDesc task, UCMMergeHostAddress& address,
                                 int layerID, const std::string& reqID);

    void ReleaseUCMMergeHostAddress(UCMMergeHostAddress& address);

    bool MemcpyUCMMergeHostAddress(const UCMMergeHostAddress& address,
                                   std::vector<UCMMergeBlockContext>& blockContexts, int layerID,
                                   int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
                                   const std::string& reqID, size_t previousLoadNum);
    size_t FillFailedUCMMerge(const std::vector<UCMMergeBlockContext>& blockContexts, int layerID,
                              int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
                              const std::string& reqID, size_t previousLoadNum);

    // UCMTensorMerge 分解辅助方法（huge_method 序号51）
    bool ValidateUCMMergeRequest(int layerID, const std::string& reqID);
    void BuildUCMMergeBlockContexts(int layerID, const std::string& reqID, uint32_t requestIndex,
                                    std::vector<UCMMergeBlockContext>& blockContexts,
                                    UC::Detail::TaskDesc& task,
                                    std::map<int, std::vector<size_t>>& blockIdxToTokenIdx);
    // IndexerInitLoad
    bool ProcessIndexerInitLoadLayer(std::vector<std::vector<int>>& blockTable,
                                     std::vector<std::vector<int>>& slabIdx,
                                     const std::string& reqID, int layerID, int number);

    bool LoadNonUcmIndexerBlock(const std::string& reqID, int layerID, int blockIdx,
                                int deviceBlockId, std::uint8_t* indexerAddr);

    // LastChunkMove
    bool PrepareLastChunkSources(int layerID, const std::string& reqID,
                                 const std::vector<int>& moveMap,
                                 const std::vector<int>& blockTable,
                                 std::vector<LastChunkSourceBlock>& sourceBlocks,
                                 UC::Detail::TaskDesc& task, std::vector<int>& loadedBlockIndices);

    bool LoadLastChunkSources(int layerID, const std::string& reqID, UC::Detail::TaskDesc& task,
                              const std::vector<int>& loadedBlockIndices,
                              std::vector<LastChunkSourceBlock>& sourceBlocks,
                              UC::Detail::TaskHandle& hostTaskId, bool& hasHostTask);

    void CopyLastChunkTokens(int layerID, const std::string& reqID, const std::vector<int>& moveMap,
                             const std::vector<LastChunkSourceBlock>& sourceBlocks,
                             UC::Detail::TaskHandle hostTaskId, bool hasHostTask);

    // CallPrefetchProcess
    void PreparePrefetchResults(const TaskInfo& taskInfo);

    void ProcessPrefetchRequest(const TaskInfo& taskInfo, size_t i, int layerID,
                                const PrefetchShmLayout& layout);

    void WritePrefetchResult(const std::string& oneReqID, size_t i, int layerID,
                             const std::vector<int>& useBlockTable,
                             const PrefetchShmLayout& layout);

    void ReadPrefetchResults(const TaskInfo& taskInfo, const PrefetchShmLayout& layout);

    void MergeAndCopyPrefetchLayer(const TaskInfo& taskInfo, ThreadInfo& threadInfo, int layerID,
                                   PrefetchTiming& timing);

    // MergeAndCopyPrefetchLayer 分解辅助方法（huge_method 序号50）
    void CopyPrefetchToH2DBuffers(int streamIdx, int layerID);
    void ValidateAndPublishHbmSlotIndex(int streamIdx, int layerID,
                                        std::vector<uint64_t>& hbmSlotIndex);

    void TransferAndScatterPrefetchLayer(ThreadInfo& threadInfo, int layerID, int streamIdx,
                                         PrefetchTiming& timing, PrefetchTimePoint syncEnd);

    void CopyPreSwapTokens(uintptr_t useTokenBase, uintptr_t spaTokenBase,
                           const std::vector<int>& swapRange, const std::vector<int>& needAppend,
                           int sparseLen, int spaMaxBs, int mtpToken, int changeLen);

    bool SubmitIndexerInitLoadTask(UC::Detail::TaskDesc& task, const std::string& reqID,
                                   int layerID);

    void FinishOneLayerPrefetch(const std::string& reqID, int layerID,
                                std::map<int, int>& hitBlocksIdx, std::vector<int>& missIdxs);

    bool PrepareUCMMergeSources(const UCMMergeHostAddress& address,
                                std::vector<UCMMergeBlockContext>& blockContexts, int layerID,
                                const std::string& reqID);

    void CopyUCMMergeTokensSerial(const std::vector<UCMMergeBlockContext>& blockContexts,
                                  int layerID, int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
                                  const std::string& reqID, size_t previousLoadNum);

    // CopyUCMMergeTokensSerial 分解辅助（huge_depth 序号41）
    void CopySingleTokenKV(std::byte* kDstBase, std::byte* vDstBase,
                           const UCMMergeBlockContext& context, size_t tokenOffset,
                           size_t dstOffset, size_t kTokenBytes, size_t vTokenBytes);

    void CopyUCMMergeTokensParallel(const std::vector<UCMMergeBlockContext>& blockContexts,
                                    int layerID, int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
                                    const std::string& reqID, size_t previousLoadNum,
                                    int tokenThreadNum);

    bool BuildLastChunkSources(int layerID, const std::string& reqID,
                               const std::vector<int>& moveMap,
                               std::vector<LastChunkSourceBlock>& sourceBlocks,
                               UC::Detail::TaskDesc& task, std::vector<int>& loadedBlockIndices);

    void AddIndexerMapForRequest(const std::string& reqID);

    void DeleteIndexerMapForRequest(const std::string& reqID);

    void SetIndexerMapForLayer(const std::string& reqID, int layerID,
                               std::vector<int>& blockTableList,
                               std::vector<std::vector<int>>& remainIdx,
                               std::vector<std::vector<int>>& prefetchIdx,
                               std::vector<int>& blockSlot);

    void ProcessTokensMapTopk(int layerID, const std::string& reqID, torch::Tensor& qTopkTensor,
                              int maxIdx, size_t sparseSeqLen);

    void ReleasePrefetchStreamResources(int layerID);

    void InitBufferContainers();
};  //
}  // namespace ucmprefetch

#endif