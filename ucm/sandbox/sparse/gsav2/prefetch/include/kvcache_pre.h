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
#include "kvcache_hbm_manger_adapt.h"
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
public:
    std::mutex mMutexUseToken;
    // Serializes access to the per-request maps (mReqUseBlockTable, mDocsTables,
    // mBlocksMap, mUseBlockTableIdx, ...) that are touched concurrently by
    // DelBlocksMap (Python thread), SetIndexerMap (Python thread) and
    // IndexerLoadProcess (indexer worker thread).
    std::mutex mUpdateReqMutex;
    // mUpdateReqMutex 埋点：dump 由 UCM_PREFETCH_LOCK_PROFILE 门控。
    // 公开给线程函数（MultiBSThreadFun）直接访问。
    UReqLockStats mUReqLockStats[static_cast<size_t>(UReqLockSite::Count)];
    bool mUReqLockProfileEnabled = false;
    uint64_t mUReqLockDumpIntervalSteps = 100;
    uint64_t mUReqLockNextDumpStep = 0;

    UReqLockStats& UReqLockStatsAt(UReqLockSite site)
    {
        return mUReqLockStats[static_cast<size_t>(site)];
    }
    void DumpUReqLockStats();
    std::atomic<bool> mIsPrefetchRunning{false};
    std::atomic<bool> mIsIndexerLoadRunning{false};
    // 任务完成事件：worker 尾部在 mTaskDoneMutex 内翻转状态位、锁外 notify_all
    // （杜绝丢唤醒）。本锁不与任何引擎锁嵌套。
    std::mutex mTaskDoneMutex;
    std::condition_variable mPrefetchDoneCv;
    std::condition_variable mIndexerDoneCv;
    int mRank = -1;
    uint32_t mSpaTopkLen = 5120;
    std::vector<std::vector<float>> mAvgHitRate;
    // task manager class
    std::unique_ptr<TaskManager> mTaskManger;
    std::unique_ptr<TaskManager> mIndexerTaskManger;
    std::vector<bool> mIsIndexerOffload;
    std::map<std::string, std::vector<std::vector<int>>> mUseBlockTableIdx;
    std::map<std::string, std::vector<std::vector<int>>> mRealTimeIdx;
    std::map<std::string, std::vector<std::vector<int>>> mPrefetchBlockTableIdx;
    std::map<std::string, int> mSeqLen;

    /* UCM store for Host-only KV load */
    UC::StoreV1* mStoreV1_{nullptr};
    bool mISSPAC8{false};
    size_t mStoreTensorCount{kStoreTensorCount};
    Logger mLogger;

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

    // SetTokensMap 分解辅助（持有方调用，内部判别式访问 per-request 状态）
    void ResetReqTokenState(const std::string& reqID, const std::vector<int>& remainBlockTable,
                            int maxIdx, const std::vector<int>& slots);

    void RunLastChunkMoveGuarded(int layerID, const std::string& reqID, int curTopkIndex);

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
    void SetStoreV1(UC::StoreV1* store) { mStoreV1_ = store; }

    void SetStoreV1Raw(uintptr_t storePtr) { mStoreV1_ = reinterpret_cast<UC::StoreV1*>(storePtr); }

    uintptr_t GetStoreV1Raw() const { return reinterpret_cast<uintptr_t>(mStoreV1_); }

    void SetStoreTensorCount(size_t count)
    {
        if (count > 0) { mStoreTensorCount = count; }
    }

    size_t GetStoreTensorCount() { return mStoreTensorCount; }

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

    /* 配置 HBM 驱逐模式（Python 入口）：mode 1 = Decay（衰减驱逐 + Budget5120
     * 分层窗口：indexer layer<=18 → 4096，19..62 → 5120，>=63 → 4096，clamp 到
     * [topkSize, mSpaTopkLen]）；其他值 = Lru（默认，与原版逐字节一致）。
     * 引擎级开关：对调用后 set_tokens_map 重建的所有管理器生效。 */
    void SetHbmEvictMode(int mode);

    int GetHbmEvictMode() const { return mHbmEvictMode; }

    void PreSwapTokens(std::vector<uintptr_t>& useTokenPtrs, std::vector<int32_t>& seqLens,
                       std::vector<std::vector<int>>& swapIndex,
                       std::vector<uintptr_t>& spaTokensPtr, std::vector<std::string>& reqIDs,
                       int sparseLen, int spaMaxBs, int mtpToken);

    // Prefetch running status
    bool GetPrefetchStatus();

    // 任务完成事件等待（带超时）：替代 Python 侧 sleep(1ms) 轮询，唤醒延迟
    // notify 后 µs 级。绑定侧 gil_scoped_release 释放 GIL，见 pybinds.cpp。
    void WaitPrefetchIdle(uint64_t timeoutMs);
    void WaitIndexerIdle(uint64_t timeoutMs);

    // indexer offload
    void SetIndexerMap(std::string reqID, std::vector<int>& blockTableList,
                       std::vector<std::vector<int>>& remainIdx,
                       std::vector<std::vector<int>>& prefetchIdx, std::vector<int>& blockSlot,
                       torch::Tensor& dsaSBT);

    void UpdateIndexerMap(std::string reqID, int idx);

    int IndexerLoadProcess(const TaskInfo& taskInfo, ThreadInfo& threadInfo);

    int RunOneBsPrefetch(std::string reqID, int topkLen, int topkIndex,
                         const torch::Tensor& topkList);

    // RunOneBsPrefetch 分解辅助（提前识别/判别在并行域外完成）
    int AcquireIndexerTopkBase(const torch::Tensor& topkList, const std::string& reqID, int topkLen,
                               PrefetchTopkView& topkView);

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
                            const std::string& reqID);

    bool GetIndexerStatus();

    void SetLastIndexerBlock(const std::string& reqID, const std::vector<int>& blockIdxs,
                             const std::vector<uintptr_t>& indexerAddrs);

private:
#ifdef KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC
public:
#endif
    // HBM manager mapping
    // tokens topks
    std::map<std::string, std::vector<int>> mAllReqIdSlots;
    std::map<std::string, std::vector<int>> mReqUseBlockTable;
    std::map<std::string, int> mPromptLen;
    std::map<std::string, int> mSparseSeqLen;
    std::map<std::string, std::vector<std::unique_ptr<HBMManagerLayered>>> mHBMMangers;

    /* HBM 驱逐模式（Python 经 set_hbm_evict_mode 配置，引擎级生效于后续
     * set_tokens_map 重建的所有 (req,layer) 管理器）：
     *   0 = Lru + Uniform：与原 HBMManager 行为逐字节一致（默认）
     *   1 = Decay + Budget5120 分层窗口表（decay λ0.85 / steps20 / 不去重） */
    int mHbmEvictMode = 0;
    int mHbmWindowPlan = 0;

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

    uint32_t mIndexerSparseLen = 0;
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

    // LoadIndexerRequest 分解辅助（表查找+任务组装，内部持锁）
    bool ResolveIndexerLoadTask(const std::string& reqID, const std::vector<int>& hbmBlockIDs,
                                int layerID, std::uint8_t* indexerBaseAddr,
                                UC::Detail::TaskDesc& task, size_t& totalNumber);

    // 从KVCacheInplaceScatter拆分出来
    void PrepareScatterAclTensor(const std::string& type, int streamIdx, torch::Tensor& dstIndexND,
                                 torch::Tensor& hbmBufferND, aclTensor*& dstIndexTensor,
                                 aclTensor*& dstHBMBufferTensor);

    void* GetScatterWorkspace(const std::string& type, int layerID, int streamIdx,
                              uint64_t workspaceSize);

    void RunInplaceScatter(const std::string& type, int layerID, int streamIdx,
                           aclTensor* dstTensor, aclTensor* dstIndexTensor,
                           aclTensor* dstHBMBufferTensor, aclrtStream stream);

    // SubmitTask / PreSwapTokens 分解辅助
    void SyncIndexerSubmitState(std::vector<std::string>& reqIDList, std::vector<int>& seqLens);
    void SwapOneBsTokens(size_t bs, std::vector<uintptr_t>& useTokenPtrs,
                         std::vector<int32_t>& seqLens, std::vector<std::vector<int>>& swapIndex,
                         std::vector<uintptr_t>& spaTokensPtr, const std::string& reqID,
                         int sparseLen, int spaMaxBs, int mtpToken, uint64_t& maxTokensUs,
                         uint64_t& cpuCopyUs);

    // 0923 开始
    // UCMTensorMerge
    bool BuildUCMMergeLoadPlan(UCMTokenLoadPlan& plan,
                               const std::vector<UCMMergeBlockContext>& blockContexts, int layerID,
                               int streamIdx, const std::string& reqID, size_t previousLoadNum);

    bool SubmitUCMTokenLoad(const UCMTokenLoadPlan& plan, const char* stage, int layerID,
                            const std::string& reqID);

    bool SubmitUCMTokenLoadLeg(const UC::Detail::TaskDesc& task,
                               const std::vector<UC::Detail::TokenFilter>& filters,
                               const char* tensorName, const char* stage, int layerID,
                               const std::string& reqID);

    bool MemcpyUCMMergeTokens(std::vector<UCMMergeBlockContext>& blockContexts, int layerID,
                              int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
                              const std::string& reqID, size_t previousLoadNum);
    size_t FillFailedUCMMerge(const std::vector<UCMMergeBlockContext>& blockContexts, int layerID,
                              int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
                              const std::string& reqID, size_t previousLoadNum,
                              bool zeroUcmOnly = false);

    // BuildUCMMergeLoadPlan / FillFailedUCMMerge 分解辅助
    void AppendMergeBlockShard(const UCMMergeBlockContext& context,
                               const std::vector<BlockId>& hashVec,
                               const std::vector<int>& missGlobalIds, int layerID,
                               std::byte* kDstBase, std::byte* vDstBase, size_t previousLoadNum,
                               UCMTokenLoadPlan& plan);
    void BuildFallbackSlotIndices(const UCMMergeBlockContext& context, size_t dstBase,
                                  const std::vector<int>& missSlots,
                                  const std::vector<int>& blockTable,
                                  std::vector<uint64_t>& hbmSlotIndex, int layerID,
                                  const std::string& reqID);

    // UCMTensorMerge 分解辅助方法（huge_method 序号51）
    bool ValidateUCMMergeRequest(int layerID, const std::string& reqID, size_t previousLoadNum);
    void BuildUCMMergeBlockContexts(int layerID, const std::string& reqID, uint32_t requestIndex,
                                    std::vector<UCMMergeBlockContext>& blockContexts,
                                    std::map<int, std::vector<size_t>>& blockIdxToTokenIdx);

    void UCMTensorMergeBatch(const TaskInfo& taskInfo, const std::vector<uint32_t>& requestIndices,
                             int layerID, int streamIdx, std::vector<uint64_t>& hbmSlotIndex);

    void CollectUCMMergeRequests(const TaskInfo& taskInfo,
                                 const std::vector<uint32_t>& requestIndices, int layerID,
                                 int streamIdx, UCMTokenLoadPlan& plan,
                                 std::vector<UCMMergeRequestState>& requests,
                                 size_t& runningLoadNum, bool& planBuildOk);
    bool SubmitUCMMergeBatchLoad(const UCMTokenLoadPlan& plan,
                                 const std::vector<UCMMergeRequestState>& requests, int layerID,
                                 bool planOk);
    void FallbackUCMMergeRequests(const std::vector<UCMMergeRequestState>& requests, int layerID,
                                  int streamIdx, std::vector<uint64_t>& hbmSlotIndex, bool planOk,
                                  bool loadOk, const std::vector<std::string>& localFailed);
    bool ValidateMergeStagingWindow(int streamIdx);
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
                                 UCMTokenLoadPlan& plan);

    bool LoadLastChunkSources(int layerID, const std::string& reqID, const UCMTokenLoadPlan& plan);

    void CopyLastChunkTokens(int layerID, const std::string& reqID, const std::vector<int>& moveMap,
                             const std::vector<LastChunkSourceBlock>& sourceBlocks);

    // CallPrefetchProcess
    int PreparePrefetchResults(const TaskInfo& taskInfo);

    // PreparePrefetchResults 分解辅助（提前识别/判别在并行域外完成）
    int AcquirePrefetchTopkBase(const TaskInfo& taskInfo, uint32_t oneRunBsLen,
                                PrefetchTopkView& topkView);
    void ZeroFillAllPrefetchResults(uint32_t oneRunBsLen, const PrefetchShmLayout& layout,
                                    uint32_t* versionPtr);
    int PublishPrefetchResultsRank0(const TaskInfo& taskInfo, uint32_t oneRunBsLen,
                                    const PrefetchShmLayout& layout, uint32_t* versionPtr);
    int SyncPrefetchResultsFromRank0(const TaskInfo& taskInfo, const PrefetchShmLayout& layout,
                                     uint32_t* versionPtr);

    /* oneTopkListScratch/useBlockTableScratch 由调用方（OMP 每 layerID 迭代）
     * 提供并跨请求复用，免 per-(req,layer) 堆分配；本函数对二者只做
     * assign/clear + 全量覆写，不跨调用持有内容。 */
    int ProcessPrefetchRequest(const TaskInfo& taskInfo, const PrefetchTopkView& topkView, size_t i,
                               int layerID, const PrefetchShmLayout& layout,
                               std::vector<int>& oneTopkListScratch,
                               std::vector<int>& useBlockTableScratch);

    // ProcessPrefetchRequest 分解辅助
    HBMManagerLayered* FindReqLayerManager(const std::string& reqID, int layerID);
    /* 校验单请求查找输入（topkLen/平坦读边界）。返回 1=就绪，0=topkLen 空
     *（软跳过，已清槽），-1=非法（已日志+清槽）。 */
    int AcquirePrefetchRequestState(const TaskInfo& taskInfo, const PrefetchTopkView& topkView,
                                    size_t i, int layerID, const PrefetchShmLayout& layout,
                                    const int*& topkPtr, int& oneTopkLen);
    // 尾段：尺寸校验 + 张量拷贝 + shm 写结果；0=成功，-1=失败（已清槽）。
    int FinishPrefetchRequest(size_t i, int layerID, const PrefetchShmLayout& layout,
                              const std::string& oneReqID, std::vector<int>& useBlockTable,
                              std::vector<int>& missGlobalIds, std::vector<int>& loadSlots);
    /* missGlobalIds/loadSlots 由调用方检索后传入（每 (req,layer) 一次 find），
     * 本函数不再访问 mMissGlobalToken/mLoadSlot 两张 map。 */
    bool RunHbmTopkLookup(const std::string& oneReqID, std::vector<int>& oneTopkList, int layerID,
                          int bsIndex, int oneTopkLen, std::vector<int>& useBlockTable,
                          std::vector<int>& missGlobalIds, std::vector<int>& loadSlots);

    void WritePrefetchResult(size_t i, int layerID, const std::vector<int>& useBlockTable,
                             const PrefetchShmLayout& layout, const std::vector<int>& missGlobalIds,
                             const std::vector<int>& loadSlots);

    void ClearPrefetchResultSlot(size_t i, int layerID, const PrefetchShmLayout& layout);

    void ReadPrefetchResults(const TaskInfo& taskInfo, const PrefetchShmLayout& layout);

    // ReadPrefetchResults 分解辅助
    void ReadOneReqLayerResults(const TaskInfo& taskInfo, const PrefetchShmLayout& layout, size_t i,
                                int layerID, std::vector<int>& useBlockTable);

    void MergeAndCopyPrefetchLayer(const TaskInfo& taskInfo, ThreadInfo& threadInfo, int layerID,
                                   PrefetchTiming& timing);

    // MergeAndCopyPrefetchLayer 分解辅助方法（huge_method 序号50）
    void CopyPrefetchToH2DBuffers(int streamIdx, int layerID);
    void ValidateAndPublishHbmSlotIndex(int streamIdx, int layerID,
                                        std::vector<uint64_t>& hbmSlotIndex);

    void TransferAndScatterPrefetchLayer(ThreadInfo& threadInfo, int layerID, int streamIdx,
                                         PrefetchTiming& timing, PrefetchTimePoint syncEnd);

    /* PreSwapTokens 分解辅助：逐层计算 swap 追加末尾边界。decay 模式取
     * (req,layer) 管理器的分层窗口（如 sparseLen=5120、layer0 窗口 4096），
     * Lru / 管理器未就绪时为整段 sparseLen（与原行为一致）。 */
    void CollectLayerAppendEnds(const std::string& reqID, int sparseLen,
                                std::vector<int>& appendEnds);

    void CopyPreSwapTokens(uintptr_t useTokenBase, uintptr_t spaTokenBase,
                           const std::vector<int>& swapRange, const std::vector<int>& needAppend,
                           int sparseLen, int spaMaxBs, int mtpToken, int changeLen,
                           const std::vector<int>& appendEnds);

    bool SubmitIndexerInitLoadTask(UC::Detail::TaskDesc& task, const std::string& reqID,
                                   int layerID);

    // ProcessIndexerInitLoadLayer 分解辅助
    bool BuildIndexerInitLoadShards(std::vector<std::vector<int>>& blockTable,
                                    std::vector<std::vector<int>>& slabIdx,
                                    const std::string& reqID, int layerID, int number,
                                    std::uint8_t* indexerBaseAddr,
                                    const std::vector<BlockId>& hashVec,
                                    UC::Detail::TaskDesc& task);
    // BuildIndexerInitLoadShards 分解辅助（单块 shard 追加，false=中止）
    bool TryAppendInitShard(int i, std::vector<std::vector<int>>& blockTable,
                            std::vector<std::vector<int>>& slabIdx, const std::string& reqID,
                            int layerID, int number, std::uint8_t* indexerBaseAddr,
                            const std::vector<BlockId>& hashVec, size_t shardIndex,
                            UC::Detail::TaskDesc& task);

    void FinishOneLayerPrefetch(const std::string& reqID, int layerID,
                                std::map<int, int>& hitBlocksIdx, std::vector<int>& missIdxs);

    bool PrepareUCMMergeSources(std::vector<UCMMergeBlockContext>& blockContexts, int layerID,
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
                               UCMTokenLoadPlan& plan);

    // BuildLastChunkSources 分解辅助
    bool AppendLastChunkTokens(int layerID, const std::string& reqID,
                               const std::vector<int>& moveMap, const std::vector<BlockId>& hashVec,
                               const std::unordered_map<int, LastKVBlockAddress>& lastKVCopy,
                               int lastUCMBlockIdx, std::vector<LastChunkSourceBlock>& sourceBlocks,
                               UCMTokenLoadPlan& plan);
    void OpenUcmShard(size_t i, size_t blockIdxSz, int layerID, const std::vector<BlockId>& hashVec,
                      std::vector<size_t>& blockFirstDst, std::vector<int>& blockShardIdx,
                      UCMTokenLoadPlan& plan);
    bool BindNonUcmSourceBlock(int blockIdx, int layerID,
                               const std::unordered_map<int, LastKVBlockAddress>& lastKVCopy,
                               std::vector<LastChunkSourceBlock>& sourceBlocks);
    void CopybackLastChunkBlocks(int layerID, const std::vector<int>& blockTable,
                                 const std::vector<int>& moveMap);

    void SetIndexerMapForLayer(const std::string& reqID, int layerID,
                               std::vector<int>& blockTableList,
                               std::vector<std::vector<int>>& remainIdx,
                               std::vector<std::vector<int>>& prefetchIdx,
                               std::vector<int>& blockSlot);

    // SetIndexerMapForLayer 分解辅助
    void BuildIndexerLayerTables(int layerID, std::vector<int>& blockTableList,
                                 std::vector<std::vector<int>>& remainIdx,
                                 std::vector<std::vector<int>>& prefetchIdx,
                                 std::vector<int>& blockSlot, std::map<int, int>& oneDocTable,
                                 std::map<int, int>& oneBlockMap, std::vector<int>& onePrefetchIdx,
                                 std::vector<std::pair<int, int>>& oneDsaSparseBlock);
    void WriteIndexerLayerTables(const std::string& reqID, int layerID,
                                 std::vector<std::pair<int, int>>& oneDsaSparseBlock);

    void ProcessTokensMapTopk(int layerID, const std::string& reqID, torch::Tensor& qTopkTensor,
                              int maxIdx, size_t sparseSeqLen);

    // ProcessTokensMapTopk 分解辅助
    HBMManagerLayered* InitLayerManager(const std::string& reqID, int layerID, int maxIdx,
                                        size_t sparseSeqLen, int topkSize);
    int AcquireQTopkView(torch::Tensor& qTopkTensor, const std::string& reqID, int layerID,
                         PrefetchTopkView& topkView, int& qTopkNum, int& topkSize);
    int CopyUseBlockTableToIndicesTensor(const std::string& reqID, int layerID,
                                         const std::vector<int>& useBlockTable);

    void ReleasePrefetchStreamResources(int layerID);

    void InitBufferContainers();
};  //
}  // namespace ucmprefetch

#endif