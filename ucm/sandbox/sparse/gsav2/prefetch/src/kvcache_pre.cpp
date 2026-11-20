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
#include "kvcache_pre.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <kvcache_log.h>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include "kvcache_pre_detail.h"
#include "securec.h"

#define MAX_LOAD_NUM 2000u
#define H2D_INFO_SIZE 6
#define MAX_TOPK_SIZE 3200lu

namespace ucmprefetch {

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "KV pack requires lock-free uint64_t atomics");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "KV pack requires lock-free uint32_t atomics");
static_assert(std::atomic<int32_t>::is_always_lock_free,
              "KV pack requires lock-free int32_t atomics");

namespace {

constexpr auto kKVPackTimeout = std::chrono::seconds(30);
constexpr auto kKVPackPollInterval = std::chrono::microseconds(100);
using KVPackClock = std::chrono::steady_clock;

KVPackRankState* KVPackStates(SharedKVPackHeader* header)
{
    void* statesPtr = header + 1;
    return static_cast<KVPackRankState*>(statesPtr);
}

template <typename Predicate>
int KVPackWait(SharedKVPackHeader* header, Predicate ready)
{
    const auto deadline = KVPackClock::now() + kKVPackTimeout;
    for (;;) {
        if (header->fatalError.load(std::memory_order_acquire) != 0) { return -1; }
        if (ready()) { return 0; }
        if (KVPackClock::now() >= deadline) { return KVPackFail(header); }
        std::this_thread::sleep_for(kKVPackPollInterval);
    }
}
}  // namespace

// 创建或连接共享控制区和 KV staging。让 16 个 rank 映射同一份物理存储，各进程的虚拟地址可以不同。
int GSAPrefetchEngineC::InitKVPackControl(int dpRank)
{
    // 只在初始化时检查固定 rank/stream 配置，不在逐层路径重复检查。
    if (mRank < 0 || mRank >= static_cast<int>(mTpSize) || mStreamNum <= 0) { return -1; }
    const size_t streamCount = static_cast<size_t>(mStreamNum);
    const size_t kBytes = mKBufferHost[0].numel() * mKBufferHost.element_size();
    const size_t vBytes = mISSPAC8 ? 0 : mVBufferHost[0].numel() * mVBufferHost.element_size();
    const size_t stateCount = streamCount * mTpSize;
    const size_t controlBytes = sizeof(SharedKVPackHeader) + stateCount * sizeof(KVPackRankState);
    const size_t bytes = controlBytes + streamCount * (kBytes + vBytes);

    mKVPackControlName = "/gsa_kv_pack_control_dp" + std::to_string(dpRank);
    mKVPackPreviousEpoch.assign(streamCount, 0);
    mSharedKBufferHostPtr.resize(streamCount);
    if (!mISSPAC8) { mSharedVBufferHostPtr.resize(streamCount); }

    int fd = -1;
    auto fail = [&](const char* operation) {
        mLogger.log(LogLevel::ERROR, "|KVPack| %s failed, errno=%d\n", operation, errno);
        if (fd >= 0) { close(fd); }
        CleanupKVPackControl();
        return -1;
    };
    const auto deadline = KVPackClock::now() + kKVPackTimeout;
    if (mRank == 0) {
        const char* creatorFail = CreateKVPackControlFile(bytes, fd);
        if (creatorFail != nullptr) { return fail(creatorFail); }
    } else {
        const char* followerFail = OpenKVPackControlFollower(deadline, bytes, fd);
        if (followerFail != nullptr) { return fail(followerFail); }
    }

    void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) { return fail("mmap"); }
    close(fd);
    fd = -1;
    mKVPackControl = static_cast<SharedKVPackHeader*>(mapping);
    mKVPackControlSize = bytes;
    if (mIsKVPackControlCreator) {
        InitKVPackSharedStates(stateCount);
    } else {
        const char* waitFail = WaitKVPackControlInitialized(deadline);
        if (waitFail != nullptr) { return fail(waitFail); }
    }
    SetupKVPackSharedBuffers(controlBytes, streamCount, kBytes, vBytes);
    return 0;
}

const char* GSAPrefetchEngineC::WaitKVPackControlInitialized(
    const std::chrono::steady_clock::time_point& deadline)
{
    while (mKVPackControl->initialized.load(std::memory_order_acquire) != 1) {
        if (KVPackClock::now() >= deadline) { return "wait initialized"; }
        std::this_thread::sleep_for(kKVPackPollInterval);
    }
    return nullptr;
}

const char* GSAPrefetchEngineC::CreateKVPackControlFile(size_t bytes, int& fd)
{
    fd = shm_open(mKVPackControlName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) { return "shm_open creator"; }
    mIsKVPackControlCreator = true;
    if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) { return "ftruncate"; }
    return nullptr;
}

const char* GSAPrefetchEngineC::OpenKVPackControlFollower(
    const std::chrono::steady_clock::time_point& deadline, size_t bytes, int& fd)
{
    while ((fd = shm_open(mKVPackControlName.c_str(), O_RDWR, 0600)) < 0) {
        if ((errno != ENOENT && errno != EINTR) || KVPackClock::now() >= deadline) {
            return "shm_open follower";
        }
        std::this_thread::sleep_for(kKVPackPollInterval);
    }
    // 文件名出现与 ftruncate 完成之间存在启动窗口。
    for (;;) {
        struct stat info{};
        if (fstat(fd, &info) != 0) { return "fstat"; }
        if (info.st_size == static_cast<off_t>(bytes)) { return nullptr; }
        if (info.st_size != 0 || KVPackClock::now() >= deadline) { return "shared memory size"; }
        std::this_thread::sleep_for(kKVPackPollInterval);
    }
}

void GSAPrefetchEngineC::InitKVPackSharedStates(size_t stateCount)
{
    ::new (mKVPackControl) SharedKVPackHeader();
    auto* states = KVPackStates(mKVPackControl);
    for (size_t i = 0; i < stateCount; ++i) {
        ::new (static_cast<void*>(states + i)) KVPackRankState();
    }
    mKVPackControl->initialized.store(1, std::memory_order_release);
}

void GSAPrefetchEngineC::SetupKVPackSharedBuffers(size_t controlBytes, size_t streamCount,
                                                  size_t kBytes, size_t vBytes)
{
    auto* mapping = static_cast<std::byte*>(static_cast<void*>(mKVPackControl));
    auto* kBase = mapping + controlBytes;
    auto* vBase = kBase + streamCount * kBytes;
    for (size_t i = 0; i < streamCount; ++i) {
        mSharedKBufferHostPtr[i] = reinterpret_cast<uint64_t>(kBase + i * kBytes);
        if (!mISSPAC8) {
            mSharedVBufferHostPtr[i] = reinterpret_cast<uint64_t>(vBase + i * vBytes);
        }
    }
}

// 使用共享槽前，等待上一轮全部 rank 的 H2D 完成；首次使用不等待。确保旧数据已被消费，允许覆盖该槽。
int GSAPrefetchEngineC::BeginKVPackSlot(int streamIdx, uint64_t epoch)
{
    const uint64_t previousEpoch = mKVPackPreviousEpoch[streamIdx];
    auto* states = KVPackStates(mKVPackControl) + static_cast<size_t>(streamIdx) * mTpSize;
    // epoch 由 version/layer 生成；首次使用不等待。
    (void)epoch;
    return KVPackWait(mKVPackControl, [&] {
        if (previousEpoch == 0) { return true; }
        for (size_t rank = 0; rank < mTpSize; ++rank) {
            if (states[rank].h2dDoneEpoch.load(std::memory_order_acquire) != previousEpoch) {
                return false;
            }
        }
        return true;
    });
}

// 确保共享 KV 已完整写好，随后才能 H2H。
int GSAPrefetchEngineC::FinishKVPack(int streamIdx, uint64_t epoch)
{
    auto* states = KVPackStates(mKVPackControl) + static_cast<size_t>(streamIdx) * mTpSize;
    states[mRank].packedEpoch.store(epoch, std::memory_order_release);
    return KVPackWait(mKVPackControl, [&] {
        for (size_t rank = 0; rank < mTpSize; ++rank) {
            if (states[rank].packedEpoch.load(std::memory_order_acquire) != epoch) { return false; }
        }
        return true;
    });
}

// 本 rank 的 H2D 同步成功后，发布完成标记。通知其他 rank：本卡已完成本轮消费。
void GSAPrefetchEngineC::MarkKVH2DDone(int streamIdx, uint64_t epoch)
{
    auto* states = KVPackStates(mKVPackControl) + static_cast<size_t>(streamIdx) * mTpSize;
    states[mRank].h2dDoneEpoch.store(epoch, std::memory_order_release);
    mKVPackPreviousEpoch[streamIdx] = epoch;
}

// 解除本进程的共享映射，仅创建者删除共享内存名称。释放共享缓冲资源，不释放原私有 pinned staging。
void GSAPrefetchEngineC::CleanupKVPackControl()
{
    if (mKVPackControl != nullptr) {
        munmap(mKVPackControl, mKVPackControlSize);
        mKVPackControl = nullptr;
    }
    if (mIsKVPackControlCreator) { shm_unlink(mKVPackControlName.c_str()); }
    mKVPackControlSize = 0;
    mIsKVPackControlCreator = false;
    mKVPackPreviousEpoch.clear();
    mSharedKBufferHostPtr.clear();
    mSharedVBufferHostPtr.clear();
}

float average(const std::vector<float>& vec)
{
    if (vec.empty()) { throw std::runtime_error("Cannot compute average of empty vector"); }
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return static_cast<float>(sum / vec.size());
}

float average2d(const std::vector<std::vector<float>>& matrix)
{
    if (matrix.empty()) { return 0.0f; }

    double sum = std::accumulate(matrix.begin(), matrix.end(), 0.0,
                                 [](double acc, const std::vector<float>& row) {
                                     return acc + std::accumulate(row.begin(), row.end(), 0.0);
                                 });

    size_t count = 0;
    for (const auto& row : matrix) { count += row.size(); }
    /* Non-empty matrix of empty rows: count==0 must return 0.0f instead of
     * dividing by zero (NaN). */
    if (count == 0) { return 0.0f; }

    return static_cast<float>(sum / count);
}

void MultiBSThreadFun(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(threadInfo.args_);
    int ret = engine->CallPrefetchProcess(taskInfo, threadInfo);
    if (ret != 0) {
        engine->mLogger.log(LogLevel::ERROR,
                            "|KVCache Prefetch| [ERROR] MultiBSThreadFun task %d failed, ret=%d\n",
                            taskInfo.taskID, ret);
    }
    {
        UReqLockGuard lock(engine->mUpdateReqMutex,
                           engine->UReqLockStatsAt(UReqLockSite::WorkerCleanup));
        // indexer worker 在跑则延迟冲刷（DelReqIDRun 幂等），杜绝并发 erase
        // 与其无锁 IO 段迭代的 use-after-free；只门控对侧 worker。
        if (!engine->GetIndexerStatus()) { engine->DelReqIDRun(); }
    }
    // 状态位在 mTaskDoneMutex 内翻转、锁外 notify（杜绝丢唤醒）。
    {
        std::lock_guard<std::mutex> doneLock(engine->mTaskDoneMutex);
        engine->mIsPrefetchRunning.store(false, std::memory_order_release);
    }
    engine->mPrefetchDoneCv.notify_all();
}

void MultiBSThreadFunIndexer(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(threadInfo.args_);
    int ret = engine->IndexerLoadProcess(taskInfo, threadInfo);
    if (ret != 0) {
        engine->mLogger.log(LogLevel::ERROR,
                            "|KVCache Prefetch| [ERROR] MultiBSThreadFunIndexer task %d failed, "
                            "ret=%d\n",
                            taskInfo.taskID, ret);
    }
    // 与 MultiBSThreadFun 尾部同协议：mutex 内翻转、锁外 notify。
    {
        std::lock_guard<std::mutex> doneLock(engine->mTaskDoneMutex);
        engine->mIsIndexerLoadRunning.store(false, std::memory_order_release);
    }
    engine->mIndexerDoneCv.notify_all();
}

enum class ModelInfoIndex {
    LayerNum = 0,
    MaxBlockLen,
    SpaTopkLen,
    StreamNum,
    SparseLen,
    PrefetchType,
    DpRank,
    TpSize,
    IndexerOffload,
    IsSfaC8,
    MaxBs,
};

enum class SizeInfoIndex { BlockSize = 0, NumHead, KDim, VDim, ElementSize };

template <typename T>
constexpr size_t ToIndex(T value)
{
    return static_cast<size_t>(value);
}

GSAPrefetchEngineC::GSAPrefetchEngineC(std::vector<int>& modelInfo, std::vector<int>& sizeinfo,
                                       bool isLog, int rank, int threadNum,
                                       std::vector<int> cpuToBind, std::vector<int> topkLayerMap,
                                       std::string logFileName)
    : mLogger(logFileName, LogLevel::INFO, isLog)
{
    /* UReqLock 埋点 dump 门控：UCM_PREFETCH_LOCK_PROFILE 未设/0=关；
     * 1=每 100 步 dump；N(>1)=每 N 步 dump。计数本身常开（见
     * kvcache_pre_detail.h UReqLockStats），此处仅解析 dump 开关与间隔。 */
    const char* lockProfileVal = std::getenv("UCM_PREFETCH_LOCK_PROFILE");
    if (lockProfileVal != nullptr && lockProfileVal[0] != '\0' && lockProfileVal[0] != '0') {
        mUReqLockProfileEnabled = true;
        char* parseEnd = nullptr;
        const long parsedSteps = std::strtol(lockProfileVal, &parseEnd, 10);
        if (parseEnd != lockProfileVal && parsedSteps > 1) {
            mUReqLockDumpIntervalSteps = static_cast<uint64_t>(parsedSteps);
        }
    }
    InitModelInfo(modelInfo, isLog, cpuToBind, topkLayerMap, rank, threadNum);
    InitBufferInfo(sizeinfo);
    InitAclRuntime();
    InitPrefetchResultShm(modelInfo);
}

void GSAPrefetchEngineC::DumpUReqLockStats()
{
    /* 埋点 dump：exchange 清零取"距上次 dump"的区间值。wait 落点指示
     * "哪里在等"，hold 排名指示"谁在挡"——是 Tier 1 优化
     * （IndexerLoadProcess 整程锁收窄 / LastChunkMove 分相 / OMP 预取化）
     * 的裁决依据。名称表顺序必须与 UReqLockSite 枚举一一对应。 */
    static const char* kUReqLockSiteNames[static_cast<size_t>(UReqLockSite::Count)] = {
        "WorkerCleanup",     "DelBlocksMap",        "SetBlocksMap",     "SetLastIndexerBlock",
        "SetIndexerMap",     "UpdateIndexerMap",    "SubmitTaskIndexer", "SetTopkIndices",
        "ResetReqTokenState", "ProcessTokensMapTopk", "RunLastChunkMoveGuarded",
        "IndexerInitLoad",   "SubmitTask",          "IndexerLoadProcess", "IndexerLoadResolve",
        "ProcessPrefetchRequest", "ReadPrefetchResults",
    };
    static_assert(sizeof(kUReqLockSiteNames) / sizeof(kUReqLockSiteNames[0]) ==
                      static_cast<size_t>(UReqLockSite::Count),
                  "UReqLock site name table out of sync with UReqLockSite enum");
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| UReqLock profile (rank %d, since last "
                "dump): site, acq, contended(>1us), waitUs, holdUs\n",
                mDecodeStep, mRank);
    for (size_t s = 0; s < static_cast<size_t>(UReqLockSite::Count); ++s) {
        const uint64_t acq = mUReqLockStats[s].acquisitions.exchange(0, std::memory_order_relaxed);
        const uint64_t contended =
            mUReqLockStats[s].contended.exchange(0, std::memory_order_relaxed);
        const uint64_t waitNs = mUReqLockStats[s].waitNs.exchange(0, std::memory_order_relaxed);
        const uint64_t holdNs = mUReqLockStats[s].holdNs.exchange(0, std::memory_order_relaxed);
        if (acq == 0) { continue; }
        mLogger.log(LogLevel::INFO,
                     "  %-24s acq=%-10llu contended=%-8llu waitUs=%-12llu holdUs=%-12llu\n",
                     kUReqLockSiteNames[s], static_cast<unsigned long long>(acq),
                     static_cast<unsigned long long>(contended),
                     static_cast<unsigned long long>(waitNs / 1000),
                     static_cast<unsigned long long>(holdNs / 1000));
    }
}

void GSAPrefetchEngineC::InitModelInfo(std::vector<int>& modelInfo, bool isLog,
                                       std::vector<int> cpuToBind, std::vector<int> topkLayerMap,
                                       int rank, int threadNum)
{
    mLayerNum = modelInfo[ToIndex(ModelInfoIndex::LayerNum)];
    mMaxLen = modelInfo[ToIndex(ModelInfoIndex::MaxBlockLen)];
    mSpaTopkLen = modelInfo[ToIndex(ModelInfoIndex::SpaTopkLen)];
    mStreamNum = modelInfo[ToIndex(ModelInfoIndex::StreamNum)];
    mSparseLen = modelInfo[ToIndex(ModelInfoIndex::SparseLen)];
    mOpenIndexerOffload = modelInfo[ToIndex(ModelInfoIndex::IndexerOffload)];
    mISSPAC8 = modelInfo[ToIndex(ModelInfoIndex::IsSfaC8)];
    mMaxBs = modelInfo[ToIndex(ModelInfoIndex::MaxBs)];
    mTopkLayerMap = topkLayerMap;
    if (!mTopkLayerMap.empty()) { mTopkIndexerNum = mTopkLayerMap.back() + 1; }
    mRank = rank;
    mKVPackDpRank = modelInfo[ToIndex(ModelInfoIndex::DpRank)];
    mTpSize = modelInfo[ToIndex(ModelInfoIndex::TpSize)];
    mThreadNum = threadNum;
    mLogger.log(LogLevel::INFO,
                "GSAPrefetchEngineC new hbm Init mLayerNum: %d mMaxLen: %d "
                "mSpaTopkLen: %d mSparseLen: %d, prefetch type: %d, maxBs: %d\n",
                mLayerNum, mMaxLen, mSpaTopkLen, mSparseLen,
                modelInfo[ToIndex(ModelInfoIndex::PrefetchType)], mMaxBs);
    mTaskManger = std::make_unique<TaskManager>(mRank, modelInfo[ToIndex(ModelInfoIndex::DpRank)],
                                                modelInfo[ToIndex(ModelInfoIndex::TpSize)], this, 1,
                                                mStreamNum, "kv", cpuToBind);
    if (mOpenIndexerOffload) {
        mIndexerTaskManger = std::make_unique<TaskManager>(
            mRank, modelInfo[ToIndex(ModelInfoIndex::DpRank)],
            modelInfo[ToIndex(ModelInfoIndex::TpSize)], this, 1, mStreamNum, "indexer", cpuToBind);
    }
}

void GSAPrefetchEngineC::InitBufferInfo(std::vector<int>& sizeinfo)
{
    InitBufferContainers();
    mBlockSize = sizeinfo[ToIndex(SizeInfoIndex::BlockSize)];
    mTensorElemSize = sizeinfo[ToIndex(SizeInfoIndex::ElementSize)];
    mNumHead = sizeinfo[ToIndex(SizeInfoIndex::NumHead)];
    if (!mISSPAC8) {
        mVDim = sizeinfo[ToIndex(SizeInfoIndex::VDim)];
        mKVSizeBytesBlockRe = mTensorElemSize * mBlockSize * mNumHead * mVDim;
        mKVSizeBytesTokenRe = mTensorElemSize * mNumHead * mVDim;
    }
    mKDim = sizeinfo[ToIndex(SizeInfoIndex::KDim)];
    mKVSizeBytesBlock = mTensorElemSize * mBlockSize * mNumHead * mKDim;
    mKVSizeBytesToken = mTensorElemSize * mNumHead * mKDim;
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| init kvcache mKVSizeBytesBlock: %u, "
                "mKVSizeBytesBlockRe: %u,"
                "mKVSizeBytesToken %u, mKVSizeBytesTokenRe %u, mKDim: %d, mVDim: %d, mNumHead: %d "
                "mTensorElemSize: %d\n",
                mDecodeStep, mKVSizeBytesBlock, mKVSizeBytesBlockRe, mKVSizeBytesToken,
                mKVSizeBytesTokenRe, mKDim, mVDim, mNumHead, mTensorElemSize);
}

void GSAPrefetchEngineC::InitBufferContainers()
{
    if (!mISSPAC8) {
        mVcachePtr.resize(mLayerNum);
        mVcacheTensor.resize(mLayerNum);
        mVCacheAcl.resize(mLayerNum);
        mLastCHunkVBuffer.resize(mLayerNum);
        mLastCHunkVPtr.resize(mLayerNum);
        mVWorkspaceAddr.resize(mStreamNum);
        mVMaxWorkspaceSize.resize(mStreamNum, 0);
        mVHBMBufferAcl.resize(mStreamNum, nullptr);
        mVBufferHostPtr.resize(mStreamNum);
        mVDstIndexAcl.resize(mStreamNum, nullptr);
    }
    mKcachePtr.resize(mLayerNum);
    mKcacheTensor.resize(mLayerNum);
    mLoadNum.resize(mLayerNum);
    mKCacheAcl.resize(mLayerNum);
    mLastCHunkKBuffer.resize(mLayerNum);
    mLastCHunkKPtr.resize(mLayerNum);
    mAvgHitRate.resize(mTopkIndexerNum);
    if (mOpenIndexerOffload) {
        mIndexerTensor.resize(mLayerNum);
        mIndexerPtr.resize(mLayerNum);
    }
    // value for prefetch load
    mKWorkspaceAddr.resize(mStreamNum);
    mKMaxWorkspaceSize.resize(mStreamNum, 0);
    mKDstIndexAcl.resize(mStreamNum, nullptr);
    mKHBMBufferAcl.resize(mStreamNum, nullptr);
    mKBufferHostPtr.resize(mStreamNum);
}

void GSAPrefetchEngineC::InitAclRuntime()
{
    mContext = (void*)malloc(sizeof(aclrtContext));
    mStream = (void*)malloc(sizeof(aclrtStream));
    if (mContext == nullptr || mStream == nullptr) {
        mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| InitAclRuntime malloc failed\n");
        throw std::runtime_error("InitAclRuntime: malloc failed");
    }

    auto ret = aclrtGetCurrentContext(static_cast<aclrtContext*>(mContext));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtGetCurrentContext Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
        throw std::runtime_error("InitAclRuntime: aclrtGetCurrentContext failed");
    }

    ret = aclrtCreateStream(static_cast<aclrtStream*>(mStream));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtCreateStream Error: %d, Error: %s\n",
                    ret, errMsg ? errMsg : "Unknown error");
        throw std::runtime_error("InitAclRuntime: aclrtCreateStream failed");
    }

    mLogger.log(LogLevel::INFO,
                "GSAPrefetchEngineC aclrtGetCurrentContext: %p stream: %p, test memory 5\n",
                mContext, mStream);
}

void GSAPrefetchEngineC::CreatePrefetchResultShm()
{
    // Rank 0: Create shared memory with read-write access
    int shmFd = shm_open(mPrefetchResultShmName.c_str(), O_CREAT | O_RDWR, 0666);
    if (shmFd == -1) {
        mLogger.log(LogLevel::ERROR, "Rank 0 shm_open failed: %s\n", strerror(errno));
    } else {
        if (ftruncate(shmFd, mPrefetchResultShmSize) == -1) {
            mLogger.log(LogLevel::ERROR, "Rank 0 ftruncate failed: %s\n", strerror(errno));
        }
        mPrefetchResultShm =
            mmap(NULL, mPrefetchResultShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        if (mPrefetchResultShm == MAP_FAILED) {
            mLogger.log(LogLevel::ERROR, "Rank 0 mmap failed: %s\n", strerror(errno));
            mPrefetchResultShm = nullptr;
        }
        close(shmFd);
        mIsPrefetchResultShmCreator = true;
        mPrefetchResultVersion = 0;
        mLogger.log(LogLevel::INFO, "Rank 0 created shared memory: size=%zu\n",
                    mPrefetchResultShmSize);
    }
}

void GSAPrefetchEngineC::AttachPrefetchResultShm()
{
    // Other ranks: Wait for rank 0 to create, then attach with read-only access
    int retryCount = 0;
    int shmFd = -1;
    while (shmFd == -1 && retryCount < 3000) {
        shmFd = shm_open(mPrefetchResultShmName.c_str(), O_RDONLY, 0666);
        if (shmFd == -1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Wait 1ms before retry
            retryCount++;
        }
    }
    if (shmFd == -1) {
        mLogger.log(LogLevel::ERROR, "Rank %d failed to open shared memory after %d retries: %s\n",
                    mRank, retryCount, strerror(errno));
    } else {
        mPrefetchResultShm = mmap(NULL, mPrefetchResultShmSize, PROT_READ, MAP_SHARED, shmFd, 0);
        if (mPrefetchResultShm == MAP_FAILED) {
            mLogger.log(LogLevel::ERROR, "Rank %d mmap failed: %s\n", mRank, strerror(errno));
            mPrefetchResultShm = nullptr;
        } else {
            mIsPrefetchResultShmCreator = false;
            mPrefetchResultVersion = 0;
            mLogger.log(LogLevel::INFO, "Rank %d attached to shared memory (read-only): size=%zu\n",
                        mRank, mPrefetchResultShmSize);
        }
        close(shmFd);
    }
}

void GSAPrefetchEngineC::InitPrefetchResultShm(std::vector<int>& modelInfo)
{
    mPrefetchResultShm = nullptr;
    mPrefetchResultShmSize = 0;
    mPrefetchResultVersion = 0;
    mPrefetchResultShmName =
        "gsa_prefetch_result_dp" + std::to_string(modelInfo[ToIndex(ModelInfoIndex::DpRank)]);

    const uint32_t versionSize = 4;
    const uint32_t countSize = 4;

    size_t perReqPerLayerTotalSize = 2 * countSize + (mMaxLoadNum * 2 + mSpaTopkLen) * sizeof(int);

    mPrefetchResultShmSize = versionSize + mMaxBs * mTopkIndexerNum * perReqPerLayerTotalSize;

    if (mRank == 0) {
        CreatePrefetchResultShm();
    } else {
        AttachPrefetchResultShm();
    }
    if (mPrefetchResultShm == nullptr) {
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| prefetch result shm is nullptr, cannot continue\n");
        KVPackAbort(nullptr);
    }
    if (mRank == 0) {
        __atomic_store_n(static_cast<uint32_t*>(mPrefetchResultShm), uint32_t(0), __ATOMIC_RELEASE);
    }
}

GSAPrefetchEngineC::~GSAPrefetchEngineC()
{
    mTaskManger.reset();
    mIndexerTaskManger.reset();
    CleanupKVPackControl();
    // Clean up shared memory
    if (mPrefetchResultShm != nullptr) {
        munmap(mPrefetchResultShm, mPrefetchResultShmSize);
        if (mIsPrefetchResultShmCreator) { shm_unlink(mPrefetchResultShmName.c_str()); }
        mPrefetchResultShm = nullptr;
    }
    for (int layerID = 0; layerID < mStreamNum; layerID++) {
        ReleasePrefetchStreamResources(layerID);
    }
    free(mContext);
    auto ret = aclrtDestroyStream((*static_cast<aclrtStream*>(mStream)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtDestroyStream Error: %d, Error: %s\n",
                    ret, errMsg ? errMsg : "Unknown error");
    }
    free(mStream);
}

void GSAPrefetchEngineC::ReleasePrefetchStreamResources(int layerID)
{
    if (mKMaxWorkspaceSize[layerID] > 0) {
        auto ret = aclrtFree(mKWorkspaceAddr[layerID]);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtFree Error: %d, Error: %s\n", ret,
                        errMsg ? errMsg : "Unknown error");
        }
    }
    if (mKDstIndexAcl[layerID] != nullptr) {
        ReleaseAclTensor(mKDstIndexAcl[layerID]);
        mKDstIndexAcl[layerID] = nullptr;
    }
    if (mKHBMBufferAcl[layerID] != nullptr) {
        ReleaseAclTensor(mKHBMBufferAcl[layerID]);
        mKHBMBufferAcl[layerID] = nullptr;
    }
    if (mISSPAC8) { return; }
    if (mVMaxWorkspaceSize[layerID] > 0) {
        auto ret = aclrtFree(mVWorkspaceAddr[layerID]);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| aclrtFree Error: %d, Error: %s\n", ret,
                        errMsg ? errMsg : "Unknown error");
        }
    }
    if (mVDstIndexAcl[layerID] != nullptr) {
        ReleaseAclTensor(mVDstIndexAcl[layerID]);
        mVDstIndexAcl[layerID] = nullptr;
    }
    if (mVHBMBufferAcl[layerID] != nullptr) {
        ReleaseAclTensor(mVHBMBufferAcl[layerID]);
        mVHBMBufferAcl[layerID] = nullptr;
    }
}

void GSAPrefetchEngineC::SetBlocksMap(const std::string& reqID,
                                      const std::vector<BlockId>& blocksHash, int maxIdx,
                                      const std::vector<int>& blockIdxs,
                                      const std::vector<uintptr_t>& kAddrs,
                                      const std::vector<uintptr_t>& vAddrs)
{
    if (blockIdxs.size() != kAddrs.size()) {
        mLogger.log(LogLevel::ERROR, "|SetBlocksMap| kAddrs size mismatch\n");
        return;
    }
    if (!mISSPAC8 && blockIdxs.size() != vAddrs.size()) {
        mLogger.log(LogLevel::ERROR, "|SetBlocksMap| vAddrs size mismatch\n");
        return;
    }
    /* admission 写侧与 indexer worker（其段1 表事务 / 段2 IndexerLoadResolve
     * mini 段均持 mUpdateReqMutex find 本函数写入的两张 map）对称加锁，杜绝
     * std::map 新 key 插入 vs 并发 find 的结构级 UB。每请求一次，非解码热路径。 */
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SetBlocksMap));
    mAllBlocksHash[reqID] = blocksHash;
    mPromptLen[reqID] = maxIdx;
    auto& blockMap = mLastKVBlockAddress[reqID];
    blockMap.clear();
    for (size_t i = 0; i < blockIdxs.size(); ++i) {
        blockMap[blockIdxs[i]].kAddr = kAddrs[i];
        if (!mISSPAC8) { blockMap[blockIdxs[i]].vAddr = vAddrs[i]; }
    }
}

void GSAPrefetchEngineC::SetLastIndexerBlock(const std::string& reqID,
                                             const std::vector<int>& blockIdxs,
                                             const std::vector<uintptr_t>& indexerAddrs)
{
    if (!mOpenIndexerOffload) { return; }
    if (blockIdxs.size() != indexerAddrs.size()) {
        mLogger.log(LogLevel::ERROR, "[ERROR] SetLastIndexerBlocks size mismatch, reqID=%s\n",
                    reqID.c_str());
        return;
    }
    /* 与 SetBlocksMap 同理：写侧补 mUpdateReqMutex 对称（读侧 indexer worker
     * 已持同锁）。 */
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SetLastIndexerBlock));
    auto reqIt = mLastKVBlockAddress.find(reqID);
    if (reqIt == mLastKVBlockAddress.end()) {
        mLogger.log(LogLevel::ERROR, "[ERROR] SetLastIndexerBlock no block info, reqID=%s\n",
                    reqID.c_str());
        return;
    }
    auto& blockMap = reqIt->second;
    for (size_t i = 0; i < blockIdxs.size(); ++i) {
        auto it = blockMap.find(blockIdxs[i]);
        if (it == blockMap.end()) { continue; }
        it->second.indexerAddr = indexerAddrs[i];
        it->second.indexerBlockIdx = blockIdxs[i];
    }
}

void GSAPrefetchEngineC::SetTokensMap(int layerID, const std::string& reqID,
                                      torch::Tensor& qTopkTensor,
                                      const std::vector<int>& remainBlockTable, int maxIdx,
                                      const std::vector<int>& slots, bool isNeedInit)
{
    std::lock_guard<std::mutex> lock(mMutexUseToken);
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch|"
                " Start SetTokensMap layerID: %d, reqID: %s, isNeedInit: %d\n",
                mDecodeStep, layerID, reqID.c_str(), isNeedInit);
    const auto start = std::chrono::high_resolution_clock::now();
    if (static_cast<size_t>(layerID) >= mTopkLayerMap.size()) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| SetTokensMap layerID out of range: "
                    "reqID=%s layerID=%d mapSize=%zu\n",
                    mDecodeStep, reqID.c_str(), layerID, mTopkLayerMap.size());
        return;
    }
    int curTopkIndex = mTopkLayerMap[layerID];
    if (layerID == 0) {
        ResetReqTokenState(reqID, remainBlockTable, maxIdx, slots);
    }
    /* 提前识别：curTopkIndex 越界即拒绝（后续 mAvgHitRate/[manager] 下标依赖）。 */
    if (curTopkIndex < 0 || static_cast<size_t>(curTopkIndex) >= mAvgHitRate.size()) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| SetTokensMap topkIndex out of range: "
                    "reqID=%s layerID=%d topkIndex=%d avgHitRateSize=%zu\n",
                    mDecodeStep, reqID.c_str(), layerID, curTopkIndex, mAvgHitRate.size());
        return;
    }
    if (isNeedInit) {
        mAvgHitRate[curTopkIndex].clear();
        const size_t sparseSeqLen = remainBlockTable.size() * static_cast<size_t>(mBlockSize);
        ProcessTokensMapTopk(curTopkIndex, reqID, qTopkTensor, maxIdx, sparseSeqLen);
    }
    const auto managerEnd = std::chrono::high_resolution_clock::now();
    const auto managerDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(managerEnd - start);
    RunLastChunkMoveGuarded(layerID, reqID, curTopkIndex);
    const auto totalEnd = std::chrono::high_resolution_clock::now();
    const auto totalDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - start);
    mLogger.log(LogLevel::DEBUG,
                "Decode step: %u, |KVCache Prefetch| End SetTokensMap "
                "layerID=%d reqID=%s managerCost=%lld us totalCost=%lld us\n",
                mDecodeStep, layerID, reqID.c_str(),
                static_cast<long long>(managerDuration.count()),
                static_cast<long long>(totalDuration.count()));
}

void GSAPrefetchEngineC::ResetReqTokenState(const std::string& reqID,
                                            const std::vector<int>& remainBlockTable, int maxIdx,
                                            const std::vector<int>& slots)
{
    /* layer0 每步重建 per-request 容器；写七张 per-request map（operator[] 含
     * 插入/重平衡），与 worker 清理段 erase 并发，持锁。调用方已持
     * mMutexUseToken，锁序 mMutexUseToken → mUpdateReqMutex。 */
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::ResetReqTokenState));
    auto it = mHBMMangers.find(reqID);
    if (it != mHBMMangers.end()) {
        it->second.clear();
        mAllReqIdSlots[reqID].clear();
        mReqUseBlockTable[reqID].clear();
        mMissGlobalToken[reqID].clear();
        mLoadSlot[reqID].clear();
    }
    mAllReqIdSlots[reqID].assign(slots.begin(), slots.end());
    mReqUseBlockTable[reqID].assign(remainBlockTable.begin(), remainBlockTable.end());
    mSparseSeqLen[reqID] = remainBlockTable.size() * static_cast<size_t>(mBlockSize);
    mPromptLen[reqID] = maxIdx;
    mHBMMangers[reqID].resize(mTopkIndexerNum);
    mMissGlobalToken[reqID].resize(mTopkIndexerNum);
    mLoadSlot[reqID].resize(mTopkIndexerNum);
}

void GSAPrefetchEngineC::RunLastChunkMoveGuarded(int layerID, const std::string& reqID,
                                                 int curTopkIndex)
{
    /* moveMap 值快照 mini 段：mSlotToGlobal 原先跨 IO 全程持有，值拷贝后
     * 锁外 erase 不再影响。调用方仍全程持 mMutexUseToken（锁序不变），
     * LastChunkMove 的 DDR IO 不再持 mUpdateReqMutex。 */
    std::vector<int> moveMap;
    {
        UReqLockGuard lock(mUpdateReqMutex,
                           UReqLockStatsAt(UReqLockSite::RunLastChunkMoveGuarded));
        // manager 判别：未就绪/已被清理一律日志+跳过，杜绝空指针 core。
        auto managersIt = mHBMMangers.find(reqID);
        if (managersIt == mHBMMangers.end() || curTopkIndex < 0 ||
            static_cast<size_t>(curTopkIndex) >= managersIt->second.size() ||
            !managersIt->second[static_cast<size_t>(curTopkIndex)]) {
            mLogger.log(LogLevel::ERROR,
                        "Decode step: %u, |KVCache Prefetch| SetTokensMap manager not ready, "
                        "skip last chunk move: reqID=%s layerID=%d topkIndex=%d tableSize=%zu\n",
                        mDecodeStep, reqID.c_str(), layerID, curTopkIndex,
                        managersIt == mHBMMangers.end() ? 0 : managersIt->second.size());
            return;
        }
        moveMap = managersIt->second[static_cast<size_t>(curTopkIndex)]->mSlotToGlobal;
    }
    LastChunkMove(layerID, reqID, moveMap);
}

void GSAPrefetchEngineC::ProcessTokensMapTopk(int layerID, const std::string& reqID,
                                              torch::Tensor& qTopkTensor, int maxIdx,
                                              size_t sparseSeqLen)
{
    /* 循环外一次性提取只读视图（含读范围校验）；写 mHBMMangers/miss/loadSlot
     * 与 worker erase 并发，持锁。调用方已持 mMutexUseToken。 */
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::ProcessTokensMapTopk));
    PrefetchTopkView topkView;
    int qTopkNum = 0;
    int topkSize = 0;
    if (AcquireQTopkView(qTopkTensor, reqID, layerID, topkView, qTopkNum, topkSize) != 0) {
        return;
    }
    auto* manager = InitLayerManager(reqID, layerID, maxIdx, sparseSeqLen, topkSize);
    if (manager == nullptr) { return; }
    auto& missGlobalTokens = mMissGlobalToken[reqID][layerID];
    auto& loadSlots = mLoadSlot[reqID][layerID];
    std::vector<int> useBlockTable;
    useBlockTable.reserve(static_cast<size_t>(mSpaTopkLen));
    std::vector<int> oneTopkList(static_cast<size_t>(topkSize));
    for (int i = 0; i < qTopkNum; ++i) {
        const int* topkPtr = topkView.base + static_cast<int64_t>(i) * topkView.stride0;
        std::memcpy(oneTopkList.data(), topkPtr, static_cast<size_t>(topkSize) * sizeof(int));
        missGlobalTokens.clear();
        loadSlots.clear();
        useBlockTable.clear();
        manager->processTopkWithDebug(oneTopkList, useBlockTable, missGlobalTokens, loadSlots,
                                     mLogger);
    }
    if (useBlockTable.size() != static_cast<size_t>(mSpaTopkLen)) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] mAllTopkIndicesTensor greater than mSpaTopkLen !!! "
                    "layer_id %d, mSpaTopkLen: %d, useBlockTable size: %zu\n",
                    layerID, mSpaTopkLen, useBlockTable.size());
        return;
    }
    CopyUseBlockTableToIndicesTensor(reqID, layerID, useBlockTable);
}

// ProcessTokensMapTopk 分解辅助：重建 (req,layer) 的 HBM 管理器，未注册/越界返回 nullptr。
HBMManagerLayered* GSAPrefetchEngineC::InitLayerManager(const std::string& reqID, int layerID,
                                                        int maxIdx, size_t sparseSeqLen,
                                                        int topkSize)
{
    auto managersIt = mHBMMangers.find(reqID);
    if (managersIt == mHBMMangers.end() ||
        static_cast<size_t>(layerID) >= managersIt->second.size()) {
        mLogger.log(LogLevel::ERROR,
                     "Decode step: %u, |KVCache Prefetch| HBM managers not initialized, "
                     "skip topk map: reqID=%s layerID=%d managerSize=%zu\n",
                     mDecodeStep, reqID.c_str(), layerID,
                     managersIt == mHBMMangers.end() ? 0 : managersIt->second.size());
        return nullptr;
    }
    /* layerId 供分层窗口表取值（与 mTopkIndexerNum 同一索引空间）；evictMode /
     * windowPlan 来自 Python 配置（SetHbmEvictMode），默认 Lru + Uniform 时与
     * 原 HBMManager 行为逐字节一致。 */
    managersIt->second[static_cast<size_t>(layerID)] = std::make_unique<HBMManagerLayered>();
    auto& manager = *managersIt->second[static_cast<size_t>(layerID)];
    manager.init(maxIdx, sparseSeqLen, mSpaTopkLen, topkSize, mLogger, /*preloadHbm=*/true,
                 /*maxTotalSlots=*/32768, /*layerId=*/layerID, mHbmEvictMode, mHbmWindowPlan);
    return &manager;
}

void GSAPrefetchEngineC::SetHbmEvictMode(int mode)
{
    mHbmEvictMode = (mode == 1) ? 1 : 0;
    /* Decay 与分层窗口表联动（gsa_layer_window::Budget5120）：indexer layer<=18 →
     * 4096、19..62 → 5120、>=63 → 4096，窗口值 clamp 到 [topkSize, mSpaTopkLen]。 */
    mHbmWindowPlan = (mHbmEvictMode == 1) ? 2 : 0;
    mLogger.log(LogLevel::INFO,
                "GSAPrefetchEngineC set HBM evict mode: %d (%s), window plan: %d (%s)\n",
                mHbmEvictMode, mHbmEvictMode == 1 ? "decay" : "lru", mHbmWindowPlan,
                mHbmWindowPlan == 2 ? "budget5120" : "uniform");
}

int GSAPrefetchEngineC::AcquireQTopkView(torch::Tensor& qTopkTensor, const std::string& reqID,
                                         int layerID, PrefetchTopkView& topkView, int& qTopkNum,
                                         int& topkSize)
{
    const bool topkDefined = qTopkTensor.defined();
    const int64_t topkDim = topkDefined ? qTopkTensor.dim() : 0;
    if (!topkDefined || topkDim < 3 || qTopkTensor.scalar_type() != torch::kInt ||
        static_cast<int64_t>(qTopkTensor.size(2)) <= 0) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| invalid qTopkTensor, skip topk map: "
                    "reqID=%s layerID=%d defined=%d dim=%lld dtype=%d\n",
                    mDecodeStep, reqID.c_str(), layerID, topkDefined ? 1 : 0,
                    static_cast<long long>(topkDim),
                    topkDefined ? static_cast<int>(qTopkTensor.scalar_type()) : -1);
        return -1;
    }
    qTopkNum = static_cast<int>(qTopkTensor.sizes()[0]);
    topkSize = static_cast<int>(qTopkTensor.sizes()[2]);
    topkView.base = qTopkTensor.data_ptr<int>();
    topkView.stride0 = qTopkTensor.stride(0);
    topkView.numel = PrefetchTopkReadableNumel(qTopkTensor);
    /* 平坦读边界：末行读取范围不得超过张量元素数（防 OOB 读）。 */
    const int64_t lastOffset = static_cast<int64_t>(qTopkNum > 0 ? qTopkNum - 1 : 0) *
                                   topkView.stride0 +
                               static_cast<int64_t>(topkSize);
    if (lastOffset > topkView.numel) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| qTopkTensor read out of range, "
                    "skip topk map: reqID=%s layerID=%d qTopkNum=%d topkSize=%d "
                    "lastOffset=%lld numel=%lld\n",
                    mDecodeStep, reqID.c_str(), layerID, qTopkNum, topkSize,
                    static_cast<long long>(lastOffset),
                    static_cast<long long>(topkView.numel));
        return -1;
    }
    return 0;
}

int GSAPrefetchEngineC::CopyUseBlockTableToIndicesTensor(const std::string& reqID, int layerID,
                                                         const std::vector<int>& useBlockTable)
{
    // find 判别杜绝 operator[] 并发插入；张量校验失败跳过拷贝并报错。
    // 本函数不加锁，调用方必须已持 mUpdateReqMutex。
    auto tensorIt = mAllTopkIndicesTensor.find(reqID);
    if (tensorIt == mAllTopkIndicesTensor.end() || !tensorIt->second.defined() ||
        tensorIt->second.dim() < 1 ||
        static_cast<int64_t>(tensorIt->second.size(0)) <= static_cast<int64_t>(layerID)) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| mAllTopkIndicesTensor invalid, "
                    "skip copy: reqID=%s layerID=%d defined=%d dim=%lld size0=%lld\n",
                    mDecodeStep, reqID.c_str(), layerID,
                    tensorIt == mAllTopkIndicesTensor.end() ? 0
                                                            : (tensorIt->second.defined() ? 1 : 0),
                    tensorIt != mAllTopkIndicesTensor.end() && tensorIt->second.defined()
                        ? static_cast<long long>(tensorIt->second.dim())
                        : -1LL,
                    tensorIt != mAllTopkIndicesTensor.end() && tensorIt->second.defined() &&
                            tensorIt->second.dim() >= 1
                        ? static_cast<long long>(tensorIt->second.size(0))
                        : -1LL);
        return -1;
    }
    auto dstTensor = tensorIt->second[layerID];
    if (dstTensor.numel() < static_cast<int64_t>(useBlockTable.size())) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| mAllTopkIndicesTensor too small, "
                    "skip copy: reqID=%s layerID=%d numel=%lld need=%zu\n",
                    mDecodeStep, reqID.c_str(), layerID,
                    static_cast<long long>(dstTensor.numel()), useBlockTable.size());
        return -1;
    }
    std::memcpy(dstTensor.data_ptr(), useBlockTable.data(),
                useBlockTable.size() * sizeof(int));
    return 0;
}

void GSAPrefetchEngineC::SetIndexerMap(std::string reqID, std::vector<int>& blockTableList,
                                       std::vector<std::vector<int>>& remainIdx,
                                       std::vector<std::vector<int>>& prefetchIdx,
                                       std::vector<int>& blockSlot, torch::Tensor& dsaSBT)
{
    if (!mOpenIndexerOffload) { return; }
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::SetIndexerMap));
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch|"
                " Start SetIndexerMap\n",
                mDecodeStep);
    if (mBlocksMap.find(reqID) != mBlocksMap.end()) {
        mBlocksMap[reqID].clear();
        mDocsTables[reqID].clear();
        mPrefetchIdx[reqID].clear();
        mUseBlockTableIdx[reqID].clear();
        mPrefetchBlockTableIdx[reqID].clear();
        mRealTimeIdx[reqID].clear();
    }
    mIndexerLoadResultT[reqID] = dsaSBT;
    if (mIndexerSparseLen == 0) { mIndexerSparseLen = dsaSBT.sizes()[1]; }
    mUseBlockTableIdx[reqID].resize(mLayerNum);
    mPrefetchBlockTableIdx[reqID].resize(mLayerNum);
    for (int i = 0; i < mLayerNum; i++) {
        SetIndexerMapForLayer(reqID, i, blockTableList, remainIdx, prefetchIdx, blockSlot);
    }
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, "
                "|KVCache Prefetch| End SetIndexerMap\n",
                mDecodeStep);
}

void GSAPrefetchEngineC::SetIndexerMapForLayer(const std::string& reqID, int layerID,
                                               std::vector<int>& blockTableList,
                                               std::vector<std::vector<int>>& remainIdx,
                                               std::vector<std::vector<int>>& prefetchIdx,
                                               std::vector<int>& blockSlot)
{
    std::map<int, int> oneDocTable;
    std::map<int, int> oneBlockMap;
    std::vector<int> onePrefetchIdx;
    std::vector<std::pair<int, int>> oneDsaSparseBlock;
    BuildIndexerLayerTables(layerID, blockTableList, remainIdx, prefetchIdx, blockSlot, oneDocTable,
                            oneBlockMap, onePrefetchIdx, oneDsaSparseBlock);
    mDocsTables[reqID].push_back(oneDocTable);
    mBlocksMap[reqID].push_back(oneBlockMap);
    mPrefetchIdx[reqID].push_back(onePrefetchIdx);
    WriteIndexerLayerTables(reqID, layerID, oneDsaSparseBlock);
}

// 消费序：prefetch 段在前、remain 段在中、back() 给 last 块。
void GSAPrefetchEngineC::BuildIndexerLayerTables(int layerID, std::vector<int>& blockTableList,
                                                 std::vector<std::vector<int>>& remainIdx,
                                                 std::vector<std::vector<int>>& prefetchIdx,
                                                 std::vector<int>& blockSlot,
                                                 std::map<int, int>& oneDocTable,
                                                 std::map<int, int>& oneBlockMap,
                                                 std::vector<int>& onePrefetchIdx,
                                                 std::vector<std::pair<int, int>>& oneDsaSparseBlock)
{
    uint32_t offset = 0;
    for (uint32_t index = 0; index < prefetchIdx[layerID].size(); index++) {
        int idx = prefetchIdx[layerID][index];
        oneDocTable[idx] = blockTableList[index + offset];
        oneBlockMap[blockTableList[index + offset]] = idx;
        onePrefetchIdx.push_back(idx);
    }
    offset = prefetchIdx[layerID].size();
    for (uint32_t index = 0; index < remainIdx[layerID].size(); index++) {
        int idx = remainIdx[layerID][index];
        oneDocTable[idx] = blockTableList[index + offset];
        oneBlockMap[blockTableList[index + offset]] = idx;
        oneDsaSparseBlock.emplace_back(std::make_pair(idx, blockTableList[index + offset]));
    }
    oneDsaSparseBlock.emplace_back(std::make_pair(blockSlot.size() - 1, blockTableList.back()));
}

void GSAPrefetchEngineC::WriteIndexerLayerTables(const std::string& reqID, int layerID,
                                                 std::vector<std::pair<int, int>>& oneDsaSparseBlock)
{
    std::sort(oneDsaSparseBlock.begin(), oneDsaSparseBlock.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<int> sortBlockTable;
    mUseBlockTableIdx[reqID][layerID].resize(oneDsaSparseBlock.size());
    mPrefetchBlockTableIdx[reqID][layerID].resize(oneDsaSparseBlock.size());
    int pairIdx = 0;
    for (auto onePair : oneDsaSparseBlock) {
        sortBlockTable.emplace_back(onePair.second);
        mUseBlockTableIdx[reqID][layerID][pairIdx] = onePair.first;
        mPrefetchBlockTableIdx[reqID][layerID][pairIdx] = onePair.first;
        pairIdx += 1;
    }
    if (mIndexerSparseLen < sortBlockTable.size()) {
        mLogger.log(LogLevel::ERROR,
                    "[ERROR] mIndexerLoadResultT greater than mIndexerSparseLen !!! "
                    "layer_id %d, mIndexerSparseLen: %d, useBlockTable size %d\n",
                    layerID, mIndexerSparseLen, static_cast<int>(oneDsaSparseBlock.size()));
    } else {
        std::memcpy(mIndexerLoadResultT[reqID][layerID].data_ptr(), sortBlockTable.data(),
                    sortBlockTable.size() * sizeof(int));
    }
}

void GSAPrefetchEngineC::UpdateIndexerMap(std::string reqID, int idx)
{
    /* 提前识别 + 判别：生产调用链（Python update_indexer_map）只使用 "add"
     * 语义（新分配块后追加槽位索引）。守卫三连：offload 开关、请求已初始化
     * （find 判别，杜绝 operator[] 插入空条目后 [layerID] 越界 UB）、层数就绪。 */
    if (!mOpenIndexerOffload || mLayerNum <= 0) { return; }
    UReqLockGuard lock(mUpdateReqMutex, UReqLockStatsAt(UReqLockSite::UpdateIndexerMap));
    auto it = mUseBlockTableIdx.find(reqID);
    if (it == mUseBlockTableIdx.end() || it->second.size() < static_cast<size_t>(mLayerNum)) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| UpdateIndexerMap skipped: "
                    "reqID=%s not initialized, idx=%d, layerNum=%d, tableSize=%zu\n",
                    mDecodeStep, reqID.c_str(), idx, mLayerNum,
                    it == mUseBlockTableIdx.end() ? 0 : it->second.size());
        return;
    }
    for (int layerID = 0; layerID < mLayerNum; layerID++) {
        it->second[static_cast<size_t>(layerID)].emplace_back(idx);
    }
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| UpdateIndexerMap add for reqID: %s. "
                "add slot idx: %d, after add len: %zu\n",
                mDecodeStep, reqID.c_str(), idx, it->second[0].size());
}

void GSAPrefetchEngineC::TokenIndexMapInter(std::vector<int>& oneTopkList, int bsIndex, int layerID,
                                            int topkLen, const std::string& reqID)
{
    if (!mOpenIndexerOffload) { return; }
    if (mBlockSize <= 0) {
        mLogger.log(LogLevel::ERROR, "[ERROR] Trans topk blockSize invalid: %d, reqID %s\n",
                    mBlockSize, reqID.c_str());
        return;
    }
    /* 循环外一次性 find 取表与 seqLen，杜绝插空+越界 UB。 */
    auto realTimeIt = mRealTimeIdx.find(reqID);
    auto seqLenIt = mSeqLen.find(reqID);
    if (realTimeIt == mRealTimeIdx.end() || seqLenIt == mSeqLen.end() ||
        static_cast<size_t>(layerID) >= realTimeIt->second.size()) {
        mLogger.log(LogLevel::ERROR, "[ERROR] Decode step: %u Trans topk state missing: "
                    "req_id: %s, layer_id %d\n", mDecodeStep, reqID.c_str(), layerID);
        return;
    }
    const auto& realTimeIdx = realTimeIt->second[static_cast<size_t>(layerID)];
    const int seqLen = seqLenIt->second;
    /* 判别式门控：负 token / 块序号越界 / 翻译后越界三类非法输入一律
     * "日志 + 跳过翻译（保留原值）"，杜绝 OOB 读与负 slot 伪装。 */
    for (int j = 0; j < topkLen; j++) {
        const int ori = oneTopkList[j];
        if (ori < 0) {
            mLogger.log(LogLevel::ERROR, "[ERROR] Decode step: %u Trans topk ori token < 0: "
                        "layer_id %d, ori token %d, reqID %s\n",
                        mDecodeStep, layerID, ori, reqID.c_str());
            continue;
        }
        const uint32_t revIdx = static_cast<uint32_t>(ori / mBlockSize);
        if (revIdx >= realTimeIdx.size()) {
            mLogger.log(LogLevel::ERROR, "[ERROR] Decode step: %u Trans topk revIdx out of "
                        "range: layer_id %d, ori %d, revIdx %u, size %zu, reqID %s\n",
                        mDecodeStep, layerID, ori, revIdx, realTimeIdx.size(), reqID.c_str());
            continue;
        }
        const int translated = realTimeIdx[revIdx] * mBlockSize + ori % mBlockSize;
        if (translated < 0 || translated >= seqLen) {
            mLogger.log(LogLevel::ERROR, "[ERROR] Decode step: %u Trans topk out of range: "
                        "layer_id %d, ori %d, new %d, block_index %u, seq_len %d, reqID %s\n",
                        mDecodeStep, layerID, ori, translated, revIdx, seqLen, reqID.c_str());
            continue;
        }
        oneTopkList[j] = translated;
    }
}

void GSAPrefetchEngineC::LastChunkMove(int layerID, const std::string& reqID,
                                       const std::vector<int>& moveMap)
{
    if (moveMap.empty() || mBlockSize <= 0 ||
        moveMap.size() % static_cast<size_t>(mBlockSize) != 0) {
        return;
    }
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| mStoreV1_ is nullptr");
        return;
    }
    /* blockTable/hashCount 值快照 mini 段：blockTable 引用原先跨 IO 全程
     * 持有，值拷贝后锁外 erase/rehash 不再影响。 */
    std::vector<int> blockTableCopy;
    size_t hashCount = 0;
    {
        UReqLockGuard lock(mUpdateReqMutex,
                           UReqLockStatsAt(UReqLockSite::RunLastChunkMoveGuarded));
        auto hashIt = mAllBlocksHash.find(reqID);
        auto blockTableIt = mReqUseBlockTable.find(reqID);
        if (hashIt == mAllBlocksHash.end() || blockTableIt == mReqUseBlockTable.end()) {
            mLogger.log(LogLevel::ERROR, "|LastChunkMove| Cannot find hash/blockTable");
            return;
        }
        hashCount = hashIt->second.size();
        blockTableCopy = blockTableIt->second;
    }
    const std::vector<int>& blockTable = blockTableCopy;
    // sourceBlocks 仅保留非 UCM 块的 host 源地址；UCM 块的 token 由 LoadOnToken
    // 在接口内直接拷贝到 last chunk staging buffer。
    std::vector<LastChunkSourceBlock> sourceBlocks(hashCount + 1);
    UCMTokenLoadPlan plan;
    if (!PrepareLastChunkSources(layerID, reqID, moveMap, blockTable, sourceBlocks, plan)) {
        return;
    }
    if (!LoadLastChunkSources(layerID, reqID, plan)) { return; }
    if (Synchronize(mContext, mStream, 1, mLogger) != 0) { return; }
    CopyLastChunkTokens(layerID, reqID, moveMap, sourceBlocks);
    CopybackLastChunkBlocks(layerID, blockTable, moveMap);
    if (Synchronize(mContext, mStream, 1, mLogger) != 0) { return; }
}

// staging buffer → HBM 槽位整块 H2D 回写。
void GSAPrefetchEngineC::CopybackLastChunkBlocks(int layerID, const std::vector<int>& blockTable,
                                                 const std::vector<int>& moveMap)
{
    const size_t blockSize = static_cast<size_t>(mBlockSize);
    const size_t blockCount = moveMap.size() / blockSize;
    const size_t kBlockBytes = blockSize * static_cast<size_t>(mKVSizeBytesToken);
    const size_t vBlockBytes = blockSize * static_cast<size_t>(mKVSizeBytesTokenRe);
    for (size_t i = 0; i < blockCount; ++i) {
        const int hbmSlot = blockTable[i];
        void* kSrc = reinterpret_cast<void*>(mLastCHunkKPtr[layerID] + i * kBlockBytes);
        void* kDst = reinterpret_cast<void*>(mKcachePtr[layerID] +
                                             static_cast<size_t>(hbmSlot) * kBlockBytes);
        const int32_t kRet = CopyTrans(static_cast<void**>(&kSrc), static_cast<void**>(&kDst),
                                       kBlockBytes, 1, "h2d", mStream);
        if (kRet == -1) { mLogger.log(LogLevel::ERROR, "|LastChunkMove| CopyTrans H2D failed,"); }
        if (!mISSPAC8) {
            void* vSrc = reinterpret_cast<void*>(mLastCHunkVPtr[layerID] + i * vBlockBytes);
            void* vDst = reinterpret_cast<void*>(mVcachePtr[layerID] +
                                                 static_cast<size_t>(hbmSlot) * vBlockBytes);
            const int32_t vRet = CopyTrans(static_cast<void**>(&vSrc), static_cast<void**>(&vDst),
                                           vBlockBytes, 1, "h2d", mStream);
            if (vRet == -1) {
                mLogger.log(LogLevel::ERROR, "|LastChunkMove| CopyTrans H2D failed,");
            }
        }
    }
}

bool GSAPrefetchEngineC::PrepareLastChunkSources(int layerID, const std::string& reqID,
                                                 const std::vector<int>& moveMap,
                                                 const std::vector<int>& blockTable,
                                                 std::vector<LastChunkSourceBlock>& sourceBlocks,
                                                 UCMTokenLoadPlan& plan)
{
    const size_t blockSize = static_cast<size_t>(mBlockSize);
    const size_t blockCount = moveMap.size() / blockSize;
    if (blockCount > static_cast<size_t>(mLastChunkNum) || blockTable.size() < blockCount) {
        mLogger.log(
            LogLevel::ERROR,
            "|LastChunkMove| Invalid block count, reqID=%s blockCount=%zu blockTableSize=%zu\n",
            reqID.c_str(), blockCount, blockTable.size());
        return false;
    }
    for (size_t i = 0; i < blockCount; ++i) {
        if (blockTable[i] < 0 || blockTable[i] >= mKVHBMBlockLen) {
            mLogger.log(LogLevel::ERROR,
                        "|LastChunkMove| HBM slot out of range, reqID=%s hbmSlot=%d\n",
                        reqID.c_str(), blockTable[i]);
            return false;
        }
    }
    return BuildLastChunkSources(layerID, reqID, moveMap, sourceBlocks, plan);
}

bool GSAPrefetchEngineC::BuildLastChunkSources(int layerID, const std::string& reqID,
                                               const std::vector<int>& moveMap,
                                               std::vector<LastChunkSourceBlock>& sourceBlocks,
                                               UCMTokenLoadPlan& plan)
{
    /* hashVec 值快照 mini 段：token 循环及其后的 LoadOnToken DDR IO 不持锁。 */
    std::vector<BlockId> hashVecCopy;
    {
        UReqLockGuard lock(mUpdateReqMutex,
                           UReqLockStatsAt(UReqLockSite::RunLastChunkMoveGuarded));
        auto hashItLastChunk = mAllBlocksHash.find(reqID);
        if (hashItLastChunk == mAllBlocksHash.end()) {
            mLogger.log(LogLevel::ERROR,
                        "|LastChunkMove| Cannot find hashVec, reqID=%s\n", reqID.c_str());
            return false;
        }
        hashVecCopy = hashItLastChunk->second;
    }
    const auto& hashVec = hashVecCopy;
    if (layerID < 0 || static_cast<size_t>(layerID) >= mLastCHunkKPtr.size() ||
        (!mISSPAC8 && static_cast<size_t>(layerID) >= mLastCHunkVPtr.size())) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| layerID out of range, layerID=%d\n", layerID);
        return false;
    }
    auto* kDstBase = reinterpret_cast<std::byte*>(mLastCHunkKPtr[layerID]);
    std::byte* vDstBase =
        mISSPAC8 ? nullptr : reinterpret_cast<std::byte*>(mLastCHunkVPtr[layerID]);
    if (kDstBase == nullptr || (!mISSPAC8 && vDstBase == nullptr)) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| last chunk buffer is nullptr\n");
        return false;
    }
    const int lastUCMBlockIdx = static_cast<int>(hashVec.size());
    // 非 UCM 条目（blockIdx >= lastUCMBlockIdx）快照，通常仅最后一块。
    std::unordered_map<int, LastKVBlockAddress> lastKVCopy;
    {
        UReqLockGuard lock(mUpdateReqMutex,
                           UReqLockStatsAt(UReqLockSite::RunLastChunkMoveGuarded));
        auto lastKVReqIt = mLastKVBlockAddress.find(reqID);
        if (lastKVReqIt == mLastKVBlockAddress.end()) {
            mLogger.log(LogLevel::ERROR, "|LastChunkMove| no non-UCM block\n");
            return false;
        }
        for (const auto& kv : lastKVReqIt->second) {
            if (kv.first >= lastUCMBlockIdx) { lastKVCopy.emplace(kv.first, kv.second); }
        }
    }
    return AppendLastChunkTokens(layerID, reqID, moveMap, hashVec, lastKVCopy, lastUCMBlockIdx,
                                 sourceBlocks, plan);
}

// 每 UCM 块一个 shard：目的地址为该块首 token 在 staging buffer 的落位；
// 块内 token 逐条追加 K/V TokenRange（dstOffset 相对 shard 首地址）。
bool GSAPrefetchEngineC::AppendLastChunkTokens(int layerID, const std::string& reqID,
                                              const std::vector<int>& moveMap,
                                              const std::vector<BlockId>& hashVec,
                                              const std::unordered_map<int, LastKVBlockAddress>& lastKVCopy,
                                              int lastUCMBlockIdx,
                                              std::vector<LastChunkSourceBlock>& sourceBlocks,
                                              UCMTokenLoadPlan& plan)
{
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    std::vector<size_t> blockFirstDst(hashVec.size(), 0);
    std::vector<int> blockShardIdx(hashVec.size(), -1);
    for (size_t i = 0; i < moveMap.size(); ++i) {
        const int globalToken = moveMap[i];
        if (globalToken < 0) {
            mLogger.log(LogLevel::ERROR,
                        "|LastChunkMove| Invalid global token, reqID=%s token=%d\n", reqID.c_str(),
                        globalToken);
            return false;
        }
        const int blockIdx = globalToken / mBlockSize;
        if (blockIdx >= lastUCMBlockIdx) {
            if (!BindNonUcmSourceBlock(blockIdx, layerID, lastKVCopy, sourceBlocks)) {
                return false;
            }
            continue;
        }
        const size_t blockIdxSz = static_cast<size_t>(blockIdx);
        if (blockShardIdx[blockIdxSz] < 0) { OpenUcmShard(i, blockIdxSz, layerID, hashVec,
                                                           blockFirstDst, blockShardIdx, plan); }
        const size_t shardIdx = static_cast<size_t>(blockShardIdx[blockIdxSz]);
        const size_t tokenIdxInBlock = static_cast<size_t>(globalToken % mBlockSize);
        const size_t dstOffset = i - blockFirstDst[blockIdxSz];
        plan.kFilters[shardIdx].ranges[kKTensorIndex].push_back(UC::Detail::TokenRange{
            tokenIdxInBlock * kTokenBytes, kTokenBytes, dstOffset * kTokenBytes});
        if (!mISSPAC8) {
            plan.vFilters[shardIdx].ranges[kVTensorIndex].push_back(
                UC::Detail::TokenRange{tokenIdxInBlock * vTokenBytes, vTokenBytes,
                                       dstOffset * vTokenBytes});
        }
    }
    return true;
}

// 块首触：为该 UCM 块开 K/V shard（目的窗口 = staging buffer 中首 token 落位）。
void GSAPrefetchEngineC::OpenUcmShard(size_t i, size_t blockIdxSz, int layerID,
                                     const std::vector<BlockId>& hashVec,
                                     std::vector<size_t>& blockFirstDst,
                                     std::vector<int>& blockShardIdx, UCMTokenLoadPlan& plan)
{
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDstBase = reinterpret_cast<std::byte*>(mLastCHunkKPtr[layerID]);
    std::byte* vDstBase =
        mISSPAC8 ? nullptr : reinterpret_cast<std::byte*>(mLastCHunkVPtr[layerID]);
    blockFirstDst[blockIdxSz] = i;
    blockShardIdx[blockIdxSz] = static_cast<int>(plan.kTask.size());
    plan.kTask.emplace_back(UC::Detail::Shard{
        hashVec[blockIdxSz], static_cast<size_t>(layerID),
        std::vector<void*>{kDstBase + i * kTokenBytes}});
    UC::Detail::TokenFilter kFilter;
    kFilter.ranges.resize(mStoreTensorCount);
    plan.kFilters.push_back(std::move(kFilter));
    if (!mISSPAC8) {
        plan.vTask.emplace_back(UC::Detail::Shard{
            hashVec[blockIdxSz], static_cast<size_t>(layerID),
            std::vector<void*>{vDstBase + i * vTokenBytes}});
        UC::Detail::TokenFilter vFilter;
        vFilter.ranges.resize(mStoreTensorCount);
        plan.vFilters.push_back(std::move(vFilter));
    }
}

// 非 UCM 块：绑定 host 源地址（+layerID 块内偏移）到 sourceBlocks。
bool GSAPrefetchEngineC::BindNonUcmSourceBlock(int blockIdx, int layerID,
                                              const std::unordered_map<int, LastKVBlockAddress>& lastKVCopy,
                                              std::vector<LastChunkSourceBlock>& sourceBlocks)
{
    auto blockIt = lastKVCopy.find(blockIdx);
    if (blockIt == lastKVCopy.end()) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| non-UCM block not found\n");
        return false;
    }
    const size_t kBlockBytes =
        static_cast<size_t>(mBlockSize) * static_cast<size_t>(mKVSizeBytesToken);
    const size_t vBlockBytes =
        static_cast<size_t>(mBlockSize) * static_cast<size_t>(mKVSizeBytesTokenRe);
    if (sourceBlocks.size() <= static_cast<size_t>(blockIdx))
        sourceBlocks.resize(static_cast<size_t>(blockIdx) + 1);
    sourceBlocks[blockIdx].kBase =
        reinterpret_cast<const std::byte*>(blockIt->second.kAddr) +
        static_cast<size_t>(layerID) * kBlockBytes;
    if (!mISSPAC8) {
        sourceBlocks[blockIdx].vBase =
            reinterpret_cast<const std::byte*>(blockIt->second.vAddr) +
            static_cast<size_t>(layerID) * vBlockBytes;
    }
    return true;
}

bool GSAPrefetchEngineC::LoadLastChunkSources(int layerID, const std::string& reqID,
                                              const UCMTokenLoadPlan& plan)
{
    if (plan.kTask.empty() && plan.vTask.empty()) { return true; }
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| mStoreV1_ is nullptr, reqID=%s layerID=%d\n",
                    reqID.c_str(), layerID);
        return false;
    }
    return SubmitUCMTokenLoad(plan, "LastChunkMove", layerID, reqID);
}

bool GSAPrefetchEngineC::SubmitUCMTokenLoad(const UCMTokenLoadPlan& plan, const char* stage,
                                            int layerID, const std::string& reqID)
{
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR, "|%s| mStoreV1_ is nullptr, reqID=%s layerID=%d\n", stage,
                    reqID.c_str(), layerID);
        return false;
    }
    if (!SubmitUCMTokenLoadLeg(plan.kTask, plan.kFilters, "K", stage, layerID, reqID)) {
        return false;
    }
    if (!mISSPAC8 &&
        !SubmitUCMTokenLoadLeg(plan.vTask, plan.vFilters, "V", stage, layerID, reqID)) {
        return false;
    }
    return true;
}

bool GSAPrefetchEngineC::SubmitUCMTokenLoadLeg(const UC::Detail::TaskDesc& task,
                                               const std::vector<UC::Detail::TokenFilter>& filters,
                                               const char* tensorName, const char* stage,
                                               int layerID, const std::string& reqID)
{
    if (task.empty()) { return true; }
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR,
                    "|%s| LoadOnToken(%s) mStoreV1_ is nullptr, reqID=%s layerID=%d\n", stage,
                    tensorName, reqID.c_str(), layerID);
        return false;
    }
    if (filters.size() != task.size()) {
        mLogger.log(LogLevel::ERROR,
                    "|%s| LoadOnToken(%s) filter size mismatch, reqID=%s layerID=%d "
                    "shards=%zu filters=%zu\n",
                    stage, tensorName, reqID.c_str(), layerID, task.size(), filters.size());
        return false;
    }
    UC::Detail::TaskDesc submitTask = task;
    submitTask.brief = std::string(stage) + "LoadOnToken" + tensorName;
    submitTask.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    auto loadResult = mStoreV1_->LoadOnToken(std::move(submitTask), filters);
    if (!loadResult) {
        mLogger.log(LogLevel::ERROR,
                    "|%s| LoadOnToken(%s) failed, reqID=%s layerID=%d shards=%zu error=%s\n", stage,
                    tensorName, reqID.c_str(), layerID, task.size(),
                    loadResult.Error().ToString().c_str());
        return false;
    }
    auto waitStatus = mStoreV1_->Wait(loadResult.Value());
    if (waitStatus.Failure()) {
        const std::string error = waitStatus.ToString();
        mLogger.log(LogLevel::ERROR,
                    "|%s| LoadOnToken(%s) Wait failed, reqID=%s layerID=%d shards=%zu error=%s\n",
                    stage, tensorName, reqID.c_str(), layerID, task.size(), error.c_str());
        return false;
    }
    return true;
}

void GSAPrefetchEngineC::CopyLastChunkTokens(int layerID, const std::string& reqID,
                                             const std::vector<int>& moveMap,
                                             const std::vector<LastChunkSourceBlock>& sourceBlocks)
{
    constexpr size_t kTwoThreadTokenThreshold = 256;
    constexpr size_t kFourThreadTokenThreshold = 512;
    constexpr int kMaxTokenThreadNum = 4;
    const size_t tokenCount = moveMap.size();
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDstBase = reinterpret_cast<std::byte*>(mLastCHunkKPtr[layerID]);
    std::byte* vDstBase = nullptr;
    if (!mISSPAC8) { vDstBase = reinterpret_cast<std::byte*>(mLastCHunkVPtr[layerID]); }
    int tokenThreadNum = 1;
    if (tokenCount >= kFourThreadTokenThreshold) {
        tokenThreadNum = mThreadNum < kMaxTokenThreadNum ? mThreadNum : kMaxTokenThreadNum;
    } else if (tokenCount >= kTwoThreadTokenThreshold) {
        tokenThreadNum = (mThreadNum < 2) ? mThreadNum : 2;
    }
    if (tokenThreadNum < 1) tokenThreadNum = 1;
    auto copyOneToken = [&](size_t i) {
        const int globalToken = moveMap[i];
        const size_t blockIdx = static_cast<size_t>(globalToken / mBlockSize);
        if (blockIdx >= sourceBlocks.size()) { return; }
        const auto& src = sourceBlocks[blockIdx];
        // UCM 块 kBase 为空：其 token 已由 LoadOnToken 在接口内完成落位。
        if (src.kBase == nullptr) { return; }
        const size_t tokenIdx = static_cast<size_t>(globalToken % mBlockSize);
        std::memcpy(kDstBase + i * kTokenBytes, src.kBase + tokenIdx * kTokenBytes, kTokenBytes);
        if (!mISSPAC8) {
            std::memcpy(vDstBase + i * vTokenBytes, src.vBase + tokenIdx * vTokenBytes,
                        vTokenBytes);
        }
    };
    if (tokenThreadNum == 1) {
        for (size_t i = 0; i < tokenCount; ++i) copyOneToken(i);
    } else {
        const long long count = static_cast<long long>(tokenCount);
#pragma omp parallel for num_threads(tokenThreadNum) proc_bind(close) schedule(static)
        for (long long i = 0; i < count; ++i) copyOneToken(static_cast<size_t>(i));
    }
}

bool GSAPrefetchEngineC::BuildUCMMergeLoadPlan(UCMTokenLoadPlan& plan,
                                               const std::vector<UCMMergeBlockContext>& blockContexts,
                                               int layerID, int streamIdx, const std::string& reqID,
                                               size_t previousLoadNum)
{
    // 当前 rank 无 UCM 加载任务时，plan 为空，不依赖 Store。
    const auto& hashVec = mAllBlocksHash.at(reqID);
    const auto& missGlobalIds = mMissGlobalToken[reqID][mTopkLayerMap[layerID]];
    if (streamIdx < 0 || static_cast<size_t>(streamIdx) >= mSharedKBufferHostPtr.size() ||
        (!mISSPAC8 && static_cast<size_t>(streamIdx) >= mSharedVBufferHostPtr.size())) {
        mLogger.log(LogLevel::ERROR,
                    "|UCMTensorMerge| streamIdx out of range, reqID=%s streamIdx=%d\n",
                    reqID.c_str(), streamIdx);
        return false;
    }
    auto* kDstBase = reinterpret_cast<std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    std::byte* vDstBase =
        mISSPAC8 ? nullptr : reinterpret_cast<std::byte*>(mSharedVBufferHostPtr[streamIdx]);
    if (kDstBase == nullptr || (!mISSPAC8 && vDstBase == nullptr)) {
        mLogger.log(LogLevel::ERROR, "|UCMTensorMerge| shared buffer is nullptr, reqID=%s\n",
                    reqID.c_str());
        return false;
    }
    // 仅本 rank 拥有的 UCM 块需要 LoadOnToken；目的首地址为该块首个 token 在
    // 共享 staging buffer 中的落位，块内 token 逐条追加 TokenRange。
    for (const auto& context : blockContexts) {
        if (!context.owned || context.isnoUCMblock) { continue; }
        AppendMergeBlockShard(context, hashVec, missGlobalIds, layerID, kDstBase, vDstBase,
                              previousLoadNum, plan);
    }
    return true;
}

// 单个 owned UCM 块的 K/V shard 与 TokenRange 组装。
void GSAPrefetchEngineC::AppendMergeBlockShard(const UCMMergeBlockContext& context,
                                              const std::vector<BlockId>& hashVec,
                                              const std::vector<int>& missGlobalIds, int layerID,
                                              std::byte* kDstBase, std::byte* vDstBase,
                                              size_t previousLoadNum, UCMTokenLoadPlan& plan)
{
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    const auto& tokenIndices = *context.tokenIndices;
    const size_t firstDst = previousLoadNum + context.tokenBase;
    plan.kTask.emplace_back(UC::Detail::Shard{
        hashVec[static_cast<size_t>(context.blockIdx)], static_cast<size_t>(layerID),
        std::vector<void*>{kDstBase + firstDst * kTokenBytes}});
    UC::Detail::TokenFilter kFilter;
    kFilter.ranges.resize(mStoreTensorCount);
    for (size_t localTokenIdx = 0; localTokenIdx < tokenIndices.size(); ++localTokenIdx) {
        const int globalToken = missGlobalIds[tokenIndices[localTokenIdx]];
        const size_t tokenIdxInBlock = static_cast<size_t>(globalToken % mBlockSize);
        kFilter.ranges[kKTensorIndex].push_back(UC::Detail::TokenRange{
            tokenIdxInBlock * kTokenBytes, kTokenBytes, localTokenIdx * kTokenBytes});
    }
    plan.kFilters.push_back(std::move(kFilter));
    if (mISSPAC8) { return; }
    plan.vTask.emplace_back(UC::Detail::Shard{
        hashVec[static_cast<size_t>(context.blockIdx)], static_cast<size_t>(layerID),
        std::vector<void*>{vDstBase + firstDst * vTokenBytes}});
    UC::Detail::TokenFilter vFilter;
    vFilter.ranges.resize(mStoreTensorCount);
    for (size_t localTokenIdx = 0; localTokenIdx < tokenIndices.size(); ++localTokenIdx) {
        const int globalToken = missGlobalIds[tokenIndices[localTokenIdx]];
        const size_t tokenIdxInBlock = static_cast<size_t>(globalToken % mBlockSize);
        vFilter.ranges[kVTensorIndex].push_back(UC::Detail::TokenRange{
            tokenIdxInBlock * vTokenBytes, vTokenBytes, localTokenIdx * vTokenBytes});
    }
    plan.vFilters.push_back(std::move(vFilter));
}

bool GSAPrefetchEngineC::MemcpyUCMMergeTokens(std::vector<UCMMergeBlockContext>& blockContexts,
                                               int layerID, int streamIdx,
                                               std::vector<uint64_t>& hbmSlotIndex,
                                               const std::string& reqID, size_t previousLoadNum)
{
    // constexpr int kMaxTokenThreadNum = 4;
    // constexpr size_t kTwoThreadTokenThreshold = 256;
    // constexpr size_t kFourThreadTokenThreshold = 512;
    int curTopkIndex = mTopkLayerMap[layerID];
    const size_t tokenCount = mMissGlobalToken[reqID][curTopkIndex].size();
    const size_t finalLoadNum = previousLoadNum + tokenCount;
    if (hbmSlotIndex.capacity() < finalLoadNum) {
        hbmSlotIndex.reserve(static_cast<size_t>(mLoadIndexBufferLen));
    }
    hbmSlotIndex.resize(finalLoadNum);
    if (!PrepareUCMMergeSources(blockContexts, layerID, reqID)) { return false; }
    int tokenThreadNum = 1;
    // if (tokenCount >= kFourThreadTokenThreshold) {
    //     tokenThreadNum = kMaxTokenThreadNum;
    // } else if (tokenCount >= kTwoThreadTokenThreshold) {
    //     tokenThreadNum = 2;
    // }
    if (tokenThreadNum == 1) {
        CopyUCMMergeTokensSerial(blockContexts, curTopkIndex, streamIdx, hbmSlotIndex, reqID,
                                 previousLoadNum);
    } else {
        CopyUCMMergeTokensParallel(blockContexts, curTopkIndex, streamIdx, hbmSlotIndex, reqID,
                                   previousLoadNum, tokenThreadNum);
    }
    return true;
}

bool GSAPrefetchEngineC::PrepareUCMMergeSources(std::vector<UCMMergeBlockContext>& blockContexts,
                                                int layerID, const std::string& reqID)
{
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    const size_t kBlockBytes = static_cast<size_t>(mBlockSize) * kTokenBytes;
    const size_t vBlockBytes = static_cast<size_t>(mBlockSize) * vTokenBytes;
    const auto addressIt = mLastKVBlockAddress.find(reqID);
    for (auto& context : blockContexts) {
        /* 仅 owned 的非 UCM 块仍需引擎侧源地址：owned 的 UCM 块已由
         * LoadOnToken 在接口内直接落位，非 owned 块由拥有它的 rank 填充。 */
        if (!context.owned || !context.isnoUCMblock) { continue; }
        {
            if (addressIt == mLastKVBlockAddress.end()) { return false; }
            auto blockAddressIt = addressIt->second.find(context.blockIdx);
            if (blockAddressIt == addressIt->second.end()) { return false; }
            const auto& blockAddress = blockAddressIt->second;
            if (blockAddress.kAddr == 0) { return false; }
            if (!mISSPAC8 && blockAddress.vAddr == 0) { return false; }
            context.kSrc = reinterpret_cast<const std::byte*>(blockAddress.kAddr) +
                           static_cast<size_t>(layerID) * kBlockBytes;
            if (!mISSPAC8) {
                context.vSrc = reinterpret_cast<const std::byte*>(blockAddress.vAddr) +
                               static_cast<size_t>(layerID) * vBlockBytes;
            }
        }
    }
    return true;
}

void GSAPrefetchEngineC::CopyUCMMergeTokensSerial(
    const std::vector<UCMMergeBlockContext>& blockContexts, int layerID, int streamIdx,
    std::vector<uint64_t>& hbmSlotIndex, const std::string& reqID, size_t previousLoadNum)
{
    const auto& missGlobalIds = mMissGlobalToken[reqID][layerID];
    const auto& missSlots = mLoadSlot[reqID][layerID];
    const auto& blockTable = mReqUseBlockTable.at(reqID);
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDstBase = reinterpret_cast<std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    std::byte* vDstBase = nullptr;
    if (!mISSPAC8) { vDstBase = reinterpret_cast<std::byte*>(mSharedVBufferHostPtr[streamIdx]); }
    for (const auto& context : blockContexts) {
        const auto& tokenIndices = *context.tokenIndices;
        for (size_t localTokenIdx = 0; localTokenIdx < tokenIndices.size(); ++localTokenIdx) {
            const size_t idx = tokenIndices[localTokenIdx];
            const int globalToken = missGlobalIds[idx];
            const size_t tokenOffset = static_cast<size_t>(globalToken % mBlockSize);
            const int slotID = missSlots[idx];
            const int hbmIdx = blockTable[static_cast<size_t>(slotID / mBlockSize)];
            const int hbmSlot = slotID % mBlockSize;
            const size_t dstOffset = previousLoadNum + context.tokenBase + localTokenIdx;
            /* owned 的 UCM 块已由 LoadOnToken 落位，仅 owned 非 UCM 块需要本地拷贝。 */
            if (context.owned && context.isnoUCMblock) {
                CopySingleTokenKV(kDstBase, vDstBase, context, tokenOffset, dstOffset, kTokenBytes,
                                  vTokenBytes);
            }
            hbmSlotIndex[dstOffset] =
                static_cast<uint64_t>(hbmIdx) * static_cast<uint64_t>(mBlockSize) +
                static_cast<uint64_t>(hbmSlot);
        }
    }
}

void GSAPrefetchEngineC::CopySingleTokenKV(std::byte* kDstBase, std::byte* vDstBase,
                                           const UCMMergeBlockContext& context, size_t tokenOffset,
                                           size_t dstOffset, size_t kTokenBytes, size_t vTokenBytes)
{
    std::memcpy(kDstBase + dstOffset * kTokenBytes, context.kSrc + tokenOffset * kTokenBytes,
                kTokenBytes);
    if (!mISSPAC8) {
        std::memcpy(vDstBase + dstOffset * vTokenBytes, context.vSrc + tokenOffset * vTokenBytes,
                    vTokenBytes);
    }
}

void GSAPrefetchEngineC::CopyUCMMergeTokensParallel(
    const std::vector<UCMMergeBlockContext>& blockContexts, int layerID, int streamIdx,
    std::vector<uint64_t>& hbmSlotIndex, const std::string& reqID, size_t previousLoadNum,
    int tokenThreadNum)
{
    const auto& missGlobalIds = mMissGlobalToken[reqID][layerID];
    const auto& missSlots = mLoadSlot[reqID][layerID];
    const auto& blockTable = mReqUseBlockTable.at(reqID);
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDstBase = reinterpret_cast<std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    std::byte* vDstBase = nullptr;
    if (!mISSPAC8) { vDstBase = reinterpret_cast<std::byte*>(mSharedVBufferHostPtr[streamIdx]); }
    std::vector<TokenCopyTask> tokenCopyTasks;
    BuildTokenCopyTasks(blockContexts, tokenCopyTasks, missGlobalIds, missSlots, blockTable,
                        kTokenBytes, vTokenBytes, previousLoadNum);
    const long long taskCount = static_cast<long long>(tokenCopyTasks.size());
#pragma omp parallel for num_threads(tokenThreadNum) proc_bind(close) schedule(static)
    for (long long i = 0; i < taskCount; ++i) {
        const auto& copyTask = tokenCopyTasks[static_cast<size_t>(i)];
        if (copyTask.localCopy) {
            std::memcpy(kDstBase + copyTask.dstOffset * kTokenBytes, copyTask.kSrc, kTokenBytes);
            if (!mISSPAC8) {
                std::memcpy(vDstBase + copyTask.dstOffset * vTokenBytes, copyTask.vSrc,
                            vTokenBytes);
            }
        }
        hbmSlotIndex[copyTask.dstOffset] = copyTask.hbmSlot;
    }
}

void GSAPrefetchEngineC::BuildTokenCopyTasks(const std::vector<UCMMergeBlockContext>& blockContexts,
                                             std::vector<TokenCopyTask>& tokenCopyTasks,
                                             const std::vector<int>& missGlobalIds,
                                             const std::vector<int>& missSlots,
                                             const std::vector<int>& blockTable, size_t kTokenBytes,
                                             size_t vTokenBytes, size_t previousLoadNum)
{
    for (const auto& context : blockContexts) {
        const auto& tokenIndices = *context.tokenIndices;
        for (size_t localTokenIdx = 0; localTokenIdx < tokenIndices.size(); ++localTokenIdx) {
            const size_t idx = tokenIndices[localTokenIdx];
            const int globalToken = missGlobalIds[idx];
            const size_t tokenOffset = static_cast<size_t>(globalToken % mBlockSize);
            const int slotID = missSlots[idx];
            const int hbmIdx = blockTable[static_cast<size_t>(slotID / mBlockSize)];
            const int hbmSlot = slotID % mBlockSize;
            const size_t dstOffset = previousLoadNum + context.tokenBase + localTokenIdx;
            /* owned 的 UCM 块已由 LoadOnToken 落位：仅 owned 非 UCM 块需要本地拷贝。 */
            const bool localCopy = context.owned && context.isnoUCMblock;
            if (!mISSPAC8) {
                tokenCopyTasks.push_back(
                    {localCopy ? context.kSrc + tokenOffset * kTokenBytes : nullptr,
                     localCopy ? context.vSrc + tokenOffset * vTokenBytes : nullptr, dstOffset,
                     static_cast<uint64_t>(hbmIdx) * static_cast<uint64_t>(mBlockSize) +
                         static_cast<uint64_t>(hbmSlot),
                     localCopy});
            } else {
                tokenCopyTasks.push_back(
                    {localCopy ? context.kSrc + tokenOffset * kTokenBytes : nullptr, nullptr,
                     dstOffset,
                     static_cast<uint64_t>(hbmIdx) * static_cast<uint64_t>(mBlockSize) +
                         static_cast<uint64_t>(hbmSlot),
                     localCopy});
            }
        }
    }
}

size_t GSAPrefetchEngineC::FillFailedUCMMerge(
    const std::vector<UCMMergeBlockContext>& blockContexts, int layerID, int streamIdx,
    std::vector<uint64_t>& hbmSlotIndex, const std::string& reqID, size_t previousLoadNum,
    bool zeroUcmOnly)
{
    const int topkIndex = mTopkLayerMap[layerID];
    const auto& missSlots = mLoadSlot.at(reqID)[topkIndex];
    const auto& blockTable = mReqUseBlockTable.at(reqID);
    const size_t finalLoadNum = previousLoadNum + missSlots.size();
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDst = reinterpret_cast<std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    auto* vDst =
        mISSPAC8 ? nullptr : reinterpret_cast<std::byte*>(mSharedVBufferHostPtr[streamIdx]);

    // Full request layout is preserved on every rank, including non-owned tokens.
    if (hbmSlotIndex.capacity() < finalLoadNum) {
        hbmSlotIndex.reserve(static_cast<size_t>(mLoadIndexBufferLen));
    }
    hbmSlotIndex.resize(finalLoadNum);

    size_t zeroTokens = 0;
    for (const auto& context : blockContexts) {
        const auto& tokenIndices = *context.tokenIndices;
        const size_t dstBase = previousLoadNum + context.tokenBase;
        /* zeroUcmOnly: the store-side load failed AFTER the local non-UCM
         * copies already landed, so keep them and only zero the UCM windows
         * (the store never splices a failed task). */
        const bool zeroThis = context.owned && !tokenIndices.empty() &&
                              (!zeroUcmOnly || !context.isnoUCMblock);
        if (zeroThis) {
            std::memset(kDst + dstBase * kTokenBytes, 0, tokenIndices.size() * kTokenBytes);
            if (!mISSPAC8) {
                std::memset(vDst + dstBase * vTokenBytes, 0, tokenIndices.size() * vTokenBytes);
            }
            zeroTokens += tokenIndices.size();
        }
        // All ranks build indices for ALL tokens, ownership aside.
        BuildFallbackSlotIndices(context, dstBase, missSlots, blockTable, hbmSlotIndex, layerID,
                                 reqID);
    }
    return zeroTokens;
}

// Zero-fallback 期间的 HBM 槽位索引回填：非法目的元数据按 abort 语义处理。
void GSAPrefetchEngineC::BuildFallbackSlotIndices(const UCMMergeBlockContext& context, size_t dstBase,
                                                  const std::vector<int>& missSlots,
                                                  const std::vector<int>& blockTable,
                                                  std::vector<uint64_t>& hbmSlotIndex, int layerID,
                                                  const std::string& reqID)
{
    const auto& tokenIndices = *context.tokenIndices;
    for (size_t localIdx = 0; localIdx < tokenIndices.size(); ++localIdx) {
        const size_t idx = tokenIndices[localIdx];
        const int slotID = missSlots[idx];
        if (slotID < 0 || static_cast<size_t>(slotID / mBlockSize) >= blockTable.size() ||
            blockTable[static_cast<size_t>(slotID / mBlockSize)] < 0) {
            mLogger.log(LogLevel::ERROR,
                        "|KVPack| invalid HBM slot during zero fallback: "
                        "rank=%d layer=%d reqID=%s slot=%d tableSize=%zu\n",
                        mRank, layerID, reqID.c_str(), slotID, blockTable.size());
            // Invalid destination metadata is separate from load failure.
            KVPackAbort(mKVPackControl);
        }
        const int hbmIdx = blockTable[static_cast<size_t>(slotID / mBlockSize)];
        hbmSlotIndex[dstBase + localIdx] =
            static_cast<uint64_t>(hbmIdx) * static_cast<uint64_t>(mBlockSize) +
            static_cast<uint64_t>(slotID % mBlockSize);
    }
}

void GSAPrefetchEngineC::UCMTensorMerge(int layerID, int streamIdx,
                                        std::vector<uint64_t>& hbmSlotIndex,
                                        const std::string& reqID, uint32_t requestIndex)
{
    /* Single-request compatibility entry (tests / legacy callers): delegate to
     * the whole-batch merge with exactly one request. */
    TaskInfo one;
    one.reqIdList = {reqID};
    one.topkLenList = {1};
    one.layerID = layerID;
    UCMTensorMergeBatch(one, {requestIndex}, layerID, streamIdx, hbmSlotIndex);
}

bool GSAPrefetchEngineC::ValidateMergeStagingWindow(int streamIdx)
{
    if (streamIdx < 0 || static_cast<size_t>(streamIdx) >= mSharedKBufferHostPtr.size() ||
        (!mISSPAC8 && static_cast<size_t>(streamIdx) >= mSharedVBufferHostPtr.size())) {
        mLogger.log(LogLevel::ERROR,
                    "|UCMTensorMerge| streamIdx out of range, streamIdx=%d\n", streamIdx);
        return false;
    }
    auto* kDstBase = reinterpret_cast<const std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    const std::byte* vDstBase =
        mISSPAC8 ? nullptr
                 : reinterpret_cast<const std::byte*>(mSharedVBufferHostPtr[streamIdx]);
    if (kDstBase == nullptr || (!mISSPAC8 && vDstBase == nullptr)) {
        mLogger.log(LogLevel::ERROR, "|UCMTensorMerge| shared buffer is nullptr\n");
        return false;
    }
    return true;
}

void GSAPrefetchEngineC::UCMTensorMergeBatch(const TaskInfo& taskInfo,
                                             const std::vector<uint32_t>& requestIndices,
                                             int layerID, int streamIdx,
                                             std::vector<uint64_t>& hbmSlotIndex)
{
    const bool stagingOk = ValidateMergeStagingWindow(streamIdx);
    UCMTokenLoadPlan plan;
    std::vector<UCMMergeRequestState> requests;
    size_t runningLoadNum = static_cast<size_t>(mLoadNum[layerID]);
    bool planBuildOk = true;
    CollectUCMMergeRequests(taskInfo, requestIndices, layerID, streamIdx, plan, requests,
                            runningLoadNum, planBuildOk);
    const bool planOk = stagingOk && planBuildOk;
    const bool loadOk = SubmitUCMMergeBatchLoad(plan, requests, layerID, planOk);
    std::vector<std::string> localFailed;
    if (planOk) {
        for (auto& request : requests) {
            if (!MemcpyUCMMergeTokens(request.contexts, layerID, streamIdx, hbmSlotIndex,
                                      request.reqID, request.previousLoadNum)) {
                const size_t zeroTokens =
                    FillFailedUCMMerge(request.contexts, layerID, streamIdx, hbmSlotIndex,
                                       request.reqID, request.previousLoadNum);
                mLogger.log(LogLevel::ERROR,
                            "|KVPack| zero fallback (local copy): rank=%d layer=%d reqID=%s "
                            "zeroOwnedTokens=%zu\n",
                            mRank, layerID, request.reqID.c_str(), zeroTokens);
                localFailed.push_back(request.reqID);
            }
        }
    }
    FallbackUCMMergeRequests(requests, layerID, streamIdx, hbmSlotIndex, planOk, loadOk,
                             localFailed);
    mLoadNum[layerID] = static_cast<uint32_t>(runningLoadNum);
}

void GSAPrefetchEngineC::CollectUCMMergeRequests(const TaskInfo& taskInfo,
                                                 const std::vector<uint32_t>& requestIndices,
                                                 int layerID, int streamIdx, UCMTokenLoadPlan& plan,
                                                 std::vector<UCMMergeRequestState>& requests,
                                                 size_t& runningLoadNum, bool& planBuildOk)
{
    /* Phase 1: build the whole-batch plan (one shard per owned UCM block,
     * concatenated across requests; per-shard dst = the block's landing window
     * in the shared staging buffer). Ownership stays per-request:
     * (blockOrdinalInRequest + requestIndex) % mTpSize == mRank. */
    for (uint32_t requestIndex : requestIndices) {
        if (requestIndex >= taskInfo.reqIdList.size() ||
            requestIndex >= taskInfo.topkLenList.size()) {
            continue;
        }
        const std::string& reqID = taskInfo.reqIdList[requestIndex];
        if (!ValidateUCMMergeRequest(layerID, reqID, runningLoadNum)) { continue; }
        UCMMergeRequestState request;
        request.reqID = reqID;
        request.previousLoadNum = runningLoadNum;
        BuildUCMMergeBlockContexts(layerID, reqID, requestIndex, request.contexts,
                                   request.tokenMap);
        if (!BuildUCMMergeLoadPlan(plan, request.contexts, layerID, streamIdx, reqID,
                                   runningLoadNum)) {
            /* Plan-build failures are staging-window conditions shared by every
             * request: keep accumulating so each request still gets its zero
             * fallback and advances mLoadNum (same as the per-request flow). */
            mLogger.log(LogLevel::ERROR,
                        "|UCMTensorMerge| plan build failed, layer=%d reqID=%s\n", layerID,
                        reqID.c_str());
            planBuildOk = false;
        }
        runningLoadNum += mMissGlobalToken[reqID][mTopkLayerMap[layerID]].size();
        requests.push_back(std::move(request));
    }
}

bool GSAPrefetchEngineC::SubmitUCMMergeBatchLoad(const UCMTokenLoadPlan& plan,
                                                 const std::vector<UCMMergeRequestState>& requests,
                                                 int layerID, bool planOk)
{
    /* 批量 LoadOnToken（K 后 V）在本线程直接提交（原为 std::async worker，
     * 现 worker 路径无异常源、spawn 失败不会静默跳过 zero fallback）。 */
    if (!planOk || (plan.kTask.empty() && plan.vTask.empty())) {
        return planOk;
    }
    if (requests.empty()) {
        mLogger.log(LogLevel::ERROR,
                    "|UCMTensorMerge| no valid requests to load, layer=%d\n", layerID);
        return false;
    }
    const std::string& firstReqID = requests.front().reqID;
    return SubmitUCMTokenLoad(plan, "UCMTensorMerge", layerID, firstReqID);
}

void GSAPrefetchEngineC::FallbackUCMMergeRequests(const std::vector<UCMMergeRequestState>& requests,
                                                  int layerID, int streamIdx,
                                                  std::vector<uint64_t>& hbmSlotIndex, bool planOk,
                                                  bool loadOk,
                                                  const std::vector<std::string>& localFailed)
{
    /* Fallback：plan 失败时本地拷贝未跑，零填全部 owned token；load 失败时
     * 本地拷贝已落位，仅零填 UCM 窗口。hbmSlotIndex 对所有 rank 全量回填。 */
    if (planOk && loadOk) {
        return;
    }
    for (auto& request : requests) {
        if (std::find(localFailed.begin(), localFailed.end(), request.reqID) !=
            localFailed.end()) {
            continue;
        }
        const size_t zeroTokens =
            FillFailedUCMMerge(request.contexts, layerID, streamIdx, hbmSlotIndex,
                               request.reqID, request.previousLoadNum, /*zeroUcmOnly=*/planOk);
        mLogger.log(LogLevel::ERROR,
                    "|KVPack| zero fallback (batched load): rank=%d layer=%d reqID=%s "
                    "stage=%s zeroOwnedTokens=%zu\n",
                    mRank, layerID, request.reqID.c_str(), planOk ? "load" : "plan", zeroTokens);
    }
}

bool GSAPrefetchEngineC::ValidateUCMMergeRequest(int layerID, const std::string& reqID,
                                                 size_t previousLoadNum)
{
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR, "|UCMTensorMerge| mStoreV1_ is nullptr reqID=%s\n",
                    reqID.c_str());
    }
    int curTopkIndex = mTopkLayerMap[layerID];
    const auto& missGlobalIds = mMissGlobalToken[reqID][curTopkIndex];
    const auto& missSlots = mLoadSlot[reqID][curTopkIndex];
    if (missGlobalIds.empty()) { return false; }
    if (missGlobalIds.size() != missSlots.size() || mBlockSize <= 0) {
        KVPackAbort(mKVPackControl);
    }
    const auto hashIt = mAllBlocksHash.find(reqID);
    if (hashIt == mAllBlocksHash.end() ||
        mReqUseBlockTable.find(reqID) == mReqUseBlockTable.end()) {
        KVPackAbort(mKVPackControl);
    }
    const size_t finalLoadNum = previousLoadNum + missGlobalIds.size();
    if (finalLoadNum > static_cast<size_t>(mLoadIndexBufferLen) ||
        finalLoadNum > static_cast<size_t>(mKVBufferLen)) {
        KVPackAbort(mKVPackControl);
    }
    return true;
}

void GSAPrefetchEngineC::BuildUCMMergeBlockContexts(
    int layerID, const std::string& reqID, uint32_t requestIndex,
    std::vector<UCMMergeBlockContext>& blockContexts,
    std::map<int, std::vector<size_t>>& blockIdxToTokenIdx)
{
    const auto& hashVec = mAllBlocksHash.at(reqID);
    const auto& missGlobalIds = mMissGlobalToken[reqID][mTopkLayerMap[layerID]];

    for (size_t i = 0; i < missGlobalIds.size(); ++i) {
        if (missGlobalIds[i] < 0) { KVPackAbort(mKVPackControl); }
        blockIdxToTokenIdx[missGlobalIds[i] / mBlockSize].push_back(i);
    }

    blockContexts.reserve(blockIdxToTokenIdx.size());

    size_t tokenBase = 0;
    for (const auto& [blockIdx, tokenIndices] : blockIdxToTokenIdx) {
        const bool isnoUCMblock = static_cast<size_t>(blockIdx) >= hashVec.size();
        const bool owned =
            ((blockContexts.size() + requestIndex) % mTpSize) == static_cast<size_t>(mRank);

        blockContexts.push_back(
            {&tokenIndices, tokenBase, isnoUCMblock, blockIdx, nullptr, nullptr, owned});

        tokenBase += tokenIndices.size();
    }
}
}  // namespace ucmprefetch
