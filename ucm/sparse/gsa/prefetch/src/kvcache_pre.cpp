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
#include <kvcache_log.h>
#include <sched.h>
#include <cstdint>
#include <unistd.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cerrno>
#include <numeric>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "securec.h"
#include "kvcache_pre.h"
#include "kvcache_pre_detail.h"

#define MAX_LOAD_NUM 2000u
#define H2D_INFO_SIZE 6
#define MAX_TOPK_SIZE 3200lu
// 0表示使用原来的HOST预取、组合，1表示优化后的UCM统一管理
#define IS_UCM_BUFFER_OPEN 1

namespace ucmprefetch {

static_assert(std::atomic<uint64_t>::is_always_lock_free, "KV pack requires lock-free uint64_t atomics");
static_assert(std::atomic<uint32_t>::is_always_lock_free, "KV pack requires lock-free uint32_t atomics");
static_assert(std::atomic<int32_t>::is_always_lock_free, "KV pack requires lock-free int32_t atomics");

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
        if (header->fatalError.load(std::memory_order_acquire) != 0) {
            return -1;
        }
        if (ready()) { return 0; }
        if (KVPackClock::now() >= deadline) { return KVPackFail(header); }
        std::this_thread::sleep_for(kKVPackPollInterval);
    }
}
} // namespace

// 创建或连接共享控制区和 KV staging。让 16 个 rank 映射同一份物理存储，各进程的虚拟地址可以不同。
int GSAPrefetchEngineC::InitKVPackControl(int dpRank)
{
    // 只在初始化时检查固定 rank/stream 配置，不在逐层路径重复检查。
    if (mRank < 0 || mRank >= static_cast<int>(mTpSize) ||
        mStreamNum <= 0) {
        return -1;
    }
    const size_t streamCount = static_cast<size_t>(mStreamNum);
    const size_t kBytes = mKBufferHost[0].numel() * mKBufferHost.element_size();
    const size_t vBytes = mISSPAC8 ? 0 :
        mVBufferHost[0].numel() * mVBufferHost.element_size();
    const size_t stateCount = streamCount * mTpSize;
    const size_t controlBytes = sizeof(SharedKVPackHeader) +
        stateCount * sizeof(KVPackRankState);
    const size_t bytes = controlBytes + streamCount * (kBytes + vBytes);

    mKVPackControlName = "/gsa_kv_pack_control_dp" + std::to_string(dpRank);
    mKVPackPreviousEpoch.assign(streamCount, 0);
    mSharedKBufferHostPtr.resize(streamCount);
    if (!mISSPAC8) { mSharedVBufferHostPtr.resize(streamCount); }

    int fd = -1;
    auto fail = [&](const char* operation) {
        mLogger.log(LogLevel::ERROR, "|KVPack| %s failed, errno=%d\n",
            operation, errno);
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
        if (KVPackClock::now() >= deadline) {
            return "wait initialized";
        }
        std::this_thread::sleep_for(kKVPackPollInterval);
    }
    return nullptr;
}

const char* GSAPrefetchEngineC::CreateKVPackControlFile(size_t bytes, int& fd)
{
    fd = shm_open(mKVPackControlName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) { return "shm_open creator"; }
    mIsKVPackControlCreator = true;
    if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        return "ftruncate";
    }
    return nullptr;
}

const char* GSAPrefetchEngineC::OpenKVPackControlFollower(
    const std::chrono::steady_clock::time_point& deadline, size_t bytes, int& fd)
{
    while ((fd = shm_open(mKVPackControlName.c_str(), O_RDWR, 0600)) < 0) {
        if ((errno != ENOENT && errno != EINTR) ||
            KVPackClock::now() >= deadline) {
            return "shm_open follower";
        }
        std::this_thread::sleep_for(kKVPackPollInterval);
    }
    // 文件名出现与 ftruncate 完成之间存在启动窗口。
    for (;;) {
        struct stat info {};
        if (fstat(fd, &info) != 0) { return "fstat"; }
        if (info.st_size == static_cast<off_t>(bytes)) { return nullptr; }
        if (info.st_size != 0 || KVPackClock::now() >= deadline) {
            return "shared memory size";
        }
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

void GSAPrefetchEngineC::SetupKVPackSharedBuffers(
    size_t controlBytes, size_t streamCount, size_t kBytes, size_t vBytes)
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
    auto* states = KVPackStates(mKVPackControl) +
        static_cast<size_t>(streamIdx) * mTpSize;
    // epoch 由 version/layer 生成；首次使用不等待。
    (void)epoch;
    return KVPackWait(mKVPackControl, [&] {
        if (previousEpoch == 0) { return true; }
        for (size_t rank = 0; rank < mTpSize; ++rank) {
            if (states[rank].h2dDoneEpoch.load(std::memory_order_acquire) !=
                previousEpoch) {
                return false;
            }
        }
        return true;
    });
}

// 确保共享 KV 已完整写好，随后才能 H2H。
int GSAPrefetchEngineC::FinishKVPack(int streamIdx, uint64_t epoch)
{
    auto* states = KVPackStates(mKVPackControl) +
        static_cast<size_t>(streamIdx) * mTpSize;
    states[mRank].packedEpoch.store(epoch, std::memory_order_release);
    return KVPackWait(mKVPackControl, [&] {
        for (size_t rank = 0; rank < mTpSize; ++rank) {
            if (states[rank].packedEpoch.load(std::memory_order_acquire) != epoch) {
                return false;
            }
        }
        return true;
    });
}

// 本 rank 的 H2D 同步成功后，发布完成标记。通知其他 rank：本卡已完成本轮消费。
void GSAPrefetchEngineC::MarkKVH2DDone(int streamIdx, uint64_t epoch)
{
    auto* states = KVPackStates(mKVPackControl) +
        static_cast<size_t>(streamIdx) * mTpSize;
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
    if (mIsKVPackControlCreator) {
        shm_unlink(mKVPackControlName.c_str());
    }
    mKVPackControlSize = 0;
    mIsKVPackControlCreator = false;
    mKVPackPreviousEpoch.clear();
    mSharedKBufferHostPtr.clear();
    mSharedVBufferHostPtr.clear();
}

float average(const std::vector<float>& vec)
{
    if (vec.empty()) {
        throw std::runtime_error("Cannot compute average of empty vector");
    }
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return static_cast<float>(sum / vec.size());
}

float average2d(const std::vector<std::vector<float>>& matrix)
{
    if (matrix.empty()) {
        return 0.0f;
    }

    double sum = std::accumulate(matrix.begin(), matrix.end(), 0.0,
        [](double acc, const std::vector<float>& row) {
            return acc + std::accumulate(row.begin(), row.end(), 0.0);
        });

    size_t count = 0;
    for (const auto& row : matrix) {
        count += row.size();
    }

    return static_cast<float>(sum / count);
}

void MultiBSThreadFun(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(threadInfo.args_);
    int ret = engine->CallPrefetchProcess(taskInfo, threadInfo);
    engine->mDelReqMutex.lock();
    engine->DelReqIDRun();
    engine->mDelReqMutex.unlock();
    engine->mIsPrefetchRunning.store(false, std::memory_order_release);
    if (ret == 0) { return; }
}

void MultiBSThreadFunIndexer(const TaskInfo& taskInfo, ThreadInfo& threadInfo)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(threadInfo.args_);
    int ret = engine->IndexerLoadProcess(taskInfo, threadInfo);
    engine->mIsIndexerLoadRunning.store(false, std::memory_order_release);
    if (ret == 0) { return; }
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

enum class SizeInfoIndex {
    BlockSize = 0,
    NumHead,
    KDim,
    VDim,
    ElementSize
};

template <typename T>
constexpr size_t ToIndex(T value)
{
    return static_cast<size_t>(value);
}

GSAPrefetchEngineC::GSAPrefetchEngineC(std::vector<int>& modelInfo,
    std::vector<int>& sizeinfo, bool isLog, int rank, int threadNum,
    std::vector<int> cpuToBind, std::vector<int> topkLayerMap,
    std::string logFileName)
    : mLogger(logFileName, LogLevel::INFO, isLog)
{
    InitModelInfo(modelInfo, isLog, cpuToBind,
        topkLayerMap, rank, threadNum);
    InitBufferInfo(sizeinfo);
    InitAclRuntime();
    InitPrefetchResultShm(modelInfo);
}

void GSAPrefetchEngineC::InitModelInfo(
    std::vector<int>& modelInfo, bool isLog,
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
    if (!mTopkLayerMap.empty()) {
        mTopkIndexerNum = mTopkLayerMap.back() + 1;
    }
    mIsLog = isLog;
    mRank = rank;
    mKVPackDpRank = modelInfo[ToIndex(ModelInfoIndex::DpRank)];
    mTpSize = modelInfo[ToIndex(ModelInfoIndex::TpSize)];
    mThreadNum = threadNum;
    mLogger.log(LogLevel::INFO,
        "GSAPrefetchEngineC new hbm Init mLayerNum: %d mMaxLen: %d "
        "mSpaTopkLen: %d mSparseLen: %d, prefetch type: %d, maxBs: %d\n",
        mLayerNum, mMaxLen, mSpaTopkLen, mSparseLen,
        modelInfo[ToIndex(ModelInfoIndex::PrefetchType)], mMaxBs);
    mTaskManger = std::make_unique<TaskManager>(
        mRank,
        modelInfo[ToIndex(ModelInfoIndex::DpRank)],
        modelInfo[ToIndex(ModelInfoIndex::TpSize)],
        this,
        1,
        mStreamNum,
        "kv",
        cpuToBind);
    if (mOpenIndexerOffload) {
        mIndexerTaskManger = std::make_unique<TaskManager>(
            mRank,
            modelInfo[ToIndex(ModelInfoIndex::DpRank)],
            modelInfo[ToIndex(ModelInfoIndex::TpSize)],
            this, 1, mStreamNum, "indexer",
            cpuToBind
        );
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
        "Decode step: %u, |KVCache Prefetch| init kvcache mKVSizeBytesBlock: %u, mKVSizeBytesBlockRe: %u,"
        "mKVSizeBytesToken %u, mKVSizeBytesTokenRe %u, mKDim: %d, mVDim: %d, mNumHead: %d mTensorElemSize: %d\n",
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

    auto ret = aclrtGetCurrentContext(static_cast<aclrtContext*>(mContext));
    if (ret != ACL_SUCCESS) {
        const char *errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
            "|KVCache Prefetch| aclrtGetCurrentContext Error: %d, Error: %s\n",
            ret, errMsg ? errMsg : "Unknown error");
    }

    ret = aclrtCreateStream(static_cast<aclrtStream*>(mStream));
    if (ret != ACL_SUCCESS) {
        const char *errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
            "|KVCache Prefetch| aclrtCreateStream Error: %d, Error: %s\n",
            ret, errMsg ? errMsg : "Unknown error");
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
        mLogger.log(LogLevel::ERROR, "Rank 0 shm_open failed: %s\n",
            strerror(errno));
    } else {
        if (ftruncate(shmFd, mPrefetchResultShmSize) == -1) {
            mLogger.log(LogLevel::ERROR, "Rank 0 ftruncate failed: %s\n",
                strerror(errno));
        }
        mPrefetchResultShm = mmap(NULL, mPrefetchResultShmSize,
            PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        if (mPrefetchResultShm == MAP_FAILED) {
            mLogger.log(LogLevel::ERROR, "Rank 0 mmap failed: %s\n", strerror(errno));
            mPrefetchResultShm = nullptr;
        }
        close(shmFd);
        mIsPrefetchResultShmCreator = true;
        mPrefetchResultVersion = 0;
        mLogger.log(LogLevel::INFO,
            "Rank 0 created shared memory: size=%zu\n",
            mPrefetchResultShmSize);
    }
}

void GSAPrefetchEngineC::AttachPrefetchResultShm()
{
    // Other ranks: Wait for rank 0 to create, then attach with read-only access
    int retryCount = 0;
    int shmFd = -1;
    while (shmFd == -1 && retryCount < 100) {
        shmFd = shm_open(mPrefetchResultShmName.c_str(), O_RDONLY, 0666);
        if (shmFd == -1) {
            sleep(1); // Wait 1ms before retry
            retryCount++;
        }
    }
    if (shmFd == -1) {
        mLogger.log(LogLevel::ERROR,
            "Rank %d failed to open shared memory after %d retries: %s\n",
            mRank, retryCount, strerror(errno));
    } else {
        mPrefetchResultShm = mmap(NULL, mPrefetchResultShmSize,
            PROT_READ, MAP_SHARED, shmFd, 0);
        if (mPrefetchResultShm == MAP_FAILED) {
            mLogger.log(LogLevel::ERROR, "Rank %d mmap failed: %s\n",
                mRank, strerror(errno));
            mPrefetchResultShm = nullptr;
        } else {
            mIsPrefetchResultShmCreator = false;
            mPrefetchResultVersion = 0;
            mLogger.log(LogLevel::INFO,
                "Rank %d attached to shared memory (read-only): size=%zu\n",
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

    size_t perReqPerLayerTotalSize =
        2 * countSize + (mMaxLoadNum * 2 + mSpaTopkLen) * sizeof(int);

    mPrefetchResultShmSize =
        versionSize + mMaxBs * mTopkIndexerNum * perReqPerLayerTotalSize;

    if (mRank == 0) {
        CreatePrefetchResultShm();
    } else {
        AttachPrefetchResultShm();
    }
    if (mPrefetchResultShm == nullptr) { KVPackAbort(nullptr); }
    if (mRank == 0) {
        __atomic_store_n(static_cast<uint32_t*>(mPrefetchResultShm),
            uint32_t(0), __ATOMIC_RELEASE);
    }
}

void GSAPrefetchEngineC::CheckInputIndex(uint32_t maxLen, uint32_t index)
{
    if (index >= maxLen) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| index error! index: %u, maxLen: %u\n",
                    mDecodeStep, index, maxLen);
        std::abort();
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
        if (mIsPrefetchResultShmCreator) {
            shm_unlink(mPrefetchResultShmName.c_str());
        }
        mPrefetchResultShm = nullptr;
    }
    for (int layerID = 0; layerID < mStreamNum; layerID++) {
        ReleasePrefetchStreamResources(layerID);
    }
    free(mContext);
    auto ret = aclrtDestroyStream((*static_cast<aclrtStream*>(mStream)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
            "|KVCache Prefetch| aclrtDestroyStream Error: %d, Error: %s\n",
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
            mLogger.log(LogLevel::ERROR,
                "|KVCache Prefetch| aclrtFree Error: %d, Error: %s\n",
                ret, errMsg ? errMsg : "Unknown error");
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
            mLogger.log(LogLevel::ERROR,
                "|KVCache Prefetch| aclrtFree Error: %d, Error: %s\n",
                ret, errMsg ? errMsg : "Unknown error");
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
    mAllBlocksHash[reqID] = blocksHash;
    mPromptLen[reqID] = maxIdx;
    auto& blockMap = mLastKVBlockAddress[reqID];
    blockMap.clear();
    for (size_t i = 0; i < blockIdxs.size(); ++i) {
        blockMap[blockIdxs[i]].kAddr = kAddrs[i];
        if (!mISSPAC8) {
            blockMap[blockIdxs[i]].vAddr = vAddrs[i];
        }
    }
}

void GSAPrefetchEngineC::SetLastIndexerBlock(
    const std::string& reqID,
    const std::vector<int>& blockIdxs,
    const std::vector<uintptr_t>& indexerAddrs)
{
    if (!mOpenIndexerOffload) {
        return;
    }
    if (blockIdxs.size() != indexerAddrs.size()) {
        mLogger.log(LogLevel::ERROR, "[ERROR] SetLastIndexerBlocks size mismatch, reqID=%s\n", reqID.c_str());
        return;
    }
    auto reqIt = mLastKVBlockAddress.find(reqID);
    if (reqIt == mLastKVBlockAddress.end()) {
        mLogger.log(LogLevel::ERROR, "[ERROR] SetLastIndexerBlock no block info, reqID=%s\n", reqID.c_str());
        return;
    }
    auto& blockMap = reqIt->second;
    for (size_t i = 0; i < blockIdxs.size(); ++i) {
        auto it = blockMap.find(blockIdxs[i]);
        if (it == blockMap.end()) {
            continue;
        }
        it->second.indexerAddr = indexerAddrs[i];
        it->second.indexerBlockIdx = blockIdxs[i];
    }
}

void GSAPrefetchEngineC::SetTokensMap(
    int layerID, const std::string& reqID,
    torch::Tensor& qTopkTensor, const std::vector<int>& remainBlockTable,
    int maxIdx, const std::vector<int>& slots,
    bool isNeedInit)
{
    std::lock_guard<std::mutex> lock(mMutexUseToken);
    mLogger.log(LogLevel::DEBUG, "Decode step: %u, |KVCache Prefetch|"
        " Start SetTokensMap layerID: %d, reqID: %s, isNeedInit: %d\n",
        mDecodeStep, layerID, reqID.c_str(), isNeedInit);
    const auto start = std::chrono::high_resolution_clock::now();
    int curTopkIndex = mTopkLayerMap[layerID];
    if (layerID == 0) {
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
    if (isNeedInit) {
        mAvgHitRate[curTopkIndex].clear();
        const size_t sparseSeqLen = remainBlockTable.size() * static_cast<size_t>(mBlockSize);
        ProcessTokensMapTopk(curTopkIndex, reqID, qTopkTensor, maxIdx, sparseSeqLen);
    }
    auto& manager = *mHBMMangers[reqID][curTopkIndex];
    const auto managerEnd = std::chrono::high_resolution_clock::now();
    const auto managerDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(managerEnd - start);
    LastChunkMove(layerID, reqID, manager.mSlotToGlobal);
    const auto totalEnd = std::chrono::high_resolution_clock::now();
    const auto totalDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - start);
    mLogger.log(LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| End SetTokensMap "
        "layerID=%d reqID=%s managerCost=%lld us totalCost=%lld us\n",
        mDecodeStep, layerID, reqID.c_str(),
        static_cast<long long>(managerDuration.count()),
        static_cast<long long>(totalDuration.count()));
}

void GSAPrefetchEngineC::ProcessTokensMapTopk(
    int layerID, const std::string& reqID,
    torch::Tensor& qTopkTensor, int maxIdx, size_t sparseSeqLen)
{
    const int qTopkNum = static_cast<int>(qTopkTensor.sizes()[0]);
    const int topkSize = static_cast<int>(qTopkTensor.sizes()[2]);
    auto& reqManagers = mHBMMangers[reqID];
    auto& missGlobalTokens = mMissGlobalToken[reqID][layerID];
    auto& loadSlots = mLoadSlot[reqID][layerID];
    reqManagers[layerID] = std::make_unique<HBMManager>();
    auto& manager = *reqManagers[layerID];
    manager.init(maxIdx, sparseSeqLen, mSpaTopkLen, topkSize, mLogger);
    std::vector<int> useBlockTable;
    useBlockTable.reserve(static_cast<size_t>(mSpaTopkLen));
    std::vector<int> oneTopkList(static_cast<size_t>(topkSize));
    for (int i = 0; i < qTopkNum; ++i) {
        const int* topkPtr = qTopkTensor[i][0].data_ptr<int>();
        std::memcpy(
            oneTopkList.data(), topkPtr, static_cast<size_t>(topkSize) * sizeof(int));
        missGlobalTokens.clear();
        loadSlots.clear();
        useBlockTable.clear();
        manager.processTopkWithDebug(
            oneTopkList, useBlockTable, missGlobalTokens, loadSlots, mLogger);
    }
    if (useBlockTable.size() != static_cast<size_t>(mSpaTopkLen)) {
        mLogger.log(LogLevel::ERROR,
            "[ERROR] mAllTopkIndicesTensor greater than mSpaTopkLen !!! "
            "layer_id %d, mSpaTopkLen: %d, useBlockTable size: %zu\n",
            layerID, mSpaTopkLen, useBlockTable.size());
    } else {
        std::memcpy(mAllTopkIndicesTensor[reqID][layerID].data_ptr(),
            useBlockTable.data(), useBlockTable.size() * sizeof(int));
    }
}

void GSAPrefetchEngineC::SetIndexerMap(
    std::string reqID, std::vector<int>& blockTableList,
    std::vector<std::vector<int>>& remainIdx, std::vector<std::vector<int>>& prefetchIdx,
    std::vector<int>& blockSlot, torch::Tensor& dsaSBT)
{
    if (!mOpenIndexerOffload) {
        return;
    }
    std::lock_guard<std::mutex> lock(mUpdateReqMutex);
    mLogger.log(LogLevel::INFO, "Decode step: %u, |KVCache Prefetch|"
        " Start SetIndexerMap\n", mDecodeStep);
    if (mBlocksMap.find(reqID) != mBlocksMap.end()) {
        mBlocksMap[reqID].clear();
        mDocsTables[reqID].clear();
        mIndexerSlot[reqID].clear();
        mPrefetchIdx[reqID].clear();
        mUseBlockTableIdx[reqID].clear();
        mPrefetchBlockTableIdx[reqID].clear();
        mRealTimeIdx[reqID].clear();
    }
    mIndexerSlot[reqID] = blockSlot;
    mIndexerLoadResultT[reqID] = dsaSBT;
    if (mIndexerSparseLen == 0) {
        mIndexerSparseLen = dsaSBT.sizes()[1];
    }
    mUseBlockTableIdx[reqID].resize(mLayerNum);
    mPrefetchBlockTableIdx[reqID].resize(mLayerNum);
    for (int i = 0; i < mLayerNum; i++) {
        SetIndexerMapForLayer(reqID, i, blockTableList, remainIdx, prefetchIdx, blockSlot);
    }
    mLogger.log(LogLevel::INFO, "Decode step: %u, "
        "|KVCache Prefetch| End SetIndexerMap\n", mDecodeStep);
}

void GSAPrefetchEngineC::SetIndexerMapForLayer(
    const std::string& reqID, int layerID, std::vector<int>& blockTableList,
    std::vector<std::vector<int>>& remainIdx,
    std::vector<std::vector<int>>& prefetchIdx, std::vector<int>& blockSlot)
{
    std::map<int, int> oneDocTable;
    std::map<int, int> oneBlockMap;
    std::vector<int> onePrefetchIdx;
    std::vector<std::pair<int, int>> oneDsaSparseBlock;
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
    mDocsTables[reqID].push_back(oneDocTable);
    mBlocksMap[reqID].push_back(oneBlockMap);
    mPrefetchIdx[reqID].push_back(onePrefetchIdx);
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
            layerID, mIndexerSparseLen, remainIdx[layerID].size());
    } else {
        std::memcpy(mIndexerLoadResultT[reqID][layerID].data_ptr(), sortBlockTable.data(),
            sortBlockTable.size() * sizeof(int));
    }
}

void GSAPrefetchEngineC::UpdateIndexerMap(std::string reqID, int idx, int blockID, std::string type)
{
    mLogger.log(LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch|Start new UpdateIndexerMap for type: %s req_id: %s, idx: %d, blockID: %d\n",
        mDecodeStep, type.c_str(), reqID.c_str(), idx, blockID);
    if (type == "add") {
        for (int layerID = 0; layerID < mLayerNum; layerID++) {
            mUseBlockTableIdx[reqID][layerID].emplace_back(idx);
        }
        mLogger.log(LogLevel::INFO,
            "Decode step: %u, |KVCache Prefetch| "
            "add mUseBlockTableIdx for reqID: %s. "
            "add slot block id: %d, add blockID: %d, "
            "ori mIndexerSlot size:%d, after add len: %d \n",
            mDecodeStep, reqID.c_str(), idx, blockID,
            mIndexerSlot[reqID].size(), mUseBlockTableIdx[reqID][0].size());
    } else if (type == "lastdel") {
        for (int layerID = 0; layerID < mLayerNum; layerID++) {
            mUseBlockTableIdx[reqID][layerID].pop_back();
        }
        mLogger.log(LogLevel::INFO,
            "Decode step: %u, |KVCache Prefetch| lastdel mUseBlockTableIdx for reqID: %s. "
            "lastdel blockID: %d, ori mIndexerSlot size:%d, after add len: %d \n",
            mDecodeStep, reqID.c_str(), blockID, mIndexerSlot[reqID].size(), mUseBlockTableIdx[reqID][0].size());
    } else {
        mLogger.log(LogLevel::INFO,
                    "Decode step: %u, |KVCache Prefetch| del mUseBlockTableIdx for reqID: %s. "
                    "del blockID: %d, after add len: %d \n",
                    mDecodeStep, reqID.c_str(), blockID, mUseBlockTableIdx[reqID][0].size());
    }
    mLogger.log(LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| end UpdateIndexerMap\n",
        mDecodeStep);
}

void GSAPrefetchEngineC::TokenIndexMapInter(
    std::vector<int>& oneTopkList, int bsIndex,
    int layerID, int topkLen, std::string reqID)
{
    if (!mOpenIndexerOffload) {
        return;
    }
    for (int j = 0; j < topkLen; j++) {
        int ori = oneTopkList[j];
        int slot = oneTopkList[j] % mBlockSize;
        uint32_t revIdx = oneTopkList[j] / mBlockSize;
        if (revIdx >= mRealTimeIdx[reqID][layerID].size()) {
            mLogger.log(LogLevel::ERROR,
                "[ERROR] revIdx >= mRealTimeIdx[reqID][layerID].size()!!! "
                "req_id: %s, layer_id %d, ori token %d, revIdx %d, mRealTimeIdx size %d\n",
                reqID.c_str(), layerID, ori, revIdx, mRealTimeIdx[reqID][layerID].size());
        }
        int idx = mRealTimeIdx[reqID][layerID][oneTopkList[j] / mBlockSize];
        oneTopkList[j] = idx * mBlockSize + slot;
        if (oneTopkList[j] < 0 || oneTopkList[j] >= mSeqLen[reqID]) {
            mLogger.log(LogLevel::ERROR,
                "[ERROR] Decode step: %u Trans topk is < 0 !!! "
                "layer_id %d, ori token %d, new token %d, block_index %d, "
                "idx: %d seq_len: %d for reqID %s\n",
                mDecodeStep, layerID, ori, oneTopkList[j],
                revIdx, idx, mSeqLen[reqID], reqID.c_str());
        }
    }
}

void GSAPrefetchEngineC::AddIndexerMap()
{
    if (!mOpenIndexerOffload) {
        return;
    }
    mLogger.log(LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| Start AddIndexerMap\n",
        mDecodeStep);
    for (auto it = mIndexerAdd.begin(); it != mIndexerAdd.end(); it++) {
        std::string reqID = it->first;
        if (mBlocksMap.find(reqID) == mBlocksMap.end()) {
            mLogger.log(LogLevel::ERROR,
                "Decode step: %u, |KVCache Prefetch| reqID is not init: %s\n",
                mDecodeStep, reqID.c_str());
            continue;
        }
        AddIndexerMapForRequest(reqID);
    }
    mIndexerAdd.clear();
    mLogger.log(LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| End AddIndexerMap\n",
        mDecodeStep);
}

void GSAPrefetchEngineC::AddIndexerMapForRequest(const std::string& reqID)
{
    for (int i = 0; i < mLayerNum; i++) {
        for (uint32_t j = 0; j < mIndexerAdd[reqID].size(); j++) {
            int idx = mIndexerSlot[reqID].size();
            int blockID = mIndexerAdd[reqID][j].second;
            mDocsTables[reqID][i][idx] = blockID;
            mBlocksMap[reqID][i][blockID] = idx;
        }
    }
    for (uint32_t j = 0; j < mIndexerAdd[reqID].size(); j++) {
        mIndexerSlot[reqID].emplace_back(mIndexerAdd[reqID][j].first);
    }
}

void GSAPrefetchEngineC::DeleteIndexerMap()
{
    if (!mOpenIndexerOffload) {
        return;
    }
    mLogger.log(LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| Start DeleteIndexerMap\n",
        mDecodeStep);
    for (auto it = mIndexerDel.begin(); it != mIndexerDel.end(); it++) {
        std::string reqID = it->first;
        if (mBlocksMap.find(reqID) == mBlocksMap.end()) {
            mLogger.log(LogLevel::ERROR,
                "Decode step: %u, |KVCache Prefetch| reqID is not init: %s\n",
                mDecodeStep, reqID.c_str());
            continue;
        }
        DeleteIndexerMapForRequest(reqID);
    }
    mIndexerDel.clear();
    mLogger.log(LogLevel::INFO,
        "Decode step: %u, |KVCache Prefetch| End DeleteIndexerMap\n",
        mDecodeStep);
}

void GSAPrefetchEngineC::DeleteIndexerMapForRequest(
    const std::string& reqID)
{
    for (int i = 0; i < mLayerNum; i++) {
        for (uint32_t j = 0; j < mIndexerDel[reqID].size(); j++) {
            int blockID = mIndexerDel[reqID][j];
            int idx = mBlocksMap[reqID][i][blockID];
            mBlocksMap[reqID][i].erase(blockID);
            mDocsTables[reqID][i].erase(idx);
            mPrefetchIdx[reqID][i].erase(
                std::remove(
                    mPrefetchIdx[reqID][i].begin(),
                    mPrefetchIdx[reqID][i].end(), idx),
                mPrefetchIdx[reqID][i].end());
        }
    }
}

void GSAPrefetchEngineC::LastChunkMove(
    int layerID, const std::string& reqID, const std::vector<int>& moveMap)
{
    if (moveMap.empty() || mBlockSize <= 0 ||
        moveMap.size() % static_cast<size_t>(mBlockSize) != 0) { return; }
    if (!mStoreV1_) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| mStoreV1_ is nullptr");
        return;
    }
    auto hashIt = mAllBlocksHash.find(reqID);
    auto blockTableIt = mReqUseBlockTable.find(reqID);
    if (hashIt == mAllBlocksHash.end() || blockTableIt == mReqUseBlockTable.end()) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| Cannot find hash/blockTable");
        return;
    }
    const auto& blockTable = blockTableIt->second;
    const size_t blockSize = static_cast<size_t>(mBlockSize);
    const size_t blockCount = moveMap.size() / blockSize;
    const size_t kBlockBytes = blockSize * static_cast<size_t>(mKVSizeBytesToken);
    const size_t vBlockBytes = blockSize * static_cast<size_t>(mKVSizeBytesTokenRe);
    std::vector<LastChunkSourceBlock> sourceBlocks(hashIt->second.size() + 1);
    std::vector<int> loadedBlockIndices;
    UC::Detail::TaskDesc task;
    if (!PrepareLastChunkSources(layerID, reqID, moveMap, blockTable,
        sourceBlocks, task, loadedBlockIndices)) { return; }
    UC::Detail::TaskHandle hostTaskId{};
    bool hasHostTask = false;
    if (!LoadLastChunkSources(layerID, reqID, task, loadedBlockIndices,
        sourceBlocks, hostTaskId, hasHostTask)) { return; }
    Synchronize(mContext, mStream, 1, mLogger);
    CopyLastChunkTokens(layerID, reqID, moveMap, sourceBlocks, hostTaskId, hasHostTask);
    for (size_t i = 0; i < blockCount; ++i) {
        const int hbmSlot = blockTable[i];
        void* kSrc = reinterpret_cast<void*>(mLastCHunkKPtr[layerID] + i * kBlockBytes);
        void* kDst = reinterpret_cast<void*>(mKcachePtr[layerID] + static_cast<size_t>(hbmSlot) * kBlockBytes);
        const int32_t kRet = CopyTrans(static_cast<void**>(&kSrc), static_cast<void**>(&kDst),
            kBlockBytes, 1, "h2d", mStream);
        if (kRet == -1) {
            mLogger.log(LogLevel::ERROR, "|LastChunkMove| CopyTrans H2D failed,");
        }
        if (!mISSPAC8) {
            void* vSrc = reinterpret_cast<void*>(mLastCHunkVPtr[layerID] + i * vBlockBytes);
            void* vDst = reinterpret_cast<void*>(mVcachePtr[layerID] + static_cast<size_t>(hbmSlot) * vBlockBytes);
            const int32_t vRet = CopyTrans(static_cast<void**>(&vSrc), static_cast<void**>(&vDst),
                vBlockBytes, 1, "h2d", mStream);
            if (vRet == -1) {
                mLogger.log(LogLevel::ERROR, "|LastChunkMove| CopyTrans H2D failed,");
            }
        }
    }
    Synchronize(mContext, mStream, 1, mLogger);
}

bool GSAPrefetchEngineC::PrepareLastChunkSources(
    int layerID, const std::string& reqID,
    const std::vector<int>& moveMap, const std::vector<int>& blockTable,
    std::vector<LastChunkSourceBlock>& sourceBlocks,
    UC::Detail::TaskDesc& task, std::vector<int>& loadedBlockIndices)
{
    const size_t blockSize = static_cast<size_t>(mBlockSize);
    const size_t blockCount = moveMap.size() / blockSize;
    if (blockCount > static_cast<size_t>(mLastChunkNum) || blockTable.size() < blockCount) {
        mLogger.log(LogLevel::ERROR,
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
    return BuildLastChunkSources(
        layerID, reqID, moveMap, sourceBlocks, task, loadedBlockIndices);
}

bool GSAPrefetchEngineC::BuildLastChunkSources(
    int layerID, const std::string& reqID, const std::vector<int>& moveMap,
    std::vector<LastChunkSourceBlock>& sourceBlocks,
    UC::Detail::TaskDesc& task, std::vector<int>& loadedBlockIndices)
{
    const auto& hashVec = mAllBlocksHash.at(reqID);
    const size_t blockSize = static_cast<size_t>(mBlockSize);
    const size_t kBlockBytes = blockSize * static_cast<size_t>(mKVSizeBytesToken);
    const size_t vBlockBytes = blockSize * static_cast<size_t>(mKVSizeBytesTokenRe);
    const int lastUCMBlockIdx = static_cast<int>(hashVec.size());
    std::vector<uint8_t> blockAdded(hashVec.size(), 0);
    const size_t reserveCount = std::min(moveMap.size(), hashVec.size());
    loadedBlockIndices.reserve(reserveCount);
    task.reserve(reserveCount);
    auto lastKVReqIt = mLastKVBlockAddress.find(reqID);
    if (lastKVReqIt == mLastKVBlockAddress.end()) {
        mLogger.log(LogLevel::ERROR, "|LastChunkMove| no non-UCM block\n");
        return false;
    }
    for (size_t i = 0; i < moveMap.size(); ++i) {
        const int globalToken = moveMap[i];
        if (globalToken < 0) {
            mLogger.log(LogLevel::ERROR,
                "|LastChunkMove| Invalid global token, reqID=%s token=%d\n",
                reqID.c_str(), globalToken);
            return false;
        }
        const int blockIdx = globalToken / mBlockSize;
        if (blockIdx >= lastUCMBlockIdx) {
            auto blockIt = lastKVReqIt->second.find(blockIdx);
            if (blockIt == lastKVReqIt->second.end()) {
                mLogger.log(LogLevel::ERROR, "|LastChunkMove| non-UCM block not found\n");
                return false;
            }
            if (sourceBlocks.size() <= static_cast<size_t>(blockIdx))
                sourceBlocks.resize(static_cast<size_t>(blockIdx) + 1);
            sourceBlocks[blockIdx].kBase = reinterpret_cast<const std::byte*>(blockIt->second.kAddr)
                + static_cast<size_t>(layerID) * kBlockBytes;
            if (!mISSPAC8) {
                sourceBlocks[blockIdx].vBase = reinterpret_cast<const std::byte*>(blockIt->second.vAddr)
                    + static_cast<size_t>(layerID) * vBlockBytes;
            }
            continue;
        }
        if (!blockAdded[blockIdx]) {
            blockAdded[blockIdx] = 1;
            loadedBlockIndices.push_back(blockIdx);
            task.emplace_back(UC::Detail::Shard{
                hashVec[blockIdx], static_cast<size_t>(layerID), {}});
        }
    }
    return true;
}

bool GSAPrefetchEngineC::LoadLastChunkSources(
    int layerID, const std::string& reqID, UC::Detail::TaskDesc& task,
    const std::vector<int>& loadedBlockIndices,
    std::vector<LastChunkSourceBlock>& sourceBlocks,
    UC::Detail::TaskHandle& hostTaskId, bool& hasHostTask)
{
    if (task.empty()) {
        return true;
    }
    const size_t kBlockBytes =
        static_cast<size_t>(mBlockSize) * static_cast<size_t>(mKVSizeBytesToken);
    const size_t normalBlockCount = task.size();
    auto hostResult = mStoreV1_->LoadToHostBuffer(std::move(task));
    if (!hostResult) {
        mLogger.log(LogLevel::ERROR,
            "|LastChunkMove| LoadToHostBuffer failed, layerID=%d reqID=%s error=%s\n",
            layerID, reqID.c_str(), hostResult.Error().ToString().c_str());
        return false;
    }
    auto result = std::move(hostResult.Value());
    hostTaskId = result.taskId;
    hasHostTask = true;
    if (result.views.size() != normalBlockCount) {
        mLogger.log(LogLevel::ERROR,
            "|LastChunkMove| HostBufferView size mismatch, reqID=%s views=%zu expected=%zu\n",
            reqID.c_str(), result.views.size(), normalBlockCount);
        mStoreV1_->ReleaseHostBuffer(hostTaskId);
        return false;
    }
    for (size_t i = 0; i < normalBlockCount; ++i) {
        if (!result.views[i].hostAddr) {
            mLogger.log(LogLevel::ERROR,
                "|LastChunkMove| HostBufferView is nullptr, reqID=%s index=%zu\n",
                reqID.c_str(), i);
            mStoreV1_->ReleaseHostBuffer(hostTaskId);
            return false;
        }
        const auto* base = static_cast<const std::byte*>(result.views[i].hostAddr);
        sourceBlocks[loadedBlockIndices[i]].kBase = base;
        if (!mISSPAC8) {
            sourceBlocks[loadedBlockIndices[i]].vBase = base + kBlockBytes;
        }
    }
    return true;
}

void GSAPrefetchEngineC::CopyLastChunkTokens(
    int layerID, const std::string& reqID,
    const std::vector<int>& moveMap,
    const std::vector<LastChunkSourceBlock>& sourceBlocks,
    UC::Detail::TaskHandle hostTaskId, bool hasHostTask)
{
    constexpr size_t kTwoThreadTokenThreshold = 256;
    constexpr size_t kFourThreadTokenThreshold = 512;
    constexpr int kMaxTokenThreadNum = 4;
    const size_t tokenCount = moveMap.size();
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDstBase = reinterpret_cast<std::byte*>(mLastCHunkKPtr[layerID]);
    std::byte* vDstBase = nullptr;
    if (!mISSPAC8) {
        vDstBase = reinterpret_cast<std::byte*>(mLastCHunkVPtr[layerID]);
    }
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
        const size_t tokenIdx = static_cast<size_t>(globalToken % mBlockSize);
        const auto& src = sourceBlocks[blockIdx];
        std::memcpy(kDstBase + i * kTokenBytes,
            src.kBase + tokenIdx * kTokenBytes, kTokenBytes);
        if (!mISSPAC8) {
            std::memcpy(vDstBase + i * vTokenBytes,
                src.vBase + tokenIdx * vTokenBytes, vTokenBytes);
        }
    };
    if (tokenThreadNum == 1) {
        for (size_t i = 0; i < tokenCount; ++i) copyOneToken(i);
    } else {
        const long long count = static_cast<long long>(tokenCount);
#pragma omp parallel for num_threads(tokenThreadNum) proc_bind(close) schedule(static)
        for (long long i = 0; i < count; ++i) copyOneToken(static_cast<size_t>(i));
    }
    if (hasHostTask) {
        auto status = mStoreV1_->ReleaseHostBuffer(hostTaskId);
        if (status.Failure()) {
            const std::string error = status.ToString();
            mLogger.log(LogLevel::ERROR,
                "|LastChunkMove| ReleaseHostBuffer failed, reqID=%s status=%s\n",
                reqID.c_str(), error.c_str());
        }
    }
}

UCMMergeHostAddress::~UCMMergeHostAddress()
{
    owner.ReleaseUCMMergeHostAddress(*this);
}

void GSAPrefetchEngineC::ReleaseUCMMergeHostAddress(
    UCMMergeHostAddress& address)
{
    if (!address.hasHostTask) {
        return;
    }
    const auto status = mStoreV1_->ReleaseHostBuffer(address.taskId);
    if (status.Failure()) {
        mLogger.log(
            LogLevel::ERROR,
            "|UCMTensorMerge| ReleaseHostBuffer failed, "
            "taskId=%zu status=%s\n",
            static_cast<std::size_t>(address.taskId),
            status.ToString().c_str());
    }
    // 与原实现相同：释放失败仅记录错误，不自动重试。
    address.hasHostTask = false;
    address.addresses.clear();
    address.views.clear();
}

bool GSAPrefetchEngineC::LoadUCMMergeHostAddress(UC::Detail::TaskDesc task,
    UCMMergeHostAddress& address, int layerID, const std::string& reqID)
{
    // 每次 merge 使用一个新的 address, 避免覆盖尚未释放的 task
    if (address.hasHostTask || !address.views.empty() || !address.addresses.empty()) {
        return false;
    }
    const size_t normalBlockCount = task.size();
    // 当前 rank 无 UCM 加载任务时，不依赖 Store。
    if (normalBlockCount == 0) { return true; }
    if (!mStoreV1_) { return false; }
    auto hostResult = mStoreV1_->LoadToHostBuffer(std::move(task));
    if (!hostResult) {
        mLogger.log(LogLevel::ERROR, "|UCMTensorMerge| LoadToHostBuffer failed, "
            "layerID=%d, reqID=%s, error=%s\n", layerID, reqID.c_str(),
            hostResult.Error().ToString().c_str());
        return false;
    }
    auto result = std::move(hostResult.Value());
    address.taskId = result.taskId;
    address.hasHostTask = true;
    address.views = std::move(result.views);
    if (address.views.size() != normalBlockCount) {
        mLogger.log(LogLevel::ERROR, "|UCMTensorMerge| host views count mismatch, expected=%zu actual=%zu \n",
            normalBlockCount, address.views.size());
        return false;
    }
    address.addresses.reserve(normalBlockCount);
    for (const auto& view : address.views) {
        if (view.hostAddr == nullptr) {
            mLogger.log(LogLevel::ERROR, "|UCMTensorMerge| null hostAddr, layerID=%d reqID=%s\n",
                layerID, reqID.c_str());
            return false;
        }
        address.addresses.push_back(static_cast<const std::byte*>(view.hostAddr));
    }
    return true; // 此处不能释放，后面的 memcpy 仍需访问这些地址。
}

bool GSAPrefetchEngineC::MemcpyUCMMergeHostAddress(
    const UCMMergeHostAddress& address,
    std::vector<UCMMergeBlockContext>& blockContexts, int layerID, int streamIdx,
    std::vector<uint64_t>& hbmSlotIndex, const std::string& reqID,
    size_t previousLoadNum)
{
    constexpr int kMaxTokenThreadNum = 4;
    constexpr size_t kTwoThreadTokenThreshold = 256;
    constexpr size_t kFourThreadTokenThreshold = 512;
    int curTopkIndex = mTopkLayerMap[layerID];
    const size_t tokenCount = mMissGlobalToken[reqID][curTopkIndex].size();
    const size_t finalLoadNum = previousLoadNum + tokenCount;
    if (hbmSlotIndex.capacity() < finalLoadNum) {
        hbmSlotIndex.reserve(static_cast<size_t>(mLoadIndexBufferLen));
    }
    hbmSlotIndex.resize(finalLoadNum);
    if (!PrepareUCMMergeSources(address, blockContexts, layerID, reqID)) {
        return false;
    }
    int tokenThreadNum = 1;
    if (tokenCount >= kFourThreadTokenThreshold) {
        tokenThreadNum = kMaxTokenThreadNum;
    } else if (tokenCount >= kTwoThreadTokenThreshold) {
        tokenThreadNum = 2;
    }
    if (tokenThreadNum == 1) {
        CopyUCMMergeTokensSerial(
            blockContexts, curTopkIndex, streamIdx, hbmSlotIndex,
            reqID, previousLoadNum);
    } else {
        CopyUCMMergeTokensParallel(
            blockContexts, curTopkIndex, streamIdx, hbmSlotIndex,
            reqID, previousLoadNum, tokenThreadNum);
    }
    return true;
}

bool GSAPrefetchEngineC::PrepareUCMMergeSources(
    const UCMMergeHostAddress& address,
    std::vector<UCMMergeBlockContext>& blockContexts,
    int layerID, const std::string& reqID)
{
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    const size_t kBlockBytes = static_cast<size_t>(mBlockSize) * kTokenBytes;
    const size_t vBlockBytes = static_cast<size_t>(mBlockSize) * vTokenBytes;
    const auto addressIt = mLastKVBlockAddress.find(reqID);
    for (auto& context : blockContexts) {
        if (!context.owned) { continue; }
        if (context.isnoUCMblock) {
            if (addressIt == mLastKVBlockAddress.end()) {
                return false;
            }
            auto blockAddressIt = addressIt->second.find(context.blockIdx);
            if (blockAddressIt == addressIt->second.end()) {
                return false;
            }
            const auto& blockAddress = blockAddressIt->second;
            if (blockAddress.kAddr == 0) {
                return false;
            }
            if (!mISSPAC8 && blockAddress.vAddr == 0) {
                return false;
            }
            context.kSrc = reinterpret_cast<const std::byte*>(blockAddress.kAddr)
                + static_cast<size_t>(layerID) * kBlockBytes;
            if (!mISSPAC8) {
                context.vSrc = reinterpret_cast<const std::byte*>(blockAddress.vAddr)
                    + static_cast<size_t>(layerID) * vBlockBytes;
            }
        } else {
            const auto* shardBase = address.addresses[context.viewIndex];
            context.kSrc = shardBase;
            if (!mISSPAC8) {
                context.vSrc = shardBase + kBlockBytes;
            }
        }
    }
    return true;
}

void GSAPrefetchEngineC::CopyUCMMergeTokensSerial(
    const std::vector<UCMMergeBlockContext>& blockContexts,
    int layerID, int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
    const std::string& reqID, size_t previousLoadNum)
{
    const auto& missGlobalIds = mMissGlobalToken[reqID][layerID];
    const auto& missSlots = mLoadSlot[reqID][layerID];
    const auto& blockTable = mReqUseBlockTable.at(reqID);
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDstBase = reinterpret_cast<std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    std::byte* vDstBase = nullptr;
    if (!mISSPAC8) {
        vDstBase = reinterpret_cast<std::byte*>(mSharedVBufferHostPtr[streamIdx]);
    }
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
            if (context.owned) {
                CopySingleTokenKV(kDstBase, vDstBase, context, tokenOffset,
                    dstOffset, kTokenBytes, vTokenBytes);
            }
            hbmSlotIndex[dstOffset] = static_cast<uint64_t>(hbmIdx)
                * static_cast<uint64_t>(mBlockSize) + static_cast<uint64_t>(hbmSlot);
        }
    }
}

void GSAPrefetchEngineC::CopySingleTokenKV(std::byte* kDstBase, std::byte* vDstBase,
    const UCMMergeBlockContext& context, size_t tokenOffset,
    size_t dstOffset, size_t kTokenBytes, size_t vTokenBytes)
{
    std::memcpy(kDstBase + dstOffset * kTokenBytes,
        context.kSrc + tokenOffset * kTokenBytes, kTokenBytes);
    if (!mISSPAC8) {
        std::memcpy(vDstBase + dstOffset * vTokenBytes,
            context.vSrc + tokenOffset * vTokenBytes, vTokenBytes);
    }
}

void GSAPrefetchEngineC::CopyUCMMergeTokensParallel(
    const std::vector<UCMMergeBlockContext>& blockContexts,
    int layerID, int streamIdx, std::vector<uint64_t>& hbmSlotIndex,
    const std::string& reqID, size_t previousLoadNum, int tokenThreadNum)
{
    const auto& missGlobalIds = mMissGlobalToken[reqID][layerID];
    const auto& missSlots = mLoadSlot[reqID][layerID];
    const auto& blockTable = mReqUseBlockTable.at(reqID);
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDstBase = reinterpret_cast<std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    std::byte* vDstBase = nullptr;
    if (!mISSPAC8) {
        vDstBase = reinterpret_cast<std::byte*>(mSharedVBufferHostPtr[streamIdx]);
    }
    std::vector<TokenCopyTask> tokenCopyTasks;
    BuildTokenCopyTasks(blockContexts, tokenCopyTasks,
        missGlobalIds, missSlots, blockTable,
        kTokenBytes, vTokenBytes, previousLoadNum);
    const long long taskCount = static_cast<long long>(tokenCopyTasks.size());
#pragma omp parallel for num_threads(tokenThreadNum) proc_bind(close) schedule(static)
    for (long long i = 0; i < taskCount; ++i) {
        const auto& copyTask = tokenCopyTasks[static_cast<size_t>(i)];
        if (copyTask.owned) {
            std::memcpy(kDstBase + copyTask.dstOffset * kTokenBytes,
                copyTask.kSrc, kTokenBytes);
            if (!mISSPAC8) {
                std::memcpy(vDstBase + copyTask.dstOffset * vTokenBytes,
                    copyTask.vSrc, vTokenBytes);
            }
        }
        hbmSlotIndex[copyTask.dstOffset] = copyTask.hbmSlot;
    }
}

void GSAPrefetchEngineC::BuildTokenCopyTasks(
    const std::vector<UCMMergeBlockContext>& blockContexts,
    std::vector<TokenCopyTask>& tokenCopyTasks,
    const std::vector<int>& missGlobalIds,
    const std::vector<int>& missSlots,
    const std::vector<int>& blockTable,
    size_t kTokenBytes, size_t vTokenBytes, size_t previousLoadNum)
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
            if (!mISSPAC8) {
                tokenCopyTasks.push_back({
                    context.owned ? context.kSrc + tokenOffset * kTokenBytes : nullptr,
                    context.owned ? context.vSrc + tokenOffset * vTokenBytes : nullptr, dstOffset,
                    static_cast<uint64_t>(hbmIdx) * static_cast<uint64_t>(mBlockSize)
                        + static_cast<uint64_t>(hbmSlot), context.owned});
            } else {
                tokenCopyTasks.push_back({
                    context.owned ? context.kSrc + tokenOffset * kTokenBytes : nullptr,
                    nullptr, dstOffset,
                    static_cast<uint64_t>(hbmIdx) * static_cast<uint64_t>(mBlockSize)
                        + static_cast<uint64_t>(hbmSlot), context.owned});
            }
        }
    }
}

size_t GSAPrefetchEngineC::FillFailedUCMMerge(
    const std::vector<UCMMergeBlockContext>& blockContexts,
    int layerID, int streamIdx,
    std::vector<uint64_t>& hbmSlotIndex,
    const std::string& reqID, size_t previousLoadNum)
{
    const int topkIndex = mTopkLayerMap[layerID];
    const auto& missSlots = mLoadSlot.at(reqID)[topkIndex];
    const auto& blockTable = mReqUseBlockTable.at(reqID);
    const size_t finalLoadNum = previousLoadNum + missSlots.size();
    const size_t kTokenBytes = static_cast<size_t>(mKVSizeBytesToken);
    const size_t vTokenBytes = static_cast<size_t>(mKVSizeBytesTokenRe);
    auto* kDst = reinterpret_cast<std::byte*>(mSharedKBufferHostPtr[streamIdx]);
    auto* vDst = mISSPAC8 ? nullptr :
        reinterpret_cast<std::byte*>(mSharedVBufferHostPtr[streamIdx]);

    // Full request layout is preserved on every rank, including non-owned tokens.
    if (hbmSlotIndex.capacity() < finalLoadNum) {
        hbmSlotIndex.reserve(static_cast<size_t>(mLoadIndexBufferLen));
    }
    hbmSlotIndex.resize(finalLoadNum);

    size_t zeroTokens = 0;
    for (const auto& context : blockContexts) {
        const auto& tokenIndices = *context.tokenIndices;
        const size_t dstBase = previousLoadNum + context.tokenBase;
        if (context.owned && !tokenIndices.empty()) {
            std::memset(kDst + dstBase * kTokenBytes,
                0, tokenIndices.size() * kTokenBytes);
            if (!mISSPAC8) {
                std::memset(vDst + dstBase * vTokenBytes,
                    0, tokenIndices.size() * vTokenBytes);
            }
            zeroTokens += tokenIndices.size();
        }
        // Every rank must build indices for ALL tokens, regardless of ownership.
        for (size_t localIdx = 0; localIdx < tokenIndices.size(); ++localIdx) {
            const size_t idx = tokenIndices[localIdx];
            const int slotID = missSlots[idx];
            if (slotID < 0 ||
                static_cast<size_t>(slotID / mBlockSize) >= blockTable.size() ||
                blockTable[static_cast<size_t>(slotID / mBlockSize)] < 0) {
                mLogger.log(LogLevel::ERROR,
                    "|KVPack| invalid HBM slot during zero fallback: "
                    "rank=%d layer=%d reqID=%s slot=%d tableSize=%zu\n",
                    mRank, layerID, reqID.c_str(), slotID, blockTable.size());
                // Invalid destination metadata is separate from load failure.
                // Preserve the abort version's handling of structural errors.
                KVPackAbort(mKVPackControl);
            }
            const int hbmIdx = blockTable[static_cast<size_t>(slotID / mBlockSize)];
            hbmSlotIndex[dstBase + localIdx] =
                static_cast<uint64_t>(hbmIdx) * static_cast<uint64_t>(mBlockSize)
                + static_cast<uint64_t>(slotID % mBlockSize);
        }
    }
    return zeroTokens;
}

void GSAPrefetchEngineC::UCMTensorMerge(
    int layerID, int streamIdx,
    std::vector<uint64_t>& hbmSlotIndex,
    const std::string& reqID, uint32_t requestIndex)
{
    if (!ValidateUCMMergeRequest(layerID, reqID)) {
        return;
    }

    // context.tokenIndices 指向此 map 中的 vector。
    // map 必须存活到正常拷贝和失败兜底全部结束。
    std::map<int, std::vector<size_t>> blockIdxToTokenIdx;
    std::vector<UCMMergeBlockContext> blockContexts;
    UC::Detail::TaskDesc task;

    BuildUCMMergeBlockContexts(
        layerID, reqID, requestIndex,
        blockContexts, task, blockIdxToTokenIdx);

    const auto& missGlobalIds =
        mMissGlobalToken[reqID][mTopkLayerMap[layerID]];
    const size_t previousLoadNum = mLoadNum[layerID];
    const size_t finalLoadNum =
        previousLoadNum + missGlobalIds.size();

    UCMMergeHostAddress address(*this);
    const char* failedStage = nullptr;

    if (!LoadUCMMergeHostAddress(
        std::move(task), address, layerID, reqID)) {
        failedStage = "load";
    } else if (!MemcpyUCMMergeHostAddress(
        address, blockContexts, layerID, streamIdx,
        hbmSlotIndex, reqID, previousLoadNum)) {
        failedStage = "prepare/copy";
    }

    if (failedStage != nullptr) {
        const size_t zeroTokens = FillFailedUCMMerge(
            blockContexts, layerID, streamIdx,
            hbmSlotIndex, reqID, previousLoadNum);

        mLogger.log(
            LogLevel::ERROR,
            "|KVPack| zero fallback: rank=%d layer=%d reqID=%s "
            "requestIndex=%u stage=%s zeroOwnedTokens=%zu totalTokens=%zu\n",
            mRank, layerID, reqID.c_str(), requestIndex,
            failedStage, zeroTokens, missGlobalIds.size());
    }

    mLoadNum[layerID] = static_cast<uint32_t>(finalLoadNum);
}

bool GSAPrefetchEngineC::ValidateUCMMergeRequest(int layerID, const std::string& reqID)
{
    if (!mStoreV1_) {
        mLogger.log(
            LogLevel::ERROR,
            "|UCMTensorMerge| mStoreV1_ is nullptr reqID=%s\n",
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
    const size_t finalLoadNum = mLoadNum[layerID] + missGlobalIds.size();
    if (finalLoadNum > static_cast<size_t>(mLoadIndexBufferLen) ||
        finalLoadNum > static_cast<size_t>(mKVBufferLen)) {
        KVPackAbort(mKVPackControl);
    }
    return true;
}

void GSAPrefetchEngineC::BuildUCMMergeBlockContexts(
    int layerID, const std::string& reqID,
    uint32_t requestIndex,
    std::vector<UCMMergeBlockContext>& blockContexts,
    UC::Detail::TaskDesc& task,
    std::map<int, std::vector<size_t>>& blockIdxToTokenIdx)
{
    const auto& hashVec = mAllBlocksHash.at(reqID);
    const auto& missGlobalIds =
        mMissGlobalToken[reqID][mTopkLayerMap[layerID]];

    for (size_t i = 0; i < missGlobalIds.size(); ++i) {
        if (missGlobalIds[i] < 0) {
            KVPackAbort(mKVPackControl);
        }
        blockIdxToTokenIdx[missGlobalIds[i] / mBlockSize].push_back(i);
    }

    blockContexts.reserve(blockIdxToTokenIdx.size());
    task.reserve(blockIdxToTokenIdx.size());

    size_t tokenBase = 0;
    for (const auto& [blockIdx, tokenIndices] : blockIdxToTokenIdx) {
        const bool isnoUCMblock =
            static_cast<size_t>(blockIdx) >= hashVec.size();
        const bool owned =
            ((blockContexts.size() + requestIndex) % mTpSize)
            == static_cast<size_t>(mRank);

        blockContexts.push_back({
            &tokenIndices, task.size(), tokenBase,
            isnoUCMblock, blockIdx, nullptr, nullptr, owned
        });

        if (owned && !isnoUCMblock) {
            task.push_back(UC::Detail::Shard{
                hashVec[blockIdx], static_cast<size_t>(layerID), {}
            });
        }

        tokenBase += tokenIndices.size();
    }
}
}  // namespace ucmprefetch
