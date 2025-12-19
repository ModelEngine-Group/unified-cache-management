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
#include <kvcache_log.h>
#include <sched.h>
#include <stdint.h>

namespace ucmprefetch {
ThreadPool::ThreadPool(size_t threadCount) : stop(false), maxThreads(threadCount)
{
    for (size_t i = 0; i < maxThreads; i++) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    this->condition.wait(lock,
                                         [this] { return this->stop || !this->tasks.empty(); });

                    if (this->stop && this->tasks.empty()) { return; }

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                    ++activeThreads;
                }

                task();
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    --activeThreads;
                    condition.notify_all();
                }
            }
        });
    }
}
ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) { worker.join(); }
}

template <class F, class... Args>
auto ThreadPool::Enqueue(F&& f,
                         Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);

        condition.wait(lock, [this] {
            if (!(activeThreads < maxThreads || tasks.size() < maxThreads * 2)) {
                std::cout << "Need wait: " << activeThreads << " " << tasks.size() << std::endl;
            }
            return (activeThreads < maxThreads || tasks.size() < maxThreads * 2);
        });
        // don't allow enqueueing after stopping the pool
        if (stop) { throw std::runtime_error("enqueue on stopped ThreadPool"); }

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

size_t ThreadPool::GetActiveThreads() const { return activeThreads; }

void MutliBSThreadFun(void* args)
{
    GSAPrefetchEngineC* engine = static_cast<GSAPrefetchEngineC*>(args);
    int ret = engine->CallPrefetchProcessFun();
    engine->mMutex.lock();
    engine->DelReqIDRun();
    engine->mMutex.unlock();
    if (ret == 0) { engine->SetPrefetchStatus(true); }
}

GSAPrefetchEngineC::GSAPrefetchEngineC(torch::Tensor& loadSuccessBlocks,
                                       torch::Tensor& successTableLen,
                                       std::vector<uint32_t>& kvShape, bool useMla, bool isLog,
                                       int tpSize, int rank, int extraTopkLen, bool isPythonLoad)
    : mLogger("./log/kvcache_pre_log.txt", LogLevel::INFO, isLog)
{
    mLoadSuccessBlocks = loadSuccessBlocks;
    mLayerNum = mLoadSuccessBlocks.sizes()[0];
    mMaxBs = mLoadSuccessBlocks.sizes()[1];
    mMaxTopkLen = mLoadSuccessBlocks.sizes()[2];
    //mFreeBlock = freeBlock;
    //mFreeBlockLen = freeBlockLen;
    mSuccessTableLen = successTableLen;
    mIsLog = isLog;
    mBsIndexList = (int*)malloc(sizeof(int) * mMaxBs);
    mTopkLenList = (int*)malloc(sizeof(int) * mMaxBs);
    mIsPrefetchDone = true;
    mThreadPool = ThreadPool::GetInst();
    mUseMla = useMla;
    mHeadSzie = kvShape[2];
    mHeadNum = kvShape[1];
    mBlockSize = kvShape[0];
    mTPSize = tpSize;
    mRank = rank;
    mIsPythonLoad = isPythonLoad;
    if (mRank != 0) {
        mLogger.SetLevel(LogLevel::WARNING);
        mIsLog = false;
    }
    //mExtraTopkLen = extraTopkLen;
    mLogger.log(LogLevel::INFO,
                "GSAPrefetchEngineC Init mLayerNum %d mMaxBs %u, mUseMla %d, mHeadSzie %u, mTPSize "
                "%u mBlockSize %u mHeadNum %u\n",
                mLayerNum, mMaxBs, mUseMla, mHeadSzie, mTPSize, mBlockSize, mHeadNum);
}

size_t GSAPrefetchEngineC::GetOffset(uint32_t layerID, bool isV)
{
    size_t kMinDataBlockSize =
        static_cast<size_t>(mBlockSize) * mHeadNum * mHeadSzie * mTensorElemSize;
    size_t vMinDataBlockSize = kMinDataBlockSize;
    size_t layerSize = (kMinDataBlockSize + vMinDataBlockSize) * mTPSize;
    if (mUseMla) {
        vMinDataBlockSize = 0;
        layerSize = kMinDataBlockSize;
    }
    size_t kOffset = 0;
    if (mUseMla) {
        kOffset = layerSize * layerID;
    } else {
        kOffset = layerSize * layerID + layerSize / mTPSize * mRank;
    }
    size_t vOffset = kOffset + kMinDataBlockSize;
    if (isV) {
        return vOffset;
    } else {
        return kOffset;
    }
}

size_t GSAPrefetchEngineC::GetOffsetNew(uint32_t layerID, bool isV)
{
    size_t kMinDataBlockSize =
        static_cast<size_t>(mBlockSize) * mHeadNum * mHeadSzie * mTensorElemSize;
    size_t layerSize = kMinDataBlockSize * 2;
    size_t kOffset = layerSize * layerID;
    if (mUseMla) {
        layerSize = kMinDataBlockSize;
        kOffset = layerSize * layerID;
        return kOffset;
    }
    size_t vOffset = kOffset + kMinDataBlockSize;

    if (isV) {
        return vOffset;
    } else {
        return kOffset;
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
    free(mBsIndexList);
    free(mTopkLenList);
}

void GSAPrefetchEngineC::SetBlocksMap(std::string reqID, std::vector<int>& blockTableList,
                                      std::vector<int>& remainIdx, std::vector<int>& preftchIndex,
                                      std::vector<std::string>& blocksHash, int maxIdx)
{
    if (mBlocksMap.find(reqID) != mBlocksMap.end()) {
        mBlocksMap[reqID].clear();
        mDocsTables[reqID].clear();
        mAllBlcoksHash[reqID].clear();
        mPrefetchIdx[reqID].clear();
    }
    mAllBlcoksHash[reqID] = blocksHash;
    for (int i = 0; i < mLayerNum; i++) {
        std::map<int, int> oneDocTable;
        std::map<int, int> oneBlockMap;
        std::map<int> onePrefetchIdx;
        for (auto idx : remainIdx) {
            oneDocTable[idx] = blockTableList[idx];
            oneBlockMap[blockTableList[idx]] = idx;
        }
        for (auto idx : preftchIndex) {
            oneDocTable[idx] = blockTableList[idx];
            oneBlockMap[blockTableList[idx]] = idx;
            onePrefetchIdx.push_back(idx);
        }
        mDocsTables[reqID].push_back(oneDocTable);
        mBlocksMap[reqID].push_back(oneBlockMap);
        mPrefetchIdx[reqID].push_back(onePrefetchIdx);
    }
    mPromptLen[reqID] = maxIdx;
    PrintMap(reqID, 0);
}

void GSAPrefetchEngineC::SetBlocksMapMultiLayer(std::string reqID,
                                                std::vector<std::map<int, int>>& remainMap,
                                                std::vector<std::map<int, int>>& prefetchMap,
                                                std::vector<std::string>& blocksHash, int maxIdx)
{
    if (mBlocksMap.find(reqID) != mBlocksMap.end()) {
        mBlocksMap[reqID].clear();
        mDocsTables[reqID].clear();
        mAllBlcoksHash[reqID].clear();
        mPrefetchIdx[reqID].clear();
    }
    mAllBlcoksHash[reqID] = blocksHash;
    for (int i = 0; i < mLayerNum; i++) {
        std::map<int, int> oneDocTable;
        std::map<int, int> oneBlockMap;
        std::vector<int> onePrefetchIdx;
        for (auto it = remainMap[i].begin(); it != remainMap[i].end(); it++) {
            oneDocTable[it->first] = it->second;
            oneBlockMap[it->second] = it->first;
        }
        for (auto it = prefetchMap[i].begin(); it != prefetchMap[i].end(); it++) {
            oneDocTable[it->first] = it->second;
            oneBlockMap[it->second] = it->first;
            onePrefetchIdx.push_back(it->first);
        }
        mDocsTables[reqID].push_back(oneDocTable);
        mBlocksMap[reqID].push_back(oneBlockMap);
        mPrefetchIdx[reqID].push_back(onePrefetchIdx);
    }
    mPromptLen[reqID] = maxIdx;
}

void GSAPrefetchEngineC::AddBlocksMap(std::string reqID, int idx, int blockID)
{
    if (mBlocksMap.find(reqID) == mBlocksMap.end()) {
        for (int i = 0; i < mLayerNum; ++i) {
            std::map<int, int> oneDocTable;
            std::map<int, int> oneBlockMap;
            oneDocTable[idx] = blockID;
            oneBlockMap[blockID] = idx;
            mDocsTables[reqID].push_back(oneDocTable);
            mBlocksMap[reqID].push_back(oneBlockMap);
        }
    } else {
        for (int i = 0; i < mLayerNum; i++) {
            mDocsTables[reqID][i][idx] = blockID;
            mBlocksMap[reqID][i][blockID] = idx;
        }
    }
}

void GSAPrefetchEngineC::DelBlocksMap(std::string reqID)
{
    mMutex.lock();
    mDelSeqIds.insert(reqID);
    if (mIsPrefetchDone) { DelReqIDRun(); }
    mMutex.unlock();
}

void GSAPrefetchEngineC::DelReqIDRun()
{
    for (auto it = mDelSeqIds.begin(); it != mDelSeqIds.end(); it++) {
        if (mBlocksMap.find(*it) == mBlocksMap.end()) {
            continue;
        } else {
            mBlocksMap.erase(*it);
            mDocsTables.erase(*it);
            mAllBlcoksHash.erase(*it);
            mPromptLen.erase(*it);
            mPrefetchIdx.erase(*it);
            std::cout << "Del reqID: " << *it << std::endl;
        }
        if (mPromptLen.find(*it) == mPromptLen.end()) {
            continue;
        } else {
            mPromptLen.erase(*it);
        }
    }
    mDelSeqIds.clear();
}

void GSAPrefetchEngineC::PrintMap(std::string reqID, int i)
{
    std::ostringstream oss;
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: " << reqID
        << " layerID: " << i << "mDocsTables";
    for (auto it : mDocsTables[reqID][i]) { oss << "(" << it.first << ", " << it.second << ")"; }
    oss << "------\n";
    mLogger.log(LogLevel::INFO, oss.str().c_str());
    oss.str("");
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: " << reqID
        << " layerID: " << i << "mBlocksMap";
    for (auto it : mBlocksMap[reqID][i]) { oss << "(" << it.first << ", " << it.second << ")"; }
    oss << "------\n";
    mLogger.log(LogLevel::INFO, oss.str().c_str());
    oss.str("");
}

void GSAPrefetchEngineC::GetHitAndMissBlock(PrefetchReqInfo oneBsInfo,
                                            std::unordered_set<int>& hitBlocks,
                                            std::map<int, int>& hitBlocksIdx,
                                            std::vector<int>& missIdxs)
{
    int topkLen = oneBsInfo.topkLen;
    int layerID = oneBsInfo.layerID;
    std::string reqID = oneBsInfo.reqID;
    int topkIndex = oneBsInfo.topkIndex;

    std::ostringstream oss;
    oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: " << reqID
        << " layerID: " << layerID << " topk len: " << topkLen << " topk: ";
    for (int j = 0; j < topkLen; j++) {
        int64_t item = 0;
        if (mUseTopkIdxs.scalar_type() == torch::kInt32) {
            item = mUseTopkIdxs[layerID][topkIndex][j].item<int32_t>();
        } else {
            item = mUseTopkIdxs[layerID][topkIndex][j].item<int64_t>();
        }
        oss << item << " ";
        if (mDocsTables[reqID][layerID].find(item) != mDocsTables[reqID][layerID].end()) {
            int blockID = mDocsTables[reqID][layerID][item];
            hitBlocks.insert(blockID);
            hitBlocksIdx.insert(std::make_pair(item, blockID));
            //if (hitBlocks.size() == (topkLen - mExtraTopkLen)) { break; }
        } else {
            missIdxs.push_back(item);
        }
    }
    oss << "------\n";
    mLogger.log(LogLevel::DEBUG, oss.str().c_str());
    oss.str("");
    //if ((hitBlocks.size() + missIdxs.size()) != (uint32_t)topkLen &&
    //    hitBlocks.size() != (topkLen - mExtraTopkLen)) {
    if ((hitBlocks.size() + missIdxs.size()) != (uint32_t)topkLen) {
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| Decode step: %u, Rank: %d, reqID: %s, layer: %d, hit size: "
                    "%lu, miss size: %lu , topkLen: %d, not equal error\n",
                    mDecodeStep, mRank, reqID, layerID, hitBlocks.size(), missIdxs.size(), topkLen);
        PrintMap(reqID, layerID);
    }
}

void GSAPrefetchEngineC::RunPrefetchH2D(PrefetchReqInfo oneBsInfo,
                                        std::unordered_set<int>& hitBlocks,
                                        std::map<int, int>& hitBlocksIdx,
                                        std::vector<int>& missIdxs)
{
    int layerID = oneBsInfo.layerID;
    std::string reqID = oneBsInfo.reqID;
    //uint32_t topkLen = oneBsInfo.topkLen;
    int oneFreeBlockLen = mPrefetchIdx[reqID][layerID].size();
    std::vector<int> oneFreeBlockTable;
    uint32_t index = 0;
    std::vector<int> onePrefetchIdx = mPrefetchIdx[reqID][layerID];
    int oneFreeBlockIndex = 0;
    //while (oneFreeBlockIndex < oneFreeBlockLen && index < missIdxs.size() &&
    //       hitBlocks.size() < (topkLen - mExtraTopkLen)) {
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
    LoadKVToHBM(oneFreeBlockTable, missIdxs, layerID, reqID);
}

void GSAPrefetchEngineC::RunOneBsPrefetch(std::string reqID, int topkLen, int bsIndex,
                                          int topkIndex)
{
#pragma omp parallel for num_threads(16) proc_bind(master)
    for (int i = 0; i < mLayerNum; i++) {
        mLoadSuccessBlocks[i][bsIndex].fill_(0);
        //int* freeBlockPtr = mFreeBlock[i][bsIndex].data_ptr<int>();
        std::unordered_set<int> hitBlocks;
        std::map<int, int> hitBlocksIdx;
        std::vector<int> missIdxs;
        PrefetchReqInfo oneBsInfo;
        oneBsInfo.topkLen = topkLen;
        oneBsInfo.reqID = reqID;
        oneBsInfo.topkIndex = topkIndex;
        oneBsInfo.bsIndex = bsIndex;
        oneBsInfo.layerID = i;
        GetHitAndMissBlock(oneBsInfo, hitBlocks, hitBlocksIdx, missIdxs);
        //if (missIdxs.size() != 0 && hitBlocksIdx.size() < (topkLen - mExtraTopkLen)) {
        if (missIdxs.size() != 0) {
            RunPrefetchH2D(oneBsInfo, hitBlocks, hitBlocksIdx, missIdxs);
        }
        int successIndex = 0;
        for (auto it = hitBlocksIdx.begin(); it != hitBlocksIdx.end(); it++) {
            mLoadSuccessBlocks[i][bsIndex][successIndex] = it->second;
            successIndex += 1;
        }
        mPrefetchIdx[reqID][i].clear();
        for (auto it = mDocsTables[reqID][i].begin(); it != mDocsTables[reqID][i].end(); it++) {
            if (it->first >= mPromptLen[reqID]) { break; }
            if (hitBlocksIdx.find(it->first) != hitBlocksIdx.end()) {
                continue;
            } else {
                //freeBlockPtr[oneFreeBlockIndex] = it->second;
                //oneFreeBlockIndex += 1;
                mPrefetchIdx[reqID][i].push_back(it->first);
            }
        }
        //mFreeBlockLen[i][bsIndex] = oneFreeBlockIndex;
        mSuccessTableLen[i][bsIndex] = (int)(hitBlocks.size());
    }
}

void GSAPrefetchEngineC::LoadKVToHBM(std::vector<int> loadNPUBlockIDs, std::vector<int> missIdxs,
                                     int layerID, std::string reqID)
{
    for (size_t i = 0; i < loadNPUBlockIDs.size(); i++) {
        if (!mIsPythonLoad) {
            if (mDelSeqIds.find(reqID) != mDelSeqIds.end()) {
                mLogger.log(LogLevel::INFO,
                            "Decode step: %u, Rank: %d, reqID: %s, layer: %d, stop prefetch\n",
                            mDecodeStep, mRank, reqID.c_str(), layerID);
                return;
            }
            while (mStopPrefetch) { std::this_thread::sleep_for(std::chrono::microseconds(2)); }
            UC::Task task{UC::Task::Type::LOAD, UC::Task::Location::DEVICE, "NFS::S2D"};
            std::string blockId = mAllBlcoksHash[reqID][missIdxs[i]];
            size_t kOffset = GetOffsetNew(layerID, false);
            size_t vOffset = GetOffsetNew(layerID, true);
            if (!mUseMla) {
                task.Append(blockId, kOffset,
                            reinterpret_cast<uintptr_t>(
                                mKvCaches[layerID][0][loadNPUBlockIDs[i]].data_ptr()),
                            mKVSzieBytes);
                task.Append(blockId, vOffset,
                            reinterpret_cast<uintptr_t>(
                                mKvCaches[layerID][1][loadNPUBlockIDs[i]].data_ptr()),
                            mKVSzieBytes);
            } else {
                task.Append(
                    blockId, kOffset,
                    reinterpret_cast<uintptr_t>(mKvCaches[layerID][loadNPUBlockIDs[i]].data_ptr()),
                    mKVSzieBytes);
            }
            size_t taskID = mStore->Submit(std::move(task));
            auto ret = mStore->Wait(taskID);
            if (ret != 0) {
                mLogger.log(LogLevel::ERROR,
                            "Decode step: %u, Rank: %d, reqID: %s, layer: %d, blockID: %lu, miss "
                            "idx: %u, load blockid: %u load k error\n",
                            mDecodeStep, mRank, reqID.c_str(), layerID, blockId, missIdxs[i],
                            loadNPUBlockIDs[i]);
                return;
            }
        }

        int oriIdx = mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]];
        mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]] = missIdxs[i];
        mDocsTables[reqID][layerID].erase(oriIdx);
        mDocsTables[reqID][layerID][missIdxs[i]] = loadNPUBlockIDs[i];
    }
}

void GSAPrefetchEngineC::RunAsyncPrefetchBs(std::vector<std::string>& reqIDsInput,
                                            std::vector<int>& topkLensInput,
                                            std::vector<int>& bsIndexInput,
                                            std::vector<torch::Tensor>& kvCaches, void* storePtr)
{
    if (mKVSzieBytes == 0) {
        mTensorElemSize = kvCaches[0].element_size();
        if (mUseMla) {
            mKVSzieBytes = kvCaches[0].element_size() * kvCaches[0][0].numel();
        } else {
            mKVSzieBytes = kvCaches[0].element_size() * kvCaches[0][0][0].numel();
        }
        if (storePtr == nullptr) {
            mLogger.log(LogLevel::ERROR,
                        "Decode step: %u, |KVCache Prefetch| storePtr is nullptr error\n",
                        mDecodeStep);
            std::abort();
        }
        mStore = static_cast<UC::CCStore<>*>(storePtr);
        mLogger.log(LogLevel::INFO,
                    "Decode step: %u, |KVCache Prefetch| start mKVSzieBytes: %u, mTensorElemSize "
                    "%u, store %p\n",
                    mDecodeStep, mKVSzieBytes, mTensorElemSize, mStore);
    }
    mKvCaches = kvCaches;
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| start async pretch batch size: %lu\n",
                mDecodeStep, reqIDsInput.size());
    runBsLen = reqIDsInput.size();
    if (runBsLen > mMaxBs) {
        mLogger.log(LogLevel::ERROR, "Decode step: %u, |KVCache Prefetch| runBsLen %u, maxBs: %d\n",
                    mDecodeStep, runBsLen, mMaxBs);
        std::abort();
    }
    mReqIdList.clear();
    mReqIdList.assign(reqIDsInput.begin(), reqIDsInput.end());
    memcpy(mTopkLenList, topkLensInput.data(), sizeof(int) * runBsLen);
    memcpy(mBsIndexList, bsIndexInput.data(), sizeof(int) * runBsLen);
    mMutex.lock();
    mIsPrefetchDone = false;
    mMutex.unlock();
    if (mIsPythonLoad) {
        MutliBSThreadFun(this);
    } else {
        mThreadPool->Enqueue(MutliBSThreadFun, this);
    }
}

void GSAPrefetchEngineC::SetBlockTableInfo(torch::Tensor& blockTables, torch::Tensor& blockLengths,
                                           torch::Tensor& inputTopkBuf, int step)
{
    mLoadSuccessBlocks = blockTables;
    mSuccessTableLen = blockLengths;
    mUseTopkIdxs = inputTopkBuf.clone();
    mDecodeStep = step;
}

int GSAPrefetchEngineC::CallPrefetchProcessFun()
{
    auto start = std::chrono::high_resolution_clock::now();
    allNeedLoadBlock.clear();
    allMissIdxs.clear();
    for (size_t i = 0; i < runBsLen; i++) {
        if (mDocsTables.find(mReqIdList[i]) == mDocsTables.end() || mTopkLenList[i] <= 0) {
            mLogger.log(LogLevel::ERROR,
                        "Decode step: %u, |KVCache Prefetch| topk len is zero: %d\n", mDecodeStep,
                        mTopkLenList[i]);
            continue;
        }
        allMissIdxs.insert({mReqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        allNeedLoadBlock.insert({mReqIdList[i], std::vector<std::vector<int>>(mLayerNum)});
        RunOneBsPrefetch(mReqIdList[i], mTopkLenList[i], mBsIndexList[i], i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Finish async pretch cost: %lu\n", mDecodeStep,
                duration.count());
    return 0;
}

bool GSAPrefetchEngineC::GetPrefetchStatus() { return mIsPrefetchDone; }

void GSAPrefetchEngineC::SetPrefetchStatus(bool flag)
{
    mMutex.lock();
    mIsPrefetchDone = flag;
    mMutex.unlock();
}

void GSAPrefetchEngineC::SetModelRunningStatus(bool flag) { mStopPrefetch = flag; }

std::map<std::string, std::vector<std::vector<int>>> GSAPrefetchEngineC::ObtainLoadBlocks()
{
    return allNeedLoadBlock;
}

std::map<std::string, std::vector<std::vector<int>>> GSAPrefetchEngineC::ObtainMissIdxs()
{
    return allMissIdxs;
}

std::map<std::string, std::vector<std::map<int, int>>> GSAPrefetchEngineC::ObtainBlocksMap()
{
    return mBlocksMap;
}

std::map<std::string, std::vector<std::map<int, int>>> GSAPrefetchEngineC::ObtainDocsMap()
{
    return mDocsTables;
}
} // namespace ucmprefetch
