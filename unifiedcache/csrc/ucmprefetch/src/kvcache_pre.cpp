#include "kvcache_pre.h"
#include <kvcache_log.h>
#include <stdint.h>
#include <sched.h>

namespace ucmprefetch
{

    GSAPrefetchEngineC::GSAPrefetchEngineC(torch::Tensor &freeBlock,
        torch::Tensor &loadSuccessBlocks,
        torch::Tensor &freeBlockLen,
        torch::Tensor &successTableLen,
        bool isLog)
        :mLogger("./log/kvcache_pre_log.txt", LogLevel::INFO, isLog)
    {
        mLoadSuccessBlocks = loadSuccessBlocks;
        mLoadSuccessBlocksCPU = mLoadSuccessBlocks.cpu();
        mLayerNum = mLoadSuccessBlocks.sizes()[0];
        mMaxBs = mLoadSuccessBlocks.sizes()[1];
        mMaxTopkLen = mLoadSuccessBlocks.sizes()[2];
        mFreeBlock = freeBlock;
        mFreeBlockLen = freeBlockLen;
        mSuccessTableLen = successTableLen;
        mIsLog = isLog;
        mReqIdList = (int *)malloc(sizeof(int) * mMaxBs);
        mBsIndexList = (int *)malloc(sizeof(int) * mMaxBs);
        mTopkLenList = (int *)malloc(sizeof(int) * mMaxBs);
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
        free(mReqIdList);
        free(mBsIndexList);
        free(mTopkLenList);
    }

    void GSAPrefetchEngineC::SetBlocksMap(int reqID, std::vector<int> &blockTableList,
        std::vector<int> &selectIndex)
    {
        if (mBlocksMap.find(reqID) != mBlocksMap.end()) {
            mBlocksMap[reqID].clear();
            mDocsTables[reqID].clear();
        }
        for (int i = 0; i < mLayerNum; i++) {
            std::map<int, int> oneDocTable;
            std::map<int, int> oneBlockMap;
            for (auto idx:selectIndex) {
                oneDocTable[idx] = blockTableList[idx];
                oneBlockMap[blockTableList[idx]] = idx;
            }
            mDocsTables[reqID].push_back(oneDocTable);
            mBlocksMap[reqID].push_back(oneBlockMap);
        }
    }

    void GSAPrefetchEngineC::AddBlocksMap(int reqID, int idx, int blockID)
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

    void GSAPrefetchEngineC::DelBlocksMap(int reqID)
    {
        if (mBlocksMap.find(reqID) == mBlocksMap.end()) {
            return;
        } else {
            mBlocksMap.erase(reqID);
            mDocsTables.erase(reqID);
            std::cout << "Del reqID: " << reqID << std::endl;
        }
    }

    void GSAPrefetchEngineC::PrintMap(int reqID, int i)
    {
        std::ostringstream oss;
        oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: "
            << reqID << " layerID: " << i << "mDocsTables";
        for (auto it : mDocsTables[reqID][i]) {
            oss << "(" << it.first << ", " << it.second << ")";
        }
        oss << "------\n";
        mLogger.log(LogLevel::INFO, oss.str().c_str());
        oss.str("");
        oss << "Decode step: " << mDecodeStep << " Rnak: " << mRank << " reqID: "
            << reqID << " layerID: " << i << "mBlocksMap";
        for (auto it : mBlocksMap[reqID][i]) {
            oss << "(" << it.first << ", " << it.second << ")";
        }
        oss << "------\n";
        mLogger.log(LogLevel::INFO, oss.str().c_str());
        oss.str("");
    }

    void GSAPrefetchEngineC::GetHitAndMissBlock(PrefetchReqInfo oneBsInfo,
            std::unordered_set<int> &hitBlocks,
            std::map<int, int> &hitBlocksIdx,
            std::vector<int> &missIdxs)
    {
        int topkLen = oneBsInfo.topkLen;
        int layerID = oneBsInfo.layerID;
        int reqID = oneBsInfo.reqID;
        int topkIndex = oneBsInfo.topkIndex;

        for (int j = 0; j < topkLen; j++) {
            int64_t item = 0;
            if (mUseTopkIdxs.scalar_type() == torch::kInt32) {
                item = mUseTopkIdxs[layerID][topkIndex][j].item<int32_t>();
            } else {
                item = mUseTopkIdxs[layerID][topkIndex][j].item<int64_t>();
            }
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
                "|KVCache Prefetch| Decode step: %u, Rank: %d, reqID: %d, layer: %d, not equel error\n",
                mDecodeStep, mRank, reqID, layerID);
            PrintMap(reqID, layerID);
        }

    }
    
    void GSAPrefetchEngineC::RunPrefetchH2D(PrefetchReqInfo oneBsInfo,
            std::unordered_set<int> &hitBlocks,
            std::map<int, int> &hitBlocksIdx,
            std::vector<int> &missIdxs)
    {
        int layerID = oneBsInfo.layerID;
        int reqID = oneBsInfo.reqID;
        int topkIndex = oneBsInfo.topkIndex;
        int bsIndex = oneBsInfo.bsIndex;

        int oneFreeBlockLen = mFreeBlockLen[layerID][bsIndex].item<int>();
        int *freeBlockPtr = mFreeBlock[layerID][bsIndex].data_ptr<int>();
        std::vector<int> oneFreeBlockTable;

        uint32_t index = 0;
        int oneFreeBlockIndex = 0;
        while(oneFreeBlockIndex < oneFreeBlockLen && index < missIdxs.size()) {
            int oneFreeBlockID = freeBlockPtr[oneFreeBlockIndex];
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
        allNeedLoadBlock[topkIndex][layerID] = oneFreeBlockTable;
        allMissIdxs[topkIndex][layerID] = missIdxs;
        LoadKVToHBM(oneFreeBlockTable, missIdxs, layerID, reqID);
    }

    void GSAPrefetchEngineC::RunOneBsPrefetch(int reqID,
        int topkLen, int bsIndex, int topkIndex)
    {
#pragma omp parallel for num_threads(16) proc_bind(master)
        for (int i = 0; i < mLayerNum; i++) {
            mLoadSuccessBlocksCPU[i][bsIndex].fill_(-1);
            int *freeBlockPtr = mFreeBlock[i][bsIndex].data_ptr<int>();
            std::unordered_set<int> hitBlocks;
            std::map<int, int> hitBlocksIdx;
            std::vector<int> missIdxs;
            PrefetchReqInfo oneBsInfo;
            oneBsInfo.topkLen = topkLen;
            oneBsInfo.reqID = reqID;
            oneBsInfo.topkIndex = topkIndex;
            oneBsInfo.bsIndex = bsIndex;
            oneBsInfo.layerID = i;
            GetHitAndMissBlock(oneBsInfo, hitBlocks,hitBlocksIdx, missIdxs);
            if (missIdxs.size() != 0) {
                RunPrefetchH2D(oneBsInfo, hitBlocks,hitBlocksIdx, missIdxs);
            }
            int successIndex = 0;
            for (auto it = hitBlocksIdx.begin(); it != hitBlocksIdx.end(); it++) {
                mLoadSuccessBlocksCPU[i][bsIndex][successIndex] = it->second;
                successIndex += 1;
            }
            int oneFreeBlockIndex = 0;
            for (auto it = mBlocksMap[reqID][i].begin(); it != mBlocksMap[reqID][i].end(); it++) {
                if (hitBlocks.find(it->first) != hitBlocks.end()) {
                    continue;
                } else {
                    freeBlockPtr[oneFreeBlockIndex] = it->first;
                    oneFreeBlockIndex += 1;
                }
            }
            mFreeBlockLen[i][bsIndex] = oneFreeBlockIndex;
            mSuccessTableLen[i][bsIndex] = (int)(hitBlocks.size());
            mLoadSuccessBlocks[i][bsIndex].copy_(mLoadSuccessBlocksCPU[i][bsIndex]);
        }
    }

    void GSAPrefetchEngineC::LoadKVToHBM(std::vector<int> loadNPUBlockIDs,
        std::vector<int> missIdxs, int layerID, int reqID)
    {
        for (size_t i = 0; i < loadNPUBlockIDs.size(); i++) {
            int oriIdx = mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]];
            mBlocksMap[reqID][layerID][loadNPUBlockIDs[i]] = missIdxs[i];
            mDocsTables[reqID][layerID].erase(oriIdx);
            mDocsTables[reqID][layerID][missIdxs[i]] = loadNPUBlockIDs[i];
        }
    }
    
    void GSAPrefetchEngineC::RunAsyncPrefetchBs(std::vector<int> &reqIDsInput,
        std::vector<int> &topkLensInput,
        std::vector<int> &bsIndexInput, int rank)
    {
        if (mRank == -1) {
            mRank = rank;
        }
        if(mRank != 0) {
            mLogger.SetLevel(LogLevel::WARNING);
            mIsLog = false;
        }
        mLogger.log(LogLevel::INFO,
            "Decode step: %u, |KVCache Prefetch| start async pretch batch size: %lu\n",
            mDecodeStep, reqIDsInput.size());
        runBsLen = reqIDsInput.size();
        if (runBsLen > mMaxBs) {
            mLogger.log(LogLevel::ERROR,
                "Decode step: %u, |KVCache Prefetch| runBsLen %u, maxBs: %d\n",
                mDecodeStep, runBsLen, mMaxBs);
            std::abort();
        }
        memcpy(mReqIdList, reqIDsInput.data(), sizeof(int) * runBsLen);
        memcpy(mTopkLenList, topkLensInput.data(), sizeof(int) * runBsLen);
        memcpy(mBsIndexList, bsIndexInput.data(), sizeof(int) * runBsLen);
        CallPrefetchProcessFun();
    }

    void GSAPrefetchEngineC::SetBlockTableInfo(torch::Tensor &blockTables, torch::Tensor &blockLengths,
        torch::Tensor &inputTopkBuf, int step)
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
        allNeedLoadBlock.resize(runBsLen, std::vector<std::vector<int>>(mLayerNum));
        allMissIdxs.clear();
        allMissIdxs.resize(runBsLen, std::vector<std::vector<int>>(mLayerNum));
        for (size_t i = 0; i < runBsLen; i++) {
            if (mDocsTables.find(mReqIdList[i]) == mDocsTables.end() || mTopkLenList[i] <= 0) {
                mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| topk len is zero: %d\n",
                    mDecodeStep, mTopkLenList[i]);
                    continue;
            }
            RunOneBsPrefetch(mReqIdList[i], mTopkLenList[i], mBsIndexList[i], i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        mLogger.log(LogLevel::INFO,
            "Decode step: %u, |KVCache Prefetch| Finish async pretch cost: %lu\n",
            mDecodeStep, duration.count());
        return 0;
    }

    
    std::vector<std::vector<std::vector<int>>> GSAPrefetchEngineC::ObtainLoadBlocks()
    {
        return allNeedLoadBlock;
    }

    std::vector<std::vector<std::vector<int>>> GSAPrefetchEngineC::ObtainMissIdxs()
    {
        return allMissIdxs;
    }

    std::map<int, std::vector<std::map<int, int>>> GSAPrefetchEngineC::ObtainBlocksMap()
    {
        return mBlocksMap;
    }
} // namespace uc