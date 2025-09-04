#ifndef ATB_KV_CACHE_PRE_H
#define ATB_KV_CACHE_PRE_H
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <pybind11/numpy.h>
#include <chrono>
#include <stdio.h>
#include <stdarg.h>
#include <torch/torch.h>
#include <cstring>
#include <omp.h>
#include <unordered_set>
#include <kvcache_log.h>
#include <sstream>
#include <map>
#include <stdexcept>

namespace py = pybind11;

namespace ucmprefetch
{
    typedef struct {
        int topkLen;
        int reqID;
        int layerID;
        int topkIndex;
        int bsIndex;
    } PrefetchReqInfo;

    class __attribute__((visibility("hidden"))) GSAPrefetchEngineC 
    {
    private:
        std::map<int, std::vector<std::map<int, int>>> mDocsTables;
        std::map<int, std::vector<std::map<int, int>>> mBlocksMap;
        torch::Tensor mLoadSuccessBlocks;
        torch::Tensor mFreeBlock;
        torch::Tensor mFreeBlockLen;
        torch::Tensor mSuccessTableLen;
        torch::Tensor mUseTopkIdxs;
        int mLayerNum;
        int mRank = -1;
        torch::Tensor mLoadSuccessBlocksCPU;
        uint32_t mMaxBs = 30;
        int *mReqIdList = NULL;
        int *mTopkLenList = NULL;
        int *mBsIndexList = NULL;
        uint32_t runBsLen = 0;
        bool mIsLog = false;
        Logger mLogger;
        uint32_t mDecodeStep = 0;
        uint32_t mMaxTopkLen = 0;
        uint32_t mMaxBlocksLen = 0;
        std::unordered_set<int> mDelSeqIds;
        std::vector<std::vector<std::vector<int>>> allNeedLoadBlock;
        std::vector<std::vector<std::vector<int>>> allMissIdxs;
    
    private:
        void LoadKVToHBM(std::vector<int> loadNPUBlockIDs,
            std::vector<int> missIdxs, int layerID, int reqID);
        
        void GetHitAndMissBlock(PrefetchReqInfo oneBsInfo,
            std::unordered_set<int> &hitBlocks,
            std::map<int, int> &hitBlocksIdx,
            std::vector<int> &missIdxs);
        
        void RunPrefetchH2D(PrefetchReqInfo oneBsInfo,
            std::unordered_set<int> &hitBlocks,
            std::map<int, int> &hitBlocksIdx,
            std::vector<int> &missIdxs);
        
        void RunOneBsPrefetch(int reqID, int topkLen,
            int bsIndex, int topkIndex);

    public:
        ~GSAPrefetchEngineC();

        GSAPrefetchEngineC(torch::Tensor &freeBlock,
            torch::Tensor &loadSuccessBlocks,
            torch::Tensor & freeBlockLen,
            torch::Tensor &successTableLen,
            bool isLog);

        void SetBlocksMap(int reqID, std::vector<int> &blockTableList,
            std::vector<int> &selectIndex);

        void CheckInputIndex(uint32_t maxLen, uint32_t index);

        void AddBlocksMap(int reqID, int idx, int blockID);

        void DelBlocksMap(int reqID);

        void SetBlockTableInfo(torch::Tensor &blockTables,
            torch::Tensor &blockLengths,
            torch::Tensor &inputTopkBuf, int step);

        void RunAsyncPrefetchBs(std::vector<int> &reqIDsInput,
            std::vector<int> &topkLensInput,
            std::vector<int> &bsIndexInput, int rank);
        
        int CallPrefetchProcessFun();

        void PrintMap(int reqID, int i);

        std::vector<std::vector<std::vector<int>>> ObtainLoadBlocks();

        std::vector<std::vector<std::vector<int>>> ObtainMissIdxs();

        std::map<int, std::vector<std::map<int, int>>> ObtainBlocksMap();
    };
    
} // namespace uc

#endif
