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
#include "../../../../store/ucmstore.h"

namespace py = pybind11;

namespace ucmprefetch
{
    typedef struct {
        int topkLen;
        std::string reqID;
        int layerID;
        int topkIndex;
        int bsIndex;
    } PrefetchReqInfo;

    class ThreadPool 
    {
    public:
        static ThreadPool *GetInst() 
        {
            static ThreadPool pool(1);
            return &pool;
        }

        ~ThreadPool();

        template<class F, class ... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

        size_t GetActiveThreads() const;
    
    private:
        explicit ThreadPool(size_t threadCount);
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        mutable std::mutex queueMutex;
        bool stop;
        std::condition_variable condition;
        std::atomic<size_t> activeThreads{0};
        size_t maxThreads;
    };

    void MutliBSThreadFun(void *args);

    class __attribute__((visibility("hidden"))) GSAPrefetchEngineC 
    {
    private:
        std::map<std::string, std::vector<std::map<int, int>>> mDocsTables;
        std::map<std::string, std::vector<std::map<int, int>>> mBlocksMap;
        torch::Tensor mLoadSuccessBlocks;
        torch::Tensor mFreeBlock;
        torch::Tensor mFreeBlockLen;
        torch::Tensor mSuccessTableLen;
        torch::Tensor mUseTopkIdxs;
        int mLayerNum;
        int mRank = -1;
        uint32_t mMaxBs = 30;
        std::vector<std::string> mReqIdList;
        int *mTopkLenList = NULL;
        int *mBsIndexList = NULL;
        uint32_t runBsLen = 0;
        bool mIsLog = false;
        bool mIsPrefetchDone = true;
        bool mUseMla = false;
        Logger mLogger;
        ThreadPool *mThreadPool;
        uint32_t mDecodeStep = 0;
        uint32_t mMaxTopkLen = 0;
        uint32_t mMaxBlocksLen = 0;
        std::unordered_set<std::string> mDelSeqIds;
        std::map<std::string, std::vector<std::vector<int>>> allNeedLoadBlock;
        std::map<std::string, std::vector<std::vector<int>>> allMissIdxs;
        std::map<std::string, int> mPromptLen;
        UC::CCStore<> *mStore = nullptr;
        std::vector<torch::Tensor> mKvCaches;
        uint32_t mBlockSize = 128;
        uint32_t mTensorElemSize = 2; // fp16
        uint32_t mHeadNum = 40;
        uint32_t mHeadSzie = 128;
        uint32_t mTPSize = 2;
        std::map<std::string, std::vector<std::string>> mAllBlcoksHash;
        uint32_t mKVSzieBytes = 0;
        uint32_t mExtraTopkLen = 16;
    public:
        std::mutex mMutex;
        bool mStopPrefetch = false;
    
    private:
        void LoadKVToHBM(std::vector<int> loadNPUBlockIDs,
            std::vector<int> missIdxs, int layerID, std::string reqID);
        
        void GetHitAndMissBlock(PrefetchReqInfo oneBsInfo,
            std::unordered_set<int> &hitBlocks,
            std::map<int, int> &hitBlocksIdx,
            std::vector<int> &missIdxs);
        
        void RunPrefetchH2D(PrefetchReqInfo oneBsInfo,
            std::unordered_set<int> &hitBlocks,
            std::map<int, int> &hitBlocksIdx,
            std::vector<int> &missIdxs);
        
        void RunOneBsPrefetch(std::string reqID, int topkLen,
            int bsIndex, int topkIndex);

    public:
        ~GSAPrefetchEngineC();

        GSAPrefetchEngineC(torch::Tensor &freeBlock,
            torch::Tensor &loadSuccessBlocks,
            torch::Tensor &freeBlockLen,
            torch::Tensor &successTableLen,
            std::vector<uint32_t> &kvShape,
            bool useMla,
            bool isLog,
            int tpSize,
            int rank,
            int extraTopkLen
        );

        void SetBlocksMap(std::string reqID, std::vector<int> &blockTableList,
            std::vector<int> &selectIndex, std::vector<std::string> &blocksHash,
            int maxIdx);
        
        void SetBlocksMapMultiLayer(std::string reqID,
            std::vector<std::map<int, int>> &remainMap,
            std::vector<std::map<int, int>> &prefetchMap,
            std::vector<std::string> &blocksHash,
            int maxIdx);

        void CheckInputIndex(uint32_t maxLen, uint32_t index);

        void AddBlocksMap(std::string reqID, int idx, int blockID);

        void DelBlocksMap(std::string reqID);

        void DelReqIDRun();

        void SetBlockTableInfo(torch::Tensor &blockTables,
            torch::Tensor &blockLengths,
            torch::Tensor &inputTopkBuf, int step);

        void RunAsyncPrefetchBs(std::vector<std::string> &reqIDsInput,
            std::vector<int> &topkLensInput,
            std::vector<int> &bsIndexInput,
            std::vector<torch::Tensor> &kvCaches,
            void *storePtr);
        
        int CallPrefetchProcessFun();

        void PrintMap(std::string reqID, int i);

        bool GetPrefetchStatus();

        void SetPrefetchStatus(bool flag);

        void SetModelRunningStatus(bool flag);
        
        size_t GetOffset(uint32_t layerID, bool isV);

        std::map<std::string, std::vector<std::vector<int>>> ObtainLoadBlocks();

        std::map<std::string, std::vector<std::vector<int>>> ObtainMissIdxs();

        std::map<std::string, std::vector<std::map<int, int>>> ObtainBlocksMap();

        std::map<std::string, std::vector<std::map<int, int>>> ObtainDocsMap();
    };
    
} // namespace uc

#endif
