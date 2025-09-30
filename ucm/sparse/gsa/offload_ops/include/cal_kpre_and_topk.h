#ifndef CAL_KPRE_AND_TOPK_H
#define CAL_KPRE_AND_TOPK_H
#include <atomic>
#include <vector>
#include <mutex>
#include <torch/torch.h>
#include "k_repre.h"
#include "select_topk_block.h"
#include "thread_safe_queue.h"

class __attribute__((visibility("hidden"))) CalKpreAndTopk
{
public:
    std::vector<torch::Tensor> m_kCache;
    std::vector<torch::Tensor> m_qCache;
private:
    std::vector<std::atomic<bool>> m_kReady;
    std::vector<std::atomic<bool>> m_qReady;
    std::vector<torch::Tensor> m_kfCache;
    std::vector<torch::Tensor> m_topkCache;
    std::vector<std::atomic<bool>> m_topkFlag;
    torch::Tensor m_calcKpreBlockTable;
    // std::vector<uint32_t> m_calcKpreBlockTable;
    std::vector<uint32_t> m_calcRepreSlotMapping;
    std::vector<std::vector<uint32_t>> m_repreSlotMapping;
    bool m_needCalPre;
    bool m_needCalTopk;
    uint32_t m_layerNum;
    uint32_t m_headSize;
    uint32_t m_qNumHeads;
    uint32_t m_kNumHeads;
    uint32_t m_blockSize;   
    uint32_t m_numKpre;
    std::vector<uint32_t> m_topkLens;
    std::vector<uint32_t> m_calTopkIdx;
    std::vector<bool> m_isDecode;
    std::mutex m_calLock;
    std::thread m_calculateThread;
    std::thread m_copyThread;
    std::atomic<bool> m_running{false};
    std::condition_variable m_dataReady;
    ThreadSafeQueue m_copyQueue;
    uint32_t m_count;
    SelectTopkBlock::TopkBlockSelector* m_topkComputer = nullptr;
    KRepre::KRepreComputer* m_kpreComputer = nullptr;
public:
    CalKpreAndTopk(uint32_t layerNum, uint32_t blockSize, uint32_t maxBs, uint32_t numHeads, uint32_t headSize);
    ~CalKpreAndTopk();
    void SetKpreMethodParam(uint32_t numHeads, uint32_t numKpre);
    void SetKpreCache(std::vector<torch::Tensor>& kpreCache);
    void SetTopkCache(std::vector<torch::Tensor>& topkCache, std::vector<uint32_t>& topkLens);
    void SetCommonParam(std::vector<uint32_t>& calTopkIdx, std::vector<bool>& isDecode);
    void SetTopkParam(std::vector<std::vector<uint32_t>>& repreSlotMapping);
    // void SetKpreParam(std::vector<uint32_t>& calcKpreBlockTable, std::vector<uint32_t>& calcRepreSlotMapping);
    void SetKpreParam(torch::Tensor& calcKpreBlockTable, std::vector<uint32_t>& calcRepreSlotMapping);
    bool SetKpreDataReady(uint32_t layerIdx);
    void SetTopkDataReady(uint32_t layerIdx);
    void AddCopyReq(bool needCalKpre, uint32_t layerIdx, std::vector<int32_t>& locations, torch::Tensor& npuTensor);
    bool IsCalculateFinish();
private:
    void Calculate();
    void CopyData();
    void CalForOneLayer(uint32_t curLayer);
    void CalculateKpre(uint32_t curLayer);
    void CalculateTopk(uint32_t curLayer); 
    bool CheckDataStatus() const;
    void ResetReady();
};

#endif