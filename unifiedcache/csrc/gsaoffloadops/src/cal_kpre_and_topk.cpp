#include <thread>
#include <iostream>
#include "cal_kpre_and_topk.h"
#include "k_repre.h"
#include "select_topk_block.h"

CalKpreAndTopk::CalKpreAndTopk(uint32_t layerNum, uint32_t blockSize, uint32_t maxBs, uint32_t numHeads, uint32_t headSize)
:m_kReady(layerNum), m_qReady(layerNum), m_topkFlag(layerNum)
{
    m_layerNum = layerNum;
    m_blockSize = blockSize;
    m_qNumHeads = numHeads;
    m_headSize = headSize;
    auto optionsForQCache = torch::TensorOptions().device("cpu").dtype(torch::kFloat32);
    for (uint32_t i = 0; i < m_layerNum; i++) {
        torch::Tensor layerQCache = torch::zeros({maxBs, m_qNumHeads, m_headSize}, optionsForQCache);
        m_qCache.push_back(layerQCache);
        m_kReady[i].store(false);
        m_qReady[i].store(false);
        m_topkFlag[i].store(true);
    }
    m_topkComputer = new SelectTopkBlock::TopkBlockSelector();
    m_kpreComputer = new KRepre::KRepreComputer();
    m_numKpre = 1;
}

void CalKpreAndTopk::SetKpreMethodParam(uint32_t maxBlockNum, uint32_t numHeads, uint32_t numKpre)
{
    m_kNumHeads = numHeads;
    m_numKpre = numKpre;
    auto optionsForKCache = torch::TensorOptions().device("cpu").dtype(torch::kFloat32);
    for (uint32_t i = 0; i < m_layerNum; i++) {
        torch::Tensor layerKCache = torch::zeros({maxBlockNum, m_kNumHeads, m_blockSize, m_headSize}, optionsForKCache);
        m_kCache.push_back(layerKCache);
    }
}

void CalKpreAndTopk::SetKpreCache(std::vector<torch::Tensor>& kpreCache)
{
    m_kfCache = kpreCache;
    m_running = true;
    m_calculateThread = std::thread([this]() {
        this->Calculate();
    });
}

void CalKpreAndTopk::SetTopkCache(std::vector<torch::Tensor>& topkCache, std::vector<uint32_t>& topkLens)
{
    m_topkCache = topkCache;
    m_topkLens = topkLens;
}

CalKpreAndTopk::~CalKpreAndTopk()
{
    m_running = false;
    m_dataReady.notify_all();
    if (m_calculateThread.joinable()) {
        m_calculateThread.join();
    }
    delete m_topkComputer;
    m_topkComputer = nullptr;
    delete m_kpreComputer;
    m_kpreComputer = nullptr;
}

void CalKpreAndTopk::SetCommonParam(std::vector<uint32_t>& calTopkIdx, std::vector<bool>& isDecode)
{
    m_calTopkIdx = calTopkIdx;
    m_isDecode = isDecode;
    m_needCalTopk = false;
    m_needCalPre = false;
}

void CalKpreAndTopk::SetTopkParam(std::vector<std::vector<uint32_t>>& repreSlotMapping)
{
    m_needCalTopk = true;
    m_repreSlotMapping = repreSlotMapping;
    for (auto& atomic_val : m_topkFlag) {
        atomic_val.store(false, std::memory_order_relaxed);
    }
}

void CalKpreAndTopk::SetKpreParam(std::vector<uint32_t>& calcKpreBlockTable, std::vector<uint32_t>& calcRepreSlotMapping)
{
    m_needCalPre = true;
    m_calcKpreBlockTable = calcKpreBlockTable;
    m_calcRepreSlotMapping = calcRepreSlotMapping;
}

void CalKpreAndTopk::SetKpreDataReady(uint32_t layerIdx)
{
    if (m_needCalPre && layerIdx == 0) {
        std::lock_guard<std::mutex> lock(m_calLock);
        m_kReady[layerIdx].store(true, std::memory_order_release);
        m_dataReady.notify_one();
    } else {
        m_kReady[layerIdx].store(true, std::memory_order_release);
    }
}

void CalKpreAndTopk::SetTopkDataReady(uint32_t layerIdx)
{
    if (!m_needCalPre && layerIdx == 0) {
        std::lock_guard<std::mutex> lock(m_calLock);
        m_qReady[layerIdx].store(true, std::memory_order_release);
        m_dataReady.notify_one();
    } else {
        m_qReady[layerIdx].store(true, std::memory_order_release);
    }
}

bool CalKpreAndTopk::CheckDataStatus() const
{
    if (m_needCalPre) {
        return m_running && m_kReady[0];
    } else {
        return m_running && m_qReady[0];
    }
}

void CalKpreAndTopk::Calculate()
{
    while (m_running) {
        std::unique_lock<std::mutex> lock(m_calLock);
        m_dataReady.wait(lock, [this]() { 
            return this -> CheckDataStatus();
        });
        for (uint32_t i = 0; i < m_layerNum; i++) {
            CalForOneLayer(i);
            m_topkFlag[i].store(true, std::memory_order_release);
        }
        ResetReady();
        if (!m_running) {
            break;
        }
    }
}

void CalKpreAndTopk::CalForOneLayer(uint32_t curLayer)
{
    if (m_needCalPre) {
        while(!m_kReady[curLayer].load(std::memory_order_acquire));
        CalculateKpre(curLayer);
    }
    if (m_needCalTopk) {
        while(!m_qReady[curLayer].load(std::memory_order_acquire));
        CalculateTopk(curLayer);
    }
}

void CalKpreAndTopk::CalculateKpre(uint32_t curLayer)
{
    if (m_calcKpreBlockTable.size() == 1) {
        m_kpreComputer -> ComputeKRepreBlock(m_kCache[curLayer][0].data_ptr<float>(),
                                             m_kNumHeads, m_blockSize, m_headSize,
                                             m_kfCache[curLayer][m_calcRepreSlotMapping[0]].data_ptr<float>());
    } else {
        uint32_t numBlock = m_calcKpreBlockTable.size();
        std::vector<float*> kArray;
        std::vector<float*> kRepreBlockArray;
        for (uint32_t i = 0; i < numBlock; i++) {
            kArray.push_back(m_kCache[curLayer][i].data_ptr<float>());
            kRepreBlockArray.push_back(m_kfCache[curLayer][m_calcRepreSlotMapping[i]].data_ptr<float>());
        }
        m_kpreComputer -> ComputeKRepre(kArray, numBlock, m_kNumHeads, m_blockSize, m_headSize, kRepreBlockArray);
    }
}

void CalKpreAndTopk::CalculateTopk(uint32_t curLayer)
{
    std::vector<float*> qCache;
    for (auto i: m_calTopkIdx) {
        qCache.push_back(m_qCache[curLayer][i].data_ptr<float>());
    }
    std::vector<const float*> kfCacheVec;
    std::vector<uint32_t> numBlockVec;
    std::vector<int32_t*> topkCacheVec;
    for (uint32_t i = 0; i < m_isDecode.size(); i++) {
        if (m_isDecode[i] == false) {
            continue;
        }
        uint32_t startPosition = m_repreSlotMapping[i][0];
        uint32_t size = m_repreSlotMapping[i].size();
        kfCacheVec.push_back(m_kfCache[curLayer][startPosition].data_ptr<float>());
        numBlockVec.push_back(size);
        topkCacheVec.push_back(m_topkCache[curLayer][i].data_ptr<int32_t>());
    }
    m_topkComputer -> SelectTopKBS(qCache, kfCacheVec, topkCacheVec, m_calTopkIdx.size(),
                                   numBlockVec, m_kNumHeads, m_qNumHeads, m_numKpre, m_headSize,
                                   m_topkLens);
}

void CalKpreAndTopk::ResetReady()
{
    if (m_needCalTopk) {
        for (auto& atomic_val : m_qReady) {
            atomic_val.store(false, std::memory_order_relaxed);
        }
    }
    if (m_needCalPre) {
        for (auto& atomic_val : m_kReady) {
            atomic_val.store(false, std::memory_order_relaxed);
        }
    }
}

bool CalKpreAndTopk::IsCalculateFinish()
{
    return m_topkFlag[m_layerNum - 1] == true;
}