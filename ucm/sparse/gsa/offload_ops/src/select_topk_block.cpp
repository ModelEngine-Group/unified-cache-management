#include "select_topk_block.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace SelectTopkBlock {
#define OMP_THREAD_NUM 16u

bool TopkBlockSelector::ValidateParameters(float* q, const float* kRepre, uint32_t numBlock,
                                           uint32_t kHead, uint32_t qHead, uint32_t numKrepre,
                                           uint32_t headSize)
{
    return (q != nullptr) && (kRepre != nullptr) && (numBlock > 0) && (kHead > 0) && (qHead > 0) &&
           (numKrepre > 0) && (headSize > 0);
}

void TopkBlockSelector::TopKImpl(const float* scores, uint32_t numScores, uint32_t k,
                                 int32_t* topkIndices)
{
    if (startWindow_ + endWindow_ >= numScores || k >= numScores || k == 0) {
        for (uint32_t i = 0; i < numScores; ++i) { topkIndices[i] = i; }
        return;
    }
    uint32_t idx = 0;
    for (uint32_t i = 0; i < startWindow_; ++i) { topkIndices[idx++] = i; }
    for (uint32_t i = 0; i < endWindow_; ++i) { topkIndices[idx++] = numScores - endWindow_ + i; }
    int32_t midCount = k - startWindow_ - endWindow_;
    if (midCount > 0) {
        std::vector<uint32_t> middleIndices;
        middleIndices.reserve(numScores - startWindow_ - endWindow_);
        for (uint32_t i = startWindow_; i < numScores - endWindow_; ++i) {
            middleIndices.push_back(i);
        }
        std::stable_sort(
            middleIndices.begin(), middleIndices.end(),
            [scores](uint32_t lhs, uint32_t rhs) { return scores[lhs] > scores[rhs]; });
        for (int32_t i = 0; i < midCount; ++i) { topkIndices[idx++] = middleIndices[i]; }
    }
}

float TopkBlockSelector::ComputeBlockScore(float* qMean, const float* blockBase, uint32_t kHead,
                                           uint32_t numKrepre, uint32_t headSize,
                                           const VecProductClass& vecProduct)
{
    const size_t headOffset = headSize;
    const size_t normOffset = headSize;
    const size_t headBlockOffset = static_cast<size_t>(numKrepre) * headSize;
    float blockScore = 0.0f;
    for (uint32_t idxHead = 0; idxHead < kHead; ++idxHead) {
        const float* q = qMean + idxHead * headOffset;
        const float* headBase = blockBase + idxHead * headBlockOffset;
        float maxScore = -std::numeric_limits<float>::max();
        for (uint32_t idxNorm = 0; idxNorm < numKrepre; ++idxNorm) {
            const float* k = headBase + idxNorm * normOffset;
            if (idxNorm + 1 < numKrepre) {
                __builtin_prefetch(headBase + (idxNorm + 1) * normOffset, 0, 3);
            }
            const float score = vecProduct.VectorDot(q, k, headSize);
            maxScore = std::fmax(maxScore, score);
        }
        blockScore += maxScore;
    }
    return blockScore;
}

const VecProductClass& TopkBlockSelector::ThreadLocalVecProduct::GetInstance()
{
    static thread_local VecProductClass instance;
    return instance;
}

std::vector<float> TopkBlockSelector::ComputeKQDotScores(const float* __restrict qMean,
                                                         const float* __restrict kRepre,
                                                         uint32_t numBlock, uint32_t kHead,
                                                         uint32_t numKrepre, uint32_t headSize)
{
    std::vector<float> blockScores(numBlock, 0.0f);
    const size_t blockOffset = static_cast<size_t>(kHead * numKrepre * headSize);
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
    for (uint32_t idxBlock = 0; idxBlock < numBlock; ++idxBlock) {
        const VecProductClass& vecProduct = ThreadLocalVecProduct::GetInstance();
        const float* blockBase = kRepre + idxBlock * blockOffset;
        if (idxBlock + 1 < numBlock) {
            __builtin_prefetch(kRepre + (idxBlock + 1) * blockOffset, 0, 1);
        }
        blockScores[idxBlock] = ComputeBlockScore(const_cast<float*>(qMean), blockBase, kHead,
                                                  numKrepre, headSize, vecProduct);
    }
    return blockScores;
}

void TopkBlockSelector::ComputeQHeadMean(float* __restrict q, uint32_t kHead, uint32_t qHead,
                                         uint32_t headSize)
{
    if (kHead == qHead) { return; }
    const VecProductClass& vecProduct = ThreadLocalVecProduct::GetInstance();
    const uint32_t groupSize = qHead / kHead;
    for (uint32_t kIdx = 0; kIdx < kHead; ++kIdx) {
        const uint32_t sourceIdx = kIdx * groupSize;
        const float* groupVectors = q + sourceIdx * headSize;
        float* meanVector = q + kIdx * headSize;
        vecProduct.VectorMean(groupVectors, meanVector, headSize, groupSize);
    }
}

void TopkBlockSelector::SelectTopK(float* q, const float* kRepre, uint32_t numBlock, uint32_t kHead,
                                   uint32_t qHead, uint32_t numKrepre, uint32_t headSize,
                                   uint32_t topkLength, int32_t* topkResult)
{
    if (!ValidateParameters(q, kRepre, numBlock, kHead, qHead, numKrepre, headSize) ||
        topkResult == nullptr || topkLength == 0) {
        return;
    }
    ComputeQHeadMean(q, kHead, qHead, headSize);
    const std::vector<float> scores =
        ComputeKQDotScores(q, kRepre, numBlock, kHead, numKrepre, headSize);
    TopKImpl(scores.data(), numBlock, topkLength, topkResult);
}

void TopkBlockSelector::SelectTopKBS(const std::vector<float*>& qCacheVec,
                                     const std::vector<const float*>& kfCacheVec,
                                     const std::vector<int32_t*>& topkCacheVec, uint32_t numBatch,
                                     const std::vector<uint32_t>& numBlockVec, uint32_t kHead,
                                     uint32_t qHead, uint32_t numKrepre, uint32_t headSize,
                                     const std::vector<uint32_t>& topkLengthVec)
{
    for (uint32_t bs = 0; bs < numBatch; ++bs) {
        const uint32_t numBlock = numBlockVec[bs];
        const uint32_t topkLength = topkLengthVec[bs];
        float* q = qCacheVec[bs];
        const float* kRepre = kfCacheVec[bs];
        int32_t* topkResult = topkCacheVec[bs];
        SelectTopK(q, kRepre, numBlock, kHead, qHead, numKrepre, headSize, topkLength, topkResult);
    }
}

} // namespace SelectTopkBlock