#include <stdexcept>
#include <cassert>
#include "k_repre.h"


namespace KRepre {
#define OMP_THREAD_NUM 32u

const VecProductClass& KRepreComputer::ThreadLocalVecProduct::GetInstance()
{
    thread_local static VecProductClass instance;
    return instance;
}

void KRepreComputer::ComputeKRepreBlock(const float* __restrict kArray,
                        uint32_t kHead,
                        uint32_t blockSize,
                        uint32_t headSize,
                        float* __restrict kRepreBlock) const
{
    // 获取本地线程实例
    const auto& vecProduct = ThreadLocalVecProduct::GetInstance();

    for (uint32_t idxHead = 0; idxHead < kHead; ++idxHead) {
        const float* kArraySingleHead = kArray + idxHead * blockSize * headSize;
        float* kRepreBlockSingleHead = kRepreBlock + idxHead * headSize;

        vecProduct.VectorMean(
            kArraySingleHead,
            kRepreBlockSingleHead,
            headSize,
            blockSize
        );
    }
}
    
void KRepreComputer::ComputeKRepre(const std::vector<float*>& kArray,
                   uint32_t numBlock,
                   uint32_t kHead,
                   uint32_t blockSize,
                   uint32_t headSize,
                   const std::vector<float*>& kRepreBlockArray) const
{
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
    for (uint32_t idxBlock = 0; idxBlock < numBlock; ++idxBlock) {
        const float* kArrayCurrentBlock = kArray[idxBlock];
        float * KRepreCurrentBlock = kRepreBlockArray[idxBlock];

        ComputeKRepreBlock(
            kArrayCurrentBlock,
            kHead,
            blockSize,
            headSize,
            KRepreCurrentBlock
        );
    }
}
}