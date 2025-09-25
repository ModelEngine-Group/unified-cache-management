#include <stdexcept>
#include <cassert>
#include "k_repre.h"


namespace Kpre {
#define OMP_THREAD_NUM 32u

const VecProductClass& KpreComputer::ThreadLocalVecProduct::GetInstance()
{
    thread_local static VecProductClass instance;
    return instance;
}

void KpreComputer::ComputeKpreBlock(const float* __restrict kArray,
                        uint32_t kHead,
                        uint32_t blockSize,
                        uint32_t headSize,
                        float* __restrict kpreBlock) const
{
    // 获取本地线程实例
    const auto& vecProduct = ThreadLocalVecProduct::GetInstance();

    for (uint32_t idxHead = 0; idxHead < kHead; ++idxHead) {
        const float* kArraySingleHead = kArray + idxHead * blockSize * headSize;
        float* kpreBlockSingleHead = kpreBlock + idxHead * headSize;

        vecProduct.VectorMean(
            kArraySingleHead,
            kpreBlockSingleHead,
            headSize,
            blockSize
        );
    }
}
    
void KpreComputer::ComputeKpre(const std::vector<float*>& kArray,
                   uint32_t numBlock,
                   uint32_t kHead,
                   uint32_t blockSize,
                   uint32_t headSize,
                   const std::vector<float*>& kpreBlockArray) const
{
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
    for (uint32_t idxBlock = 0; idxBlock < numBlock; ++idxBlock) {
        const float* kArrayCurrentBlock = kArray[idxBlock];
        float * KpreCurrentBlock = kpreBlockArray[idxBlock];

        ComputeKpreBlock(
            kArrayCurrentBlock,
            kHead,
            blockSize,
            headSize,
            KpreCurrentBlock
        );
    }
}
}