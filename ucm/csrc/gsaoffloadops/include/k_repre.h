#ifndef K_REPRE_H
#define K_REPRE_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include "vec_product.h"

namespace KRepre {

using VecProductClass = VecProduct::VecProduct;

/**
* @brief Key表征计算器
*
* 提供基于向量均值的Key表征计算功能，支持多线程并行计算和SIMD优化
*/
class KRepreComputer {
public:
    KRepreComputer() = default;
    
    /**
    * @brief 禁用拷贝构造和赋值
    */
    KRepreComputer(const KRepreComputer&) = delete;
    KRepreComputer& operator=(const KRepreComputer&) = delete;

    /**
    * @brief 计算单个Block的K表征
    * 
    * @param kArray k向量指针数组 [kHead, blockSize, headSize]
    * @param kHead k头数量
    * @param blockSize block内k向量数量
    * @param headSize 向量维度
    * @param kRepreBlock 单block 表征 [kHead, headSize]
    */
    void ComputeKRepreBlock(const float* __restrict kArray,
                            uint32_t kHead,
                            uint32_t blockSize,
                            uint32_t headSize,
                            float* __restrict kRepreBlock) const;
    
    /**
    * @brief 计算多个Block的K表征（使用OpenMP并行优化）
    * 
    * @param kArray k向量指针数组 [kHead, blockSize, headSize]
    * @param numBlock block数量
    * @param kHead k头数量
    * @param blockSize block内k向量数量
    * @param headSize 向量维度
    * @param kRepreBlockArray 全量K表征 [numBlock, kHead, x, headSize]
    */
    void ComputeKRepre(const std::vector<float*>& kArray,
                       uint32_t numBlock,
                       uint32_t kHead,
                       uint32_t blockSize,
                       uint32_t headSize,
                       const std::vector<float*>& kRepreBlockArray) const;

private:
    // 线程本地VecProduct实例管理
    class ThreadLocalVecProduct {
    public:
        static const VecProductClass& GetInstance();
    private:
        ThreadLocalVecProduct() = default;
    };
};

}

#endif
