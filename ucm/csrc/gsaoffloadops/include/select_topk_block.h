#ifndef SELECT_TOPK_BLOCK_H
#define SELECT_TOPK_BLOCK_H

#include <cstdint>
#include <vector>
#include <string>
#include "vec_product.h"

namespace SelectTopkBlock {

using VecProductClass = VecProduct::VecProduct;

/**
* @brief Topk块选择器
*
* 提供基于向量点积的TopK块选择功能，支持多线程并行计算和SIMD优化
*/
class TopkBlockSelector {
public:
    TopkBlockSelector() : startWindow_(1), endWindow_(2) {};
    
    /**
    * @brief 禁用拷贝构造和赋值
    */
    TopkBlockSelector(const TopkBlockSelector&) = delete;
    TopkBlockSelector& operator=(const TopkBlockSelector&) = delete;

    /**
    * @brief 设置窗口参数
    * @param startWindow 头窗口尺寸
    * @param endWindow 尾窗口尺寸
    */
    void SetWindowSize(uint32_t startWindow, uint32_t endWindow) {
        startWindow_ = startWindow;
        endWindow_ = endWindow;
    }
    
    /**
    * @brief 计算K-Q点积分数
    * @param qMean 查询向量 [qHead, headSize]
    * @param kRepre 键表示向量 [numBlock, kHead, numKrepre, headSize]
    * @param numBlock 块数量
    * @param kHead key注意力头数量
    * @param numKrepre 最大归一化维度
    * @param headSize 单头维度
    * @return std::vector<float> 每个块分数
    */
    std::vector<float> ComputeKQDotScores(const float* __restrict qMean, const float* __restrict kRepre, 
                                          uint32_t numBlock,
                                          uint32_t kHead,
                                          uint32_t numKrepre,
                                          uint32_t headSize);
    /**
    * @brief 计算单kHead对应q均值，计算结果用qHead尺寸的vector原址存储
    * @param q 查询向量 [qHead, headSize]
    * @param kHead key注意力头数量
    * @param qHead query注意力头数量
    * @param headSize 单头维度
    */
    void ComputeQHeadMean(float* __restrict q, 
                          uint32_t kHead,
                          uint32_t qHead,
                          uint32_t headSize);

    /**
    * @brief 选择TopK块
    * 
    * @param q 查询向量均值
    * @param kRepre 键表示向量
    * @param numBlock 块数量
    * @param kHead key注意力头数量
    * @param qHead query注意力头数量
    * @param numKrepre 最大归一化维度
    * @param headSize 单头维度
    * @param topkLength TopK中的K值
    * @param topkResult 输出Topk结果 [topkLength]
    */
    void SelectTopK(float* q, const float* kRepre, 
                    uint32_t numBlock, uint32_t kHead, uint32_t qHead,
                    uint32_t numKrepre, uint32_t headSize,
                    uint32_t topkLength, int32_t* topkResult);

    /**
    * @brief 批量选择TopK块
    * 
    * @param qCacheVec 查询向量缓存 std::vector<float*>，每个元素指向 [qHead, headSize]
    * @param kfCacheVec 键表示向量缓存 std::vector<float*>，每个元素指向 [numBlock[i], kHead, numKrepre, headSize]
    * @param topkCacheVec 输出Topk结果缓存 std::vector<uint32_t*>，每个元素指向 [maxTopkLength]
    * @param numBatch 批次数量
    * @param numBlockVec 每个批次的块数量 std::vector<uint32_t
    * @param kHead key注意力头数量
    * @param qHead query注意力头数量
    * @param numKrepre 最大归一化维度
    * @param headSize 单头维度
    * @param topkLengthVec 每个批次的TopK长度 std::vector<uint32_t>
    */
    void SelectTopKBS(const std::vector<float*>& qCacheVec,
                      const std::vector<const float*>& kfCacheVec,
                      const std::vector<int32_t*>& topkCacheVec,
                      uint32_t numBatch, 
                      const std::vector<uint32_t>& numBlockVec,
                      uint32_t kHead, uint32_t qHead,
                      uint32_t numKrepre, uint32_t headSize,
                      const std::vector<uint32_t>& topkLengthVec);

private:
    uint32_t startWindow_;
    uint32_t endWindow_;
    class ThreadLocalVecProduct {
    public:
        static const VecProductClass& GetInstance();
    private:
        ThreadLocalVecProduct() = default;
    };

    /**
    * @brief TopK算法实现
    * @param scores 分数向量
    * @param numScores 分数向量
    * @param k TopK中的K值
    * @param topkIndices 输出TopK索引 [k]
    */
   void TopKImpl(const float* scores, uint32_t numScores, uint32_t k, int32_t* topkIndices);

    /** 
    * @brief 计算单块分数
    */
   static float ComputeBlockScore(float* qMean, const float* blockBase,
                                  uint32_t kHead, uint32_t numKrepre,
                                  uint32_t headSize, const VecProductClass& vecProduct);
    
    /** 
    * @brief 参数验证
    */
   static bool ValidateParameters(float* qMean, const float* kRepre,
                                  uint32_t numBlock, uint32_t kHead, uint32_t qHead,
                                  uint32_t numKrepre, uint32_t headSize);
};

}

#endif
