#include <cmath>
#include <cstring>
#include <cfloat>
#include "vec_product.h"

namespace VecProduct {

// ==================================================================================
// 编译时选择唯一的实现
// ==================================================================================

#ifdef ARCH_X86

#ifdef __AVX512F__
// ============= AVX512实现 =============
float VecProduct::VectorDotImpl(const float* __restrict q, const float* __restrict k, uint32_t dim)
{
    __m512 sum = _mm512_setzero_ps();
    uint32_t i = 0;
    
    // 512位向量化计算，使用FMA指令
    for (; i + SimdElements::VEC16 <= dim; i += SimdElements::VEC16) {
        const __m512 qVec = _mm512_loadu_ps(&q[i]);
        const __m512 kVec = _mm512_loadu_ps(&k[i]);
        sum = _mm512_fmadd_ps(qVec, kVec, sum);
    }

    // AVX512内置归约指令
    float result = _mm512_reduce_add_ps(sum);

    // 处理剩余元素
    for (; i < dim; ++i) {
        result += q[i] * k[i];
    }

    return result;
}

void VecProduct::VectorMeanImpl(const float* __restrict vectors, float* __restrict mean, uint32_t dim, uint32_t numK)
{
    if (numK == 0) {
        std::fill(mean, mean + dim, 0.0f);
        return;
    }

    const __m512 invNumK = _mm512_set1_ps(1.0f / static_cast<float>(numK));
    uint32_t i = 0;

    // mean初始化为vectors中第一个向量
    for (; i + SimdElements::VEC16 <= dim; i += SimdElements::VEC16) {
        _mm512_storeu_ps(&mean[i], _mm512_loadu_ps(&vectors[i]));
    }
    for (; i < dim; ++i) {
        mean[i] = vectors[i];
    }

    // 累加所有向量
    for (uint32_t k = 1; k < numK; ++k) {
        const float* currentVec = vectors + k * dim;
        i = 0;

        // 向量化累加
        for (; i + SimdElements::VEC16 <= dim; i += SimdElements::VEC16) {
            const __m512 meanVec = _mm512_loadu_ps(&mean[i]);
            const __m512 currentVecAvx512 = _mm512_loadu_ps(&currentVec[i]);
            _mm512_storeu_ps(&mean[i], _mm512_add_ps(meanVec, currentVecAvx512));
        }

        // 处理剩余元素
        for (; i < dim; ++i) {
            mean[i] += currentVec[i];
        }
    }

    // 计算均值
    i = 0;
    for (; i + SimdElements::VEC16 <= dim; i += SimdElements::VEC16) {
        const __m512 sumVec = _mm512_loadu_ps(&mean[i]);
        _mm512_storeu_ps(&mean[i], _mm512_mul_ps(sumVec, invNumK));
    }

    for (; i < dim; ++i) {
        mean[i] *= (1.0f / static_cast<float>(numK));
    }
}

#elif defined(__AVX2__)
// ============= AVX2实现 =============
float VecProduct::VectorDotImpl(const float* __restrict q, const float* __restrict k, uint32_t dim)
{
    __m256 sum = _mm256_setzero_ps();
    uint32_t i = 0;
    
    // 256位向量化计算，使用FMA指令
    for (; i + SimdElements::VEC8 <= dim; i += SimdElements::VEC8) {
        const __m256 qVec = _mm256_loadu_ps(&q[i]);
        const __m256 kVec = _mm256_loadu_ps(&k[i]);
        sum = _mm256_fmadd_ps(qVec, kVec, sum);
    }

    // 向量归约求和
    const __m128 sumHigh = _mm256_extractf128_ps(sum, 1);
    __m128 sumLow = _mm256_extractf128_ps(sum, 0);
    sumLow = _mm_add_ps(sumHigh, sumLow);
    sumLow = _mm_hadd_ps(sumLow, sumLow);
    sumLow = _mm_hadd_ps(sumLow, sumLow);

    float result = _mm_cvtss_f32(sumLow);

    // 处理剩余元素
    for (; i < dim; ++i) {
        result += q[i] * k[i];
    }

    return result;
}

void VecProduct::VectorMeanImpl(const float* __restrict vectors, float* __restrict mean, uint32_t dim, uint32_t numK)
{
    if (numK == 0) {
        std::fill(mean, mean + dim, 0.0f);
        return;
    }

    const __m256 invNumK = _mm256_set1_ps(1.0f / static_cast<float>(numK));
    uint32_t i = 0;

    // mean初始化为vectors中第一个向量
    for (; i + SimdElements::VEC8 <= dim; i += SimdElements::VEC8) {
        _mm256_storeu_ps(&mean[i], _mm256_loadu_ps(&vectors[i]));
    }
    for (; i < dim; ++i) {
        mean[i] = vectors[i];
    }

    // 累加所有向量
    for (uint32_t k = 1; k < numK; ++k) {
        const float* currentVec = vectors + k * dim;
        i = 0;

        // 向量化累加
        for (; i + SimdElements::VEC8 <= dim; i += SimdElements::VEC8) {
            const __m256 meanVec = _mm256_loadu_ps(&mean[i]);
            const __m256 currentVecAvx = _mm256_loadu_ps(&currentVec[i]);
            _mm256_storeu_ps(&mean[i], _mm256_add_ps(meanVec, currentVecAvx));
        }

        // 处理剩余元素
        for (; i < dim; ++i) {
            mean[i] += currentVec[i];
        }
    }

    // 计算均值
    i = 0;
    for (; i + SimdElements::VEC8 <= dim; i += SimdElements::VEC8) {
        const __m256 sumVec = _mm256_loadu_ps(&mean[i]);
        _mm256_storeu_ps(&mean[i], _mm256_mul_ps(sumVec, invNumK));
    }

    for (; i < dim; ++i) {
        mean[i] *= (1.0f / static_cast<float>(numK));
    }
}

#elif defined(__SSE2__)
// ============= SSE2实现 =============
float VecProduct::VectorDotImpl(const float* __restrict q, const float* __restrict k, uint32_t dim)
{
    __m128 sum = _mm_setzero_ps();
    uint32_t i = 0;
    
    // 128位向量化计算
    for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
        const __m128 qVec = _mm_loadu_ps(&q[i]);
        const __m128 kVec = _mm_loadu_ps(&k[i]);
        sum = _mm_add_ps(sum, _mm_mul_ps(qVec, kVec));
    }

    // 向量归约求和
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);

    // 处理剩余元素
    for (; i < dim; ++i) {
        result += q[i] * k[i];
    }

    return result;
}

void VecProduct::VectorMeanImpl(const float* __restrict vectors, float* __restrict mean, uint32_t dim, uint32_t numK)
{
    if (numK == 0) {
        std::fill(mean, mean + dim, 0.0f);
        return;
    }

    const __m128 invNumK = _mm_set1_ps(1.0f / static_cast<float>(numK));
    uint32_t i = 0;

    // mean初始化为vectors中第一个向量
    for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
        _mm_storeu_ps(&mean[i], _mm_loadu_ps(&vectors[i]));
    }
    for (; i < dim; ++i) {
        mean[i] = vectors[i];
    }

    // 累加所有向量
    for (uint32_t k = 1; k < numK; ++k) {
        const float* currentVec = vectors + k * dim;
        i = 0;

        // 向量化累加
        for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
            const __m128 meanVec = _mm_loadu_ps(&mean[i]);
            const __m128 currentVecSse = _mm_loadu_ps(&currentVec[i]);
            _mm_storeu_ps(&mean[i], _mm_add_ps(meanVec, currentVecSse));
        }

        // 处理剩余元素
        for (; i < dim; ++i) {
            mean[i] += currentVec[i];
        }
    }

    // 计算均值
    i = 0;
    for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
        const __m128 sumVec = _mm_loadu_ps(&mean[i]);
        _mm_storeu_ps(&mean[i], _mm_mul_ps(sumVec, invNumK));
    }

    for (; i < dim; ++i) {
        mean[i] *= (1.0f / static_cast<float>(numK));
    }
}
#endif

#elif defined(ARCH_ARM)
// ============= ARM NEON 实现 =============
float VecProduct::VectorDotImpl(const float* __restrict q, const float* __restrict k, uint32_t dim)
{
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    // SIMD向量化计算
    for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
        const float32x4_t qVec = vld1q_f32(&q[i]);
        const float32x4_t kVec = vld1q_f32(&k[i]);
        sum = vmlaq_f32(sum, qVec, kVec);
    }

    // 向量归约求和
    float result = vaddvq_f32(sum);

    // 处理剩余元素
    for (; i < dim; ++i) {
        result += q[i] * k[i];
    }

    return result;
}

void VecProduct::VectorMeanImpl(const float* __restrict vectors, float* __restrict mean, uint32_t dim, uint32_t numK)
{
    if (numK == 0) {
        std::fill(mean, mean + dim, 0.0f);
        return;
    }

    const float32x4_t invNumK = vdupq_n_f32(1.0f / static_cast<float>(numK));
    uint32_t i = 0;

    // mean初始化为vectors中第一个向量
    for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
       vst1q_f32(&mean[i], vld1q_f32(&vectors[i]));
    }
    for (; i < dim; ++i) {
        mean[i] = vectors[i];
    }

    // 累加所有向量
    for (uint32_t k = 1; k < numK; ++k) {
        const float* currentVec = vectors + k * dim;
        i = 0;

        // 向量化累加
        for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
            const float32x4_t meanVec = vld1q_f32(&mean[i]);
            const float32x4_t currentVecNeon = vld1q_f32(&currentVec[i]);
            vst1q_f32(&mean[i],vaddq_f32(meanVec, currentVecNeon));
        }

        // 处理剩余元素
        for (; i < dim; ++i) {
            mean[i] += currentVec[i];
        }
    }

    // 计算均值
    i = 0;
    for (; i + SimdElements::VEC4 <= dim; i += SimdElements::VEC4) {
        const float32x4_t sumVec = vld1q_f32(&mean[i]);
        vst1q_f32(&mean[i], vmulq_f32(sumVec, invNumK));
    }

    for (; i < dim; ++i) {
        mean[i] *= (1.0f / static_cast<float>(numK));
    }
}

#else
// ============= 通用标量实现 =============
float VecProduct::VectorDotImpl(const float* __restrict q, const float* __restrict k, uint32_t dim)
{
    // 4路展开减少循环开销
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    float sum4 = 0.0;
    uint32_t i = 0;

    // 4路循环展开
    for (; i + SCALAR_UNROLL_FACTOR <= dim; i += SCALAR_UNROLL_FACTOR) {
        sum1 += q[i] * k[i];
        sum2 += q[i+1] * k[i+1];
        sum3 += q[i+2] * k[i+2];
        sum4 += q[i+3] * k[i+3];
    }

    float sum = sum1 + sum2 + sum3 + sum4;

    // 处理剩余元素
    for (; i < dim; ++i) {
        sum += q[i] * k[i];
    }

    return sum;
}

void VecProduct::VectorMeanImpl(const float* __restrict vectors, float* __restrict mean, uint32_t dim, uint32_t numK)
{
    if (numK == 0) {
        std::fill(mean, mean + dim, 0.0f);
        return;
    }

    // mean初始化为vectors中第一个向量
    std::memcpy(mean, vectors, sizeof(float) * dim);

    const float invNumK = 1.0 / static_cast<float>(numK);

    // 累加所有向量
    for (uint32_t k = 1; k < numK; ++k) {
        const float* currentVec = vectors + k * dim;
        
        uint32_t i = 0;
        // 4路展开优化
        for (; i + SCALAR_UNROLL_FACTOR <= dim; i += SCALAR_UNROLL_FACTOR) {
            mean[i]   += currentVec[i];
            mean[i+1] += currentVec[i+1];
            mean[i+2] += currentVec[i+2];
            mean[i+3] += currentVec[i+3];
        }

        // 处理剩余元素
        for (; i < dim; ++i) {
            mean[i] += currentVec[i];
        }
    }    

    // 计算均值
    uint32_t i = 0;
    for (; i + SCALAR_UNROLL_FACTOR <= dim; i += SCALAR_UNROLL_FACTOR) {
        mean[i]   = mean[i] * invNumK;
        mean[i+1] = mean[i+1] * invNumK;
        mean[i+2] = mean[i+2] * invNumK;
        mean[i+3] = mean[i+3] * invNumK;
    }

    for (; i < dim; ++i) {
        mean[i] *= invNumK;
    }
}
#endif

}